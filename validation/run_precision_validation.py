"""Resolution- and steady-state-aware PSC-IMB validation campaign.

The lightweight suite in ``run_psc_imb_validation.py`` is useful as a smoke
test.  This driver is deliberately stricter: every reported comparison has an
explicit reference value, a relative error, and a time-convergence diagnostic.
Short runs that have not reached steady state remain useful, but are marked as
such instead of being interpreted as spatial-discretization error.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_d3q19_psc_moving_sphere_benchmark import hasimoto_factor  # noqa: E402
from src.lbm_dem.psc3d import C, CS2, Particle3D, PscD3Q19  # noqa: E402


DEFAULT_OUT = ROOT / "workspace" / "validation" / "precision"
SUMMARY_FIELDS = [
    "case",
    "profile",
    "n",
    "radius_lu",
    "diameter_lu",
    "subsamples",
    "steps",
    "time_converged",
    "reference_name",
    "reference_value",
    "measured_value",
    "signed_relative_error",
    "absolute_relative_error",
    "steady_std_relative",
    "steady_drift_relative",
    "force_ratio_tail",
    "effective_force_ratio_tail",
    "force_balance_relative",
    "force_balance_pass",
    "re_tail",
    "hasimoto_lambda",
    "hasimoto_factor",
    "exact_porosity",
    "psc_porosity",
    "solid_volume_relative_error",
    "momentum_residual_tail",
]


@dataclass(frozen=True)
class GridRun:
    n: int
    radius: float
    min_steps: int
    max_steps: int
    sample_every: int


@dataclass(frozen=True)
class Profile:
    subsamples: int
    target_re: float
    steady_tolerance: float
    force_balance_tolerance: float
    settling: GridRun
    hasimoto: tuple[GridRun, ...]
    permeability: tuple[GridRun, ...]


PROFILES = {
    "quick": Profile(
        subsamples=2,
        target_re=5.0e-3,
        steady_tolerance=2.0e-2,
        force_balance_tolerance=5.0e-2,
        settling=GridRun(20, 2.5, 300, 800, 20),
        hasimoto=(GridRun(20, 2.5, 300, 900, 30), GridRun(28, 3.5, 500, 1200, 40)),
        permeability=(GridRun(20, 2.5, 600, 1800, 50),),
    ),
    "standard": Profile(
        subsamples=3,
        target_re=5.0e-3,
        steady_tolerance=1.0e-2,
        force_balance_tolerance=2.0e-2,
        settling=GridRun(32, 4.0, 800, 3000, 25),
        hasimoto=(
            GridRun(24, 3.0, 900, 2400, 40),
            GridRun(32, 4.0, 1400, 4200, 50),
            GridRun(40, 5.0, 2200, 6500, 60),
        ),
        permeability=(
            GridRun(24, 3.0, 1800, 5000, 60),
            GridRun(32, 4.0, 3000, 8500, 80),
            GridRun(40, 5.0, 4500, 13000, 100),
        ),
    ),
    "high": Profile(
        subsamples=4,
        target_re=2.5e-3,
        steady_tolerance=5.0e-3,
        force_balance_tolerance=1.0e-2,
        settling=GridRun(48, 6.0, 3000, 10000, 50),
        hasimoto=(
            GridRun(32, 4.0, 2500, 7000, 60),
            GridRun(48, 6.0, 5000, 15000, 100),
            GridRun(64, 8.0, 9000, 26000, 140),
        ),
        permeability=(
            GridRun(32, 4.0, 5000, 14000, 100),
            GridRun(48, 6.0, 10000, 30000, 160),
            GridRun(64, 8.0, 18000, 52000, 220),
        ),
    ),
}


def write_rows(path: Path, rows: Iterable[dict[str, object]], fields: Iterable[str] = SUMMARY_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_timeseries(path: Path, rows: list[list[float]], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows, dtype=float), delimiter=",", header=header, comments="")


def fluid_statistics(solver: PscD3Q19) -> tuple[float, float, float]:
    fluid_weight = 1.0 - np.clip(solver.solid, 0.0, 1.0)
    fluid_volume = float(np.sum(fluid_weight))
    intrinsic_uz = float(np.sum(fluid_weight * solver.uz) / max(fluid_volume, 1.0e-30))
    superficial_uz = float(np.mean(fluid_weight * solver.uz))
    return intrinsic_uz, superficial_uz, fluid_volume / float(solver.n**3)


def steady_metrics(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("inf"), float("inf")
    scale = max(abs(float(np.mean(values))), 1.0e-30)
    std_rel = float(np.std(values, ddof=1) / scale) if len(values) > 1 else float("inf")
    split = max(1, len(values) // 2)
    old = float(np.mean(values[:split]))
    new = float(np.mean(values[split:]))
    drift_rel = abs(new - old) / max(abs(new), 1.0e-30)
    return std_rel, drift_rel


def geometry_metrics(n: int, radius: float, psc_porosity: float) -> tuple[float, float]:
    exact_solid = 4.0 * np.pi * radius**3 / 3.0
    exact_porosity = 1.0 - exact_solid / float(n**3)
    psc_solid = (1.0 - psc_porosity) * float(n**3)
    volume_error = abs(psc_solid - exact_solid) / exact_solid
    return exact_porosity, volume_error


def common_summary(
    case: str,
    profile_name: str,
    run: GridRun,
    subsamples: int,
    steps: int,
    time_converged: bool,
    reference_name: str,
    reference_value: float,
    measured_value: float,
    std_rel: float,
    drift_rel: float,
    force_ratio_tail: float,
    effective_force_ratio_tail: float,
    force_balance_rel: float,
    force_balance_pass: int | str,
    re_tail: float,
    psc_porosity: float,
    momentum_residual_tail: float,
) -> dict[str, object]:
    exact_porosity, volume_error = geometry_metrics(run.n, run.radius, psc_porosity)
    signed_error = measured_value / max(reference_value, 1.0e-30) - 1.0
    return {
        "case": case,
        "profile": profile_name,
        "n": run.n,
        "radius_lu": run.radius,
        "diameter_lu": 2.0 * run.radius,
        "subsamples": subsamples,
        "steps": steps,
        "time_converged": int(time_converged),
        "reference_name": reference_name,
        "reference_value": reference_value,
        "measured_value": measured_value,
        "signed_relative_error": signed_error,
        "absolute_relative_error": abs(signed_error),
        "steady_std_relative": std_rel,
        "steady_drift_relative": drift_rel,
        "force_ratio_tail": force_ratio_tail,
        "effective_force_ratio_tail": effective_force_ratio_tail,
        "force_balance_relative": force_balance_rel,
        "force_balance_pass": force_balance_pass,
        "re_tail": re_tail,
        "hasimoto_lambda": run.radius / float(run.n),
        "hasimoto_factor": hasimoto_factor(run.radius, float(run.n)),
        "exact_porosity": exact_porosity,
        "psc_porosity": psc_porosity,
        "solid_volume_relative_error": volume_error,
        "momentum_residual_tail": momentum_residual_tail,
    }


def should_stop(
    step: int,
    run: GridRun,
    values: list[float],
    steady_tolerance: float,
    extra_check: Callable[[], bool] | None = None,
) -> bool:
    if step < run.min_steps or len(values) < 8:
        return False
    std_rel, drift_rel = steady_metrics(np.asarray(values[-8:], dtype=float))
    return std_rel <= steady_tolerance and drift_rel <= steady_tolerance and (extra_check is None or extra_check())


def run_settling(
    out_dir: Path, profile_name: str, profile: Profile, tau: float, density_ratio: float
) -> dict[str, object]:
    run = profile.settling
    nu = CS2 * (tau - 0.5)
    correction = hasimoto_factor(run.radius, float(run.n))
    reference_speed = profile.target_re * nu / (2.0 * run.radius)
    external_force = -6.0 * np.pi * nu * run.radius * correction * reference_speed
    volume = 4.0 * np.pi * run.radius**3 / 3.0
    mass = density_ratio * volume
    sphere = Particle3D(0.5 * run.n, 0.5 * run.n, 0.5 * run.n, run.radius)
    solver = PscD3Q19(run.n, [sphere], tau=tau, subsamples=profile.subsamples)
    rows: list[list[float]] = []
    relative_speeds: list[float] = []
    net_forces: list[float] = []
    residuals: list[float] = []
    psc_porosity = float("nan")
    for step in range(1, run.max_steps + 1):
        diag = solver.collide_stream_psc()
        hydro_force = float(solver.hydro_forces[0, 2])
        net_force = external_force + hydro_force
        sphere.vz += net_force / mass
        sphere.z = (sphere.z + sphere.vz) % float(run.n)
        if step % run.sample_every == 0 or step == run.max_steps:
            intrinsic_uz, _, psc_porosity = fluid_statistics(solver)
            relative_uz = sphere.vz - intrinsic_uz
            relative_speeds.append(abs(relative_uz))
            net_forces.append(net_force)
            residuals.append(float(diag["momentum_residual"]))
            rows.append(
                [
                    step,
                    sphere.z,
                    sphere.vz,
                    intrinsic_uz,
                    relative_uz,
                    external_force,
                    hydro_force,
                    net_force,
                    abs(relative_uz) / reference_speed - 1.0,
                    abs(net_force) / abs(external_force),
                    diag["momentum_residual"],
                    diag["solid_volume_lu"],
                ]
            )
            if should_stop(
                step,
                run,
                relative_speeds,
                profile.steady_tolerance,
                extra_check=lambda: abs(float(np.mean(net_forces[-8:]))) / abs(external_force)
                <= profile.force_balance_tolerance,
            ):
                break
    tail_count = min(8, len(relative_speeds))
    tail_speed = np.asarray(relative_speeds[-tail_count:])
    tail_net = np.asarray(net_forces[-tail_count:])
    std_rel, drift_rel = steady_metrics(tail_speed)
    force_balance = abs(float(np.mean(tail_net))) / abs(external_force)
    force_ratio = abs(float(np.mean(tail_net)) - external_force) / abs(external_force)
    converged = std_rel <= profile.steady_tolerance and drift_rel <= profile.steady_tolerance
    write_timeseries(
        out_dir / "settling_timeseries.csv",
        rows,
        "step,z,vz,mean_fluid_uz,relative_uz,external_force_z,hydro_force_z,net_force_z,"
        "signed_velocity_error,instant_force_balance_relative,momentum_residual,solid_volume_lu",
    )
    return common_summary(
        "single_particle_settling",
        profile_name,
        run,
        profile.subsamples,
        int(rows[-1][0]),
        converged,
        "Hasimoto-corrected Stokes terminal relative speed",
        reference_speed,
        float(np.mean(tail_speed)),
        std_rel,
        drift_rel,
        force_ratio,
        float("nan"),
        force_balance,
        int(force_balance <= profile.force_balance_tolerance),
        2.0 * run.radius * float(np.mean(tail_speed)) / nu,
        psc_porosity,
        float(np.mean(residuals[-tail_count:])),
    )


def run_hasimoto_case(
    out_dir: Path, profile_name: str, profile: Profile, tau: float, run: GridRun
) -> dict[str, object]:
    nu = CS2 * (tau - 0.5)
    correction = hasimoto_factor(run.radius, float(run.n))
    imposed_velocity = profile.target_re * nu / (2.0 * run.radius)
    sphere = Particle3D(0.5 * run.n, 0.5 * run.n, 0.5 * run.n, run.radius, vz=imposed_velocity)
    solver = PscD3Q19(
        run.n,
        [sphere],
        tau=tau,
        subsamples=profile.subsamples,
        static_geometry=True,
    )
    rows: list[list[float]] = []
    normalized_ratios: list[float] = []
    residuals: list[float] = []
    psc_porosity = float("nan")
    for step in range(1, run.max_steps + 1):
        diag = solver.collide_stream_psc()
        if step % run.sample_every == 0 or step == run.max_steps:
            intrinsic_uz, _, psc_porosity = fluid_statistics(solver)
            relative_uz = imposed_velocity - intrinsic_uz
            force_z = float(solver.hydro_forces[0, 2])
            stokes_drag = 6.0 * np.pi * nu * run.radius * abs(relative_uz)
            resistance_ratio = abs(force_z) / max(stokes_drag, 1.0e-30)
            normalized_ratio = resistance_ratio / correction
            normalized_ratios.append(normalized_ratio)
            residuals.append(float(diag["momentum_residual"]))
            rows.append(
                [
                    step,
                    imposed_velocity,
                    intrinsic_uz,
                    relative_uz,
                    force_z,
                    resistance_ratio,
                    normalized_ratio,
                    2.0 * run.radius * abs(relative_uz) / nu,
                    diag["momentum_residual"],
                    diag["solid_volume_lu"],
                ]
            )
            if should_stop(step, run, normalized_ratios, profile.steady_tolerance):
                break
    tail_count = min(8, len(normalized_ratios))
    tail = np.asarray(normalized_ratios[-tail_count:])
    std_rel, drift_rel = steady_metrics(tail)
    converged = std_rel <= profile.steady_tolerance and drift_rel <= profile.steady_tolerance
    tag = f"n{run.n:03d}_r{run.radius:g}".replace(".", "p")
    write_timeseries(
        out_dir / f"hasimoto_timeseries_{tag}.csv",
        rows,
        "step,imposed_sphere_uz,mean_fluid_uz,relative_uz,force_z,resistance_ratio,"
        "ratio_over_hasimoto,re,momentum_residual,solid_volume_lu",
    )
    re_tail = float(np.mean(np.asarray(rows[-tail_count:])[:, 7]))
    return common_summary(
        "periodic_moving_sphere_hasimoto",
        profile_name,
        run,
        profile.subsamples,
        int(rows[-1][0]),
        converged,
        "Hasimoto-normalized resistance ratio",
        1.0,
        float(np.mean(tail)),
        std_rel,
        drift_rel,
        float("nan"),
        float("nan"),
        float("nan"),
        "",
        re_tail,
        psc_porosity,
        float(np.mean(residuals[-tail_count:])),
    )


def run_permeability_case(
    out_dir: Path, profile_name: str, profile: Profile, tau: float, run: GridRun
) -> dict[str, object]:
    nu = CS2 * (tau - 0.5)
    correction = hasimoto_factor(run.radius, float(run.n))
    target_intrinsic_uz = profile.target_re * nu / (2.0 * run.radius)
    body_force = target_intrinsic_uz * 6.0 * np.pi * nu * run.radius * correction / float(run.n**3)
    sphere = Particle3D(0.5 * run.n, 0.5 * run.n, 0.5 * run.n, run.radius)
    solver = PscD3Q19(
        run.n,
        [sphere],
        tau=tau,
        subsamples=profile.subsamples,
        body_force_z=body_force,
        static_geometry=True,
    )
    exact_porosity, _ = geometry_metrics(run.n, run.radius, 1.0)
    reference_k = exact_porosity * run.n**3 / (6.0 * np.pi * run.radius * correction)
    rows: list[list[float]] = []
    permeabilities: list[float] = []
    force_balances: list[float] = []
    force_ratios: list[float] = []
    effective_force_ratios: list[float] = []
    re_values: list[float] = []
    residuals: list[float] = []
    psc_porosity = float("nan")
    expected_force = body_force * run.n**3
    for step in range(1, run.max_steps + 1):
        momentum_before = float(np.sum(solver.f * C[:, 2, None, None, None]))
        diag = solver.collide_stream_psc()
        momentum_after = float(np.sum(solver.f * C[:, 2, None, None, None]))
        if step % run.sample_every == 0 or step == run.max_steps:
            intrinsic_uz, superficial_uz, psc_porosity = fluid_statistics(solver)
            force_z = float(solver.hydro_forces[0, 2])
            permeability = nu * superficial_uz / body_force
            force_balance = abs(abs(force_z) - expected_force) / expected_force
            force_ratio = abs(force_z) / expected_force
            momentum_change = momentum_after - momentum_before
            effective_external_force = momentum_change + force_z
            effective_force_ratio = effective_external_force / expected_force
            re = 2.0 * run.radius * abs(intrinsic_uz) / nu
            permeabilities.append(permeability)
            force_balances.append(force_balance)
            force_ratios.append(force_ratio)
            effective_force_ratios.append(effective_force_ratio)
            re_values.append(re)
            residuals.append(float(diag["momentum_residual"]))
            rows.append(
                [
                    step,
                    body_force,
                    intrinsic_uz,
                    superficial_uz,
                    force_z,
                    expected_force,
                    momentum_change,
                    effective_external_force,
                    effective_force_ratio,
                    force_balance,
                    permeability,
                    reference_k,
                    permeability / reference_k - 1.0,
                    re,
                    diag["momentum_residual"],
                    diag["solid_volume_lu"],
                ]
            )
            if should_stop(step, run, permeabilities, profile.steady_tolerance):
                break
    tail_count = min(8, len(permeabilities))
    tail_k = np.asarray(permeabilities[-tail_count:])
    std_rel, drift_rel = steady_metrics(tail_k)
    force_balance = float(np.mean(force_balances[-tail_count:]))
    converged = std_rel <= profile.steady_tolerance and drift_rel <= profile.steady_tolerance
    tag = f"n{run.n:03d}_r{run.radius:g}".replace(".", "p")
    write_timeseries(
        out_dir / f"permeability_timeseries_{tag}.csv",
        rows,
        "step,body_force_z,intrinsic_uz,superficial_uz,hydro_force_z,expected_force_z,"
        "fluid_momentum_change_z,effective_external_force_z,effective_force_ratio,"
        "force_balance_relative,permeability_lu,reference_permeability_lu,signed_permeability_error,"
        "re,momentum_residual,solid_volume_lu",
    )
    re_tail = float(np.mean(re_values[-tail_count:]))
    return common_summary(
        "periodic_fixed_array_permeability",
        profile_name,
        run,
        profile.subsamples,
        int(rows[-1][0]),
        converged,
        "Darcy permeability from Hasimoto periodic drag",
        reference_k,
        float(np.mean(tail_k)),
        std_rel,
        drift_rel,
        float(np.mean(force_ratios[-tail_count:])),
        float(np.mean(effective_force_ratios[-tail_count:])),
        force_balance,
        int(force_balance <= profile.force_balance_tolerance),
        re_tail,
        psc_porosity,
        float(np.mean(residuals[-tail_count:])),
    )


def format_value(value: object, scientific: bool = True) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4e}" if scientific else f"{number:.4f}"


def write_report(out_dir: Path, profile_name: str, rows: list[dict[str, object]], command: str) -> Path:
    report = out_dir / "precision_validation_report.md"
    lines = [
        "# PSC-IMB precision validation",
        "",
        f"Profile: `{profile_name}`",
        "",
        f"Reproduce: `{command}`",
        "",
        "A spatial/reference error is only interpretable when `time converged` is 1.",
        "Force-driven cases must additionally pass the separate force-balance check.",
        "",
        "| case | N | d/lattice | steps | time converged | measured/reference | abs. relative error | steady drift | force balance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = float(row["measured_value"]) / max(float(row["reference_value"]), 1.0e-30)
        lines.append(
            f"| {row['case']} | {row['n']} | {format_value(row['diameter_lu'], False)} | {row['steps']} | "
            f"{row['time_converged']} | {ratio:.6f} | {format_value(row['absolute_relative_error'])} | "
            f"{format_value(row['steady_drift_relative'])} | {format_value(row['force_balance_relative'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Settling compares the measured particle-fluid relative terminal speed with the Hasimoto-corrected Stokes value.",
        "- Moving-sphere resistance compares drag/Stokes with the Hasimoto correction at fixed low Reynolds number.",
        "- Fixed-array permeability uses the same periodic drag reference plus Darcy's law; prescribed-force balance is a separate validation check.",
        "- The exact sphere volume defines the reference porosity. `solid_volume_relative_error` separately reports PSC geometry quadrature error.",
        "",
        "Full numeric fields are in `precision_error_table.csv`; every simulated point also has a time-series CSV.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def convergence_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for case_name in sorted({str(row["case"]) for row in rows}):
        case_rows = sorted(
            [row for row in rows if str(row["case"]) == case_name],
            key=lambda row: float(row["diameter_lu"]),
        )
        coarse = case_rows[-2] if len(case_rows) >= 2 else None
        fine = case_rows[-1]
        observed_order: float | str = ""
        if coarse is not None:
            coarse_error = float(coarse["absolute_relative_error"])
            fine_error = float(fine["absolute_relative_error"])
            diameter_ratio = float(fine["diameter_lu"]) / float(coarse["diameter_lu"])
            if coarse_error > 0.0 and fine_error > 0.0 and diameter_ratio > 1.0:
                observed_order = float(np.log(coarse_error / fine_error) / np.log(diameter_ratio))
        result.append(
            {
                "case": case_name,
                "n_points": len(case_rows),
                "all_time_converged": int(all(int(row["time_converged"]) == 1 for row in case_rows)),
                "finest_n": fine["n"],
                "finest_diameter_lu": fine["diameter_lu"],
                "finest_absolute_relative_error": fine["absolute_relative_error"],
                "observed_order_finest_pair": observed_order,
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    parser.add_argument("--case", choices=("all", "settling", "hasimoto", "permeability"), default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tau", type=float, default=0.72)
    parser.add_argument("--particle-density-ratio", type=float, default=2.5)
    parser.add_argument("--only-n", type=int, nargs="+", help="run only matching Hasimoto/permeability grids")
    parser.add_argument("--min-steps", type=int, help="override profile minimum steps for selected cases")
    parser.add_argument("--max-steps", type=int, help="override profile maximum steps for selected cases")
    parser.add_argument("--dry-run", action="store_true", help="write only the planned run matrix")
    return parser.parse_args()


def override_steps(profile: Profile, min_steps: int | None, max_steps: int | None) -> Profile:
    def update(run: GridRun) -> GridRun:
        new_min = run.min_steps if min_steps is None else min_steps
        new_max = run.max_steps if max_steps is None else max_steps
        if new_min > new_max:
            raise ValueError(f"min_steps={new_min} exceeds max_steps={new_max}")
        return replace(run, min_steps=new_min, max_steps=new_max)

    return replace(
        profile,
        settling=update(profile.settling),
        hasimoto=tuple(update(run) for run in profile.hasimoto),
        permeability=tuple(update(run) for run in profile.permeability),
    )


def selected_runs(runs: tuple[GridRun, ...], only_n: list[int] | None) -> tuple[GridRun, ...]:
    return runs if only_n is None else tuple(run for run in runs if run.n in only_n)


def planned_rows(
    profile_name: str, profile: Profile, case: str, only_n: list[int] | None
) -> list[dict[str, object]]:
    plans: list[dict[str, object]] = []
    selected = ("settling", "hasimoto", "permeability") if case == "all" else (case,)
    if "settling" in selected and (only_n is None or profile.settling.n in only_n):
        plans.append({"case": "settling", **profile.settling.__dict__})
    if "hasimoto" in selected:
        plans.extend({"case": "hasimoto", **run.__dict__} for run in selected_runs(profile.hasimoto, only_n))
    if "permeability" in selected:
        plans.extend(
            {"case": "permeability", **run.__dict__} for run in selected_runs(profile.permeability, only_n)
        )
    for row in plans:
        row["profile"] = profile_name
        row["subsamples"] = profile.subsamples
        row["target_re"] = profile.target_re
    return plans


def main() -> None:
    args = parse_args()
    profile = override_steps(PROFILES[args.profile], args.min_steps, args.max_steps)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = planned_rows(args.profile, profile, args.case, args.only_n)
    if not plan:
        raise ValueError("No runs match the selected case/profile/--only-n combination")
    write_rows(args.out_dir / "run_plan.csv", plan, plan[0].keys())
    command = "python validation/run_precision_validation.py " + " ".join(sys.argv[1:])
    manifest = {
        "command": command,
        "profile": args.profile,
        "case": args.case,
        "tau": args.tau,
        "particle_density_ratio": args.particle_density_ratio,
        "dry_run": args.dry_run,
        "plan": plan,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"wrote {args.out_dir / 'run_plan.csv'}")
        return

    selected = ("settling", "hasimoto", "permeability") if args.case == "all" else (args.case,)
    summaries: list[dict[str, object]] = []
    if "settling" in selected and (args.only_n is None or profile.settling.n in args.only_n):
        summary = run_settling(args.out_dir, args.profile, profile, args.tau, args.particle_density_ratio)
        summaries.append(summary)
        print(f"settling N={summary['n']}: error={float(summary['absolute_relative_error']):.3e}")
    if "hasimoto" in selected:
        for run in selected_runs(profile.hasimoto, args.only_n):
            summary = run_hasimoto_case(args.out_dir, args.profile, profile, args.tau, run)
            summaries.append(summary)
            print(f"Hasimoto N={summary['n']}: error={float(summary['absolute_relative_error']):.3e}")
    if "permeability" in selected:
        for run in selected_runs(profile.permeability, args.only_n):
            summary = run_permeability_case(args.out_dir, args.profile, profile, args.tau, run)
            summaries.append(summary)
            print(
                f"permeability N={summary['n']}: error={float(summary['absolute_relative_error']):.3e}, "
                f"force-balance={float(summary['force_balance_relative']):.3e}"
            )

    write_rows(args.out_dir / "precision_error_table.csv", summaries)
    convergence = convergence_rows(summaries)
    write_rows(args.out_dir / "resolution_convergence.csv", convergence, convergence[0].keys())
    for case_name in sorted({str(row["case"]) for row in summaries}):
        case_rows = [row for row in summaries if row["case"] == case_name]
        write_rows(args.out_dir / f"{case_name}_summary.csv", case_rows)
    report = write_report(args.out_dir, args.profile, summaries, command)
    print(f"wrote {args.out_dir / 'precision_error_table.csv'}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
