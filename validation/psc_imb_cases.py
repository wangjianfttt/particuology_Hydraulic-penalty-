"""Lightweight PSC-IMB validation cases.

The cases in this module are intentionally smoke-sized by default. They reuse
the audited D3Q19 PSC-IMB kernel and the existing benchmark routines, while
writing fresh validation artifacts under ``workspace/`` instead of mutating the
historical ``data/validation`` evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_d3q19_psc_moving_sphere_benchmark import (  # noqa: E402
    hasimoto_factor,
    run_case as run_moving_sphere_case,
)
from scripts.run_d3q19_psc_sphere_drag_benchmark import (  # noqa: E402
    run_case as run_fixed_sphere_case,
)
from src.lbm_dem.psc3d import CS2, Particle3D, PscD3Q19  # noqa: E402


DEFAULT_OUT = ROOT / "workspace" / "validation" / "psc_imb"


@dataclass(frozen=True)
class SettlingConfig:
    n: int = 28
    radius: float = 3.5
    tau: float = 0.72
    subsamples: int = 3
    steps: int = 180
    sample_every: int = 5
    particle_density: float = 2.5
    fluid_density: float = 1.0
    gravity_z: float = 2.0e-6


@dataclass(frozen=True)
class PeriodicResistanceConfig:
    n: int = 32
    radius: float = 4.0
    tau: float = 0.72
    subsamples: int = 3
    steps: int = 220
    sample_every: int = 20
    tail_fraction: float = 0.5
    velocities: tuple[float, ...] = (5.0e-5, 1.0e-4, 2.0e-4)
    body_forces: tuple[float, ...] = (5.0e-8, 1.0e-7)


def fluid_weighted_mean_uz(solver: PscD3Q19) -> float:
    fluid_weight = 1.0 - np.clip(solver.solid, 0.0, 1.0)
    return float(np.sum(fluid_weight * solver.uz) / max(float(np.sum(fluid_weight)), 1.0e-30))


def _write_csv(path: Path, rows: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


def run_single_particle_settling(out_dir: Path, cfg: SettlingConfig) -> dict[str, float]:
    """Run a small periodic single-particle settling check.

    This is a method-level validation, not a calibrated physical experiment:
    the particle is advanced through a periodic cubic domain under a constant
    excess body force, and the PSC-IMB hydrodynamic load is integrated back into
    a simple point-particle update.
    """

    volume = 4.0 / 3.0 * np.pi * cfg.radius**3
    mass = cfg.particle_density * volume
    buoyant_force_z = -(cfg.particle_density - cfg.fluid_density) * volume * cfg.gravity_z
    sphere = Particle3D(0.5 * cfg.n, 0.5 * cfg.n, 0.5 * cfg.n, cfg.radius)
    solver = PscD3Q19(
        n=cfg.n,
        particles=[sphere],
        tau=cfg.tau,
        subsamples=cfg.subsamples,
        body_force_z=0.0,
    )
    rows: list[list[float]] = []
    for step in range(cfg.steps + 1):
        diag = solver.collide_stream_psc()
        hydro_z = float(solver.hydro_forces[0, 2])
        net_z = buoyant_force_z + hydro_z
        sphere.vz += net_z / mass
        sphere.z = (sphere.z + sphere.vz) % float(cfg.n)
        if step % cfg.sample_every == 0 or step == cfg.steps:
            mean_fluid_uz = fluid_weighted_mean_uz(solver)
            rel_uz = sphere.vz - mean_fluid_uz
            rows.append(
                [
                    float(step),
                    sphere.z,
                    sphere.vz,
                    mean_fluid_uz,
                    rel_uz,
                    buoyant_force_z,
                    hydro_z,
                    net_z,
                    diag["momentum_residual"],
                    diag["solid_volume_lu"],
                ]
            )
    arr = np.asarray(rows, dtype=float)
    tail = arr[max(0, len(arr) - 6) :]
    terminal_vz = float(np.mean(tail[:, 2]))
    force_balance_residual = float(abs(np.mean(tail[:, 7])) / max(abs(buoyant_force_z), 1.0e-30))
    nu = CS2 * (cfg.tau - 0.5)
    summary = {
        "case": "single_particle_settling",
        "n": float(cfg.n),
        "radius_lu": float(cfg.radius),
        "tau": float(cfg.tau),
        "steps": float(cfg.steps),
        "buoyant_force_z": float(buoyant_force_z),
        "terminal_vz_tail": terminal_vz,
        "hydro_force_z_tail": float(np.mean(tail[:, 6])),
        "force_balance_residual_tail": force_balance_residual,
        "re_tail": float(2.0 * cfg.radius * abs(np.mean(tail[:, 4])) / max(nu, 1.0e-30)),
        "momentum_residual_tail": float(np.mean(tail[:, 8])),
        "solid_volume_lu": float(arr[-1, 9]),
    }
    _write_csv(
        out_dir / "single_particle_settling_timeseries.csv",
        arr,
        "step,z,vz,mean_fluid_uz,relative_uz,buoyant_force_z,hydro_force_z,net_force_z,momentum_residual,solid_volume_lu",
    )
    _write_csv(
        out_dir / "single_particle_settling_summary.csv",
        np.asarray([[summary[k] for k in summary if k != "case"]], dtype=float),
        ",".join(k for k in summary if k != "case"),
    )
    return summary


def run_periodic_resistance(out_dir: Path, cfg: PeriodicResistanceConfig) -> dict[str, float]:
    """Run fixed/moving periodic sphere resistance checks against Hasimoto."""

    moving_summaries = []
    for velocity in cfg.velocities:
        arr, summary = run_moving_sphere_case(
            cfg.n,
            cfg.radius,
            cfg.tau,
            cfg.subsamples,
            float(velocity),
            cfg.steps,
            cfg.sample_every,
            cfg.tail_fraction,
        )
        moving_summaries.append(summary)
        _write_csv(
            out_dir / f"periodic_moving_sphere_timeseries_velocity_{velocity:.1e}.csv",
            arr,
            "step,sphere_vz,mean_fluid_uz,relative_uz,force_z,abs_force_z,stokes_drag,resistance_ratio,re,momentum_residual,solid_volume_lu",
        )

    fixed_summaries = []
    for body_force in cfg.body_forces:
        arr, summary = run_fixed_sphere_case(
            cfg.n,
            cfg.radius,
            cfg.tau,
            cfg.subsamples,
            float(body_force),
            cfg.steps,
            cfg.sample_every,
        )
        fixed_summaries.append(summary)
        _write_csv(
            out_dir / f"periodic_fixed_sphere_timeseries_force_{body_force:.1e}.csv",
            arr,
            "step,body_force_z,mean_uz_fluid,force_z,abs_force_z,stokes_drag,resistance_ratio,re,momentum_residual,solid_volume_lu",
        )

    moving_header = list(moving_summaries[0].keys())
    fixed_header = list(fixed_summaries[0].keys())
    moving_arr = np.asarray([[row[k] for k in moving_header] for row in moving_summaries], dtype=float)
    fixed_arr = np.asarray([[row[k] for k in fixed_header] for row in fixed_summaries], dtype=float)
    _write_csv(out_dir / "periodic_moving_sphere_summary.csv", moving_arr, ",".join(moving_header))
    _write_csv(out_dir / "periodic_fixed_sphere_summary.csv", fixed_arr, ",".join(fixed_header))

    velocity = moving_arr[:, moving_header.index("sphere_vz")]
    force_z = moving_arr[:, moving_header.index("force_z_tail")]
    slope, intercept = np.polyfit(velocity, force_z, 1)
    pred = slope * velocity + intercept
    r2 = 1.0 - float(np.sum((force_z - pred) ** 2)) / max(float(np.sum((force_z - force_z.mean()) ** 2)), 1.0e-30)

    hasimoto = hasimoto_factor(cfg.radius, float(cfg.n))
    ratio_over_hasimoto = moving_arr[:, moving_header.index("ratio_over_hasimoto_tail")]
    fixed_ratio = fixed_arr[:, fixed_header.index("resistance_ratio_tail")]
    summary = {
        "case": "periodic_sphere_resistance",
        "n": float(cfg.n),
        "radius_lu": float(cfg.radius),
        "hasimoto_lambda": float(cfg.radius / float(cfg.n)),
        "hasimoto_factor": float(hasimoto),
        "moving_ratio_tail_mean": float(np.mean(moving_arr[:, moving_header.index("resistance_ratio_tail")])),
        "moving_ratio_over_hasimoto_mean": float(np.mean(ratio_over_hasimoto)),
        "moving_ratio_over_hasimoto_std": float(np.std(ratio_over_hasimoto, ddof=1)) if len(ratio_over_hasimoto) > 1 else 0.0,
        "moving_force_velocity_r2": float(r2),
        "fixed_body_force_ratio_tail_mean": float(np.mean(fixed_ratio)),
        "fixed_body_force_ratio_over_hasimoto_mean": float(np.mean(fixed_ratio) / hasimoto),
        "fixed_body_force_ratio_tail_std": float(np.std(fixed_ratio, ddof=1)) if len(fixed_ratio) > 1 else 0.0,
        "max_momentum_residual_tail": float(
            max(
                np.max(moving_arr[:, moving_header.index("momentum_residual_tail")]),
                np.max(fixed_arr[:, fixed_header.index("momentum_residual_tail")]),
            )
        ),
    }
    _write_csv(
        out_dir / "periodic_resistance_summary.csv",
        np.asarray([[summary[k] for k in summary if k != "case"]], dtype=float),
        ",".join(k for k in summary if k != "case"),
    )
    return summary


def run_validation_suite(
    out_dir: Path = DEFAULT_OUT,
    settling: SettlingConfig | None = None,
    periodic: PeriodicResistanceConfig | None = None,
) -> list[dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    settling_summary = run_single_particle_settling(out_dir, settling or SettlingConfig())
    periodic_summary = run_periodic_resistance(out_dir, periodic or PeriodicResistanceConfig())
    return [settling_summary, periodic_summary]
