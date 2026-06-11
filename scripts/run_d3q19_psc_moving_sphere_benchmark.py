"""Moving-sphere resistance check for the D3Q19 PSC-IMB kernel.

The fixed-sphere benchmark validates pressure-gradient-driven flow around a
stationary particle. This companion check imposes a rigid translational velocity
on the sphere in an initially quiescent periodic domain. The particle centre is
held fixed while the PSC solid velocity is non-zero, so the test isolates the
moving-boundary force path without mesh motion or DEM displacement.

Because the periodic box is small, the result is not expected to equal
unbounded Stokes drag. The useful checks are force direction, velocity
linearity, resistance-scale consistency and particle-fluid momentum closure.
The summary also reports the Hasimoto low-volume-fraction correction for a
translating sphere in a cubic periodic array, using lambda = a / L.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lbm_dem.psc3d import CS2, Particle3D, PscD3Q19


OUT = ROOT / "data" / "validation" / "d3q19_psc_moving_sphere"
FIG = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=0.72)
    parser.add_argument("--subsamples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--tail-fraction", type=float, default=0.5)
    parser.add_argument("--velocities", type=float, nargs="+", default=[0.5e-4, 1.0e-4, 2.0e-4, 4.0e-4])
    parser.add_argument("--tag", default="", help="Optional suffix for isolated output directories and figures.")
    return parser.parse_args()


def hasimoto_factor(radius: float, box_length: float) -> float:
    """Periodic Stokes drag correction for a sphere in a cubic array."""
    lam = radius / box_length
    denominator = 1.0 - 2.837297 * lam + 4.19 * lam**3 - 27.4 * lam**6
    return float(1.0 / denominator)


def fluid_weighted_mean_uz(solver: PscD3Q19) -> float:
    fluid_weight = 1.0 - np.clip(solver.solid, 0.0, 1.0)
    return float(np.sum(fluid_weight * solver.uz) / max(float(np.sum(fluid_weight)), 1.0e-30))


def run_case(
    n: int,
    radius: float,
    tau: float,
    subsamples: int,
    velocity: float,
    steps: int,
    sample_every: int,
    tail_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    sphere = Particle3D(0.5 * n, 0.5 * n, 0.5 * n, radius, vz=velocity)
    solver = PscD3Q19(n=n, particles=[sphere], tau=tau, subsamples=subsamples, body_force_z=0.0)
    nu = CS2 * (tau - 0.5)
    rows = []
    for step in range(steps + 1):
        diag = solver.collide_stream_psc()
        if step % sample_every == 0 or step == steps:
            mean_fluid_uz = fluid_weighted_mean_uz(solver)
            force_z = float(solver.hydro_forces[0, 2])
            rel_u = velocity - mean_fluid_uz
            stokes = 6.0 * np.pi * nu * radius * max(abs(rel_u), 1.0e-30)
            resistance_ratio = abs(force_z) / max(stokes, 1.0e-30)
            re = 2.0 * radius * abs(rel_u) / max(nu, 1.0e-30)
            rows.append(
                [
                    step,
                    velocity,
                    mean_fluid_uz,
                    rel_u,
                    force_z,
                    abs(force_z),
                    stokes,
                    resistance_ratio,
                    re,
                    diag["momentum_residual"],
                    diag["solid_volume_lu"],
                ]
            )
    arr = np.asarray(rows, dtype=float)
    tail_start = max(0, int(np.floor((1.0 - tail_fraction) * len(arr))))
    tail = arr[tail_start:]
    hasimoto = hasimoto_factor(radius, float(n))
    ratio_tail = tail[:, 7]
    summary = {
        "n": float(n),
        "radius_lu": float(radius),
        "diameter_lu": float(2.0 * radius),
        "tau": float(tau),
        "nu_lu": float(nu),
        "steps": float(steps),
        "tail_fraction": float(tail_fraction),
        "hasimoto_lambda": float(radius / float(n)),
        "hasimoto_factor": float(hasimoto),
        "sphere_vz": float(velocity),
        "mean_fluid_uz_tail": float(np.mean(tail[:, 2])),
        "relative_uz_tail": float(np.mean(tail[:, 3])),
        "force_z_tail": float(np.mean(tail[:, 4])),
        "abs_force_tail": float(np.mean(tail[:, 5])),
        "stokes_drag_tail": float(np.mean(tail[:, 6])),
        "resistance_ratio_tail": float(np.mean(ratio_tail)),
        "resistance_ratio_std_tail": float(np.std(ratio_tail, ddof=1)) if len(ratio_tail) > 1 else 0.0,
        "ratio_over_hasimoto_tail": float(np.mean(ratio_tail) / hasimoto),
        "ratio_over_hasimoto_std_tail": float(np.std(ratio_tail / hasimoto, ddof=1)) if len(ratio_tail) > 1 else 0.0,
        "re_tail": float(np.mean(tail[:, 8])),
        "momentum_residual_tail": float(np.mean(tail[:, 9])),
        "solid_volume_lu": float(arr[-1, 10]),
    }
    return arr, summary


def main() -> None:
    args = parse_args()
    out_dir = OUT / args.tag if args.tag else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    summaries = []
    series = []
    for velocity in args.velocities:
        arr, summary = run_case(
            args.n,
            args.radius,
            args.tau,
            args.subsamples,
            velocity,
            args.steps,
            args.sample_every,
            args.tail_fraction,
        )
        summaries.append(summary)
        series.append(arr)
        np.savetxt(
            out_dir / f"timeseries_velocity_{velocity:.1e}.csv",
            arr,
            delimiter=",",
            header="step,sphere_vz,mean_fluid_uz,relative_uz,force_z,abs_force_z,stokes_drag,resistance_ratio,re,momentum_residual,solid_volume_lu",
            comments="",
        )
        print(
            f"vz={velocity:.1e} relU={summary['relative_uz_tail']:.3e} "
            f"Fz={summary['force_z_tail']:.3e} |F|/Stokes={summary['resistance_ratio_tail']:.3f} "
            f"Hasimoto-normalized={summary['ratio_over_hasimoto_tail']:.3f} "
            f"Re={summary['re_tail']:.3e} residual={summary['momentum_residual_tail']:.1e}"
        )
    header = list(summaries[0].keys())
    arr = np.asarray([[row[k] for k in header] for row in summaries], dtype=float)
    np.savetxt(out_dir / "summary.csv", arr, delimiter=",", header=",".join(header), comments="")

    v = arr[:, header.index("sphere_vz")]
    fz = arr[:, header.index("force_z_tail")]
    absf = arr[:, header.index("abs_force_tail")]
    slope, intercept = np.polyfit(v, fz, 1)
    pred = slope * v + intercept
    ss_res = float(np.sum((fz - pred) ** 2))
    ss_tot = float(np.sum((fz - fz.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)
    np.savetxt(
        out_dir / "linearity.csv",
        np.asarray([[slope, intercept, r2, float(np.max(np.abs(arr[:, header.index("momentum_residual_tail")])))]], dtype=float),
        delimiter=",",
        header="force_velocity_slope,force_velocity_intercept,r2,max_momentum_residual_tail",
        comments="",
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2), constrained_layout=True)
    for ts in series:
        axes[0].plot(ts[:, 0], ts[:, 4], label=f"v={ts[0,1]:.1e}")
    axes[0].set_xlabel("LBM step")
    axes[0].set_ylabel(r"$F_z$ on moving sphere")
    axes[0].set_title("force relaxation")
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].plot(v, fz, "o", color="#3f6db5", label="PSC-IMB")
    xfit = np.linspace(0.0, float(v.max()) * 1.05, 100)
    axes[1].plot(xfit, slope * xfit + intercept, "--", color="#333333", label=f"linear fit $R^2$={r2:.6f}")
    axes[1].set_xlabel("imposed sphere velocity")
    axes[1].set_ylabel(r"tail $F_z$")
    axes[1].set_title("moving-boundary linearity")
    axes[1].legend(frameon=False, fontsize=7)

    re_tail = arr[:, header.index("re_tail")]
    ratio_tail = arr[:, header.index("resistance_ratio_tail")]
    ratio_std = arr[:, header.index("resistance_ratio_std_tail")]
    hasimoto = float(arr[0, header.index("hasimoto_factor")])
    axes[2].errorbar(re_tail, ratio_tail, yerr=ratio_std, fmt="s-", color="#c7665a", capsize=2.5, label="PSC-IMB")
    axes[2].axhline(hasimoto, color="#4d5562", linestyle="--", linewidth=1.2, label=f"Hasimoto {hasimoto:.3f}")
    axes[2].set_xlabel(r"$Re$")
    axes[2].set_ylabel(r"$|F|/(6\pi\nu r |U_\mathrm{rel}|)$")
    axes[2].set_title("periodic resistance")
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig_stem = "d3q19_psc_moving_sphere_benchmark" if not args.tag else f"d3q19_psc_moving_sphere_benchmark_{args.tag}"
    fig_path = FIG / f"{fig_stem}.pdf"
    fig.savefig(fig_path)
    fig.savefig(fig_path.with_suffix(".png"), dpi=240)
    print(f"force-velocity slope={slope:.6e}, intercept={intercept:.3e}, R2={r2:.8f}")
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
