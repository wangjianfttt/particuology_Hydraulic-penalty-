"""Run D3Q19 LBM before/after permeability probes for a local fines dump."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_local_fines_geometry import read_last_dump
from scripts.run_3d_window_lbm import run_lbm


BASE = ROOT / "data" / "production" / "3d_pebble_bed" / "local_windows" / "strong_force_chain"
FIG = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-glob", default=str(BASE / "fines_packed_slug_900_traj_*.dump"))
    parser.add_argument("--output-stem", default="local_fines_packed_slug_900_lbm")
    parser.add_argument("--box-size", type=float, default=0.009)
    parser.add_argument("--resolution", type=int, default=56)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--tau", type=float, default=0.72)
    parser.add_argument("--force-z", type=float, default=2.0e-7)
    parser.add_argument("--force-sweep", action="store_true")
    parser.add_argument("--fine-radius-inflate-dx", type=float, default=0.0)
    return parser.parse_args()


def voxelize_variable(particles: np.ndarray, box_size: float, n: int, fine_radius_inflate_dx: float = 0.0) -> np.ndarray:
    dx = box_size / n
    axes = [(np.arange(n) + 0.5) * dx for _ in range(3)]
    solid = np.zeros((n, n, n), dtype=bool)
    for _, ptype, x, y, z, radius in particles:
        if int(ptype) == 2:
            radius = radius + fine_radius_inflate_dx * dx
        margin = radius + np.sqrt(3.0) * dx
        if x < -margin or x > box_size + margin or y < -margin or y > box_size + margin or z < -margin or z > box_size + margin:
            continue
        ix0 = max(0, int(np.floor((x - radius) / dx)))
        ix1 = min(n - 1, int(np.ceil((x + radius) / dx)))
        iy0 = max(0, int(np.floor((y - radius) / dx)))
        iy1 = min(n - 1, int(np.ceil((y + radius) / dx)))
        iz0 = max(0, int(np.floor((z - radius) / dx)))
        iz1 = min(n - 1, int(np.ceil((z + radius) / dx)))
        xx = axes[0][ix0 : ix1 + 1][:, None, None]
        yy = axes[1][iy0 : iy1 + 1][None, :, None]
        zz = axes[2][iz0 : iz1 + 1][None, None, :]
        solid[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1] |= (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2 <= radius * radius
    return solid


def run_force_set(solid: np.ndarray, tau: float, force_z: float, steps: int, force_sweep: bool) -> tuple[dict[str, float | np.ndarray], float, float, float]:
    forces = [0.5 * force_z, force_z, 2.0 * force_z] if force_sweep else [force_z]
    results = [(force, run_lbm(solid, tau=tau, force_z=force, steps=steps)) for force in forces]
    result = results[len(results) // 2][1]
    force_arr = np.array([item[0] for item in results], dtype=float)
    superficial = np.array([item[1]["superficial_uz"] for item in results], dtype=float)
    if len(force_arr) > 1:
        slope, intercept = np.polyfit(force_arr, superficial, 1)
        pred = slope * force_arr + intercept
        ss_res = float(np.sum((superficial - pred) ** 2))
        ss_tot = float(np.sum((superficial - superficial.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)
    else:
        slope = float(superficial[0] / max(force_arr[0], 1.0e-30))
        intercept = 0.0
        r2 = float("nan")
    return result, float(slope), float(intercept), float(r2)


def main() -> None:
    args = parse_args()
    dumps = [Path(path) for path in sorted(glob.glob(args.dump_glob))]
    if not dumps:
        raise FileNotFoundError(args.dump_glob)
    particles = read_last_dump(dumps[-1])
    skeleton = particles[particles[:, 1] == 1]
    cases = [("skeleton", skeleton), ("final", particles)]
    rows = []
    slices = []
    out_dir = BASE / "fines_lbm"
    out_dir.mkdir(exist_ok=True)
    for label, pts in cases:
        solid = voxelize_variable(pts, args.box_size, args.resolution, fine_radius_inflate_dx=args.fine_radius_inflate_dx)
        result, slope, intercept, r2 = run_force_set(solid, args.tau, args.force_z, args.steps, args.force_sweep)
        rows.append(
            [
                label,
                args.resolution,
                args.steps,
                args.tau,
                args.force_z,
                args.fine_radius_inflate_dx,
                result["porosity_voxel"],
                result["mean_uz_fluid"],
                result["superficial_uz"],
                result["max_uz"],
                result["permeability_lu"],
                slope,
                intercept,
                r2,
            ]
        )
        slices.append((label, result["speed_slice"], result["solid_slice"]))
        np.savez_compressed(out_dir / f"{args.output_stem}_{label}.npz", solid=solid, speed_slice=result["speed_slice"], solid_slice=result["solid_slice"])
        print(f"{label}: porosity={result['porosity_voxel']:.4f}, K_lu={result['permeability_lu']:.6e}, R2={r2:.6f}")
    k_ratio = rows[1][10] / max(rows[0][10], 1.0e-30)
    summary = BASE / f"{args.output_stem}_permeability.csv"
    with summary.open("w") as fh:
        fh.write("case,resolution,steps,tau,force_z,fine_radius_inflate_dx,voxel_porosity,mean_uz_fluid,superficial_uz,max_uz,permeability_lu,darcy_slope,darcy_intercept,darcy_r2,K_final_over_K_skeleton\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row + [k_ratio]) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for ax, (label, speed, solid_slice) in zip(axes, slices):
        im = ax.imshow(np.ma.array(speed, mask=solid_slice).T, origin="lower", cmap="viridis")
        ax.contour(solid_slice.T, levels=[0.5], colors="white", linewidths=0.25)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
    fig.suptitle(f"D3Q19 local permeability ratio K_final/K_skeleton = {k_ratio:.3f}")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{args.output_stem}_permeability.pdf")
    fig.savefig(FIG / f"{args.output_stem}_permeability.png", dpi=220)
    print(f"K_final/K_skeleton={k_ratio:.6f}")
    print(f"wrote {summary}")
    print(f"wrote figures/{args.output_stem}_permeability.pdf")


if __name__ == "__main__":
    main()
