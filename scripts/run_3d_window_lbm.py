"""D3Q19 LBM flow probes through selected local 3-D pebble-bed windows.

This is a bridge calculation: the 30k LIGGGHTS bed supplies realistic local
geometry, while a compact single-phase D3Q19 solver estimates relative
permeability/flow focusing in selected windows before full IMB coupling.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "production" / "3d_pebble_bed"
FIG = ROOT / "figures"


C = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
    ],
    dtype=int,
)
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=float)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
CS2 = 1.0 / 3.0


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> np.ndarray:
    uu = ux * ux + uy * uy + uz * uz
    feq = np.empty((19, *rho.shape), dtype=np.float64)
    for i, (cx, cy, cz) in enumerate(C):
        cu = cx * ux + cy * uy + cz * uz
        feq[i] = W[i] * rho * (1.0 + cu / CS2 + 0.5 * cu * cu / (CS2 * CS2) - 0.5 * uu / CS2)
    return feq


def guo_force(fx: float, fy: float, fz: float, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, tau: float) -> np.ndarray:
    term = np.empty((19, *ux.shape), dtype=np.float64)
    prefactor = 1.0 - 0.5 / tau
    uf = ux * fx + uy * fy + uz * fz
    for i, (cx, cy, cz) in enumerate(C):
        cu = cx * ux + cy * uy + cz * uz
        cf = cx * fx + cy * fy + cz * fz
        term[i] = prefactor * W[i] * ((cf - uf) / CS2 + cu * cf / (CS2 * CS2))
    return term


def voxelize(points: np.ndarray, center: np.ndarray, size: np.ndarray, radius: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    lo = center - 0.5 * size
    hi = center + 0.5 * size
    dx = size / n
    axes = [lo[k] + (np.arange(n) + 0.5) * dx[k] for k in range(3)]
    solid = np.zeros((n, n, n), dtype=bool)
    margin = radius + np.linalg.norm(dx)
    near = np.all((points >= lo - margin) & (points <= hi + margin), axis=1)
    for x, y, z in points[near]:
        ix0 = max(0, int(np.floor((x - radius - lo[0]) / dx[0])))
        ix1 = min(n - 1, int(np.ceil((x + radius - lo[0]) / dx[0])))
        iy0 = max(0, int(np.floor((y - radius - lo[1]) / dx[1])))
        iy1 = min(n - 1, int(np.ceil((y + radius - lo[1]) / dx[1])))
        iz0 = max(0, int(np.floor((z - radius - lo[2]) / dx[2])))
        iz1 = min(n - 1, int(np.ceil((z + radius - lo[2]) / dx[2])))
        xx = axes[0][ix0 : ix1 + 1][:, None, None]
        yy = axes[1][iy0 : iy1 + 1][None, :, None]
        zz = axes[2][iz0 : iz1 + 1][None, None, :]
        solid[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1] |= (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2 <= radius * radius
    return solid, dx


def run_lbm(solid: np.ndarray, tau: float, force_z: float, steps: int, return_fields: bool = False) -> dict[str, float | np.ndarray]:
    shape = solid.shape
    fluid = ~solid
    rho = np.ones(shape)
    ux = np.zeros(shape)
    uy = np.zeros(shape)
    uz = np.zeros(shape)
    f = equilibrium(rho, ux, uy, uz)
    nu = CS2 * (tau - 0.5)
    last_mean = 0.0
    for step in range(steps):
        rho = np.sum(f, axis=0)
        ux = np.sum(f * C[:, 0, None, None, None], axis=0) / rho
        uy = np.sum(f * C[:, 1, None, None, None], axis=0) / rho
        uz = (np.sum(f * C[:, 2, None, None, None], axis=0) + 0.5 * force_z) / rho
        ux[solid] = 0.0
        uy[solid] = 0.0
        uz[solid] = 0.0
        feq = equilibrium(rho, ux, uy, uz)
        fpost = f - (f - feq) / tau + guo_force(0.0, 0.0, force_z, ux, uy, uz, tau)
        for i, (cx, cy, cz) in enumerate(C):
            f[i] = np.roll(fpost[i], shift=(cx, cy, cz), axis=(0, 1, 2))
        for i in range(19):
            vals = f[OPP[i], solid].copy()
            f[i, solid] = vals
        if step % 50 == 0 or step == steps - 1:
            last_mean = float(np.mean(uz[fluid]))
    rho = np.sum(f, axis=0)
    uz = (np.sum(f * C[:, 2, None, None, None], axis=0) + 0.5 * force_z) / rho
    uz[solid] = 0.0
    mean_uz_fluid = float(np.mean(uz[fluid]))
    superficial_uz = float(np.mean(uz))
    permeability_lu = nu * superficial_uz / max(force_z, 1e-30)
    result = {
        "mean_uz_fluid": mean_uz_fluid,
        "superficial_uz": superficial_uz,
        "max_uz": float(np.max(uz[fluid])) if np.any(fluid) else 0.0,
        "permeability_lu": permeability_lu,
        "porosity_voxel": float(np.mean(fluid)),
        "speed_slice": uz[:, :, shape[2] // 2].copy(),
        "solid_slice": solid[:, :, shape[2] // 2].copy(),
    }
    if return_fields:
        rho = np.sum(f, axis=0)
        ux = np.sum(f * C[:, 0, None, None, None], axis=0) / rho
        uy = np.sum(f * C[:, 1, None, None, None], axis=0) / rho
        uz = (np.sum(f * C[:, 2, None, None, None], axis=0) + 0.5 * force_z) / rho
        ux[solid] = 0.0
        uy[solid] = 0.0
        uz[solid] = 0.0
        result.update({"ux": ux.copy(), "uy": uy.copy(), "uz": uz.copy()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default="pebble_bed_30000_dense_contact")
    parser.add_argument("--resolution", type=int, default=56)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.72)
    parser.add_argument("--force-z", type=float, default=2.0e-7)
    parser.add_argument("--force-sweep", action="store_true", help="run three force levels for Darcy-linearity checks")
    args = parser.parse_args()

    meta = json.loads((BASE / f"{args.stem}_metadata.json").read_text())
    points_data = np.genfromtxt(BASE / f"{args.stem}_centres.csv", delimiter=",", names=True)
    points = np.column_stack([points_data["x_m"], points_data["y_m"], points_data["z_m"]])
    windows = np.genfromtxt(BASE / "selected_lbm_windows.csv", delimiter=",", names=True, dtype=None, encoding=None)
    out_dir = BASE / "window_lbm"
    out_dir.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)

    rows = []
    slices = []
    force_levels = [0.5 * args.force_z, args.force_z, 2.0 * args.force_z] if args.force_sweep else [args.force_z]
    fit_rows = []
    for row in np.atleast_1d(windows):
        label = str(row["label"])
        center = np.array([row["cx"], row["cy"], row["cz"]], dtype=float)
        size = np.array([row["sx"], row["sy"], row["sz"]], dtype=float)
        solid, dx = voxelize(points, center, size, radius=0.5 * meta["diameter_m"], n=args.resolution)
        force_results = [(force, run_lbm(solid, tau=args.tau, force_z=force, steps=args.steps)) for force in force_levels]
        result = force_results[len(force_results) // 2][1]
        forces = np.array([item[0] for item in force_results], dtype=float)
        superficial = np.array([item[1]["superficial_uz"] for item in force_results], dtype=float)
        if len(force_results) > 1:
            slope, intercept = np.polyfit(forces, superficial, 1)
            pred = slope * forces + intercept
            ss_res = float(np.sum((superficial - pred) ** 2))
            ss_tot = float(np.sum((superficial - superficial.mean()) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        else:
            slope = float(superficial[0] / max(forces[0], 1e-30))
            intercept = 0.0
            r2 = float("nan")
        fit_rows.append([label, slope, intercept, r2])
        np.savez_compressed(out_dir / f"{label}_fields.npz", solid=solid, speed_slice=result["speed_slice"], solid_slice=result["solid_slice"], dx=dx)
        rows.append(
            [
                label,
                args.resolution,
                args.steps,
                args.tau,
                force_results[len(force_results) // 2][0],
                result["porosity_voxel"],
                result["mean_uz_fluid"],
                result["superficial_uz"],
                result["max_uz"],
                result["permeability_lu"],
                float(row["porosity"]),
                result["porosity_voxel"] - float(row["porosity"]),
                float(row["n_particles"]),
                float(row["n_contacts"]),
                float(row["force_sum"]),
                slope,
                intercept,
                r2,
            ]
        )
        slices.append((label, result["speed_slice"], result["solid_slice"]))
        print(
            f"{label}: voxel porosity={result['porosity_voxel']:.3f}, "
            f"Us={result['superficial_uz']:.3e}, k_lu={result['permeability_lu']:.3e}, Darcy R2={r2:.6f}"
        )

    summary = out_dir / "window_lbm_summary.csv"
    header = (
        "label,resolution,steps,tau,force_z,voxel_porosity,mean_uz_fluid,superficial_uz,"
        "max_uz,permeability_lu,window_porosity,porosity_error,n_particles,n_contacts,"
        "force_sum,darcy_slope,darcy_intercept,darcy_r2"
    )
    with summary.open("w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.2), constrained_layout=True)
    for idx, (ax, item) in enumerate(zip(axes.ravel(), slices)):
        label, speed, solid_slice = item
        masked = np.ma.array(speed, mask=solid_slice)
        im = ax.imshow(masked.T, origin="lower", cmap="viridis")
        ax.contour(solid_slice.T, levels=[0.5], colors="white", linewidths=0.25)
        ax.text(-0.06, 1.04, chr(ord("a") + idx), transform=ax.transAxes, fontweight="bold", va="top")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
    if len(slices) < len(axes.ravel()):
        axes.ravel()[-1].axis("off")
    fig.savefig(FIG / "window_3d_lbm_flow_slices.pdf")
    fig.savefig(FIG / "window_3d_lbm_flow_slices.png", dpi=220)

    data = np.genfromtxt(summary, delimiter=",", names=True, dtype=None, encoding=None)
    fig2, axes2 = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    labels = [str(v) for v in np.atleast_1d(data["label"])]
    vals = np.atleast_1d(data["permeability_lu"]).astype(float)
    axes2[0].bar(labels, vals, color="#3f6db5")
    axes2[0].set_ylabel("LBM permeability proxy (l.u.)")
    axes2[0].text(-0.13, 1.06, "a", transform=axes2[0].transAxes, fontweight="bold", va="top")
    axes2[0].tick_params(axis="x", labelrotation=25)
    axes2[0].grid(True, axis="y", alpha=0.25)
    forces = np.atleast_1d(data["force_sum"]).astype(float)
    colors = ["#3f6db5" if label != "strong_force_chain" else "#c23b30" for label in labels]
    axes2[1].bar(labels, forces, color=colors)
    axes2[1].set_ylabel("contact-force sum (N)")
    axes2[1].text(-0.13, 1.06, "b", transform=axes2[1].transAxes, fontweight="bold", va="top")
    axes2[1].tick_params(axis="x", labelrotation=25)
    axes2[1].grid(True, axis="y", alpha=0.25)
    fig2.savefig(FIG / "window_3d_lbm_permeability.pdf")
    fig2.savefig(FIG / "window_3d_lbm_permeability.png", dpi=220)
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
