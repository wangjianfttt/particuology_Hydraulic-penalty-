"""Estimate local geometric blockage from a LIGGGHTS fines trajectory dump."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "production" / "3d_pebble_bed" / "local_windows" / "strong_force_chain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-glob", default=str(BASE / "fines_packed_slug_900_traj_*.dump"))
    parser.add_argument("--output-stem", default="local_fines_packed_slug_900_geometry")
    parser.add_argument("--box-size", type=float, default=0.009)
    parser.add_argument("--nxy", type=int, default=72)
    parser.add_argument("--nz", type=int, default=72)
    return parser.parse_args()


def read_last_dump(path: Path) -> np.ndarray:
    rows = []
    columns: list[str] | None = None
    in_atoms = False
    for line in path.read_text().splitlines():
        if line.startswith("ITEM: ATOMS"):
            columns = line.split()[2:]
            in_atoms = True
            continue
        if line.startswith("ITEM:"):
            in_atoms = False
            continue
        if in_atoms:
            rows.append([float(value) for value in line.split()])
    if columns is None:
        raise ValueError(f"No ITEM: ATOMS section in {path}")
    table = np.asarray(rows, dtype=float)
    idx = {name: i for i, name in enumerate(columns)}
    return np.column_stack([table[:, idx["id"]], table[:, idx["type"]], table[:, idx["x"]], table[:, idx["y"]], table[:, idx["z"]], table[:, idx["radius"]]])


def open_area_profile(particles: np.ndarray, box_size: float, nxy: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, box_size, nxy, endpoint=False) + 0.5 * box_size / nxy
    ys = np.linspace(0.0, box_size, nxy, endpoint=False) + 0.5 * box_size / nxy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    z_centres = np.linspace(0.0, box_size, nz, endpoint=False) + 0.5 * box_size / nz
    open_fraction = np.empty(nz, dtype=float)
    for iz, z in enumerate(z_centres):
        occupied = np.zeros((nxy, nxy), dtype=bool)
        dz = np.abs(particles[:, 4] - z)
        active = particles[dz <= particles[:, 5]]
        for _, _, x, y, zp, radius in active:
            r2 = radius * radius - (z - zp) ** 2
            occupied |= (xx - x) ** 2 + (yy - y) ** 2 <= r2
        open_fraction[iz] = 1.0 - float(np.mean(occupied))
    return z_centres, open_fraction


def main() -> None:
    args = parse_args()
    dumps = [Path(path) for path in sorted(glob.glob(args.dump_glob))]
    if not dumps:
        raise FileNotFoundError(args.dump_glob)
    particles = read_last_dump(dumps[-1])
    skeleton = particles[particles[:, 1] == 1]
    z, open_skeleton = open_area_profile(skeleton, args.box_size, args.nxy, args.nz)
    _, open_final = open_area_profile(particles, args.box_size, args.nxy, args.nz)
    ratio = np.divide(open_final, open_skeleton, out=np.ones_like(open_final), where=open_skeleton > 1.0e-12)
    summary = np.asarray(
        [
            [
                float(np.min(open_skeleton)),
                float(np.min(open_final)),
                float(np.min(ratio)),
                float(z[int(np.argmin(ratio))]),
                float(np.mean(ratio)),
            ]
        ]
    )
    out_csv = BASE / f"{args.output_stem}.csv"
    np.savetxt(
        out_csv,
        summary,
        delimiter=",",
        header="min_open_skeleton,min_open_final,min_open_area_ratio,z_at_min_ratio_m,mean_open_area_ratio",
        comments="",
    )
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    z_mm = z / 1.0e-3
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.10},
    )
    ax, ax_ratio = axes
    ax.plot(z_mm, open_skeleton, color="#3f6db5", lw=1.5, label="skeleton only")
    ax.plot(z_mm, open_final, color="#c7665a", lw=1.5, label="skeleton + retained fines")
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=2.0, columnspacing=1.0)
    ax.set_ylabel("open area fraction")
    ax.text(-0.075, 1.05, "a", transform=ax.transAxes, fontweight="bold", va="top")
    ax.grid(True, alpha=0.25)

    ax_ratio.plot(z_mm, ratio, color="#4f8f5b", lw=1.5)
    ax_ratio.axhline(1.0, color="#333333", lw=0.8, alpha=0.75)
    ax_ratio.text(z_mm[-1] + 0.08, 0.985, "final / skeleton", color="#4f8f5b", va="center", fontsize=8)
    ax_ratio.set_xlabel("z position (mm)")
    ax_ratio.set_ylabel("open-area ratio")
    ax_ratio.text(-0.075, 1.05, "b", transform=ax_ratio.transAxes, fontweight="bold", va="top")
    ax_ratio.grid(True, alpha=0.25)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(float(np.min(z_mm)), float(np.max(z_mm) + 1.25))
    fig.subplots_adjust(left=0.11, right=0.88, top=0.96, bottom=0.14)
    fig.savefig(ROOT / "figures" / f"{args.output_stem}.pdf")
    fig.savefig(ROOT / "figures" / f"{args.output_stem}.png", dpi=220)
    print(f"wrote {out_csv}")
    print(f"wrote figures/{args.output_stem}.pdf")


if __name__ == "__main__":
    main()
