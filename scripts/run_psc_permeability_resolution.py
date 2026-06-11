#!/usr/bin/env python3
"""Run one PSC-IMB permeability probe for a local pore-window geometry."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lbm_dem.psc3d import Particle3D, SparsePscD3Q19


BOX_SIZE_M = 9.0e-3
OUTDIR = ROOT / "convergence" / "permeability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=["skeleton", "baseline", "packed_slug"])
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--source-dump", required=True)
    parser.add_argument("--particle-filter", default="all", choices=["all", "type==1"])
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--tau", type=float, default=0.72)
    parser.add_argument("--force-z", type=float, default=2.0e-7)
    parser.add_argument("--subsamples", type=int, default=2)
    parser.add_argument(
        "--geometry-workers",
        type=int,
        default=1,
        help="Number of worker processes used only for sparse PSC solid-fraction mapping.",
    )
    parser.add_argument("--outdir", default=str(OUTDIR))
    return parser.parse_args()


def parse_dump(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    cols = lines[header].split()[2:]
    idx = {name: i for i, name in enumerate(cols)}
    required = ["id", "type", "x", "y", "z", "radius"]
    missing = [name for name in required if name not in idx]
    if missing:
        raise ValueError(f"{path} missing dump columns: {missing}")
    rows = []
    for line in lines[header + 1 :]:
        if not line.strip():
            continue
        parts = line.split()
        rows.append(
            [
                int(float(parts[idx["id"]])),
                int(float(parts[idx["type"]])),
                float(parts[idx["x"]]),
                float(parts[idx["y"]]),
                float(parts[idx["z"]]),
                float(parts[idx["radius"]]),
            ]
        )
    return np.asarray(rows, dtype=float)


def to_particles(rows: np.ndarray, resolution: int) -> list[Particle3D]:
    scale = resolution / BOX_SIZE_M
    particles = []
    for _, _, x, y, z, r in rows:
        particles.append(Particle3D(x * scale, y * scale, z * scale, r * scale))
    return particles


def filter_particles(rows: np.ndarray, particle_filter: str) -> np.ndarray:
    if particle_filter == "type==1":
        return rows[rows[:, 1] == 1]
    return rows


def write_row(path: Path, row: dict[str, object]) -> None:
    fields = [
        "case",
        "resolution",
        "steps",
        "tau",
        "force_z",
        "subsamples",
        "geometry_workers",
        "source_dump",
        "particle_filter",
        "n_particles",
        "porosity_psc",
        "mean_uz_fluid",
        "superficial_uz",
        "max_uz",
        "permeability_lu",
        "K_over_skeleton_same_resolution",
        "momentum_residual",
        "particle_force_sum_x_lu",
        "particle_force_sum_y_lu",
        "particle_force_sum_z_lu",
        "particle_force_norm_sum_lu",
        "fluid_imb_sum_x_lu",
        "fluid_imb_sum_y_lu",
        "fluid_imb_sum_z_lu",
        "particle_fluid_momentum_closure_lu",
        "domain_body_force_z_lu",
        "fluid_volume_body_force_z_lu",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows_all = parse_dump(Path(args.source_dump))
    rows = filter_particles(rows_all, args.particle_filter)
    particles = to_particles(rows, args.resolution)
    solver = SparsePscD3Q19(
        n=args.resolution,
        particles=particles,
        tau=args.tau,
        subsamples=args.subsamples,
        body_force_z=args.force_z,
        static_geometry=True,
        geometry_workers=args.geometry_workers,
    )
    last = None
    for step in range(args.steps):
        last = solver.collide_stream_psc()
        if step == 0 or (step + 1) % 60 == 0 or step + 1 == args.steps:
            print(
                f"{args.case} n={args.resolution} step={step + 1}/{args.steps} "
                f"mean_uz={last['mean_uz']:.3e} residual={last['momentum_residual']:.3e}"
            )
    fluid_weight = 1.0 - solver.solid
    fluid_volume = float(np.sum(fluid_weight))
    weighted_uz = solver.uz * fluid_weight
    mean_uz_fluid = float(np.sum(weighted_uz) / max(fluid_volume, 1.0e-300))
    superficial_uz = float(np.sum(weighted_uz) / solver.n**3)
    nu = (args.tau - 0.5) / 3.0
    permeability = float(nu * superficial_uz / max(args.force_z, 1.0e-300))
    particle_force_sum = np.sum(solver.hydro_forces, axis=0) if len(solver.hydro_forces) else np.zeros(3)
    fluid_imb_sum = np.array(
        [
            float(np.sum(solver.imb_mx)),
            float(np.sum(solver.imb_my)),
            float(np.sum(solver.imb_mz)),
        ],
        dtype=float,
    )
    particle_fluid_closure = float(np.linalg.norm(particle_force_sum + fluid_imb_sum))
    row = {
        "case": args.case,
        "resolution": args.resolution,
        "steps": args.steps,
        "tau": args.tau,
        "force_z": args.force_z,
        "subsamples": args.subsamples,
        "geometry_workers": args.geometry_workers,
        "source_dump": str(Path(args.source_dump).resolve()),
        "particle_filter": args.particle_filter,
        "n_particles": len(rows),
        "porosity_psc": fluid_volume / solver.n**3,
        "mean_uz_fluid": mean_uz_fluid,
        "superficial_uz": superficial_uz,
        "max_uz": float(np.max(solver.uz)),
        "permeability_lu": permeability,
        "K_over_skeleton_same_resolution": "",
        "momentum_residual": float(last["momentum_residual"] if last else np.nan),
        "particle_force_sum_x_lu": float(particle_force_sum[0]),
        "particle_force_sum_y_lu": float(particle_force_sum[1]),
        "particle_force_sum_z_lu": float(particle_force_sum[2]),
        "particle_force_norm_sum_lu": float(np.linalg.norm(particle_force_sum)),
        "fluid_imb_sum_x_lu": float(fluid_imb_sum[0]),
        "fluid_imb_sum_y_lu": float(fluid_imb_sum[1]),
        "fluid_imb_sum_z_lu": float(fluid_imb_sum[2]),
        "particle_fluid_momentum_closure_lu": particle_fluid_closure,
        "domain_body_force_z_lu": float(args.force_z * solver.n**3),
        "fluid_volume_body_force_z_lu": float(args.force_z * fluid_volume),
    }
    out = Path(args.outdir) / f"{args.case}_n{args.resolution:03d}.csv"
    write_row(out, row)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
