"""Merge per-resolution PSC-IMB permeability CSV files.

The convergence driver writes one ``case_nXXX.csv`` file per completed run.
This helper rebuilds the raw convergence table from those independent files so
that high-resolution points computed on another machine can be copied back
without overwriting the existing 64/96/128 campaign.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "convergence" / "permeability"

SUMMARY_FIELDS = [
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


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_one_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise ValueError(f"{path} should contain exactly one data row")
    return rows[0]


def collect_rows(input_dir: Path) -> list[dict[str, str]]:
    pattern = re.compile(r"^(?P<case>.+)_n(?P<n>\d{3})\.csv$")
    rows: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("*_n*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        row = read_one_row(path)
        rows.append(row)
    return sorted(rows, key=lambda r: (r["case"], int(r["resolution"])))


def add_raw_ratios(rows: list[dict[str, str]]) -> None:
    skeleton_by_resolution: dict[str, float] = {}
    for row in rows:
        if row.get("case") == "skeleton":
            skeleton_by_resolution[row["resolution"]] = float(row["permeability_lu"])
    for row in rows:
        try:
            row["K_over_skeleton_same_resolution"] = repr(
                float(row["permeability_lu"]) / skeleton_by_resolution[row["resolution"]]
            )
        except (KeyError, ValueError, ZeroDivisionError):
            row["K_over_skeleton_same_resolution"] = ""


def observed_order(h1: float, h2: float, h3: float, k1: float, k2: float, k3: float) -> float:
    e12 = k1 - k2
    e23 = k2 - k3
    if e12 == 0.0 or e23 == 0.0 or e12 * e23 <= 0.0:
        return 2.0
    target = abs(e12 / e23)

    def ratio(p: float) -> float:
        return (h1**p - h2**p) / max(h2**p - h3**p, 1.0e-300)

    lo, hi = 0.1, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ratio(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def richardson_for_case(rows: list[dict[str, str]], case: str) -> dict[str, object]:
    case_rows = sorted([r for r in rows if r.get("case") == case], key=lambda r: int(r["resolution"]))
    if len(case_rows) < 3:
        return {
            "case": case,
            "coarse_resolution": "",
            "mid_resolution": "",
            "fine_resolution": "",
            "richardson_order": "",
            "permeability_extrapolated_lu": "",
            "fine_grid_permeability_lu": "",
            "relative_fine_to_extrapolated": "",
            "K_over_skeleton_extrapolated": "",
        }
    selected = case_rows[-3:]
    n1, n2, n3 = [int(r["resolution"]) for r in selected]
    k1, k2, k3 = [float(r["permeability_lu"]) for r in selected]
    h1, h2, h3 = 1.0 / n1, 1.0 / n2, 1.0 / n3
    p = observed_order(h1, h2, h3, k1, k2, k3)
    denom = h2**p - h3**p
    k_inf = (k3 * h2**p - k2 * h3**p) / denom if abs(denom) > 1.0e-300 else k3
    rel = abs(k3 - k_inf) / max(abs(k_inf), 1.0e-300)
    return {
        "case": case,
        "coarse_resolution": n1,
        "mid_resolution": n2,
        "fine_resolution": n3,
        "richardson_order": p,
        "permeability_extrapolated_lu": k_inf,
        "fine_grid_permeability_lu": k3,
        "relative_fine_to_extrapolated": rel,
        "K_over_skeleton_extrapolated": "",
    }


def add_extrapolated_ratios(rows: list[dict[str, object]]) -> None:
    skeleton = next((r for r in rows if r["case"] == "skeleton"), None)
    if not skeleton:
        return
    try:
        k_skeleton = float(skeleton["permeability_extrapolated_lu"])
    except (TypeError, ValueError):
        return
    for row in rows:
        try:
            row["K_over_skeleton_extrapolated"] = float(row["permeability_extrapolated_lu"]) / k_skeleton
        except (TypeError, ValueError, ZeroDivisionError):
            row["K_over_skeleton_extrapolated"] = ""


def main() -> None:
    rows = collect_rows(IN_DIR)
    add_raw_ratios(rows)
    raw_path = IN_DIR / "permeability_convergence_raw.csv"
    write_rows(raw_path, SUMMARY_FIELDS, rows)

    cases = sorted({row["case"] for row in rows})
    extrap_rows = [richardson_for_case(rows, case) for case in cases]
    add_extrapolated_ratios(extrap_rows)
    extrap_fields = [
        "case",
        "coarse_resolution",
        "mid_resolution",
        "fine_resolution",
        "richardson_order",
        "permeability_extrapolated_lu",
        "fine_grid_permeability_lu",
        "relative_fine_to_extrapolated",
        "K_over_skeleton_extrapolated",
    ]
    richardson_path = IN_DIR / "permeability_richardson.csv"
    write_rows(richardson_path, extrap_fields, extrap_rows)
    print(f"wrote {raw_path}")
    print(f"wrote {richardson_path}")


if __name__ == "__main__":
    main()
