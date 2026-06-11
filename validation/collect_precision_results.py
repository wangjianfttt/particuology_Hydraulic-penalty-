"""Collect completed precision-validation runs into one auditable error table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "workspace" / "validation" / "precision_campaign"
DEFAULT_INPUTS = [
    ROOT / "workspace" / "validation" / "precision_quick",
    ROOT / "workspace" / "validation" / "precision_standard_settling",
    ROOT / "workspace" / "validation" / "precision_standard_hasimoto_n24_n32",
    ROOT / "workspace" / "validation" / "precision_standard_permeability_n24",
]


def read_rows(directory: Path) -> list[dict[str, str]]:
    path = directory / "precision_error_table.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["source_directory"] = str(directory)
        time_ok = row.get("time_converged") == "1"
        force_value = row.get("force_balance_pass", "")
        force_ok = force_value in ("", "1")
        row["result_status"] = (
            "USABLE" if time_ok and force_ok else "FORCE_BALANCE_FAIL" if time_ok else "NOT_TIME_CONVERGED"
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def write_report(path: Path, rows: list[dict[str, str]], command: str) -> None:
    lines = [
        "# Precision validation campaign summary",
        "",
        f"Reproduce: `{command}`",
        "",
        "`USABLE` means time-converged and, where applicable, prescribed-force balance passed.",
        "",
        "| case | profile | N | d | subsamples | steps | abs. error | time | force ratio | effective force ratio | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['profile']} | {row['n']} | {number(row, 'diameter_lu'):.1f} | "
            f"{row['subsamples']} | {row['steps']} | {number(row, 'absolute_relative_error'):.4e} | "
            f"{row['time_converged']} | {number(row, 'force_ratio_tail'):.4f} | "
            f"{number(row, 'effective_force_ratio_tail'):.4f} | {row['result_status']} |"
        )
    lines += [
        "",
        "For the fixed-array permeability check, both force ratios should approach 1.0.",
        "A stable `effective_force_ratio_tail` different from 1.0 indicates that the applied",
        "fluid momentum input differs from the prescribed body force, independent of run length.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [row for directory in args.inputs for row in read_rows(directory)]
    if not rows:
        raise FileNotFoundError("No precision_error_table.csv files found in selected inputs")
    rows.sort(key=lambda row: (row["case"], number(row, "diameter_lu"), number(row, "steps")))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "combined_error_table.csv"
    write_csv(out_csv, rows)
    command = "python validation/collect_precision_results.py --inputs " + " ".join(str(path) for path in args.inputs)
    report = args.out_dir / "campaign_report.md"
    write_report(report, rows, command)
    print(f"wrote {out_csv}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
