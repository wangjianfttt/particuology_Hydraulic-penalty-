"""Run synchronized PSC-IMB-DEM cases from a CSV manifest.

The manifest is intentionally plain CSV so the same file can drive local
checks, Slurm array jobs, and later manuscript provenance audits.  This
wrapper does not invent results: rows are either executed by calling
``couple_psc_liggghts.py`` or listed with ``--dry-run``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "coupling_sync" / "synced_campaign_cases.csv"
DEFAULT_OUTDIR = ROOT / "coupling_sync" / "runs" / "campaign"
BOX_SIZE_M = 0.009
CS2 = 1.0 / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--case-id", default="", help="Only run rows with this case_id.")
    parser.add_argument("--seed", default="", help="Only run rows with this seed.")
    parser.add_argument("--row-index", type=int, default=None, help="Zero-based manifest row index for Slurm arrays.")
    parser.add_argument("--start-index", type=int, default=None, help="First zero-based manifest row index to run.")
    parser.add_argument("--end-index", type=int, default=None, help="Last zero-based manifest row index to run, inclusive.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of selected rows to run.")
    parser.add_argument("--local-sample", action="store_true", help="Override selected row to a short laptop-scale check.")
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python executable used to launch couple_psc_liggghts.py; defaults to PYTHON_BIN or this interpreter.",
    )
    parser.add_argument("--skip-completed", action="store_true", help="Skip rows whose output CSV already has an ok final row.")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete rows from their latest DEM checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but do not execute them.")
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def selected_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[tuple[int, dict[str, str]]]:
    indexed = list(enumerate(rows))
    if args.row_index is not None:
        indexed = [item for item in indexed if item[0] == args.row_index]
    if args.start_index is not None:
        indexed = [item for item in indexed if item[0] >= args.start_index]
    if args.end_index is not None:
        indexed = [item for item in indexed if item[0] <= args.end_index]
    if args.case_id:
        indexed = [item for item in indexed if item[1].get("case_id") == args.case_id]
    if args.seed:
        indexed = [item for item in indexed if item[1].get("seed") == args.seed]
    if args.limit is not None:
        indexed = indexed[: args.limit]
    return indexed


def output_csv_for(row: dict[str, str], args: argparse.Namespace) -> Path:
    return args.outdir / row["case_id"] / row["seed"] / f"{row['dump_prefix']}_synced.csv"


def physical_lattice_dt(row: dict[str, str]) -> float:
    """Return the viscosity-matched physical duration of one lattice step."""
    resolution = int(row["resolution"])
    tau = float(row.get("tau", "0.72") or "0.72")
    temperature_k = float(row.get("temperature_k", "773.15") or "773.15")
    pressure_pa = float(row.get("pressure_pa", "2.0e5") or "2.0e5")
    mu_he = float(row.get("mu_he", "3.7e-5") or "3.7e-5")
    rho_he = pressure_pa / (2077.1 * temperature_k)
    nu_phys = mu_he / rho_he
    nu_lu = CS2 * (tau - 0.5)
    dx = BOX_SIZE_M / resolution
    return nu_lu * dx * dx / nu_phys


def exchange_dt(row: dict[str, str]) -> float:
    if row.get("coupling_mode", "quasi_steady").strip() == "transient":
        return physical_lattice_dt(row) * int(row.get("lbm_steps_per_exchange", "1") or "1")
    return float(row["dt_lbm_s"])


def expected_macro_steps(row: dict[str, str], args: argparse.Namespace) -> int:
    total_time = float(row["total_time_s"])
    dt_lbm = exchange_dt(row)
    if args.local_sample:
        total_time = 9.6e-8
    return int(math.ceil(total_time / dt_lbm))


def completed_output(path: Path, row: dict[str, str], args: argparse.Namespace) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except csv.Error:
        return False
    if not rows:
        return False
    metadata_path = path.with_name(path.stem + "_metadata.json")
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    requested_mode = row.get("coupling_mode", "quasi_steady").strip() or "quasi_steady"
    checks = [
        metadata.get("coupling_regime") == requested_mode,
        bool(metadata.get("use_si_scaling")) == as_bool(row.get("use_si_scaling", "0")),
        int(metadata.get("resolution", -1)) == int(row["resolution"]),
        abs(float(metadata.get("youngs_modulus_pa", -1.0)) - float(row["youngs_modulus_pa"])) <= 1.0,
    ]
    if requested_mode == "transient":
        checks.extend(
            [
                metadata.get("body_acceleration_z_m_s2") is not None,
                abs(float(metadata["body_acceleration_z_m_s2"]) - float(row["body_acceleration_z"])) <= 1.0e-12,
                int(metadata.get("lbm_steps_per_exchange", -1))
                == int(row.get("lbm_steps_per_exchange", "1") or "1"),
            ]
        )
    early_stop_ok = bool(metadata.get("early_stop_triggered")) and rows[-1].get("status") == "ok"
    full_run_ok = rows[-1].get("status") == "ok" and len(rows) >= expected_macro_steps(row, args)
    return all(checks) and (full_run_ok or early_stop_ok)


def command_for(row_index: int, row: dict[str, str], args: argparse.Namespace) -> list[str]:
    outdir = args.outdir / row["case_id"] / row["seed"]
    outdir.mkdir(parents=True, exist_ok=True)
    total_time = row["total_time_s"]
    resolution = row["resolution"]
    psc_min = row["psc_min_steps"]
    psc_max = row["psc_max_steps"]
    if args.local_sample:
        total_time = "9.6e-8"
        resolution = "16"
        psc_min = "20"
        psc_max = "80"
    output_csv = output_csv_for(row, args)
    cmd = [
        args.python_bin,
        str(ROOT / "coupling_sync" / "couple_psc_liggghts.py"),
        "--case-id",
        row["case_id"],
        "--seed",
        row["seed"],
        "--data-file",
        row["data_file"],
        "--dump-prefix",
        row["dump_prefix"],
        "--T_total",
        total_time,
        "--dt_LBM",
        row["dt_lbm_s"],
        "--auto-subcycles",
        row["auto_subcycles"],
        "--resolution",
        resolution,
        "--psc-min-steps",
        psc_min,
        "--psc-max-steps",
        psc_max,
        "--lbm-residual-threshold",
        row["residual_tol"],
        "--force-z",
        row["force_z"],
        "--youngs-modulus",
        row["youngs_modulus_pa"],
        "--output-csv",
        str(output_csv),
        "--metadata-json",
        str(output_csv.with_name(output_csv.stem + "_metadata.json")),
        "--schema-csv",
        str(output_csv.with_name(output_csv.stem + "_schema.csv")),
        "--checkpoint-dir",
        str(output_csv.with_name(output_csv.stem + "_checkpoints")),
        "--checkpoint-every",
        row.get("checkpoint_every", "1"),
    ]
    optional_value_args = {
        "coupling_mode": "--coupling-mode",
        "lbm_steps_per_exchange": "--lbm-steps-per-exchange",
        "dt_dem_max": "--dt-dem-max",
        "body_acceleration_z": "--body-acceleration-z",
        "tau": "--tau",
        "temperature_k": "--temperature-k",
        "pressure_pa": "--pressure-pa",
        "mu_he": "--mu-he",
        "velocity_feedback_factor": "--velocity-feedback-factor",
        "angular_feedback_factor": "--angular-feedback-factor",
        "stop_when_retained_below": "--stop-when-retained-below",
        "min_macro_before_stop": "--min-macro-before-stop",
    }
    for field, flag in optional_value_args.items():
        value = row.get(field, "").strip()
        if value:
            cmd.extend([flag, value])
    if args.resume:
        cmd.append("--resume")
    if as_bool(row.get("apply_psc_torque", "0")):
        cmd.append("--apply-psc-torque")
    if as_bool(row.get("use_si_scaling", "0")):
        cmd.append("--use-si-scaling")
    if as_bool(row.get("allow_unscaled_regression", "0")):
        cmd.append("--allow-unscaled-regression")
    return cmd


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest)
    picked = selected_rows(rows, args)
    if not picked:
        raise SystemExit("No manifest rows selected.")
    for row_index, row in picked:
        output_csv = output_csv_for(row, args)
        if args.skip_completed and completed_output(output_csv, row, args):
            print(f"skip completed row {row_index}: {output_csv}", flush=True)
            continue
        cmd = command_for(row_index, row, args)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
