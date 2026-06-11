"""Analyze true-SI transient PSC-IMB-DEM campaign outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "coupling_sync" / "runs" / "transient_physical_factorial_n32"
DEFAULT_OUT = ROOT / "analysis" / "transient_physical" / "outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def load_run(path: Path) -> dict[str, object] | None:
    meta_path = path.with_name(path.stem + "_metadata.json")
    rows = read_csv(path)
    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return None
    final = valid[-1]
    provenance = "metadata"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("coupling_regime") != "transient":
            raise ValueError(f"{path} is not transient: {meta.get('coupling_regime')}")
        if not bool(meta.get("use_si_scaling")):
            raise ValueError(f"{path} is not SI-scaled")
        if meta.get("physical_time_basis") != "explicit_lbm_and_dem_transient_time":
            raise ValueError(f"{path} has invalid time basis: {meta.get('physical_time_basis')}")
    else:
        meta = {
            "case_id": final.get("case_id", ""),
            "seed": final.get("seed", ""),
            "body_acceleration_z_m_s2": math.nan,
            "resolution": final.get("resolution", ""),
            "macro_steps_requested": "",
        }
        provenance = "csv_fields_no_metadata"
        if final.get("coupling_mode") != "transient":
            raise ValueError(f"{path} is not transient: {final.get('coupling_mode')}")
        if final.get("use_si_scaling") not in {"1", "1.0", "true", "True"}:
            raise ValueError(f"{path} is not SI-scaled")
    case_id = str(meta["case_id"])
    concentration = "high" if "high_concentration" in case_id else "low"
    drive = "high" if "high_drive" in case_id else "low"
    contacts = [as_float(row, "n_fine_contacts", 0.0) for row in valid]
    fine_fine = [as_float(row, "n_fine_fine_contacts", 0.0) for row in valid]
    first_contact_time = math.nan
    for row, count in zip(valid, contacts):
        if count > 0:
            first_contact_time = as_float(row, "physical_time_s")
            break
    return {
        "path": str(path),
        "provenance": provenance,
        "case_id": case_id,
        "seed": str(meta.get("seed", "")),
        "concentration": concentration,
        "drive": drive,
        "body_acceleration_z_m_s2": meta.get("body_acceleration_z_m_s2", math.nan),
        "resolution": meta.get("resolution", ""),
        "requested_time_s": meta.get("total_time_s", math.nan),
        "requested_macro_steps": meta.get("macro_steps_requested", ""),
        "completed_macro_steps": len(valid),
        "completed_time_s": as_float(final, "physical_time_s"),
        "completed_fraction_of_request": (
            as_float(final, "physical_time_s") / float(meta.get("total_time_s", math.nan))
            if meta.get("total_time_s") not in ("", None, 0)
            else math.nan
        ),
        "early_stop_triggered": bool(meta.get("early_stop_triggered", False)),
        "early_stop_reason": meta.get("early_stop_reason", ""),
        "final_status": final.get("status", ""),
        "final_retained_fraction": as_float(final, "retained_fraction"),
        "final_fine_contacts": as_float(final, "n_fine_contacts", 0.0),
        "max_fine_contacts": max(contacts),
        "final_fine_fine_contacts": as_float(final, "n_fine_fine_contacts", 0.0),
        "max_fine_fine_contacts": max(fine_fine),
        "first_contact_time_s": first_contact_time,
        "final_superficial_uz_lu": as_float(final, "superficial_uz_lu"),
        "final_force_norm_mean_n": as_float(final, "force_norm_mean_n"),
        "max_force_norm_mean_n": max(as_float(row, "force_norm_mean_n", 0.0) for row in valid),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["concentration"]), str(row["drive"])), []).append(row)
    out = []
    metrics = [
        "completed_macro_steps",
        "completed_time_s",
        "completed_fraction_of_request",
        "final_retained_fraction",
        "max_fine_contacts",
        "max_fine_fine_contacts",
        "max_force_norm_mean_n",
    ]
    for (conc, drive), items in sorted(grouped.items()):
        entry: dict[str, object] = {"concentration": conc, "drive": drive, "n_runs": len(items)}
        entry["n_early_stop"] = sum(1 for item in items if bool(item.get("early_stop_triggered", False)))
        for metric in metrics:
            vals = [float(item[metric]) for item in items if not math.isnan(float(item[metric]))]
            entry[f"{metric}_mean"] = mean(vals) if vals else math.nan
            entry[f"{metric}_sd"] = stdev(vals) if len(vals) > 1 else 0.0
        out.append(entry)
    return out


def write_markdown(path: Path, rows: list[dict[str, object]], grouped: list[dict[str, object]]) -> None:
    lines = [
        "# True-SI transient campaign summary",
        "",
        "All included runs passed provenance checks: `coupling_regime=transient`, `use_si_scaling=true`, and `physical_time_basis=explicit_lbm_and_dem_transient_time`.",
        "",
        f"Included run count: {len(rows)}",
        "",
        "| concentration | drive | n | early stops | completed steps mean | completed time mean (s) | completed/requested mean | retained mean | max fine contacts mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(
            f"| {row['concentration']} | {row['drive']} | {row['n_runs']} | {row['n_early_stop']} | "
            f"{float(row['completed_macro_steps_mean']):.1f} | {float(row['completed_time_s_mean']):.3e} | "
            f"{float(row['completed_fraction_of_request_mean']):.3f} | "
            f"{float(row['final_retained_fraction_mean']):.3f} | {float(row['max_fine_contacts_mean']):.1f} |"
        )
    lines += [
        "",
        "Rows marked as early stops ended after a specified local-window condition, such as retained_fraction reaching zero.",
        "Such rows should be interpreted as completed local-window outcomes, not as residence for the full requested horizon.",
        "Runs without early stop and with completed/requested below one remain partial transient results.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.input.rglob("*_synced.csv")):
        loaded = load_run(path)
        if loaded:
            rows.append(loaded)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_summary(rows)
    write_csv(args.out_dir / "run_summary.csv", rows)
    write_csv(args.out_dir / "group_summary.csv", grouped)
    write_markdown(args.out_dir / "transient_campaign_summary.md", rows, grouped)
    print(f"wrote {len(rows)} run summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
