"""Synthetic smoke test for the transient campaign analyzer."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "analysis" / "transient_physical" / "analyze_transient_runs.py"


FIELDS = [
    "case_id",
    "seed",
    "data_file",
    "dump_prefix",
    "coupling_mode",
    "use_si_scaling",
    "macro_step",
    "physical_time_s",
    "dt_lbm_s",
    "dt_dem_s",
    "dem_substeps",
    "lbm_iterations",
    "lbm_residual",
    "lbm_converged",
    "imb_momentum_residual",
    "porosity_psc",
    "mean_uz_lu",
    "superficial_uz_lu",
    "n_particles",
    "n_fines",
    "retained_fraction",
    "force_norm_mean_n",
    "force_norm_max_n",
    "torque_norm_mean_nm",
    "torque_norm_max_nm",
    "force_norm_sliding_average_n",
    "torque_norm_sliding_average_nm",
    "n_fine_contacts",
    "n_fine_fine_contacts",
    "max_fine_contact_force_n",
    "psc_force_scale_n_per_lu",
    "psc_torque_scale_nm_per_lu",
    "velocity_lu_per_m_s",
    "angular_lu_per_rad_s",
    "status",
]


def write_case(
    root: Path,
    case_id: str,
    seed: str,
    contacts: int,
    *,
    write_metadata: bool = True,
    early_stop: bool = False,
) -> None:
    out = root / case_id / seed
    out.mkdir(parents=True)
    stem = f"{case_id}_{seed}_synced"
    csv_path = out / f"{stem}.csv"
    metadata_payload = {
        "case_id": case_id,
        "seed": seed,
        "coupling_regime": "transient",
        "use_si_scaling": True,
        "physical_time_basis": "explicit_lbm_and_dem_transient_time",
        "body_acceleration_z_m_s2": 1000.0,
        "resolution": 32,
        "macro_steps_requested": 2,
        "total_time_s": 2e-5,
        "early_stop_triggered": early_stop,
        "early_stop_reason": "retained_fraction 0 <= 0" if early_stop else "",
    }
    if write_metadata:
        csv_path.with_name(f"{stem}_metadata.json").write_text(json.dumps(metadata_payload), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for step in (1, 2):
            writer.writerow(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "coupling_mode": "transient",
                    "use_si_scaling": "1",
                    "physical_time_s": step * 1e-5,
                    "retained_fraction": 0.0 if early_stop and step == 2 else 1.0,
                    "n_fine_contacts": contacts if step == 2 else 0,
                    "n_fine_fine_contacts": 0,
                    "superficial_uz_lu": 1e-6,
                    "force_norm_mean_n": 2e-9,
                    "status": "ok",
                }
            )


def run_analyzer(root: Path, out: Path) -> list[dict[str, str]]:
    subprocess.run([sys.executable, str(SCRIPT), "--input", str(root), "--out-dir", str(out)], check=True)
    return list(csv.DictReader((out / "run_summary.csv").open()))


def test_analyzer_reads_metadata_runs(tmp_path: Path) -> None:
    write_case(tmp_path, "transient_upstream_low_concentration_low_drive_n32", "260601", 0)
    write_case(tmp_path, "transient_upstream_high_concentration_high_drive_n32", "260601", 3)
    rows = run_analyzer(tmp_path, tmp_path / "out")
    assert len(rows) == 2
    assert {row["provenance"] for row in rows} == {"metadata"}
    assert max(float(row["max_fine_contacts"]) for row in rows) == 3


def test_analyzer_uses_csv_provenance_when_metadata_is_missing(tmp_path: Path) -> None:
    write_case(
        tmp_path,
        "transient_upstream_high_concentration_high_drive_n32_10s",
        "260601",
        0,
        write_metadata=False,
    )
    rows = run_analyzer(tmp_path, tmp_path / "out")
    assert len(rows) == 1
    assert rows[0]["provenance"] == "csv_fields_no_metadata"
    assert rows[0]["final_retained_fraction"] == "1.0"


def test_analyzer_reports_early_stop_metadata(tmp_path: Path) -> None:
    write_case(
        tmp_path,
        "transient_upstream_high_concentration_high_drive_n16_10s_earlystop",
        "260601",
        0,
        early_stop=True,
    )
    rows = run_analyzer(tmp_path, tmp_path / "out")
    assert len(rows) == 1
    assert rows[0]["early_stop_triggered"] == "True"
    assert "retained_fraction" in rows[0]["early_stop_reason"]
    grouped = list(csv.DictReader((tmp_path / "out" / "group_summary.csv").open()))
    assert grouped[0]["n_early_stop"] == "1"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_case(root, "transient_upstream_low_concentration_low_drive_n32", "260601", 0)
        write_case(root, "transient_upstream_high_concentration_high_drive_n32", "260601", 3)
        out = root / "out"
        rows = run_analyzer(root, out)
        assert len(rows) == 2
        assert max(float(row["max_fine_contacts"]) for row in rows) == 3
    print("PASS transient analyzer synthetic test")


if __name__ == "__main__":
    main()
