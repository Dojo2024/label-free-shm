"""
QA pass over generated time-history output CSVs: checks structural
correctness (columns, row count matches the source ground motion's own
sample count, no NaN/Inf) and basic physical plausibility (nonzero,
finite peak, no runaway/instability), and prints a per-variant summary of
peak response so obviously-wrong runs stand out.
"""
from pathlib import Path
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fe_model import SENSOR_LOCATIONS
from knet_parser import read_event

OUT_DIR = str(Path(__file__).resolve().parents[1] / "output" / "complete_simulation_dataset")
RAW_GM_DIR = str(Path(__file__).resolve().parents[1] / "data" / "raw_ground_motion" / "selecteddata" / "knet" / "ascii")

EXPECTED_COLS = ["Time"] + [
    f"Sensor_{sid}_{d}"
    for sid in sorted(k for k, v in SENSOR_LOCATIONS.items())
    for d in ("Long", "Lat", "Vert")
]


def qa_file(path, event_id):
    problems = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if header != EXPECTED_COLS:
        problems.append(f"header mismatch: {len(header)} cols vs expected {len(EXPECTED_COLS)}")

    data = np.array(rows, dtype=np.float64) if rows else np.zeros((0, len(header)))

    if not np.isfinite(data).all():
        n_bad = (~np.isfinite(data)).sum()
        problems.append(f"{n_bad} non-finite values (NaN/Inf)")

    # cross-check row count against the source ground motion's own sample count
    gm = read_event(RAW_GM_DIR, event_id)
    expected_steps = min(len(rec["accel_gal"]) for rec in gm.values()) if gm else None
    if expected_steps is not None and len(rows) != expected_steps:
        problems.append(f"{len(rows)} rows vs expected {expected_steps} (source GM sample count)")

    accel_cols = data[:, 1:] if data.size else data
    peak = float(np.abs(accel_cols).max()) if accel_cols.size else 0.0
    if peak == 0.0:
        problems.append("peak response is exactly zero (suspicious)")
    if peak > 50.0:  # m/s^2 -- >5g structural response would indicate instability, not real damage
        problems.append(f"peak {peak:.2f} m/s^2 (>5g) -- looks like numerical instability")

    return problems, peak, len(rows)


def main():
    variants = sorted(d for d in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, d)))
    total_files = 0
    total_problems = 0
    print(f"{'variant':45s} {'n_files':>8s} {'peak range (m/s^2)':>22s} {'problems':>9s}")
    for variant in variants:
        files = sorted(glob.glob(os.path.join(OUT_DIR, variant, "*.csv")))
        peaks = []
        n_problems = 0
        for path in files:
            event_id = os.path.splitext(os.path.basename(path))[0]
            problems, peak, n_rows = qa_file(path, event_id)
            total_files += 1
            if problems:
                n_problems += 1
                total_problems += 1
                print(f"  PROBLEM {variant}/{event_id}: {'; '.join(problems)}")
            else:
                peaks.append(peak)
        if peaks:
            print(f"{variant:45s} {len(files):8d} {min(peaks):9.4f} - {max(peaks):9.4f} {n_problems:9d}")
        else:
            print(f"{variant:45s} {len(files):8d} {'(no clean files)':>22s} {n_problems:9d}")

    print(f"\n{total_files} files checked, {total_problems} with problems")


if __name__ == "__main__":
    main()
