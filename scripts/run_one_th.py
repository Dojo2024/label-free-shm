"""
One dataset-generation task: runs a full-duration time-history analysis for
one (event_id, damage_case) pair and writes the per-sensor CSV. Run once per
line of the gen_run_manifest.py manifest -- kept a standalone script
(not a function import) so a single run's crash/OOM can't take down others.

Usage: run_one_th.py <event_id> <damage_case|NONE>
"""
from pathlib import Path
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from time_history import run_time_history

OUT_DIR = str(Path(__file__).resolve().parents[1] / "output" / "complete_simulation_dataset")


def main():
    event_id, damage_case = sys.argv[1], sys.argv[2]
    damage_case = None if damage_case == "NONE" else damage_case

    params = json.load(open(os.path.join(os.path.dirname(__file__), "..", "calibrated_params.json")))

    variant_dir = os.path.join(OUT_DIR, damage_case or "healthy_baseline")
    os.makedirs(variant_dir, exist_ok=True)
    out_csv = os.path.join(variant_dir, f"{event_id}.csv")
    if os.path.exists(out_csv):
        print(f"skip (exists): {out_csv}")
        return

    run_time_history(event_id, damage_case=damage_case, params=params, out_csv=out_csv)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
