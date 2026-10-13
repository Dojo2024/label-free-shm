"""
Generates the run manifest for bulk dataset generation: one line per
(event_id, damage_case) combination, "NONE" meaning the healthy model.
30 events x 9 variants (healthy + 8 damage cases) = 270 lines; run_one_th.py
runs one line.
"""
import csv
import os

from fe_model import DAMAGE_TARGETS

HERE = os.path.dirname(__file__)


def event_ids():
    path = os.path.join(HERE, "..", "data", "knet_wsn_matched_events.csv")
    seen = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["EventID"] not in seen:
                seen.append(row["EventID"])
    return seen


if __name__ == "__main__":
    variants = ["NONE"] + list(DAMAGE_TARGETS.keys())
    out_path = os.path.join(HERE, "run_manifest.txt")
    with open(out_path, "w") as f:
        for eid in event_ids():
            for variant in variants:
                f.write(f"{eid},{variant}\n")
    n = len(event_ids()) * len(variants)
    print(f"wrote {n} lines to {out_path}")
