"""Side-by-side scores of the sbi_bridge.py prior/temperature variants (2026-10-01 runs).

Every variant was scored on the same held-out bridges (500 sparse-prior + 500 hier-prior draws)
and the same 20 draws per Table 3 case, so the columns are paired.

    python scripts/compare_sbi_variants.py
"""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results" / "sbi_bridge"
TAGS = ["sparse_notemp", "sparse_temp0.3", "hier_notemp", "hier_temp0.1", "hier_temp0.3", "hier_temp0.6"]
TYPES = {"B": "bearings", "P": "piers", "G": "girders", "C": "cables"}
CASES = {"healthy": "healthy", "damage_case_1_bearing_minor_local": "1 bearing minor local",
         "damage_case_2_girder_moderate_local": "2 girder moderate local",
         "damage_case_3_bearing_severe_widespread": "3 bearing severe widespread",
         "damage_case_5_foundation_moderate_local": "5 foundation moderate local",
         "damage_case_7_cable_severe_widespread": "7 cable severe widespread",
         "damage_case_8_multiple_mixed_damage": "8 multiple mixed"}

res = {t: json.loads((R / f"results_{t}_n200000.json").read_text()) for t in TAGS if (R / f"results_{t}_n200000.json").exists()}
cols = list(res)
w = 15
print("columns:", ", ".join(cols))

for test in ["sparse", "hier"]:
    print(f"\n== held-out test set drawn from the {test} prior: detection AUROC / 90% coverage, by part type")
    print(f"{'':10s}" + "".join(f"{c:>{w}s}" for c in cols))
    for t, name in TYPES.items():
        print(f"{name:10s}" + "".join(f"{res[c]['test'][test]['by_type'][t]['auroc']:>8.3f} /{res[c]['test'][test]['by_type'][t]['coverage_90']:.2f}" for c in cols))

for cond in ["nominal", "random"]:
    print(f"\n== Table 3 cases, {cond} environment/model (20 draws each)")
    print("   dmg P = mean P(damaged) on the truly damaged zones; in90 = mean share of draws with the truth inside")
    print("   the 90% interval (worst damaged zone in brackets); FA = highest P(damaged) on an intact zone")
    for key, name in CASES.items():
        print(f" {name}")
        for c in cols:
            r = res[c]["cases"][f"{cond}/{key}"]
            truth, pd, in90 = (np.array(r[k]) for k in ("truth", "p_damaged", "truth_in_90"))
            dmg = truth < 0.9
            s = f"   {c:15s} FA {pd[~dmg].max():.2f}"
            if dmg.any():
                s += f" | dmg P {pd[dmg].mean():.2f} | in90 {in90[dmg].mean():.2f} (worst {in90[dmg].min():.2f})"
            print(s)
