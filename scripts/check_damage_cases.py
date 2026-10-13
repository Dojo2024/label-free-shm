"""
Sanity check for the Table 3 damage-case wiring in fe_model.py: build the
healthy calibrated model and each of the 8 damage cases, compare frequencies
mode-by-mode (same eigen index, not by classify_modes direction bucket).

Direction-bucket comparison was tried first and rejected: classify_modes()
labels a mode by argmax of summed |displacement| per direction, and at least
one mode (mode 7, healthy) is a near-degenerate longitudinal/vertical
coupled mode whose label flips across runs even though its frequency barely
moves (0.44535 -> 0.44533 Hz) -- that flip alone produced a spurious "+138%"
longitudinal shift. Comparing by raw mode index avoids the mislabeling
entirely and is valid here because none of these damage magnitudes are large
enough to actually reorder the mode sequence (checked below).

Real damage (stiffness reduction) should only ever lower frequencies.
Single build+eigen per case, ~8ms each -> whole script runs in under a
second, single-threaded.
"""
import json
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fe_model import build_model, get_frequencies, DAMAGE_TARGETS

with open(__file__.rsplit('/', 1)[0] + "/../calibrated_params.json") as f:
    params = json.load(f)

build_model(params)
healthy = get_frequencies(25)

for case in DAMAGE_TARGETS:
    build_model(params, damage_case=case)
    damaged = get_frequencies(25)
    pcts = [100 * (d - h) / h for h, d in zip(healthy, damaged)]
    max_drop = min(pcts)
    max_rise = max(pcts)
    flag = "  <-- unexpected increase, investigate" if max_rise > 0.01 else ""
    print(f"{case:45s} max_drop={max_drop:+7.3f}%  max_rise={max_rise:+7.3f}%  "
          f"mean={sum(pcts) / len(pcts):+6.3f}%{flag}")
