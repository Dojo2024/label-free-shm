"""
Transient time-history analysis: drives the calibrated FE model (healthy or
one of the 8 Table 3 damage cases) with a real K-NET ground-motion record,
and writes per-sensor acceleration output to CSV.

Ground-motion component -> model direction mapping is an ASSUMPTION, not
measured: EW -> X (longitudinal, dir=1), NS -> Y (lateral, dir=2),
UD -> Z (vertical, dir=3). We don't have the bridge's true compass
orientation relative to K-NET's EW/NS axes, so this treats the recording
station's fixed compass frame as if it lined up with the bridge's own
long/lateral axes -- a real simplification worth revisiting if azimuth data
ever surfaces, not hidden here.

Rayleigh damping uses an ASSUMED 3% critical damping ratio (typical range
for a steel cable-stayed bridge; not a measured value -- no measured modal
damping exists in the source material) targeted at the first and
`num_modes_for_damping`-th global modes.
"""
from pathlib import Path
import csv
import os
import sys
import tempfile

import numpy as np
import openseespy.opensees as ops

sys.path.insert(0, os.path.dirname(__file__))
from fe_model import build_model, get_frequencies, sensor_output_nodes

RAW_GM_DIR = str(Path(__file__).resolve().parents[1] / "data" / "raw_ground_motion" / "selecteddata" / "knet" / "ascii")
DAMPING_RATIO = 0.03
COMPONENT_DIR = {"EW": 1, "NS": 2, "UD": 3}


def run_time_history(event_id, damage_case=None, params=None, out_csv=None,
                      num_modes_for_damping=2, max_steps=None):
    from knet_parser import read_event

    components = read_event(RAW_GM_DIR, event_id)
    if len(components) != 3:
        raise FileNotFoundError(
            f"expected 3 components (EW/NS/UD) for {event_id} in {RAW_GM_DIR}, "
            f"found {sorted(components.keys())}"
        )

    model = build_model(params, damage_case=damage_case)

    freqs = get_frequencies(num_modes_for_damping)
    w1, w2 = 2 * np.pi * freqs[0], 2 * np.pi * freqs[num_modes_for_damping - 1]
    alpha_m = DAMPING_RATIO * 2 * w1 * w2 / (w1 + w2)
    beta_k = DAMPING_RATIO * 2 / (w1 + w2)
    ops.rayleigh(alpha_m, 0.0, beta_k, 0.0)

    dts = {rec["dt"] for rec in components.values()}
    assert len(dts) == 1, f"components have mismatched dt: {dts}"
    dt = dts.pop()
    n_steps = min(len(rec["accel_gal"]) for rec in components.values())
    if max_steps:
        n_steps = min(n_steps, max_steps)

    for i, (comp, rec) in enumerate(sorted(components.items()), start=1):
        accel_m_s2 = (rec["accel_gal"][:n_steps] / 100.0).tolist()  # gal -> m/s^2
        ops.timeSeries("Path", i, "-dt", dt, "-values", *accel_m_s2)
        ops.pattern("UniformExcitation", i, COMPONENT_DIR[comp], "-accel", i)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-6, 10)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    sensor_nodes = sensor_output_nodes(model)
    sensor_ids = sorted(sensor_nodes.keys())
    node_tags = [sensor_nodes[sid] for sid in sensor_ids]

    rec_fd, rec_path = tempfile.mkstemp(suffix=".recorder_tmp.txt")
    os.close(rec_fd)
    ops.recorder("Node", "-file", rec_path, "-time", "-node", *node_tags,
                 "-dof", 1, 2, 3, "accel")

    for step in range(n_steps):
        ok = ops.analyze(1, dt)
        if ok != 0:
            ops.remove("recorders")
            raise RuntimeError(
                f"transient analysis failed to converge at step {step}, "
                f"t={step * dt:.3f}s (event={event_id}, damage_case={damage_case})"
            )
    ops.remove("recorders")

    raw = np.loadtxt(rec_path)
    os.remove(rec_path)

    header = ["Time"] + [f"Sensor_{sid}_{d}" for sid in sensor_ids for d in ("Long", "Lat", "Vert")]
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(raw.tolist())

    return header, raw


if __name__ == "__main__":
    import json
    import time

    params = json.load(open(os.path.join(os.path.dirname(__file__), "..", "calibrated_params.json")))
    t0 = time.time()
    header, raw = run_time_history(
        "IBR0071809050511", damage_case=None, params=params,
        out_csv="/tmp/test_time_history.csv", max_steps=500,
    )
    print(f"{raw.shape[0]} steps x {raw.shape[1]} cols in {time.time() - t0:.1f}s "
          f"(max_steps=500 smoke test)")
    print(f"peak |accel| across all sensors/dirs: {np.abs(raw[:, 1:]).max():.4f} m/s^2")
