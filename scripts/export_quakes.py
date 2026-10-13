"""
Exports a 6-record subset of the simulated earthquake dataset
(output/complete_simulation_dataset) to JSON for the interactive viewer:
pick a record, a damage case and a sensor channel, compare healthy vs damaged
time histories and Welch spectra.

    cd scripts && VIRTUAL_ENV= ../.venv/bin/python export_quakes.py              # index.json
    cd scripts && VIRTUAL_ENV= ../.venv/bin/python export_quakes.py <EventID>    # <EventID>.json

One record per call (9 CSVs of ~7 MB each, a few seconds), so every call stays
short on a laptop. Writes figures/interactive/data/quakes/.

What the data is (checked against time_history.py, which made the dataset):
- ground input: the K-NET IBR007 record, mean-removed (knet_parser), gal/100
  -> m/s^2 with no other scaling, cut to the shortest component, applied as
  UniformExcitation EW -> longitudinal, NS -> lateral, UD -> vertical (an
  assumed mapping). Sample i is at t = i*dt.
- sensor output: OpenSees `recorder Node ... accel` without `-timeSeries`,
  which under UniformExcitation is acceleration RELATIVE to the ground, not
  absolute. First row is t = dt (the recorder writes after each step).
- the FE model is a centerline idealization, so the 60 channels hold only 36
  distinct series (left/right sensor pairs and the pylon sensors S27/S28 share
  the deck node of S26/S32). Each distinct series is stored once, under the
  lowest sensor id; `alias` maps every other channel to it.

Series are decimated 100 -> 10 Hz (scipy.signal.decimate, q=10, order-8
Chebyshev I, zero_phase=True; passband edge 4 Hz) and stored as integers
q with value = q * scale, scale = peak / 9999 (4 significant digits at the
series' own peak). Welch PSDs use make_figures.fig_time_history's settings on
the raw 100 Hz series and are kept for 0.1-2.5 Hz. Peak and RMS are from the
raw 100 Hz series.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate, welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_model as fm
from knet_parser import read_event
from sbi_bridge import REF_MODES
from time_history import COMPONENT_DIR, RAW_GM_DIR

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "output" / "complete_simulation_dataset"
OUT = ROOT / "figures" / "interactive" / "data" / "quakes"
CAL = json.loads((ROOT / "calibrated_params.json").read_text())

# Spans M 3.7-6.7 and catalog PGA ~7-130 gal (index.json also has PGA recomputed from the
# waveforms); IBR0071906170800 is figures/05_time_history.png's record.
SELECTED = [
    "IBR0071906170800",  # M5.1, 77 km, 129 gal -- largest PGA of the 30
    "IBR0071912041038",  # M4.9, 9 km, 55 gal -- shallow, 2nd largest PGA
    "IBR0071809050511",  # M5.5, 60 km, 37 gal -- time_history.py's smoke-test record
    "IBR0071907280331",  # M6.6, 393 km, 14 gal -- deep event, longest record (132 s)
    "IBR0071906182222",  # M6.7, 14 km, 7 gal -- largest magnitude, weak shaking at IBR007
    "IBR0071907201027",  # M3.7, 56 km, 8 gal -- smallest magnitude
]
CASES = ["healthy_baseline"] + sorted(fm.DAMAGE_TARGETS)
DIRS = {"Long": "longitudinal", "Lat": "lateral", "Vert": "vertical"}
SENSORS = sorted(fm.SENSOR_LOCATIONS)
CHANNELS = [f"Sensor_{s}_{d}" for s in SENSORS for d in DIRS]
PIER_X = {"P31": fm.X_P31, "P32": fm.X_P32, "P33": fm.X_P33, "PYLON": fm.X_PYLON, "P35": fm.X_P35}
PAPER_NAMES = ["1GVe", "1GLo", "1GLa", "2GVe", "3GVe", "4GVe", "2GLa", "5GVe"]  # order of REF_MODES
FIELD_HZ = [0.38, 0.42, 0.51, 0.72, 1.08, 1.29, 1.43, 1.62]
NAME_DIR = {"Ve": "vertical", "Lo": "longitudinal", "La": "lateral"}
WELCH = dict(nperseg=4096)  # make_figures.fig_time_history; scipy defaults otherwise (hann, 50% overlap)
PSD_BAND = (0.1, 2.5)
Q = 10
QMAX = 9999


def quantize(x):
    scale = float(f"{np.abs(x).max() / QMAX:.6g}")
    return {"scale": scale, "q": np.rint(x / scale).astype(int).tolist()}


def sig4(x):
    return [float(f"{v:.4g}") for v in x]


def write(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))
    print(f"wrote {path.relative_to(ROOT)}  {path.stat().st_size / 1e6:.2f} MB")


def ground(event_id):
    """The three K-NET components exactly as time_history.run_time_history applies them."""
    comps = read_event(RAW_GM_DIR, event_id)
    assert sorted(comps) == ["EW", "NS", "UD"], sorted(comps)
    dts = {c["dt"] for c in comps.values()}
    assert len(dts) == 1, dts
    n = min(len(c["accel_gal"]) for c in comps.values())
    return comps, dts.pop(), n


def event_meta(event_id, row):
    comps, dt, n = ground(event_id)
    h = comps["EW"]["header"]
    assert float(h["Mag."]) == row.Magnitude and float(h["Depth. (km)"]) == row.Depth_km, event_id
    jst = datetime.strptime(h["Origin Time"], "%Y/%m/%d %H:%M:%S")
    pga = {c: float(np.abs(comps[c]["accel_gal"][:n]).max()) for c in comps}
    return {
        "event_id": event_id,
        "station": h["Station Code"],
        # K-NET headers give origin time in JST; the events CSV's DateTime_UTC column is this same JST value.
        "origin_time_jst": jst.strftime("%Y-%m-%d %H:%M:%S"),
        "origin_time_utc": (jst - timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"),
        "magnitude": float(row.Magnitude),
        "depth_km": float(row.Depth_km),
        "epicenter_lat": float(h["Lat."]), "epicenter_lon": float(h["Long."]),
        "station_lat": float(h["Station Lat."]), "station_lon": float(h["Station Long."]),
        "pga_gal": round(max(pga.values()), 2),
        "pga_m_s2": round(max(pga.values()) / 100, 4),
        "pga_gal_by_component": {c: round(v, 2) for c, v in pga.items()},
        "pga_gal_catalog": round(float(row.PGA_gal), 2),
        "duration_s": round(n * dt, 2),
        "duration_s_catalog": float(row.Duration_s),
    }


def index():
    events = pd.read_csv(ROOT / "data" / "knet_wsn_matched_events.csv").set_index("EventID")
    for case in CASES:
        missing = [e for e in SELECTED if not (DATASET / case / f"{e}.csv").exists()]
        assert not missing, (case, missing)

    p = {**fm.DEFAULT_PARAMS, **CAL}
    anchors = sorted(x for side in fm.cable_anchor_xs(p).values() for x in side)
    locs = {**PIER_X, "all": None}

    def desc(spec):
        if spec["target"] == "multiple":
            return " + ".join(desc(s) for s in spec["specs"])
        k = f"k x {1 - spec['factor']:.2f}"
        if spec["target"] == "bearing":
            return f"bearing springs (all 6 DOF) at {', '.join(spec['piers'])}: {k}"
        if spec["target"] == "pier":
            return f"pier {', '.join(spec['piers'])} section (A, Iy, Iz): {k}"
        where = spec["near"]
        if spec["target"] == "girder":
            span = "the whole deck" if where == "all" else f"deck within {fm.NEAR_WINDOW:.0f} m of {where}"
            return f"girder section (A, Iy, Iz) over {span}: {k}"
        hit = len(anchors) if where == "all" else sum(abs(x - locs[where]) <= fm.NEAR_WINDOW for x in anchors)
        which = "all" if where == "all" else f"within {fm.NEAR_WINDOW:.0f} m of {where}"
        return f"stay-cable area, {hit} of {len(anchors)} cables ({which}): {k}"

    cases = [{"id": "healthy_baseline", "label": "Healthy", "desc": "calibrated model, no damage"}]
    for case in CASES[1:]:
        n = case.split("_")[2]
        cases.append({"id": case, "label": f"Case {n}: {' '.join(case.split('_')[3:])}",
                      "desc": desc(fm.DAMAGE_TARGETS[case])})

    m = fm.build_model(CAL)
    nodes = fm.sensor_output_nodes(m)
    node_x = {t: x for x, t in m["deck_nodes"].items()}
    first = {}
    channels = []
    for s in SENSORS:
        loc = fm.SENSOR_LOCATIONS[s]
        first.setdefault(nodes[s], s)
        pier = loc.get("pier")
        for d in DIRS:
            channels.append({
                "channel": f"Sensor_{s}_{d}", "sensor": s, "label": f"S{s} {DIRS[d]}",
                "dir": d, "direction": DIRS[d], "kind": loc["kind"],
                "x": loc["x"] if pier is None else PIER_X[pier],
                "x_model": node_x.get(nodes[s], PIER_X.get(pier)),
                "pier": pier,
                "data": f"Sensor_{first[nodes[s]]}_{d}",
            })

    modes = fm.classify_modes(m, max(REF_MODES))
    identified = []
    for name, field, i in zip(PAPER_NAMES, FIELD_HZ, REF_MODES):
        mm = modes[i - 1]
        assert mm["direction"] == NAME_DIR[name[-2:]], (name, mm)
        identified.append({"name": name, "fe_mode": i, "freq_hz": round(mm["freq_hz"], 4),
                           "field_hz": field, "direction": mm["direction"]})
    fe_only = [{"fe_mode": mm["mode"], "freq_hz": round(mm["freq_hz"], 4), "direction": mm["direction"]}
               for mm in modes if mm["mode"] not in REF_MODES]

    write(OUT / "index.json", {
        "records": [event_meta(e, events.loc[e]) for e in SELECTED],
        "cases": cases,
        "channels": channels,
        "modes": identified,
        "fe_only_modes": fe_only,
        "notes": {
            "response": "acceleration relative to the ground, m/s^2 (OpenSees UniformExcitation, "
                        "recorder without -timeSeries)",
            "ground": "K-NET IBR007, EW -> longitudinal, NS -> lateral, UD -> vertical (assumed mapping), m/s^2",
            "encoding": "series value = q[i] * scale at t = t0 + i * dt",
            "duplicates": "36 distinct series behind 60 channels; channel['data'] names the stored one",
        },
    })


def record(event_id):
    assert event_id in SELECTED, event_id
    events = pd.read_csv(ROOT / "data" / "knet_wsn_matched_events.csv").set_index("EventID")
    comps, dt_g, n = ground(event_id)
    gnd = {}
    for comp, axis in COMPONENT_DIR.items():
        a = comps[comp]["accel_gal"][:n] / 100.0
        gnd[{1: "Long", 2: "Lat", 3: "Vert"}[axis]] = {"source": comp, "peak": float(f"{np.abs(a).max():.4g}"),
                                                     **quantize(decimate(a, Q, zero_phase=True))}

    cases, alias, f_keep, t0, n_resp = {}, None, None, None, None
    for case in CASES:
        r = pd.read_csv(DATASET / case / f"{event_id}.csv")
        assert list(r.columns) == ["Time"] + CHANNELS, case
        dt = r.Time.iloc[1] - r.Time.iloc[0]
        assert abs(dt - dt_g) < 1e-9 and len(r) == n, (case, dt, len(r), n)
        if alias is None:  # identical columns -> the lowest sensor id is stored, the rest alias it
            seen, alias = {}, {}
            for c in CHANNELS:
                key = r[c].to_numpy().tobytes()
                if key in seen:
                    alias[c] = seen[key]
                else:
                    seen[key] = c
            t0 = float(r.Time.iloc[0])
        chans = {}
        for c in CHANNELS:
            x = r[c].to_numpy()
            if c in alias:
                assert np.array_equal(x, r[alias[c]].to_numpy()), (case, c)
                continue
            f, pxx = welch(x, fs=1 / dt, **WELCH)
            keep = (f >= PSD_BAND[0]) & (f <= PSD_BAND[1])
            f_keep = f[keep]
            y = decimate(x, Q, zero_phase=True)
            n_resp = len(y)
            chans[c] = {**quantize(y), "psd": sig4(pxx[keep]),
                        "peak": float(f"{np.abs(x).max():.4g}"), "rms": float(f"{np.sqrt(np.mean(x ** 2)):.4g}")}
        cases[case] = chans

    write(OUT / f"{event_id}.json", {
        "event": event_meta(event_id, events.loc[event_id]),
        # Most of the peak acceleration is above the 4 Hz anti-alias edge (S49 Vert keeps ~13% of its peak,
        # the ground ~10%); `peak` fields are from the raw 100 Hz series, not the decimated one.
        "ground": {"t0": 0.0, "dt": round(dt_g * Q, 6), "n": len(gnd["Long"]["q"]), "units": "m/s^2",
                   "band_hz": [0, 4.0], **gnd},
        "response": {"t0": t0, "dt": round(dt_g * Q, 6), "n": n_resp, "units": "m/s^2",
                     "kind": "acceleration relative to ground", "band_hz": [0, 4.0]},
        "psd": {"f": [round(v, 5) for v in f_keep], "units": "(m/s^2)^2/Hz",
                "welch": {"fs": round(1 / dt_g, 6), "nperseg": WELCH["nperseg"], "window": "hann",
                          "noverlap": WELCH["nperseg"] // 2, "source": "raw 100 Hz series"}},
        "alias": alias,
        "cases": cases,
    })


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 1:
        index()
    else:
        record(sys.argv[1])
