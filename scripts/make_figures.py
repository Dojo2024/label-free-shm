"""
Intuition figures for the FE project: what was built (bridge model, calibration,
mode shapes, damage sensitivity, one simulated event) and the IASC-ASCE SBI
prototype (frame, frequency check, posterior on the published patterns,
coverage). Everything is recomputed from the code + committed results; nothing
is hand-drawn except the pipeline diagram.

    cd scripts && VIRTUAL_ENV= ../.venv/bin/python make_figures.py
    cd scripts && VIRTUAL_ENV= ../.venv/bin/python make_figures.py --paper   # paper/figures/, no headings
    cd scripts && VIRTUAL_ENV= ../.venv/bin/python make_figures.py --only fig_paper_modes,fig_temperature_vs_damage

Writes figures/NN_*.png. Runs in well under a minute on a laptop
(a handful of ~10 ms eigen solves; no time-history analysis -- event responses
are read from the committed dataset with `git show`, since output/ is outside
the sparse checkout).
"""
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from scipy.signal import welch
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openseespy.opensees as ops
import iasc_asce as ia
import sbi_bridge as sb
from scipy.optimize import linear_sum_assignment
from calibrate import TARGETS, match_modes
from fe_model import (CABLE_ZONES, DAMAGE_TARGETS, GIRDER_ZONES, SENSOR_LOCATIONS, X_P31, X_P32, X_P33,
                      X_P35, X_PYLON, build_model, zone_of, classify_modes, get_frequencies, sensor_output_nodes)
from knet_parser import read_event

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PARAMS = json.loads((ROOT / "calibrated_params.json").read_text())
SBI = json.loads((ROOT / "results" / "sbi_iasc" / "results_n20000.json").read_text())
RAW_GM_DIR = ROOT / "data" / "raw_ground_motion" / "selecteddata" / "knet" / "ascii"
DATASET = "output/complete_simulation_dataset"

# Paper (Siringoringo et al. 2022, Fig. 12) mode names, in calibrate.TARGETS order
PAPER_NAMES = ["1GVe", "1GLo", "1GLa", "2GVe", "3GVe", "4GVe", "2GLa", "5GVe"]
PIERS = {"P31": X_P31, "P32": X_P32, "P33": X_P33, "P34 (pylon)": X_PYLON, "P35": X_P35}
EVENT = "IBR0071906170800"   # largest PGA of the 30 WSN-matched events (129 gal, M5.1)
EVENT_SENSOR = "Sensor_49_Vert"  # mid main span, where the 1st vertical mode peaks
FACES = ["W", "E", "N", "S"]
THETA_NAMES = [f"S{s}-{f}" for s in range(1, ia.N_FLOORS + 1) for f in FACES]
N_TEST = 200  # sbi_iasc.py --n-test default, the held-out draws behind coverage_90

# reference categorical palette (dataviz skill, light mode), fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
DIR_COLOR = {"vertical": BLUE, "longitudinal": ORANGE, "lateral": AQUA}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white",
})


PAPER = "--paper" in sys.argv  # paper/figures/, no numbered heading (the LaTeX caption replaces it)
if PAPER:
    FIG = ROOT / "paper" / "figures"


def save(fig, name):
    if PAPER and fig._suptitle is not None:
        fig._suptitle.remove()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG.relative_to(ROOT)}/{name}", flush=True)


def case_label(case):
    n = case.split("_")[2]
    words = " ".join(case.split("_")[3:])
    return f"{n}: {words}"


def case_desc(spec):
    if spec["target"] == "multiple":
        return " + ".join(case_desc(s) for s in spec["specs"])
    where = ",".join(spec["piers"]) if "piers" in spec else (
        "everywhere" if spec["near"] == "all" else f"within 40 m of {spec['near']}")
    return f"{spec['target']} {where}: k x {1 - spec['factor']:.2f}"


def model_geometry(m):
    """Element segments from the live OpenSees model, grouped by member type."""
    tags = {"girder": {e for e, *_ in m["girder_elements"]},
            "cable": {e for e, _ in m["cable_elements"]},
            "pier": set(m["pier_elements"].values()),
            "pylon": set(m["pylon_elements"])}
    segs = {k: [] for k in ("girder", "cable", "pier", "pylon_steel", "pylon_concrete")}
    for kind, ts in tags.items():
        for e in ts:
            i, j = ops.eleNodes(e)
            ci, cj = np.array(ops.nodeCoord(i)), np.array(ops.nodeCoord(j))
            if kind == "pylon":
                kind_ = "pylon_concrete" if max(ci[2], cj[2]) <= 0 else "pylon_steel"
            else:
                kind_ = kind
            segs[kind_].append((i, j, ci, cj))
    return segs


# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, text, color, dashed=False, w=2.3, h=1.25):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=color + "22", ec=color, lw=1.4, ls="--" if dashed else "-"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, color=INK, wrap=True)
        return (x, y, w, h)

    def arrow(a, b, dashed=False):
        (x0, y0, w0, h0), (x1, y1, w1, h1) = a, b
        if abs(y0 - y1) < 1e-6:
            p0, p1 = (x0 + w0, y0 + h0 / 2), (x1, y1 + h1 / 2)
        else:
            p0, p1 = (x0 + w0 / 2, y0 + h0), (x1 + w1 / 2, y1)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, color=INK2,
                                     lw=1.2, ls="--" if dashed else "-"))

    ax.text(0.1, 5.0, "Bridge track: Shin-Nakagawa cable-stayed bridge (530.8 m)", fontsize=10,
            weight="bold", color=BLUE, va="top")
    ax.text(0.1, 2.35, "Method track: IASC-ASCE Phase I benchmark (known damage truth)", fontsize=10,
            weight="bold", color=VIOLET, va="top")
    y1, y2, dx = 3.2, 0.55, 2.5
    b = [box(0.1 + i * dx, y1, t, BLUE) for i, t in enumerate([
        "Paper's measured modes\nSiringoringo et al. 2022\n8 girder frequencies\n(0.38 - 1.62 Hz)",
        "Calibration\ndifferential evolution\n14 free params, 3 seeds\nobjective 0.0002",
        "Calibrated FE model\nOpenSeesPy, 3D\nall 8 modes within 1.3%\n(paper's own FE: ~10%)",
        "K-NET IBR007 records\n30 real earthquakes\nx 9 variants\n(healthy + 8 damage cases)",
        "270 simulated datasets\n20 sensors x 3 dirs, 100 Hz\nGitHub + HF Hub"])]
    nxt = box(0.1 + 5 * dx, y1, "next: bridge SBI\nposterior over damage +\nnuisance params from\nsimulated responses",
              MUTED, dashed=True)
    c = [box(0.1 + i * dx, y2, t, VIOLET) for i, t in enumerate([
        "IASC-ASCE 12DOF\nJohnson et al. 2004\n4-storey braced frame",
        "Port from the PDF\nall 13 published cases\nwithin 0.07%",
        "20,000 simulations\ntheta ~ U(0,1)^16\nface stiffness fractions",
        "Neural posterior\nestimation (sbi NPE)\nx = 12 freqs + 6 shapes\n(+ noise)",
        "Posterior over brace\ndamage per face\npatterns i-vi, 90% coverage"])]
    for a, bb in zip(b, b[1:] + [nxt]):
        arrow(a, bb, dashed=bb is nxt)
    for a, bb in zip(c[:-1], c[1:]):
        arrow(a, bb)
    arrow(c[-1], nxt, dashed=True)
    fig.suptitle("00  Pipeline: a calibrated bridge simulator on one track, the inference method validated on a "
                 "benchmark with known truth on the other; they meet at bridge SBI (not built yet)",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    save(fig, "00_pipeline.png")


# ---------------------------------------------------------------------------
def fig_bridge_model():
    m = build_model(PARAMS)
    segs = model_geometry(m)
    sensor_nodes = sensor_output_nodes(m)
    anchors = sorted({x for _, x in m["cable_elements"]})

    fig, (ae, ap) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw=dict(height_ratios=[2.1, 1]))
    style = {"cable": (MUTED, 0.7, "stay cable (15 per side, harp)"),
             "girder": (BLUE, 2.4, "steel box girder (deck)"),
             "pier": (INK2, 2.4, f"pier (h = {PARAMS['pier_height']:.1f} m, fixed base)"),
             "pylon_steel": (ORANGE, 3.0, "pylon, steel (above deck)"),
             "pylon_concrete": (VIOLET, 5.0, f"pylon leg, concrete ({PARAMS['pylon_base_depth']} m)")}
    for kind, (col, lw, lab) in style.items():
        for n, (_, _, ci, cj) in enumerate(segs[kind]):
            ae.plot([ci[0], cj[0]], [ci[2], cj[2]], color=col, lw=lw, solid_capstyle="round",
                    label=lab if n == 0 else None, zorder=3 if kind != "cable" else 2)
    for name, x in PIERS.items():
        ae.plot(x, -1.2, marker="^", ms=9, color=YELLOW, mec=INK2, mew=0.6, zorder=5,
                label="laminated rubber bearing" if name == "P31" else None)
        ae.text(x, -34, name, ha="center", va="top", fontsize=8.5, color=INK)
    for x in [X_P31, X_P32, X_P33, X_P35]:
        ae.add_patch(Rectangle((x - 5, -PARAMS["pier_height"] - 2.5), 10, 2.5, fc=GRID, ec=MUTED, hatch="///"))
    ae.add_patch(Rectangle((X_PYLON - 7, -PARAMS["pylon_base_depth"] - 2.5), 14, 2.5, fc=GRID, ec=MUTED, hatch="///"))
    gx = sorted({ops.nodeCoord(sensor_nodes[s])[0] for s, v in SENSOR_LOCATIONS.items() if v["kind"] == "girder"})
    ae.plot(gx, [3.0] * len(gx), "o", ms=7, color=RED, mec="white", mew=0.8, zorder=6,
            label=f"sensor output node (20 sensors -> {len(set(sensor_nodes.values()))} FE nodes)")
    px = sorted({ops.nodeCoord(sensor_nodes[s])[0] for s, v in SENSOR_LOCATIONS.items() if v["kind"] == "pier"})
    ae.plot(px, [-5.0] * len(px), "v", ms=6, color=VIOLET, mec="white", mew=0.6, zorder=6,
            label="pier-top sensor node")
    ae.plot(anchors, [0] * len(anchors), "|", ms=6, color=INK2, zorder=4)
    ae.annotate(f"cable anchorage zone\n(top {PARAMS['pylon_anchor_zone']:.0f} m)",
                (X_PYLON + 2, PARAMS["pylon_top_height"] - 25), (X_PYLON + 40, 95), fontsize=8,
                arrowprops=dict(arrowstyle="-", color=MUTED), color=INK2)
    for (a, b_), lab in [((X_P31, X_P32), "58.9"), ((X_P32, X_P33), "75.0"), ((X_P33, X_PYLON), "113.0"),
                         ((X_PYLON, X_P35), "283.9 m")]:
        ae.annotate("", (a, -44), (b_, -44), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
        ae.text((a + b_) / 2, -46, lab, ha="center", va="top", fontsize=8, color=INK2)
    ae.set_aspect("equal")
    ae.set_ylim(-55, 112)
    ae.set_xlim(-15, 545)
    ae.set_ylabel("z, height above deck [m]")  # x label only on the plan below: it collided with the plan title
    ae.set_title("Elevation (x-z), drawn from the built OpenSees model: single pylon, 30 harp stays, "
                 "deck on rubber bearings at every support", loc="left")
    ae.legend(loc="upper left", ncols=2, fontsize=8)

    # plan: FE model is a centerline; real sensors have lateral offsets
    ap.add_patch(Rectangle((0, -11.45), X_P35, 22.9, fc=BLUE + "18", ec=BLUE, lw=0.8,
                           label="real deck width 22.9 m"))
    ap.plot([0, X_P35], [0, 0], color=BLUE, lw=2.4, label="FE deck = centerline beam (y = 0)")
    ap.plot(anchors, [0] * len(anchors), "|", ms=8, color=INK2, label="cable anchors (single plane)")
    for name, x in PIERS.items():
        ap.plot(x, 0, "s" if "pylon" in name else "^", ms=10 if "pylon" in name else 8,
                color=ORANGE if "pylon" in name else YELLOW, mec=INK2, mew=0.6, zorder=4)
    nodes = pd.read_csv(ROOT / "data" / "bridge_nodes.csv")
    for kind, mk, col, lab in [("girder", "o", RED, "girder sensor (real survey position)"),
                               ("pier", "v", VIOLET, "pier-top sensor"),
                               ("ground", "x", MUTED, "free-field sensor (no FE node, excluded)")]:
        d = nodes[nodes.type == kind]
        ap.plot(d.x_coord, d.y_coord, mk, ms=6, color=col, mew=1.2, label=lab, zorder=5)
        for _, r in d.iterrows():
            dy = 3.2 if r.y_coord >= 0 else -3.2
            if kind == "pier":
                dx = 7
            elif kind == "ground":
                dx, dy = 0, -4.5
            else:
                dx = -7
            ap.text(r.x_coord + dx, r.y_coord + dy, r.node_id, fontsize=6.5, ha="center", va="center",
                    color=INK2)
    ap.set_xlim(-15, 545)
    ap.set_ylim(-26, 26)
    ap.set_xlabel("x, along bridge [m]")
    ap.set_ylabel("y, lateral [m]\n(exaggerated)")
    ap.set_title("Plan (x-y): the two +-y sensors at each station read the same FE node "
                 "(centerline idealisation)", loc="left")
    ap.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=3, fontsize=8)
    fig.suptitle("01  Calibrated FE model of the Shin-Nakagawa Bridge (530.8 m, 4 spans)",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    save(fig, "01_bridge_model.png")


# ---------------------------------------------------------------------------
def fig_calibration():
    m = build_model(PARAMS)
    matched = match_modes(classify_modes(m, 30))
    order = sorted(range(len(TARGETS)), key=lambda i: (["vertical", "longitudinal", "lateral"].index(TARGETS[i][0]),
                                                         TARGETS[i][1]))
    fig, (ax, ae) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    errs = []
    for k, i in enumerate(order):
        d, f_t, mm = matched[i]
        f_s = mm["freq_hz"]
        err = 100 * (f_s - f_t) / f_t
        errs.append(err)
        ax.bar(k - 0.19, f_t, 0.36, color=MUTED, label="measured (paper Fig. 12)" if k == 0 else None)
        ax.bar(k + 0.19, f_s, 0.36, color=DIR_COLOR[d], label=f"FE, {d}" if d not in
               [matched[j][0] for j in order[:k]] else None)
        ax.text(k, max(f_t, f_s) + 0.04, f"{err:+.1f}%", ha="center", fontsize=8.5, color=INK)
        ae.barh(k, err, color=DIR_COLOR[d], height=0.6)
    names = [f"{PAPER_NAMES[i]}\n(FE mode {matched[i][2]['mode']})" for i in order]
    ax.set_xticks(range(8), names)
    ax.set_ylabel("natural frequency [Hz]")
    ax.set_ylim(0, 1.85)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Measured vs calibrated frequency, grouped vertical / longitudinal / lateral", loc="left")
    ae.axvspan(-10, 10, color=GRID, zorder=0, label="paper's own FE model accuracy (~10%)")
    ae.axvline(0, color=INK2, lw=0.8)
    ae.set_yticks(range(8), [PAPER_NAMES[i] for i in order])
    ae.invert_yaxis()
    ae.set_xlim(-11, 11)
    ae.set_xlabel("FE error vs measured [%]")
    ae.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=8)
    ae.set_title("Error on the paper's own +-10% scale", loc="left")
    fig.suptitle(f"02  Calibration: all 8 measured modes reproduced within {max(map(abs, errs)):.1f}% "
                 "(one-to-one Hungarian match within each direction)", x=0.01, ha="left",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "02_calibration.png")


# ---------------------------------------------------------------------------
def fig_mode_shapes(n_show=6):
    m = build_model(PARAMS)
    modes = classify_modes(m, 30)
    paper = {mm["mode"]: PAPER_NAMES[i] for i, (_, _, mm) in enumerate(match_modes(modes))}
    segs = model_geometry(m)
    draw = ["cable", "pier", "pylon_concrete", "pylon_steel", "girder"]
    pylon_tags = {n for k in ("pylon_steel", "pylon_concrete") for i, j, *_ in segs[k] for n in (i, j)}
    deck_tags = set(m["deck_nodes"].values())
    col = {"cable": GRID, "pier": INK2, "pylon_concrete": VIOLET, "pylon_steel": ORANGE, "girder": BLUE}
    fig, axes = plt.subplots(3, 2, figsize=(14, 9.5))
    for mode, ax in zip(modes[:n_show], axes.ravel()):
        k = mode["mode"]
        lat = mode["direction"] == "lateral"
        ev = {t: np.array(ops.nodeEigenvector(t, k)[:3]) for t in pylon_tags | deck_tags}
        mag = lambda ts: sum(np.abs(ev[t]).sum() for t in ts)
        share = 100 * mag(pylon_tags) / (mag(pylon_tags) + mag(deck_tags))
        umax = max(np.abs(v[1] if lat else v[[0, 2]]).max() for v in ev.values())
        s = 30.0 / umax

        def pos(tag, c):
            u = ev.get(tag, np.zeros(3))
            if lat:  # y-displacement rotated into the drawing plane
                return (c[0] + s * u[1], c[2]) if tag in pylon_tags else (c[0], c[2] + s * u[1])
            return c[0] + s * u[0], c[2] + s * u[2]

        for kind in draw:
            for i, j, ci, cj in segs[kind]:
                ax.plot([ci[0], cj[0]], [ci[2], cj[2]], color=GRID if kind != "cable" else "#f1f0ed",
                        lw=1.0, zorder=1)
        for kind in draw:
            for i, j, ci, cj in segs[kind]:
                (x0, z0), (x1, z1) = pos(i, ci), pos(j, cj)
                ax.plot([x0, x1], [z0, z1], color=col[kind] if kind != "cable" else MUTED,
                        lw=2.2 if kind != "cable" else 0.5, solid_capstyle="round", zorder=3)
        tag = f"= paper {paper[k]}" if k in paper else "no measured counterpart"
        ax.set_title(f"Mode {k}: {mode['freq_hz']:.3f} Hz, {mode['direction']}  ({tag}); "
                     f"pylon holds {share:.0f}% of motion", loc="left", color=INK,
                     fontsize=9.5)
        ax.text(0.99, 0.04, "lateral (y) displacement rotated into the page" if lat else
                "x-z displacement, exaggerated", transform=ax.transAxes, ha="right", fontsize=7.5,
                color=MUTED)
        ax.set_aspect("equal")
        ax.set_xlim(-45, 575)
        ax.set_ylim(-45, 140)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
    fig.suptitle("03  First 6 mode shapes: measured modes are main-span girder modes; FE adds two lateral modes "
                 "the paper does not list\n      (5: pylon sway, 89% of motion in the pylon; 6: back-span deck "
                 "sway)", x=0.01, ha="left",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "03_mode_shapes.png")


# ---------------------------------------------------------------------------
def fig_damage_sensitivity():
    m = build_model(PARAMS)
    matched = match_modes(classify_modes(m, 30))
    idx = [mm["mode"] - 1 for _, _, mm in matched]
    build_model(PARAMS)  # a second ops.eigen on the same build fails, so rebuild
    healthy = np.array(get_frequencies(25))
    cases = list(DAMAGE_TARGETS)
    pct = []
    for c in cases:
        build_model(PARAMS, damage_case=c)
        pct.append(100 * (np.array(get_frequencies(25)) - healthy) / healthy)
    pct = np.array(pct)
    assert pct.max() < 0.01, "a damage case raised a frequency"
    sub = np.abs(pct[:, idx])  # all drops (asserted above); abs avoids "-0.00"

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[3, 1.05], hspace=0.55, wspace=0.3)
    ax, ab, at = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    im = ax.imshow(np.clip(sub, 0.01, None), cmap="Blues", norm=LogNorm(0.01, 25), aspect="auto")
    for r in range(sub.shape[0]):
        for c in range(sub.shape[1]):
            v = sub[r, c]
            ax.text(c, r, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 3 else (INK if v >= 0.5 else MUTED),
                    weight="bold" if v >= 0.5 else "normal")
            if v < 0.5:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="....", ec=GRID, lw=0))
    ax.set_xticks(range(len(idx)), [f"{PAPER_NAMES[i]}\n{healthy[j]:.2f} Hz" for i, j in enumerate(idx)])
    ax.set_yticks(range(len(cases)), [case_label(c) for c in cases])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.035)
    cb.set_label("frequency drop vs healthy [%] (log)")
    ax.set_title("Frequency drop per measured mode [%] (hatched = below 0.5%)", loc="left")

    worst = sub.max(axis=1)
    worst_all = -pct.min(axis=1)
    ab.barh(range(len(cases)), worst, color=BLUE, height=0.6, label="worst of the 8 measured modes")
    ab.plot(worst_all, range(len(cases)), "|", ms=12, mew=2, color=ORANGE, label="worst of first 25 FE modes")
    ab.axvspan(0.4, 1.0, color=RED + "30", label="~0.4-1%: frequency\nidentification noise")
    ab.axvline(5, color=RED, ls="--", lw=1, label="5%: SRIM vs M-ERA spread")
    ab.set_xscale("log")
    ab.set_xlim(0.01, 40)
    ab.set_ylim(len(cases) - 0.5, -0.5)
    ab.set_yticks(range(len(cases)), [])
    ab.set_xlabel("worst frequency drop [%] (log)")
    ab.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, fontsize=7.2)
    ab.set_title("Is it visible at all?", loc="left")
    for r, v in enumerate(worst):
        ab.text(v * 1.15, r, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)

    at.axis("off")
    rows = [f"{case_label(c).split(':')[0]:>2}  {case_desc(DAMAGE_TARGETS[c])}" for c in cases]
    for k, line in enumerate(rows):
        at.text(0.0 + 0.5 * (k // 4), 0.95 - 0.26 * (k % 4), line, fontsize=8.5, family="monospace",
                va="top", color=INK, transform=at.transAxes)
    at.set_title("What each damage case does (fractional stiffness reduction, fe_model.DAMAGE_TARGETS)",
                 loc="left")
    fig.suptitle("04  Damage sensitivity: cases 3 and 7 shift frequencies 10-20%; cases 4 and 5 stay under "
                 "0.4% on every measured mode (frequency-invisible);\n      case 1 shows only on one mode (2GLa, 2.2%)",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    save(fig, "04_damage_sensitivity.png")
    return healthy, sub


# ---------------------------------------------------------------------------
def load_response(variant):
    csv = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{DATASET}/{variant}/{EVENT}.csv"],
                         capture_output=True, text=True, check=True).stdout
    return pd.read_csv(io.StringIO(csv))


def fig_time_history():
    gm = read_event(RAW_GM_DIR, EVENT)
    m = build_model(PARAMS)
    healthy_f = [mm["freq_hz"] for d, _, mm in match_modes(classify_modes(m, 30)) if d == "vertical"]
    variants = {"healthy_baseline": ("healthy", INK2, "-"),
                "damage_case_7_cable_severe_widespread": ("case 7: all cables k x 0.60", ORANGE, "-"),
                "damage_case_4_cable_minor_local": ("case 4: cables near pylon k x 0.90", AQUA, "--")}
    resp = {v: load_response(v) for v in variants}

    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), gridspec_kw=dict(height_ratios=[1, 1, 1.3]))
    a0, a1, a2 = axes
    for comp, col, lab in [("EW", ORANGE, "EW -> longitudinal"), ("NS", AQUA, "NS -> lateral"),
                           ("UD", BLUE, "UD -> vertical")]:
        rec = gm[comp]
        a = rec["accel_gal"] / 100.0
        a0.plot(np.arange(len(a)) * rec["dt"], a, color=col, lw=0.6, label=lab)
    a0.set_ylabel("ground accel. [m/s$^2$]")
    a0.set_xlabel("time [s]")
    a0.legend(loc="upper right", ncols=3)
    a0.set_title(f"Input: K-NET IBR007 record {EVENT} (2019-06-17, M5.1, PGA 1.29 m/s$^2$), "
                 "applied as uniform support excitation", loc="left")

    for v in list(variants)[:2]:
        lab, col, ls = variants[v]
        r = resp[v]
        a1.plot(r.Time, r[EVENT_SENSOR], color=col, lw=0.7, label=lab, alpha=0.9)
    a1.set_xlim(0, 60)
    a1.set_ylabel("deck accel. [m/s$^2$]")
    a1.set_xlabel("time [s]")
    a1.legend(loc="upper right")
    a1.set_title("Output: vertical acceleration (relative to ground) at girder sensor S49, mid main span "
                 "(x = 459.8 m), first 60 s -- traces look alike by eye", loc="left")

    for v, (lab, col, ls) in variants.items():
        r = resp[v]
        f, pxx = welch(r[EVENT_SENSOR].to_numpy(), fs=1 / (r.Time[1] - r.Time[0]), nperseg=4096)
        a2.semilogy(f, pxx, color=col, ls=ls, lw=1.3, label=lab)
    for k, fh in enumerate(healthy_f):
        a2.axvline(fh, color=MUTED, lw=0.7, ls=":", zorder=0)
        a2.text(fh, 1.02, f"{fh:.2f}", transform=a2.get_xaxis_transform(), ha="center", fontsize=7.5,
                color=INK2)
    a2.set_xlim(0.2, 2.0)
    a2.set_ylim(1e-9, 1e-3)
    a2.set_xlabel("frequency [Hz]  (dotted: healthy FE vertical modes)")
    a2.set_ylabel("PSD [(m/s$^2$)$^2$/Hz]")
    a2.legend(loc="lower left")
    a2.set_title("Welch PSD of the whole record: case 7 moves every peak left; case 4 overlays healthy",
                 loc="left", pad=14)
    fig.suptitle("05  One simulated event: severe cable loss is obvious in the spectrum, minor local cable "
                 "damage is indistinguishable from healthy", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "05_time_history.png")


# ---------------------------------------------------------------------------
def fig_iasc_frame():
    fig = plt.figure(figsize=(13, 5.6))
    af = fig.add_axes([0.0, 0.02, 0.36, 0.86])
    ab = fig.add_axes([0.44, 0.13, 0.55, 0.72])
    c30, s30 = np.cos(np.pi / 6), np.sin(np.pi / 6)

    def P(x, y, z):  # isometric projection, viewer at the south-west-ish (+x, +y) corner
        return (-x + y) * c30, z + (-x - y) * s30 * 0.7

    B, H = ia.BAY, ia.STORY_H
    for lev in range(ia.N_FLOORS + 1):
        z = lev * H
        corners = [P(x, y, z) for x, y in [(B, B), (B, -B), (-B, -B), (-B, B)]]
        af.add_patch(Polygon(corners, closed=True, fc=GRID if lev else "none", ec=MUTED, lw=0.8, alpha=0.5,
                             zorder=1))
    for x in ia.PLAN:
        for y in ia.PLAN:
            af.plot(*zip(P(x, y, 0), P(x, y, ia.N_FLOORS * H)), color=INK2, lw=1.4, zorder=2)
    damaged = {"S1-WN": (RED, "pattern iii/iv/vi brace (S1-WN)"), "S3-NW": (ORANGE, "pattern iv brace (S3-NW)")}
    labelled = set()
    for s in range(1, ia.N_FLOORS + 1):
        for face, (bot, top) in ia._FACE_BRACES.items():
            bid = f"S{s}-{face}"
            back = face[0] in "NE"  # far faces from this viewpoint
            col, lab = damaged.get(bid, (BLUE, "brace (8 per storey; far faces N/E faded)"))
            af.plot(*zip(P(*bot, (s - 1) * H), P(*top, s * H)), color=col,
                    lw=2.6 if bid in damaged else 1.0, zorder=3, alpha=0.35 if back and bid not in damaged else 1,
                    label=lab if lab not in labelled else None)
            labelled.add(lab)
    for name, (x, y, z) in {"W face (+x), near": (B * 1.5, 0, -0.35), "S face (+y), near": (0, B * 1.5, -0.35),
                            "N face (-y), far": (0, -B * 1.3, 4 * H + 0.35),
                            "E face (-x), far": (-B * 1.3, 0, 4 * H + 0.35)}.items():
        af.text(*P(x, y, z), name, ha="center", va="center", fontsize=9, weight="bold", color=VIOLET)
    for s in range(1, ia.N_FLOORS + 1):
        af.text(*P(B, -B, (s - 0.5) * H), f"storey {s}  ", ha="right", va="center", fontsize=8, color=INK2)
    af.set_aspect("equal")
    af.axis("off")
    af.legend(loc="upper center", fontsize=7.5, bbox_to_anchor=(0.5, -0.02), ncols=1)
    af.set_title("4-storey, 2x2-bay steel frame (2.5 x 2.5 m plan, 0.9 m storeys)", fontsize=9.5)

    ours = ia.frequencies(*ia.build_12dof(None, "symmetric", False))
    ref = np.array(ia.PUBLISHED_FREQS[("T2", None, "symmetric", False)])
    worst = max(np.abs(100 * (ia.frequencies(*ia.build_12dof(d, ms, lu)) - np.array(r)) / np.array(r)).max()
                for (_, d, ms, lu), r in ia.PUBLISHED_FREQS.items())
    k = np.arange(1, 13)
    ab.bar(k - 0.2, ref, 0.38, color=MUTED, label="published (Johnson et al. 2004, Table 2)")
    ab.bar(k + 0.2, ours, 0.38, color=VIOLET, label="our port (iasc_asce.py)")
    for kk, (o, r) in enumerate(zip(ours, ref), 1):
        ab.text(kk, max(o, r) + 1.5, f"{100 * (o - r) / r:+.2f}%", ha="center", fontsize=7.5, color=INK2,
                rotation=90, va="bottom")
    ab.set_xticks(k)
    ab.set_xlabel("mode number")
    ab.set_ylabel("natural frequency [Hz]")
    ab.set_ylim(0, 105)
    ab.legend(loc="upper left")
    ab.set_title(f"Undamaged 12DOF, consistent mass. Worst error over all 13 published cases: {worst:.3f}%",
                 loc="left")
    fig.suptitle("06  IASC-ASCE Phase I benchmark: the 12DOF port reproduces the published frequencies "
                 "to printed rounding", x=0.01, ha="left", fontsize=12, weight="bold")
    save(fig, "06_iasc_frame.png")
    return worst


# ---------------------------------------------------------------------------
def fig_sbi_patterns():
    desc = {"i": "all storey-1 braces removed", "ii": "all storey-1 and storey-3 braces removed",
            "iii": "one storey-1 brace on the W face removed (W face -> 0.5)",
            "iv": "iii + one storey-3 brace on the N face removed",
            "vi": "one storey-1 W brace at 2/3 stiffness (W face -> 0.83)"}
    xs = np.array([s * 5 + f for s in range(4) for f in range(4)], dtype=float)
    fig, axes = plt.subplots(len(desc), 1, figsize=(12, 12.5), sharex=True)
    for ax, (name, d) in zip(axes, desc.items()):
        r = SBI["patterns"][name]
        t, mu, sd = map(np.array, (r["truth"], r["post_mean"], r["post_sd"]))
        bad = (np.abs(mu - t) > 2 * sd) & (np.abs(mu - t) > 0.1)  # material, not a prior-edge squeeze
        for s in range(4):
            ax.axvspan(s * 5 - 0.6, s * 5 + 3.6, color="#f4f3f0", zorder=0)
        ax.errorbar(xs, mu, yerr=2 * sd, fmt="none", ecolor=BLUE, elinewidth=1.0, capsize=0, alpha=0.55)
        ax.errorbar(xs, mu, yerr=sd, fmt="none", ecolor=BLUE, elinewidth=3.2, capsize=0)
        ax.plot(xs[~bad], mu[~bad], "o", ms=6, color=BLUE, mec="white", mew=0.8,
                label="posterior mean, +-1 sd (thick), +-2 sd (thin)")
        ax.plot(xs[bad], mu[bad], "o", ms=7, color=RED, mec="white", mew=0.8,
                label="overconfident miss: truth outside +-2 sd and |err| > 0.1")
        ax.plot(xs, t, "_", ms=16, mew=2.4, color=INK, label="truth")
        ax.set_ylim(-0.08, 1.15)
        ax.set_ylabel("remaining\nstiffness")
        err = np.abs(mu - t)
        ax.set_title(f"pattern ({name}): {d}.   mean |err| {err.mean():.3f}, max |err| {err.max():.2f}",
                     loc="left")
        for i in np.where(bad)[0]:
            ax.annotate(f"{THETA_NAMES[i]}: {mu[i]:.2f} +- {sd[i]:.2f}", (xs[i], mu[i]), (xs[i] + 0.4, 0.12),
                        fontsize=7.5, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
    axes[0].legend(loc="lower center", ncols=3, fontsize=8, bbox_to_anchor=(0.5, 1.18))
    axes[-1].set_xticks(xs, [f"{f}\nS{s + 1}" if f == "W" else f for s in range(4) for f in FACES])
    axes[-1].set_xlabel("face (W/E/N/S) within storey 1-4")
    fig.text(0.01, 0.005, "Undamaged faces sit at the upper edge of the U(0,1) prior, so their posterior means "
             "are pulled slightly below 1 (and broken faces slightly above 0) -- expected, not a miss.", fontsize=8, color=INK2)
    fig.suptitle("07  NPE on the published damage patterns: i, iii, iv, vi recovered; ii only partially -- "
                 "storeys 2 and 4 become uncertain, S2-E is confidently wrong", x=0.01, ha="left",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    save(fig, "07_sbi_patterns.png")


# ---------------------------------------------------------------------------
def fig_sbi_coverage():
    cov = np.array([SBI["coverage_90"][n] for n in THETA_NAMES])
    half = 1.645 * np.sqrt(0.9 * 0.1 / N_TEST)
    xs = np.array([s * 5 + f for s in range(4) for f in range(4)], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.axhspan(0.9 - half, 0.9 + half, color=GRID, label=f"expected spread for {N_TEST} test draws (90% binomial)")
    out = np.abs(cov - 0.9) > half
    ax.bar(xs[~out], cov[~out], 0.75, color=BLUE, label="per-parameter coverage")
    ax.bar(xs[out], cov[out], 0.75, color=RED, label="outside expected spread")
    ax.axhline(0.9, color=INK, ls="--", lw=1.1, label="nominal 0.90")
    for x, c in zip(xs, cov):
        ax.text(x, 0.705, f"{c:.3f}", ha="center", va="bottom", fontsize=7.5, color="white")
    ax.set_xticks(xs, [f"{f}\nS{s + 1}" if f == "W" else f for s in range(4) for f in FACES])
    ax.set_xlabel("face (W/E/N/S) within storey 1-4")
    ax.set_ylabel("90% interval coverage")
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower center", ncols=2, fontsize=8, bbox_to_anchor=(0.5, -0.45))
    ax.set_title(f"Held-out prior draws: fraction whose true value falls in the posterior 5-95% interval "
                 f"(mean {cov.mean():.3f})", loc="left")
    fig.suptitle("08  Posterior calibration: coverage sits near 0.90 for 15 of 16 faces; S2-W is "
                 "under-covered (slightly overconfident)", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "08_sbi_coverage.png")


# ---------------------------------------------------------------------------
# Bridge SBI (sbi_bridge.py)
ZONE_TYPE_COLOR = {"B": ORANGE, "P": VIOLET, "G": BLUE, "C": AQUA}
ZONE_TYPE_NAME = {"B": "bearing", "P": "pier", "G": "girder zone", "C": "cable group"}


def zone_color(z):
    return ZONE_TYPE_COLOR[z[0]]


def bridge_sbi_results():
    """Largest-n full run on disk (the smoke file is never used)."""
    runs = sorted((ROOT / "results" / "sbi_bridge").glob("results_n*.json"),
                  key=lambda f: int(f.stem.split("_n")[1]))
    return runs[-1].stem.split("_")[1], json.loads(runs[-1].read_text())


def bridge_logf_shift(zone_mult=None, env=None):
    """Signed d log f of the 8 paired modes vs the healthy calibrated model, in noise sigmas."""
    env = env or dict(bearing=1.0, deck_E=1.0)
    sb.reference_shapes()
    x0 = sb.modal_features(build_model(sb.state_params({}, dict(bearing=1.0, deck_E=1.0))))
    x = sb.modal_features(build_model(sb.state_params({}, env), zone_mult=zone_mult))
    return (x[:sb.N_FREQS] - x0[:sb.N_FREQS]) / sb.FREQ_NOISE


def bridge_feature_shift(zone_mult=None, env=None):
    """max |d log f| over the 8 paired modes vs the healthy calibrated model, in noise sigmas."""
    return np.abs(bridge_logf_shift(zone_mult, env)).max()


def fig_bridge_zones():
    sb.reference_shapes()
    m = build_model(PARAMS)
    segs = model_geometry(m)
    fig, (ae, ah) = plt.subplots(2, 1, figsize=(14, 9.5), gridspec_kw=dict(height_ratios=[1.35, 1]))

    # elevation, every element coloured by the damage zone it belongs to
    cable_zone = {et: zone_of(ax_, CABLE_ZONES) for et, ax_ in m["cable_elements"]}
    for et, ax_ in m["cable_elements"]:
        i, j = ops.eleNodes(et)
        ci, cj = np.array(ops.nodeCoord(i)), np.array(ops.nodeCoord(j))
        z = cable_zone[et]
        shade = 1.0 if z.endswith("near") else 0.5
        ae.plot([ci[0], cj[0]], [ci[2], cj[2]], color=AQUA, alpha=shade, lw=0.9, zorder=2)
    for _, _, ci, cj in segs["pylon_steel"] + segs["pylon_concrete"]:
        ae.plot([ci[0], cj[0]], [ci[2], cj[2]], color=MUTED, lw=3.0, zorder=2)
    for k, (z, (lo, hi)) in enumerate(GIRDER_ZONES.items()):
        ae.plot([lo, hi], [0, 0], color=BLUE, alpha=1.0 if k % 2 == 0 else 0.55, lw=5,
                solid_capstyle="butt", zorder=3)
        ae.text((lo + hi) / 2, 5, z, ha="center", va="bottom", fontsize=9, color=INK, weight="bold")
    for name, x in {"P31": X_P31, "P32": X_P32, "P33": X_P33, "PYLON": X_PYLON, "P35": X_P35}.items():
        if name != "PYLON":
            ae.plot([x, x], [-1.5, -PARAMS["pier_height"]], color=VIOLET, lw=5, solid_capstyle="butt", zorder=3)
            ae.text(x + 4, -PARAMS["pier_height"] / 2 - 1, f"P-{name}", ha="left", va="center", fontsize=8.5, color=INK)
        ae.plot(x, -1.2, marker="^", ms=10, color=ORANGE, mec=INK2, mew=0.6, zorder=5)
        ae.text(x, -24 if name != "PYLON" else -30, f"B-{name}", ha="center", va="top", fontsize=8.5, color=INK)
    for z, (lo, hi) in CABLE_ZONES.items():
        xs_ = [ax_ for et, ax_ in m["cable_elements"] if cable_zone[et] == z]
        xm = np.mean(xs_)
        ae.text(xm, 40 + (12 if z.endswith("near") else 0), z, ha="center", fontsize=8.5, color=INK,
                bbox=dict(fc="white", ec="none", pad=1))
    handles = [plt.Line2D([], [], color=c, lw=5 if t != "B" else 0, marker="^" if t == "B" else None,
                          ms=9, mec=INK2, label=f"{ZONE_TYPE_NAME[t]} ({sum(z[0] == t for z in sb.ZONES)})")
               for t, c in ZONE_TYPE_COLOR.items()]
    ae.legend(handles=handles, loc="upper right", ncols=4, fontsize=8.5)
    ae.set_aspect("equal")
    ae.set_ylim(-38, 112)
    ae.set_xlim(-15, 545)
    ae.set_xlabel("x, along bridge [m]")
    ae.set_ylabel("z [m]")
    ae.set_title("The 19 damage zones SBI infers a remaining-stiffness multiplier for "
                 "(alternate shades separate neighbouring zones of one type)", loc="left")

    # detectability: 50% loss per zone vs measurement noise and a temperature swing
    shift = np.array([bridge_feature_shift({z: 0.5}) for z in sb.ZONES])
    temp = max(bridge_feature_shift(env=dict(bearing=b, deck_E=1.0)) for b in (sb.ENV_BEARING, 1 / sb.ENV_BEARING))
    order = np.argsort(shift)
    ys = np.arange(len(order))
    ah.barh(ys, shift[order], 0.72, color=[zone_color(sb.ZONES[i]) for i in order])
    ah.axvline(1, color=INK, ls="--", lw=1.1)
    ah.axvline(temp, color=RED, ls="-", lw=1.4)
    ah.text(1.05, 0, "1 sigma of frequency noise", color=INK, fontsize=8, va="center")
    ah.text(temp * 1.05, 1.0, f"bearing temperature swing\n(k x/÷ {sb.ENV_BEARING}, no damage): "
            f"{temp:.1f} sigma", color=RED, fontsize=8, va="bottom")
    for y, i in zip(ys, order):
        ah.text(shift[i] * 1.06, y, f"{shift[i]:.2f}", va="center", fontsize=7.5, color=INK2)
    ah.set_yticks(ys, [sb.ZONES[i] for i in order])
    ah.set_xscale("log")
    ah.set_xlim(0.05, 60)
    ah.set_xlabel("largest shift of the 8 paired log-frequencies, in units of the 1% noise sigma")
    ah.set_title("Detectability of a 50% stiffness loss in each zone alone (healthy calibrated model, "
                 "noise-free): piers sit below the noise, behind far softer bearings", loc="left")
    ah.grid(axis="y", visible=False)
    fig.suptitle("09  Bridge SBI: what is inferred, and what the 8 identified modes can see",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "09_bridge_zones.png")


def fig_bridge_mode_pairing(n_draws=300):
    """Worst matched MAC per nuisance draw under three ways of picking 'the same 8 modes'."""
    ref, logf_ref = sb.reference_shapes()
    rng = np.random.default_rng(1)
    worst = {k: [] for k in ("frequency order", "MAC only", "MAC + frequency (used)")}
    for _ in range(n_draws):
        mm = build_model(sb.state_params(sb.sample_nuisance(rng), sb.sample_env(rng)))
        lam = np.array(ops.eigen(sb.N_EIGEN))
        logf = np.log(np.sqrt(lam) / (2 * np.pi))
        sh = sb.sensor_shapes(mm, sb.N_EIGEN)
        mac = (ref @ sh.T) ** 2
        picks = {"frequency order": np.array(sb.REF_MODES) - 1,
                 "MAC only": linear_sum_assignment(-mac)[1]}
        nb = int((logf < np.log(sb.F_BAND)).sum())
        picks["MAC + frequency (used)"] = linear_sum_assignment(
            1 - mac[:, :nb] + np.abs(logf_ref[:, None] - logf[None, :nb]))[1]
        for k, p in picks.items():
            worst[k].append((mac[np.arange(len(ref)), p].min(), np.abs(logf[p] - logf_ref).max()))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    cols = [MUTED, ORANGE, BLUE]
    for (k, v), c in zip(worst.items(), cols):
        v = np.array(v)
        a1.hist(v[:, 0], bins=np.linspace(0, 1, 41), histtype="step", lw=2, color=c,
                label=f"{k}: {np.mean(v[:, 0] < 0.5):.0%} of draws < 0.5")
        a2.hist(np.exp(v[:, 1]), bins=np.logspace(0, np.log10(12), 41), histtype="step", lw=2, color=c,
                label=f"{k}: {np.mean(np.exp(v[:, 1]) > 1.6):.0%} of draws > 1.6x")
    a1.set_xlabel("worst MAC among the 8 paired modes (1 = same shape at the sensors)")
    a1.set_ylabel(f"nuisance draws (of {n_draws})")
    a1.set_title("Shape agreement with the healthy reference", loc="left")
    a2.set_xscale("log")
    a2.set_xlabel("worst frequency ratio f / f_ref among the 8 paired modes")
    a2.set_title("Frequency agreement: MAC alone jumps to aliased 3-5 Hz modes", loc="left")
    a1.legend(loc="upper center", fontsize=8)
    a2.legend(loc="upper right", fontsize=8)
    fig.suptitle("10  Which simulated mode is '2GVe'? Pairing the paper's 8 identified modes under the "
                 "nuisance prior", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "10_bridge_mode_pairing.png")


def fig_bridge_sbi_cases():
    tag, res = bridge_sbi_results()
    cases = list(res["cases"])
    pd_ = np.array([res["cases"][c]["p_damaged"] for c in cases]).T
    truth = np.array([res["cases"][c]["truth"] for c in cases]).T
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(pd_, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(pd_.shape[0]):
        for j in range(pd_.shape[1]):
            if truth[i, j] < 0.999:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=RED, lw=2))
                ax.text(j, i, f"{pd_[i, j]:.2f}\nk={truth[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if pd_[i, j] > 0.55 else INK)
            elif pd_[i, j] >= 0.2:
                ax.text(j, i, f"{pd_[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(cases)), ["healthy"] + [f"case {c.split('_')[2]}\n" + " ".join(c.split("_")[3:5])
                                                    for c in cases[1:]], fontsize=8)
    ax.set_yticks(range(len(sb.ZONES)), sb.ZONES)
    for t in ax.get_yticklabels():
        t.set_color(zone_color(t.get_text()))
        t.set_weight("bold")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("posterior P(zone multiplier < 0.9)")
    ax.set_title(f"One noisy observation per case at the calibrated params, nominal temperature ({tag} sims). "
                 "Red box = zone the case reduces (k = true zone-mean multiplier);\nP counts only losses > 10%, so k >= 0.9 "
                 "boxes (case 4, case 6) should stay low. Prior P(multiplier < 0.9) = 0.15 x 0.875 = 0.13", loc="left")
    fig.suptitle("11  Bridge SBI on the 8 Table 3 damage cases: which zones does the posterior flag?",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "11_bridge_sbi_cases.png")


def fig_bridge_sbi_detection():
    tag, res = bridge_sbi_results()
    zones = sb.ZONES
    auc = np.array([res["auroc"][z] if res["auroc"][z] is not None else np.nan for z in zones])
    cov = np.array([res["coverage_90"][z] for z in zones])
    n_test = 300  # sbi_bridge.py --n-test default
    half = 1.645 * np.sqrt(0.9 * 0.1 / n_test)
    xs = np.arange(len(zones))
    colors = [zone_color(z) for z in zones]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    a1.bar(xs, auc, 0.72, color=colors)
    chance = a1.axhline(0.5, color=MUTED, ls=":", lw=1.4, label="chance (AUROC 0.5)")
    for x, v in zip(xs, auc):
        a1.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    a1.set_ylim(0.3, 1.02)
    a1.set_ylabel("detection AUROC")
    a1.set_title("Held-out prior draws: does P(k < 0.9) rank truly damaged zones above intact ones?", loc="left")
    a2.axhspan(0.9 - half, 0.9 + half, color=GRID, label=f"expected spread for {n_test} test draws")
    a2.bar(xs, cov, 0.72, color=colors)
    a2.axhline(0.9, color=INK, ls="--", lw=1.1, label="nominal 0.90")
    a2.set_ylim(0.7, 1.0)
    a2.set_ylabel("90% interval coverage")
    a2.set_title(f"Calibration: fraction of true values inside the 5-95% posterior interval (mean {cov.mean():.3f})",
                 loc="left")
    a2.set_xticks(xs, zones, rotation=45, ha="right")
    handles = [Rectangle((0, 0), 1, 1, color=c, label=ZONE_TYPE_NAME[t]) for t, c in ZONE_TYPE_COLOR.items()]
    a2.legend(handles=handles + a2.get_legend_handles_labels()[0] + [chance], loc="lower center", ncols=7, fontsize=8,
              bbox_to_anchor=(0.5, -0.38))
    for a in (a1, a2):
        a.grid(axis="x", visible=False)
    fig.suptitle(f"12  Bridge SBI ({tag} sims): calibrated everywhere; detection tracks the physics "
                 "(piers at chance)", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "12_bridge_sbi_detection.png")


# ---------------------------------------------------------------------------
# Paper figures, sized for a 6.5 in text width so font sizes are the printed sizes
def fig_paper_modes():
    m = build_model(PARAMS)
    matched = match_modes(classify_modes(m, 30))
    assert [mm["mode"] for _, _, mm in matched] == sb.REF_MODES
    segs = model_geometry(m)
    draw = ["cable", "pier", "pylon_concrete", "pylon_steel", "girder"]
    col = {"cable": MUTED, "pier": INK2, "pylon_concrete": VIOLET, "pylon_steel": ORANGE, "girder": BLUE}
    lw = {"cable": 0.35, "pier": 1.4, "pylon_concrete": 2.0, "pylon_steel": 1.5, "girder": 1.6}
    sensors = sorted(set(sensor_output_nodes(m).values()))
    tags = {n for k in draw for i, j, *_ in segs[k] for n in (i, j)} | set(sensors)
    fig, axes = plt.subplots(4, 2, figsize=(6.5, 5.3), sharex=True)
    for (d, f_field, mm), name, ax in zip(matched, PAPER_NAMES, axes.ravel()):
        k = mm["mode"]
        plan = d == "lateral"  # lateral modes in plan (x-y), the rest in elevation (x-z)
        ev = {t: np.array(ops.nodeEigenvector(t, k)[:3]) for t in tags}
        comp = [1] if plan else [0, 2]
        t_max = max(tags, key=lambda t: np.abs(ev[t][comp]).max())
        u_max = ev[t_max][comp][np.argmax(np.abs(ev[t_max][comp]))]
        s = (45.0 if plan else 30.0) / u_max  # signed: the largest displacement points up / +y

        def pos(t, c, s_):
            u = s_ * ev[t]
            return (c[0] + u[0], u[1]) if plan else (c[0] + u[0], c[2] + u[2])

        for kind in draw:
            for i, j, ci, cj in segs[kind]:
                (x0, y0), (x1, y1) = pos(i, ci, 0), pos(j, cj, 0)
                ax.plot([x0, x1], [y0, y1], color="#f1f0ed" if kind == "cable" else GRID, lw=1.0, zorder=1)
        for kind in draw:
            for i, j, ci, cj in segs[kind]:
                (x0, y0), (x1, y1) = pos(i, ci, s), pos(j, cj, s)
                ax.plot([x0, x1], [y0, y1], color=col[kind], lw=lw[kind], solid_capstyle="round",
                        zorder=2 if kind == "cable" else 3)
        xy = np.array([pos(t, np.array(ops.nodeCoord(t)), s) for t in sensors])
        ax.plot(xy[:, 0], xy[:, 1], "o", ms=3.2, color=RED, mec="white", mew=0.5, zorder=5)
        ax.set_title(f"{name}   FE {mm['freq_hz']:.3f} Hz,  field {f_field:.2f} Hz", loc="left", fontsize=8,
                     color=INK, pad=2)
        ax.text(0.995, 0.96, "plan x-y (lateral)" if plan else "elevation x-z", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color=MUTED)
        ax.set_aspect("equal")
        ax.set_xlim(-20, 551)
        ax.set_ylim(*((-80, 80) if plan else (-40, 120)))
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.set_xticks(list(PIERS.values()), ["P31", "P32", "P33", "P34", "P35"], fontsize=6.5)
        ax.tick_params(length=2, pad=1.5)
    handles = [plt.Line2D([], [], color=col[k], lw=lw[k] if k != "cable" else 0.8, label=lab) for k, lab in
               [("girder", "deck (girder)"), ("pylon_steel", "pylon, steel"), ("pylon_concrete", "pylon, concrete"),
                ("pier", "pier"), ("cable", "stay cable")]]
    handles += [plt.Line2D([], [], color=GRID, lw=1.0, label="undeformed"),
                plt.Line2D([], [], color=RED, marker="o", ms=3.2, mec="white", mew=0.5, lw=0,
                           label=f"sensor station ({len(sensors)} FE nodes)")]
    fig.legend(handles=handles, loc="lower center", ncols=4, fontsize=7, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("13  The 8 field-identified modes of the calibrated model (FE modes " +
                 ", ".join(map(str, sb.REF_MODES)) + "; displacement exaggerated)",
                 x=0.01, ha="left", fontsize=10, weight="bold")
    fig.tight_layout(h_pad=0.4, w_pad=1.0)
    save(fig, "13_bridge_modes.png")


def fig_temperature_vs_damage(n_draws=80):
    """(a) d log f signature of each environment change and each 50%-loss zone; (b, c) SBI-style
    baseline/current pairs, healthy vs one damaged zone, against the healthy (temperature) cloud."""
    env_rows = {"bearings x1.3 (cold)": dict(bearing=sb.ENV_BEARING, deck_E=1.0),
                "bearings /1.3 (warm)": dict(bearing=1 / sb.ENV_BEARING, deck_E=1.0),
                f"deck E +{sb.ENV_DECK_E:.0%}": dict(bearing=1.0, deck_E=1 + sb.ENV_DECK_E),
                f"deck E -{sb.ENV_DECK_E:.0%}": dict(bearing=1.0, deck_E=1 - sb.ENV_DECK_E)}
    env_sig = np.array([bridge_logf_shift(env=e) for e in env_rows.values()])
    zone_sig = np.array([bridge_logf_shift({z: 0.5}) for z in sb.ZONES])
    # the two temperature directions (half the cold-warm and E+/E- differences), orthonormalised
    basis, _ = np.linalg.qr(np.array([env_sig[0] - env_sig[1], env_sig[2] - env_sig[3]]).T / 2)
    resid = lambda v: v - basis @ (basis.T @ v)
    frac_t = lambda v: 1 - (resid(v) ** 2).sum() / (v ** 2).sum()
    order = sorted(range(len(sb.ZONES)), key=lambda i: -frac_t(zone_sig[i]))
    rows = list(env_rows) + [sb.ZONES[i] for i in order]
    sig = np.vstack([env_sig, zone_sig[order]])

    # SBI-style pairs: shared nuisance, independent baseline/current environments, 1% frequency noise.
    # Each nuisance draw gives one baseline and two current states (healthy, and one zone damaged)
    rng = np.random.default_rng(0)
    N = sb.N_FREQS
    logf = lambda nuis, env, zm=None: sb.modal_features(build_model(sb.state_params(nuis, env), zone_mult=zm))[:N]
    noise = lambda: rng.normal(0, sb.FREQ_NOISE, N)
    healthy, damaged, dzone = [], [], []
    for _ in range(n_draws):
        nuis, env_b, env_c = sb.sample_nuisance(rng), sb.sample_env(rng), sb.sample_env(rng)
        base = logf(nuis, env_b)
        z = rng.integers(len(sb.ZONES))
        mult = rng.uniform(0.98, 1.0, len(sb.ZONES))  # the prior's near-intact range
        mult[z] = rng.uniform(sb.M_MIN, sb.DAMAGE_THRESHOLD)
        healthy.append(logf(nuis, env_c) + noise() - (base + noise()))
        damaged.append(logf(nuis, env_c, dict(zip(sb.ZONES, mult))) + noise() - (base + noise()))
        dzone.append(sb.ZONES[z])
    healthy, damaged = np.array(healthy) / sb.FREQ_NOISE, np.array(damaged) / sb.FREQ_NOISE
    dzone = np.array(dzone)
    mu = healthy.mean(0)
    _, sv, vt = np.linalg.svd(healthy - mu, full_matrices=False)
    vt[:2] *= np.sign(vt[:2, 1])[:, None]  # PC1 positive on 1GLo (cold = stiffer bearings)
    var = sv ** 2 / (sv ** 2).sum()
    cov_i = np.linalg.inv(np.cov(healthy.T))
    d_m = lambda X: np.sqrt(np.einsum("ij,jk,ik->i", X - mu, cov_i, X - mu))
    thr8 = np.sqrt(chi2.ppf(0.99, N))
    pc = lambda X: (X - mu) @ vt[:2].T
    c2 = np.cov(pc(healthy).T)
    out2 = lambda X: np.einsum("ij,jk,ik->i", pc(X), np.linalg.inv(c2), pc(X)) > chi2.ppf(0.99, 2)

    fig = plt.figure(figsize=(6.5, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.22, 1], hspace=0.55, wspace=0.42)
    ah = fig.add_subplot(gs[:, 0])
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f3f2ef", ORANGE])
    vmax = 10
    im = ah.imshow(sig, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for i, j in np.ndindex(sig.shape):
        v = sig[i, j]
        if abs(v) >= 0.5:
            ah.text(j, i, f"{v:.0f}" if abs(v) >= 9.5 else f"{v:.1f}", ha="center", va="center", fontsize=5.3,
                    color="white" if abs(v) > 0.6 * vmax else INK)
    for i, v in enumerate(sig):
        ah.text(7.75, i, f"{frac_t(v):.0%}", ha="left", va="center", fontsize=5.8, color=INK2, clip_on=False)
        ah.text(9.35, i, f"{np.linalg.norm(resid(v)):.1f}", ha="right", va="center", fontsize=5.8, color=INK2,
                clip_on=False)
    ah.text(7.75, -0.9, "T-\nlike", ha="left", va="bottom", fontsize=5.8, color=INK2, clip_on=False)
    ah.text(9.35, -0.9, "rest\n[σ]", ha="right", va="bottom", fontsize=5.8, color=INK2, clip_on=False)
    ah.axhline(len(env_rows) - 0.5, color=INK, lw=1.0)
    ah.xaxis.tick_top()
    ah.set_xticks(range(N), PAPER_NAMES, fontsize=6.5, rotation=90)
    ah.set_yticks(range(len(rows)), rows, fontsize=6.5)
    for t in ah.get_yticklabels()[len(env_rows):]:
        t.set_color(zone_color(t.get_text()))
        t.set_weight("bold")
    ah.tick_params(length=0, pad=1.5)
    ah.grid(False)
    cb = fig.colorbar(im, ax=ah, orientation="horizontal", fraction=0.035, pad=0.02, extend="both")
    cb.set_label("shift in log-frequency / 1% noise sigma  (zones: 50% stiffness loss)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5)
    ah.set_title("(a) one change at a time, calibrated model", loc="left", fontsize=8, pad=26)

    colors = {t: c for t, c in ZONE_TYPE_COLOR.items()}
    for k, ax in enumerate([fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]):
        yh = pc(healthy)[:, 1] if k == 0 else d_m(healthy)
        yd = pc(damaged)[:, 1] if k == 0 else d_m(damaged)
        ax.scatter(pc(healthy)[:, 0], yh, s=9, color=MUTED, lw=0, alpha=0.8, zorder=2,
                   label=f"healthy ({n_draws})")
        for t, c in colors.items():
            sel = np.array([z[0] == t for z in dzone])
            out = (out2(damaged[sel]) if k == 0 else d_m(damaged[sel]) > thr8).sum()
            ax.scatter(pc(damaged[sel])[:, 0], yd[sel], s=11, color=c, lw=0.4, ec="white", zorder=3,
                       label=f"{ZONE_TYPE_NAME[t]}: {out}/{sel.sum()} outside")
        if k == 0:
            w, v = np.linalg.eigh(c2)
            r = np.sqrt(chi2.ppf(0.99, 2) * w)
            ax.add_patch(Ellipse((0, 0), 2 * r[1], 2 * r[0], angle=np.degrees(np.arctan2(v[1, 1], v[0, 1])),
                                 fill=False, ec=INK, ls="--", lw=0.8, zorder=4))
            ax.set_ylabel(f"healthy PC2 ({var[1]:.0%}) [σ]", fontsize=6.5)
            ax.set_title(f"(b) plane of the healthy cloud: {out2(damaged).mean():.0%} of damaged\n"
                         "pairs outside the healthy 99% ellipse", loc="left", fontsize=8)
        else:
            ax.axhline(thr8, color=INK, ls="--", lw=0.8, zorder=4)
            ax.set_yscale("log")
            ax.set_ylabel("Mahalanobis distance to healthy\n(all 8 log-freq. changes)", fontsize=6.5)
            ax.set_title(f"(c) all 8 modes: {(d_m(damaged) > thr8).mean():.0%} of damaged, "
                         f"{(d_m(healthy) > thr8).mean():.0%} of\nhealthy pairs outside the healthy 99% region",
                         loc="left", fontsize=8)
            ax.set_xlabel(f"healthy PC1 ({var[0]:.0%} of variance; 1GLo, 1GLa, 2GLa) [σ]", fontsize=6.5)
        ax.tick_params(labelsize=6.5, length=2)
        ax.legend(fontsize=5.8, loc="upper left", handletextpad=0.1, borderaxespad=0.2, labelspacing=0.25)
    fig.suptitle("14  Temperature vs damage in the 8 identified frequencies", x=0.01, ha="left", fontsize=10,
                 weight="bold", y=1.02)
    print(f"  PC1 loadings {np.round(vt[0], 2)}, PC2 {np.round(vt[1], 2)}, var {np.round(var[:3], 2)}")
    print("  T-like frac / rest sigma: " + ", ".join(f"{r} {frac_t(v):.2f}/{np.linalg.norm(resid(v)):.1f}"
                                                     for r, v in zip(rows, sig)))
    for t in colors:
        sel = np.array([z[0] == t for z in dzone])
        print(f"  {ZONE_TYPE_NAME[t]}: n={sel.sum()}, outside 2D {out2(damaged[sel]).sum()}, "
              f"outside 8D {(d_m(damaged[sel]) > thr8).sum()}")
    save(fig, "14_temperature_vs_damage.png")


ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None


# ---------------------------------------------------------------------------
# 2026-10-01 upgrade vs the previous model: the widespread-damage (hier) prior and a temperature
# reading. Every variant was scored on the same held-out bridges and the same 20 draws per Table 3
# case (sbi_bridge.py fixed test seeds), so the comparisons are paired. "Previous model" is the
# sparse-prior recipe of the 27 Sep run, retrained and scored on those same bridges.
UPGRADE = {"sparse_notemp": ("previous model", MUTED),
           "hier_notemp": ("+ widespread-damage prior", BLUE),
           "hier_temp0.3": ("+ prior + temperature reading", ORANGE)}
TEMP_STEPS = [("hier_notemp", "none"), ("hier_temp0.6", "poor\n(error 0.6)"),
              ("hier_temp0.3", "fair\n(error 0.3)"), ("hier_temp0.1", "good\n(error 0.1)")]
PART_NAMES = {"B": "bearings", "G": "girders", "C": "cables", "P": "piers"}
N_CASE_DRAWS, N_TEST_DRAWS = 20, 500  # sbi_bridge.py --n-repeat / --n-test defaults used for these runs


def upgrade_results():
    tags = list(UPGRADE) + [t for t, _ in TEMP_STEPS]
    return {t: json.loads((ROOT / "results" / "sbi_bridge" / f"results_{t}_n200000.json").read_text())
            for t in dict.fromkeys(tags)}


def case_arrays(r, key):
    c = r["cases"][key]
    truth = np.array(c["truth"])
    return truth, np.array(c["p_damaged"]), np.array(c["truth_in_90"]), np.array(c["post_mean"]), \
        truth < sb.DAMAGE_THRESHOLD


def upgrade_case_label(key):
    if key.endswith("healthy"):
        return "healthy"
    words = key.split("_")
    return f"case {words[2]}\n{words[3]}\n{' '.join(words[4:])}"


def upgrade_bars(ax, groups, values, label_groups=()):
    """values {tag: [one value per group]} -> side-by-side bars per group, white gaps between them;
    numbers only on label_groups (selective direct labels)."""
    w = 0.8 / len(values)
    for i, (tag, vals) in enumerate(values.items()):
        xs = np.arange(len(groups)) + (i - (len(values) - 1) / 2) * w
        ax.bar(xs, vals, w, color=UPGRADE[tag][1], edgecolor="white", linewidth=1.2, label=UPGRADE[tag][0])
        for g in label_groups:
            ax.text(xs[g], vals[g] + 0.012, f"{vals[g]:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis="x", visible=False)


def upgrade_legend(fig, band_ax):
    """One legend under the panels: the three models, plus the honest-model band drawn on band_ax."""
    handles, labels = band_ax.get_legend_handles_labels()  # [band, model bars...]
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="upper center", ncols=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))


def honest_band(ax, n, label):
    half = 1.645 * np.sqrt(0.9 * 0.1 / n)
    ax.axhspan(0.9 - half, min(0.9 + half, 1.0), color=GRID, zorder=0, label=label)
    ax.axhline(0.9, color=INK, ls="--", lw=1.1, zorder=1)


def fig_upgrade_cases():
    res = upgrade_results()
    base = res["sparse_notemp"]
    keys = [k for k in base["cases"] if k.startswith("random/")]
    dmg_keys = [k for k in keys if case_arrays(base, k)[4].any()]
    headline = [i for i, k in enumerate(dmg_keys) if "widespread" in k]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 10.5))

    upgrade_bars(a1, [upgrade_case_label(k) for k in dmg_keys],
                 {t: [case_arrays(res[t], k)[1][case_arrays(res[t], k)[4]].mean() for k in dmg_keys] for t in UPGRADE},
                 headline)
    a1.set_ylim(0, 1.08)
    a1.set_ylabel("P(damaged), damaged zones")
    a1.set_title("Does it see the damage? Mean probability of damage on the zones the case really damaged "
                 "(higher is better)", loc="left")

    honest_band(a2, N_CASE_DRAWS, f"where an honest model lands with {N_CASE_DRAWS} draws")
    upgrade_bars(a2, [upgrade_case_label(k) for k in dmg_keys],
                 {t: [case_arrays(res[t], k)[2][case_arrays(res[t], k)[4]].mean() for k in dmg_keys] for t in UPGRADE},
                 headline)
    a2.set_ylim(0, 1.08)
    a2.set_ylabel("truth inside 90% range")
    a2.set_title("Is it honest? Share of draws whose 90% range contains the true stiffness, damaged zones "
                 "(dashed = 0.90 target)", loc="left")

    upgrade_bars(a3, [upgrade_case_label(k) for k in keys],
                 {t: [case_arrays(res[t], k)[1][~case_arrays(res[t], k)[4]].max() for k in keys] for t in UPGRADE})
    a3.set_ylim(0, 0.7)
    a3.set_ylabel("worst P(damaged), intact zones")
    a3.set_title("False alarms: highest damage probability on a zone the case left intact (lower is better)",
                 loc="left")
    for a in (a1, a2, a3):
        a.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"15  Upgraded bridge model vs the previous one on the Table 3 damage cases\n     {N_CASE_DRAWS} draws "
                 "per case with random temperature and model parameters, the same draws for every model",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    upgrade_legend(fig, a2)
    save(fig, "15_sbi_upgrade_cases.png")


def fig_upgrade_widespread():
    res = upgrade_results()
    keys = [("random/damage_case_3_bearing_severe_widespread", "Case 3: bearings degraded together"),
            ("random/damage_case_7_cable_severe_widespread", "Case 7: cables weakened together")]
    widths = [case_arrays(res["sparse_notemp"], k)[4].sum() for k, _ in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, gridspec_kw=dict(width_ratios=widths))
    for ax, (key, title) in zip(axes, keys):
        truth, _, _, _, dmg = case_arrays(res["sparse_notemp"], key)
        idx = np.flatnonzero(dmg)
        ax.axhline(1.0, color=MUTED, lw=1, ls=":")
        for j, i in enumerate(idx):
            ax.plot([j - 0.4, j + 0.4], [truth[i]] * 2, color=INK, lw=2.2, zorder=3,
                    label="true stiffness" if j == 0 else None)
        for k, tag in enumerate(UPGRADE):
            mean, in90 = case_arrays(res[tag], key)[3], case_arrays(res[tag], key)[2]
            xs = np.arange(len(idx)) + (k - 1) * 0.24
            ax.scatter(xs, mean[idx], s=70, color=UPGRADE[tag][1], edgecolor="white", linewidth=1.5, zorder=4,
                       label=UPGRADE[tag][0])
            for x, i in zip(xs, idx):
                ax.text(x, 0.215, f"{in90[i]:.0%}".replace("%", ""), ha="center", fontsize=7, color=INK2)
        ax.set_xticks(range(len(idx)), [sb.ZONES[i] for i in idx])
        ax.set_xlim(-0.6, len(idx) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
    axes[0].set_ylim(0.15, 1.06)
    axes[0].set_ylabel("stiffness (1.0 = healthy)")
    axes[0].text(-0.55, 0.25, "% of draws with the truth\ninside the 90% range:", fontsize=7, color=INK2, va="bottom")
    axes[0].text(-0.55, 1.015, "healthy", fontsize=7.5, color=MUTED, va="bottom")
    axes[1].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("16  The two cases the previous model got confidently wrong: estimated vs true stiffness, zone by zone\n"
                 f"     (mean estimate over {N_CASE_DRAWS} draws, random temperature and model parameters)",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "16_sbi_upgrade_widespread.png")


def fig_upgrade_detection():
    res = upgrade_results()
    parts = list(PART_NAMES)
    tests = [("sparse", "Bridges with ordinary damage (zones damaged one at a time)"),
             ("hier", "Bridges where whole part types can degrade together")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey="row")
    for col, (test, title) in enumerate(tests):
        a_auc, a_cov = axes[:, col]
        a_auc.axhline(0.5, color=MUTED, ls=":", lw=1.2)
        upgrade_bars(a_auc, [PART_NAMES[p] for p in parts],
                     {t: [res[t]["test"][test]["by_type"][p]["auroc"] for p in parts] for t in UPGRADE},
                     range(len(parts)) if test == "hier" else [parts.index("P")])
        a_auc.set_ylim(0.45, 1.04)
        a_auc.set_title(title, loc="left")
        honest_band(a_cov, N_TEST_DRAWS, f"where an honest model lands with {N_TEST_DRAWS} test bridges")
        upgrade_bars(a_cov, [PART_NAMES[p] for p in parts],
                     {t: [res[t]["test"][test]["by_type"][p]["coverage_90"] for p in parts] for t in UPGRADE})
        a_cov.set_ylim(0.75, 1.0)
    axes[0, 0].set_ylabel("detection AUROC (0.5 = chance)")
    axes[1, 0].set_ylabel("truth inside 90% range")
    fig.suptitle(f"17  Held-out test bridges ({N_TEST_DRAWS} per set, identical for every model): detection (top) and "
                 "honesty of the 90% ranges (bottom)\n     Single training run per model: gaps below ~0.02 are "
                 "within run-to-run noise", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    upgrade_legend(fig, axes[1, 0])
    save(fig, "17_sbi_upgrade_detection.png")


def fig_upgrade_temperature():
    res = upgrade_results()
    tags, labels = zip(*TEMP_STEPS)
    xs = np.arange(len(tags))
    ring = {t: UPGRADE[t][1] for t in tags if t in UPGRADE}  # same colour as in figures 15-17
    panels = [("Bearings detected, ordinary damage", "AUROC",
               lambda t: res[t]["test"]["sparse"]["by_type"]["B"]["auroc"], (0.85, 0.95)),
              ("Bearings detected, whole types degrading", "AUROC",
               lambda t: res[t]["test"]["hier"]["by_type"]["B"]["auroc"], (0.85, 0.95)),
              ("Case 3 honesty: truth inside 90% range", "share of draws",
               lambda t: case_arrays(res[t], "random/damage_case_3_bearing_severe_widespread")[2][
                   case_arrays(res[t], "random/damage_case_3_bearing_severe_widespread")[4]].mean(), (0.4, 1.05))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (title, ylabel, get, ylim) in zip(axes, panels):
        ys = [get(t) for t in tags]
        prev = get("sparse_notemp")
        if title.startswith("Case 3"):
            honest_band(ax, N_CASE_DRAWS, None)
        ax.axhline(prev, color=MUTED, ls="--", lw=1.2, zorder=1)
        ax.text(len(tags) - 0.55, prev, "previous model", fontsize=7.5, color=MUTED, ha="right",
                va="bottom" if prev < np.mean(ylim) else "top")
        ax.plot(xs, ys, color=INK2, lw=1.4, zorder=2)
        ax.scatter(xs, ys, s=70, color=[ring.get(t, INK2) for t in tags], edgecolor="white", linewidth=1.5, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + (ylim[1] - ylim[0]) * 0.03, f"{y:.3f}" if ylabel == "AUROC" else f"{y:.2f}",
                    ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(xs, labels, fontsize=8)
        ax.set_xlabel("temperature reading given to the model")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("18  How much the temperature reading helps, by how accurate it is (all with the widespread-damage "
                 "prior)\n     Clearest gain where whole part types degrade; a near-perfect reading makes case 3 overconfident.\n"
                 "     Single training run per model: gaps below ~0.02 are noise",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "18_sbi_upgrade_temperature.png")


def main():
    t0 = time.time()
    FIG.mkdir(exist_ok=True)
    if ONLY:  # --only fig_a,fig_b: just these, into figures/ or (with --paper) paper/figures/
        for name in ONLY:
            globals()[name]()
        print(f"done in {time.time() - t0:.0f}s")
        return
    if PAPER:  # the figures main.tex includes, plus 13 and 14 written for it
        for f in (fig_bridge_model, fig_sbi_patterns, fig_bridge_zones, fig_bridge_mode_pairing,
                  fig_bridge_sbi_cases, fig_bridge_sbi_detection, fig_paper_modes, fig_temperature_vs_damage):
            f()
        print(f"done in {time.time() - t0:.0f}s")
        return
    fig_pipeline()
    fig_bridge_model()
    fig_calibration()
    fig_mode_shapes()
    fig_damage_sensitivity()
    fig_time_history()
    fig_iasc_frame()
    fig_sbi_patterns()
    fig_sbi_coverage()
    fig_bridge_zones()
    fig_bridge_mode_pairing()
    fig_bridge_sbi_cases()
    fig_bridge_sbi_detection()
    fig_paper_modes()
    fig_temperature_vs_damage()
    fig_upgrade_cases()
    fig_upgrade_widespread()
    fig_upgrade_detection()
    fig_upgrade_temperature()
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
