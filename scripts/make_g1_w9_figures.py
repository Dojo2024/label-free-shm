"""Figures 28-31: Gate 1 on Shin-Nakagawa (g1_shin_nakagawa.py) and W9 stage 0 on SeisGPT's data
(w9_stage0.py), both run 13 Oct 2026.
  28 modal superposition against OpenSees: two sensor traces, then every record's error per simulator
  29 identifying modes from earthquake records: output-only SSI vs input-output PO-MOESP, clean and 10% noise
  30 the SeisGPT shake table: first frequency per run, measured vs SeisGPT's FE response
  31 storey damage on SeisGPT's condensed stiffness matrices: five definitions tested
Figure 28 rebuilds one record's simulations (OpenSees model + modes, about 10 s); the rest reads results/.
    python scripts/make_g1_w9_figures.py [--paper | --out DIR]
(--paper and --out leave out the figure headlines and explanation paragraphs, for use under a caption)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = "--paper" in sys.argv
OUTDIR = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
STRIP = PAPER or OUTDIR is not None
FIG = Path(OUTDIR).resolve() if OUTDIR else ROOT / ("paper/figures" if PAPER else "figures")
FIG.mkdir(parents=True, exist_ok=True)

# validated categorical slots 1-4 (dataviz palette; aqua and yellow sit below 3:1 on the surface, so every
# series carrying them is also direct-labelled)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "normal",
    "axes.titlelocation": "left", "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "white",
    "lines.linewidth": 1.6})


def save(fig, name):
    if STRIP:
        if fig._suptitle is not None:
            fig._suptitle.remove()
        for t in [t for t in fig.texts if t.get_gid() == "explain"]:
            t.remove()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG}/{name}", flush=True)


def title(fig, text, y=1.0):
    fig.suptitle(text, x=0.01, y=y, ha="left", fontsize=11.5, fontweight="bold", color=INK)


def explain(fig, text, y):
    fig.text(0.01, y, text, ha="left", va="top", fontsize=8.5, color=INK2, gid="explain", wrap=True)


def strip(ax, x, vals, color, rng, size=7, alpha=0.55):
    vals = np.asarray(vals, float)
    ax.scatter(x + rng.uniform(-0.18, 0.18, vals.size), vals, s=size, color=color, alpha=alpha, lw=0)
    ax.plot([x - 0.26, x + 0.26], [np.median(vals)] * 2, color=INK, lw=1.6)


G1 = {nz: json.load(open(RES / f"g1_shin_nakagawa/{f}")) for nz, f in ((0.0, "g1.json"), (0.1, "g1_noise0.1.json"))}
S0 = json.load(open(RES / "w9_stage0/stage0.json"))


def fig28():
    import g1_shin_nakagawa as g
    variant, ev = "healthy_baseline", "IBR0071906170800"  # the strongest of the 30 events (PGA 1.27 m/s2)
    csv = ROOT / f"output/complete_simulation_dataset/{variant}/{ev}.csv"
    hdr = open(csv).readline().strip().split(",")[1:]
    y_os = np.loadtxt(csv, delimiter=",", skiprows=1)[:, 1:]
    d = g.modal_data(variant)
    ug, dt = g.ground(ev, y_os.shape[0])
    y_re = g.newmark_coupled(-(ug @ d["gamma"].T), d["w"], d["Cr"], dt) @ d["S"].T
    y_cx = g.complex_superpose(ug, d, dt)
    rms = np.sqrt((y_os ** 2).mean(0))
    deck = max(np.where(d["is_mass"])[0], key=lambda c: rms[c])
    pier = max(np.where(~d["is_mass"])[0], key=lambda c: rms[c])
    t = (np.arange(y_os.shape[0]) + 1) * dt
    k = int(np.argmax(np.abs(y_os[:, deck])))
    win = (t > t[k] - 6) & (t < t[k] + 6)

    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.22, top=0.80)
    for i, (c, where) in enumerate(((deck, "deck sensor (has mass)"), (pier, "pier-top sensor (massless node)"))):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(t[win], y_os[win, c], color=INK, lw=2.4, label="OpenSees (reference)")
        ax.plot(t[win], y_re[win, c], color=ORANGE, lw=1.3, label="real modes, projected damping")
        ax.plot(t[win], y_cx[win, c], color=BLUE, lw=1.3, ls=(0, (4, 2)), label="complex modes")
        e_re = np.sqrt(((y_re[:, c] - y_os[:, c]) ** 2).mean()) / rms[c]
        e_cx = np.sqrt(((y_cx[:, c] - y_os[:, c]) ** 2).mean()) / rms[c]
        ax.set_title(f"{hdr[c].replace('_', ' ')}: {where}\nwhole-record error: real modes {100 * e_re:.1f}%, "
                     f"complex modes {100 * e_cx:.3f}%")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("relative acceleration (m/s²)")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    res = G1[0.0]["records"]
    sims = (("textbook modal\ndamping", "sp_pooled_classical", None, MUTED),
            ("real modes,\nprojected damping", "sp_pooled", "sp_ch_max", ORANGE),
            ("complex modes\n(all 1,013 kept)", "sp_cx_pooled", "sp_cx_ch_max", BLUE))
    rng = np.random.default_rng(0)
    for j, (key_i, lab) in enumerate(((1, "all 60 channels together"), (2, "worst single channel"))):
        ax = fig.add_subplot(gs[1, j])
        xs = []
        for x, s in enumerate(sims):
            if s[key_i] is None:
                continue
            strip(ax, x, [100 * r[s[key_i]] for r in res], s[3], rng)
            xs.append(x)
        ax.axhline(2, color=INK2, lw=1, ls=(0, (3, 2)))
        ax.text(max(xs) + 0.85, 2.3, "gate 2%", ha="right", va="bottom", fontsize=8, color=INK2)
        ax.set_yscale("log")
        ax.set_xticks(xs, [sims[x][0] for x in xs])
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.9)
        ax.set_ylabel("RMS error vs OpenSees (%)")
        ax.set_title(f"Every record: {lab}\n(270 records; bar = median)")
        ax.grid(axis="x", visible=False)
    title(fig, "Gate 1: a modal simulator that reproduces OpenSees to 0.2% needs complex modes", y=0.985)
    explain(fig, "Top: the strongest of the 30 earthquakes, 12 s around its peak. Real-mode superposition is "
            "close on the deck but wrong on the pier tops: the model's piers carry no mass and the bearings and "
            "cables no Rayleigh damping, so 354 of 684 degrees of freedom are massless but damped and real "
            "modes cannot represent them. Complex modes of the full damped system, stepped with the same rule "
            "as OpenSees (Newmark average acceleration = trapezoidal), lie on top of the reference. Bottom: "
            "every record. The only complex-mode records above 2% on a single channel are one event's "
            "near-silent channel at the end of its record (all 9 bridge states of IBR0072002062019).", y=0.945)
    save(fig, "28_g1_superposition.png")


def fig29():
    fig = plt.figure(figsize=(11, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28, top=0.70)
    ax = fig.add_subplot(gs[0])
    rng = np.random.default_rng(1)
    cats = []
    for x, (nz, key, lab, col) in enumerate(((0.0, "targets", "SSI\nclean", ORANGE), (0.0, "targets_io", "PO-MOESP\nclean", BLUE),
                                              (0.1, "targets", "SSI\n10% noise", ORANGE), (0.1, "targets_io", "PO-MOESP\n10% noise", BLUE))):
        tg = [t for r in G1[nz]["records"] for t in r[key]]
        found = [100 * abs(t["err"]) for t in tg if t["found"]]
        strip(ax, x, np.maximum(found, 1e-4), col, rng, size=5, alpha=0.35)
        ax.text(x, 13, f"found {100 * len(found) / len(tg):.0f}%", ha="center", fontsize=8, color=INK2)
        cats.append(lab)
    ax.axhline(1, color=INK2, lw=1, ls=(0, (3, 2)))
    ax.text(3.9, 1.15, "gate 1%", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(8e-5, 30)
    ax.set_xlim(-0.5, 3.95)
    ax.set_xticks(range(4), cats)
    ax.set_ylabel("|frequency error| of found modes (%)")
    ax.set_title("Frequency error, every (record, excited mode) pair\n(dots at 0.0001% are below it; bar = median)")
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[1])
    groups = (("clean\nper record", 0.0, "per"), ("clean\npooled", 0.0, "pool"),
              ("10% noise\nper record", 0.1, "per"), ("10% noise\npooled", 0.1, "pool"))
    w = 0.36
    for gi, (lab, nz, how) in enumerate(groups):
        for k, (key, col, name) in enumerate((("targets", ORANGE, "output-only SSI"), ("targets_io", BLUE, "input-output PO-MOESP"))):
            src = G1[nz]["records"] if how == "per" else G1[nz]["pooled"]
            tg = [t for r in src for t in r[key]]
            v = 100 * np.mean([t.get("passed", False) for t in tg])
            x = gi + (k - 0.5) * (w + 0.04)
            ax.bar(x, v, width=w, color=col, label=name if gi == 0 else None)
            ax.text(x, v + 1.5, f"{v:.1f}" if 99 < v < 100 else f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(4), [gr[0] for gr in groups])
    ax.set_ylim(0, 112)
    ax.set_ylabel("targets passing the gate (%)")
    ax.set_title("Pass rate: within 1% and MAC ≥ 0.95\n(pooled = the 30 events of one bridge state together)")
    ax.legend(loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="x", visible=False)
    title(fig, "Gate 1: earthquake records need input-output identification", y=0.985)
    explain(fig, "270 simulated records (9 bridge states × 30 K-NET events). A target is an FE mode below the band "
            "that holds ≥ 1% of the record's response. Output-only SSI assumes white, steady excitation; an "
            "earthquake is neither, and on ~100 s records it finds few modes and shifts close pairs by ~1%. "
            "PO-MOESP uses the ground motion as the input (the WSN has free-field sensors), and passes 99.6% of "
            "clean single records. With 10% sensor noise a single record is too short; pooling the 30 events "
            "of a bridge state recovers 95.5%.", y=0.93)
    save(fig, "29_g1_identification.png")


def fig30():
    st = S0["shake_table"]
    runs = st["runs"]
    fig = plt.figure(figsize=(11, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.25, top=0.70)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(runs))
    m = np.array([r["f1_arx_meas_cps"] for r in runs])
    f = np.array([r["f1_arx_fem_cps"] for r in runs])
    for v, col, lab in ((m, BLUE, "measured specimen"), (f, ORANGE, "SeisGPT's FE response")):
        ax.plot(x, 100 * (v / v[0] - 1), color=col, marker="o", ms=5)
        ax.text(x[-1] + 0.15, 100 * (v[-1] / v[0] - 1), lab, va="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [f"run {i + 1}\n{r['pga'] / 1000:.1f}" for i, r in enumerate(runs)])
    ax.set_xlim(-0.3, len(runs) + 1.6)
    ax.set_xlabel("run, in test order, with peak table acceleration (×1000, file units)")
    ax.set_ylabel("first frequency, change from run 1 (%)")
    ax.set_title("First natural frequency per run (time-domain ARX fit)")
    ax = fig.add_subplot(gs[1])
    ax.bar(x, [r["corr_reported"] for r in runs], color=BLUE, width=0.6)
    ax.set_xticks(x, [f"{i + 1}" for i in x])
    ax.set_xlabel("run")
    ax.set_ylim(0, 1)
    ax.set_title("SeisGPT's reported measured-vs-FE correlation")
    ax.grid(axis="x", visible=False)
    title(fig, "W9 stage 0: the shake-table specimen does not soften from run to run", y=0.985)
    explain(fig, "SeisGPT's 1/10-scale 43-storey test, 7 runs of rising intensity. Measured f1 stays within 0.6% "
            "of run 1 while the shaking grows about 7×, so the runs are not a progressive damage series (Old ADA "
            "stays P2's fourth structure). SeisGPT's own FE response softens about 9% at the strongest runs, "
            "which is where their reported agreement with the measurements falls. The files give no sampling "
            "interval (recovered from the FE response), so changes are shown relative to run 1.", y=0.93)
    save(fig, "30_w9_shake_table.png")


def fig31():
    fig = plt.figure(figsize=(11, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.3, top=0.70)
    ax = fig.add_subplot(gs[0])
    rng = np.random.default_rng(2)
    approx = (("shear\nchain\n(entry-wise)", "storey_springs", MUTED), ("shear +\nbending\n(entry-wise)", "shear_flexure", ORANGE),
              ("shear +\nbending\n(fit to modes)", "shear_flexure_modal", BLUE))
    for x, (lab, key, col) in enumerate(approx):
        strip(ax, x, [100 * abs(r["f_err_chain"][0]) for r in S0[key]["models"]], col, rng)
    ax.axhline(1, color=INK2, lw=1, ls=(0, (3, 2)))
    ax.text(-0.45, 1.15, "target 1%", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(range(3), [a[0] for a in approx])
    ax.set_xlim(-0.5, 2.95)
    ax.set_ylabel("|error| in healthy f1 (%)")
    ax.set_title("Approximate definitions: how far the fitted part\nis from the building (130 matrices; bar = median)")
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[1])
    pos = ("bottom", "middle", "top")
    defs = (("drift congruence (exact)", "drift_congruence", lambda r, p: r[f"f1_drop_{p}"], AQUA),
            ("series drift flexibility (exact)", "drift_flexibility", lambda r, p: r[f"f1_drop_{p}"], YELLOW),
            ("shear + bending, entry-wise", "shear_flexure", lambda r, p: r["dmg20"][p]["f1_drop"], ORANGE),
            ("shear + bending, fit to modes", "shear_flexure_modal", lambda r, p: r["dmg20"][p]["f1_drop"], BLUE))
    for lab, key, get, col in defs:
        v = [100 * np.median([max(get(r, p), 1e-6) for r in S0[key]["models"]]) for p in pos]
        ax.plot(range(3), v, color=col, marker="o", ms=5)
        ax.text(2.1, v[-1], lab, va="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(3), ["bottom storey", "middle storey", "top storey"])
    ax.set_xlim(-0.2, 3.9)
    ax.set_ylabel("median drop in f1 (%)")
    ax.set_title("Damage behaviour: 20% loss in one storey\n(physics: largest at the bottom, smallest at the top)")
    ax.grid(axis="x", visible=False)
    title(fig, "W9 stage 0: no simple storey-damage definition works on SeisGPT's matrices", y=0.985)
    explain(fig, "SeisGPT releases only condensed lateral stiffness matrices (65 buildings × 2 directions), so "
            "storey damage must be defined on them. Fitted chains miss the healthy building (left; the shear "
            "chain also breaks positive definiteness for 77% of mid-storey damages and is left out on the right). "
            "Exact definitions keep the healthy building but respond wrongly: damage is nearly invisible, or "
            "does not fall with height as bending buildings require (right). Stage 1 therefore trains on our own "
            "OpenSeesPy buildings with damage defined on elements.", y=0.93)
    save(fig, "31_w9_storey_damage.png")


if __name__ == "__main__":
    fig29()
    fig30()
    fig31()
    fig28()
