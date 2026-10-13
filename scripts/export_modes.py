"""
Export the calibrated healthy bridge's geometry, sensors and first 12 mode
shapes to figures/interactive/data/modes.json for the interactive 3D mode viewer.

Axes: x longitudinal (P31 at 0 -> P35 at 530.8 m), y lateral, z vertical
(deck at 0). The model is a centreline stick model with no deck width.

Component shares per mode are given two ways:
- kinetic energy with the model's lumped nodal masses (ops.nodeMass). Pier
  and cable elements carry no mass in fe_model, so their kinetic share is 0 by
  construction; it only splits deck vs pylon.
- modal strain energy per element group. Beams come from OpenSees element
  forces (eleResponse 'globalForce' with the eigenvector imposed through
  setNodeDisp). zeroLength bearings and corotTruss cables do not refresh their
  forces on setNodeDisp (they read back 0), so those two are computed
  analytically from the spring / EA/L stiffness. Every mode is checked
  against the Rayleigh identity sum(U) = 0.5 * omega^2 * phi' M phi.

The bearing-vs-pier number: each pier (fixed base, massless top node) is in
series with its rubber bearing, so under the same force the pier deforms
k_bearing / k_pier as much as the bearing. Longitudinal and lateral bearings
are 1e4x / 60x softer than the pier's bending stiffness, so the pier barely
moves -- this is why pier damage is nearly invisible to deck sensors.

One build + eigen + energy pass, well under a second on a laptop.
"""
import json
import sys
from pathlib import Path

import numpy as np
import openseespy.opensees as ops
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fe_model as fm
from calibrate import TARGETS

ROOT = Path(__file__).parents[1]
OUT = ROOT / "figures/interactive/data/modes.json"
N_MODES = 12
REF_MODES = [1, 2, 3, 4, 7, 9, 10, 11]  # = sbi_bridge.REF_MODES
PAPER_NAMES = ["1GVe", "1GLo", "1GLa", "2GVe", "3GVe", "4GVe", "2GLa", "5GVe"]  # = make_figures.PAPER_NAMES
PIERS = ["P31", "P32", "P33", "P35"]
DIRS = ["longitudinal", "lateral", "vertical"]


def sig(v, n=4):
    if isinstance(v, dict):
        return {k: sig(x, n) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [sig(x, n) for x in v]
    if isinstance(v, (float, np.floating)):
        return 0.0 if v == 0 else float(f"{float(v):.{n - 1}e}")
    if isinstance(v, np.integer):
        return int(v)
    return v


CAL = json.loads((ROOT / "calibrated_params.json").read_text())
m = fm.build_model(CAL)
p = m["params"]

# --- nodes: compact index in a fixed order deck (by x), pylon (by z), piers ---
node_rows = [(t, "deck", {"x": x}) for x, t in sorted(m["deck_nodes"].items())]
pylon_tags = sorted({t for e in m["pylon_elements"] for t in ops.eleNodes(e)}, key=lambda t: ops.nodeCoord(t)[2])
node_rows += [(t, "pylon", {}) for t in pylon_tags]
for name in PIERS:
    node_rows += [(m["pier_bottom"][name], "pier_base", {"pier": name}),
                  (m["pier_top"][name], "pier_top", {"pier": name})]
tags = [t for t, _, _ in node_rows]
assert sorted(tags) == sorted(ops.getNodeTags()), "exported node set != model node set"
idx = {t: i for i, t in enumerate(tags)}
xyz = np.array([ops.nodeCoord(t) for t in tags])
mass = np.array([ops.nodeMass(t)[0] for t in tags])
comp_of_node = np.array([c.split("_")[0].replace("pier", "piers") for _, c, _ in node_rows])

# --- elements ---
girder = [[idx[m["deck_nodes"][x0]], idx[m["deck_nodes"][x1]]] for _, x0, x1 in m["girder_elements"]]
pylon = [[idx[a] for a in ops.eleNodes(e)] for e in m["pylon_elements"]]
piers = [[idx[m["pier_bottom"][n]], idx[m["pier_top"][n]]] for n in PIERS]
cables = []
anchors = fm.cable_anchor_xs(p)
for et, ax in m["cable_elements"]:
    a, b = ops.eleNodes(et)
    assert a in pylon_tags and b == m["deck_nodes"][ax], "cable is not a single pylon->deck truss"
    cables.append(dict(nodes=[idx[a], idx[b]], deck_x=ax, side="back" if ax in anchors["back"] else "main"))
bearings = []
for name, et in m["bearing_elements"].items():
    a, b = ops.eleNodes(et)
    assert np.allclose(ops.nodeCoord(a), ops.nodeCoord(b)), f"bearing {name} nodes do not coincide"
    bearings.append(dict(name=name, nodes=[idx[a], idx[b]]))
n_ele = len(girder) + len(pylon) + len(piers) + len(cables) + len(bearings)
assert n_ele == len(ops.getEleTags()), (n_ele, len(ops.getEleTags()))
fixed = [idx[m["pier_bottom"][n]] for n in PIERS] + [idx[m["pylon_base"]]]

# --- sensors ---
real = pd.read_csv(ROOT / "data/bridge_nodes.csv").set_index("sensor_id")
out_nodes = fm.sensor_output_nodes(m)
readers = {}
for sid in fm.SENSOR_LOCATIONS:
    readers.setdefault(out_nodes[sid], []).append(sid)
sensors = []
for sid, loc in fm.SENSOR_LOCATIONS.items():
    r = real.loc[sid]
    assert r["node_id"] == loc["node_id"] and r["type"] == loc["kind"]
    t = out_nodes[sid]
    sensors.append(dict(
        id=sid, name=loc["node_id"], kind=loc["kind"], pier=loc.get("pier"),
        node=idx[t], node_xyz=xyz[idx[t]].tolist(),
        real_xyz=[float(r["x_coord"]), float(r["y_coord"]), float(r["z_coord"])],
        shares_node_with=[s for s in readers[t] if s != sid]))

# --- modes ---
lam = np.array(ops.eigen(N_MODES))
freqs = np.sqrt(lam) / (2 * np.pi)
field = {mode: (name, f) for mode, name, (_, f) in zip(REF_MODES, PAPER_NAMES, TARGETS)}
deck_mask = comp_of_node == "deck"
bearing_k = {n: np.array([p["bearing_k_long"], p["bearing_k_lat"], p["bearing_k_vert"]] + 3 * [p["bearing_k_rot"]])
             for n in PIERS}
bearing_k["PYLON"] = np.array([p["pylon_bearing_k_long"], p["pylon_bearing_k_lat"], p["bearing_k_vert"]]
                              + 3 * [p["bearing_k_rot"]])
cable_geom = []
for et, _ in m["cable_elements"]:
    a, b = ops.eleNodes(et)
    d = np.array(ops.nodeCoord(b)) - np.array(ops.nodeCoord(a))
    cable_geom.append((a, b, d / np.linalg.norm(d), p["cable_E"] * p["cable_A"] / np.linalg.norm(d)))

modes = []
for k in range(1, N_MODES + 1):
    phi6 = np.array([ops.nodeEigenvector(t, k) for t in tags])
    for t, v in zip(tags, phi6):
        for d in range(6):
            ops.setNodeDisp(t, d + 1, v[d], "-commit")

    ke = mass[:, None] * phi6[:, :3] ** 2
    ke_share = {c: ke[comp_of_node == c].sum() / ke.sum() for c in ["deck", "pylon", "piers"]}
    ke_share["cables"] = 0.0

    def beam_u(e):
        u = np.concatenate([ops.nodeDisp(n) for n in ops.eleNodes(e)])
        return 0.5 * np.array(ops.eleResponse(e, "globalForce")) @ u
    su = dict(deck=sum(beam_u(e) for e, _, _ in m["girder_elements"]),
              pylon=sum(beam_u(e) for e in m["pylon_elements"]),
              piers=sum(beam_u(e) for e in m["pier_elements"].values()),
              cables=sum(0.5 * kax * ((phi6[idx[b], :3] - phi6[idx[a], :3]) @ e) ** 2 for a, b, e, kax in cable_geom),
              bearings=sum(0.5 * bearing_k[n] @ (phi6[idx[ops.eleNodes(et)[1]]] - phi6[idx[ops.eleNodes(et)[0]]]) ** 2
                           for n, et in m["bearing_elements"].items()))
    u_total = sum(su.values())
    rayleigh = 0.5 * lam[k - 1] * ke.sum()
    assert abs(u_total / rayleigh - 1) < 1e-4, (k, u_total, rayleigh)
    su_share = {c: v / u_total for c, v in su.items()}

    # series bearing/pier: bearing slip = deck - support top, pier deformation = top - base (base fixed)
    per_support = {}
    for n in PIERS + ["PYLON"]:
        a, b = ops.eleNodes(m["bearing_elements"][n])  # support top node, deck node
        base = m["pier_bottom"][n] if n != "PYLON" else m["pylon_base"]
        slip = phi6[idx[b], :3] - phi6[idx[a], :3]
        defo = phi6[idx[a], :3] - phi6[idx[base], :3]
        per_support[n] = dict(bearing_slip=slip, pier_deformation=defo,
                              ratio=np.linalg.norm(defo) / np.linalg.norm(slip))

    trans = phi6[:, :3]
    scale = np.abs(trans).max()
    flat = trans.ravel()
    sign = np.sign(flat[np.argmax(np.abs(flat))])
    disp = trans * sign / scale
    deck_abs = np.abs(trans[deck_mask]).sum(0)
    deck_sq = (trans[deck_mask] ** 2).sum(0)
    for s in per_support.values():
        s["bearing_slip"] = (s["bearing_slip"] * sign / scale).tolist()
        s["pier_deformation"] = (s["pier_deformation"] * sign / scale).tolist()

    name, f_field = field.get(k, ("FE-only", None))
    direction = DIRS[int(np.argmax(deck_abs))]
    if k in field:
        assert direction == TARGETS[REF_MODES.index(k)][0], (k, direction)
    modes.append(dict(
        mode=k, freq_hz=freqs[k - 1], period_s=1 / freqs[k - 1],
        label=name, identified=k in field, field_freq_hz=f_field,
        direction=direction, deck_direction_share=dict(zip(DIRS, deck_sq / deck_sq.sum())),
        kinetic_energy_share=ke_share, strain_energy_share=su_share,
        pier_over_bearing=dict(mean_4_piers=np.mean([per_support[n]["ratio"] for n in PIERS]),
                               max_pier_deformation=max(np.linalg.norm(per_support[n]["pier_deformation"]) for n in PIERS),
                               per_support=per_support),
        disp=disp.tolist()))

data = dict(
    meta=dict(
        model="Shin-Nakagawa cable-stayed bridge, OpenSeesPy centreline stick model (scripts/fe_model.py), calibrated healthy state",
        source="calibrated_params.json", generator="scripts/export_modes.py", date="2026-09-28",
        units=dict(length="m", frequency="Hz", period="s"),
        axes=dict(x="longitudinal (P31 at x=0, P35 at x=530.8)", y="lateral (transverse)", z="vertical (deck at z=0)"),
        supports=dict(P31=fm.X_P31, P32=fm.X_P32, P33=fm.X_P33, PYLON=fm.X_PYLON, P35=fm.X_P35),
        identified_modes=REF_MODES, identified_names=PAPER_NAMES,
        n_nodes=len(tags), n_elements=n_ele,
        rounding="all floats rounded to 4 significant digits",
        disp_scaling=("modes[i].disp[j] = [ux, uy, uz] of node j, eigenvector translations divided by the largest |translation| "
                      "over all nodes and axes in that mode (so max is exactly 1), sign chosen so that entry is +1. "
                      "Rotations are not exported. Mode shapes are dimensionless; pick any visual amplitude."),
        kinetic_energy_share=("share of sum(m * phi^2) over the model's lumped translational nodal masses. Pier and cable "
                              "elements have no mass in the model, so piers and cables are 0 by construction."),
        strain_energy_share=("share of modal strain energy 0.5 * phi' K phi per element group (deck girders, pylon incl. "
                             "the concrete leg below the deck, pier columns, stay cables, the 5 rubber bearings). "
                             "Checked per mode against 0.5 * omega^2 * phi' M phi to 1e-4. This is the number that shows "
                             "how little the piers deform."),
        pier_over_bearing=("per support: bearing_slip = deck node minus support-top node translation, pier_deformation = "
                           "pier top minus fixed pier base (for PYLON: pylon node at deck level minus pylon base, i.e. the "
                           "concrete leg); both in the same scaled units as disp. ratio = |pier_deformation| / |bearing_slip|. "
                           "mean_4_piers averages ratio over P31, P32, P33, P35 (PYLON excluded). Series springs: "
                           "ratio ~ k_bearing / k_pier, so small means the pier is hidden behind a softer bearing. In vertical modes the "
                           "bearings barely slip at all, so the ratio is larger (0.1-0.8) while both motions are tiny; "
                           "max_pier_deformation = largest |pier_deformation| over the 4 piers, i.e. as a fraction of the "
                           "mode's peak translation, is the unambiguous 'piers barely move' number."),
        sensors_note=("The FE model has no deck width. Real sensors at some stations come in +y/-y pairs (e.g. S42/S45 at "
                      "x=96.4); both read the same centreline FE node (shares_node_with lists the others). real_xyz is the "
                      "surveyed position from data/bridge_nodes.csv (y = transverse offset, m). Pier sensors read the "
                      "pier-top node (deck level, below the bearing); the PYLON pier sensors S27/S28 read the deck node "
                      "above the pylon bearing. S22/S23/S24 are surveyed at x=533.0 but read the deck-end/P35 node at 530.8. "
                      "Ground (free-field) sensors have no FE node and are not exported."),
    ),
    geometry=dict(
        nodes=[dict(i=i, tag=t, component=c, xyz=xyz[i].tolist(), **extra) for i, (t, c, extra) in enumerate(node_rows)],
        fixed_nodes=fixed,
        elements=dict(deck=girder, pylon=pylon, piers=piers, cables=cables, bearings=bearings),
    ),
    sensors=sensors,
    modes=modes,
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(sig(data), separators=(",", ":")))
print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB): {len(tags)} nodes, {n_ele} elements, "
      f"{len(sensors)} sensors, {N_MODES} modes")
for md in modes:
    print(f"  mode {md['mode']:2d} {md['freq_hz']:.4f} Hz {md['label']:7s} {md['direction']:12s} "
          f"SE piers {md['strain_energy_share']['piers']:.2e} bearings {md['strain_energy_share']['bearings']:.3f} "
          f"pier/bearing {md['pier_over_bearing']['mean_4_piers']:.2e} "
          f"max pier def {md['pier_over_bearing']['max_pier_deformation']:.1e}")
