"""
Shin-Nakagawa Bridge — parametric 3D OpenSeesPy model.

Shin-Nakagawa Bridge is a real, documented structure: a 530 m single-plane,
single-pylon, steel box-girder cable-stayed bridge in Ibaraki, Japan, seismically
isolated with laminated rubber bearings (LRB) at all piers and the pylon, damaged
in the 2011 Tohoku earthquake, monitored 2017-2020 by a WSN recording 63 events
(Siringoringo et al., J. Bridge Eng. 2022; SHM2019 proceedings). Span layout here
(58.9 + 75.0 + 113.0 + 283.9 = 530.8 m) matches the real bridge length.

Exact section properties (girder plate thickness, cable areas, bearing stiffness)
are not publicly available to us (paywalled). Instead of inventing precise-looking
numbers, this model exposes them as named CALIBRATABLE parameters (see
`DEFAULT_PARAMS`) and `calibrate.py` fits them against the six field-identified
frequencies in Table 2 of the project write-up. Treat every value in
`DEFAULT_PARAMS` as an assumption until calibration confirms it reproduces the
target modal behavior — this is standard FE model updating, not a claim that
these are the bridge's real properties.

Units: kN, m, t (tonne), s.  X = longitudinal, Y = lateral, Z = vertical (up).

Geometry (from the write-up's Fig. 2, confirmed against real bridge length):
  P31 (x=0) -- 58.9m -- P32 (x=58.9) -- 75.0m -- P33 (x=133.9)
  -- 113.0m -- P34/pylon (x=246.9) -- 283.9m -- P35 (x=530.8)

Simplifications, documented rather than hidden:
  - Uniform assumed pier height (`pier_height`), not per-pier real heights.
  - Pylon geometry from Siringoringo et al. 2022 (J. Bridge Eng., Fig. 2):
    top 104.6m above deck, cable anchorage zone = upper 54.0m, a 22.7m concrete
    leg below deck (fixed at its base; the 48m caisson is not modeled). Steel
    section above the deck, concrete below, as in the paper's own FE model.
  - Deck-pylon and deck-pier connections are all 6-DOF bearings (zeroLength
    springs) for the real laminated rubber bearings; the paper: "seismically
    isolated with laminated rubber bearings at the main pylon and piers".
    The post-2011 viscous dampers at the pylon are not modeled (no static
    stiffness, so they don't enter the eigen problem).
  - Cables: single plane, harp pairing (nearest deck anchor <-> lowest pylon
    anchor), N_CABLES_PER_SIDE per side, linear-elastic truss elements (no
    sag/Ernst-modulus correction). The paper says "15 stay cables"; Fig. 2
    reads as 15 per side, which is what we use -- ambiguous, not confirmed.
    Deck anchors are spaced evenly along each side (assumed; not in the paper).
"""
import numpy as np
import openseespy.opensees as ops

# ---------------------------------------------------------------------------
# Geometry (confirmed against real bridge span layout)
# ---------------------------------------------------------------------------
X_P31, X_P32, X_P33, X_PYLON, X_P35 = 0.0, 58.9, 133.9, 246.9, 530.8
DECK_Z = 0.0
DECK_NODE_SPACING = 10.0  # m; resolution of deck discretization

N_CABLES_PER_SIDE = 15


def cable_anchor_xs(p):
    """Deck cable anchorages per side: evenly spaced between the pylon and
    `back_anchor_start` / `main_anchor_end`, endpoints excluded. The extent is
    not in the paper, so it's a calibrated parameter."""
    return {side: [round(float(x), 3) for x in np.linspace(a, b, N_CABLES_PER_SIDE + 2)[1:-1]]
            for side, (a, b) in [("back", (p['back_anchor_start'], X_PYLON)),
                                 ("main", (X_PYLON, p['main_anchor_end']))]}

# "near <pier/PYLON>" in DAMAGE_TARGETS means within this many meters of that
# location along the deck — a documented locality assumption, not a measured
# damage-zone size.
NEAR_WINDOW = 40.0

# ---------------------------------------------------------------------------
# Sensor placement along the deck, from the WSN sensor survey (data/bridge_nodes.csv,
# not distributed) -- actual sensor x-positions, not invented. y_coord is the lateral offset
# of each sensor from the deck centerline (structures instrumented on both
# sides); our FE model is a centerline idealization with no deck width, so
# both left/right sensors at a given x necessarily read the same simulated
# value here -- a documented simplification, not distinct physics per side.
# "ground" sensors are free-field/downhole reference accelerometers, not
# points on the bridge structure -- they have no corresponding FE node and
# are excluded from SENSOR_LOCATIONS below (listed only for completeness).
# Pier sensor x-coords (0.0, 58.9, 133.9, 246.9, 533.0) match our pier
# locations (X_P31/P32/P33/PYLON/P35) to within 2.2m (533.0 vs 530.8) --
# a minor discrepancy between two real sources (write-up Fig. 2 span sum vs
# actual sensor survey), not adjusted, since it's <0.5% of total span.
SENSOR_LOCATIONS = {
    38: dict(node_id="S38", kind="girder", x=0.0),
    34: dict(node_id="S34", kind="girder", x=58.9),
    42: dict(node_id="S42", kind="girder", x=96.4),
    45: dict(node_id="S45", kind="girder", x=96.4),
    30: dict(node_id="S30", kind="girder", x=133.9),
    44: dict(node_id="S44", kind="girder", x=190.4),
    36: dict(node_id="S36", kind="girder", x=190.4),
    26: dict(node_id="S26", kind="girder", x=246.9),
    32: dict(node_id="S32", kind="girder", x=246.9),
    49: dict(node_id="S49", kind="girder", x=459.825),
    51: dict(node_id="S51", kind="girder", x=459.825),
    22: dict(node_id="S22", kind="girder", x=533.0),
    39: dict(node_id="S39", kind="pier", pier="P31"),
    40: dict(node_id="S40", kind="pier", pier="P31"),
    35: dict(node_id="S35", kind="pier", pier="P32"),
    31: dict(node_id="S31", kind="pier", pier="P33"),
    27: dict(node_id="S27", kind="pier", pier="PYLON"),
    28: dict(node_id="S28", kind="pier", pier="PYLON"),
    23: dict(node_id="S23", kind="pier", pier="P35"),
    24: dict(node_id="S24", kind="pier", pier="P35"),
}
# Exact girder-sensor x-coords -> forced into the deck node grid so "sensor
# output" is read at a real node, never interpolated from the nearest one.
SENSOR_GIRDER_XS = sorted({v["x"] for v in SENSOR_LOCATIONS.values() if v["kind"] == "girder"})

# ---------------------------------------------------------------------------
# Calibratable parameters — starting assumptions, fit by calibrate.py
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    # Deck (steel box girder): assumed section, scaled by calibration
    deck_E=210e6,       # kPa (steel)
    deck_A=1.2,          # m^2, assumed box girder cross-section area
    deck_Iy=8.0,          # m^4, weak-axis (lateral bending)
    deck_Iz=4.0,          # m^4, strong-axis (vertical bending)
    deck_J=6.0,           # m^4, torsional constant (box girders are torsionally stiff)
    deck_mass_per_len=18.0,  # t/m, assumed steel box girder + deck slab mass

    # Pylon above deck: steel A-frame (paper: "steel beam above the girder").
    # Section values are assumed and calibrated; geometry is from the paper.
    pylon_E=210e6,       # kPa (steel)
    pylon_A=1.5,          # m^2, assumed (two steel box legs)
    pylon_Iy=200.0,        # m^4
    pylon_Iz=200.0,        # m^4
    pylon_J=50.0,          # m^4
    pylon_mass_per_len=15.0,  # t/m, assumed
    pylon_top_height=104.6,   # m above deck (paper Fig. 2)
    pylon_anchor_zone=54.0,   # m, cable anchorage zone at the top (paper Fig. 2)
    # Pylon leg below deck: concrete, 22.7m to the caisson (paper Fig. 2), base fixed.
    pylon_base_depth=22.7,
    pylon_leg_E=30e6,
    pylon_leg_A=60.0,       # m^2, assumed (42m-wide pier head per Fig. 2)
    pylon_leg_I=2000.0,     # m^4, assumed near-rigid
    pylon_leg_J=3000.0,
    pylon_leg_mass_per_len=150.0,  # t/m, ~60m^2 x 2.5 t/m^3

    # Piers P31/P32/P33/P35: assumed uniform section + height
    pier_E=30e6,          # kPa
    pier_A=8.0,            # m^2
    pier_Iy=15.0,           # m^4
    pier_Iz=15.0,           # m^4
    pier_J=20.0,            # m^4
    pier_height=20.0,       # m, assumed uniform (real piers likely differ)

    # LRB isolation bearings (deck-to-pier connection): calibratable spring stiffness
    bearing_k_long=5e3,    # kN/m, longitudinal (LRBs are soft this direction by design)
    bearing_k_lat=5e3,     # kN/m, lateral
    bearing_k_vert=5e6,    # kN/m, vertical (bearings are stiff vertically)
    bearing_k_rot=1e7,     # kN*m/rad, rotational (assumed near-rigid)
    # Deck-pylon bearing: the real bridge is "seismically isolated with laminated
    # rubber bearings at the main pylon and piers" (Siringoringo et al., J. Bridge
    # Eng. 2022 abstract), so the deck is NOT framed rigidly into the tower. Own
    # params because the pylon bearing carries far more load than a pier bearing.
    pylon_bearing_k_long=5e3,  # kN/m
    pylon_bearing_k_lat=5e3,   # kN/m

    # Deck extent of the cable fans (not in the paper; calibrated)
    back_anchor_start=0.0,    # m, x where the back-span fan starts (P31 = 0)
    main_anchor_end=530.8,    # m, x where the main-span fan ends (P35 = 530.8)

    # Stay cables: assumed steel strand properties
    cable_E=195e6,        # kPa
    cable_A=0.02,          # m^2 per cable, assumed
)

# Sensor -> structural element mapping used by damage cases (Table 3), keyed by
# the sensor IDs in the write-up's Fig. 1 / Table 3.
DAMAGE_TARGETS = {
    "damage_case_1_bearing_minor_local":      dict(target="bearing", piers=["P31"], factor=0.15),
    "damage_case_2_girder_moderate_local":    dict(target="girder", near="P33", factor=0.30),
    "damage_case_3_bearing_severe_widespread": dict(target="bearing", piers=["P31", "P32", "P33", "P35", "PYLON"], factor=0.50),
    "damage_case_4_cable_minor_local":        dict(target="cable", near="PYLON", factor=0.10),
    "damage_case_5_foundation_moderate_local": dict(target="pier", piers=["P32"], factor=0.25),
    "damage_case_6_girder_minor_widespread":  dict(target="girder", near="all", factor=0.05),
    "damage_case_7_cable_severe_widespread":  dict(target="cable", near="all", factor=0.40),
    "damage_case_8_multiple_mixed_damage":    dict(target="multiple", specs=[
        dict(target="pier", piers=["P33"], factor=0.20),
        dict(target="girder", near="P32", factor=0.20),
    ]),
}


# Continuous damage zones for SBI (build_model's `zone_mult`): every bearing and
# pier by name, the deck split by span (main span in thirds), and each cable fan
# split into the half near the pylon and the half far from it. Girder elements
# are assigned by midpoint x, cables by deck-anchor x. Boundaries are fixed x
# coordinates, so they don't move when the fan-extent params change.
GIRDER_ZONES = {"G1": (X_P31, X_P32), "G2": (X_P32, X_P33), "G3": (X_P33, X_PYLON),
                "G4": (X_PYLON, 341.5), "G5": (341.5, 436.2), "G6": (436.2, X_P35)}
CABLE_ZONES = {"Cb-far": (X_P31, 134.0), "Cb-near": (134.0, X_PYLON),
               "Cm-near": (X_PYLON, 360.0), "Cm-far": (360.0, X_P35)}
ZONE_NAMES = ([f"B-{n}" for n in ["P31", "P32", "P33", "PYLON", "P35"]]
              + [f"P-{n}" for n in ["P31", "P32", "P33", "P35"]]
              + list(GIRDER_ZONES) + list(CABLE_ZONES))


def zone_of(x, zones):
    return next(z for z, (lo, hi) in zones.items() if lo <= x <= hi)


def _pier_locs():
    return {"P31": X_P31, "P32": X_P32, "P33": X_P33, "P35": X_P35}


def _target_locs():
    locs = _pier_locs()
    locs["PYLON"] = X_PYLON
    return locs


def resolve_damage_specs(damage_case):
    """DAMAGE_TARGETS[damage_case] -> list of single-target specs (case 8's
    'multiple' wrapper is unpacked into its constituent specs)."""
    if damage_case is None:
        return []
    spec = DAMAGE_TARGETS[damage_case]
    if spec.get("target") == "multiple":
        return spec["specs"]
    return [spec]


def _pier_multiplier(pier_name, specs):
    mult = 1.0
    for s in specs:
        if s.get("target") == "pier" and pier_name in s.get("piers", []):
            mult *= (1 - s["factor"])
    return mult


def _bearing_multiplier(pier_name, specs):
    mult = 1.0
    for s in specs:
        if s.get("target") == "bearing" and pier_name in s.get("piers", []):
            mult *= (1 - s["factor"])
    return mult


def _near_multiplier(x, target, specs):
    """Shared 'near <location>' matching for girder/cable damage specs."""
    locs = _target_locs()
    mult = 1.0
    for s in specs:
        if s.get("target") != target:
            continue
        near = s["near"]
        if near == "all" or abs(x - locs[near]) <= NEAR_WINDOW:
            mult *= (1 - s["factor"])
    return mult


def build_model(params=None, damage_case=None, zone_mult=None):
    """Build the OpenSeesPy model in the current interpreter. Returns a dict of
    tag registries (deck_nodes, pier_top_nodes, pier_bottom_nodes, pylon_top,
    cable_elements, girder_elements, bearing_elements, pier_elements) needed by
    downstream analysis (eigen, time-history) and by damage-case application.

    `damage_case`, if given, must be a key of DAMAGE_TARGETS (Table 3). It
    applies a fractional stiffness reduction to the targeted element(s):
    bearings and piers by name, girder/cable elements within NEAR_WINDOW
    meters of a named pier/pylon location (or all of them, for 'near: all').
    The deck-pylon connection is a rubber-bearing spring like the piers'
    (named "PYLON" in bearing_elements), so damage_case_3's "PYLON" bearing
    target is a real target. Before 2026-09-27 it was a rigid equalDOF and
    that sub-target was a no-op.

    `zone_mult`, if given, maps ZONE_NAMES entries to a stiffness multiplier,
    applied on top of any damage_case; zones left out stay at 1.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    specs = resolve_damage_specs(damage_case)
    zm = zone_mult or {}
    assert set(zm) <= set(ZONE_NAMES), set(zm) - set(ZONE_NAMES)

    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    tag = 1
    def next_tag():
        nonlocal tag
        t = tag
        tag += 1
        return t

    transf_deck = next_tag()
    ops.geomTransf('Linear', transf_deck, 0.0, 1.0, 0.0)
    transf_vert = next_tag()
    ops.geomTransf('Linear', transf_vert, 1.0, 0.0, 0.0)

    # --- deck nodes/elements ---
    anchors = cable_anchor_xs(p)
    xs = list(np.arange(X_P31, X_P35, DECK_NODE_SPACING)) + [X_P35]
    xs = sorted(set(round(x, 3) for x in xs) | {X_P31, X_P32, X_P33, X_PYLON, X_P35}
                | {round(x, 3) for x in SENSOR_GIRDER_XS if x <= X_P35}
                | {x for xs_ in anchors.values() for x in xs_})
    # Lumped mass by tributary length (half of each adjacent segment): the grid is
    # non-uniform once sensor nodes are inserted, so a flat DECK_NODE_SPACING per node
    # over-counted total deck mass by ~15% (61 nodes x 10m vs 530.8m of deck).
    trib = np.diff(xs, prepend=xs[0]) / 2 + np.diff(xs, append=xs[-1]) / 2
    deck_nodes = {}
    for x, L in zip(xs, trib):
        nt = next_tag()
        ops.node(nt, x, 0.0, DECK_Z)
        ops.mass(nt, *(3 * [p['deck_mass_per_len'] * L]), 0, 0, 0)
        deck_nodes[x] = nt

    girder_elements = []
    sorted_xs = sorted(deck_nodes.keys())
    for x0, x1 in zip(sorted_xs[:-1], sorted_xs[1:]):
        gm = (_near_multiplier((x0 + x1) / 2, "girder", specs)
              * zm.get(zone_of((x0 + x1) / 2, GIRDER_ZONES), 1.0))
        et = next_tag()
        ops.element('elasticBeamColumn', et, deck_nodes[x0], deck_nodes[x1],
                    p['deck_A'] * gm, p['deck_E'], p['deck_E'] / 2.4, p['deck_J'],
                    p['deck_Iy'] * gm, p['deck_Iz'] * gm, transf_deck)
        girder_elements.append((et, x0, x1))

    # --- piers: P31, P32, P33, P35 ---
    pier_bottom, pier_top, pier_elements, bearing_elements = {}, {}, {}, {}
    for name, x in _pier_locs().items():
        b = next_tag()
        ops.node(b, x, 0.0, -p['pier_height'])
        ops.fix(b, 1, 1, 1, 1, 1, 1)
        t_top = next_tag()
        ops.node(t_top, x, 0.0, DECK_Z)  # coincident with deck node; bearing is the zeroLength link
        pier_bottom[name] = b
        pier_top[name] = t_top

        pm = _pier_multiplier(name, specs) * zm.get(f"P-{name}", 1.0)
        pe = next_tag()
        ops.element('elasticBeamColumn', pe, b, t_top,
                    p['pier_A'] * pm, p['pier_E'], p['pier_E'] / 2.4, p['pier_J'],
                    p['pier_Iy'] * pm, p['pier_Iz'] * pm, transf_vert)
        pier_elements[name] = pe

        bm = _bearing_multiplier(name, specs) * zm.get(f"B-{name}", 1.0)
        be = _add_bearing(next_tag, t_top, deck_nodes[x], p, name, mult=bm)
        bearing_elements[name] = be

    # --- pylon: concrete leg below deck, steel above, cable anchors in the top zone ---
    top, zone = p['pylon_top_height'], p['pylon_anchor_zone']
    anchor_zs = list(np.linspace(top - zone, top, N_CABLES_PER_SIDE))
    zs = sorted({round(float(z), 3) for z in
                 [-p['pylon_base_depth'], -p['pylon_base_depth'] / 2, 0.0]
                 + list(np.linspace(0.0, top - zone, 4)) + anchor_zs})
    ztrib = np.diff(zs, prepend=zs[0]) / 2 + np.diff(zs, append=zs[-1]) / 2
    pylon_nodes, pylon_elements = {}, []
    for z, L in zip(zs, ztrib):
        nt = next_tag()
        ops.node(nt, X_PYLON, 0.0, z)
        # split a node's tributary mass between leg and tower at z=0
        below = min(L, max(0.0, -z + L / 2)) if z <= 0 else 0.0
        m = p['pylon_leg_mass_per_len'] * below + p['pylon_mass_per_len'] * (L - below)
        ops.mass(nt, *(3 * [m]), 0, 0, 0)
        pylon_nodes[z] = nt
    pylon_base, pylon_top = pylon_nodes[zs[0]], pylon_nodes[zs[-1]]
    ops.fix(pylon_base, 1, 1, 1, 1, 1, 1)
    for z0, z1 in zip(zs[:-1], zs[1:]):
        et = next_tag()
        if z1 <= 0:
            ops.element('elasticBeamColumn', et, pylon_nodes[z0], pylon_nodes[z1],
                        p['pylon_leg_A'], p['pylon_leg_E'], p['pylon_leg_E'] / 2.4, p['pylon_leg_J'],
                        p['pylon_leg_I'], p['pylon_leg_I'], transf_vert)
        else:
            ops.element('elasticBeamColumn', et, pylon_nodes[z0], pylon_nodes[z1],
                        p['pylon_A'], p['pylon_E'], p['pylon_E'] / 2.6, p['pylon_J'],
                        p['pylon_Iy'], p['pylon_Iz'], transf_vert)
        pylon_elements.append(et)

    # deck-pylon laminated rubber bearing (see pylon_bearing_* in DEFAULT_PARAMS)
    pb = dict(p, bearing_k_long=p['pylon_bearing_k_long'], bearing_k_lat=p['pylon_bearing_k_lat'])
    bearing_elements["PYLON"] = _add_bearing(next_tag, pylon_nodes[0.0], deck_nodes[X_PYLON], pb,
                                             "PYLON", mult=_bearing_multiplier("PYLON", specs) * zm.get("B-PYLON", 1.0))

    # --- cables: harp pairing, nearest deck anchor <-> lowest pylon anchor ---
    cable_mat = next_tag()
    ops.uniaxialMaterial('Elastic', cable_mat, p['cable_E'])
    cable_elements = []
    for side, axs in anchors.items():
        by_dist = sorted(axs, key=lambda x: abs(x - X_PYLON))
        for ax, az in zip(by_dist, anchor_zs):
            cm = _near_multiplier(ax, "cable", specs) * zm.get(zone_of(ax, CABLE_ZONES), 1.0)
            et = next_tag()
            ops.element('corotTruss', et, pylon_nodes[round(float(az), 3)], deck_nodes[ax],
                        p['cable_A'] * cm, cable_mat)
            cable_elements.append((et, ax))

    return dict(
        params=p, damage_case=damage_case, deck_nodes=deck_nodes,
        girder_elements=girder_elements,
        pier_bottom=pier_bottom, pier_top=pier_top, pier_elements=pier_elements,
        bearing_elements=bearing_elements, pylon_base=pylon_base, pylon_top=pylon_top,
        pylon_elements=pylon_elements, cable_elements=cable_elements,
    )


def sensor_output_nodes(model):
    """Maps each structural sensor_id in SENSOR_LOCATIONS to the OpenSees
    node tag to record output from, for the given build_model() result.
    Girder sensors snap to the nearest deck node (exact match for all but
    sensor 22 at x=533.0, just past our X_P35=530.8 deck end -- see
    SENSOR_LOCATIONS docstring). Pier sensors map to that pier's top node
    (deck-level, where the bearing/pier meet -- matches where real bridge
    accelerometers are mounted). Ground (free-field) sensors are omitted:
    they have no corresponding FE node."""
    out = {}
    for sid, loc in SENSOR_LOCATIONS.items():
        if loc["kind"] == "girder":
            nearest_x = min(model["deck_nodes"].keys(), key=lambda k: abs(k - loc["x"]))
            out[sid] = model["deck_nodes"][nearest_x]
        elif loc["kind"] == "pier" and loc["pier"] == "PYLON":
            # Sensors 27/28 sit at the pylon location (x=246.9), which isn't
            # in pier_top -- read off the deck node above the pylon bearing,
            # matching the deck-level mounting of the other pier sensors.
            out[sid] = model["deck_nodes"][X_PYLON]
        elif loc["kind"] == "pier":
            out[sid] = model["pier_top"][loc["pier"]]
    return out


def _add_bearing(next_tag, node_i, node_j, p, name, mult=1.0):
    mats = []
    dirs = [1, 2, 3, 4, 5, 6]
    stiffs = [p['bearing_k_long'] * mult, p['bearing_k_lat'] * mult, p['bearing_k_vert'] * mult,
              p['bearing_k_rot'] * mult, p['bearing_k_rot'] * mult, p['bearing_k_rot'] * mult]
    mat_tags = []
    for d, k in zip(dirs, stiffs):
        mt = 100000 + node_i * 10 + d
        ops.uniaxialMaterial('Elastic', mt, k)
        mat_tags.append(mt)
    et = next_tag()
    ops.element('zeroLength', et, node_i, node_j, '-mat', *mat_tags, '-dir', *dirs)
    return et


def get_frequencies(num_modes=8):
    eigvals = ops.eigen(num_modes)
    freqs = [np.sqrt(max(ev, 0.0)) / (2 * np.pi) for ev in eigvals]
    return freqs


def classify_modes(model, num_modes=8):
    """Classify each mode by dominant displacement direction (vertical Z,
    longitudinal X, lateral Y) using modal displacements at deck nodes only --
    the measured modes come from deck sensors, so pylon/pier motion must not vote."""
    freqs = get_frequencies(num_modes)
    classified = []
    for i in range(1, num_modes + 1):
        sums = sum(np.abs(ops.nodeEigenvector(t, i)[:3]) for t in model["deck_nodes"].values())
        direction = ['longitudinal', 'lateral', 'vertical'][int(np.argmax(sums))]
        classified.append(dict(mode=i, freq_hz=freqs[i - 1], direction=direction))
    return classified
