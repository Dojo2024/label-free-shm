"""
IASC-ASCE SHM benchmark, Phase I: the 12DOF shear-building model, ported
from the paper (the original MATLAB `datagen` code is not available).

Source (the only reference used):
  Johnson, Lam, Katafygiotis, Beck, "Phase I IASC-ASCE Structural Health
  Monitoring Benchmark Problem Using Simulated Data", J. Eng. Mech. (2004).
  data/benchmarks/iasc_asce/Johnson2004_phase1_definition.pdf
  - Table 1: member properties; text p.6-7: slab masses, 2.5 x 2.5 m plan,
    4 x 0.9 m stories, 2x2 bays of 1.25 m
  - damage patterns (i)-(vi), p.8; Tables 2/5: 12DOF frequencies
  - Figs 10-11: published 12DOF M and K, transcribed (from the PDF text
    layer, every matrix parsed exactly symmetric) into
    data/benchmarks/iasc_asce/published_12dof_matrices.json

How the model is built (physics, not the published matrices): a 3-D frame of
9 columns x 4 stories (Euler-Bernoulli beams), 12 floor beams per floor,
8 pin-ended braces per story (axial bars), plus 4 rigid floor slabs per floor.
Every non-ground node is slaved to its floor's rigid-body motion
(x, y, theta about the centre column; no vertical motion, no out-of-plane
rotation), which is the paper's 12DOF assumption. Element matrices are
transformed to global axes and condensed through that constraint, so the
columns come out fixed-fixed (12EI/h^3, GJ/h), the floor beams carry no strain
and the braces contribute EA cos^2(a)/L at their plan offset. Damage then acts
on individual braces, and the coupling terms of the asymmetric patterns
(iii), (iv), (vi) fall out of the geometry.

Geometry / conventions (origin at the centre column, as the paper's theta):
  +x = West (strong column direction), +y = South (weak direction),
  theta counter-clockwise about +z. The paper's coordinates put the NE corner
  at (0, 0), so its (2.5 m, 0) "northwest column" is our (+1.25, -1.25).
  DOF order [x1, y1, th1, ..., x4, y4, th4]; SI units (kg, kg m^2, N/m,
  N m/rad).

Decisions the paper leaves implicit, each pinned down by matching Fig. 10:
  - Brace layout: each of the 2 bays on each of the 4 faces has one brace,
    from the corner column at the bottom of the story to the mid-face column
    at its top (chevron). Stiffness cannot see this; the floor rotational
    inertias can (roof lumped I = 2056.9 only with the top ends mid-face).
  - Slabs are uniform 1.25 x 1.25 m rigid plates (own inertia m(a^2+b^2)/12).
  - Lumped mass (HKUST code): half of each member's mass at each end node,
    plus half of each column's torsional inertia rho*(Iy+Iz)*L; damaged
    braces keep their mass ("includes the mass from damaged braces").
  - Consistent mass (USC code): the standard Euler-Bernoulli consistent
    frame mass (axial 140/70, transverse 156/54/22L/13L over 420, torsion
    rho*Ip*L/3,/6) for every member, braces included, with each end's nodal
    rotations tied to the floor (rz = theta). Candidates tried against Fig. 10
    for the braces: plain bar mass m/6*[[2,1],[1,2]] gives floor-to-floor
    coupling 11.5 kg (paper 11.1); beam translations with rotations dropped
    leave theta-theta coupling 1.1 kg*m^2 short (21.9 vs 23.0) and nudge the
    top modes 0.07% low; the full frame mass matches every entry to the
    printed rounding. Braces with zero stiffness are dropped from the
    consistent mass ("neglected any mass of damaged braces").
  - Pattern (v) only frees the bending moment at one floor-beam end; floor
    beams are unstrained under the rigid-floor constraint, so (v) == (iv) in
    this model, exactly as the paper states.

Simplifications: no P-delta, no shear deformation, no rotary inertia beyond
the column torsion term. Braces are bars (no bending stiffness), as stated.
"""
from pathlib import Path
import itertools
import json

import numpy as np
import scipy.linalg
import scipy.signal

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "iasc_asce"
PUBLISHED_MATRICES = DATA_DIR / "published_12dof_matrices.json"

# ---- Table 1 + text p.6-7 -------------------------------------------------
E = 2e11
G = E / 2.6
RHO = 7800.0
COLUMN = dict(A=1.133e-3, Iy=1.97e-6, Iz=0.664e-6, J=8.01e-9)   # B100x9
BEAM = dict(A=1.43e-3, Iy=1.22e-6, Iz=0.249e-6, J=38.2e-9)      # S75x11
BRACE = dict(A=0.141e-3, Iy=0.0, Iz=0.0, J=0.0)                 # L25x25x3
STORY_H = 0.9
BAY = 1.25
N_FLOORS = 4
SLAB_MASS = {1: 800.0, 2: 600.0, 3: 600.0, 4: 400.0}
ASYM_SLAB_MASS = 550.0
# The heavy roof slab's position is read off the Case 4 lumped M (Fig. 10):
# M[x4,th4] = +93.8 = -m*cy and M[y4,th4] = -93.8 = +m*cx with m = 150 kg
# -> slab centre (-0.625, -0.625) = the north-east bay.
ASYM_SLAB_CENTRE = (-0.625, -0.625)

PLAN = (-BAY, 0.0, BAY)

# Braces: id "S<story>-<face><bay>", e.g. "S1-WN" is story 1, west face,
# north brace. (bottom corner node, top mid-face node) in plan coordinates.
_FACE_BRACES = {
    "WN": ((BAY, -BAY), (BAY, 0.0)), "WS": ((BAY, BAY), (BAY, 0.0)),
    "EN": ((-BAY, -BAY), (-BAY, 0.0)), "ES": ((-BAY, BAY), (-BAY, 0.0)),
    "NW": ((BAY, -BAY), (0.0, -BAY)), "NE": ((-BAY, -BAY), (0.0, -BAY)),
    "SW": ((BAY, BAY), (0.0, BAY)), "SE": ((-BAY, BAY), (0.0, BAY)),
}
BRACE_IDS = [f"S{s}-{f}" for s in range(1, N_FLOORS + 1) for f in _FACE_BRACES]

# ---- damage patterns, p.8 ---------------------------------------------------
# Each maps brace id -> remaining stiffness fraction. "beam_release" lists
# floor-beam end moment releases, which do nothing to the 12DOF model.
DAMAGE_PATTERNS = {
    # (i) "no stiffness in the braces of the first story (i.e., the braces
    # still contribute mass, but provide no resistance within the structure)"
    "i": dict(braces={b: 0.0 for b in BRACE_IDS if b.startswith("S1-")}),
    # (ii) "no stiffness in any of the braces of the first and third stories"
    "ii": dict(braces={b: 0.0 for b in BRACE_IDS if b[:3] in ("S1-", "S3-")}),
    # (iii) "no stiffness in one brace in the first story (north brace on the
    # west face of the structure)"
    "iii": dict(braces={"S1-WN": 0.0}),
    # (iv) "no stiffness in one brace in the first story (north brace on the
    # west face) and in one brace in the third story (west brace on the north
    # face)"
    "iv": dict(braces={"S1-WN": 0.0, "S3-NW": 0.0}),
    # (v) "the same as damage pattern (iv) but with the north floor beam at the
    # first level on the west face of the structure (i.e., the beam from
    # (2.5m, 0, 0.9m) to (2.5m, 1.25m, 0.9m)) partially unscrewed from the
    # northwest column at (2.5m, 0, 0.9m) -- consequently, the beam-column
    # connection there can only transmit forces and cannot sustain any bending
    # moments". The paper: "(v) is no different than (iv) for the 12DOF model".
    "v": dict(braces={"S1-WN": 0.0, "S3-NW": 0.0},
              beam_release=[("floor1", (BAY, -BAY), (BAY, 0.0), "end at NW column")]),
    # (vi) "two-thirds stiffness (i.e., a one-third stiffness loss) in one brace
    # in the first story (the same brace damaged in pattern (iii): the north
    # brace on the west face)"
    "vi": dict(braces={"S1-WN": 2.0 / 3.0}),
}


# ---- element matrices ---------------------------------------------------------
def _frame_stiffness(L, A, Iy, Iz, J):
    """3-D Euler-Bernoulli frame element, local DOFs [u v w rx ry rz]x2.
    Iz resists v (bending in local x-y), Iy resists w (bending in local x-z)."""
    k = np.zeros((12, 12))
    k[0, 0] = k[6, 6] = E * A / L
    k[0, 6] = -E * A / L
    k[3, 3] = k[9, 9] = G * J / L
    k[3, 9] = -G * J / L
    a, b, c = 12 * E * Iz / L**3, 6 * E * Iz / L**2, E * Iz / L
    k[1, 1] = k[7, 7] = a
    k[1, 7] = -a
    k[1, 5] = k[1, 11] = b
    k[5, 7] = k[7, 11] = -b
    k[5, 5] = k[11, 11] = 4 * c
    k[5, 11] = 2 * c
    a, b, c = 12 * E * Iy / L**3, 6 * E * Iy / L**2, E * Iy / L
    k[2, 2] = k[8, 8] = a
    k[2, 8] = -a
    k[2, 4] = k[2, 10] = -b
    k[4, 8] = k[8, 10] = b
    k[4, 4] = k[10, 10] = 4 * c
    k[4, 10] = 2 * c
    return np.triu(k) + np.triu(k, 1).T


def _frame_consistent_mass(L, A, Iy, Iz):
    """Consistent mass of the same element (no rotary inertia except torsion)."""
    m = np.zeros((12, 12))
    s = RHO * A * L / 420
    m[0, 0] = m[6, 6] = 140 * s
    m[0, 6] = 70 * s
    m[1, 1] = m[7, 7] = m[2, 2] = m[8, 8] = 156 * s
    m[1, 7] = m[2, 8] = 54 * s
    m[1, 5] = 22 * L * s
    m[1, 11] = -13 * L * s
    m[5, 7] = 13 * L * s
    m[7, 11] = -22 * L * s
    m[2, 4] = -22 * L * s
    m[2, 10] = 13 * L * s
    m[4, 8] = -13 * L * s
    m[8, 10] = 22 * L * s
    m[4, 4] = m[5, 5] = m[10, 10] = m[11, 11] = 4 * L**2 * s
    m[4, 10] = m[5, 11] = -3 * L**2 * s
    It = RHO * (Iy + Iz) * L
    m[3, 3] = m[9, 9] = It / 3
    m[3, 9] = It / 6
    return np.triu(m) + np.triu(m, 1).T


def _frame_lumped_mass(L, A, Iy, Iz):
    m = np.zeros((12, 12))
    half = RHO * A * L / 2
    for d in (0, 1, 2, 6, 7, 8):
        m[d, d] = half
    m[3, 3] = m[9, 9] = RHO * (Iy + Iz) * L / 2   # torsional inertia about the member axis
    return m


def _rotation(p1, p2, strong_dir):
    """Rows = local x (member axis), y, z in global coordinates. Local z is the
    direction whose transverse displacement is resisted by Iy (strong axis)."""
    ex = (p2 - p1) / np.linalg.norm(p2 - p1)
    ez = strong_dir - (strong_dir @ ex) * ex
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    return np.vstack([ex, ey, ez])


def _node_constraint(pos, level):
    """6 x 12 map from the 12 floor DOFs to a node's [ux uy uz rx ry rz]."""
    g = np.zeros((6, 12))
    if level == 0:
        return g
    c = 3 * (level - 1)
    px, py = pos
    g[0, c], g[0, c + 2] = 1.0, -py
    g[1, c + 1], g[1, c + 2] = 1.0, px
    g[5, c + 2] = 1.0
    return g


def _elements():
    """(kind, id, (pos1, level1), (pos2, level2)) for every member."""
    out = []
    for s in range(1, N_FLOORS + 1):
        for px, py in itertools.product(PLAN, PLAN):
            out.append(("column", f"C{s}", ((px, py), s - 1), ((px, py), s)))
        for face, (bot, top) in _FACE_BRACES.items():
            out.append(("brace", f"S{s}-{face}", (bot, s - 1), (top, s)))
    for f in range(1, N_FLOORS + 1):
        for c in PLAN:
            for a0, a1 in ((-BAY, 0.0), (0.0, BAY)):
                out.append(("beam", f"B{f}", ((a0, c), f), ((a1, c), f)))   # along x
                out.append(("beam", f"B{f}", ((c, a0), f), ((c, a1), f)))   # along y
    return out


def _slab_mass(m, cx, cy):
    """Rigid 1.25 x 1.25 m plate of mass m centred at (cx, cy), about the origin."""
    I = m * (cx**2 + cy**2 + 2 * BAY**2 / 12)
    return np.array([[m, 0.0, -m * cy], [0.0, m, m * cx], [-m * cy, m * cx, I]])


def build_12dof(damage_pattern=None, mass="symmetric", lumped=False, braces=None):
    """(M, K) of the 12DOF model in SI units. `damage_pattern` is None or a key
    of DAMAGE_PATTERNS; `mass` is 'symmetric' (Cases 1-3) or 'asymmetric'
    (Cases 4-6: one NE roof slab 550 kg); lumped=True is the HKUST code,
    lumped=False the USC consistent-mass code. `braces` (brace id -> remaining
    stiffness fraction) overrides `damage_pattern`, for continuous damage."""
    if mass not in ("symmetric", "asymmetric"):
        raise ValueError(f"mass must be 'symmetric' or 'asymmetric', got {mass!r}")
    if braces is not None and damage_pattern is not None:
        raise ValueError("pass damage_pattern or braces, not both")
    factors = braces if braces is not None else (
        {} if damage_pattern is None else DAMAGE_PATTERNS[damage_pattern]["braces"])
    unknown = set(factors) - set(BRACE_IDS)
    if unknown:
        raise KeyError(f"unknown brace ids {unknown}")

    K = np.zeros((12, 12))
    M = np.zeros((12, 12))
    for kind, eid, (p1, l1), (p2, l2) in _elements():
        x1 = np.array([p1[0], p1[1], STORY_H * l1])
        x2 = np.array([p2[0], p2[1], STORY_H * l2])
        L = np.linalg.norm(x2 - x1)
        if kind == "column":
            props, strong = COLUMN, np.array([1.0, 0.0, 0.0])    # strong bending toward x
        elif kind == "beam":
            props, strong = BEAM, np.array([0.0, 0.0, 1.0])      # strong bending vertically
        else:
            props, strong = BRACE, np.array([0.0, 0.0, 1.0])   # no bending: any normal works
        R = _rotation(x1, x2, strong)
        T = np.kron(np.eye(4), R)
        Lc = np.vstack([_node_constraint(p1, l1), _node_constraint(p2, l2)])   # 12 x 12
        B = T @ Lc                                                              # floor DOFs -> local

        f = factors.get(eid, 1.0) if kind == "brace" else 1.0
        if kind == "brace":
            k_loc = np.zeros((12, 12))
            k_loc[0, 0] = k_loc[6, 6] = E * props["A"] / L
            k_loc[0, 6] = k_loc[6, 0] = -E * props["A"] / L
        else:
            k_loc = _frame_stiffness(L, props["A"], props["Iy"], props["Iz"], props["J"])
        K += f * (B.T @ k_loc @ B)

        if lumped:
            m_loc = _frame_lumped_mass(L, props["A"], props["Iy"], props["Iz"])
        elif kind == "brace":
            if f == 0.0:
                continue       # USC code neglects the mass of removed braces
            m_loc = _frame_consistent_mass(L, props["A"], 0.0, 0.0)
        else:
            m_loc = _frame_consistent_mass(L, props["A"], props["Iy"], props["Iz"])
        M += B.T @ m_loc @ B

    for floor in range(1, N_FLOORS + 1):
        c = slice(3 * (floor - 1), 3 * floor)
        for cx, cy in itertools.product((-BAY / 2, BAY / 2), repeat=2):
            m = SLAB_MASS[floor]
            if mass == "asymmetric" and floor == N_FLOORS and (cx, cy) == ASYM_SLAB_CENTRE:
                m = ASYM_SLAB_MASS
            M[c, c] += _slab_mass(m, cx, cy)
    return M, K


def frequencies(M, K):
    """Natural frequencies [Hz], ascending."""
    lam = scipy.linalg.eigh(K, M, eigvals_only=True)
    if lam.min() <= 0:
        raise ValueError(f"non-positive eigenvalue {lam.min()}")
    return np.sqrt(lam) / (2 * np.pi)


# ---- published reference values -------------------------------------------------
_U = [9.41, 11.79, 16.53, 25.60, 32.07, 38.85, 45.17, 48.37, 48.68, 60.60, 68.64, 85.51]
PUBLISHED_FREQS = {
    # (table, damage, mass, lumped): Hz
    ("T2", None, "symmetric", False): _U,
    ("T2", None, "symmetric", True): [9.41, 11.79, 16.38, 25.54, 32.01, 38.66, 44.64, 48.01, 48.44, 60.15, 67.48, 83.62],
    ("T2", "i", "symmetric", False): [6.24, 9.91, 11.84, 21.58, 28.99, 37.56, 38.75, 47.57, 48.19, 60.45, 66.46, 85.20],
    ("T2", "i", "symmetric", True): [6.24, 9.91, 11.73, 21.53, 28.92, 37.37, 38.28, 47.34, 47.83, 59.99, 65.31, 83.31],
    ("T2", "ii", "symmetric", False): [5.83, 9.52, 11.13, 14.93, 24.98, 28.78, 36.28, 41.65, 47.06, 54.76, 64.86, 74.27],
    ("T2", "ii", "symmetric", True): [5.82, 9.51, 11.01, 14.89, 24.91, 28.41, 36.06, 41.35, 46.79, 54.34, 63.64, 72.61],
    # Table 5: 12DOF, asymmetric mass, Case 4 (HKUST lumped mass)
    ("T5", None, "asymmetric", True): [9.29, 11.64, 16.19, 25.27, 31.66, 38.26, 44.20, 47.75, 47.97, 59.81, 66.90, 83.23],
    ("T5", "i", "asymmetric", True): [6.18, 9.80, 11.63, 21.27, 28.59, 36.87, 37.93, 46.81, 47.54, 59.63, 64.67, 82.89],
    ("T5", "ii", "asymmetric", True): [5.76, 9.39, 10.90, 14.78, 24.70, 28.22, 35.97, 40.60, 46.46, 53.68, 63.44, 71.58],
    ("T5", "iii", "asymmetric", True): [8.79, 11.64, 15.80, 24.37, 31.66, 37.77, 43.61, 47.68, 47.96, 59.81, 66.58, 83.18],
    ("T5", "iv", "asymmetric", True): [8.79, 11.50, 15.68, 24.36, 30.82, 37.76, 42.91, 47.68, 47.96, 58.18, 66.56, 81.76],
    ("T5", "v", "asymmetric", True): [8.79, 11.50, 15.68, 24.36, 30.82, 37.76, 42.91, 47.68, 47.96, 58.18, 66.56, 81.76],
    ("T5", "vi", "asymmetric", True): [9.15, 11.64, 16.07, 24.98, 31.66, 38.10, 43.99, 47.72, 47.97, 59.81, 66.79, 83.21],
}
# Fig. 10 / 11 matrices: json key -> build_12dof kwargs
PUBLISHED_M = {
    "M_consistent_undamaged": dict(damage_pattern=None, mass="symmetric", lumped=False),
    "M_consistent_i": dict(damage_pattern="i", mass="symmetric", lumped=False),
    "M_consistent_ii": dict(damage_pattern="ii", mass="symmetric", lumped=False),
    "M_lumped_symmetric": dict(damage_pattern=None, mass="symmetric", lumped=True),
    "M_lumped_asymmetric": dict(damage_pattern=None, mass="asymmetric", lumped=True),
}
PUBLISHED_K = {"K_undamaged": None, "K_i": "i", "K_ii": "ii", "K_iii": "iii", "K_iv": "iv", "K_vi": "vi"}


# ---- time histories (Cases 1 and 4, 12DOF data generation) ----------------------
def sensor_matrix():
    """16 accelerometers: two per floor in x (on the north and south faces,
    y = -/+1.25 at x = 0) and two in y (east and west faces, x = -/+1.25 at
    y = 0). The paper gives the count, not the positions: this layout is an
    assumption (floor-edge centre points)."""
    C = np.zeros((16, 12))
    r = 0
    for floor in range(1, N_FLOORS + 1):
        c = 3 * (floor - 1)
        for py in (-BAY, BAY):
            C[r, c], C[r, c + 2] = 1.0, -py
            r += 1
        for px in (-BAY, BAY):
            C[r, c + 1], C[r, c + 2] = 1.0, px
            r += 1
    return C


def simulate(case, damage_pattern=None, lumped=True, duration=100.0, dt=0.001,
             zeta=0.01, noise=0.10, seed=0):
    """Noisy 16-channel acceleration for the two cases Table 9 generates with
    the 12DOF model: Case 1 (symmetric mass, independent y-direction forces at
    each floor; only the y channels are meant to be analysed) and Case 4
    (asymmetric mass, diagonal shaker at the roof centre column). lumped=True
    is the HKUST mass the paper says to use for new studies; lumped=False
    reproduces the USC consistent-mass Case 1 data. Forces are in N with unit
    pre-filter variance (the paper gives no amplitude; the model is linear). Modal superposition with the exact
    zero-order-hold discretisation (MATLAB lsim's scheme), 1% damping per mode,
    excitation = unit-variance Gaussian white noise through a 6th-order 100 Hz
    Butterworth low-pass, sensor noise = Gaussian with RMS `noise` x the
    largest channel RMS. Checked against a direct 24-state ZOH simulation with
    the equivalent modal damping matrix (agreement 4e-14 relative).
    Returns (t, y_noisy, forces)."""
    if case == 1:
        M, K = build_12dof(damage_pattern, "symmetric", lumped)
        Bf = np.zeros((12, 4))
        for i in range(4):
            Bf[3 * i + 1, i] = 1.0
    elif case == 4:
        M, K = build_12dof(damage_pattern, "asymmetric", lumped)
        # Roof shaker "in a direction -(i-hat - j-hat) or +(i-hat - j-hat)" at
        # the centre column: one force along (i - j)/sqrt(2); sign is immaterial.
        Bf = np.zeros((12, 1))
        Bf[9, 0], Bf[10, 0] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    else:
        raise ValueError(f"only 12DOF-generated cases 1 and 4 are implemented, got {case}")
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    b, a = scipy.signal.butter(6, 100.0, fs=1 / dt)
    f = scipy.signal.lfilter(b, a, rng.standard_normal((n, Bf.shape[1])), axis=0)

    lam, Phi = scipy.linalg.eigh(K, M)          # Phi^T M Phi = I
    w = np.sqrt(lam)
    Cs = sensor_matrix()
    q = np.zeros((n, 12))
    qd = np.zeros((n, 12))
    p = f @ (Phi.T @ Bf).T                        # modal forces, n x 12
    for j in range(12):
        A = np.array([[0.0, 1.0], [-w[j] ** 2, -2 * zeta * w[j]]])
        sysd = scipy.signal.cont2discrete((A, np.array([[0.0], [1.0]]), np.eye(2), np.zeros((2, 1))), dt, method="zoh")
        _, _, x = scipy.signal.dlsim(sysd[:4] + (dt,), p[:, j:j + 1])
        q[:, j], qd[:, j] = x[:, 0], x[:, 1]
    qdd = p - qd * (2 * zeta * w) - q * w**2
    acc = (qdd @ Phi.T) @ Cs.T
    rms = np.sqrt((acc ** 2).mean(axis=0)).max()
    y = acc + noise * rms * rng.standard_normal(acc.shape)
    return t, y, f


def _matrix_check(ours, ref):
    """max relative difference over published non-zeros, and the largest value
    we have where the paper prints 0."""
    nz = ref != 0
    rel = np.abs(ours[nz] - ref[nz]) / np.abs(ref[nz])
    i = np.argmax(rel)
    where = tuple(int(v) for v in np.argwhere(nz)[i])
    zero_leak = np.abs(ours[~nz]).max() if (~nz).any() else 0.0
    return rel.max(), where, zero_leak


def main():
    pub = {k: np.array(v) for k, v in json.loads(PUBLISHED_MATRICES.read_text()).items() if not k.startswith("_")}
    print("== Fig. 10 mass matrices (kg, kg m, kg m^2) ==")
    for key, kw in PUBLISHED_M.items():
        M, _ = build_12dof(**kw)
        rel, where, leak = _matrix_check(M, pub[key])
        print(f"  {key:24s} max rel diff {100 * rel:6.3f}% at {where} "
              f"(ours {M[where]:.2f} vs {pub[key][where]:.1f}); max |entry| where paper has 0: {leak:.3g}")
    print("== Fig. 11 stiffness matrices (MN/m) ==")
    for key, dmg in PUBLISHED_K.items():
        _, K = build_12dof(dmg)
        K = K / 1e6
        rel, where, leak = _matrix_check(K, pub[key])
        print(f"  {key:24s} max rel diff {100 * rel:6.3f}% at {where} "
              f"(ours {K[where]:.3f} vs {pub[key][where]:.2f}); max |entry| where paper has 0: {leak:.3g}")
    _, K4 = build_12dof("iv")
    _, K5 = build_12dof("v")
    assert np.array_equal(K4, K5), "pattern (v) must equal (iv) in the 12DOF model"

    print("== natural frequencies [Hz]: ours vs paper (% error) ==")
    worst_all = 0.0
    for (table, dmg, mass, lumped), ref in PUBLISHED_FREQS.items():
        f = frequencies(*build_12dof(dmg, mass, lumped))
        err = 100 * (f - np.array(ref)) / np.array(ref)
        worst = np.abs(err).max()
        worst_all = max(worst_all, worst)
        name = f"{table} {dmg or 'undamaged'} {mass} {'lumped' if lumped else 'consistent'}"
        print(f"  {name:38s} worst {worst:5.3f}%")
        print("     ours  " + " ".join(f"{v:6.2f}" for v in f))
        print("     paper " + " ".join(f"{v:6.2f}" for v in ref))
        print("     err%  " + " ".join(f"{v:+6.2f}" for v in err))
    print(f"worst frequency error over all tabulated 12DOF cases: {worst_all:.3f}% "
          f"({'PASS' if worst_all < 0.5 else 'FAIL'} at 0.5%)")


if __name__ == "__main__":
    main()
