"""KW51 finite element model: StaBIL-equivalent assembly, constraints and modes, in Python.

Takes the model definition from kw51_model.build (a translation of KU Leuven's model_kw51.m) and
does what StaBIL 3.1's getdof / asmkm / tconstr / eigfem do, from the StaBIL manual (no StaBIL code
was used):
  beam    Euler-Bernoulli, cubic deflection; shear deflection with A_s = ky A, kz A when finite.
          Local x from node 1 to node 2, local y towards the reference node (third node). Consistent
          mass of the shear-deformable beam with rotary inertia (StaBIL's default); torsional inertia
          rho (Iyy + Izz).
  shell4  flat shell: membrane = bilinear + two incompatible modes (1 - xi^2, 1 - eta^2; Taylor's
          constant-Jacobian correction), condensed; bending = four overlaid DKT triangles (each diagonal
          split once, so each weighs 1/2). Mass: consistent bilinear for u, v; for w and the rotations,
          the cubic w interpolation of each DKT triangle (3 interior points), also halved. Drilling
          rotations get 1/1000 of the smallest non-zero diagonal stiffness (StaBIL's choice, after
          Bathe). Formulations checked against the StaBIL 3.1 source (13 Oct), reimplemented here.
  tconstr Constant = CoefS u_slave + sum CoefMk u_masterk, slave eliminated: U = T Ur (Q0 = 0 here).
DOF labels: node + 0.01 .. 0.06 = ux uy uz rx ry rz.

Derived from Maes, van de Velde, Lombaert (2025), Zenodo 10.5281/zenodo.17366772 (CC BY-NC-SA 4.0),
through kw51_model.py; this file is therefore CC BY-NC-SA 4.0 as well.

    .venv/bin/python scripts/kw51_fe.py --test     # element tests + the StaBIL manual's plate example
    .venv/bin/python scripts/kw51_fe.py            # modal_analysis.m: the updated model before retrofit
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------- beam
def beam_local(L, A, ky, kz, Ixx, Iyy, Izz, E, nu, rho):
    G = E / (2 * (1 + nu))
    K = np.zeros((12, 12))
    M = np.zeros((12, 12))
    a = [0, 6]
    K[np.ix_(a, a)] = E * A / L * np.array([[1, -1], [-1, 1]])
    t = [3, 9]
    K[np.ix_(t, t)] = G * Ixx / L * np.array([[1, -1], [-1, 1]])
    # x-y plane: v, rz with Izz (shear area ky A); x-z plane: w, ry with Iyy (shear area kz A)
    for idx, I, k, s in (([1, 5, 7, 11], Izz, ky, 1), ([2, 4, 8, 10], Iyy, kz, -1)):
        ph = 0.0 if not np.isfinite(k) else 12 * E * I / (G * k * A * L ** 2)
        c = E * I / ((1 + ph) * L ** 3)
        K[np.ix_(idx, idx)] = c * np.array([[12, s * 6 * L, -12, s * 6 * L],
                                            [s * 6 * L, (4 + ph) * L ** 2, -s * 6 * L, (2 - ph) * L ** 2],
                                            [-12, -s * 6 * L, 12, -s * 6 * L],
                                            [s * 6 * L, (2 - ph) * L ** 2, -s * 6 * L, (4 + ph) * L ** 2]])
        # consistent mass of the shear-deformable beam with rotary inertia (Przemieniecki), r^2 = I / A
        q = (I / A) / L ** 2
        c0 = rho * A * L / (1 + ph) ** 2
        m11 = c0 * (13 / 35 + 7 / 10 * ph + ph ** 2 / 3 + 6 / 5 * q)
        m13 = c0 * (9 / 70 + 3 / 10 * ph + ph ** 2 / 6 - 6 / 5 * q)
        m12 = c0 * (11 / 210 + 11 / 120 * ph + ph ** 2 / 24 + (1 / 10 - ph / 2) * q) * L
        m14 = -c0 * (13 / 420 + 3 / 40 * ph + ph ** 2 / 24 - (1 / 10 - ph / 2) * q) * L
        m22 = c0 * (1 / 105 + ph / 60 + ph ** 2 / 120 + (2 / 15 + ph / 6 + ph ** 2 / 3) * q) * L ** 2
        m24 = -c0 * (1 / 140 + ph / 60 + ph ** 2 / 120 + (1 / 30 + ph / 6 - ph ** 2 / 6) * q) * L ** 2
        M[np.ix_(idx, idx)] = np.array([[m11, s * m12, m13, s * m14],
                                        [s * m12, m22, -s * m14, m24],
                                        [m13, -s * m14, m11, -s * m12],
                                        [s * m14, m24, -s * m12, m22]])
    M[np.ix_(a, a)] = rho * A * L / 6 * np.array([[2, 1], [1, 2]])
    M[np.ix_(t, t)] = rho * (Iyy + Izz) * L / 6 * np.array([[2, 1], [1, 2]])
    return K, M


def beam_global(X1, X2, Xref, sec, mat):
    A, ky, kz, Ixx, Iyy, Izz = sec
    E, nu, rho = mat
    ex = X2 - X1
    L = np.linalg.norm(ex)
    ex = ex / L
    v = Xref - X1
    ey = v - (v @ ex) * ex
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    R = np.vstack([ex, ey, ez])
    T = np.kron(np.eye(4), R)
    K, M = beam_local(L, A, ky, kz, Ixx, Iyy, Izz, E, nu, rho)
    return T.T @ K @ T, T.T @ M @ T


# ---------------------------------------------------------------- shell4 = Q4 membrane + 4 x DKT/2
def dkt(x, y, D):
    """DKT plate bending (Batoz, Bathe, Ho 1980), local DOFs per node (w, rx, ry), rx = w,y, ry = -w,x."""
    x12, x23, x31 = x[0] - x[1], x[1] - x[2], x[2] - x[0]
    y12, y23, y31 = y[0] - y[1], y[1] - y[2], y[2] - y[0]
    A2 = x31 * y12 - x12 * y31  # 2 x area
    xs, ys = (x23, x31, x12), (y23, y31, y12)  # edges k = 4, 5, 6
    l2 = [xs[k] ** 2 + ys[k] ** 2 for k in range(3)]
    P = [-6 * xs[k] / l2[k] for k in range(3)]
    q = [3 * xs[k] * ys[k] / l2[k] for k in range(3)]
    r = [3 * ys[k] ** 2 / l2[k] for k in range(3)]
    t = [-6 * ys[k] / l2[k] for k in range(3)]
    P4, P5, P6 = P
    q4, q5, q6 = q
    r4, r5, r6 = r
    t4, t5, t6 = t
    K = np.zeros((9, 9))
    for xi, eta in ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5)):
        Hx_xi = np.array([P6 * (1 - 2 * xi) + (P5 - P6) * eta, q6 * (1 - 2 * xi) - (q5 + q6) * eta,
                          -4 + 6 * (xi + eta) + r6 * (1 - 2 * xi) - eta * (r5 + r6),
                          -P6 * (1 - 2 * xi) + eta * (P4 + P6), q6 * (1 - 2 * xi) - eta * (q6 - q4),
                          -2 + 6 * xi + r6 * (1 - 2 * xi) + eta * (r4 - r6),
                          -eta * (P5 + P4), eta * (q4 - q5), -eta * (r5 - r4)])
        Hy_xi = np.array([t6 * (1 - 2 * xi) + eta * (t5 - t6), 1 + r6 * (1 - 2 * xi) - eta * (r5 + r6),
                          -q6 * (1 - 2 * xi) + eta * (q5 + q6),
                          -t6 * (1 - 2 * xi) + eta * (t4 + t6), -1 + r6 * (1 - 2 * xi) + eta * (r4 - r6),
                          -q6 * (1 - 2 * xi) - eta * (q4 - q6),
                          -eta * (t4 + t5), eta * (r4 - r5), -eta * (q4 - q5)])
        Hx_eta = np.array([-P5 * (1 - 2 * eta) - xi * (P6 - P5), q5 * (1 - 2 * eta) - xi * (q5 + q6),
                           -4 + 6 * (xi + eta) + r5 * (1 - 2 * eta) - xi * (r5 + r6),
                           xi * (P4 + P6), xi * (q4 - q6), -xi * (r6 - r4),
                           P5 * (1 - 2 * eta) - xi * (P4 + P5), q5 * (1 - 2 * eta) + xi * (q4 - q5),
                           -2 + 6 * eta + r5 * (1 - 2 * eta) + xi * (r4 - r5)])
        Hy_eta = np.array([-t5 * (1 - 2 * eta) - xi * (t6 - t5), 1 + r5 * (1 - 2 * eta) - xi * (r5 + r6),
                           -q5 * (1 - 2 * eta) + xi * (q5 + q6),
                           xi * (t4 + t6), xi * (r4 - r6), -xi * (q4 - q6),
                           t5 * (1 - 2 * eta) - xi * (t4 + t5), -1 + r5 * (1 - 2 * eta) + xi * (r4 - r5),
                           -q5 * (1 - 2 * eta) - xi * (q4 - q5)])
        B = np.vstack([y31 * Hx_xi + y12 * Hx_eta,
                       -x31 * Hy_xi - x12 * Hy_eta,
                       -x31 * Hx_xi - x12 * Hx_eta + y31 * Hy_xi + y12 * Hy_eta]) / A2
        K += A2 / 6 * B.T @ D @ B  # weight 1/6 on the unit triangle, Jacobian A2
    # Batoz's nodal DOFs are (w, theta_x = w,y, theta_y = -w,x) = (w, rx, ry) here: no mapping. Checked
    # 13 Oct by the patch tests (rigid tilts give zero energy, constant curvatures the exact energy)
    return K


def dkt_mass(x, y, rho_h):
    """Mass of a DKT triangle from its cubic w interpolation (DOFs w, rx, ry per node), 3 interior points."""
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    area = (b[1] * c[2] - b[2] * c[1]) / 2
    M = np.zeros((9, 9))
    for Lg in (np.array([4, 1, 1]) / 6, np.array([1, 4, 1]) / 6, np.array([1, 1, 4]) / 6):
        M += np.outer(*(2 * [cubic_w(Lg, b, c)])) * rho_h * area / 3
    return M


def cubic_w(L, b, c):
    """Incomplete-cubic w shape functions in area coordinates (Bazeley et al. 1965), ordered
    (w1, rx1, ry1, w2, ...); reproduces any linear w with rx = w,y, ry = -w,x (checked in --test)."""
    j, k = [1, 2, 0], [2, 0, 1]
    L123 = L[0] * L[1] * L[2] / 2
    N = np.zeros(9)
    N[0::3] = L + L ** 2 * L[j] + L ** 2 * L[k] - L * L[j] ** 2 - L * L[k] ** 2
    N[1::3] = b[j] * (L[k] * L ** 2 + L123) - b[k] * (L ** 2 * L[j] + L123)
    N[2::3] = c[j] * (L[k] * L ** 2 + L123) - c[k] * (L ** 2 * L[j] + L123)
    return N


def shell4_global(X, h, mat):
    E, nu, rho = mat
    ez = np.cross(X[2] - X[0], X[3] - X[1])
    ez /= np.linalg.norm(ez)
    ex = X[1] - X[0]
    ex = ex - (ex @ ez) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    R = np.vstack([ex, ey, ez])
    xl = (X - X.mean(0)) @ R.T
    x, y = xl[:, 0], xl[:, 1]
    K = np.zeros((24, 24))
    M = np.zeros((24, 24))
    # membrane: bilinear + incompatible modes (condensed), 2x2 Gauss; consistent bilinear mass
    Dm = E * h / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    xin, etn = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
    XY = np.column_stack([x, y])
    J0 = np.vstack([xin / 4, etn / 4]) @ XY
    K11, K12, K22 = np.zeros((8, 8)), np.zeros((8, 4)), np.zeros((4, 4))
    Mm = np.zeros((8, 8))
    g = 1 / np.sqrt(3)
    for a in (-g, g):
        for b_ in (-g, g):
            N = (1 + xin * a) * (1 + etn * b_) / 4
            dN = np.vstack([xin * (1 + etn * b_) / 4, etn * (1 + xin * a) / 4])
            J = dN @ XY
            dJ = np.linalg.det(J)
            dNx = np.linalg.solve(J, dN)
            B = np.zeros((3, 8))
            B[0, 0::2], B[1, 1::2] = dNx[0], dNx[1]
            B[2, 0::2], B[2, 1::2] = dNx[1], dNx[0]
            # incompatible modes 1 - xi^2, 1 - eta^2, derivatives through J0 scaled by det(J0)/det(J) (Taylor)
            dNi = np.linalg.solve(J0, np.diag([-2 * a, -2 * b_])) * (np.linalg.det(J0) / dJ)
            Bi = np.zeros((3, 4))
            Bi[0, [0, 2]], Bi[1, [1, 3]] = dNi[0], dNi[1]
            Bi[2, [0, 2]], Bi[2, [1, 3]] = dNi[1], dNi[0]
            K11 += B.T @ Dm @ B * dJ
            K12 += B.T @ Dm @ Bi * dJ
            K22 += Bi.T @ Dm @ Bi * dJ
            Nm = np.zeros((2, 8))
            Nm[0, 0::2], Nm[1, 1::2] = N, N
            Mm += rho * h * Nm.T @ Nm * dJ
    um = np.array([[6 * i, 6 * i + 1] for i in range(4)]).ravel()
    K[np.ix_(um, um)] = K11 - K12 @ np.linalg.solve(K22, K12.T)
    M[np.ix_(um, um)] = Mm
    # bending and its mass: four DKT triangles, each split of the quad counted once -> weight 1/2
    Db = E * h ** 3 / (12 * (1 - nu ** 2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    for tri in ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3)):
        idx = np.array([[6 * n + 2, 6 * n + 3, 6 * n + 4] for n in tri]).ravel()
        K[np.ix_(idx, idx)] += dkt(x[list(tri)], y[list(tri)], Db) / 2
        M[np.ix_(idx, idx)] += dkt_mass(x[list(tri)], y[list(tri)], rho * h) / 2
    dK = np.diag(K)
    for e in range(5, 24, 6):
        K[e, e] = dK[dK != 0].min() / 1000  # drilling
    T = np.kron(np.eye(8), R)
    return T.T @ K @ T, T.T @ M @ T


# ---------------------------------------------------------------- StaBIL-equivalent assembly
def getdof(Elements, Types):
    names = dict(Types)
    out = []
    for e in Elements:
        typ = names[int(e[1])]
        nodes = e[4:6] if typ in ("beam", "truss") else e[4:8]
        nd = 3 if typ == "truss" else 6
        out += [int(n) + 0.01 * (d + 1) for n in nodes for d in range(nd)]
    return np.unique(np.round(np.array(out), 2))


def asmkm(m, DOF):
    names = dict(m["Types"])
    pos = {round(d, 2): i for i, d in enumerate(DOF)}
    X = {int(r[0]): r[1:4] for r in m["Nodes"]}
    sec = {int(r[0]): r[1:] for r in m["Sections"]}
    mat = {int(r[0]): r[1:4] for r in m["Materials"]}
    rows, cols, kv, mv = [], [], [], []
    for e in m["Elements"]:
        typ = names[int(e[1])]
        s, mt = sec[int(e[2])], mat[int(e[3])]
        if typ == "beam":
            n = [int(e[4]), int(e[5])]
            Ke, Me = beam_global(X[n[0]], X[n[1]], X[int(e[6])], s[:6], mt)
        elif typ == "shell4":
            n = [int(v) for v in e[4:8]]
            Ke, Me = shell4_global(np.array([X[v] for v in n]), s[0], mt)
        else:
            raise ValueError(f"element type {typ} not implemented")
        idx = np.array([pos[round(v + 0.01 * (d + 1), 2)] for v in n for d in range(6)])
        rows.append(np.repeat(idx, idx.size)), cols.append(np.tile(idx, idx.size))
        kv.append(Ke.ravel()), mv.append(Me.ravel())
    r, c = np.concatenate(rows), np.concatenate(cols)
    nd = len(DOF)
    K = sp.csr_matrix((np.concatenate(kv), (r, c)), shape=(nd, nd))
    M = sp.csr_matrix((np.concatenate(mv), (r, c)), shape=(nd, nd))
    return K, M


def tconstr(Constr, DOF):
    """T (nDOF x nMaster) with U = T Ur for Constant = CoefS u_s + sum CoefMk u_mk (Constant = 0 here)."""
    Constr = np.atleast_2d(Constr)
    assert np.all(Constr[:, 0] == 0), "non-zero constants (Q0) not needed for KW51"
    slave = np.round(Constr[:, 2], 2)
    pos = {round(d, 2): i for i, d in enumerate(DOF)}
    assert len(set(slave)) == len(slave), "a DOF is slaved twice in one stage"
    is_slave = np.isin(np.round(DOF, 2), slave)
    master = DOF[~is_slave]
    mpos = {round(d, 2): i for i, d in enumerate(master)}
    rows, cols, vals = [], [], []
    for i, d in enumerate(DOF):
        if not is_slave[i]:
            rows.append(i), cols.append(mpos[round(d, 2)]), vals.append(1.0)
    for row in Constr:
        cs, s = row[1], round(row[2], 2)
        assert s in pos, f"slave DOF {s} not in DOF"
        for k in range(3, len(row) - 1, 2):
            cm, md = row[k], row[k + 1]
            if np.isnan(cm) or cm == 0:
                continue
            md = round(md, 2)
            assert md in mpos, f"master DOF {md} of slave {s} is not a master in this stage"
            rows.append(pos[s]), cols.append(mpos[md]), vals.append(-cm / cs)
    T = sp.csr_matrix((vals, (rows, cols)), shape=(len(DOF), len(master)))
    return T, master


def assemble(m):
    DOF = np.concatenate([getdof(m["Elements"], m["Types"]), np.round(np.asarray(m["dof_extra"], float), 2)])
    K, M = asmkm(m, DOF)
    M = M.tolil()
    pos = {round(d, 2): i for i, d in enumerate(DOF)}
    for d, mass in m["box_masses"]:
        M[pos[round(d, 2)], pos[round(d, 2)]] += mass
    M = M.tocsr()
    T, cur = sp.identity(len(DOF), format="csr"), DOF
    for C in m["constraints"]:
        Ti, cur = tconstr(C, cur)
        T = T @ Ti
    Kr, Mr = (T.T @ K @ T).tocsr(), (T.T @ M @ T).tocsr()
    if m["springs"]:
        Kr = Kr.tolil()
        p = {round(d, 2): i for i, d in enumerate(cur)}
        for d, k in m["springs"]:
            Kr[p[round(d, 2)], p[round(d, 2)]] += k
        Kr = Kr.tocsr()
    return dict(K=Kr, M=Mr, T=T, DOF=DOF, MasterDOF=cur)


def modes(K, M, n=50):
    """eigfem: lowest n modes (shift-invert); frequencies in Hz, mass-normalised shapes."""
    # DOFs with neither stiffness nor mass (e.g. drilling rotations left unconstrained) are dropped
    keep = np.where((np.abs(K).sum(1).A.ravel() > 0) | (np.abs(M).sum(1).A.ravel() > 0))[0]
    K, M = K[keep][:, keep], M[keep][:, keep]
    w2, phi = spla.eigsh(K, k=n, M=M, sigma=-1.0, which="LM")
    o = np.argsort(w2)
    return np.sqrt(np.clip(w2[o], 0, None)) / (2 * np.pi), phi[:, o], keep


# ---------------------------------------------------------------- tests
def test_elements():
    rng = np.random.default_rng(0)
    # rigid-body motions must give zero strain energy: u = a + theta x X, rotations = theta
    X = rng.normal(size=(4, 3))
    X[:, 2] = 0.3 * X[:, 0] - 0.2 * X[:, 1]  # planar, tilted
    Ke, _ = shell4_global(X, 0.1, (2e11, 0.3, 7850))
    Xb = rng.normal(size=(3, 3))
    Kb, _ = beam_global(Xb[0], Xb[1], Xb[2], (0.01, np.inf, np.inf, 1e-5, 2e-5, 3e-5), (2e11, 0.3, 7850))
    for name, Kmat, P in (("shell4", Ke, X), ("beam", Kb, Xb[:2])):
        worst = 0
        for _ in range(6):
            a, th = rng.normal(size=3), rng.normal(size=3)
            U = np.concatenate([np.concatenate([a + np.cross(th, p), th]) for p in P])
            worst = max(worst, abs(U @ Kmat @ U) / (np.abs(Kmat).max() * (U @ U)))
        print(f"  rigid-body energy, {name}: {worst:.1e} (relative)")
    # cantilever beam: first bending frequency vs Euler-Bernoulli 1.875^2 sqrt(EI / (rho A L^4)) / 2 pi
    L, n, E, rho, A, I = 10.0, 40, 2e11, 7850, 0.01, 1e-5
    Kg, Mg = np.zeros((6 * (n + 1),) * 2), np.zeros((6 * (n + 1),) * 2)
    for i in range(n):
        Ke, Me = beam_global(np.array([i * L / n, 0, 0]), np.array([(i + 1) * L / n, 0, 0]), np.array([0, 1, 0]),
                             (A, np.inf, np.inf, 2 * I, I, I), (E, 0.3, rho))
        s = slice(6 * i, 6 * i + 12)
        Kg[s, s] += Ke
        Mg[s, s] += Me
    import scipy.linalg as sl
    w = np.sqrt(np.sort(sl.eigh(Kg[6:, 6:], Mg[6:, 6:], eigvals_only=True))[:2]) / (2 * np.pi)
    ref = 1.875104 ** 2 * np.sqrt(E * I / (rho * A * L ** 4)) / (2 * np.pi)
    print(f"  cantilever f1 {w[0]:.5f} Hz vs analytical {ref:.5f} ({100 * (w[0] / ref - 1):+.3f}%)")


def test_cubic_w():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=3), rng.normal(size=3)
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    worst = 0
    for w0, wx, wy in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        U = np.concatenate([[w0 + wx * xi + wy * yi, wy, -wx] for xi, yi in zip(x, y)])
        for _ in range(5):
            L = rng.dirichlet(np.ones(3))
            worst = max(worst, abs(cubic_w(L, b, c) @ U - (w0 + wx * (L @ x) + wy * (L @ y))))
    print(f"  cubic w reproduces linear fields: max error {worst:.1e}")


def test_plate():
    """StaBIL manual example 2.2: simply supported 10 x 10 x 1 plate, 20 x 20 shell4 mesh.
    StaBIL reports 0.1268, 0.3170, 0.3170, 0.5050 Hz (analytical 0.1270, 0.3176, 0.3176, 0.5080)."""
    Lx = Ly = 10.0
    h, E, nu, rho, m = 1.0, 2e5, 0.3, 7000, 20
    xs = np.linspace(0, Lx, m + 1)
    nid = lambda i, j: j * (m + 1) + i
    Xn = np.array([[xs[i], xs[j], 0] for j in range(m + 1) for i in range(m + 1)])
    nd = 6 * len(Xn)
    K, M = np.zeros((nd, nd)), np.zeros((nd, nd))
    for j in range(m):
        for i in range(m):
            n = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            Ke, Me = shell4_global(Xn[n], h, (E, nu, rho))
            idx = np.array([6 * v + d for v in n for d in range(6)])
            K[np.ix_(idx, idx)] += Ke
            M[np.ix_(idx, idx)] += Me
    rem = set()
    for v in range(len(Xn)):
        rem |= {6 * v, 6 * v + 1, 6 * v + 5}
        x, y = Xn[v, :2]
        edge_x = np.isclose(y, 0) or np.isclose(y, Ly)  # edges along x: w = 0 so w,x = 0, ry fixed, rx free
        edge_y = np.isclose(x, 0) or np.isclose(x, Lx)  # as the manual: [Edge1;Edge3]+0.05, [Edge2;Edge4]+0.04
        if edge_x or edge_y:
            rem.add(6 * v + 2)
        if edge_x:
            rem.add(6 * v + 4)
        if edge_y:
            rem.add(6 * v + 3)
    keep = np.array(sorted(set(range(nd)) - rem))
    import scipy.linalg as sl
    # M has no rotational inertia (singular), so solve M phi = mu K phi, mu = 1 / omega^2, largest mu
    mu = sl.eigh(M[np.ix_(keep, keep)], K[np.ix_(keep, keep)], eigvals_only=True, subset_by_index=[len(keep) - 4, len(keep) - 1])
    f = np.sort(1 / np.sqrt(mu)) / (2 * np.pi)
    ana = np.sqrt(E * h ** 2 / (12 * (1 - nu ** 2) * rho)) * np.pi ** 2 * np.array([2, 5, 5, 8]) / Lx ** 2 / (2 * np.pi)
    print(f"  plate: {np.round(f, 4)} Hz; StaBIL 0.1268 0.317 0.317 0.505; ratio to analytical {np.round(f / ana, 4)}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_elements()
        test_cubic_w()
        test_plate()
        sys.exit()
    from kw51_model import build
    import time
    t0 = time.time()
    m = build(hb_rel=1.1, kxx=np.nan, Id_rel=0.06, Ad_rel=0.78)
    sysm = assemble(m)
    print(f"assembled in {time.time() - t0:.0f} s: {len(sysm['DOF'])} DOFs -> {len(sysm['MasterDOF'])} after constraints")
    f, phi, keep = modes(sysm["K"], sysm["M"], 50)
    print("dropped empty DOFs:", len(sysm["MasterDOF"]) - len(keep))
    sel = [1, 2, 3, 4, 5, 6, 7, 15, 20, 25, 27, 30, 39, 41]
    print("FE modes 1-50 (Hz):", np.round(f, 3))
    print("table-6 modes", sel, "->", np.round(f[np.array(sel) - 1], 3))
