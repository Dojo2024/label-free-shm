"""Gate 1 on Shin-Nakagawa (the project's first validation gate):
  "on simulated records, identified frequencies are within 1% of the truth and mode shapes reach a
   MAC of at least 0.95. Modal superposition matches OpenSees within 2% RMS at the sensors."

Records: output/complete_simulation_dataset, 9 variants (healthy + 8 Table 3 damage cases) x 30
K-NET events, made by time_history.py (OpenSees, Rayleigh 3% on modes 1-2 of that variant,
Newmark average acceleration, record dt, relative accelerations at 20 sensors x 3 directions).

  A Modal superposition. The same variant's first N_MODES modes (mass-normalised on the lumped
    translational masses; the model has no rotational mass), participation from the K-NET
    components with time_history's direction map, modal damping from the same Rayleigh
    coefficients, and the same Newmark scheme, step and initial state (zero acceleration at t = 0,
    load zero after the record ends, as OpenSees's Path series). Error = RMS(superposition -
    OpenSees) / RMS(OpenSees), per channel and pooled over the record's channels.
  B SSI. Covariance-driven SSI (ssi.py) on each record, decimated to FS_ID Hz, all 60 channels.
    Truth = the variant's FE modes at the sensor channels. Targets = FE modes below F_MAX whose
    superposed response holds >= 1% of the record's pooled response energy (an unexcited mode
    cannot be identified from any record). Each target is matched to the identified mode with the
    best MAC within 5% in frequency; it passes if |error| < 1% and MAC >= 0.95.
    Output-only SSI-COV assumes white excitation; an earthquake is neither white nor stationary, and on
    ~100 s records it finds few modes and biases close pairs by ~1% (13 Oct). The WSN also records
    the ground (free-field sensors), so the main method is input-output: PO-MOESP subspace
    identification on the decimated ground motion and responses (12 principal output combinations),
    with the same stabilisation and picking as ssi.py. Both are reported, per record and pooled per
    variant (pooled = the covariances, or the MOESP data columns, of all 30 records together).

    python scripts/g1_shin_nakagawa.py --workers 30        # about 1 min per pass on 30 cores
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.signal

sys.path.insert(0, os.path.dirname(__file__))
ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "output/complete_simulation_dataset"
OUT = ROOT / "results/g1_shin_nakagawa"
N_MODES, ENERGY_MIN = 200, 0.01
# SSI settings: the 13 Oct single-record sweep (fs 10/5 Hz, i 20/40, all or 40 reference channels) put
# poles nearest the FE modes at fs 5 Hz, i 20; targets stop at 2 Hz, below that Nyquist (2.5 Hz)
FS_ID, I_BLK, ORDERS, F_MAX, TOL = 5.0, 20, range(10, 121, 2), 2.0, 0.01
# Input-output (PO-MOESP): 10 Hz (targets to 4 Hz), S_MOESP block rows (4 s of lags), R_PC output
# combinations, model orders ORDERS_ERA. OKID/ERA was tried first (13 Oct): the least-squares observer
# fitted to noise-free, low-passed data is unstable (|eig| 4-11) and its Markov parameters diverge.
FS_IO, S_MOESP, R_PC, ORDERS_ERA, F_MAX_IO = 10.0, 40, 12, range(20, 101, 2), 4.0
NO_SSI = "--no-ssi" in sys.argv
NOISE = float(sys.argv[sys.argv.index("--noise") + 1]) if "--noise" in sys.argv else 0.0

_cache = {}


def modal_data(variant):
    """Frequencies, participation (mode x direction), sensor shapes (60 x mode) and Rayleigh
    coefficients for one variant, built exactly as time_history.run_time_history builds it."""
    if variant in _cache:
        return _cache[variant]
    import openseespy.opensees as ops
    from fe_model import build_model, sensor_output_nodes
    params = json.load(open(ROOT / "calibrated_params.json"))
    model = build_model(params, damage_case=None if variant == "healthy_baseline" else variant)
    # one eigen call only: a second ops.eigen() in the same model fails ("no EigenSOE has been set").
    # Modes 1-2 of it equal get_frequencies(2), which time_history uses for the Rayleigh targets.
    lam = np.array(ops.eigen(N_MODES))
    w = np.sqrt(lam)
    w1, w2 = w[0], w[1]
    alpha, beta = 0.03 * 2 * w1 * w2 / (w1 + w2), 0.03 * 2 / (w1 + w2)
    nodes = ops.getNodeTags()
    ev = {n: np.array([ops.nodeEigenvector(n, j + 1) for j in range(N_MODES)]) for n in nodes}  # (mode, dof), read before re-analysis
    # The actual M and C of the transient analysis (same Rayleigh call as time_history). Bearings
    # (zeroLength) and cables (corotTruss) are built without -doRayleigh, so they get no
    # stiffness-proportional damping: C = alpha M + beta K_beams, which modes do NOT diagonalise.
    ops.rayleigh(alpha, 0.0, beta, 0.0)
    ops.wipeAnalysis()
    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Transformation")
    ops.algorithm("Linear")
    mats = {}
    for name, coef in (("M", (1.0, 0.0, 0.0)), ("C", (0.0, 1.0, 0.0)), ("K", (0.0, 0.0, 1.0))):
        ops.integrator("GimmeMCK", *coef)
        ops.analysis("Transient")
        ops.analyze(1, 0.0)
        nd = ops.systemSize()
        mats[name] = np.array(ops.printA("-ret")).reshape(nd, nd)
    eq = {n: ops.nodeDOFs(n) for n in nodes}  # equation numbers per node DOF (-1 = constrained)
    nd = mats["M"].shape[0]
    Phi = np.zeros((nd, N_MODES))
    R = np.zeros((nd, 3))  # influence vectors of the three ground directions
    for n in nodes:
        for k, e in enumerate(eq[n]):
            if e >= 0:
                Phi[e] = ev[n][:, k]
                if k < 3:
                    R[e, k] = 1.0
    Phi /= np.sqrt(np.einsum("ij,ik,kj->j", Phi, mats["M"], Phi))[None, :]
    gamma = (Phi.T @ mats["M"] @ R)  # (mode, 3)
    Cr = Phi.T @ mats["C"] @ Phi     # projected damping, full (non-classical)
    sens = sensor_output_nodes(model)
    S = np.stack([Phi[eq[sens[s]][k]] for s in sorted(sens) for k in range(3)])  # (60, mode)
    d = dict(w=w, gamma=gamma, S=S, Cr=Cr, zeta=alpha / (2 * w) + beta * w / 2, alpha=alpha, beta=beta,
             offdiag=float(np.linalg.norm(Cr - np.diag(np.diag(Cr))) / np.linalg.norm(Cr)))
    d.update(complex_modes(mats["M"], mats["C"], mats["K"], R, [eq[sens[s]][k] for s in sorted(sens) for k in range(3)]))
    _cache[variant] = d
    return d


def complex_modes(M, C, K, R, sidx):
    """Complex-mode superposition data. 354 of the 684 DOFs (rotations, pier nodes) are massless but
    damped, so they follow first-order dynamics C_rr u_r' = -K_rm u_m - K_rr u_r - C_rm v_m that real
    modes (which condense them statically) miss. C_rr has a (near) null direction (eigenvalue ~1e-9 of
    a 4e10 maximum): along it the DOFs are static, solved from equilibrium as OpenSees does. With
    u_r = Q_d q_d + Q_0 q_0 (damped / undamped directions of C_rr), the state is x = [u_m, v_m, q_d],
    x' = A x + B ug, q_0 and so u_r are linear in x (u_r = P x). A is diagonalised with every mode kept
    (no truncation), and each complex mode is stepped with the trapezoidal rule, which is what Newmark
    average acceleration is on this first-order form."""
    mass = np.diag(M) > 0
    m_, r_ = np.where(mass)[0], np.where(~mass)[0]
    nm, nr = m_.size, r_.size
    Minv = 1 / np.diag(M)[m_]
    sub = lambda X, a, b: X[np.ix_(a, b)]
    c, Q = np.linalg.eigh(sub(C, r_, r_))
    dmp = c > 1e-12 * c.max()
    Qd, Q0, cd = Q[:, dmp], Q[:, ~dmp], c[dmp]
    nq = Qd.shape[1]
    Krm, Crm, Krr = sub(K, r_, m_), sub(C, r_, m_), sub(K, r_, r_)
    Z = lambda r, c_: np.zeros((r, c_))
    # q_0 = H x from the undamped directions' equilibrium; u_r = P x
    H = -np.linalg.solve(Q0.T @ Krr @ Q0, Q0.T @ np.hstack([Krm, Crm, Krr @ Qd])) if Q0.shape[1] else Z(0, 2 * nm + nq)
    P = np.hstack([Z(nr, 2 * nm), Qd]) + Q0 @ H
    Gq = -(1 / cd)[:, None] * (Qd.T @ (np.hstack([Krm, Crm, Z(nr, nq)]) + Krr @ P))  # q_d' = Gq x
    nx = 2 * nm + nq
    A = np.zeros((nx, nx))
    A[:nm, nm:2 * nm] = np.eye(nm)
    A[nm:2 * nm] = Minv[:, None] * (-np.hstack([sub(K, m_, m_), sub(C, m_, m_), Z(nm, nq)]) - sub(K, m_, r_) @ P
                                    - sub(C, m_, r_) @ Qd @ Gq)
    A[2 * nm:] = Gq
    B = np.zeros((nx, 3))
    B[nm:2 * nm] = -R[m_]  # M^-1 (-M R ug)
    lam, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    pos = {e: i for i, e in enumerate(m_)}
    posr = {e: i for i, e in enumerate(r_)}
    # sensor accelerations: massed DOF -> the v_m row of x' (equilibrium, as Newmark enforces); massless
    # DOF -> Newmark's kinematic acceleration from the velocity u_r' = P x' = P (A x + B ug)
    out_acc, out_vel = np.zeros((len(sidx), nx)), np.zeros((len(sidx), nx))
    D_acc, D_vel = np.zeros((len(sidx), 3)), np.zeros((len(sidx), 3))
    is_mass = np.zeros(len(sidx), bool)
    for k, e in enumerate(sidx):
        if e in pos:
            out_acc[k], D_acc[k], is_mass[k] = A[nm + pos[e]], B[nm + pos[e]], True
        else:
            out_vel[k], D_vel[k] = P[posr[e]] @ A, P[posr[e]] @ B
    return dict(lam=lam, Bz=Vinv @ B, Ca=out_acc @ V, Cv=out_vel @ V, Da=D_acc, Dv=D_vel, is_mass=is_mass,
                n_states=nx, n_static=int((~dmp).sum()), n_sensor_massless=int((~is_mass).sum()),
                max_re_lam=float(lam.real.max()))


def complex_superpose(ug, d, dt):
    """Trapezoidal (= Newmark average acceleration) per complex mode from rest, zero acceleration at
    t = 0 (OpenSees's state: the first step's input average uses ug(0) := 0)."""
    lam = d["lam"]
    a = (1 + lam * dt / 2) / (1 - lam * dt / 2)
    b = (dt / 2) / (1 - lam * dt / 2)
    g = ug.copy()
    g[0] = 0.0
    z = np.zeros(lam.size, complex)
    n = g.shape[0] - 1
    acc = np.empty((n, d["Ca"].shape[0]))
    vel = np.empty((n, d["Ca"].shape[0]))
    for i in range(1, n + 1):
        z = a * z + b * (d["Bz"] @ (g[i - 1] + g[i]))
        acc[i - 1] = (d["Ca"] @ z).real + d["Da"] @ g[i]
        vel[i - 1] = (d["Cv"] @ z).real + d["Dv"] @ g[i]
    # massless sensor DOFs: Newmark's acceleration a_{k+1} = 2 (v_{k+1} - v_k) / dt - a_k, from rest
    mk = ~d["is_mass"]
    if mk.any():
        v_prev, a_prev = np.zeros(mk.sum()), np.zeros(mk.sum())
        for i in range(n):
            a_prev = 2 * (vel[i, mk] - v_prev) / dt - a_prev
            v_prev = vel[i, mk]
            acc[i, mk] = a_prev
    return acc


def ground(event_id, n):
    from knet_parser import read_event
    from time_history import COMPONENT_DIR, RAW_GM_DIR
    comps = read_event(RAW_GM_DIR, event_id)
    dt = {c["dt"] for c in comps.values()}.pop()
    ug = np.zeros((n + 1, 3))  # ug[k] = ground acceleration at t = k dt; zero after the record
    for c, rec in comps.items():
        a = rec["accel_gal"][:n] / 100.0
        ug[: len(a), COMPONENT_DIR[c] - 1] = a
    return ug, dt


def newmark(p, w, zeta, dt):
    """Linear SDOFs (unit mass) by Newmark average acceleration, vectorised over modes, starting
    from rest with zero acceleration (OpenSees's state at t = 0). p (steps+1, modes) at t = k dt.
    Returns accelerations at t = dt .. steps dt."""
    g, b = 0.5, 0.25
    c, k = 2 * zeta * w, w ** 2
    a1 = 1 / (b * dt ** 2) + g / (b * dt) * c
    a2 = 1 / (b * dt) + (g / b - 1) * c
    a3 = (1 / (2 * b) - 1) + dt * (g / (2 * b) - 1) * c
    kh = k + a1
    u, v, a = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
    out = np.empty((p.shape[0] - 1, w.size))
    for i in range(1, p.shape[0]):
        un = (p[i] + a1 * u + a2 * v + a3 * a) / kh
        an = (un - u) / (b * dt ** 2) - v / (b * dt) - (1 / (2 * b) - 1) * a
        v = v + dt * ((1 - g) * a + g * an)
        u, a = un, an
        out[i - 1] = a
    return out


def newmark_coupled(p, w, Cr, dt):
    """Same scheme on the reduced system I q'' + Cr q' + diag(w^2) q = p (Cr full)."""
    g, b = 0.5, 0.25
    I = np.eye(w.size)
    A1 = I / (b * dt ** 2) + g / (b * dt) * Cr
    A2 = I / (b * dt) + (g / b - 1) * Cr
    A3 = (1 / (2 * b) - 1) * I + dt * (g / (2 * b) - 1) * Cr
    Kinv = np.linalg.inv(np.diag(w ** 2) + A1)
    u, v, a = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
    out = np.empty((p.shape[0] - 1, w.size))
    for i in range(1, p.shape[0]):
        un = Kinv @ (p[i] + A1 @ u + A2 @ v + A3 @ a)
        an = (un - u) / (b * dt ** 2) - v / (b * dt) - (1 / (2 * b) - 1) * a
        v = v + dt * ((1 - g) * a + g * an)
        u, a = un, an
        out[i - 1] = a
    return out


def one(task):
    variant, csv = task
    import ssi
    t0 = time.time()
    rec = np.loadtxt(csv, delimiter=",", skiprows=1)
    y_os = rec[:, 1:]
    n = y_os.shape[0]
    d = modal_data(variant)
    ug, dt = ground(Path(csv).stem, n)
    p = -(ug @ d["gamma"].T)  # (steps+1, modes)
    rms = lambda x: np.sqrt((x ** 2).mean(0))
    # classical: diagonal modal damping from the Rayleigh formula (what a textbook simulator does)
    err_cl = newmark(p, d["w"], d["zeta"], dt) @ d["S"].T - y_os
    pooled_cl = float(np.sqrt((err_cl ** 2).sum() / (y_os ** 2).sum()))
    # coupled: the model's actual damping projected on the modes
    qdd = newmark_coupled(p, d["w"], d["Cr"], dt)
    y_ms = qdd @ d["S"].T
    err = y_ms - y_os
    ch = rms(err) / rms(y_os)
    pooled = float(np.sqrt((err ** 2).sum() / (y_os ** 2).sum()))
    if NO_SSI:
        return dict(variant=variant, event=Path(csv).stem, n=n, sp_pooled=pooled, sp_pooled_classical=pooled_cl,
                    sp_ch_median=float(np.median(ch)), sp_ch_max=float(ch.max()), cr_offdiag=d["offdiag"],
                    secs=time.time() - t0, targets=[])
    # complex modes: the model's full damped dynamics, every mode kept
    err_cx = complex_superpose(ug, d, dt) - y_os
    pooled_cx = float(np.sqrt((err_cx ** 2).sum() / (y_os ** 2).sum()))
    ch_cx = rms(err_cx) / rms(y_os)
    # the same error after a 10 Hz low-pass of both (FFT mask): the band of every target mode; above it
    # the reduction drops the damped massless rotations and OpenSees's own dt = 0.01 s Newmark is inaccurate
    fr = np.fft.rfftfreq(n, dt)
    lp = lambda x: np.fft.irfft(np.fft.rfft(x, axis=0) * (fr < 10.0)[:, None], n=n, axis=0)
    e10 = lp(y_ms) - lp(y_os)
    pooled10 = float(np.sqrt((e10 ** 2).sum() / (lp(y_os) ** 2).sum()))
    ch10 = rms(e10) / rms(lp(y_os))
    # modal energy shares at the sensors (from the superposition) -> SSI targets
    share = (qdd ** 2).sum(0) * (d["S"] ** 2).sum(0)
    share /= share.sum()
    f_fe = d["w"] / (2 * np.pi)
    targets = [j for j in range(d["w"].size) if f_fe[j] < F_MAX and share[j] >= ENERGY_MIN]
    y = decimated(noisy(y_os, csv + "y"), dt)
    Rs, cnt = covariances(y)
    modes = pick(poles_from_cov([r / c for r, c in zip(Rs, cnt)], y.shape[1]))
    rows = match(modes, d, targets, share)
    # input-output: OKID/ERA on the same record
    u_io, y_io = io_data(y_os, Path(csv).stem, csv)
    W = reducer([y_io])
    modes_io = pick_io(moesp(moesp_blocks(u_io, y_io @ W), 3, R_PC, W))
    targets_io = [j for j in range(d["w"].size) if f_fe[j] < F_MAX_IO and share[j] >= ENERGY_MIN]
    rows_io = match(modes_io, d, targets_io, share)
    return dict(variant=variant, event=Path(csv).stem, n=n, dt=dt, sp_pooled=pooled, sp_pooled_classical=pooled_cl,
                n_identified_io=len(modes_io), targets_io=rows_io,
                sp_cx_pooled=pooled_cx, sp_cx_ch_max=float(ch_cx.max()), n_states=d["n_states"],
                n_sensor_massless=d["n_sensor_massless"], max_re_lam=d["max_re_lam"],
                sp_pooled_lp10=pooled10, sp_ch_max_lp10=float(ch10.max()), sp_ch_lp10=ch10.tolist(),
                ch_rms_rel=(rms(lp(y_os)) / np.median(rms(lp(y_os)))).tolist(),
                cr_offdiag=d["offdiag"], sp_ch_median=float(np.median(ch)),
                sp_ch_max=float(ch.max()), sp_ch_max_name=int(np.argmax(ch)), n_identified=len(modes), targets=rows,
                share=share.tolist(), secs=time.time() - t0)


def noisy(x, key):
    """Independent white noise at NOISE x each channel's RMS (sensor noise), seeded by the record."""
    if NOISE == 0:
        return x
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return x + NOISE * np.sqrt((x ** 2).mean(0)) * rng.standard_normal(x.shape)


def decimated(y, dt):
    q = int(round(1 / dt / FS_ID))  # 20 = 10 x 2: decimate() wants factors of at most ~13 per stage
    for f in (10, q // 10) if q > 13 else (q,):
        y = scipy.signal.decimate(y, f, axis=0, zero_phase=True)
    return y


def covariances(y):
    """Unnormalised output covariance sums and their counts for lags 0 .. 2 I_BLK - 1 (all channels as
    references), so several records can be pooled by adding sums and counts."""
    y = y - y.mean(0)
    N = y.shape[0]
    return [y[k:].T @ y[: N - k] for k in range(2 * I_BLK)], [N - k for k in range(2 * I_BLK)]


def poles_from_cov(R, l):
    """ssi.ssi_cov from the covariances on (copied, so the pooled covariances can be passed in)."""
    i = I_BLK
    T = np.block([[R[i + a - b] for b in range(i)] for a in range(i)])
    U, sv, _ = scipy.linalg.svd(T, full_matrices=False)
    out = {}
    for n in ORDERS:
        O = U[:, :n] * np.sqrt(sv[:n])
        A = scipy.linalg.lstsq(O[:-l], O[l:])[0]
        mu, psi = scipy.linalg.eig(A)
        lam = np.log(mu.astype(complex)) * FS_ID
        keep = (lam.imag > 0) & (lam.real < 0)
        lam, phi = lam[keep], (O[:l] @ psi)[:, keep]
        o = np.argsort(np.abs(lam))
        out[n] = (np.abs(lam[o]) / (2 * np.pi), -lam.real[o] / np.abs(lam[o]), phi[:, o])
    return out


def pick(poles):
    import ssi
    return ssi.pick_modes(ssi.stable_poles(poles), len(poles), fmax=F_MAX * 1.2, min_frac=0.3, tol=TOL)


def match(modes, d, targets, share):
    """Each target FE mode against the identified mode with the best MAC within 5% in frequency."""
    import ssi
    f_fe = d["w"] / (2 * np.pi)
    rows = []
    for j in targets:
        best = None
        for md in modes:
            if abs(md["f"] / f_fe[j] - 1) < 0.05:
                mc = float(ssi.mac(md["shape"][:, None], d["S"][:, j:j + 1])[0, 0])
                if best is None or mc > best[1]:
                    best = (md["f"], mc, md["zeta"])
        if best is None:
            rows.append(dict(mode=j + 1, f_fe=float(f_fe[j]), share=float(share[j]), found=False))
        else:
            e = best[0] / f_fe[j] - 1
            rows.append(dict(mode=j + 1, f_fe=float(f_fe[j]), share=float(share[j]), found=True, f_id=float(best[0]),
                             err=float(e), mac=best[1], zeta_id=float(best[2]),
                             passed=bool(abs(e) < 0.01 and best[1] >= 0.95)))
    return rows


def reducer(ys):
    """Orthonormal basis (60 x R_PC) of the responses' leading principal directions. The 60 channels
    are far from independent (several sensors share a node; noise-free responses span only the active
    modes); identification runs on these R_PC combinations, and shapes are mapped back to 60 channels."""
    U, _, _ = scipy.linalg.svd(np.vstack(ys).T, full_matrices=False)
    return U[:, :R_PC]


def moesp_blocks(u, y):
    """Block-Hankel data matrix [U_f; U_p; Y_p; Y_f] (s = S_MOESP block rows) of one record. Records are
    pooled by concatenating these column-wise."""
    s, N = S_MOESP, y.shape[0]
    j = N - 2 * s + 1
    H = lambda x, start: np.vstack([x[start + k: start + k + j].T for k in range(s)])
    return np.vstack([H(u, s), H(u, 0), H(y, 0), H(y, s)])


def moesp(Mcols, m, l, W):
    """PO-MOESP (Verhaegen 1994): LQ of the data matrix; the part of the future outputs explained by
    past inputs and outputs, after removing the future inputs, has the extended observability matrix as
    its column space. A from its shift structure, C from its first block, at every order in ORDERS_ERA.
    Returns {order: (f, zeta, shapes on 60 channels)} like ssi.ssi_cov."""
    s = S_MOESP
    L = scipy.linalg.qr(Mcols.T, mode="r")[0].T
    a, b = s * m, s * (m + l)
    L32 = L[a + b:, a: a + b]
    U, sv, _ = scipy.linalg.svd(L32, full_matrices=False)
    out = {}
    for n in ORDERS_ERA:
        G = U[:, :n] * np.sqrt(sv[:n])
        A = scipy.linalg.lstsq(G[:-l], G[l:])[0]
        C = G[:l]
        mu, psi = scipy.linalg.eig(A)
        lam = np.log(mu.astype(complex)) * FS_IO
        keep = (lam.imag > 0) & (lam.real < 0)
        lam, phi = lam[keep], (W @ C @ psi)[:, keep]
        o = np.argsort(np.abs(lam))
        out[n] = (np.abs(lam[o]) / (2 * np.pi), -lam.real[o] / np.abs(lam[o]), phi[:, o])
    return out


def io_data(y_os, event_id, key):
    """Decimated ground motion (3) and responses (60) at FS_IO, filtered identically (zero phase), so
    the input-output relation is preserved. Sensor noise (NOISE) on both, as the WSN's free-field and
    structural accelerometers would have."""
    ug, dt = ground(event_id, y_os.shape[0])
    q = int(round(1 / dt / FS_IO))
    dec = lambda x: scipy.signal.decimate(x, q, axis=0, zero_phase=True)
    # ug[k] is at t = k dt; the record's row k at t = (k + 1) dt
    return dec(noisy(ug[1:], key + "u")), dec(noisy(y_os, key + "y"))


def pick_io(poles):
    import ssi
    return ssi.pick_modes(ssi.stable_poles(poles), len(poles), fmax=F_MAX_IO * 1.2, min_frac=0.3, tol=TOL)


def pooled(args):
    """One identification per variant from the covariances of all its records added together."""
    variant, csvs, shares = args
    d = modal_data(variant)
    Rs, cnt = None, None
    for c in csvs:
        y_os = np.loadtxt(c, delimiter=",", skiprows=1)[:, 1:]
        r, n = covariances(decimated(noisy(y_os, c + "y"), 0.01))
        Rs = r if Rs is None else [a + b for a, b in zip(Rs, r)]
        cnt = n if cnt is None else [a + b for a, b in zip(cnt, n)]
    modes = pick(poles_from_cov([r / c for r, c in zip(Rs, cnt)], 60))
    share = np.mean(shares, axis=0)
    f_fe = d["w"] / (2 * np.pi)
    targets = [j for j in range(d["w"].size) if f_fe[j] < F_MAX and share[j] >= ENERGY_MIN]
    UY = [io_data(np.loadtxt(c, delimiter=",", skiprows=1)[:, 1:], Path(c).stem, c) for c in csvs]
    W = reducer([y for _, y in UY])
    modes_io = pick_io(moesp(np.hstack([moesp_blocks(u, y @ W) for u, y in UY]), 3, R_PC, W))
    targets_io = [j for j in range(d["w"].size) if f_fe[j] < F_MAX_IO and share[j] >= ENERGY_MIN]
    return dict(variant=variant, n_records=len(csvs), n_identified=len(modes), targets=match(modes, d, targets, share),
                n_identified_io=len(modes_io), targets_io=match(modes_io, d, targets_io, share))


def ssi_summary(tg):
    found = [t for t in tg if t["found"]]
    return dict(n_targets=len(tg), frac_found=len(found) / max(len(tg), 1),
                frac_passed_of_targets=float(np.mean([t.get("passed", False) for t in tg])) if tg else None,
                abs_err_median=float(np.median([abs(t["err"]) for t in found])) if found else None,
                abs_err_p90=float(np.percentile([abs(t["err"]) for t in found], 90)) if found else None,
                mac_median=float(np.median([t["mac"] for t in found])) if found else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N records only (smoke test)")
    ap.add_argument("--no-ssi", action="store_true", help="superposition only")
    ap.add_argument("--noise", type=float, default=0.0, help="sensor noise, fraction of channel RMS (identification only)")
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    tasks = [(Path(c).parent.name, c) for c in sorted(glob.glob(str(SIM / "*/*.csv")))]
    tasks = tasks[: a.limit] if a.limit else tasks
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(one, tasks, chunksize=1)
        if not a.no_ssi:
            groups = {}
            for r, (v, c) in zip(res, tasks):
                groups.setdefault(v, ([], []))
                groups[v][0].append(c), groups[v][1].append(r["share"])
            pres = pool.map(pooled, [(v, cs, sh) for v, (cs, sh) in groups.items()], chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.array([r["sp_pooled"] for r in res])
    spx = np.array([r["sp_ch_max"] for r in res])
    tg = [t for r in res for t in r["targets"]]
    summary = dict(
        n_records=len(res), wall_s=time.time() - t0,
        superposition=dict(pooled_median=float(np.median(sp)), pooled_max=float(sp.max()),
                           classical_pooled_median=float(np.median([r["sp_pooled_classical"] for r in res])),
                           frac_pooled_le_2pct=float(np.mean(sp <= 0.02)),
                           worst_channel_median=float(np.median(spx)), worst_channel_max=float(spx.max()),
                           frac_all_channels_le_2pct=float(np.mean(spx <= 0.02)),
                           lp10_pooled_median=float(np.median([r.get("sp_pooled_lp10", np.nan) for r in res])),
                           lp10_pooled_max=float(np.max([r.get("sp_pooled_lp10", np.nan) for r in res])),
                           lp10_worst_channel_max=float(np.max([r.get("sp_ch_max_lp10", np.nan) for r in res])),
                           lp10_frac_channels_le_2pct=float(np.mean([e <= 0.02 for r in res for e in r.get("sp_ch_lp10", [])])),
                           lp10_energy_share_of_channels_gt_2pct=float(
                               sum(w ** 2 for r in res for e, w in zip(r.get("sp_ch_lp10", []), r.get("ch_rms_rel", [])) if e > 0.02)
                               / max(sum(w ** 2 for r in res for w in r.get("ch_rms_rel", [])), 1e-30)),
                           lp10_failing_channel_rms_rel_max=float(max([w for r in res for e, w in zip(r.get("sp_ch_lp10", []), r.get("ch_rms_rel", [])) if e > 0.02], default=0))),
        complex_superposition=dict(
            pooled_median=float(np.median([r["sp_cx_pooled"] for r in res])), pooled_max=float(np.max([r["sp_cx_pooled"] for r in res])),
            worst_channel_max=float(np.max([r["sp_cx_ch_max"] for r in res])),
            frac_records_all_channels_le_2pct=float(np.mean([r["sp_cx_ch_max"] <= 0.02 for r in res])),
            n_states=res[0]["n_states"], n_sensor_channels_massless=res[0]["n_sensor_massless"],
            max_re_lam=float(max(r["max_re_lam"] for r in res))),
        ssi_per_record=ssi_summary(tg),
        io_per_record=ssi_summary([t for r in res for t in r.get("targets_io", [])]),
    )
    if not a.no_ssi:
        summary["ssi_pooled_per_variant"] = ssi_summary([t for r in pres for t in r["targets"]])
        summary["io_pooled_per_variant"] = ssi_summary([t for r in pres for t in r["targets_io"]])
        summary["io_pooled_detail"] = {r["variant"]: [(t["mode"], round(t["f_fe"], 3), t["found"] and round(100 * t["err"], 2),
                                                       t["found"] and round(t["mac"], 3)) for t in r["targets_io"]] for r in pres}
        summary["ssi_pooled_detail"] = {r["variant"]: [(t["mode"], round(t["f_fe"], 3), t["found"] and round(100 * t["err"], 2),
                                                        t["found"] and round(t["mac"], 3)) for t in r["targets"]] for r in pres}
    by_mode = {}
    for t in tg:
        b = by_mode.setdefault(t["mode"], dict(f_fe=t["f_fe"], n=0, found=0, passed=0, errs=[], macs=[]))
        b["n"] += 1
        if t["found"]:
            b["found"] += 1
            b["passed"] += t["passed"]
            b["errs"].append(abs(t["err"]))
            b["macs"].append(t["mac"])
    summary["ssi_by_mode"] = {m: dict(f_fe=round(b["f_fe"], 3), n=b["n"], found=b["found"], passed=b["passed"],
                                      abs_err_median=float(np.median(b["errs"])) if b["errs"] else None,
                                      mac_median=float(np.median(b["macs"])) if b["macs"] else None)
                              for m, b in sorted(by_mode.items())}
    tag = (f"_limit{a.limit}" if a.limit else "") + (f"_noise{a.noise:g}" if a.noise else "")
    summary["noise"] = a.noise
    for r in res:
        r.pop("share", None)
    json.dump(dict(summary=summary, records=res, pooled=None if a.no_ssi else pres), open(OUT / f"g1{tag}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
