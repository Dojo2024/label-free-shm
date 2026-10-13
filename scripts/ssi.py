"""Covariance-driven stochastic subspace identification (SSI-COV; Peeters & De Roeck 1999, the classic
output-only method) with a stabilisation diagram: natural frequencies, damping ratios and mode shapes
from output-only acceleration records, no input force needed.

How it works, in four steps:
  1. output covariances R_k = E[y(t+k) y_ref(t)^T] for lags k = 1 .. 2i-1;
  2. stacked into a block Toeplitz matrix, which factorises as (observability) x (controllability)
     for a state-space model of the structure; its SVD gives the observability matrix for any
     model order n;
  3. the shift structure of that matrix gives the state matrix A, whose eigenvalues are the poles
     (frequency + damping) and whose eigenvectors, seen through the output matrix C, are the
     mode shapes at the sensors;
  4. the true order is unknown, so every order in a range is fitted. Physical modes reappear at
     the same frequency, damping and shape as the order grows ("stable"); noise poles wander.
     Modes = clusters of stable poles that show up at most of the orders.

    python scripts/ssi.py      # Gate 1: the IASC-ASCE 12-DOF frame (iasc_asce.simulate), known answer

Gate 1 passes when every identified frequency is within 1% of the truth and every mode
shape reaches MAC >= 0.95.
"""
import time

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.signal


def mac(a, b):
    """Modal assurance criterion between shape columns of a and b: 1 = same shape, 0 = orthogonal."""
    a, b = np.atleast_2d(a.T).T, np.atleast_2d(b.T).T
    return np.abs(a.conj().T @ b) ** 2 / np.outer((np.abs(a) ** 2).sum(0), (np.abs(b) ** 2).sum(0))


def ssi_cov(y, fs, i, orders, ref=None):
    """Poles at every model order in `orders`.

    y (N, l) outputs at fs Hz; ref = indices of the reference channels (default all). i = block rows:
    the longest covariance lag is 2i-1 samples, which should span a few periods of the lowest mode.
    Orders must be <= len(ref) * i. Returns {order: (f [Hz], zeta, shapes (l, k) complex)}, one pole
    per conjugate pair, decaying poles only, sorted by frequency."""
    y = y - y.mean(0)
    ref = np.arange(y.shape[1]) if ref is None else np.asarray(ref)
    N, l = y.shape
    R = [y[k:].T @ y[:N - k, ref] / (N - k) for k in range(2 * i)]
    T = np.block([[R[i + a - b] for b in range(i)] for a in range(i)])   # row a, col b: R_{i+a-b}
    U, s, _ = scipy.linalg.svd(T, full_matrices=False)
    out = {}
    for n in orders:
        O = U[:, :n] * np.sqrt(s[:n])
        A = scipy.linalg.lstsq(O[:-l], O[l:])[0]
        mu, psi = scipy.linalg.eig(A)
        lam = np.log(mu.astype(complex)) * fs
        keep = (lam.imag > 0) & (lam.real < 0)
        lam, phi = lam[keep], (O[:l] @ psi)[:, keep]
        o = np.argsort(np.abs(lam))
        out[n] = (np.abs(lam[o]) / (2 * np.pi), -lam.real[o] / np.abs(lam[o]), phi[:, o])
    return out


def stable_poles(poles, df=0.01, dz=0.05, dmac=0.02):
    """Stabilisation diagram. A pole at order n is stable if the previous order has a pole within
    df (relative) in frequency, dz (relative) in damping and 1 - MAC < dmac.
    Returns a list of (order, f, zeta, shape)."""
    orders = sorted(poles)
    out = []
    for prev, n in zip(orders, orders[1:]):
        fp, zp, sp = poles[prev]
        f, z, s = poles[n]
        if len(fp) == 0:
            continue
        m = mac(s, sp)
        for k in range(len(f)):
            j = np.argmin(np.abs(fp - f[k]) / f[k] + (1 - m[k]))
            if (abs(fp[j] - f[k]) / f[k] < df and abs(zp[j] - z[k]) / z[k] < dz and 1 - m[k, j] < dmac):
                out.append((n, f[k], z[k], s[:, k]))
    return out


def pick_modes(stable, n_orders, fmax=np.inf, min_frac=0.4, tol=0.005, min_mac=0.9, zmax=0.1):
    """Group stable poles into modes: a pole joins the cluster whose median frequency is within tol
    and whose reference shape has MAC >= min_mac with it (so close modes with different shapes stay
    apart). Keep clusters present at >= min_frac of the fitted orders, below fmax, with median
    damping <= zmax. Returns a list of dicts (f, zeta, shape, n) sorted by frequency."""
    clusters = []
    for n, f, z, s in sorted(stable, key=lambda p: -p[0]):     # high orders first: best estimates
        if f > fmax or z > zmax:
            continue
        best, best_mac = None, min_mac
        for c in clusters:
            fc = np.median(c["f"])
            m = mac(s, c["shape"])[0, 0]
            if abs(f - fc) / fc < tol and m >= best_mac:
                best, best_mac = c, m
        if best is None:
            clusters.append(dict(f=[f], z=[z], orders={n}, shape=s))
        elif n not in best["orders"]:
            best["f"].append(f), best["z"].append(z), best["orders"].add(n)
    modes = [dict(f=float(np.median(c["f"])), zeta=float(np.median(c["z"])), shape=c["shape"],
                  n=len(c["orders"])) for c in clusters if len(c["orders"]) >= min_frac * n_orders]
    modes = [m for m in modes if m["zeta"] <= zmax]
    return sorted(modes, key=lambda m: m["f"])


def identify(y, fs, i, orders, ref=None, **pick):
    poles = ssi_cov(y, fs, i, orders, ref)
    return pick_modes(stable_poles(poles), len(orders), **pick)


def gate1(duration=1800.0, noise=0.10, seed=0):
    """Gate 1 on the IASC-ASCE 12-DOF frame: asymmetric mass with multi-point
    white-noise excitation so all 12 modes are sufficiently excited, 16 accelerometers,
    10% sensor noise, 1% damping."""
    import iasc_asce as ia
    t0 = time.time()
    # Multi-point excitation: white noise at every floor in x and y, not just the roof.
    M, K = ia.build_12dof(None, "asymmetric", True)
    lam, Phi = scipy.linalg.eigh(K, M)
    f_true, shapes_true = np.sqrt(lam) / (2 * np.pi), ia.sensor_matrix() @ Phi
    zeta = 0.01
    dt = 0.001
    n = int(round(duration / dt))
    rng = np.random.default_rng(seed)
    # Forces at all 4 floors, 2 directions each = 8 independent sources
    Bf = np.zeros((12, 8))
    for fl in range(4):
        Bf[3 * fl, 2 * fl] = 1.0       # x direction
        Bf[3 * fl + 1, 2 * fl + 1] = 1.0  # y direction
    w = np.sqrt(lam)
    Cs = ia.sensor_matrix()
    b, a = scipy.signal.butter(6, 100.0, fs=1 / dt)
    f = scipy.signal.lfilter(b, a, rng.standard_normal((n, 8)), axis=0)
    p = f @ (Phi.T @ Bf).T  # modal forces
    q, qd = np.zeros((n, 12)), np.zeros((n, 12))
    for j in range(12):
        A = np.array([[0.0, 1.0], [-w[j] ** 2, -2 * zeta * w[j]]])
        sysd = scipy.signal.cont2discrete((A, np.array([[0.0], [1.0]]), np.eye(2), np.zeros((2, 1))), dt, method="zoh")
        _, _, x = scipy.signal.dlsim(sysd[:4] + (dt,), p[:, j:j + 1])
        q[:, j], qd[:, j] = x[:, 0], x[:, 1]
    qdd = p - qd * (2 * zeta * w) - q * w ** 2
    acc = (qdd @ Phi.T) @ Cs.T
    rms = np.sqrt((acc ** 2).mean(axis=0)).max()
    y = acc + noise * rms * rng.standard_normal(acc.shape)
    y = scipy.signal.decimate(y, 4, axis=0)                     # 1000 -> 250 Hz
    fs = 250.0
    sim = time.time() - t0
    modes = identify(y, fs, i=80, orders=range(30, 201, 2), fmax=110,
                     min_frac=0.3, tol=0.003)
    print(f"simulated {duration:.0f}s in {sim:.0f}s, identified in {time.time() - t0 - sim:.0f}s: "
          f"{len(modes)} modes found, 12 true")
    if not modes:
        print("GATE 1 FAILED (no modes found)")
        return False
    # Hungarian one-to-one matching on combined cost (MAC + log-frequency).
    M_match, K_match = 12, len(modes)
    f_found = np.array([md["f"] for md in modes])
    sh_found = np.array([md["shape"] for md in modes]).T  # (16, K)
    m = mac(sh_found, shapes_true[:, :M_match])  # (K, M) = MAC[j, k]
    logf_true = np.log(f_true[:M_match])         # (M,)
    logf_found = np.log(f_found)                 # (K,)
    cost = 1 - m.T + np.abs(logf_true[:, None] - logf_found[None, :])  # (M, K)
    rows, cols = scipy.optimize.linear_sum_assignment(cost)
    print(f" {'true f':>8s} {'found f':>8s} {'err %':>7s} {'MAC':>6s} {'zeta %':>7s}   (true zeta 1.00%)")
    ok = True
    matched_found = set()
    for k, j in zip(rows, cols):
        matched_found.add(j)
        err = 100 * (modes[j]["f"] / f_true[k] - 1)
        good = abs(err) < 1 and m[j, k] >= 0.95
        ok &= good
        print(f" {f_true[k]:8.2f} {modes[j]['f']:8.2f} {err:+7.2f} {m[j, k]:6.3f} {100 * modes[j]['zeta']:7.2f}"
              f"   {'ok' if good else 'FAIL'}")
    # Report any unmatched true modes
    unmatched = set(range(M_match)) - set(rows)
    for k in sorted(unmatched):
        print(f" {f_true[k]:8.2f}   (unmatched)"); ok = False
    print("GATE 1", "PASSED" if ok else "FAILED", "(every frequency within 1%, every MAC >= 0.95)")
    return ok


if __name__ == "__main__":
    gate1()
