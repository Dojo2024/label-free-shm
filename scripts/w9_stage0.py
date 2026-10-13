"""W9 stage 0: two checks on SeisGPT's public data before any training.

  A Shake table (1/10-scale 43-storey building, 7 runs of rising intensity). The files carry no
    sampling interval. SeisGPT's own FE response to each run (shifted_floor_data_fem) was computed
    from the released K_x, M_x, so its dominant peak (cycles/sample) divided by the model's first
    natural frequency (Hz) gives dt. With dt known, the measured first frequency of each run comes
    from the floor/table transfer function. Does it fall from run to run (progressive damage)?
    The Welch transfer-function peak resolves only ~8% at f1 ~ 0.3 cycles per ~100 s record, so f1
    is also fitted in the time domain (ARX, input = table acceleration, output = floor displacement,
    decimated 8x, orders 4/6/8, median over the 3 most responsive floors). dt and units are unknown,
    so the dt-free number is the ratio measured f1 / FE f1 within each run.
  B Storey-spring decomposition of the 65 real-derived buildings' condensed K (both directions):
    K = T' diag(k) T + K_rest, with T the storey-drift operator and k >= 0 fitted by least squares.
    How much of K is a shear chain, and how far are the chain-only frequencies (modes 1-3) from the
    full model's? Storey damage is then defined as a loss in k_i with K_rest held fixed.
  C The replacement if B fails: drift-space congruence. S = T^-T K T^-1 is K written in storey-drift
    coordinates (exact; diagonal only for a pure shear building). Damage d_i in storey i scales drift
    coordinate i: K_d = T' D S D T with D = diag(sqrt(1 - d)). Exact when healthy, positive definite
    for every d < 1, and equal to k_i (1 - d_i) for a shear building. Checked: f1 drop for 20% loss in
    each storey, and how many storeys modes 1-3 can tell apart (rank of their sensitivity).

Only metrics are written (no k vectors or other derived data; the data are GPL, cited as Meng 2026).

    OMP_NUM_THREADS=2 .venv/bin/python scripts/w9_stage0.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import nnls
from scipy.signal import csd, decimate, detrend, welch

SEIS = Path(os.environ.get("SEIS_DIR", Path.home() / ".cache/huggingface/seis"))
OUT = Path(__file__).resolve().parents[1] / "results/w9_stage0"


def load_km(d, tag=""):
    K = np.loadtxt(d / f"K{tag}.txt")
    M = np.loadtxt(d / f"M{tag}.txt")
    M = np.diag(M) if M.ndim == 1 else M
    return K, M


def freqs(K, M, n=3):
    w2 = eigh(K, M, eigvals_only=True)
    return np.sqrt(np.clip(w2[:n], 0, None)) / (2 * np.pi)


def peak(f, p, lo, hi):
    """Peak of p inside [lo, hi], refined by a parabola through the top bin and its neighbours."""
    idx = np.where((f >= lo) & (f <= hi))[0]
    i = idx[np.argmax(p[idx])]
    a, b, c = np.log(p[i - 1 : i + 2])
    return f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])


def shake_table():
    d = SEIS / "shaking_table_experiment"
    K, M = load_km(d, "_x")
    f_model = freqs(K, M, 3)
    runs = []
    for fn in sorted(glob.glob(str(d / "dataset/*.json"))):
        r = json.load(open(fn))
        meas = np.array(r["shifted_floor_data"])
        fem = np.array(r["shifted_floor_data_fem"])
        acc = np.array(r["earthquake_A"])
        top = int(np.argmax(np.std(fem, axis=1)))  # the floor with the largest response
        # dt from the FE response: periodogram peak (cycles/sample) over the model's f1 (Hz)
        fc, pf = welch(fem[top], fs=1.0, nperseg=2048, noverlap=1536, nfft=1 << 15)
        guess = fc[np.argmax(pf[1:]) + 1]
        p1 = peak(fc, pf, 0.5 * guess, 1.5 * guess)
        dt = p1 / f_model[0]
        # measured f1: H1 transfer function table acceleration -> floor displacement
        ft, pxy = csd(acc, meas[top], fs=1 / dt, nperseg=2048, noverlap=1536, nfft=1 << 15)
        _, pxx = welch(acc, fs=1 / dt, nperseg=2048, noverlap=1536, nfft=1 << 15)
        h = np.abs(pxy / pxx)
        _, pyy = csd(acc, fem[top], fs=1 / dt, nperseg=2048, noverlap=1536, nfft=1 << 15)
        hf = np.abs(pyy / pxx)
        band = (0.5 * f_model[0], 1.5 * f_model[0])
        name = Path(fn).stem
        runs.append(dict(
            run=name, n=int(meas.shape[1]), dt=float(dt), duration_s=float(dt * meas.shape[1]),
            pga=float(np.max(np.abs(acc))), top_row=top,
            corr_reported=float(r["average_correlation"]),
            f1_fem_tf=float(peak(ft, hf, *band)), f1_meas_tf=float(peak(ft, h, *band)),
            peak_disp_meas=float(np.max(np.abs(meas[top]))), peak_disp_fem=float(np.max(np.abs(fem[top]))),
        ))
        rows3 = np.argsort(np.std(fem, axis=1))[-3:]
        runs[-1]["f1_arx_meas_cps"] = float(np.median([arx_f1(acc, meas[i], p1) for i in rows3]))
        runs[-1]["f1_arx_fem_cps"] = float(np.median([arx_f1(acc, fem[i], p1) for i in rows3]))
        # robustness: first and second half of the record; mode 2 (needs a finer decimation)
        h = meas.shape[1] // 2
        runs[-1]["f1_arx_meas_halves_cps"] = [float(np.median([arx_f1(acc[s], meas[i, s], p1) for i in rows3]))
                                              for s in (slice(0, h), slice(h, None))]
        p2 = p1 * f_model[1] / f_model[0]
        runs[-1]["f2_arx_meas_cps"] = float(np.nanmedian([arx_f1(acc, meas[i], p2, q=3) for i in range(meas.shape[0])]))
        runs[-1]["f2_arx_fem_cps"] = float(np.nanmedian([arx_f1(acc, fem[i], p2, q=3) for i in range(fem.shape[0])]))
    dt = float(np.median([r["dt"] for r in runs]))  # one sampling rate for all runs
    for r in runs:
        r["ratio_meas_fem"] = r["f1_arx_meas_cps"] / r["f1_arx_fem_cps"]
        r["f1_arx_meas_hz"] = r["f1_arx_meas_cps"] / dt
        r["f1_arx_fem_hz"] = r["f1_arx_fem_cps"] / dt
    f0, g0 = runs[0]["f1_arx_meas_cps"], runs[0]["f2_arx_meas_cps"]
    for r in runs:
        r["f1_drop_vs_first"] = 1 - r["f1_arx_meas_cps"] / f0
        r["f2_drop_vs_first"] = 1 - r["f2_arx_meas_cps"] / g0
    return dict(f_model_hz=f_model.tolist(), dt_median=dt,
                dt_spread=float(np.ptp([r["dt"] for r in runs])), runs=runs)


def arx_f1(u, y, p_guess, q=8):
    """First-mode frequency (cycles per original sample) from ARX fits of orders 4/6/8."""
    u = decimate(detrend(u), q, zero_phase=True)
    y = decimate(detrend(y), q, zero_phase=True)
    out = []
    for na in (4, 6, 8):
        X = np.column_stack([y[na - i : len(y) - i] for i in range(1, na + 1)]
                            + [u[na - j : len(u) - j] for j in range(0, na + 1)])
        a = np.linalg.lstsq(X, y[na:], rcond=None)[0][:na]
        s = np.log(np.roots(np.r_[1.0, -a]).astype(complex))  # per decimated sample
        f = np.abs(s) / (2 * np.pi) / q
        zeta = -s.real / np.abs(s)
        ok = (np.abs(f / p_guess - 1) < 0.5) & (zeta > 0) & (zeta < 0.3) & (s.imag > 0)
        if ok.any():
            out.append(f[ok][np.argmin(np.abs(f[ok] / p_guess - 1))])
    return float(np.median(out)) if out else float("nan")


def shear_flexure(fit="frob"):
    """D: shear plus bending chain, K = T' diag(k) T + C' diag(b) C + K_rest, k, b >= 0 by NNLS.
    C maps floor displacements to curvatures (kappa_0 = u_1 at the clamped base, then second
    differences), the coupled shear-flexural beam of tall-building practice. Storey i damage d scales
    k_i and b_{i-1} (the curvature at the storey's lower floor) with K_rest held fixed.
    fit="frob" matches K entry by entry (dominated by the large local terms); fit="modal" matches
    Lam^-1/2 Phi' K_hat Phi Lam^-1/2 to the identity (Phi, Lam: the healthy modes), so every mode
    counts equally in relative terms and the low modes are no longer ignored."""
    rows = []
    for b in sorted(glob.glob(str(SEIS / "real_dataset/KM_data/realmodel_*"))):
        for ax in ("x", "y"):
            K, M = load_km(Path(b) / ax)
            n = K.shape[0]
            T = np.eye(n) - np.eye(n, k=-1)
            # rows of C: kappa_i = u_{i+1} - 2 u_i + u_{i-1} written on DOFs u_1..u_n (u_0 = 0)
            C = np.zeros((n, n))
            C[0, 0] = 1.0
            for i in range(1, n):
                C[i, i] = 1.0
                C[i, i - 1] = -2.0
                if i >= 2:
                    C[i, i - 2] = 1.0
            if fit == "frob":
                A = np.stack([np.outer(T[i], T[i]).ravel() for i in range(n)]
                             + [np.outer(C[i], C[i]).ravel() for i in range(n)], axis=1)
                x, _ = nnls(A, K.ravel())
            else:
                lam, Phi = eigh(K, M)
                W = Phi / np.sqrt(lam)[None, :]  # W' K W = I
                tw, cw = T @ W, C @ W
                A = np.stack([np.outer(tw[i], tw[i]).ravel() for i in range(n)]
                             + [np.outer(cw[i], cw[i]).ravel() for i in range(n)], axis=1)
                x, _ = nnls(A, np.eye(n).ravel())
            k, bb = x[:n], x[n:]
            Kc = T.T @ np.diag(k) @ T + C.T @ np.diag(bb) @ C
            f_full, f_chain = freqs(K, M), freqs(Kc, M)
            out = {}
            for name, i in (("bottom", 0), ("middle", n // 2), ("top", n - 1)):
                dK = 0.2 * k[i] * np.outer(T[i], T[i]) + 0.2 * bb[i] * np.outer(C[i], C[i])
                w2 = eigh(K - dK, M, eigvals_only=True)
                out[name] = dict(pd=bool(w2[0] > 0), f1_drop=float(1 - np.sqrt(max(w2[0], 0)) / (2 * np.pi) / f_full[0]))
            rows.append(dict(
                building=Path(b).name, axis=ax, n=n,
                rel_resid=float(np.linalg.norm(K - Kc) / np.linalg.norm(K)),
                f_err_chain=(f_chain / f_full - 1).tolist(),
                flex_share=float(np.trace(C.T @ np.diag(bb) @ C) / np.trace(Kc)), dmg20=out,
            ))
    err = np.abs(np.array([r["f_err_chain"] for r in rows]))
    summary = dict(
        rel_resid_median=float(np.median([r["rel_resid"] for r in rows])),
        chain_f_err_median=np.median(err, axis=0).tolist(), chain_f_err_p90=np.percentile(err, 90, axis=0).tolist(),
        frac_all3_within_1pct=float(np.mean((err < 0.01).all(axis=1))),
        frac_all3_within_5pct=float(np.mean((err < 0.05).all(axis=1))),
        flex_share_median=float(np.median([r["flex_share"] for r in rows])),
        frac_pd_after_dmg=float(np.mean([r["dmg20"][s]["pd"] for r in rows for s in ("bottom", "middle", "top")])),
        **{f"f1_drop_{s}_median": float(np.median([r["dmg20"][s]["f1_drop"] for r in rows])) for s in ("bottom", "middle", "top")},
    )
    return dict(summary=summary, models=rows)


def drift_flexibility():
    """E: storey damage as added series flexibility in storey-drift coordinates. G = S^-1 is the
    drift flexibility (diag(1/k) for a shear building). Damage d_i in storey i adds
    d_i/(1-d_i) G_ii to G_ii: K_d = T' (G + diag(c * diag(G)))^-1 T, c = d/(1-d). Exact when healthy,
    reduces to k_i (1-d_i) for a shear building, and can only soften (K_d <= K), never stiffen."""
    rows = []
    for b in sorted(glob.glob(str(SEIS / "real_dataset/KM_data/realmodel_*"))):
        for ax in ("x", "y"):
            K, M = load_km(Path(b) / ax)
            n = K.shape[0]
            T = np.eye(n) - np.eye(n, k=-1)
            Ti = np.linalg.inv(T)
            G = np.linalg.inv(Ti.T @ K @ Ti)
            f_h = freqs(K, M)
            drop = np.zeros((n, 3))
            for i in range(n):
                Gd = G.copy()
                Gd[i, i] *= 1 + 0.2 / 0.8
                drop[i] = 1 - freqs(T.T @ np.linalg.inv(Gd) @ T, M) / f_h
            sv = np.linalg.svd(drop, compute_uv=False)
            third = max(1, n // 3)
            rows.append(dict(
                building=Path(b).name, axis=ax, n=n,
                f1_drop_bottom=float(drop[0, 0]), f1_drop_middle=float(drop[n // 2, 0]), f1_drop_top=float(drop[-1, 0]),
                f1_drop_max=float(drop[:, 0].max()), argmax_storey_frac=float(np.argmax(drop[:, 0]) / (n - 1)),
                f1_drop_lower_third_share=float(drop[:third, 0].sum() / drop[:, 0].sum()),
                eff_rank_modes123=int((sv > 0.05 * sv[0]).sum()), min_f_drop=float(drop.min()),
            ))
    g = lambda k: np.array([r[k] for r in rows])
    summary = dict(
        **{f"{k}_median": float(np.median(g(k))) for k in ("f1_drop_bottom", "f1_drop_middle", "f1_drop_top",
                                                           "f1_drop_max", "argmax_storey_frac", "f1_drop_lower_third_share")},
        eff_rank_counts={int(k): int(v) for k, v in zip(*np.unique(g("eff_rank_modes123"), return_counts=True))},
        any_frequency_rise=bool((g("min_f_drop") < -1e-9).any()),
    )
    return dict(summary=summary, models=rows)


def drift_congruence():
    rows = []
    for b in sorted(glob.glob(str(SEIS / "real_dataset/KM_data/realmodel_*"))):
        for ax in ("x", "y"):
            K, M = load_km(Path(b) / ax)
            n = K.shape[0]
            T = np.eye(n) - np.eye(n, k=-1)
            Ti = np.linalg.inv(T)
            S = Ti.T @ K @ Ti
            f_h = freqs(K, M)
            drop = np.zeros((n, 3))  # 20% loss in storey i -> relative drop of f1..f3
            for i in range(n):
                D = np.ones(n)
                D[i] = np.sqrt(0.8)
                drop[i] = 1 - freqs(T.T @ (D[:, None] * S * D[None, :]) @ T, M) / f_h
            sv = np.linalg.svd(drop, compute_uv=False)
            third = max(1, n // 3)
            rows.append(dict(
                building=Path(b).name, axis=ax, n=n,
                offdiag_frac=float(np.linalg.norm(S - np.diag(np.diag(S))) / np.linalg.norm(S)),
                f1_drop_bottom=float(drop[0, 0]), f1_drop_middle=float(drop[n // 2, 0]), f1_drop_top=float(drop[-1, 0]),
                f1_drop_lower_third_share=float(drop[:third, 0].sum() / drop[:, 0].sum()),
                eff_rank_modes123=int((sv > 0.05 * sv[0]).sum()),
                max_f_drop=float(drop.max()), min_f_drop=float(drop.min()),
            ))
    g = lambda k: np.array([r[k] for r in rows])
    summary = dict(
        offdiag_frac_median=float(np.median(g("offdiag_frac"))),
        f1_drop_bottom_median=float(np.median(g("f1_drop_bottom"))),
        f1_drop_middle_median=float(np.median(g("f1_drop_middle"))),
        f1_drop_top_median=float(np.median(g("f1_drop_top"))),
        f1_drop_lower_third_share_median=float(np.median(g("f1_drop_lower_third_share"))),
        eff_rank_counts={int(k): int(v) for k, v in zip(*np.unique(g("eff_rank_modes123"), return_counts=True))},
        any_frequency_rise=bool((g("min_f_drop") < -1e-9).any()),
    )
    return dict(summary=summary, models=rows)


def storey_springs():
    rows = []
    for b in sorted(glob.glob(str(SEIS / "real_dataset/KM_data/realmodel_*"))):
        for ax in ("x", "y"):
            K, M = load_km(Path(b) / ax)
            n = K.shape[0]
            T = np.eye(n) - np.eye(n, k=-1)  # (T u)_i = u_i - u_{i-1}, ground fixed
            A = np.stack([np.outer(T[i], T[i]).ravel() for i in range(n)], axis=1)
            k, _ = nnls(A, K.ravel())
            Kc = T.T @ np.diag(k) @ T
            R = K - Kc
            f_full, f_chain = freqs(K, M), freqs(Kc, M)
            # 20% loss in one storey (bottom, middle), K_rest held fixed: is K_d still positive
            # definite, and how much does f1 drop?
            drops = {}
            for name, i in (("bottom", 0), ("middle", n // 2)):
                Kd = K - 0.2 * k[i] * np.outer(T[i], T[i])
                w2 = eigh(Kd, M, eigvals_only=True)
                drops[name] = dict(pd=bool(w2[0] > 0), f1_drop=float(1 - np.sqrt(max(w2[0], 0)) / (2 * np.pi) / f_full[0]))
            rows.append(dict(
                building=Path(b).name, axis=ax, n=n, asym=float(np.abs(K - K.T).max() / np.abs(K).max()),
                rel_resid=float(np.linalg.norm(R) / np.linalg.norm(K)),
                n_zero_k=int((k <= 0).sum()),
                f_err_chain=(f_chain / f_full - 1).tolist(), dmg20=drops,
            ))
    err = np.abs(np.array([r["f_err_chain"] for r in rows]))
    res = np.array([r["rel_resid"] for r in rows])
    summary = dict(
        n_models=len(rows), storeys_min_max=[min(r["n"] for r in rows), max(r["n"] for r in rows)],
        rel_resid_median=float(np.median(res)), rel_resid_p90=float(np.percentile(res, 90)),
        chain_f_err_median=np.median(err, axis=0).tolist(), chain_f_err_p90=np.percentile(err, 90, axis=0).tolist(),
        frac_all3_within_1pct=float(np.mean((err < 0.01).all(axis=1))),
        frac_any_zero_k=float(np.mean([r["n_zero_k"] > 0 for r in rows])),
        frac_pd_after_dmg=float(np.mean([r["dmg20"][s]["pd"] for r in rows for s in ("bottom", "middle")])),
        f1_drop_bottom_median=float(np.median([r["dmg20"]["bottom"]["f1_drop"] for r in rows])),
        f1_drop_middle_median=float(np.median([r["dmg20"]["middle"]["f1_drop"] for r in rows])),
    )
    return dict(summary=summary, models=rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    st, ss, dc, sf, df = shake_table(), storey_springs(), drift_congruence(), shear_flexure(), drift_flexibility()
    sfm = shear_flexure("modal")
    json.dump(dict(shake_table=st, storey_springs=ss, drift_congruence=dc, shear_flexure=sf, drift_flexibility=df,
                   shear_flexure_modal=sfm),
              open(OUT / "stage0.json", "w"), indent=1)
    dt = st["dt_median"]
    print("shake table: model f1-3 (Hz)", np.round(st["f_model_hz"], 3), f" dt median {st['dt_median']:.5f}s spread {st['dt_spread']:.5f}")
    for r in st["runs"]:
        print(f"  {r['run']:<36} PGA {r['pga']:7.1f}  TF f1 fem/meas {r['f1_fem_tf']:.3f}/{r['f1_meas_tf']:.3f}"
              f"  ARX f1 fem/meas {r['f1_arx_fem_hz']:.3f}/{r['f1_arx_meas_hz']:.3f} Hz  meas/fem {r['ratio_meas_fem']:.3f}"
              f"  drop vs run 1 {100 * r['f1_drop_vs_first']:5.1f}%  halves {r['f1_arx_meas_halves_cps'][0] / dt:.3f}/{r['f1_arx_meas_halves_cps'][1] / dt:.3f}"
              f"  f2 fem/meas {r['f2_arx_fem_cps'] / dt:.3f}/{r['f2_arx_meas_cps'] / dt:.3f} drop {100 * r['f2_drop_vs_first']:5.1f}%")
    print("storey springs:", json.dumps(ss["summary"]))
    print("drift congruence:", json.dumps(dc["summary"]))
    print("shear+flexure:", json.dumps(sf["summary"]))
    print("drift flexibility:", json.dumps(df["summary"]))
    print("shear+flexure, modal fit:", json.dumps(sfm["summary"]))
