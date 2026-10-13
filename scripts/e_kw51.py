"""E-KW51 (pre-registered before any KW51 value was read; answer key results/kw51/answer_key.json): does Z24's frost-memory temperature model
transfer to KW51?

Three nuisance models of log f per tracked mode, fitted exactly as our Z24 v4 temperature model (least squares,
T0 on the grid -3..4 by smallest RSS):
    standard  log f = a
    hinge     log f = a + b T + c min(T - T0, 0)
    frost     log f = a + b T + c min(T - T0, 0) + d F,  F = frost memory (tau 5 d, 30-day window, hourly)
One detector: D2 = r' Sigma^-1 r over the 8 primary modes, Sigma the covariance of the training residuals,
alarm when D2 > 20.09 (chi2_8 at 99%).
Protocol A: leave-one-out within Phase I (every record scored by a model and Sigma refitted without it, T0 included).
Protocol B: four 50-day Phase I blocks, each scored by a model fitted on the other three; the hold-out by all of Phase I.
Retrofit visibility: model fitted on all of Phase I, share of post-retrofit records alarming.

Run from FE/:  python scripts/e_kw51.py  ->  results/e_kw51/{e_kw51.json, d2.npz}
Data: KW51 trackedmodes (Maes & Lombaert, Zenodo 3745914, CC BY-NC-SA 4.0).
"""
import json, os, datetime as dt
from multiprocessing import Pool
import numpy as np, scipy.io as sio
from scipy.stats import beta, chi2

KEY = json.load(open("results/kw51/answer_key.json"))
T0S = np.arange(-3, 4.01, 0.5)
OUT = "results/e_kw51"
KINDS = ("standard", "hinge", "frost")


def load():
    m = sio.loadmat(KEY["data"]["file"], squeeze_me=True, struct_as_record=False)["modes"]
    hours = np.round((m.sdn - m.sdn[0]) * 24).astype(int)
    assert np.array_equal(hours, np.arange(len(hours)))  # regular hourly series
    t0 = dt.datetime.fromordinal(int(m.sdn[0])) - dt.timedelta(days=366)
    assert abs(m.sdn[0] % 1) < 1e-9 and t0 == dt.datetime(2018, 10, 1)
    t = np.array([t0 + dt.timedelta(hours=int(h)) for h in hours])
    lab = list(m.labels_env)
    return dict(t=t, day=np.array([x.date() for x in t]), f=m.f,
                T={"tBD31A": m.env[:, lab.index("tBD31A")], "tVL": m.env[:, lab.index("tVL")]})


def inside(day, pair):
    a, b = [dt.date.fromisoformat(s) for s in pair]
    return (day >= a) & (day <= b)


def frost(T, tau, window):
    """F(t) = (1/24) sum over hours h with 0 <= t - t_h <= window days of max(0, -T_h) e^(-(t - t_h)/tau); missing T adds 0."""
    cold = np.where(np.isnan(T), 0, np.maximum(-T, 0))
    k = np.arange(24 * window + 1)
    return np.convolve(cold, np.exp(-k / 24 / tau))[:len(T)] / 24


def design(kind, T, F, T0):
    cols = [np.ones_like(T)]
    if kind != "standard":
        cols += [T, np.minimum(T - T0, 0)]
    if kind == "frost":
        cols.append(F)
    return np.column_stack(cols)


def fit(kind, T, F, y):
    best = None
    for T0 in (T0S if kind != "standard" else [np.nan]):
        X = design(kind, T, F, T0)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ coef
        if best is None or r @ r < best[0]:
            best = (r @ r, T0, coef)
    return best[1], best[2]


def predict(ms, kind, T, F):
    return np.column_stack([design(kind, T, F, T0) @ c for T0, c in ms])


def train(kind, T, F, Y):
    ms = [fit(kind, T, F, Y[:, j]) for j in range(Y.shape[1])]
    R = Y - predict(ms, kind, T, F)
    return ms, np.linalg.inv(np.cov(R, rowvar=False)), R


def d2(det, kind, T, F, Y):
    R = Y - predict(det[0], kind, T, F)
    return np.einsum("ij,jk,ik->i", R, det[1], R)


def train_nested(T, Fs, Y):
    """tau from the Z24 grid by summed log residual SD over modes (Z24 v4's rule), refitted on these records only."""
    best = None
    for tau, F in Fs.items():
        det = train("frost", T, F, Y)
        score = np.sum(np.log(np.std(det[2], axis=0)))
        if best is None or score < best[0]:
            best = (score, tau, det)
    return best[1], best[2]


def setup(cfg, d):
    T = d["T"][cfg["channel"]]
    cols = np.array(cfg["modes"]) - 1
    f = d["f"][:, cols]
    if cfg.get("fill") == "interp":  # Neumann 2024's preprocessing: missing frequencies linearly interpolated in time
        h = np.arange(len(f))
        f = np.column_stack([np.interp(h, h[np.isfinite(c)], c[np.isfinite(c)]) for c in f.T])
    ok = np.all(np.isfinite(f), axis=1) & np.isfinite(T)
    taus = KEY["frost"]["nested_tau_grid"] if cfg["tau"] == "nested" else [cfg["tau"]]
    Fs = {tau: frost(T, tau, cfg["window"]) for tau in taus}
    return T, np.log(f), ok, Fs


def run(task):
    cfg, proto = task
    d = load()
    T, Y, ok, Fs = setup(cfg, d)
    kind = cfg["kind"]
    F = next(iter(Fs.values()))
    ph1 = ok & inside(d["day"], KEY["periods"]["phase_I"])
    score, info = np.full(len(T), np.nan), {}
    if proto == "A":
        idx = np.where(ph1)[0]
        for i in idx:
            tr = ph1.copy(); tr[i] = False
            det = train(kind, T[tr], F[tr], Y[tr])
            score[i] = d2(det, kind, T[i:i + 1], F[i:i + 1], Y[i:i + 1])[0]
    elif proto == "B":
        taus = []
        for blk in KEY["protocol_B_blocks"]:
            te = ok & inside(d["day"], blk)
            tr = ph1 & ~te
            if cfg["tau"] == "nested":
                tau, det = train_nested(T[tr], {k: v[tr] for k, v in Fs.items()}, Y[tr])
                F = Fs[tau]; taus.append(tau)
            else:
                det = train(kind, T[tr], F[tr], Y[tr])
            score[te] = d2(det, kind, T[te], F[te], Y[te])
        if cfg["tau"] == "nested":
            info["tau_per_block"] = taus
            tau, det = train_nested(T[ph1], {k: v[ph1] for k, v in Fs.items()}, Y[ph1])
            F = Fs[tau]; info["tau_all_phase_I"] = tau
        else:
            det = train(kind, T[ph1], F[ph1], Y[ph1])
        for name in ("healthy_holdout", "post_retrofit"):
            te = ok & inside(d["day"], KEY["periods"][name])
            score[te] = d2(det, kind, T[te], F[te], Y[te])
        info["fit_all_phase_I"] = [dict(mode=int(m), T0=None if np.isnan(T0) else float(T0), coef=[float(c) for c in co])
                                   for m, (T0, co) in zip(cfg["modes"], det[0])]
        info["resid_sd_pct"] = [float(100 * s) for s in np.std(det[2], axis=0)]
    return cfg["name"], proto, score, info


def rate(alarm, day):
    """Rate with a one-sided 95% Clopper-Pearson upper bound whose sample size is the number of distinct days."""
    n, k = len(alarm), int(alarm.sum())
    if n == 0:
        return dict(n=0, alarms=0, rate=None, days=0, upper95=None)
    nd = len(set(day)); ke = k / n * nd
    return dict(n=n, alarms=k, rate=k / n, days=nd, upper95=float(beta.ppf(0.95, ke + 1, nd - ke)) if ke < nd else 1.0)


def split(score, T, day, sel, limit):
    s = sel & np.isfinite(score)
    a = score > limit
    return {g: rate(a[m], day[m]) for g, m in (("below_2C", s & (T < 2)), ("above_2C", s & (T >= 2)), ("all", s))}


def main():
    os.makedirs(OUT, exist_ok=True)
    P, S = KEY["modes_primary"], KEY["modes_secondary"]
    base = dict(modes=P, channel=KEY["data"]["temperature_channel"], tau=KEY["frost"]["tau_days"], window=KEY["frost"]["window_days"])
    cfgs = [dict(base, name=k, kind=k) for k in KINDS]
    sens = [dict(base, name=f"frost_tau{t}", kind="frost", tau=t) for t in KEY["frost"]["sensitivity_tau_days"]]
    sens += [dict(base, name=f"frost_window{w}", kind="frost", window=w) for w in KEY["frost"]["sensitivity_window_days"]]
    sens += [dict(base, name=f"{k}_tVL", kind=k, channel=KEY["data"]["temperature_sensitivity_channel"]) for k in KINDS]
    sens += [dict(base, name=f"{k}_wittenberg", kind=k, modes=S) for k in KINDS]
    # Not pre-registered: reproduces Neumann's preprocessing to explain the sanity-check gap (C-PASS §2.5)
    sens += [dict(base, name=f"{k}_interp", kind=k, fill="interp") for k in KINDS]
    tasks = [(c, p) for c in cfgs + sens for p in "AB"]
    tasks += [(dict(base, name="frost_nested_tau", kind="frost", tau="nested"), "B")]
    with Pool(min(len(tasks), int(os.environ.get("SLURM_CPUS_ON_NODE", 4)))) as pool:
        res = pool.map(run, tasks, chunksize=1)
    d = load()
    cfg_of = {c["name"]: c for c, _ in tasks}
    per = {k: inside(d["day"], v) for k, v in KEY["periods"].items()}
    blocks = np.any([inside(d["day"], b) for b in KEY["protocol_B_blocks"]], axis=0)
    out, arrays = dict(primary=[c["name"] for c in cfgs], configs={}), {}
    for name, proto, score, info in res:
        c = cfg_of[name]
        T, Y, ok, _ = setup(c, d)
        limit = float(chi2.ppf(0.99, len(c["modes"])))
        if len(c["modes"]) == 8:
            assert abs(limit - KEY["alarm"]["limit"]) < 0.005
        r = dict(info, limit=limit, usable={k: [int((ok & v).sum()), int(v.sum())] for k, v in per.items()})
        if proto == "A":
            r["phase_I"] = split(score, T, d["day"], per["phase_I"], limit)
        else:
            r["blocks"] = split(score, T, d["day"], blocks, limit)
            r["holdout"] = split(score, T, d["day"], per["healthy_holdout"], limit)
            r["pooled"] = split(score, T, d["day"], blocks | per["healthy_holdout"], limit)
            r["cold_spell"] = split(score, T, d["day"], per["cold_spell"], limit)["all"]
            r["post_retrofit"] = split(score, T, d["day"], per["post_retrofit"], limit)["all"]
        out["configs"].setdefault(name, dict(cfg={k: v for k, v in c.items() if k != "name"}))[proto] = r
        arrays[f"{name}_{proto}"] = score
    np.savez_compressed(f"{OUT}/d2.npz", hours_from_2018_10_01=np.arange(len(d["t"])), **arrays)

    # Pre-registered outcome (C-PASS §2.5), on the primary configs only
    C, E = out["configs"], KEY["e_kw51"]
    fa = lambda k, p: C[k][p]["phase_I" if p == "A" else "pooled"]["below_2C"]["rate"]
    c1 = all(fa("frost", p) <= E["confirm"]["false_alarm_below_2C_max"] for p in "AB")
    c2 = C["frost"]["B"]["cold_spell"]["alarms"] < C["hinge"]["B"]["cold_spell"]["alarms"]
    c3 = C["frost"]["B"]["post_retrofit"]["rate"] >= E["confirm"]["post_retrofit_alarm_share_min"]
    gain = 100 * (fa("hinge", "B") - fa("frost", "B"))
    if c1 and c2 and c3:
        verdict = "confirm"
    elif gain >= E["partial_min_gain_points"] and fa("frost", "B") > E["confirm"]["false_alarm_below_2C_max"]:
        verdict = "partial"
    elif gain < E["kill_gain_below_points"]:
        verdict = "kill"
    else:
        verdict = "not covered by the pre-registered rules"
    out["outcome"] = dict(c1_frost_below2C_le_5_8pct_A_and_B=c1, c2_cold_spell_frost_fewer_than_hinge_B=c2,
                          c3_post_retrofit_share_ge_50pct=c3, gain_vs_hinge_B_points=gain, verdict=verdict,
                          frost_adopted=bool(c3 and verdict in ("confirm", "partial")),
                          sanity_standard_A_below_2C=fa("standard", "A"), sanity_reference_neumann_below_2C=0.38,
                          sanity_standard_A_below_2C_interp=fa("standard_interp", "A"))
    json.dump(out, open(f"{OUT}/e_kw51.json", "w"), indent=1)

    pct = lambda x: "  -  " if x is None else f"{100 * x:5.1f}"
    print(f"{'config':22s} | A: <2C  >2C  all | B: <2C  >2C  all | hold | cold(n)   | post-retro")
    for name, r in C.items():
        a = r.get("A", {}).get("phase_I"); b = r.get("B", {})
        print(f"{name:22s} | " + (" ".join(pct(a[g]["rate"]) for g in ("below_2C", "above_2C", "all")) if a else " " * 17) + " | "
              + " ".join(pct(b["pooled"][g]["rate"]) for g in ("below_2C", "above_2C", "all")) + f" | {pct(b['holdout']['all']['rate'])}"
              + f" | {b['cold_spell']['alarms']:3d}/{b['cold_spell']['n']:<4d} | {pct(b['post_retrofit']['rate'])}")
    print("usable records (primary):", C["frost"]["B"]["usable"])
    print("outcome:", json.dumps(out["outcome"]))


if __name__ == "__main__":
    main()
