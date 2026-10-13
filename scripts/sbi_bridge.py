"""Bridge SBI: neural posterior estimation of zone-level stiffness loss on the
calibrated Shin-Nakagawa model, from noisy modal features of a healthy baseline
and a current state -- the way an SHM system actually sees the bridge.

theta: remaining-stiffness multiplier for each of fe_model.ZONE_NAMES (19 zones:
5 bearings, 4 piers, 6 girder zones, 4 cable groups). Sparse prior: each zone is
damaged with probability P_DAMAGED (multiplier ~ U(M_MIN, 1)), else near-intact
(~ U(0.98, 1)).
Nuisance, sampled and marginalized, never inferred:
  - model parameters calibration could not pin down, log-uniform around the
    calibrated value (x/÷3 for the unidentified ones, x/÷1.15 for the identified);
    shared by baseline and current state, since it is the same bridge;
  - environment (temperature), drawn independently for baseline and current:
    all bearing stiffnesses x/÷1.3 (rubber), deck E +-3%.
x: for baseline and current, the 8 modes the field identification found (paper
Fig. 12: 1GVe 1GLo 1GLa 2GVe 3GVe 4GVe 2GLa 5GVe), each paired one-to-one
(Hungarian on (1 - MAC) + |log f/f_ref|, MAC at the sensor DOFs, over the modes
below F_BAND) with the calibrated healthy model's version of it. MAC alone
mispaired 27% of nuisance draws: 12 sensor nodes alias 3-5 Hz modes onto the
low ones, and one-to-one assignment pushes a loser onto a distant mode: log-frequencies (1% noise) + mode shapes at the 12
distinct sensor nodes x 3 directions (unit norm, sign aligned with the reference,
SHAPE_NOISE additive noise), then current - baseline appended (with_diff): the net
could learn the difference itself, but handing it over separates damage from the
shared nuisance directly. Neither frequency order nor a direction label is
stable under the nuisance prior -- lateral modes drop below vertical ones, and the
first modes are ~64/36 vertical/longitudinal, so direction labels flip -- so modes
are identified by shape, as field identification does. The FE-only lateral modes
(0.724 pylon sway, 0.901, 1.121 Hz) are left out: they hybridize under the nuisance
prior (10th-pct matched MAC 0.15-0.57 vs >= 0.76 for the 8 kept) and the real
bridge's identification never sees them anyway.

--prior hier adds widespread, all-of-a-type degradation to the sparse prior.
--temp-noise appends a temperature reading of the bearing state (baseline, current)
with that much error; one model is trained per value, on shared cached sims.

    python scripts/sbi_bridge.py --n-sims 20000        # workers = SLURM_CPUS_PER_TASK
    python scripts/sbi_bridge.py --n-sims 200000 --prior hier --temp-noise none 0.1 0.3
    python scripts/sbi_bridge.py --n-sims 500 --smoke  # correctness check only
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
from functools import cache
from pathlib import Path

import numpy as np
import openseespy.opensees as ops
import torch
from scipy.optimize import linear_sum_assignment
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding
from torch.distributions import Independent, SigmoidTransform, TransformedDistribution, Uniform

import fe_model as fm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sbi_bridge"
CACHE = OUT / "cache"  # training sims, ~1 GB per 200k; gitignored, regenerable
CALIBRATED = json.loads((ROOT / "calibrated_params.json").read_text())
ZONES = fm.ZONE_NAMES
REF_MODES = [1, 2, 3, 4, 7, 9, 10, 11]  # healthy-model mode numbers of the paper's 8 (0.377-1.599 Hz)
N_EIGEN = 24  # candidate pool the reference modes are paired from ...
F_BAND = 2.5  # ... restricted to the identification band, Hz
N_FREQS = len(REF_MODES)
FREQ_NOISE, SHAPE_NOISE = 0.01, 0.03
P_DAMAGED, M_MIN, DAMAGE_THRESHOLD = 0.15, 0.2, 0.9
P_TYPE_WIDE, TYPE_JITTER = 0.1, 0.05
ZONE_TYPE = np.array([z[0] for z in ZONES])  # B(earing) P(ier) G(irder) C(able)
PRIORS = ["sparse", "hier"]

# (param, x/÷ factor). Wide = not identified by the 8-frequency calibration or sat
# at a search bound (see calibrate.py); narrow = identified.
NUISANCE = [("deck_Iy", 3.0), ("bearing_k_lat", 3.0), ("pylon_bearing_k_lat", 3.0),
            ("pylon_Iy", 3.0), ("pylon_Iz", 3.0), ("pylon_mass_per_len", 3.0),
            ("pier_height", 3.0), ("bearing_k_long", 3.0),
            ("deck_Iz", 1.15), ("deck_mass_per_len", 1.15), ("pylon_bearing_k_long", 1.15),
            ("cable_A", 1.15), ("back_anchor_start", 1.15), ("main_anchor_end", 1.05)]
BEARING_KEYS = ["bearing_k_long", "bearing_k_lat", "pylon_bearing_k_long", "pylon_bearing_k_lat"]
ENV_BEARING, ENV_DECK_E = 1.3, 0.03


def meas_nodes(model):
    """Distinct sensor nodes (several sensors share a node), in a fixed order."""
    return sorted(set(fm.sensor_output_nodes(model).values()))


def sensor_shapes(model, n):
    """Mode shapes 1..n at the sensor DOFs, unit norm, shape (n, 36)."""
    nodes = meas_nodes(model)
    sh = np.array([[ops.nodeEigenvector(t, i)[:3] for t in nodes] for i in range(1, n + 1)]).reshape(n, -1)
    return sh / np.linalg.norm(sh, axis=1, keepdims=True)


@cache
def reference_shapes():
    m = fm.build_model(CALIBRATED)
    lam = np.array(ops.eigen(max(REF_MODES)))[np.array(REF_MODES) - 1]
    return (sensor_shapes(m, max(REF_MODES))[np.array(REF_MODES) - 1],
            np.log(np.sqrt(lam) / (2 * np.pi)))


def modal_features(model):
    """Needs reference_shapes() already cached: computing it rebuilds the model."""
    ref, logf_ref = reference_shapes()
    lam = np.array(ops.eigen(N_EIGEN))
    if np.any(lam <= 0):
        raise ValueError("non-positive eigenvalue")
    logf = np.log(np.sqrt(lam) / (2 * np.pi))
    n_band = int((logf < np.log(F_BAND)).sum())
    sh = sensor_shapes(model, n_band)
    cost = 1 - (ref @ sh.T) ** 2 + np.abs(logf_ref[:, None] - logf[None, :n_band])  # MAC of unit vectors = dot^2
    _, pick = linear_sum_assignment(cost)
    sh = sh[pick] * np.sign(np.sum(ref * sh[pick], axis=1))[:, None]
    return np.concatenate([logf[pick], sh.ravel()])


def sample_damage(rng, prior="sparse"):
    """sparse: each zone damaged independently w.p. P_DAMAGED. hier adds, per member type
    and w.p. P_TYPE_WIDE, one shared degradation factor with a small per-zone spread (all
    bearings ageing together). Under sparse alone, all-of-a-type damage has ~1e-4 mass:
    the cause of the confident case 3 / case 7 misses."""
    damaged = rng.random(len(ZONES)) < P_DAMAGED
    m = np.where(damaged, rng.uniform(M_MIN, 1.0, len(ZONES)), rng.uniform(0.98, 1.0, len(ZONES)))
    if prior == "hier":
        for t in "BPGC":
            i = ZONE_TYPE == t
            if rng.random() < P_TYPE_WIDE:
                m[i] *= rng.uniform(M_MIN, 1.0) * np.exp(rng.normal(0, TYPE_JITTER, i.sum()))
        m = np.clip(m, M_MIN, 1.0)
    return m


def sample_nuisance(rng):
    return {k: CALIBRATED[k] * f ** rng.uniform(-1, 1) for k, f in NUISANCE}


def sample_env(rng):
    return dict(bearing=ENV_BEARING ** rng.uniform(-1, 1), deck_E=1 + rng.uniform(-ENV_DECK_E, ENV_DECK_E))


def temp_state(env):
    """Bearing temperature state on [-1, 1]: what a temperature reading tells the net
    before its own error. Only the bearings; the deck-E share stays unobserved."""
    return np.log(env["bearing"]) / np.log(ENV_BEARING)


def state_params(nuis, env):
    p = dict(CALIBRATED) | nuis
    for k in BEARING_KEYS:
        p[k] *= env["bearing"]
    p["deck_E"] *= env["deck_E"]
    return p


def simulate_clean(args):
    """(damage, nuisance, env_base, env_cur, damage_case) -> baseline ++ current
    features, or None if the eigensolve fails."""
    damage, nuis, env_b, env_c, damage_case = args
    reference_shapes()
    try:
        base = modal_features(fm.build_model(state_params(nuis, env_b)))
        cur = modal_features(fm.build_model(state_params(nuis, env_c), damage_case=damage_case,
                                            zone_mult=dict(zip(ZONES, damage))))
    except Exception:
        return None
    return np.concatenate([base, cur])


def add_noise(x, rng):
    x = x.copy()
    half = x.size // 2
    for s in (0, half):
        x[s:s + N_FREQS] += rng.normal(0, FREQ_NOISE, N_FREQS)
        x[s + N_FREQS:s + half] += rng.normal(0, SHAPE_NOISE, half - N_FREQS)
    return x


def with_diff(x):
    """[baseline, current] -> [baseline, current, current - baseline], on noisy x."""
    half = x.shape[-1] // 2
    return np.concatenate([x, x[..., half:] - x[..., :half]], axis=-1)


def run_set(jobs, pool):
    """(damage, nuisance, env_b, env_c, case) jobs -> theta, clean x, true temperature
    state (baseline, current); failed eigensolves dropped."""
    xs = pool.map(simulate_clean, jobs, chunksize=32)
    keep = [j for j, x in zip(jobs, xs) if x is not None]
    return (np.array([j[0] for j in keep]), np.array([x for x in xs if x is not None]),
            np.array([[temp_state(j[2]), temp_state(j[3])] for j in keep]))


def observe(x, tau, rng):
    """Clean modal x -> noisy net input, plus one N(0,1) draw per temperature reading,
    fixed here so every temperature variant sees the same reading errors."""
    return with_diff(np.array([add_noise(xi, rng) for xi in x])), rng.standard_normal(tau.shape)


def with_temp(xn, tau, z, temp_noise):
    """Append the temperature readings (baseline, current, current - baseline)."""
    if temp_noise is None:
        return xn
    t = tau + temp_noise * z
    return np.concatenate([xn, t, t[:, 1:] - t[:, :1]], axis=1)


def to_u(m):   # multiplier in (M_MIN, 1) -> (0, 1); the net works on logit(u)
    return np.clip((m - M_MIN) / (1 - M_MIN), 1e-4, 1 - 1e-4)


def case_truth(damage_case):
    """Zone-level truth for a named Table 3 case: bearings/piers exact; girder zones
    the length-weighted mean multiplier, cable groups the mean over their cables.
    A partial-zone loss is not the same structure as a uniform one, so this truth
    is an approximation for cases 2, 4 and 8."""
    specs = fm.resolve_damage_specs(damage_case)
    m = fm.build_model(CALIBRATED)
    t = {f"B-{n}": fm._bearing_multiplier(n, specs) for n in ["P31", "P32", "P33", "PYLON", "P35"]}
    t |= {f"P-{n}": fm._pier_multiplier(n, specs) for n in ["P31", "P32", "P33", "P35"]}
    for z in fm.GIRDER_ZONES:
        els = [(x1 - x0, fm._near_multiplier((x0 + x1) / 2, "girder", specs))
               for _, x0, x1 in m["girder_elements"] if fm.zone_of((x0 + x1) / 2, fm.GIRDER_ZONES) == z]
        t[z] = sum(L * g for L, g in els) / sum(L for L, _ in els)
    for z in fm.CABLE_ZONES:
        t[z] = np.mean([fm._near_multiplier(ax, "cable", specs)
                        for _, ax in m["cable_elements"] if fm.zone_of(ax, fm.CABLE_ZONES) == z])
    return np.array([t[z] for z in ZONES])


def auroc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = label.sum()
    return (ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(label) - n1))


def train_npe(theta, x, smoke):
    """-> (sample(xo, k) in multiplier units, density estimator)."""
    # logit space, as in sbi_iasc.py: unbounded, so posterior sampling never rejects.
    # The training thetas come from sample_damage's prior, not this uniform; single-round
    # NPE learns the posterior under the distribution it was trained on, and this
    # object only sets the support.
    prior = Independent(TransformedDistribution(
        Uniform(torch.zeros(len(ZONES)), torch.ones(len(ZONES))), SigmoidTransform().inv), 1)
    emb = FCEmbedding(input_dim=x.shape[1], output_dim=64, num_layers=3, num_hiddens=256)
    npe = NPE(prior=prior, density_estimator=posterior_nn("maf", embedding_net=emb,
                                                          hidden_features=100, num_transforms=5))
    npe.append_simulations(torch.logit(torch.tensor(to_u(theta), dtype=torch.float32)),
                           torch.tensor(x, dtype=torch.float32))
    t0 = time.time()
    est = npe.train(max_num_epochs=20 if smoke else 500, show_train_summary=True)
    print(f"trained in {time.time() - t0:.0f}s", flush=True)
    posterior = npe.build_posterior(est)

    def sample(xo, k=2000):
        u = torch.sigmoid(posterior.sample((k,), x=torch.tensor(xo, dtype=torch.float32),
                                           show_progress_bars=False)).numpy()
        return M_MIN + (1 - M_MIN) * u
    return sample, est


def score_test(sample, theta, x):
    """Held-out prior draws: 90% coverage and detection AUROC (P(m<0.9) vs truth) per zone."""
    inside, p_dmg = [], []
    for th, xo in zip(theta, x):
        s = sample(xo, 1000)
        lo, hi = np.quantile(s, [0.05, 0.95], axis=0)
        inside.append((th >= lo) & (th <= hi))
        p_dmg.append((s < DAMAGE_THRESHOLD).mean(0))
    coverage, p_dmg, is_dmg = np.mean(inside, axis=0), np.array(p_dmg), theta < DAMAGE_THRESHOLD
    auc = np.array([auroc(p_dmg[:, i], is_dmg[:, i]) if 0 < is_dmg[:, i].sum() < len(theta) else np.nan
                    for i in range(len(ZONES))])
    return dict(coverage_90=dict(zip(ZONES, coverage.round(3).tolist())),
                auroc={z: None if np.isnan(a) else round(float(a), 3) for z, a in zip(ZONES, auc)},
                by_type={t: dict(coverage_90=round(float(coverage[ZONE_TYPE == t].mean()), 3),
                                 auroc=round(float(np.nanmean(auc[ZONE_TYPE == t])), 3)) for t in "BPGC"})


def score_case(sample, truth, x):
    """Repeated draws of one case: per-zone posterior mean, P(damaged), and how often the
    truth falls inside the 90% interval (low on a damaged zone = confidently wrong)."""
    mean, pd, inside = [], [], []
    for xo in x:
        s = sample(xo)
        lo, hi = np.quantile(s, [0.05, 0.95], axis=0)
        mean.append(s.mean(0))
        pd.append((s < DAMAGE_THRESHOLD).mean(0))
        inside.append((truth >= lo) & (truth <= hi))
    return dict(truth=truth.round(3).tolist(), post_mean=np.mean(mean, 0).round(3).tolist(),
                p_damaged=np.mean(pd, 0).round(3).tolist(), truth_in_90=np.mean(inside, 0).round(3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=500, help="held-out draws per prior")
    ap.add_argument("--n-repeat", type=int, default=20, help="draws per Table 3 case and condition")
    ap.add_argument("--prior", choices=PRIORS, default="sparse")
    ap.add_argument("--temp-noise", nargs="+", default=["none"],
                    help="one model per value: 'none' = no temperature input, else the s.d. of the "
                         "reading's error on temp_state's [-1, 1] scale")
    ap.add_argument("--threads", type=int, help="torch threads for training (default: workers)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    torch.set_num_threads(args.threads or workers)
    CACHE.mkdir(parents=True, exist_ok=True)

    # Training sims are cached per (prior, n): the temperature variants share them. Test
    # sets and case draws have their own fixed seeds, so every run -- either prior, any
    # temperature variant -- is scored on the same bridges with the same noise.
    t0 = time.time()
    nominal = dict(bearing=1.0, deck_E=1.0)
    with Pool(workers) as pool:
        f = CACHE / f"sims_{args.prior}_n{args.n_sims}.npz"
        if f.exists():
            d = np.load(f)
            train = d["theta"], d["x"], d["tau"]
        else:
            rng = np.random.default_rng(0)
            train = run_set([(sample_damage(rng, args.prior), sample_nuisance(rng), sample_env(rng),
                              sample_env(rng), None) for _ in range(args.n_sims)], pool)
            np.savez(f, theta=train[0], x=train[1], tau=train[2])
        print(f"{len(train[0])}/{args.n_sims} training sims ({args.prior}) in {time.time() - t0:.0f}s "
              f"with {workers} workers", flush=True)
        tests = {}
        for i, p in enumerate(PRIORS):
            rng = np.random.default_rng(1 + i)
            tests[p] = run_set([(sample_damage(rng, p), sample_nuisance(rng), sample_env(rng),
                                 sample_env(rng), None) for _ in range(args.n_test)], pool)
        # nominal: calibrated params and environment, noise only (the old single-draw check);
        # random: nuisance and both environments drawn, as a real bridge would present it
        rng = np.random.default_rng(3)
        ones = np.ones(len(ZONES))
        cases = {}
        for cond in ["nominal", "random"]:
            for c in [None] + list(fm.DAMAGE_TARGETS):
                jobs = [(ones, {}, nominal, nominal, c) if cond == "nominal" else
                        (ones, sample_nuisance(rng), sample_env(rng), sample_env(rng), c)
                        for _ in range(args.n_repeat)]
                cases[cond, c or "healthy"] = (ones if c is None else case_truth(c),) + run_set(jobs, pool)[1:]
    print(f"all sims done in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(10)
    obs_train = observe(train[1], train[2], rng)
    obs_test = {p: observe(t[1], t[2], rng) for p, t in tests.items()}
    obs_case = {k: observe(v[1], v[2], rng) for k, v in cases.items()}

    for tn in args.temp_noise:
        s = None if tn == "none" else float(tn)
        tag = f"{args.prior}_{'notemp' if s is None else 'temp' + tn}_{'smoke' if args.smoke else f'n{args.n_sims}'}"
        print(f"\n===== {tag}", flush=True)
        torch.manual_seed(0)
        sample, est = train_npe(train[0], with_temp(obs_train[0], train[2], obs_train[1], s), args.smoke)
        res = dict(prior=args.prior, temp_noise=s, n_sims=len(train[0]), zones=ZONES, test={}, cases={})
        for p, (th, _, tau) in tests.items():
            r = res["test"][p] = score_test(sample, th, with_temp(obs_test[p][0], tau, obs_test[p][1], s))
            print(f"test[{p}] by type (90% coverage / AUROC): "
                  + "  ".join(f"{t} {v['coverage_90']:.3f}/{v['auroc']:.3f}" for t, v in r["by_type"].items()),
                  flush=True)
        for (cond, c), (truth, _, tau) in cases.items():
            r = score_case(sample, truth, with_temp(obs_case[cond, c][0], tau, obs_case[cond, c][1], s))
            res["cases"][f"{cond}/{c}"] = r
            dmg = truth < DAMAGE_THRESHOLD
            pd, in90 = np.array(r["p_damaged"]), np.array(r["truth_in_90"])
            line = f"{cond:7s} {c:28s}"
            if (~dmg).any():
                line += f" max P(dmg) on intact {pd[~dmg].max():.2f} ({ZONES[np.argmax(np.where(dmg, -1, pd))]})"
            if dmg.any():
                worst = np.flatnonzero(dmg)[np.argmin(in90[dmg])]
                line += (f" | damaged zones: mean P(dmg) {pd[dmg].mean():.2f}, truth in 90% "
                         f"{in90[dmg].mean():.2f} (worst {ZONES[worst]} {in90[worst]:.2f}: truth "
                         f"{truth[worst]:.2f} post {r['post_mean'][worst]:.2f})")
            print(line, flush=True)
        (OUT / f"results_{tag}.json").write_text(json.dumps(res, indent=1))
        if not args.smoke:
            torch.save(est.state_dict(), OUT / f"npe_{tag}.pt")
        print(f"wrote {OUT}/results_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
