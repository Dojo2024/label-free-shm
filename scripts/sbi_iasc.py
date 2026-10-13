"""SBI prototype on the IASC-ASCE 12DOF benchmark: neural posterior estimation
of brace damage from noisy modal features, validated on the published damage
patterns where the truth is known.

theta: 16 remaining-stiffness fractions, one per (storey, face). In the 12DOF
model both braces on a face add identical stiffness, so a single broken brace
(pattern iii) is exactly that face at 0.5.
x: 12 frequencies (0.5% noise) + the first N_SHAPES mode shapes (unit norm,
sign-fixed, 0.01 additive noise).

    python scripts/sbi_iasc.py --n-sims 20000          # workers = SLURM_CPUS_PER_TASK
    python scripts/sbi_iasc.py --n-sims 300 --smoke    # correctness check only
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from sbi.inference import NPE
from torch.distributions import Independent, SigmoidTransform, TransformedDistribution, Uniform

import iasc_asce as ia

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sbi_iasc"   # output/ is outside the sparse checkout
FACES = ["W", "E", "N", "S"]
THETA_NAMES = [f"S{s}-{f}" for s in range(1, ia.N_FLOORS + 1) for f in FACES]
N_SHAPES = 6
FREQ_NOISE = 0.005
SHAPE_NOISE = 0.01


def theta_to_braces(theta):
    return {b: float(theta[THETA_NAMES.index(b[:4])]) for b in ia.BRACE_IDS}


def modal_features(M, K):
    lam, phi = scipy.linalg.eigh(K, M)
    f = np.sqrt(lam) / (2 * np.pi)
    shapes = phi[:, :N_SHAPES] / np.linalg.norm(phi[:, :N_SHAPES], axis=0)
    idx = np.abs(shapes).argmax(axis=0)
    shapes *= np.sign(shapes[idx, np.arange(N_SHAPES)])
    return np.concatenate([np.log(f), shapes.T.ravel()])


def simulate_clean(theta):
    return modal_features(*ia.build_12dof(braces=theta_to_braces(theta)))


def add_noise(x, rng):
    x = x.copy()
    x[:12] += rng.normal(0, FREQ_NOISE, 12)          # log-freq: ~relative noise
    x[12:] += rng.normal(0, SHAPE_NOISE, x.size - 12)
    return x


def pattern_theta(name):
    """Face-level ground truth of a published pattern (face = mean of its 2 braces)."""
    braces = {b: 1.0 for b in ia.BRACE_IDS} | ia.DAMAGE_PATTERNS[name]["braces"]
    return np.array([np.mean([braces[b] for b in ia.BRACE_IDS if b[:4] == t])
                     for t in THETA_NAMES])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    torch.set_num_threads(workers)
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)

    n = args.n_sims + args.n_test
    theta = rng.uniform(0, 1, (n, 16))
    t0 = time.time()
    with Pool(workers) as pool:
        x_clean = np.array(pool.map(simulate_clean, theta, chunksize=64))
    x = np.array([add_noise(xi, rng) for xi in x_clean])
    print(f"simulated {n} in {time.time() - t0:.0f}s with {workers} workers", flush=True)

    # Train in logit space: a Uniform(0,1) box prior makes sbi rejection-sample
    # posterior draws that land outside the box, which stalls when the net is
    # imperfect. The logit of a uniform is unbounded, so nothing is rejected.
    prior = Independent(TransformedDistribution(
        Uniform(torch.zeros(16), torch.ones(16)), SigmoidTransform().inv), 1)
    npe = NPE(prior=prior)
    npe.append_simulations(torch.logit(torch.tensor(theta[:args.n_sims], dtype=torch.float32)),
                           torch.tensor(x[:args.n_sims], dtype=torch.float32))
    t0 = time.time()
    est = npe.train(max_num_epochs=20 if args.smoke else 500, show_train_summary=True)
    print(f"trained in {time.time() - t0:.0f}s", flush=True)
    posterior = npe.build_posterior(est)

    def sample(xo, k=2000):
        return torch.sigmoid(posterior.sample((k,), x=torch.tensor(xo, dtype=torch.float32),
                                              show_progress_bars=False)).numpy()

    # calibration on held-out prior draws: 90% central-interval coverage per param
    th_te, x_te = theta[args.n_sims:], x[args.n_sims:]
    inside = []
    for th, xo in zip(th_te, x_te):
        s = sample(xo, 1000)
        lo, hi = np.quantile(s, [0.05, 0.95], axis=0)
        inside.append((th >= lo) & (th <= hi))
    coverage = np.mean(inside, axis=0)
    print(f"90% coverage on {len(th_te)} held-out: mean {coverage.mean():.3f} "
          f"(min {coverage.min():.3f}, max {coverage.max():.3f})", flush=True)

    # the published damage patterns, observed via the benchmark's own build (not our theta map)
    results = {"coverage_90": dict(zip(THETA_NAMES, coverage.round(3).tolist())), "patterns": {}}
    for name in ["i", "ii", "iii", "iv", "vi"]:
        truth = pattern_theta(name)
        xo = add_noise(modal_features(*ia.build_12dof(name)), rng)
        s = sample(xo)
        mean, sd = s.mean(0), s.std(0)
        err = np.abs(mean - truth)
        print(f"\npattern ({name}): mean |err| {err.mean():.3f}, max |err| {err.max():.3f}")
        for i in np.argsort(-np.abs(truth - 1) - err)[:4]:
            print(f"  {THETA_NAMES[i]}: truth {truth[i]:.2f}  post {mean[i]:.2f} +- {sd[i]:.2f}")
        results["patterns"][name] = dict(truth=truth.round(3).tolist(),
                                          post_mean=mean.round(3).tolist(),
                                          post_sd=sd.round(3).tolist())

    tag = "smoke" if args.smoke else f"n{args.n_sims}"
    (OUT / f"results_{tag}.json").write_text(json.dumps(results, indent=1))
    if not args.smoke:
        torch.save(est.state_dict(), OUT / f"npe_{tag}.pt")
    print(f"\nwrote {OUT}/results_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
