"""
Fit the free structural parameters in fe_model.DEFAULT_PARAMS against the eight
field-identified girder modes in Siringoringo et al. 2022 (J. Bridge Eng.,
DOI 10.1061/(ASCE)BE.1943-5592.0001813), Fig. 12. This is
standard FE model updating: since we don't have the bridge's real published
section properties (paywalled), we treat a handful of parameters as unknown
and solve for the values that reproduce the *measured* modal behavior.

Why every parameter has a physical [lo, hi] box (2026-09-27): natural
frequencies depend only on stiffness/mass ratios, so with masses AND
stiffnesses both free, scaling them up together leaves every frequency
unchanged. The earlier +-1-decade search exploited exactly that null
direction and drifted to a 157 t/m deck (x8.7) and a 153 t/m pylon. Masses
are the quantities we can estimate from geometry, so their boxes are tight;
stiffnesses are the genuinely uncertain part and get wider boxes. Bound
sources are in BOUNDS_PROVENANCE below.

Each build+eigen(25 modes) call costs ~8ms, so Nelder-Mead over ~9 params
runs in seconds single-threaded.
"""
import sys
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fe_model import build_model, classify_modes, DEFAULT_PARAMS

# Siringoringo et al. 2022, Fig. 12: mean SRIM/M-ERA frequencies over the 11
# largest + 3 far-field earthquakes (1GVe, 1GLo, 1GLa, 2GVe, 3GVe, 4GVe, 2GLa, 5GVe).
# These replace the write-up's Table 2 values (0.356/0.435/0.462/0.748/1.046/1.270),
# whose provenance was unknown and which don't match the paper. Their own FE model
# (TDAP) was within ~10% of these on average -- a realistic accuracy bar.
TARGETS = [
    ("vertical", 0.38),
    ("longitudinal", 0.42),
    ("lateral", 0.51),
    ("vertical", 0.72),
    ("vertical", 1.08),
    ("vertical", 1.29),
    ("lateral", 1.43),
    ("vertical", 1.62),
]

# Free parameters: (name, lo, hi) physical bounds; search is in log-space.
FREE_PARAMS = [
    ("deck_Iy", 20.0, 120.0),          # lateral bending, 22.9m-wide box
    ("deck_Iz", 1.0, 8.0),             # vertical bending, ~3m-deep box
    ("deck_mass_per_len", 8.0, 14.0),  # 3187t steel / 533m = 6.0 t/m + paving/barriers
    ("pylon_Iy", 20.0, 500.0),
    ("pylon_Iz", 20.0, 500.0),
    ("pylon_mass_per_len", 5.0, 40.0),   # steel tower above deck (paper)
    ("bearing_k_long", 10.0, 1e5),       # down to ~free: paper says supports "accommodate free longitudinal movement"
    ("bearing_k_lat", 1e3, 1e6),          # wind shoes limit lateral motion
    ("pylon_bearing_k_long", 1e3, 1e6),
    ("pylon_bearing_k_lat", 1e3, 1e6),
    ("cable_A", 0.003, 0.05),
    ("pier_height", 10.0, 40.0),         # uniform; real per-pier heights unknown
    ("back_anchor_start", 5.0, 200.0),   # back-span fan extent (not in paper)
    ("main_anchor_end", 350.0, 525.0),   # main-span fan extent (not in paper)
]
BOUNDS_PROVENANCE = "physically plausible ranges for a steel cable-stayed bridge; see the inline comments above"
MISSING_DIRECTION_PENALTY = 5.0  # per unmatched target, in squared-log-error units


def params_from_x(x):
    p = dict(DEFAULT_PARAMS)
    for (name, lo, hi), xi in zip(FREE_PARAMS, x):
        p[name] = float(10 ** xi)
    return p


def match_modes(modes):
    """One-to-one assignment of simulated modes to targets within each direction
    (Hungarian on |log f_sim/f_target|), so one mode can't satisfy two targets.
    Returns [(direction, target_f, mode_dict_or_None)] in TARGETS order."""
    out = [None] * len(TARGETS)
    for d in {t[0] for t in TARGETS}:
        ti = [i for i, t in enumerate(TARGETS) if t[0] == d]
        cand = [m for m in modes if m["direction"] == d and m["freq_hz"] > 1e-3]
        if cand:
            cost = np.array([[abs(np.log(m["freq_hz"] / TARGETS[i][1])) for m in cand] for i in ti])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                out[ti[r]] = cand[c]
    return [(d, f, m) for (d, f), m in zip(TARGETS, out)]


def objective(x):
    try:
        m = build_model(params_from_x(x))
        modes = classify_modes(m, 30)
    except Exception:
        return 1e3
    err = 0.0
    for _, target_f, m in match_modes(modes):
        err += MISSING_DIRECTION_PENALTY if m is None else float(np.log(m["freq_hz"] / target_f)) ** 2
    return err


def run(seeds=(0, 1, 2), workers=1, maxiter=400):
    """Differential evolution (global, parallel across `workers` processes),
    one run per seed. Nelder-Mead didn't converge at this many params (6
    starts, 6 different optima). Several seeds show which params the targets
    actually identify: a wide spread across seeds = not identified."""
    import time
    from scipy.optimize import differential_evolution
    bounds = [(np.log10(lo), np.log10(hi)) for _, lo, hi in FREE_PARAMS]
    results = []
    for seed in seeds:
        t0, gen = time.time(), [0]

        def progress(intermediate_result):
            gen[0] += 1
            if gen[0] % 10 == 0:
                print(f"  seed {seed} gen {gen[0]:4d}  best {intermediate_result.fun:.5f}  "
                      f"{time.time() - t0:6.0f}s", flush=True)

        res = differential_evolution(objective, bounds, seed=seed, workers=workers, updating="deferred",
                                     maxiter=maxiter, popsize=15, tol=1e-6, polish=False,
                                     callback=progress)
        print(f"seed {seed} done: objective {res.fun:.5f}, {res.nfev} evals, "
              f"{time.time() - t0:.0f}s", flush=True)
        results.append(res)
    results.sort(key=lambda r: r.fun)
    best_params = params_from_x(results[0].x)
    m = build_model(best_params)
    modes = classify_modes(m, 30)
    return results, best_params, modes


def report(results, best_params, modes):
    print("\nSeeds (objective):", ", ".join(f"{r.fun:.4f}" for r in results))
    good = [r for r in results if r.fun < 2 * results[0].fun + 1e-3]
    print(f"\nBest fit + spread over the {len(good)} seeds within 2x of best "
          "(wide spread = parameter not identified by the targets):")
    for i, (name, lo, hi) in enumerate(FREE_PARAMS):
        vals = [10 ** r.x[i] for r in good]
        at_bound = " AT BOUND" if min(abs(np.log10(best_params[name] / b)) for b in (lo, hi)) < 0.01 else ""
        print(f"  {name:20s} {best_params[name]:10.4g}  [{min(vals):.4g} .. {max(vals):.4g}]"
              f"  box [{lo:g}, {hi:g}]{at_bound}")
    print("\nFrequency match (one-to-one within direction):")
    for d, target_f, m in match_modes(modes):
        if m is None:
            print(f"  {d:12s} target={target_f:.3f} Hz  -> UNMATCHED")
        else:
            print(f"  {d:12s} target={target_f:.3f} Hz  -> sim={m['freq_hz']:.3f} Hz "
                  f"(mode {m['mode']}, {100 * (m['freq_hz'] - target_f) / target_f:+.1f}%)")


if __name__ == "__main__":
    import os
    results, best_params, modes = run(workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    report(results, best_params, modes)
    if "--save" in sys.argv:
        import json
        with open("calibrated_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print("\nSaved calibrated_params.json")
