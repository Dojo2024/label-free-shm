# Derived from the KW51 finite element model by K. Maes, M. van de Velde and G. Lombaert (KU Leuven),
# Zenodo 10.5281/zenodo.17366772, and the KW51 monitoring data (Zenodo 10.5281/zenodo.3745914).
# Licensed CC BY-NC-SA 4.0.
"""C-PASS §2.6 FE-port acceptance: the 14 mapped FE modes against the Phase I mean identified frequencies,
with MAC pairing at the deck accelerometers (select_sensors.m's Sa rows 1-6; the arch channels hold no shapes
in trackedmodes.mat). Uses Phase I (healthy calibration) data only.

Run from FE/:  python scripts/kw51_acceptance.py  ->  results/kw51/fe_acceptance.json
"""
import sys, json, datetime as dt
sys.path.insert(0, "scripts")
import numpy as np, scipy.io as sio
import kw51_fe as fe
from kw51_model import build

KEY = json.load(open("results/kw51/answer_key.json"))
ECC = (271.94 - 150) / 1000  # select_sensors.m: lateral sensors sit this far above the node (rotation x lever arm)
# select_sensors.m: (node group, x [m], direction) for aBD11Az, aBD17Ay, aBD17Az, aBD17Cz, aBD23Ay, aBD23Az
DECK = [(101, 34.558, "z"), (101, 58.158, "y"), (101, 58.158, "z"), (102, 58.158, "z"), (101, 81.758, "y"), (101, 81.758, "z")]


def node(Nodes, group, x):
    n = Nodes[(np.floor(Nodes[:, 0] / 1000) == group) & (np.round(Nodes[:, 1] * 1e4) == round(x * 1e4)), 0]
    assert len(n) == 1, (group, x, n)
    return n[0]


def mac(x, y):
    return float(np.abs(np.vdot(x, y)) ** 2 / (np.vdot(x, x).real * np.vdot(y, y).real))


def main():
    p = KEY["fe_acceptance"]["params"]
    mdl = build(Es_rel=p["Es_rel"], hb_rel=p["hb_rel"], kxx=np.nan, Id_rel=p["Id_rel"], Ad_rel=p["Ad_rel"], fixed=p["fixed"])
    s = fe.assemble(mdl)
    f, phir, keep = fe.modes(s["K"], s["M"], 50)
    full = np.zeros((len(s["MasterDOF"]), phir.shape[1])); full[keep] = phir
    phi = s["T"] @ full
    pos = {round(d, 2): i for i, d in enumerate(s["DOF"])}
    Sa = np.zeros((len(DECK), len(s["DOF"])))
    for i, (g, x, d) in enumerate(DECK):
        n = node(mdl["Nodes"], g, x)
        if d == "z":
            Sa[i, pos[round(n + 0.03, 2)]] = 1
        else:
            Sa[i, pos[round(n + 0.02, 2)]] = 1
            Sa[i, pos[round(n + 0.04, 2)]] = ECC
    phia = Sa @ phi

    m = sio.loadmat(KEY["data"]["file"], squeeze_me=True, struct_as_record=False)["modes"]
    assert list(m.labels_m[:6]) == ["aBD11Az", "aBD17Ay", "aBD17Az", "aBD17Cz", "aBD23Ay", "aBD23Az"]
    t = np.array([dt.datetime.fromordinal(int(v)) + dt.timedelta(days=v % 1) - dt.timedelta(days=366) for v in m.sdn])
    a, b = [dt.datetime.fromisoformat(d) for d in KEY["periods"]["phase_I"]]
    ph1 = (t >= a) & (t < b + dt.timedelta(days=1))
    f_meas = np.nanmean(m.f[ph1], axis=0)
    sel = np.array(KEY["fe_acceptance"]["fe_modes_for_tracked_1_to_14"]) - 1
    rows = []
    for k in range(14):
        shapes = m.m[ph1, k, :6]
        shapes = shapes[np.all(np.isfinite(shapes), axis=1)]
        med = np.median([[mac(sh, phia[:, j]) for j in range(50)] for sh in shapes], axis=0)
        near = np.where(np.abs(f / f_meas[k] - 1) < 0.10)[0]
        best_near = int(near[np.argmax(med[near])]) if len(near) else None
        rows.append(dict(tracked=k + 1, f_measured=float(f_meas[k]), fe_mode=int(sel[k] + 1), f_fe=float(f[sel[k]]),
                         rel_error=float(f[sel[k]] / f_meas[k] - 1), mac_median=float(med[sel[k]]), n_shapes=len(shapes),
                         best_mac_within_10pct=None if best_near is None else dict(fe_mode=best_near + 1, f_fe=float(f[best_near]),
                                                                                  mac_median=float(med[best_near]))))
    err = np.array([r["rel_error"] for r in rows])
    out = dict(params=p, dofs=[len(s["DOF"]), len(s["MasterDOF"])], within_3pct=int(np.sum(np.abs(err) < 0.03)),
               median_abs_error=float(np.median(np.abs(err))), max_abs_error=float(np.abs(err).max()),
               passed=bool(np.all(np.abs(err) < KEY["fe_acceptance"]["max_rel_error"])), modes=rows,
               fe_frequencies_1_to_50=[float(x) for x in f])
    json.dump(out, open("results/kw51/fe_acceptance.json", "w"), indent=1)
    for r in rows:
        bn = r["best_mac_within_10pct"]
        print(f"tracked {r['tracked']:2d}  FE {r['fe_mode']:2d}  {r['f_measured']:.3f} vs {r['f_fe']:.3f} Hz  {100 * r['rel_error']:+.2f}%  "
              f"MAC {r['mac_median']:.3f}  | best MAC within 10%: FE {bn['fe_mode']} ({bn['mac_median']:.3f})")
    print(f"within 3%: {out['within_3pct']}/14; median |error| {100 * out['median_abs_error']:.2f}%; max {100 * out['max_abs_error']:.2f}%; passed: {out['passed']}")


if __name__ == "__main__":
    main()
