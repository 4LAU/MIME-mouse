"""How many events the model spends, and whether that costs anything.

PRE REGISTERED in HANDOFF.md 2026-08-06, "## How many events the model spends,
registered properly". The three confounds, both branch sets and the prediction
that the AUC comes out MINOR were all fixed before this file existed.

Human conditional fitted on real sequences only,

    log L = a + b log distance + c log duration

then the model's slopes are compared to the human's, and the residual against
the HUMAN fit is used as a one number detector. That AUC is a LOWER BOUND on
what any detector can extract from this defect, not a claim about what the
contract detector currently uses.

No generation, no GPU, no model. Reads only the streams saved by `w4_typpos`.

Safety. Touches no evaluation data, never scoring.py, never
training/candi_polar_flow_best.pt.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_evcount.py
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from w4_beta_curve import MAX_T  # noqa: E402

BOOT = 2000
CAP_CONCERN = 0.02


def design(cond):
    return np.column_stack([np.ones(len(cond)), cond[:, 0], cond[:, 1]])


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def auc(neg, pos):
    a = np.concatenate([neg, pos])
    r = np.argsort(np.argsort(a)) + 1.0
    n0, n1 = len(neg), len(pos)
    u = r[n0:].sum() - n1 * (n1 + 1) / 2.0
    v = u / (n0 * n1)
    return max(v, 1.0 - v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="research/w4_typpos_streams.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_evcount.json")
    args = ap.parse_args()

    z = np.load(args.streams)
    cond = z["cond"].astype(np.float64)
    rL = z["real_L"].astype(np.float64)
    gL = z["gen_L"].astype(np.float64)
    n = len(rL)
    rng = np.random.default_rng(args.seed)

    keep12 = gL >= 12
    cap_r = float((rL >= MAX_T).mean())
    cap_g = float((gL >= MAX_T).mean())
    print(f"\n  {n} paired movements, all of them, no length filter")
    print(f"  the gL >= 12 filter would drop {int((~keep12).sum())}")
    print(f"  at the MAX_T cap   real {cap_r:.2%}   generated {cap_g:.2%}"
          f"   {'CONFOUND, report as bounded' if max(cap_r, cap_g) > CAP_CONCERN else 'clean'}")

    X = design(cond)
    yr, yg = np.log(np.maximum(rL, 1)), np.log(np.maximum(gL, 1))
    br, bg = fit(X, yr), fit(X, yg)
    print(f"\n  {'':<22}{'intercept':>11}{'log distance':>14}{'log duration':>14}")
    print(f"  {'human fit':<22}{br[0]:>11.4f}{br[1]:>14.4f}{br[2]:>14.4f}")
    print(f"  {'model fit':<22}{bg[0]:>11.4f}{bg[1]:>14.4f}{bg[2]:>14.4f}")
    print(f"  {'model minus human':<22}{bg[0] - br[0]:>+11.4f}"
          f"{bg[1] - br[1]:>+14.4f}{bg[2] - br[2]:>+14.4f}")

    # residual of every sequence against the HUMAN fit
    er = yr - X @ br
    eg = yg - X @ br
    a_sig = auc(er, eg)
    a_abs = auc(np.abs(er), np.abs(eg))

    boots = {"d_dist": [], "d_dur": [], "auc": [], "auc_abs": []}
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        Xi = X[i]
        b1, b2 = fit(Xi, yr[i]), fit(Xi, yg[i])
        boots["d_dist"].append(b2[1] - b1[1])
        boots["d_dur"].append(b2[2] - b1[2])
        e1, e2 = yr[i] - Xi @ b1, yg[i] - Xi @ b1
        boots["auc"].append(auc(e1, e2))
        boots["auc_abs"].append(auc(np.abs(e1), np.abs(e2)))
    se = {k: float(np.std(v, ddof=1)) for k, v in boots.items()}

    d_dur, d_dist = bg[2] - br[2], bg[1] - br[1]
    z_dur = d_dur / se["d_dur"]
    print(f"\n  duration slope difference {d_dur:+.4f}  se {se['d_dur']:.4f}"
          f"   {z_dur:+.2f} sigma")
    print(f"  distance slope difference {d_dist:+.4f}  se {se['d_dist']:.4f}"
          f"   {d_dist / se['d_dist']:+.2f} sigma")
    slope_v = ("CONFIRMED. The model over responds to commanded duration."
               if abs(z_dur) >= 3.0 else
               "NOT CONFIRMED. The exploratory event count finding is withdrawn.")
    print(f"\n  SLOPE VERDICT  {slope_v}")

    print(f"\n  residual as a one number detector, LOWER BOUND on any detector")
    print(f"    signed residual AUC   {a_sig:.4f}  se {se['auc']:.4f}")
    print(f"    absolute residual AUC {a_abs:.4f}  se {se['auc_abs']:.4f}"
          f"   (contract detector reads 0.6612)")
    best = max(a_sig, a_abs)
    if best >= 0.58:
        cost_v = (f"DOMINANT. {best:.4f} from one scalar against the contract's "
                  "0.6612. Event count is the main target.")
    elif best < 0.54:
        cost_v = (f"MINOR. {best:.4f}. A real defect that is not worth the "
                  "training budget on its own.")
    else:
        cost_v = f"PARTIAL. {best:.4f}. Report, claim neither."
    print(f"\n  COST VERDICT  {cost_v}\n")

    # the exploratory quartile table, recomputed without the length filter
    q = np.quantile(cond[:, 1], [0, .25, .5, .75, 1.0])
    print("  by commanded duration quartile, all movements")
    quarts = []
    for i in range(4):
        m = (cond[:, 1] >= q[i]) & ((cond[:, 1] <= q[i + 1]) if i == 3
                                    else (cond[:, 1] < q[i + 1]))
        row = {"lo_s": float(np.exp(q[i])), "hi_s": float(np.exp(q[i + 1])),
               "real": float(rL[m].mean()), "gen": float(gL[m].mean()),
               "ratio": float(gL[m].mean() / rL[m].mean()), "n": int(m.sum())}
        print(f"    {row['lo_s']:6.3f}s to {row['hi_s']:6.3f}s   real "
              f"{row['real']:6.2f}  gen {row['gen']:6.2f}  ratio {row['ratio']:.3f}")
        quarts.append(row)
    sp_r = quarts[3]["real"] / quarts[0]["real"]
    sp_g = quarts[3]["gen"] / quarts[0]["gen"]
    print(f"    span across quartiles   human {sp_r:.2f}x   model {sp_g:.2f}x\n")

    json.dump({"n": n, "dropped_by_filter": int((~keep12).sum()),
               "cap_real": cap_r, "cap_gen": cap_g,
               "human_fit": br.tolist(), "model_fit": bg.tolist(),
               "d_duration": float(d_dur), "d_distance": float(d_dist),
               "se": se, "z_duration": float(z_dur),
               "auc_signed": a_sig, "auc_abs": a_abs,
               "slope_verdict": slope_v, "cost_verdict": cost_v,
               "quartiles": quarts, "span_human": sp_r, "span_model": sp_g},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
