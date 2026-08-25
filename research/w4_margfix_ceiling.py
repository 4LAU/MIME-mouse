"""w4_margfix, the ceiling probe. CPU only, nothing generated, no GPU.

The prefit found the model's POOLED token marginals are already close to the
corpus, total variation 0.028 to 0.036 per channel. Yet `w4_detcap` reads
+0.0784 corrected off per trajectory UNIGRAM RATE features. Those two facts are
not in conflict, and the difference between them is exactly the question this
probe answers:

    a per class logit bias can only move the POOLED marginal, which is the MEAN
    of the per trajectory rate vectors. If the rate detector is reading the
    mean, margfix has room. If it is reading the DISPERSION or the joint around
    that mean, margfix cannot touch it and its ceiling is near zero.

The probe removes the mean difference between the two arms by hand and re reads
the same detector. The drop is the most a perfect marginal match could buy.

This is a CEILING, not a prediction. Removing a mean shift by hand is not the
same operation as sampling from a corrected model, and the real arm can land
anywhere below it, including at zero.
"""
from __future__ import annotations

import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from w4_detcap import (IDX_UNI, MK_GBM, corpus_ids, corpus_tokens,   # noqa: E402
                       cv_auc, model_tokens, build, rate_names)

STREAMS = [f"research/w4_texcover_streams_s{i}.npz" for i in (0, 1)]
NULL_SEEDS = (2001, 2002, 2003)
PLANT_P = (0.0, 0.02, 0.05, 0.10)
NAMES = rate_names()
UNI_NAMES = [NAMES[i] for i in IDX_UNI[:-1]] + ["len_frac"]
N_TOKEN_COLS = len(IDX_UNI) - 1        # everything except the trailing len_frac


def uni(s, th, dt, cond, L):
    _, FR = build(s, th, dt, cond, L)
    return FR[:, IDX_UNI]


def meanfix(A, B, cols):
    """Shift arm B onto arm A's column means over `cols`. Nothing else moves."""
    B = B.copy()
    B[:, cols] += A[:, cols].mean(0) - B[:, cols].mean(0)
    return B


def run(A, B, cols_token, cols_all, seed=0):
    # SHUFFLE BEFORE TRUNCATING. corpus_tokens sorts the ids it is handed, so
    # taking the first n rows of a longer arm takes the LOWEST ids, which is a
    # subpopulation and not a subsample. This is the w4_poskl bug, and it read
    # a divergence of 0.311 where the truth was 0.0448.
    r = np.random.default_rng(4000 + seed)
    A, B = A[r.permutation(len(A))], B[r.permutation(len(B))]
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    y = np.r_[np.zeros(n), np.ones(n)]
    out = {}
    out["raw"], _ = cv_auc(MK_GBM, np.vstack([A, B]), y)
    out["fix_token"], _ = cv_auc(
        MK_GBM, np.vstack([A, meanfix(A, B, cols_token)]), y)
    out["fix_all"], _ = cv_auc(
        MK_GBM, np.vstack([A, meanfix(A, B, cols_all)]), y)
    return out


def plant(s, th, dt, L, p, rng):
    """A pure per event iid class shift. Moves the pooled marginal, which is
    exactly the kind of difference a per class logit bias can remove."""
    s = s.copy()
    live = np.arange(s.shape[1])[None] < np.asarray(L)[:, None]
    hit = live & (rng.random(s.shape) < p) & (s < 130)
    s[hit] = np.minimum(s[hit] + 8, 129)
    return s, th, dt


def main():
    cols_tok = np.arange(N_TOKEN_COLS)
    cols_all = np.arange(len(IDX_UNI))
    rng = np.random.default_rng(11)

    M = np.vstack([uni(*model_tokens(f)) for f in STREAMS])
    hs, hth, hdt, hc, hL = corpus_tokens(corpus_ids(rng, 5000))
    H = uni(hs, hth, hdt, hc, hL)
    print(f"  model {M.shape}   human {H.shape}   "
          f"{N_TOKEN_COLS} token rate columns + len_frac\n", flush=True)

    print(f"  {'arm':>22}{'raw':>9}{'fix_token':>11}{'fix_all':>9}")
    m = run(H, M, cols_tok, cols_all)
    print(f"  {'MODEL':>22}{m['raw']:>9.4f}{m['fix_token']:>11.4f}"
          f"{m['fix_all']:>9.4f}", flush=True)

    nulls = []
    for sd in NULL_SEEDS:
        r2 = np.random.default_rng(sd)
        ids = corpus_ids(r2, 10000)
        a = uni(*corpus_tokens(ids[:5000]))
        b = uni(*corpus_tokens(ids[5000:]))
        nl = run(a, b, cols_tok, cols_all)
        nulls.append(nl)
        print(f"  {'null ' + str(sd):>22}{nl['raw']:>9.4f}"
              f"{nl['fix_token']:>11.4f}{nl['fix_all']:>9.4f}", flush=True)
    N = {k: float(np.mean([x[k] for x in nulls])) for k in m}
    print(f"  {'null mean':>22}{N['raw']:>9.4f}{N['fix_token']:>11.4f}"
          f"{N['fix_all']:>9.4f}")

    c = {k: m[k] - N[k] for k in m}
    print(f"\n  null corrected lift")
    print(f"  {'raw':>22}{c['raw']:>+9.4f}")
    print(f"  {'mean removed, tokens':>22}{c['fix_token']:>+9.4f}")
    print(f"  {'mean removed, all':>22}{c['fix_all']:>+9.4f}")
    share = 1.0 - c["fix_token"] / c["raw"] if abs(c["raw"]) > 1e-9 else 0.0
    print(f"\n  CEILING. share of the unigram rate lift that lives in the "
          f"pooled mean  {share * 100:.1f} percent")

    # ---- POWER. plant a pure marginal shift and check both arms read it ----
    print(f"\n  POWER CONTROL. human vs human with a planted per event class "
          f"shift.\n  raw must SEE it and fix_token must NOT, or the mean "
          f"removal is not doing\n  what the ceiling reading assumes.")
    r3 = np.random.default_rng(77)
    ids = corpus_ids(r3, 10000)
    ta = corpus_tokens(ids[:5000])
    tb = corpus_tokens(ids[5000:])
    A = uni(*ta)
    print(f"  {'p':>22}{'raw':>9}{'fix_token':>11}{'fix_all':>9}")
    power = {}
    for p in PLANT_P:
        s2, t2, d2 = plant(tb[0], tb[1], tb[2], tb[4], p,
                           np.random.default_rng(900 + int(p * 1000)))
        B = uni(s2, t2, d2, tb[3], tb[4])
        pr = run(A, B, cols_tok, cols_all)
        power[str(p)] = pr
        print(f"  {p:>22.2f}{pr['raw']:>9.4f}{pr['fix_token']:>11.4f}"
              f"{pr['fix_all']:>9.4f}", flush=True)
    base = power["0.0"]
    hit = [p for p in PLANT_P[1:]
           if power[str(p)]["raw"] - base["raw"] >= c["raw"] * 0.5]
    if not hit:
        print("\n  UNPOWERED. no planted size reached half the model arm's own "
              "lift.\n  the ceiling reading is NOT interpretable.")
    else:
        p = hit[0]
        pr = power[str(p)]
        kept = ((pr["fix_token"] - base["fix_token"])
                / max(pr["raw"] - base["raw"], 1e-9))
        print(f"\n  using p {p}, raw rose {pr['raw'] - base['raw']:+.4f} and "
              f"fix_token rose {pr['fix_token'] - base['fix_token']:+.4f}")
        print(f"  mean removal deleted {(1 - kept) * 100:.1f} percent of a "
              f"KNOWN pure marginal shift\n  -> "
              f"{'POWERED' if kept < 0.5 else 'THE REMOVAL DOES NOT WORK'}")

    json.dump(dict(model=m, null=N, corrected=c, share=share, power=power),
              open("research/w4_margfix_ceiling.json", "w"), indent=1)
    print("\n  wrote research/w4_margfix_ceiling.json")


if __name__ == "__main__":
    main()
