"""w4_margsurr. How detectable is the model's marginal gap, on its own?

Registered as AMENDMENT 3 of /home/aaronadmin/w4_arms/margfix_prereg.md, which
also records why the earlier feature space version was retracted.

`w4_margfix`'s prefit measured the model's pooled token marginals against the
corpus: total variation 0.0351 on speed, 0.0281 on direction, 0.0357 on timing.
A per class logit bias can move the pooled marginal and nothing else. So the
question that decides whether the GPU half is worth four hours is:

    if the ONLY difference between two arms were a pooled marginal gap of
    exactly that size, how much of the rate detector's lift would it produce?

Answered by construction. Relabel human tokens independently per event until the
pooled marginal matches the model's, and score human against that surrogate.

CPU only. Nothing is generated, no checkpoint is read, `human_eval_features` is
never touched.
"""
from __future__ import annotations

import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_ar import DT_MAX_MS                               # noqa: E402
from models.event_stream_polar import (S_PAD_CLASS, TH_BINS,        # noqa: E402
                                       TH_NULL_CLASS, TICK_CLASS)
from w4_detcap import (IDX_UNI, MK_GBM, build, corpus_ids,          # noqa: E402
                       corpus_tokens, cv_auc, model_tokens)
from w4_margfix import marginals, tv                                # noqa: E402

STREAMS = [f"research/w4_texcover_streams_s{i}.npz" for i in (0, 1)]
SEEDS = (2001, 2002, 2003)
N_ROWS = 5000
CONTRACT_SD = 0.0073


def uni(s, th, dt, cond, L):
    return build(s, th, dt, cond, L)[1][:, IDX_UNI]


def relabel_plan(p_src, p_tgt):
    """Per class replacement probability, and the distribution to replace from.

    For a class the target wants LESS of, a token of that class is replaced with
    probability (p_src - p_tgt) / p_src. The replacement is drawn from the
    classes the target wants MORE of, in proportion to how much more. In
    expectation the relabeled marginal is exactly p_tgt.
    """
    excess = np.maximum(p_src - p_tgt, 0.0)
    deficit = np.maximum(p_tgt - p_src, 0.0)
    prob = np.divide(excess, p_src, out=np.zeros_like(excess), where=p_src > 0)
    q = deficit / deficit.sum() if deficit.sum() > 0 else np.zeros_like(deficit)
    return prob, q


def apply_plan(x, mask, prob, q, rng):
    """Relabel in place under `mask`. Returns the relabeled array and the mask
    of cells that actually moved, so the caller can repair the conventions."""
    x = x.copy()
    hit = mask & (rng.random(x.shape) < prob[np.clip(x, 0, len(prob) - 1)])
    n = int(hit.sum())
    if n:
        x[hit] = rng.choice(len(q), size=n, p=q)
    return x, hit


def make_surrogate(s, th, dt, L, targets, rng, randomise_only=False):
    """Relabel human streams so the pooled marginal becomes `targets`.

    randomise_only keeps the SAME per token replacement rate but redraws from
    the arm's own marginal, so the pooled marginal does not move. That isolates
    the cost of relabeling independently from the marginal change itself.
    """
    live = np.arange(s.shape[1])[None] < np.asarray(L)[:, None]
    motion = live & (s > TICK_CLASS) & (s < S_PAD_CLASS)
    src = marginals(s, th, dt, L)
    chans = [(s, live, S_PAD_CLASS, 0), (th, motion, TH_BINS, 1),
             (dt, live, DT_MAX_MS + 1, 2)]
    out = []
    for x, mask, w, i in chans:
        prob, q = relabel_plan(src[i], targets[i])
        if randomise_only:
            q = src[i] / max(src[i].sum(), 1e-12)
        out.append(apply_plan(np.clip(x, 0, w - 1) * mask + x * ~mask,
                              mask, prob, q, rng))
    (s2, _), (th2, _), (dt2, _) = out
    s2 = np.where(live, s2, s)
    th2 = np.where(motion, th2, th)
    dt2 = np.where(live, dt2, dt)

    # repair the conventions. A relabeled stream that carries a combination no
    # model could emit would be read by the detector as the difference, and the
    # measurement would be of the repair rather than of the marginal.
    now_tick = live & (s2 == TICK_CLASS)
    th2 = np.where(now_tick, TH_NULL_CLASS, th2)
    now_moving = live & (s2 > TICK_CLASS) & (s2 < S_PAD_CLASS) \
        & (th2 >= TH_NULL_CLASS)
    k = int(now_moving.sum())
    if k:
        pth = targets[1] / max(targets[1].sum(), 1e-12)
        th2[now_moving] = rng.choice(len(pth), size=k, p=pth)
    return s2, th2, dt2


def rung(A, B, seed):
    r = np.random.default_rng(4000 + seed)
    A, B = A[r.permutation(len(A))], B[r.permutation(len(B))]
    n = min(len(A), len(B))
    y = np.r_[np.zeros(n), np.ones(n)]
    return cv_auc(MK_GBM, np.vstack([A[:n], B[:n]]), y)[0]


def main():
    # ---- the model's pooled marginal, the target the surrogate must hit -----
    acc = [marginals(*[np.load(f)[k].astype(np.int64)
                       for k in ("s", "th", "dt")]) for f in STREAMS]
    PM = tuple(np.mean([p[i] for p in acc], axis=0) for i in range(3))
    M = np.vstack([uni(*model_tokens(f)) for f in STREAMS])
    print(f"  model arm {M.shape}, target marginal from {STREAMS}\n",
          flush=True)

    print(f"  {'seed':>6}{'null':>9}{'surrogate':>11}{'randcontrol':>13}"
          f"{'tv_s':>8}{'tv_th':>8}{'tv_dt':>8}")
    rows = []
    for sd in SEEDS:
        rg = np.random.default_rng(sd)
        ids = corpus_ids(rg, 2 * N_ROWS)
        ta, tb = corpus_tokens(ids[:N_ROWS]), corpus_tokens(ids[N_ROWS:])
        A = uni(*ta)
        s, th, dt, cond, L = tb
        B0 = uni(*tb)
        rr = np.random.default_rng(7000 + sd)
        sS = make_surrogate(s, th, dt, L, PM, rr)
        sR = make_surrogate(s, th, dt, L, PM, np.random.default_rng(7000 + sd),
                            randomise_only=True)
        BS, BR = uni(*sS, cond, L), uni(*sR, cond, L)
        PS = marginals(*sS, L)
        tvs = [tv(PM[i], PS[i]) for i in range(3)]
        rows.append(dict(seed=sd, null=rung(A, B0, sd),
                         surr=rung(A, BS, sd), rand=rung(A, BR, sd), tv=tvs))
        r = rows[-1]
        print(f"  {sd:>6}{r['null']:>9.4f}{r['surr']:>11.4f}{r['rand']:>13.4f}"
              f"{tvs[0]:>8.4f}{tvs[1]:>8.4f}{tvs[2]:>8.4f}", flush=True)

    mn = {k: float(np.mean([r[k] for r in rows])) for k in ("null", "surr",
                                                            "rand")}
    print(f"  {'mean':>6}{mn['null']:>9.4f}{mn['surr']:>11.4f}"
          f"{mn['rand']:>13.4f}")
    print("\n  tv columns are the surrogate's REMAINING distance to the model "
          "marginal.\n  they should be near zero, which is the check that the "
          "relabel hit its target.")

    # ---- the model's own lift on the same rung, and detcap's truncation -----
    rg = np.random.default_rng(9001)
    ids = corpus_ids(rg, N_ROWS)
    H = uni(*corpus_tokens(ids))
    shuffled = rung(H, M, 11)
    n = min(len(H), len(M))
    y = np.r_[np.zeros(n), np.ones(n)]
    sorted_first = cv_auc(MK_GBM, np.vstack([H[:n], M[:n]]), y)[0]
    print(f"\n  MODEL arm, same rung")
    print(f"    human shuffled then truncated   {shuffled:.4f}")
    print(f"    human truncated as w4_detcap    {sorted_first:.4f}   "
          f"(corpus_tokens sorts, so this is the 4000 lowest ids)")
    d = sorted_first - shuffled
    print(f"    difference {d:+.4f}, contract noise sd {CONTRACT_SD}  -> "
          f"{'DETCAP STANDS' if abs(d) <= CONTRACT_SD else 'DETCAP NEEDS A CORRECTION'}")

    lift_model = shuffled - mn["null"]
    lift_surr = mn["surr"] - mn["null"]
    lift_rand = mn["rand"] - mn["null"]
    share = (lift_surr - lift_rand) / lift_model if lift_model > 1e-9 else 0.0
    print(f"\n  null corrected lifts")
    print(f"    model                    {lift_model:+.4f}")
    print(f"    marginal surrogate       {lift_surr:+.4f}")
    print(f"    randomisation control    {lift_rand:+.4f}")
    ok = lift_rand < lift_surr * 0.5
    print(f"\n  the randomisation control must be clearly the smaller  -> "
          f"{'OK' if ok else 'READING THROWN OUT'}")
    print(f"\n  SHARE of the model's unigram rate lift reachable by a perfect "
          f"marginal match  {share * 100:.1f} percent")
    v = ("RUN THE GPU HALF" if share > 0.5 else
         "RUN, STRONG UNREACHABLE" if share >= 0.15 else
         "CLOSE THE ARM, do not spend the GPU")
    print(f"  DECISION  {v}" + ("" if ok else
          "\n  the control failed, so the decision above is NOT readable."))

    json.dump(dict(rows=rows, mean=mn, model=shuffled,
                   model_detcap_truncation=sorted_first,
                   lift=dict(model=lift_model, surr=lift_surr, rand=lift_rand),
                   share=share, control_ok=bool(ok), decision=v),
              open("research/w4_margsurr_results.json", "w"), indent=1)
    print("\n  wrote research/w4_margsurr_results.json")


if __name__ == "__main__":
    main()
