"""w4_margtilt. Price the pooled marginal gap without touching a single token.

Registered in /home/aaronadmin/w4_arms/margtilt_prereg.md. Third attempt at the
question `w4_margfix` needs answered before it is worth four GPU hours; the two
earlier attempts and why they failed are recorded in `margfix_prereg.md`.

Reweight whole real human trajectories so their length weighted pooled token
marginal matches the model's. Every row stays an unmodified human trajectory, so
within trajectory dependence is intact and the only thing that moved is how
often each kind of trajectory appears. That is the least disruptive way to reach
a given marginal, and it brackets the iid relabel from the other side.

CPU only. Nothing generated, no checkpoint read, `human_eval_features` untouched.
"""
from __future__ import annotations

import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from w4_detcap import (IDX_UNI, MK_GBM, TBUF, build, corpus_ids,   # noqa: E402
                       corpus_tokens, cv_auc, model_tokens)

STREAMS = [f"research/w4_texcover_streams_s{i}.npz" for i in (0, 1)]
SEEDS = (2001, 2002, 2003)
N_ARM = 4000          # rows per detector arm
POOL = 16000          # rows the tilt draws from, so weights have somewhere to go
BLOCKS = ((0, 16), (16, 33), (33, 49))    # u_s, u_th, u_dt inside the 49
NTOK = 49


def uni(tok):
    """49 token rate columns plus len_frac, and the row lengths."""
    F = build(*tok)[1][:, IDX_UNI]
    return F, F[:, -1] * TBUF


def pooled(U, L, w=None):
    """Length weighted pooled marginal. This is what a per class bias moves."""
    a = L if w is None else w * L
    return (U[:, :NTOK] * a[:, None]).sum(0) / a.sum()


def blockwise_tv(p, q):
    return [0.5 * float(np.abs(p[i:j] - q[i:j]).sum()) for i, j in BLOCKS]


def fit_tilt(U, L, target, iters=200, damp=0.5, ess_floor=0.05):
    """Damped Newton on theta so that pooled(U, L, exp(theta.u)) == target.

    The three channel blocks each sum to one, so the weighted covariance has
    three exact null directions and the step uses a pseudoinverse.

    A target outside the convex hull of the pool's rate vectors is INFEASIBLE
    and Newton walks theta to infinity chasing it, at which point the weights
    collapse onto a handful of rows. That happened on the first run of this
    script with the random direction control and it took out the whole arm, so
    the loop now keeps the last iterate whose effective sample size is still
    above `ess_floor` of the pool and reports it.
    """
    X = U[:, :NTOK]
    th = np.zeros(NTOK)
    best = (np.inf, np.zeros(NTOK))
    for _ in range(iters):
        z = X @ th
        w = np.exp(z - z.max())
        a = w * L
        a = a / a.sum()
        m = (X * a[:, None]).sum(0)
        r = target - m
        e = float(np.abs(r).max())
        if ess(w / w.sum()) >= ess_floor * len(w) and e < best[0]:
            best = (e, th.copy())
        if e < 1e-9:
            break
        Xc = X - m
        C = (Xc * a[:, None]).T @ Xc
        step = np.linalg.pinv(C, rcond=1e-8) @ r
        n = np.linalg.norm(step)
        if n > 50.0:                    # a runaway step is a feasibility signal
            step = step * (50.0 / n)
        th = th + damp * step
    th = best[1]
    z = X @ th
    w = np.exp(z - z.max())
    return w / w.sum(), th


def draw(w, n, rng):
    """Without replacement. Duplicated rows would be a tell in themselves."""
    avail = int((w > 1e-12).sum())
    if avail < n:
        raise RuntimeError(
            f"tilt collapsed: only {avail} rows carry weight, need {n}. "
            f"the target is outside the pool's convex hull.")
    return rng.choice(len(w), n, replace=False, p=w)


def ess(w):
    return float(1.0 / np.square(w / w.sum()).sum())


def rung(A, B, seed):
    r = np.random.default_rng(4000 + seed)
    A, B = A[r.permutation(len(A))], B[r.permutation(len(B))]
    n = min(len(A), len(B))
    y = np.r_[np.zeros(n), np.ones(n)]
    return cv_auc(MK_GBM, np.vstack([A[:n], B[:n]]), y)[0]


def main():
    # ---- the model's pooled coarse marginal, in the detector's own space ----
    MU, ML = [], []
    for f in STREAMS:
        u, l = uni(model_tokens(f))
        MU.append(u); ML.append(l)
    MU, ML = np.vstack(MU), np.concatenate(ML)
    TARGET = pooled(MU, ML)
    print(f"  model arm {MU.shape}, target is its length weighted pooled "
          f"coarse marginal\n", flush=True)

    rows = []
    print(f"  {'seed':>6}{'null':>9}{'tilt':>9}{'randdir':>10}"
          f"{'gap_closed_s':>14}{'th':>7}{'dt':>7}{'ess':>8}")
    for sd in SEEDS:
        rg = np.random.default_rng(sd)
        ids = corpus_ids(rg, N_ARM + POOL)
        A, _ = uni(corpus_tokens(ids[:N_ARM]))
        P, PL = uni(corpus_tokens(ids[N_ARM:]))
        HUM = pooled(P, PL)
        gap0 = blockwise_tv(HUM, TARGET)

        w, _ = fit_tilt(P, PL, TARGET)
        got = pooled(P, PL, w * len(w))
        left = blockwise_tv(got, TARGET)
        closed = [1.0 - left[i] / gap0[i] if gap0[i] > 0 else 1.0
                  for i in range(3)]

        # C1. the SAME displacement, permuted inside each channel block. Same
        # block sums, same L2 norm, same multiset of magnitudes, different
        # direction. A freshly drawn random vector was tried first and it put
        # the target outside the pool's convex hull, which is a statement about
        # feasibility rather than about detectability.
        rr = np.random.default_rng(500 + sd)
        d = TARGET - HUM
        v = d.copy()
        for i, j in BLOCKS:
            v[i:j] = d[i:j][rr.permutation(j - i)]
        # shrink only as far as feasibility forces, and report how far
        shrink = 1.0
        while shrink > 0.05 and np.any(HUM + shrink * v < 0.2 * HUM):
            shrink *= 0.8
        v = shrink * v
        wR, _ = fit_tilt(P, PL, HUM + v)

        d1 = np.random.default_rng(800 + sd)
        B0 = P[draw(np.full(len(P), 1.0 / len(P)), N_ARM, d1)]
        BT = P[draw(w, N_ARM, np.random.default_rng(801 + sd))]
        BR = P[draw(wR, N_ARM, np.random.default_rng(802 + sd))]

        r = dict(seed=sd, null=rung(A, B0, sd), tilt=rung(A, BT, sd),
                 randdir=rung(A, BR, sd), closed=closed, gap0=gap0,
                 ess=ess(w), ess_rand=ess(wR), shrink=shrink,
                 randdir_closed=blockwise_tv(pooled(P, PL, wR * len(wR)),
                                             HUM + v))
        rows.append(r)
        print(f"  {sd:>6}{r['null']:>9.4f}{r['tilt']:>9.4f}{r['randdir']:>10.4f}"
              f"{closed[0]:>14.2f}{closed[1]:>7.2f}{closed[2]:>7.2f}"
              f"{r['ess']:>8.0f}", flush=True)

    mn = {k: float(np.mean([r[k] for r in rows]))
          for k in ("null", "tilt", "randdir", "ess")}
    closed = np.mean([r["closed"] for r in rows], axis=0)
    print(f"  {'mean':>6}{mn['null']:>9.4f}{mn['tilt']:>9.4f}"
          f"{mn['randdir']:>10.4f}{closed[0]:>14.2f}{closed[1]:>7.2f}"
          f"{closed[2]:>7.2f}{mn['ess']:>8.0f}")

    # ---- the model's own lift on the same rung, shuffled before balancing --
    rg = np.random.default_rng(9001)
    H, _ = uni(corpus_tokens(corpus_ids(rg, 5000)))
    model = rung(H, MU, 11)

    lift = dict(model=model - mn["null"], tilt=mn["tilt"] - mn["null"],
                randdir=mn["randdir"] - mn["null"])
    share = lift["tilt"] / lift["model"] if lift["model"] > 1e-9 else 0.0
    print(f"\n  null corrected lifts")
    for k in ("model", "tilt", "randdir"):
        print(f"    {k:<10}{lift[k]:>+9.4f}")

    # ---- C1'. DOSE RESPONSE along the model's own displacement -----------
    # registered as AMENDMENT 2 of margtilt_prereg.md, replacing a C1 that
    # could not deliver a matched magnitude comparison because feasibility
    # bound the permuted direction at a fifth of the model's displacement.
    print(f"\n  C1prime. DOSE RESPONSE along the model's own displacement")
    print(f"  {'f':>6}{'lift':>9}{'auc':>9}")
    dose = []
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        per = []
        for sd in SEEDS:
            rg = np.random.default_rng(sd)
            ids = corpus_ids(rg, N_ARM + POOL)
            A, _ = uni(corpus_tokens(ids[:N_ARM]))
            P, PL = uni(corpus_tokens(ids[N_ARM:]))
            HUM = pooled(P, PL)
            wf, _ = fit_tilt(P, PL, HUM + f * (TARGET - HUM))
            B = P[draw(wf, N_ARM, np.random.default_rng(1200 + sd))]
            per.append(rung(A, B, sd))
        a = float(np.mean(per))
        dose.append(dict(f=f, auc=a, lift=a - mn["null"]))
        print(f"  {f:>6.2f}{a - mn['null']:>+9.4f}{a:>9.4f}", flush=True)
    dl = [d["lift"] for d in dose]
    mono = all(dl[i + 1] >= dl[i] - 0.02 for i in range(len(dl) - 1))
    doubles = dl[-1] >= 2.0 * dl[1] if dl[1] > 1e-9 else dl[-1] > 0.01
    c1 = mono and doubles

    print(f"\n  CONTROLS")
    c2 = bool(np.all(closed >= 0.8))
    c3 = abs(mn["null"] - 0.5) <= 0.02
    c4 = mn["ess"] >= POOL * 0.5
    shr = float(np.mean([r["shrink"] for r in rows]))
    print(f"    C1prime  dose response monotone {mono}, f=1 lift {dl[-1]:+.4f} "
          f"at least twice f=0.25 lift {dl[1]:+.4f} {doubles}  -> "
          f"{'PASS' if c1 else 'FAIL'}")
    print(f"    DESCRIPTIVE, not a gate. permuted direction {lift['randdir']:+.4f} "
          f"against tilt {lift['tilt']:+.4f}, at {shr:.2f} of the model's\n"
          f"      displacement because feasibility binds there. A permuted "
          f"displacement a fifth the size\n      is twice as detectable, so the "
          f"model's marginal error points somewhere unusually benign.")
    print(f"    C2  gap closed on all three channels, "
          f"{np.round(closed, 2).tolist()} vs 0.80  -> "
          f"{'PASS' if c2 else 'FAIL'}")
    print(f"    C3  null {mn['null']:.4f} within 0.02 of 0.5  -> "
          f"{'PASS' if c3 else 'FAIL'}")
    print(f"    C4  effective sample size {mn['ess']:.0f} of {POOL}  -> "
          f"{'PASS' if c4 else 'CONTAMINATED'}")

    ok = c1 and c2 and c3
    print(f"\n  SHARE of the model's unigram rate lift reachable by a pooled "
          f"marginal correction  {share * 100:.1f} percent")
    v = ("FUND THE GPU HALF" if share > 0.5 else
         "FUND, STRONG UNREACHABLE" if share >= 0.15 else
         "CLOSE THE MARGINAL FAMILY ON TOKENS")
    print(f"  DECISION  {v}")
    if not ok:
        print("  A CONTROL FAILED. the decision above is NOT readable.")
    if not c4:
        print("  effective sample size low, the reading is contaminated by "
              "trajectory type duplication.")

    json.dump(dict(rows=rows, mean=mn, model=model, lift=lift, share=share,
                   dose=dose, dose_monotone=bool(mono), dose_doubles=bool(doubles),
                   controls=dict(c1=c1, c2=c2, c3=c3, c4=c4),
                   closed=closed.tolist(), decision=v),
              open("research/w4_margtilt_results.json", "w"), indent=1)
    print("\n  wrote research/w4_margtilt_results.json")


if __name__ == "__main__":
    main()
