"""w4_placebo2. WHY does the arm pipeline read low when the answer is zero?

Registered in /home/aaronadmin/w4_arms/step0_prereg.md, AMENDMENT 10.

`w4_placebo` measured a NEGATIVE placebo of up to -0.0129. Two mechanisms were
named there and they imply OPPOSITE actions:

  OVERLAP           the fixed stand in and the arm draws come from the same
                    human pool and share rows. Cannot happen against a real
                    generated arm, so the correction WOULD NOT APPLY.
  NULL CONSTRUCTION the null splits ONE double draw at random, so its halves
                    carry BINOMIAL per cell counts while both arm sides carry
                    EXACT counts. Applies to the real arm too, so the
                    correction WOULD APPLY and every published lift is
                    understated.

Four blocks, identical except for the one thing each changes:

    V0  the construction w4_placebo used. Reproduction check.
    V1  DISJOINT POOL. Stand in from half A of every cell, everything else
        from half B. Overlap impossible. Null construction unchanged.
    V2  MATCHED NULL. Same pool as V0, but the double draw is split WITHIN
        EACH CELL, so both null halves carry exactly the target counts. Same
        rows as V0's null, only the split rule changes.
    V3  both. NOT a registered read, reported as exploratory only.

CPU only. Nothing generated, no model row read except to reproduce cell counts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from w4_detcap import CORPUS, corpus_tokens, model_tokens            # noqa: E402
from w4_prefix import auc, uni_m                                     # noqa: E402
from w4_step0 import NCELL, cell_cuts, cells, draw_matched, prep      # noqa: E402

MMAX = 40
MS = (1, 2, 3, 5, 10, 20, 40)
DRAWS = 8
NULL_SEEDS = (401, 402, 403, 404, 405, 406, 407, 408)
FIXED_SEEDS = (701, 702, 703, 704)
ALL49 = np.arange(49)
PUB = {1: 0.0176, 2: 0.0141, 3: 0.0073, 5: 0.0101, 10: 0.0146,
       20: 0.0183, 40: 0.0174}


def draw_pair(pool, want, rng, mode, pseed):
    """Two disjoint matched halves.

    mode "random" reproduces w4_placebo: one draw of 2 * want[c] per cell,
    permuted as a whole and cut in the middle, so each half's per cell count is
    BINOMIAL around want[c].

    mode "cell" draws the same 2 * want[c] per cell and cuts EACH CELL in the
    middle, so each half carries EXACTLY want[c]. The pool, the seed and the
    set of rows are the same. Only the split differs."""
    if mode == "random":
        ids, short = draw_matched(pool, want, rng, mult=2)
        o = np.random.default_rng(pseed).permutation(len(ids))
        ids = ids[o]
        h = len(ids) // 2
        return ids[:h], ids[h:], short
    A, B, short = [], [], np.zeros(NCELL, dtype=np.int64)
    for c in np.flatnonzero(want):
        need = int(want[c]) * 2
        have = pool[c]
        if len(have) < need:
            short[c] = need - len(have)
            g = rng.permutation(have)
        else:
            g = rng.choice(have, need, replace=False)
        h = len(g) // 2
        A.append(g[:h]); B.append(g[h:])
    return np.concatenate(A), np.concatenate(B), short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+",
                    default=["research/w4_texcover_streams_s0.npz",
                             "research/w4_texcover_streams_s1.npz"])
    ap.add_argument("--hpool", type=int, default=400000)
    ap.add_argument("--out", default="research/w4_placebo2_results.json")
    args = ap.parse_args()
    print("w4_placebo2. overlap or null construction. CPU, nothing generated.",
          flush=True)
    rng = np.random.default_rng(8000 + MMAX)

    S, TH, DT, CD, LL = [], [], [], [], []
    for f in args.streams:
        a, b, c, d, L = model_tokens(f)
        S.append(a); TH.append(b); DT.append(c); CD.append(d); LL.append(L)
    S, TH, DT = np.concatenate(S), np.concatenate(TH), np.concatenate(DT)
    CD, LL = np.concatenate(CD), np.concatenate(LL)
    S, TH, DT, keep = prep(S, TH, DT, LL, MMAX)
    CD, nmodel = CD[keep], len(S)
    del S, TH, DT

    # Same rng call order as w4_placebo up to here, so the pool is identical.
    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")
    elig = np.flatnonzero(Lall >= MMAX)
    elig = rng.choice(elig, min(args.hpool, len(elig)), replace=False)
    hc = np.asarray(Call[elig], dtype=np.float32)
    ccuts = cell_cuts(hc)
    hcell, mcell = cells(hc, ccuts), cells(CD, ccuts)
    want = np.bincount(mcell, minlength=NCELL)
    pool = [elig[hcell == c] for c in range(NCELL)]

    # The disjoint halves. Split INSIDE each cell so both halves can still fill
    # every cell the model asks for. Splitting the pool globally would starve
    # whole cells and change the design rather than isolate the overlap.
    rs = np.random.default_rng(4242)
    poolA, poolB = [], []
    for c in range(NCELL):
        g = rs.permutation(pool[c])
        h = len(g) // 2
        poolA.append(g[:h]); poolB.append(g[h:])
    smallest = min(len(poolA[c]) for c in np.flatnonzero(want))
    print(f"  MMAX {MMAX}   standing in for {nmodel} model rows   "
          f"human pool {len(elig)}", flush=True)
    print(f"  disjoint halves, smallest half cell holds {smallest} rows "
          f"against a largest want of {int(want.max())}", flush=True)

    hcache = {}

    def human(ids):
        k = ids.tobytes()
        if k not in hcache:
            a, b, c, _, L = corpus_tokens(ids)
            a, b, c, _ = prep(a, b, c, L, MMAX)
            hcache[k] = {m: uni_m(a, b, c, m)[:, ALL49] for m in MS}
        return hcache[k]

    def block(tag, fixpool, armpool, mode):
        nac, tot_short = [], 0
        for sd in NULL_SEEDS:
            ia, ib, sh = draw_pair(armpool, want,
                                   np.random.default_rng(sd), mode, sd + 50)
            tot_short += int(sh.sum())
            A, B = human(ia), human(ib)
            nac.append([auc(A[m], B[m], sd + 7) for m in MS])
        nac = np.asarray(nac)
        null, nsd = nac.mean(0), nac.std(0, ddof=1) / np.sqrt(len(NULL_SEEDS))

        HU = [human(draw_matched(armpool, want,
                                 np.random.default_rng(8200 + d))[0])
              for d in range(DRAWS)]
        FX = [human(draw_matched(fixpool, want,
                                 np.random.default_rng(fs))[0])
              for fs in FIXED_SEEDS]
        pl = np.asarray([[np.mean([auc(HU[d][m], F[m], 8300 + d)
                                   for d in range(DRAWS)]) for m in MS]
                         for F in FX])
        pmean = pl.mean(0) - null
        psd = pl.std(0, ddof=1) / np.sqrt(len(FX))
        print(f"\n  {tag}   null shortfall {tot_short} rows over "
              f"{len(NULL_SEEDS)} null draws")
        print(f"  {'m':>5}{'null':>9}{'placebo':>10}{'p se':>9}"
              f"{'null se':>10}{'p/se':>8}")
        for i, m in enumerate(MS):
            z = pmean[i] / max(nsd[i], 1e-9)
            print(f"  {m:>5}{null[i]:>9.4f}{pmean[i]:>+10.4f}{psd[i]:>9.4f}"
                  f"{nsd[i]:>10.4f}{z:>8.2f}", flush=True)
        return {"null": null.tolist(), "null_se": nsd.tolist(),
                "placebo": pmean.tolist(), "placebo_se": psd.tolist(),
                "shortfall": tot_short}

    out = {}
    out["V0"] = block("V0 reproduction, shared pool, random split",
                      pool, pool, "random")
    out["V1"] = block("V1 DISJOINT POOL, random split",
                      poolA, poolB, "random")
    out["V2"] = block("V2 shared pool, MATCHED NULL",
                      pool, pool, "cell")
    out["V3"] = block("V3 both, EXPLORATORY, not a registered read",
                      poolA, poolB, "cell")

    def zero(v):
        return all(abs(v["placebo"][i]) <= v["null_se"][i]
                   for i in range(len(MS)))

    z1, z2 = zero(out["V1"]), zero(out["V2"])
    print(f"\n  ZERO means |placebo| within 1 null se at EVERY m, "
          f"the same bar P0 used")
    print(f"    V0 {'zero' if zero(out['V0']) else 'biased'}   "
          f"V1 {'zero' if z1 else 'biased'}   "
          f"V2 {'zero' if z2 else 'biased'}   "
          f"V3 {'zero' if zero(out['V3']) else 'biased'}")
    if z1 and not z2:
        verdict = ("Q1 OVERLAP. the correction is DISCARDED and w4_prefix's "
                   "published lifts stand")
    elif z2 and not z1:
        verdict = ("Q2 NULL CONSTRUCTION. the null in w4_prefix and "
                   "w4_prefixch is replaced and both arms are rerun")
    elif z1 and z2:
        verdict = ("Q3 ENTANGLED. both fixes are adopted and both arms are "
                   "rerun")
    else:
        verdict = ("Q4 NEITHER. something else is wrong and nothing about the "
                   "first event question is reported until it is found")
    print(f"  VERDICT  {verdict}")

    # What the published curve becomes under each variant's own placebo.
    print(f"\n  w4_prefix's published lift under each variant's correction")
    print(f"  {'m':>5}{'published':>11}{'V0':>9}{'V1':>9}{'V2':>9}{'V3':>9}")
    for i, m in enumerate(MS):
        r = [PUB[m] - out[v]["placebo"][i] for v in ("V0", "V1", "V2", "V3")]
        print(f"  {m:>5}{PUB[m]:>+11.4f}" + "".join(f"{x:>+9.4f}" for x in r))
    gr = {}
    for v in ("V0", "V1", "V2", "V3"):
        c1 = PUB[1] - out[v]["placebo"][0]
        c40 = PUB[40] - out[v]["placebo"][-1]
        gr[v] = c40 / c1 if abs(c1) > 1e-9 else float("nan")
    print("  growth lift(40)/lift(1)      " +
          "".join(f"{gr[v]:>9.2f}" for v in ("V0", "V1", "V2", "V3")) +
          "     published 0.99")

    out.update({"verdict": verdict, "growth": gr,
                "zero": {v: zero(out[v]) for v in ("V0", "V1", "V2", "V3")}})
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
