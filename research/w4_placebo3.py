"""w4_placebo3. Is the placebo an overlap artifact. CPU, nothing generated.

On 10,205 model rows the placebo of w4_placebo read -0.0055 at m=1 rising to
-0.0286 at m=20, at twenty standard errors, and it GREW from the 1,989 row run
rather than shrinking. Both facts fit one mechanism: the placebo compares two
matched human draws from the SAME pool, so they share rows, and a row that sits
in both classes with identical features is a guaranteed misranking for the
forest. At m=1 many rows share a unigram vector anyway so a twin is invisible;
at m=40 every row is unique and a twin is a perfect match. The real arm has no
such twins, the model rows are not in the human pool, so if overlap is the
mechanism the placebo does not apply to the arm at all.

Three readings, all on the same pool, want, draws and seeds as w4_placebo:

    OVERLAP   measured, not argued. Shared rows between each arm draw and each
              fixed stand in, as a fraction of rows.
    V0 DEDUP  the w4_placebo construction with the shared rows REMOVED from the
              fixed stand in before scoring. Same pool, same draws. If the
              placebo goes to zero here, overlap is the mechanism.
    V1        disjoint half pools, the w4_placebo2 design, sixteen stand ins.
              No shared rows by construction. The independent check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("EVENT_SNAP", "2.5")
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
FIXED_SEEDS = tuple(range(701, 717))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", required=True)
    ap.add_argument("--hpool", type=int, default=400000)
    ap.add_argument("--nfixed", type=int, default=16)
    ap.add_argument("--out", default="research/w4_placebo3_results.json")
    args = ap.parse_args()
    fixed = FIXED_SEEDS[:args.nfixed]
    print("w4_placebo3. is the placebo an overlap artifact. CPU, nothing "
          "generated.", flush=True)
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

    # Same rng call order as w4_placebo, so the pool is identical.
    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")
    elig = np.flatnonzero(Lall >= MMAX)
    elig = rng.choice(elig, min(args.hpool, len(elig)), replace=False)
    hc = np.asarray(Call[elig], dtype=np.float32)
    ccuts = cell_cuts(hc)
    hcell, mcell = cells(hc, ccuts), cells(CD, ccuts)
    want = np.bincount(mcell, minlength=NCELL)
    pool = [elig[hcell == c] for c in range(NCELL)]
    rs = np.random.default_rng(4242)
    poolA, poolB = [], []
    for c in range(NCELL):
        g = rs.permutation(pool[c])
        h = len(g) // 2
        poolA.append(g[:h]); poolB.append(g[h:])
    print(f"  MMAX {MMAX}   standing in for {nmodel} model rows   "
          f"human pool {len(elig)}   {len(fixed)} fixed stand ins", flush=True)

    def human(ids):
        """Features in SORTED id order, and the sorted ids, so rows can be
        matched back to corpus ids. corpus_tokens sorts; this keeps the map."""
        ids = np.sort(ids)
        a, b, c, _, L = corpus_tokens(ids)
        a, b, c, kp = prep(a, b, c, L, MMAX)
        if not kp.all():
            raise SystemExit("pool row shorter than MMAX, pool is wrong")
        return {m: uni_m(a, b, c, m) for m in MS}, ids

    def null_block(armpool):
        nac = []
        for sd in NULL_SEEDS:
            ids, _ = draw_matched(armpool, want, np.random.default_rng(sd),
                                  mult=2)
            o = np.random.default_rng(sd + 50).permutation(len(ids))
            ids = ids[o]
            h = len(ids) // 2
            (A, _), (B, _) = human(ids[:h]), human(ids[h:])
            nac.append([auc(A[m], B[m], sd + 7) for m in MS])
        nac = np.asarray(nac)
        return nac.mean(0), nac.std(0, ddof=1) / np.sqrt(len(NULL_SEEDS))

    def placebo_block(tag, HU, FX, null, nsd, dedup):
        pl, ov = [], []
        for F, fids in FX:
            row = []
            for d in range(DRAWS):
                H, hids = HU[d]
                shared = np.isin(fids, hids)
                ov.append(shared.mean())
                if dedup:
                    Fm = {m: F[m][~shared] for m in MS}
                else:
                    Fm = F
                row.append([auc(H[m], Fm[m], 8300 + d) for m in MS])
            pl.append(np.mean(row, 0))
        pl = np.asarray(pl)
        pmean, psd = pl.mean(0) - null, pl.std(0, ddof=1) / np.sqrt(len(FX))
        print(f"\n  {tag}   overlap mean {100 * np.mean(ov):.2f} percent of "
              f"fixed rows, max {100 * np.max(ov):.2f}")
        print(f"  {'m':>5}{'null':>9}{'placebo':>10}{'p se':>9}"
              f"{'null se':>10}{'p/se':>8}")
        for i, m in enumerate(MS):
            z = pmean[i] / max(nsd[i], 1e-9)
            print(f"  {m:>5}{null[i]:>9.4f}{pmean[i]:>+10.4f}{psd[i]:>9.4f}"
                  f"{nsd[i]:>10.4f}{z:>8.2f}", flush=True)
        return {"null": null.tolist(), "null_se": nsd.tolist(),
                "placebo": pmean.tolist(), "placebo_se": psd.tolist(),
                "overlap_mean": float(np.mean(ov)),
                "overlap_max": float(np.max(ov))}

    out = {}
    # ---- shared pool. the arm's draws and the null, verbatim w4_placebo ----
    null0, nsd0 = null_block(pool)
    HU0 = [human(draw_matched(pool, want, np.random.default_rng(8200 + d))[0])
           for d in range(DRAWS)]
    FX0 = [human(draw_matched(pool, want, np.random.default_rng(fs))[0])
           for fs in fixed]
    out["V0"] = placebo_block("V0 shared pool, AS PUBLISHED", HU0, FX0,
                              null0, nsd0, dedup=False)
    out["V0dedup"] = placebo_block("V0 DEDUP, shared rows removed from the "
                                   "stand in", HU0, FX0, null0, nsd0,
                                   dedup=True)
    del HU0, FX0
    # ---- disjoint half pools ---------------------------------------------
    null1, nsd1 = null_block(poolB)
    HU1 = [human(draw_matched(poolB, want, np.random.default_rng(8200 + d))[0])
           for d in range(DRAWS)]
    FX1 = [human(draw_matched(poolA, want, np.random.default_rng(fs))[0])
           for fs in fixed]
    out["V1"] = placebo_block("V1 DISJOINT POOL", HU1, FX1, null1, nsd1,
                              dedup=False)

    def zero2(v):   # within 2 null se at every m, and no 3 sd point
        return all(abs(v["placebo"][i]) <= 2 * v["null_se"][i]
                   for i in range(len(MS)))
    z0, zd, z1 = zero2(out["V0"]), zero2(out["V0dedup"]), zero2(out["V1"])
    print(f"\n  ZERO means |placebo| within 2 null se at every m")
    print(f"    V0 {'zero' if z0 else 'biased'}   V0dedup "
          f"{'zero' if zd else 'biased'}   V1 {'zero' if z1 else 'biased'}")
    if (not z0) and zd and z1:
        v = ("OVERLAP. the placebo is a construction artifact of the control, "
             "the arm carries none of it, the correction is DISCARDED and "
             "w4_prefix's published lifts stand uncorrected")
    elif (not z0) and (not zd) and (not z1):
        v = ("NOT OVERLAP. removing shared rows and using disjoint pools both "
             "leave it. mechanism unknown, correction stays applied")
    else:
        v = (f"MIXED, dedup {'zero' if zd else 'biased'} disjoint "
             f"{'zero' if z1 else 'biased'}, read by hand")
    print(f"  VERDICT  {v}")
    out["verdict"] = v
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
