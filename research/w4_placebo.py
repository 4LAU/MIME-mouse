"""w4_placebo. Is the arm minus null difference itself biased?

Registered in /home/aaronadmin/w4_arms/step0_prereg.md, AMENDMENT 9. This is the
control w4_prefix should have carried from the start.

The arm compares ONE FIXED set of model rows against eight fresh human draws.
The null compares two fresh human draws against each other. Those are different
procedures and a bias that depends on one side being fixed does not cancel in
the difference. So: put a FIXED set of HUMAN rows where the model arm goes and
read what the pipeline returns when the true answer is zero.

CPU only. Nothing generated, no model row read at all.
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

from models.event_stream_polar import TH_NULL_CLASS, TH_BINS       # noqa: E402
from w4_detcap import CORPUS, NTH_B, corpus_tokens, model_tokens    # noqa: E402
from w4_prefix import auc, uni_m                                    # noqa: E402
from w4_prefixch import COLS                                        # noqa: E402
from w4_step0 import NCELL, cell_cuts, cells, draw_matched, prep     # noqa: E402

MMAX = 40
MS = (1, 2, 3, 5, 10, 20, 40)
DRAWS = 8
NULL_SEEDS = (401, 402, 403, 404, 405, 406, 407, 408)
# Four fixed stand ins was the 2026-08-18 run. OUTCOME 8 found the model
# side of ONE draw was the binding constraint, so the rerun on ten streams
# widens this to sixteen. The first four are unchanged so the 2 stream
# numbers reproduce from the prefix of the list.
FIXED_SEEDS = tuple(range(701, 717))
ALL49 = np.arange(49)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+",
                    default=["research/w4_texcover_streams_s0.npz",
                             "research/w4_texcover_streams_s1.npz"])
    ap.add_argument("--hpool", type=int, default=400000)
    ap.add_argument("--out", default="research/w4_placebo_results.json")
    ap.add_argument("--nfixed", type=int, default=4,
                    help="fixed stand ins, 4 reproduces 2026-08-18")
    # The lift being corrected must come from the prefix run on the SAME
    # streams. Empty means the published 2 stream table below.
    ap.add_argument("--prefix-json", default="")
    args = ap.parse_args()
    fixed_seeds = FIXED_SEEDS[:args.nfixed]
    print("w4_placebo. what does the arm pipeline return when the answer is "
          "zero. CPU, nothing generated.", flush=True)
    rng = np.random.default_rng(8000 + MMAX)

    # Model rows are loaded ONLY to reproduce the matched draw's cell counts.
    S, TH, DT, CD, LL = [], [], [], [], []
    for f in args.streams:
        a, b, c, d, L = model_tokens(f)
        S.append(a); TH.append(b); DT.append(c); CD.append(d); LL.append(L)
    S, TH, DT = np.concatenate(S), np.concatenate(TH), np.concatenate(DT)
    CD, LL = np.concatenate(CD), np.concatenate(LL)
    S, TH, DT, keep = prep(S, TH, DT, LL, MMAX)
    CD, nmodel = CD[keep], len(S)
    del S, TH, DT

    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")
    elig = np.flatnonzero(Lall >= MMAX)
    elig = rng.choice(elig, min(args.hpool, len(elig)), replace=False)
    hc = np.asarray(Call[elig], dtype=np.float32)
    ccuts = cell_cuts(hc)
    hcell, mcell = cells(hc, ccuts), cells(CD, ccuts)
    want = np.bincount(mcell, minlength=NCELL)
    pool = [elig[hcell == c] for c in range(NCELL)]
    print(f"  MMAX {MMAX}   standing in for {nmodel} model rows   "
          f"human pool {len(elig)}", flush=True)

    def human(ids):
        a, b, c, _, L = corpus_tokens(ids)
        a, b, c, _ = prep(a, b, c, L, MMAX)
        return a, b, c

    # ---- the arm's own human draws and the null, verbatim ------------------
    arm_h = [human(draw_matched(pool, want, np.random.default_rng(8200 + d))[0])
             for d in range(DRAWS)]
    HU = [{m: uni_m(*h, m) for m in MS} for h in arm_h]

    null_h = []
    for sd in NULL_SEEDS:
        ids, _ = draw_matched(pool, want, np.random.default_rng(sd), mult=2)
        o = np.random.default_rng(sd + 50).permutation(len(ids))
        ids = ids[o]
        h = len(ids) // 2
        A, B = human(ids[:h]), human(ids[h:])
        null_h.append(({m: uni_m(*A, m) for m in MS},
                       {m: uni_m(*B, m) for m in MS}))

    # ---- the fixed stand ins ----------------------------------------------
    FX = []
    for fs in fixed_seeds:
        ids, _ = draw_matched(pool, want, np.random.default_rng(fs))
        h = human(ids)
        FX.append({m: uni_m(*h, m) for m in MS})
    print(f"  {len(FX)} independent fixed human stand ins, "
          f"{len(FX[0][1])} rows each", flush=True)

    SETS = {"all49": ALL49, **COLS}
    out, bad = {}, []
    print(f"\n  {'cols':>6}{'m':>5}{'null':>9}{'placebo':>10}{'p se':>9}"
          f"{'null se':>10}{'p/se':>8}")
    for name, cc in SETS.items():
        nac = np.array([[auc(A[m][:, cc], B[m][:, cc], NULL_SEEDS[i] + 7)
                         for m in MS] for i, (A, B) in enumerate(null_h)])
        null, nsd = nac.mean(0), nac.std(0, ddof=1) / np.sqrt(len(NULL_SEEDS))
        pl = np.array([[np.mean([auc(HU[d][m][:, cc], F[m][:, cc], 8300 + d)
                                 for d in range(DRAWS)]) for m in MS]
                       for F in FX])
        pmean, psd = pl.mean(0) - null, pl.std(0, ddof=1) / np.sqrt(len(FX))
        out[name] = {"null": null.tolist(), "placebo": pmean.tolist(),
                     "placebo_se": psd.tolist(), "null_se": nsd.tolist()}
        for i, m in enumerate(MS):
            z = pmean[i] / max(nsd[i], 1e-9)
            print(f"  {name:>6}{m:>5}{null[i]:>9.4f}{pmean[i]:>+10.4f}"
                  f"{psd[i]:>9.4f}{nsd[i]:>10.4f}{z:>8.2f}", flush=True)
            if name == "all49" and abs(z) > 2:
                bad.append((m, float(pmean[i]), float(z)))

    p0 = all(abs(out["all49"]["placebo"][i])
             <= out["all49"]["null_se"][i] for i in range(len(MS)))
    print(f"\n  P0 CLEAN, |placebo| within 1 null se at every m on the 49 "
          f"column detector  -> {'PASS' if p0 else 'FAIL'}")
    if bad:
        print(f"  P1 BIASED at m = {[b[0] for b in bad]}, "
              f"placebo {[round(b[1], 4) for b in bad]}")
    print(f"\n  w4_prefix's published lift, and the same lift corrected")
    if args.prefix_json:
        pj = json.load(open(args.prefix_json))["40"]
        PUB = dict(zip(pj["ms"], pj["lift"]))
        pub_growth = PUB[40] / PUB[1]
        print(f"    lift table from {args.prefix_json}")
    else:
        PUB = {1: 0.0176, 2: 0.0141, 3: 0.0073, 5: 0.0101, 10: 0.0146,
               20: 0.0183, 40: 0.0174}
        pub_growth = 0.99
    corr = {}
    for i, m in enumerate(MS):
        c = PUB[m] - out["all49"]["placebo"][i]
        corr[m] = c
        print(f"    m {m:>3}   published {PUB[m]:+.4f}   placebo "
              f"{out['all49']['placebo'][i]:+.4f}   corrected {c:+.4f}")
    g = corr[40] / corr[1] if abs(corr[1]) > 1e-9 else float("nan")
    print(f"    corrected growth lift(40)/lift(1) = {g:.2f}, "
          f"uncorrected {pub_growth:.2f}")

    # ---- P2. the th plant check -------------------------------------------
    ids, _ = draw_matched(pool, want, np.random.default_rng(999))
    _, th, _ = human(ids)
    live = th < TH_NULL_CLASS
    def frac(sh):
        a = np.minimum(th, TH_NULL_CLASS - 1) * NTH_B // TH_NULL_CLASS
        t2 = (th + sh) % TH_BINS
        b = np.minimum(t2, TH_NULL_CLASS - 1) * NTH_B // TH_NULL_CLASS
        return float(((a != b) & live).sum() / max(live.sum(), 1))
    f4, f16 = frac(4), frac(16)
    print(f"\n  P2 th PLANT CHECK. fraction of live direction tokens whose "
          f"DETECTOR bin moves")
    print(f"    4 class shift, the one w4_k0power used   {100 * f4:.1f} percent")
    print(f"    16 class shift, one full detector bin    {100 * f16:.1f} percent")
    print(f"    -> the ladder's th blindness is "
          f"{'a plant construction artifact' if f4 < 0.35 < f16 else 'NOT explained by the shift size'}")

    out.update({"p0": p0, "bad": bad, "corrected": {str(k): v for k, v
                                                    in corr.items()},
                "corrected_growth": g, "uncorrected_growth": pub_growth,
                "published": {str(k): v for k, v in PUB.items()},
                "n_fixed": len(fixed_seeds), "th_frac_shift4": f4,
                "th_frac_shift16": f16})
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
