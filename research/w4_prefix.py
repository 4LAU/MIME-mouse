"""w4_prefix. Is the defect already there in the FIRST event?

Registered in /home/aaronadmin/w4_arms/step0_prereg.md, AMENDMENT 5, which
replaces `w4_step0`'s per position histogram after that statistic was shown
under powered for arithmetic reasons rather than fixable ones. Read the
amendment before this file: it says plainly that replacing a failed primary is
the move to be most suspicious of, and why this one is different.

A per position histogram reads ONE event per row. The detector whose lift the
question is about reads about 39 and gets its power by aggregating them. So this
arm aggregates too, over the FIRST m EVENTS ONLY, and sweeps m.

If the per event defect is present from the first event and position stationary,
each event contributes independent evidence and the lift grows like sqrt(m). If
it is absent at the first event and builds, the curve sits BELOW that line at
small m.

CPU only. Nothing generated, no new rows.
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

from models.event_ar import N_DT_CLASSES                          # noqa: E402
from models.event_stream_polar import (S_PAD_CLASS, TH_BINS,      # noqa: E402
                                       TH_NULL_CLASS)
from w4_detcap import (CORPUS, NDT_B, NS_B, NTH_B, corpus_tokens,  # noqa: E402
                       model_tokens, rf_oob)
from w4_poskl import channels, cuts_from, make_cuts               # noqa: E402
from w4_step0 import (NCELL, bin_shift, cell_cuts, cells,         # noqa: E402
                      draw_matched, plant_at, prep)

DRAWS = 8
NULL_SEEDS = (401, 402, 403, 404, 405, 406, 407, 408)
MOD = {"s": S_PAD_CLASS, "th": TH_BINS, "dt": N_DT_CLASSES}
PLANT = {"s": 0.05, "th": 0.02, "dt": 0.05}   # the AMENDMENT 3 calibration


def uni_m(s, th, dt, m):
    """The 49 column coarse unigram rate vector over the FIRST m events.

    `w4_detcap.build` cannot be used here: it drops every row shorter than three
    events, and this arm has to read m = 1. The coarsening is copied from
    `bin_streams` exactly so the alphabet is the detector's own.

    The length fraction column is deliberately NOT included. Every row entering
    has length >= MMAX >= m, so it would be constant, and if it were ever not
    constant it would leak length into a question that is not about length.
    """
    sj, tj, dj = s[:, :m], th[:, :m], dt[:, :m]
    sb = np.minimum(sj, S_PAD_CLASS - 1) * NS_B // S_PAD_CLASS
    tb = np.where(tj >= TH_NULL_CLASS, NTH_B,
                  np.minimum(tj, TH_NULL_CLASS - 1) * NTH_B // TH_NULL_CLASS)
    db = np.minimum(dj, N_DT_CLASSES - 1) * NDT_B // N_DT_CLASSES
    n = len(s)
    row = np.arange(n)[:, None]

    def hist(codes, k):
        return (np.bincount((row * k + codes).ravel(),
                            minlength=n * k).reshape(n, k) / m)

    return np.concatenate([hist(sb, NS_B), hist(tb, NTH_B + 1),
                           hist(db, NDT_B)], 1).astype(np.float32)


def auc(A, B, seed):
    """Null corrected elsewhere. Balanced, shuffled, OOB so one fit per read."""
    r = np.random.default_rng(8100 + seed)
    A, B = A[r.permutation(len(A))], B[r.permutation(len(B))]
    n = min(len(A), len(B))
    X = np.vstack([A[:n], B[:n]])
    y = np.r_[np.zeros(n), np.ones(n)]
    return rf_oob(X, y)


def run(mmax, ms, args):
    rng = np.random.default_rng(8000 + mmax)

    S, TH, DT, CD, LL = [], [], [], [], []
    for f in args.streams:
        a, b, c, d, L = model_tokens(f)
        S.append(a); TH.append(b); DT.append(c); CD.append(d); LL.append(L)
    S, TH, DT = np.concatenate(S), np.concatenate(TH), np.concatenate(DT)
    CD, LL = np.concatenate(CD), np.concatenate(LL)
    S, TH, DT, keep = prep(S, TH, DT, LL, mmax)
    CD = CD[keep]

    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")
    elig = np.flatnonzero(Lall >= mmax)
    elig = rng.choice(elig, min(args.hpool, len(elig)), replace=False)
    hc = np.asarray(Call[elig], dtype=np.float32)

    ccuts = cell_cuts(hc)
    hcell, mcell = cells(hc, ccuts), cells(CD, ccuts)
    want = np.bincount(mcell, minlength=NCELL)
    pool = [elig[hcell == c] for c in range(NCELL)]
    print(f"\n  MMAX {mmax}   model rows {len(S)}   human pool {len(elig)}",
          flush=True)

    # Plant shifts come from a plain human reference, never from the model.
    ref_ids = rng.choice(elig, min(20000, len(elig)), replace=False)
    rs, rth, rdt, _, rL = corpus_tokens(ref_ids)
    rs, rth, rdt, _ = prep(rs, rth, rdt, rL, mmax)
    rcuts = make_cuts(channels(rs, rth, rdt))
    shifts = {c: bin_shift(rcuts[c]) for c in ("s", "th", "dt")}

    def human(ids):
        a, b, c, _, L = corpus_tokens(ids)
        a, b, c, _ = prep(a, b, c, L, mmax)
        return a, b, c

    MU = {m: uni_m(S, TH, DT, m) for m in ms}

    # ---- ARM ---------------------------------------------------------------
    acc = []
    for d in range(DRAWS):
        ids, _ = draw_matched(pool, want, np.random.default_rng(8200 + d))
        a, b, c = human(ids)
        acc.append([auc(uni_m(a, b, c, m), MU[m], 8300 + d) for m in ms])
    arm = np.mean(acc, 0)

    # ---- P1 NULL, identical pipeline, and the plants ride on it ------------
    def null_pair(seed, plant=None):
        ids, _ = draw_matched(pool, want, np.random.default_rng(seed), mult=2)
        o = np.random.default_rng(seed + 50).permutation(len(ids))
        ids = ids[o]
        h = len(ids) // 2
        A, B = human(ids[:h]), human(ids[h:])
        if plant is not None:
            pr, ks, ch, sh = plant
            B = plant_at(*B, np.random.default_rng(seed + 90), pr, ks, ch, sh,
                         MOD[ch])
        return [auc(uni_m(*A, m), uni_m(*B, m), seed + 7) for m in ms]

    nacc = [null_pair(sd) for sd in NULL_SEEDS]
    null = np.mean(nacc, 0)
    nsd = np.std(nacc, 0, ddof=1) / np.sqrt(len(NULL_SEEDS))
    lift = arm - null

    print(f"  {'m':>5}{'arm':>9}{'null':>9}{'lift':>9}{'null se':>10}")
    for i, m in enumerate(ms):
        print(f"  {m:>5}{arm[i]:>9.4f}{null[i]:>9.4f}{lift[i]:>+9.4f}"
              f"{nsd[i]:>10.4f}", flush=True)

    # ---- P3. does the large m end follow a square root law? ---------------
    mm = np.array(ms, dtype=np.float64)
    fitmask = mm >= 10
    c = float((np.sqrt(mm[fitmask]) * lift[fitmask]).sum()
              / (mm[fitmask]).sum())
    pred = c * np.sqrt(mm)
    resid = lift - pred
    fit_rms = float(np.sqrt(np.mean(resid[fitmask] ** 2)))
    sd_scale = float(np.mean(nsd))
    p3 = bool(fit_rms <= 2.0 * sd_scale)
    print(f"\n  P3 sqrt(m) fit on m >= 10, c = {c:.5f}")
    print(f"  {'m':>5}{'lift':>9}{'c*sqrt(m)':>12}{'resid':>9}{'resid/sd':>10}")
    for i, m in enumerate(ms):
        print(f"  {m:>5}{lift[i]:>+9.4f}{pred[i]:>12.4f}{resid[i]:>+9.4f}"
              f"{resid[i] / max(sd_scale, 1e-9):>10.2f}")
    print(f"  fitted rms {fit_rms:.4f} against noise sd {sd_scale:.4f}  -> "
          f"{'PASS' if p3 else 'FAIL, the extrapolation is not a baseline'}")

    # ---- P2. power, both directions ---------------------------------------
    ks0, kslate = np.array([0]), np.arange(5, mmax)
    # PAIRED. The first run subtracted an EIGHT seed unplanted mean from a FOUR
    # seed planted mean, so the two carried different draws and the difference
    # carried both their noise. The proof is in that run's own table: the late
    # plant touches nothing at m = 2, 3, 5, so those entries had to be exactly
    # zero and read -0.0093, -0.0053, -0.0034. Same seed, same ids, same split,
    # plant on or off, differenced per seed.
    PSEEDS = NULL_SEEDS[:4]
    base = np.array([nacc[NULL_SEEDS.index(sd)] for sd in PSEEDS])
    pw = {}
    for tag, ks in (("k0_only", ks0), ("late_only", kslate)):
        pw[tag] = {}
        for ch in ("s", "th", "dt"):
            pa = np.array([null_pair(sd, plant=(PLANT[ch], ks, ch, shifts[ch]))
                           for sd in PSEEDS])
            pw[tag][ch] = (pa - base).mean(0).tolist()
    print(f"\n  P2 POWER, calibrated plant. k0 only must show at m = 1 and "
          f"decay, late only must be zero at m = 1, 2, 3")
    print(f"  {'ch':>5}{'where':>11}" + "".join(f"{m:>8}" for m in ms))
    for tag in ("k0_only", "late_only"):
        for ch in ("s", "th", "dt"):
            print(f"  {ch:>5}{tag:>11}" +
                  "".join(f"{v:>+8.4f}" for v in pw[tag][ch]))
    k0v = np.array([pw["k0_only"][c] for c in ("s", "th", "dt")]).mean(0)
    ltv = np.array([pw["late_only"][c] for c in ("s", "th", "dt")]).mean(0)
    sees_k0 = bool(k0v[0] > 3 * sd_scale and k0v[-1] < 0.6 * k0v[0])
    blind_late = bool(abs(ltv[0]) < 2 * sd_scale and ltv[-1] > 3 * sd_scale)
    print(f"  k0 plant seen at m=1 and decaying  {sees_k0}")
    print(f"  late plant invisible at m=1 and seen at m={ms[-1]}  {blind_late}")

    p1 = bool(np.all(np.abs(null - 0.5) < 3 * np.maximum(nsd, 1e-9)))
    print(f"  P1 null within band at every m  {p1}")

    # ---- verdict -----------------------------------------------------------
    small = mm <= 3
    r_small = resid[small] / max(sd_scale, 1e-9)
    l1_sd = lift[0] / max(sd_scale, 1e-9)
    if not (p3 and sees_k0 and blind_late and p1):
        v = "UNREADABLE"
    elif l1_sd >= 3 and np.all(np.abs(r_small) <= 2):
        v = "PRESENT AT ONE"
    elif l1_sd < 2 and np.all(r_small < -2):
        v = "BUILDS EARLY"
    else:
        v = "MIXED"
    print(f"\n  lift(1) = {lift[0]:+.4f}, {l1_sd:.2f} sd. small m residuals "
          f"{np.round(r_small, 2).tolist()} sd")
    print(f"  VERDICT MMAX {mmax}  {v}")

    return dict(mmax=mmax, n_model=int(len(S)), ms=ms, arm=arm.tolist(),
                null=null.tolist(), null_se=nsd.tolist(), lift=lift.tolist(),
                c=c, resid=resid.tolist(), sd_scale=sd_scale,
                controls=dict(p1=p1, p3=p3, sees_k0=sees_k0,
                              blind_late=blind_late),
                power=pw, verdict=v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+",
                    default=["research/w4_texcover_streams_s0.npz",
                             "research/w4_texcover_streams_s1.npz"])
    ap.add_argument("--hpool", type=int, default=400000)
    ap.add_argument("--out", default="research/w4_prefix_results.json")
    args = ap.parse_args()
    print("w4_prefix. detector on the first m events only. CPU, nothing "
          "generated.", flush=True)
    out = {"40": run(40, [1, 2, 3, 5, 10, 20, 40], args),
           "20": run(20, [1, 2, 3, 5, 10, 20], args)}
    a, b = out["40"]["verdict"], out["20"]["verdict"]
    out["final"] = a if a == b else f"MIXED, {a} at MMAX 40 and {b} at MMAX 20"
    print(f"\n  FINAL  {out['final']}")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
