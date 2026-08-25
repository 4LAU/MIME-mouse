"""w4_condshare. How much of the measured defect is QUERY CONDITION MISMATCH?

Registered in /home/aaronadmin/w4_arms/condshare_prereg.md before any code.
Read it first. The short version: generated rows come from `make_specs`, which
draws distance from the eval set's distance file, angle UNIFORMLY, and duration
from a fitted model. Corpus rows carry whatever real usage produced. `w4_detcap`
null corrects with two CORPUS arms, which cannot cancel a mismatch that only the
model arm carries. So every corpus referenced number is at risk, including
detcap's 71/29 split of the defect into token rates and composition.

THE CONTRACT IS NOT IMPLICATED. `make_specs` mirrors `evaluate.py`'s spec loop
and draws its distances from the eval set's own file, so against the contract's
human reference distance is matched by construction. Nothing here puts 0.5792 in
doubt.

The ladder changes ONE thing per rung:

    A  unmatched, all rows (L >= 5), unigram rates + length fraction
    B  unmatched, all rows (L >= 5), unigram rates only
    C  unmatched, L >= 40, unigram rates only, subsampled to D's row count
    D  MATCHED,   L >= 40, unigram rates only

and the same rungs on the 18 contract features, where A and B coincide because
there is no length column to drop.

CPU only. Nothing generated, no checkpoint touched.
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

from w4_detcap import (CORPUS, IDX_UNI, TBUF, bin_streams,        # noqa: E402
                       corpus_tokens, model_tokens, rate_features, rf_oob)
from w4_views import decode_features                              # noqa: E402
from w4_prefix import MOD                                         # noqa: E402
from w4_step0 import (NCELL, bin_shift, cell_cuts, cells,         # noqa: E402
                      draw_matched, plant_at)
from w4_poskl import channels, make_cuts                          # noqa: E402

LMIN_LONG = 40
DRAWS = 8
NULL_SEEDS = (601, 602, 603, 604, 605, 606, 607, 608)
K2_SEEDS = (611, 612, 613, 614)
K2_RATES = (0.01, 0.03, 0.06)
K2_CH = "s"
# `hum` returns rate features ALREADY reduced to IDX_UNI, which is the 49
# unigram columns followed by the trailing length fraction. No rung here uses
# any other rate column, and carrying the full 1400 column block would make the
# cache below cost gigabytes instead of megabytes.
UNI = np.arange(49)                  # the 49 unigram rate columns
UNI_LEN = np.arange(50)              # the same 49 plus the length fraction


def build2(s, th, dt, cond, L):
    """`w4_detcap.build`, plus the survival mask it does not return.

    The mask is needed because the matched draw's per cell counts must be
    computed on the model rows that actually REACH the detector, not on the rows
    that were loaded. Nothing else differs; the feature code is imported, not
    copied, so it cannot drift.
    """
    ang = np.arctan2(cond[:, 3], cond[:, 2]).astype(np.float64)
    rates = rate_features(bin_streams(s, th, dt, L))
    F18, FR, keep = [], [], np.zeros(len(s), bool)
    for j in range(len(s)):
        if rates[j] is None:
            continue
        n = int(L[j])
        f = decode_features(s[j, :n], th[j, :n], dt[j, :n], float(ang[j]))
        if f is None:
            continue
        F18.append(f); FR.append(rates[j]); keep[j] = True
    return np.asarray(F18, float), np.asarray(FR, float), keep


def auc(A, B, seed):
    """Balanced, shuffled, OOB. One fit per read."""
    r = np.random.default_rng(9600 + seed)
    A, B = A[r.permutation(len(A))], B[r.permutation(len(B))]
    n = min(len(A), len(B))
    X = np.vstack([A[:n], B[:n]])
    y = np.r_[np.zeros(n), np.ones(n)]
    return rf_oob(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+",
                    default=["research/w4_texcover_streams_s0.npz",
                             "research/w4_texcover_streams_s1.npz"])
    ap.add_argument("--hpool", type=int, default=400000)
    ap.add_argument("--out", default="research/w4_condshare_results.json")
    args = ap.parse_args()
    print("w4_condshare. is the measured defect partly a condition mismatch. "
          "CPU, nothing generated.", flush=True)
    rng = np.random.default_rng(9500)

    # ---- model rows --------------------------------------------------------
    S, TH, DT, CD, LL = [], [], [], [], []
    for f in args.streams:
        a, b, c, d, L = model_tokens(f)
        S.append(a); TH.append(b); DT.append(c); CD.append(d); LL.append(L)
    S, TH, DT = np.concatenate(S), np.concatenate(TH), np.concatenate(DT)
    CD, LL = np.concatenate(CD), np.concatenate(LL)

    # Reduced to IDX_UNI on the spot, so the model side carries exactly the
    # same 50 columns the human side does. Forgetting this once cost a run.
    M18_all, MR_all, _ = build2(S, TH, DT, CD, LL)
    MR_all = MR_all[:, IDX_UNI]
    lng = np.asarray(LL) >= LMIN_LONG
    M18_lng, MR_lng, kl = build2(S[lng], TH[lng], DT[lng], CD[lng], LL[lng])
    MR_lng = MR_lng[:, IDX_UNI]
    CD_lng = CD[lng][kl]
    n_all, n_lng = len(MR_all), len(MR_lng)
    print(f"  model rows {len(S)} loaded, {n_all} reach the detector, "
          f"{n_lng} of those have length >= {LMIN_LONG}", flush=True)

    # ---- human pools -------------------------------------------------------
    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")
    pool_any = np.flatnonzero(Lall >= 5)          # detcap's own eligibility
    elig = np.flatnonzero(Lall >= LMIN_LONG)
    elig = rng.choice(elig, min(args.hpool, len(elig)), replace=False)
    hc = np.asarray(Call[elig], dtype=np.float32)

    ccuts = cell_cuts(hc)
    hcell, mcell = cells(hc, ccuts), cells(CD_lng, ccuts)
    want = np.bincount(mcell, minlength=NCELL)
    pool = [elig[hcell == c] for c in range(NCELL)]

    # ---- K4 CELL SUPPLY, read before anything is interpreted ---------------
    _, short = draw_matched(pool, want, np.random.default_rng(1))
    frac_short = float(short.sum()) / max(int(want.sum()), 1)
    print(f"  K4 cell supply shortfall {100 * frac_short:.2f} percent of "
          f"model rows  -> {'ok' if frac_short <= 0.05 else 'CONTAMINATED'}",
          flush=True)

    # ---- plant shift, from a plain human reference, never from the model ---
    ref = rng.choice(elig, 20000, replace=False)
    rs, rth, rdt, _, rL = corpus_tokens(ref)
    rcuts = make_cuts(channels(rs, rth, rdt))
    shift = bin_shift(rcuts[K2_CH])
    del rs, rth, rdt
    print(f"  K2 plant is channel {K2_CH}, shift {shift} classes, applied at "
          f"EVERY position and at a rate independent of the condition",
          flush=True)

    # ---- feature builders per source --------------------------------------
    # A, B and A18 read the SAME human draw, as do C and C18, and D and D18.
    # Without this cache each draw is decoded three times, which is most of the
    # run time and none of the information.
    hcache = {}

    def hum(ids, plant=None):
        key = (hash(np.sort(ids).tobytes()), plant)
        if key in hcache:
            return hcache[key]
        a, b, c, d, L = corpus_tokens(ids)
        if plant is not None:
            pr, sd = plant
            a, b, c = plant_at(a, b, c, np.random.default_rng(sd), pr,
                               np.arange(TBUF), K2_CH, shift, MOD[K2_CH])
        f18, fr, _ = build2(a, b, c, d, L)
        hcache[key] = (f18, fr[:, IDX_UNI])
        return hcache[key]

    def draw_any(n, sd):
        return np.random.default_rng(sd).choice(pool_any, n, replace=False)

    def draw_long(n, sd):
        return np.random.default_rng(sd).choice(elig, n, replace=False)

    def draw_match(sd, mult=1):
        ids, _ = draw_matched(pool, want, np.random.default_rng(sd), mult=mult)
        return ids

    # ---- the four rungs ----------------------------------------------------
    # Each rung is (model features, a human draw function, the column slice).
    # Rung C draws EXACTLY n_lng rows so it differs from D only in matching.
    RUNGS = {
        "A":  (MR_all, UNI_LEN, lambda sd: draw_any(n_all, sd)),
        "B":  (MR_all, UNI,     lambda sd: draw_any(n_all, sd)),
        "C":  (MR_lng, UNI,     lambda sd: draw_long(n_lng, sd)),
        "D":  (MR_lng, UNI,     lambda sd: draw_match(sd)),
        "A18": (M18_all, None,  lambda sd: draw_any(n_all, sd)),
        "C18": (M18_lng, None,  lambda sd: draw_long(n_lng, sd)),
        "D18": (M18_lng, None,  lambda sd: draw_match(sd)),
    }

    def feats(pair, cols):
        f18, fr = pair
        return f18 if cols is None else fr[:, cols]

    out = {"n_all": n_all, "n_lng": n_lng, "k4_short": frac_short,
           "shift": int(shift), "rungs": {}}
    print(f"\n  {'rung':>5}{'arm':>9}{'null':>9}{'lift':>9}{'null se':>10}"
          f"{'hrows':>8}")
    for name, (MX, cols, drawfn) in RUNGS.items():
        # The model side takes the SAME column slice as the human side. Rung B
        # drops the length fraction from both or it compares 49 columns against
        # 50 and cannot run at all.
        MXc = MX if cols is None else MX[:, cols]
        armv = [auc(feats(hum(drawfn(9700 + d)), cols), MXc, 100 + d)
                for d in range(DRAWS)]
        nulls, hrows = [], 0
        for sd in NULL_SEEDS:
            if name in ("D", "D18"):
                ids = draw_match(sd, mult=2)
            else:
                ids = drawfn(sd + 3000)
                ids = np.r_[ids, drawfn(sd + 4000)]
                ids = np.unique(ids)
            o = np.random.default_rng(sd + 50).permutation(len(ids))
            ids = ids[o]
            h = len(ids) // 2
            hrows = h
            nulls.append(auc(feats(hum(ids[:h]), cols),
                             feats(hum(ids[h:]), cols), sd + 7))
        a, nl = float(np.mean(armv)), float(np.mean(nulls))
        se = float(np.std(nulls, ddof=1) / np.sqrt(len(nulls)))
        out["rungs"][name] = {"arm": a, "null": nl, "lift": a - nl, "se": se,
                              "hrows": hrows}
        print(f"  {name:>5}{a:>9.4f}{nl:>9.4f}{a - nl:>+9.4f}{se:>10.4f}"
              f"{hrows:>8}", flush=True)

    R = out["rungs"]
    k1 = all(abs(R[k]["null"] - 0.5) <= 0.02 for k in R)
    print(f"\n  K1 every null within 0.02 of 0.5  -> "
          f"{'PASS' if k1 else 'FAIL'}")

    # ---- K2. does matching destroy a condition independent plant? ---------
    print(f"\n  K2 CONDITION INDEPENDENT PLANT through C and through D")
    print(f"  {'p':>6}{'lift C':>10}{'lift D':>10}{'D/C':>8}")
    k2 = {}
    for pr in K2_RATES:
        lc, ld = [], []
        for sd in K2_SEEDS:
            idc = draw_long(n_lng * 2, sd + 5000)
            hcut = len(idc) // 2
            lc.append(auc(feats(hum(idc[:hcut]), UNI),
                          feats(hum(idc[hcut:], plant=(pr, sd + 90)), UNI),
                          sd + 11)
                      - auc(feats(hum(idc[:hcut]), UNI),
                            feats(hum(idc[hcut:]), UNI), sd + 11))
            idm = draw_match(sd + 6000, mult=2)
            o = np.random.default_rng(sd + 6050).permutation(len(idm))
            idm = idm[o]
            hm = len(idm) // 2
            ld.append(auc(feats(hum(idm[:hm]), UNI),
                          feats(hum(idm[hm:], plant=(pr, sd + 90)), UNI),
                          sd + 11)
                      - auc(feats(hum(idm[:hm]), UNI),
                            feats(hum(idm[hm:]), UNI), sd + 11))
        mc, md = float(np.mean(lc)), float(np.mean(ld))
        ratio = md / mc if abs(mc) > 1e-9 else float("nan")
        k2[pr] = {"C": mc, "D": md, "ratio": ratio}
        print(f"  {pr:>6.2f}{mc:>+10.4f}{md:>+10.4f}{ratio:>8.2f}", flush=True)
    # Read the rate whose C lift is closest to the model's own C lift.
    tgt = min(K2_RATES, key=lambda p: abs(k2[p]["C"] - R["C"]["lift"]))
    k2ok = bool(k2[tgt]["ratio"] >= 0.7)
    print(f"  the rate matching the model's own C lift {R['C']['lift']:+.4f} "
          f"is p = {tgt:.2f}, D/C {k2[tgt]['ratio']:.2f}  -> "
          f"{'PASS' if k2ok else 'FAIL, matching destroys a known plant and '
                                 'this arm reads NOTHING'}")

    # ---- PRIMARY -----------------------------------------------------------
    share_rate = 1 - R["D"]["lift"] / R["C"]["lift"] if R["C"]["lift"] > 1e-9 \
        else float("nan")
    share_18 = 1 - R["D18"]["lift"] / R["C18"]["lift"] \
        if R["C18"]["lift"] > 1e-9 else float("nan")
    lenfrac = R["A"]["lift"] - R["B"]["lift"]
    print(f"\n  PRIMARY")
    print(f"    cond_share_rate  {100 * share_rate:6.1f} percent of the "
          f"L >= 40 unigram rate lift is condition mismatch")
    print(f"    cond_share_18    {100 * share_18:6.1f} percent of the "
          f"L >= 40 contract feature lift is condition mismatch")
    print(f"    length fraction column is worth {lenfrac:+.4f} on its own "
          f"(A minus B)")
    print(f"    length filter alone costs {R['C']['lift'] - R['B']['lift']:+.4f}"
          f" (C minus B)")
    verdict = ("VOID, K2 failed" if not k2ok else
               "CONTAMINATED, K4 failed" if frac_short > 0.05 else
               "VOID, K1 failed" if not k1 else "READ")
    print(f"\n  VERDICT  {verdict}", flush=True)

    out.update({"k1": k1, "k2": {str(k): v for k, v in k2.items()},
                "k2_target_rate": tgt, "k2_pass": k2ok,
                "cond_share_rate": share_rate, "cond_share_18": share_18,
                "len_frac_worth": lenfrac, "verdict": verdict})
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
