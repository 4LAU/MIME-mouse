"""Does sampling temperature widen the bulk, or only add tails.

`research/w4_temp.py` found that raising the temperature from 1.00 to 1.10 sets
the contract's collapse count from three or four to zero on both seeds and costs
0.0539 of AUC, nine times the replicate noise floor. Both readouts come from the
same feature matrix and the same scorer, and `dispersion_ratios` is reported
rather than fed to the forest, so the two are genuinely opposed and not an
artifact of one of them.

A std ratio crossing 0.2 upward on three separate features from a ten percent
change in temperature is not the middle of a distribution moving. This asks
whether it is the tails.

  1. Per feature: the std ratio the collapse flag reads, an interquartile ratio,
     and a ten percent trimmed std ratio. The std is a second moment and a
     handful of wild rows in fifteen hundred can move it. The IQR cannot be
     moved by anything outside the middle half.

  2. Per feature: the share of the generated variance contributed by the one
     percent of rows furthest from the median.

  3. The contract AUC at the high temperature, recomputed with the one percent
     most extreme generated rows removed.

Readout 3 is a DIAGNOSTIC. Nothing is regenerated, nothing is selected for a
deliverable, no configuration in this repository drops rows. It answers one
question: is the 0.0539 carried by fifteen rows or by fifteen hundred.

Safety. Scores through research/autoloop/scoring.py only. Never modifies scoring
code, never training/candi_polar_flow_best.pt. Paces itself on GPU temperature:
this machine crashed on this workload on 2026-08-06, kill line tightened to 79C.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_disp.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_SNAP", "2.5")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from w4_latent import cooldown, gpu_c  # noqa: E402
from w4_paired import gen, specs_for  # noqa: E402


def spreads(X):
    """Three measures of spread per column, ordered by how much of the sample
    they let a single row touch."""
    q = np.percentile(X, [25, 75, 5, 95], axis=0)
    iqr = q[1] - q[0]
    trimmed = np.array([X[(X[:, j] >= q[2, j]) & (X[:, j] <= q[3, j]), j].std()
                        for j in range(X.shape[1])])
    return X.std(0), iqr, trimmed


def tail_share(X, frac=0.01):
    """Fraction of each column's total squared deviation carried by the `frac`
    of rows furthest from that column's median."""
    d2 = (X - np.median(X, axis=0)) ** 2
    k = max(1, int(round(frac * len(X))))
    part = np.sort(d2, axis=0)[-k:].sum(0)
    return part / np.maximum(d2.sum(0), 1e-30)


def extreme_rows(G, H, frac=0.01):
    """Indices of the `frac` most extreme generated rows, extremity measured as
    the largest per feature deviation from the HUMAN median in units of the
    HUMAN interquartile range, so the ranking does not depend on the generated
    sample's own spread."""
    med = np.median(H, axis=0)
    q = np.percentile(H, [25, 75], axis=0)
    scale = np.maximum(q[1] - q[0], 1e-12)
    z = np.abs(G - med) / scale
    k = max(1, int(round(frac * len(G))))
    return np.argsort(-z.max(1))[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rngseed", type=int, default=0)
    ap.add_argument("--lo", type=float, default=1.0)
    ap.add_argument("--hi", type=float, default=1.1)
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--out", default="research/w4_disp.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    print(f"\n  EVENT_SNAP={esp._SNAP} DUR_EMPIRICAL={os.environ['DUR_EMPIRICAL']}"
          f"  human {len(H)}  specs {args.n}  seed {args.rngseed}")

    F = {}
    with torch.no_grad():
        for t in (args.lo, args.hi):
            cooldown()
            rows, meta = specs_for(args.n, args.seed, args.rngseed)
            F[t] = gen(model, rows, meta, args.batch, t, dev, args.rngseed)
            r = scoring.score_features(F[t])
            print(f"  temp {t:.2f}  contract {r['auc_rf_oob']:.4f}  "
                  f"collapse {len(r['collapse_features'])}  n {len(F[t])}  "
                  f"gpu {gpu_c()}")
            if t == args.lo:
                flagged = list(r["collapse_features"])
                auc_lo = float(r["auc_rf_oob"])
            else:
                auc_hi = float(r["auc_rf_oob"])

    hs, hq, ht = spreads(H)
    out = {"flagged_at_lo": flagged, "auc_lo": auc_lo, "auc_hi": auc_hi,
           "features": {}}

    print(f"\n  ratios to human, per feature. * marks flagged at {args.lo:.2f}")
    print(f"  {'feature':<26}{'std lo':>8}{'std hi':>8}{'IQR lo':>8}"
          f"{'IQR hi':>8}{'trim lo':>9}{'trim hi':>9}"
          f"{'tail lo':>9}{'tail hi':>9}")
    S = {t: spreads(F[t]) for t in F}
    T = {t: tail_share(F[t], args.frac) for t in F}
    for j, name in enumerate(FEATURE_NAMES):
        row = {"std_lo": S[args.lo][0][j] / hs[j],
               "std_hi": S[args.hi][0][j] / hs[j],
               "iqr_lo": S[args.lo][1][j] / hq[j],
               "iqr_hi": S[args.hi][1][j] / hq[j],
               "trim_lo": S[args.lo][2][j] / ht[j],
               "trim_hi": S[args.hi][2][j] / ht[j],
               "tail_lo": float(T[args.lo][j]),
               "tail_hi": float(T[args.hi][j]),
               "flagged": name in flagged}
        out["features"][name] = {k: float(v) if k != "flagged" else v
                                 for k, v in row.items()}
        mark = "*" if name in flagged else " "
        print(f"  {mark}{name:<25}{row['std_lo']:>8.3f}{row['std_hi']:>8.3f}"
              f"{row['iqr_lo']:>8.3f}{row['iqr_hi']:>8.3f}"
              f"{row['trim_lo']:>9.3f}{row['trim_hi']:>9.3f}"
              f"{row['tail_lo']:>9.3f}{row['tail_hi']:>9.3f}")

    # READOUT 3. The high temperature sample minus its most extreme one percent.
    idx = extreme_rows(F[args.hi], H, args.frac)
    keep = np.setdiff1d(np.arange(len(F[args.hi])), idx)
    r = scoring.score_features(F[args.hi][keep])
    auc_trim = float(r["auc_rf_oob"])
    # the same surgery on the low temperature arm, so the comparison is not
    # confounded by the removal itself lowering any sample's AUC
    idx_lo = extreme_rows(F[args.lo], H, args.frac)
    keep_lo = np.setdiff1d(np.arange(len(F[args.lo])), idx_lo)
    auc_trim_lo = float(scoring.score_features(F[args.lo][keep_lo])["auc_rf_oob"])
    out.update({"auc_hi_trimmed": auc_trim, "auc_lo_trimmed": auc_trim_lo,
                "n_removed": int(len(idx))})

    print(f"\n  removing the {args.frac:.0%} most extreme generated rows "
          f"({len(idx)} of {len(F[args.hi])})")
    print(f"    temp {args.lo:.2f}   {auc_lo:.4f} -> {auc_trim_lo:.4f}   "
          f"({auc_trim_lo - auc_lo:+.4f})")
    print(f"    temp {args.hi:.2f}   {auc_hi:.4f} -> {auc_trim:.4f}   "
          f"({auc_trim - auc_hi:+.4f})")

    fl = [FEATURE_NAMES.index(n) for n in flagged]
    if fl:
        sr = np.mean([out["features"][FEATURE_NAMES[j]]["std_hi"]
                      / max(out["features"][FEATURE_NAMES[j]]["std_lo"], 1e-9)
                      for j in fl])
        qr = np.mean([out["features"][FEATURE_NAMES[j]]["iqr_hi"]
                      / max(out["features"][FEATURE_NAMES[j]]["iqr_lo"], 1e-9)
                      for j in fl])
    else:
        sr = qr = float("nan")
    out["flagged_std_rise"], out["flagged_iqr_rise"] = float(sr), float(qr)
    print(f"\n  on the {len(fl)} flagged features, mean rise from "
          f"{args.lo:.2f} to {args.hi:.2f}:  std x{sr:.2f}   IQR x{qr:.2f}")

    if sr >= 3.0 and qr < 1.5 and abs(auc_trim - auc_lo) <= 0.010:
        verdict = (f"TAIL INFLATION. Std rises x{sr:.2f} on the flagged features "
                   f"while the IQR rises x{qr:.2f}, and removing "
                   f"{args.frac:.0%} of rows returns the AUC to "
                   f"{auc_trim:.4f} against {auc_lo:.4f}. The collapse flag is a "
                   "second moment artifact on this path and no scalar on the "
                   "logits reaches the under dispersion of the bulk.")
    elif qr >= 1.0 + 0.5 * (sr - 1.0):
        verdict = (f"GENUINE WIDENING. IQR rises x{qr:.2f} against std x{sr:.2f}. "
                   "The temperature really is widening the distribution and the "
                   "AUC loss comes from something other than tails.")
    else:
        verdict = (f"NEITHER BRANCH. std x{sr:.2f}, IQR x{qr:.2f}, trimmed AUC "
                   f"{auc_trim:.4f} against {auc_lo:.4f} at the low "
                   "temperature. Reported as measured, no verdict.")
    out["verdict"] = verdict
    print(f"\n  VERDICT  {verdict}\n")

    np.savez_compressed("research/w4_disp_cache.npz",
                        lo=F[args.lo], hi=F[args.hi])
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out} and research/w4_disp_cache.npz\n")


if __name__ == "__main__":
    main()
