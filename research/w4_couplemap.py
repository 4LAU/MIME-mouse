"""Which couplings are wrong, with the marginals taken out of the picture?

`w4_copula` established the shape of the problem. Replacing every one of the
eighteen generated feature distributions with the human one, exactly, while
leaving the model's own rank structure untouched, moves the detector from
0.6668 to 0.6066. So a perfect marginal correction buys back about a third of
the excess over the floor and leaves two thirds, and that two thirds is nothing
but the way the features co-vary inside a single trajectory.

Its pair scan then failed to name a culprit: the best pair synergy was 0.042
and a dozen pairs sat between 0.035 and 0.042, flat. But that scan ran on the
RAW generated features, so each pair's number mixes how wrong the two marginals
are with how wrong their coupling is. This re-runs it on the marginal matched
matrix, where the marginals are exact by construction and every remaining
signal is coupling.

  floor           human split half under this exact instrument, so the numbers
                  above have a grounded zero rather than a remembered one
  pairs on fixed  AUC from each pair of marginal matched columns. A pair at the
                  floor has correct coupling. A pair well above it does not.
  rank delta      Spearman correlation between each pair of features, human
                  minus generated. Signed, human readable, and free. The
                  detector's ranking says which coupling errors it can see;
                  this says which direction each one is wrong in.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_couplemap.py
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w4_copula import auc_cols, match_marginals  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="research/w4_ar_features.npy")
    ap.add_argument("--donor", default="data/human_ref_features_sir.npy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_couplemap.json")
    args = ap.parse_args()

    F = np.load(args.gen)
    F = F[np.all(np.isfinite(F), 1)]
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    H = H[np.all(np.isfinite(H), 1)]
    D = np.load(args.donor)
    D = D[np.all(np.isfinite(D), 1)]
    if len({tuple(r) for r in np.round(H, 9)}
           & {tuple(r) for r in np.round(D, 9)}):
        raise SystemExit("donor overlaps the scoring reference")
    allc = list(range(len(FEATURE_NAMES)))
    out = {}

    # the null runs a human sample through the identical transform, so every
    # number below is read against what the transform does on its own
    half = len(D) // 2
    Nm = match_marginals(D[half:], D[:half])
    floor_all = auc_cols(H, Nm, allc)
    floor_pair = float(np.median([auc_cols(H, Nm, [i, j])
                                  for i in allc for j in allc if j > i]))
    out["null_all18"] = floor_all
    out["null_pair_median"] = floor_pair
    print(f"  {len(F):,} generated, {len(H):,} human, {len(D):,} donor")
    print(f"  null all 18 {floor_all:.4f}, null median pair "
          f"{floor_pair:.4f}\n", flush=True)

    Fm = match_marginals(F, D)
    out["auc_all18_fixed"] = auc_cols(H, Fm, allc)

    rows = []
    for a in allc:
        for b in allc:
            if b <= a:
                continue
            v = auc_cols(H, Fm, [a, b])
            rh = float(spearmanr(H[:, a], H[:, b]).statistic)
            rg = float(spearmanr(F[:, a], F[:, b]).statistic)
            rows.append(dict(a=FEATURE_NAMES[a], b=FEATURE_NAMES[b], auc=v,
                             excess=v - floor_pair, rho_human=rh,
                             rho_gen=rg, drho=rg - rh))
        print(f"    through {FEATURE_NAMES[a]}", flush=True)

    rows.sort(key=lambda d: -d["excess"])
    out["pairs"] = rows
    print(f"\n  {'coupling pair':<52}{'auc':>8}{'over':>8}"
          f"{'rhoH':>8}{'rhoG':>8}{'drho':>8}")
    for d in rows[:15]:
        print(f"  {d['a'] + ' + ' + d['b']:<52}{d['auc']:>8.4f}"
              f"{d['excess']:>8.4f}{d['rho_human']:>8.3f}"
              f"{d['rho_gen']:>8.3f}{d['drho']:>8.3f}")

    big = sorted(rows, key=lambda d: -abs(d["drho"]))[:10]
    print(f"\n  {'largest rank correlation error':<52}{'rhoH':>8}"
          f"{'rhoG':>8}{'drho':>8}")
    for d in big:
        print(f"  {d['a'] + ' + ' + d['b']:<52}{d['rho_human']:>8.3f}"
              f"{d['rho_gen']:>8.3f}{d['drho']:>8.3f}")

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  every column here has exactly the human marginal, so any AUC")
    print("  above the floor is coupling and nothing else")


if __name__ == "__main__":
    main()
