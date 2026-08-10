"""Linear dependence or higher order. Splitting what survives marginal matching.

PRE REGISTERED in HANDOFF.md 2026-08-07, "## Linear dependence or higher order".
The held out split, the second sample placebo, the branch thresholds and the
prediction of HIGHER ORDER were all fixed before this file existed.

Impose the human marginals AND the human normal score correlation matrix on the
generated sample. What still separates is higher order by construction.

DIAGNOSTIC, not a generation method. Resamples nothing: same rows, same count,
no duplicates, so the out of bag estimate stays valid.

READ ONLY. No generation, no GPU. Imports research/autoloop/scoring.py and never
edits it. Never touches data/human_eval_features.npy.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_copula.py
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.special import erfinv
from scipy.stats import rankdata

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring  # noqa: E402
from w4_joint import auc, rank_match  # noqa: E402


def normal_scores(X):
    """Column by column rank to standard normal. Monotone, so this changes no
    dependence, it only fixes the marginals at Gaussian."""
    u = (rankdata(X, method="average", axis=0) - 0.5) / len(X)
    return np.sqrt(2.0) * erfinv(2.0 * u - 1.0)


def recolour(Gn, C_from, C_to):
    """Whiten Gn by C_from, recolour to C_to. Correlation becomes C_to and the
    row ordering, hence every higher order feature of the copula, is carried
    through the linear map."""
    Lf = np.linalg.cholesky(C_from + 1e-10 * np.eye(len(C_from)))
    Lt = np.linalg.cholesky(C_to + 1e-10 * np.eye(len(C_to)))
    return Gn @ np.linalg.inv(Lf).T @ Lt.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="research/w4_evprice_cache.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--out", default="research/w4_copula.json")
    args = ap.parse_args()

    G = np.load(args.cache)["F"]
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    seeds = [scoring.RF_SEED + i for i in range(args.seeds)]

    rng = np.random.default_rng(args.split)
    perm = rng.permutation(len(H))
    Hfit, Hscore = H[perm[:len(H) // 2]], H[perm[len(H) // 2:]]
    allc = list(range(G.shape[1]))

    # the placebo's correlation matrix comes from an independent HALF of the
    # generated sample, so it corrects nothing while applying the same kind of
    # linear map. The recoloured arm uses the other half, so neither arm is
    # recoloured with a matrix estimated on the rows being recoloured.
    gperm = rng.permutation(len(G))
    Ga, Gb = G[gperm[:len(G) // 2]], G[gperm[len(G) // 2:]]

    print(f"\n  generated {len(G)}, human {len(H)} split "
          f"{len(Hfit)} fit / {len(Hscore)} score")
    print(f"  {len(seeds)} forest seeds per fit\n")

    base, base_se = auc(G, Hscore, seeds)
    marg, marg_se = auc(rank_match(G, Hfit, allc), Hscore, seeds)
    excess = base - 0.5
    print(f"  BASELINE                           {base:.4f}  se {base_se:.4f}")
    print(f"  marginals matched (w4_joint)       {marg:.4f}  se {marg_se:.4f}")
    print(f"  dependence still to explain        {marg - 0.5:.4f} above chance\n")

    Gn = normal_scores(Ga)
    Ch = np.corrcoef(normal_scores(Hfit), rowvar=False)
    Cg = np.corrcoef(Gn, rowvar=False)
    Cp = np.corrcoef(normal_scores(Gb), rowvar=False)

    off = ~np.eye(len(Ch), dtype=bool)
    print(f"  correlation matrix distance, human vs generated  "
          f"max |d| {np.abs(Ch - Cg)[off].max():.4f}  "
          f"mean |d| {np.abs(Ch - Cg)[off].mean():.4f}")
    print(f"  placebo matrix distance, gen half vs gen half     "
          f"max |d| {np.abs(Cp - Cg)[off].max():.4f}  "
          f"mean |d| {np.abs(Cp - Cg)[off].mean():.4f}\n")

    # rank_match carries the recoloured rows back onto the human marginals, so
    # both arms end with identical marginals and differ only in correlation.
    real, real_se = auc(rank_match(recolour(Gn, Cg, Ch), Hfit, allc),
                        Hscore, seeds)
    plac, plac_se = auc(rank_match(recolour(Gn, Cg, Cp), Hfit, allc),
                        Hscore, seeds)
    print(f"  RECOLOURED to human correlation    {real:.4f}  se {real_se:.4f}")
    print(f"  PLACEBO, recoloured to gen         {plac:.4f}  se {plac_se:.4f}")
    print(f"  correction net of the map          {plac - real:+.4f}\n")

    if real <= 0.58 and plac > 0.58:
        verdict = (f"CORRELATION CARRIES IT. {real:.4f} with the human "
                   "correlation matrix, placebo does not follow. The model's "
                   "feature correlation matrix is a concrete target.")
    elif real >= 0.64:
        verdict = (f"HIGHER ORDER. {real:.4f} survives human marginals AND the "
                   "human correlation matrix. No tractable handle is left in "
                   "the feature space; the work moves to the generative process.")
    else:
        verdict = (f"MIXED. {real:.4f} recoloured against {plac:.4f} placebo.")
    print(f"  VERDICT  {verdict}")
    print("  an upper bound on what correcting correlations buys, not an "
          "achieved score\n")

    json.dump({"baseline": base, "marginals_only": marg,
               "recoloured_human": real, "recoloured_human_se": real_se,
               "placebo": plac, "placebo_se": plac_se,
               "corr_dist_human": float(np.abs(Ch - Cg)[off].mean()),
               "corr_dist_placebo": float(np.abs(Cp - Cg)[off].mean()),
               "verdict": verdict, "seeds": seeds},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
