"""Where the higher order residual actually lives, pair by pair.

`w4_block` leaves 0.0931 of the snap cache's 0.1519 excess as higher order
structure, unreachable by marginals or correlations. `w4_modes` excluded mode
averaging. This localises what is left.

The instrument. For a pair of standard normal variables with a GAUSSIAN copula
and correlation r, the correlation of their ABSOLUTE values is fixed by r alone:

    corr(|X|,|Y|) = (2/pi) * (sqrt(1-r^2) + r*arcsin(r) - 1) / (1 - 2/pi)

So for each pair of features, measured on normal scores, the quantity

    excess(i,j) = corr(|z_i|,|z_j|) - gaussian_implied(corr(z_i,z_j))

is zero for ANY Gaussian copula regardless of its correlation matrix. It is
non zero only for higher order dependence: volatility coupling, tail dependence,
the tendency of two features to be extreme together beyond what their linear
correlation implies. It is by construction orthogonal to everything `w4_joint`
and `w4_copula` already priced.

Comparing human excess to generated excess therefore names the pairs carrying
the part of the gap no moment based objective can reach.

DIAGNOSTIC, not a generation method. Resamples nothing, no GPU. Never touches
data/human_eval_features.npy, never modifies scoring code.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_tail.py
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w4_copula import normal_scores  # noqa: E402


def gaussian_abs_corr(r):
    """corr(|X|,|Y|) implied by a Gaussian copula with correlation r."""
    r = np.clip(r, -0.999999, 0.999999)
    return ((2.0 / np.pi) * (np.sqrt(1 - r ** 2) + r * np.arcsin(r) - 1.0)
            / (1.0 - 2.0 / np.pi))


def excess(Z):
    """Observed minus Gaussian implied absolute value correlation, per pair."""
    C = np.corrcoef(Z, rowvar=False)
    A = np.corrcoef(np.abs(Z), rowvar=False)
    return A - gaussian_abs_corr(C), C


def boot(Z, B, seed):
    """Bootstrap the excess matrix so a human minus generated difference can be
    read against its own sampling error rather than against zero."""
    rng = np.random.default_rng(seed)
    out = np.empty((B, Z.shape[1], Z.shape[1]))
    for b in range(B):
        out[b] = excess(Z[rng.integers(0, len(Z), len(Z))])[0]
    return out.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="research/w4_snap_cache.npz")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default="research/w4_tail.json")
    args = ap.parse_args()

    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    G = np.load(args.cache)["F"]
    Hn, Gn = normal_scores(H), normal_scores(G)
    Eh, Ch = excess(Hn)
    Eg, Cg = excess(Gn)

    d = len(FEATURE_NAMES)
    off = ~np.eye(d, dtype=bool)
    print(f"\n  human {len(H)}, generated {len(G)}, {args.boot} bootstrap "
          f"resamples per side")
    print(f"\n  mean |excess| over the {off.sum() // 2} pairs")
    print(f"    human      {np.abs(Eh)[off].mean():.4f}")
    print(f"    generated  {np.abs(Eg)[off].mean():.4f}")
    print(f"  a Gaussian copula would read 0 on both, up to sampling error")

    Sh = boot(Hn, args.boot, args.seed)
    Sg = boot(Gn, args.boot, args.seed + 1)
    D = Eh - Eg
    S = np.sqrt(Sh ** 2 + Sg ** 2)
    Zsc = np.where(S > 1e-12, D / S, 0.0)

    print(f"\n  mean |human minus generated excess|  "
          f"{np.abs(D)[off].mean():.4f}")
    print(f"  pairs beyond 3 sigma  "
          f"{int((np.abs(Zsc)[off] > 3).sum() // 2)} of {off.sum() // 2}")

    iu = np.triu_indices(d, 1)
    order = np.argsort(-np.abs(Zsc[iu]))[:args.top]
    print(f"\n  the {args.top} pairs where higher order dependence differs most")
    print(f"  {'feature i':<26}{'feature j':<26}{'human':>9}{'gen':>9}"
          f"{'diff':>9}{'sigma':>8}{'corr h':>9}{'corr g':>9}")
    rows = []
    for t in order:
        i, j = iu[0][t], iu[1][t]
        rows.append({"i": FEATURE_NAMES[i], "j": FEATURE_NAMES[j],
                     "excess_human": float(Eh[i, j]),
                     "excess_gen": float(Eg[i, j]),
                     "diff": float(D[i, j]), "sigma": float(Zsc[i, j]),
                     "corr_human": float(Ch[i, j]),
                     "corr_gen": float(Cg[i, j])})
        print(f"  {FEATURE_NAMES[i]:<26}{FEATURE_NAMES[j]:<26}"
              f"{Eh[i, j]:>9.3f}{Eg[i, j]:>9.3f}{D[i, j]:>+9.3f}"
              f"{Zsc[i, j]:>+8.1f}{Ch[i, j]:>+9.3f}{Cg[i, j]:>+9.3f}")

    # which feature is involved in the most badly mismatched pairs
    inv = np.zeros(d)
    for t in np.flatnonzero(np.abs(Zsc[iu]) > 3):
        inv[iu[0][t]] += 1
        inv[iu[1][t]] += 1
    print(f"\n  features by how many beyond 3 sigma pairs they appear in")
    for k in np.argsort(-inv)[:8]:
        if inv[k]:
            print(f"    {FEATURE_NAMES[k]:<26}{int(inv[k]):>4}")

    json.dump({"mean_abs_excess_human": float(np.abs(Eh)[off].mean()),
               "mean_abs_excess_gen": float(np.abs(Eg)[off].mean()),
               "mean_abs_diff": float(np.abs(D)[off].mean()),
               "pairs_beyond_3sigma": int((np.abs(Zsc)[off] > 3).sum() // 2),
               "top_pairs": rows,
               "involvement": {FEATURE_NAMES[k]: int(inv[k])
                               for k in range(d)}},
              open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}\n")


if __name__ == "__main__":
    main()
