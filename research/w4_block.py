"""What correcting ONE block of the correlation matrix would buy.

Follows w4_copula, which priced the whole matrix. This asks the narrower
question the w4_latent intervention actually addresses: if only the six wobble
features were coupled correctly, and every other entry stayed as generated, how
much of the detector's power goes away. That is the ceiling for a latent that
drives the wobble cluster and nothing else.

Registered together with w4_latent in HANDOFF.md 2026-08-07. Same held out split
and same second sample placebo as w4_copula.

DIAGNOSTIC, not a generation method. Resamples nothing.

READ ONLY. No generation, no GPU.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_block.py
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
from w4_copula import normal_scores, recolour  # noqa: E402
from w4_featmap import GROUPS  # noqa: E402
from w4_joint import auc, rank_match  # noqa: E402

WOBBLE = [10, 11, 12, 13, 16, 17]


def nearest_psd(C, eps=1e-8):
    """Splice a block from one correlation matrix into another and the result
    need not be positive semidefinite, which the Cholesky in recolour requires.
    Clip the eigenvalues and renormalise the diagonal back to one."""
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    C = V @ np.diag(np.maximum(w, eps)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def splice(Cg, Ch, cols):
    """Cg everywhere except the within-block entries of cols, taken from Ch.
    Only the block is corrected; every cross term to the rest of the matrix
    stays generated, which is what a latent driving that cluster alone does."""
    C = Cg.copy()
    C[np.ix_(cols, cols)] = Ch[np.ix_(cols, cols)]
    return nearest_psd(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="research/w4_evprice_cache.npz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--out", default="research/w4_block.json")
    args = ap.parse_args()

    G = np.load(args.cache)["F"]
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    seeds = [scoring.RF_SEED + i for i in range(args.seeds)]

    rng = np.random.default_rng(args.split)
    perm = rng.permutation(len(H))
    Hfit, Hscore = H[perm[:len(H) // 2]], H[perm[len(H) // 2:]]
    allc = list(range(G.shape[1]))

    Gn = normal_scores(G)
    Cg = np.corrcoef(Gn, rowvar=False)
    Ch = np.corrcoef(normal_scores(Hfit), rowvar=False)

    base, _ = auc(G, Hscore, seeds)
    marg, _ = auc(rank_match(G, Hfit, allc), Hscore, seeds)
    full, _ = auc(rank_match(recolour(Gn, Cg, Ch), Hfit, allc), Hscore, seeds)
    print(f"\n  BASELINE                        {base:.4f}")
    print(f"  marginals only                  {marg:.4f}")
    print(f"  marginals + FULL correlation    {full:.4f}\n")

    print(f"  {'block corrected':<18}{'AUC':>9}{'vs marginals':>14}"
          f"{'share of full':>15}")
    room = marg - full
    out = {"baseline": base, "marginals": marg, "full": full, "blocks": {}}
    for name, cols in list(GROUPS.items()) + [("WOBBLE", WOBBLE)]:
        a, ase = auc(rank_match(recolour(Gn, Cg, splice(Cg, Ch, cols)),
                                Hfit, allc), Hscore, seeds)
        out["blocks"][name] = {"auc": a, "se": ase, "bought": marg - a}
        print(f"  {name:<18}{a:>9.4f}{marg - a:>+14.4f}"
              f"{(marg - a) / room:>15.1%}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  the full correlation correction is worth {room:.4f} over "
          f"marginals alone; the block rows say how much of that one cluster "
          f"holds\n  wrote {args.out}\n")


if __name__ == "__main__":
    main()
