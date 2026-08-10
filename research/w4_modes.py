"""Is the higher order residual a missing MODE rather than a missing moment.

`research/w4_block.py` puts a floor near 0.593 on anything expressible as the
eighteen marginals plus their full linear dependence, against a served baseline
of 0.652. So 61 percent of what the detector still holds is higher order, and no
moment matching objective can reach it.

The standard generative failure with that signature is mode averaging: a model
trained by likelihood on a multimodal target puts mass between the modes instead
of on them. The result has close to the right means, variances and correlations
and the wrong joint density, which is exactly the pattern above.

Two readouts, both on normal scores so no feature's units dominate, both after a
PCA to keep the covariance estimates honest at these sample sizes.

  1. Held out log likelihood of a Gaussian mixture at k = 1 to 5, fitted on half
     a sample and scored on the other half, separately for human and generated.
     If the human sample wants more components than the generated one does, the
     human distribution is a mixture the model has not reproduced.

  2. Fit the mixture on HUMAN, then assign both samples to its components and
     compare the mixing proportions. A component that holds a healthy share of
     humans and almost no generated rows is a named, countable missing regime,
     and its mean in feature space says what that regime is.

DIAGNOSTIC, not a generation method. Resamples nothing, no GPU. Never touches
data/human_eval_features.npy, never modifies scoring code.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_modes.py
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


def pca(X, q, mean=None, W=None):
    """Project onto q directions. Fitted on the pooled sample so neither side
    gets a basis chosen to flatter it."""
    if W is None:
        mean = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
        W = Vt[:q].T
    return (X - mean) @ W, mean, W


def ll_curve(Z, ks, seed, tag):
    """Held out mean log likelihood per row at each k. Fit on half, score the
    other half, so a larger k cannot win by memorising."""
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Z))
    A, B = Z[idx[:len(Z) // 2]], Z[idx[len(Z) // 2:]]
    out = []
    for k in ks:
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4,
                            n_init=3, random_state=seed).fit(A)
        out.append(float(g.score(B)))
    best = ks[int(np.argmax(out))]
    print(f"  {tag:<12}" + "".join(f"{v:>10.3f}" for v in out)
          + f"{best:>8}", flush=True)
    return out, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="research/w4_snap_cache.npz")
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_modes.json")
    args = ap.parse_args()

    from sklearn.mixture import GaussianMixture

    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    G = np.load(args.cache)["F"]
    n = min(len(H), len(G))
    rng = np.random.default_rng(args.seed)
    # equal n so the held out log likelihoods are comparable between the two
    H = H[rng.permutation(len(H))[:n]]
    G = G[rng.permutation(len(G))[:n]]

    # normal scores computed on the POOLED sample, so the marginals of the two
    # sides are put on one common scale and any difference that survives is
    # dependence, not location or spread.
    P = normal_scores(np.vstack([H, G]))
    Hn, Gn = P[:n], P[n:]
    Z, mu, W = pca(np.vstack([Hn, Gn]), args.q)
    Zh, Zg = Z[:n], Z[n:]

    ks = list(range(1, args.kmax + 1))
    print(f"\n  n {n} per side, {args.q} PCs, mixture fitted on half and "
          f"scored on the other half")
    print(f"\n  {'sample':<12}" + "".join(f"{'k=' + str(k):>10}" for k in ks)
          + f"{'best':>8}")
    lh, kh = ll_curve(Zh, ks, args.seed, "human")
    lg, kg = ll_curve(Zg, ks, args.seed, "generated")

    gain_h = lh[kh - 1] - lh[0]
    gain_g = lg[kg - 1] - lg[0]
    print(f"\n  gain over a single Gaussian   human {gain_h:+.3f}   "
          f"generated {gain_g:+.3f}")

    # READOUT 2. Human's own mixture, both samples assigned to it.
    k = max(kh, 2)
    g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, n_init=5,
                        random_state=args.seed).fit(Zh)
    ph = np.bincount(g.predict(Zh), minlength=k) / n
    pg = np.bincount(g.predict(Zg), minlength=k) / n
    print(f"\n  human's own {k} component mixture, both samples assigned to it")
    print(f"  {'component':>10}{'human':>9}{'generated':>11}{'ratio':>9}")
    for c in range(k):
        r = pg[c] / ph[c] if ph[c] > 1e-9 else float("nan")
        print(f"  {c:>10}{ph[c]:>9.3f}{pg[c]:>11.3f}{r:>9.2f}")

    # what the most under-populated component IS, in feature terms
    valid = [c for c in range(k) if ph[c] >= 0.05]
    worst = min(valid, key=lambda c: pg[c] / ph[c])
    back = g.means_[worst] @ W.T + mu
    print(f"\n  component {worst} is the most under-populated by the model, "
          f"{pg[worst]:.3f} against {ph[worst]:.3f}")
    print(f"  its centre in normal score units, six largest:")
    for j in np.argsort(-np.abs(back))[:6]:
        print(f"    {FEATURE_NAMES[j]:<26}{back[j]:>+8.3f}")

    ratio = pg[worst] / ph[worst]
    if gain_h - gain_g >= 0.10 and ratio <= 0.6:
        verdict = (f"MISSING MODE. Human wants k={kh} against the model's "
                   f"k={kg}, and human component {worst} holds "
                   f"{ph[worst]:.1%} of humans against {pg[worst]:.1%} of "
                   "generated. A named regime the model does not produce.")
    elif ratio <= 0.6:
        verdict = (f"UNEVEN, NOT MULTIMODAL. Both samples want a similar k, but "
                   f"component {worst} is under-populated {ratio:.2f}. The "
                   "density is misshapen rather than missing a mode.")
    else:
        verdict = (f"NO MISSING MODE. Human k={kh}, generated k={kg}, worst "
                   f"component ratio {ratio:.2f}. The higher order residual is "
                   "not mode averaging and a mixture prior will not fix it.")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"ll_human": lh, "ll_generated": lg, "k_human": kh,
               "k_generated": kg, "gain_human": gain_h,
               "gain_generated": gain_g, "prop_human": ph.tolist(),
               "prop_generated": pg.tolist(), "worst_component": int(worst),
               "worst_ratio": float(ratio), "verdict": verdict},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
