"""The whole remaining gap should live in one bivariate distribution. Does it?

`w4_partial` produced a claim sharp enough to be wrong. Controlling for
`movement_duration` and `mean_velocity` collapses the model's cross feature
rank correlation error from a mean absolute 0.1689 to 0.0249, and on the largest
error pair both sides flip sign together and agree, human +0.249 against model
+0.246. Conditional on those two features the model is human.

If that is true in the strong sense then the entire 0.106 excess that `w4_copula`
left after fixing all eighteen marginals is carried by the joint distribution of
duration and mean velocity, and by nothing else. Their individual marginals are
already close to human, so the claim is specifically about how they hang
together, not about either one alone.

That is falsifiable. Fix the pair JOINTLY, leave every other column exactly as
the model made it, and the eighteen feature AUC should fall most of the way to
the instrument null. If it does not, the partial correlation reading was about
rank correlations only and does not carry the detector's actual signal, and the
honest conclusion is that the defect is spread across the joint in a way no two
features summarise.

Arms, each with its own null, because the transform itself moves the number:

  baseline            the model as it is
  pair only           the two columns alone, against the same two on the null.
                      How much these two carry by themselves.
  pair marginals      the two columns rank matched to human ONE AT A TIME. This
                      is the part `w4_copula` already bought.
  pair joint          duration matched, then mean velocity matched WITHIN
                      duration bins, so p(vel | dur) becomes human while every
                      other column and every within bin rank is untouched. The
                      decisive arm.
  all marg + joint    all eighteen marginals matched and then the conditional
                      match on top. The best a correction confined to this pair
                      plus all marginals could do.
  all but pair        the other sixteen marginals matched, the pair left alone.
                      The complement. If the pair is the whole story this stays
                      near baseline.

Every arm is scored against the untouched scoring reference and takes its
marginals from a donor bank with zero rows in common with it, for the out of bag
inversion reason written up in `w4_copula`. No GPU, no sampling.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_dvjoint.py
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
from w4_copula import auc_cols, match_marginals  # noqa: E402


def match_cols(F, D, cols):
    """Rank match only `cols`, leaving every other column untouched."""
    G = F.copy()
    G[:, cols] = match_marginals(F[:, cols], D[:, cols])
    return G


def match_conditional(F, D, dep, given, nbins):
    """Make p(F[dep] | F[given]) equal p(D[dep] | D[given]).

    Bin each side at its own quantiles of `given` so bin b on one side is the
    same slice of the conditioning distribution as bin b on the other, then
    rank match `dep` within the bin. Within bin rank order is preserved, so the
    dependence of `dep` on every OTHER column survives inside each bin.
    """
    G = F.copy()
    qs = np.linspace(0, 1, nbins + 1)[1:-1]
    fb = np.clip(np.searchsorted(np.quantile(F[:, given], qs), F[:, given]),
                 0, nbins - 1)
    db = np.clip(np.searchsorted(np.quantile(D[:, given], qs), D[:, given]),
                 0, nbins - 1)
    for b in range(nbins):
        fi = np.flatnonzero(fb == b)
        di = np.flatnonzero(db == b)
        if len(fi) < 2 or len(di) < 2:
            continue
        order = np.argsort(F[fi, dep], kind="stable")
        ranks = np.empty(len(fi), dtype=np.int64)
        ranks[order] = np.arange(len(fi))
        hs = np.sort(D[di, dep])
        G[fi, dep] = hs[(ranks * (len(hs) - 1)) // max(len(fi) - 1, 1)]
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="research/w4_ar_features.npy")
    ap.add_argument("--donor", default="data/human_ref_features_sir.npy")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--out", default="research/w4_dvjoint.json")
    args = ap.parse_args()

    F = np.load(args.gen)
    F = F[np.all(np.isfinite(F), 1)]
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    H = H[np.all(np.isfinite(H), 1)]
    D = np.load(args.donor)
    D = D[np.all(np.isfinite(D), 1)]
    shared = len({tuple(r) for r in np.round(H, 9)}
                 & {tuple(r) for r in np.round(D, 9)})
    if shared:
        raise SystemExit(f"donor shares {shared} rows with the scoring "
                         f"reference; the out of bag inversion artifact "
                         f"applies and the run is meaningless")

    DUR = FEATURE_NAMES.index("movement_duration")
    VEL = FEATURE_NAMES.index("mean_velocity")
    allc = list(range(len(FEATURE_NAMES)))
    pair = [DUR, VEL]
    rest = [i for i in allc if i not in pair]
    half = len(D) // 2
    A, B = D[half:], D[:half]
    print(f"  {len(F):,} generated, {len(H):,} human, {len(D):,} donor, "
          f"0 shared rows, {args.bins} duration bins\n")

    def joint(X, Y):
        return match_conditional(match_cols(X, Y, [DUR]), Y, VEL, DUR,
                                 args.bins)

    arms = [
        ("baseline", allc, lambda: F, lambda: A),
        ("pair only", pair, lambda: F, lambda: A),
        ("pair marginals", allc, lambda: match_cols(F, D, pair),
         lambda: match_cols(A, B, pair)),
        ("pair joint", allc, lambda: joint(F, D), lambda: joint(A, B)),
        ("pair joint, cols", pair, lambda: joint(F, D), lambda: joint(A, B)),
        ("all marg + joint", allc,
         lambda: match_conditional(match_marginals(F, D), D, VEL, DUR,
                                   args.bins),
         lambda: match_conditional(match_marginals(A, B), B, VEL, DUR,
                                   args.bins)),
        ("all but pair", allc, lambda: match_cols(F, D, rest),
         lambda: match_cols(A, B, rest)),
    ]

    out = {}
    print(f"  {'arm':<20}{'cols':>6}{'auc':>9}{'null':>9}{'excess':>9}")
    for lbl, cols, fgen, fnull in arms:
        a = auc_cols(H, fgen(), cols)
        n = auc_cols(H, fnull(), cols)
        out[lbl] = dict(auc=a, null=n, excess=a - n, n_cols=len(cols))
        print(f"  {lbl:<20}{len(cols):>6}{a:>9.4f}{n:>9.4f}{a - n:>9.4f}",
              flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  read the excess column, not the auc column. if 'pair joint'")
    print("  excess falls near zero the whole detectable gap is the duration")
    print("  and mean velocity copula and the target is that one distribution.")
    print("  if it stays near the baseline excess the partial correlation")
    print("  reading was about rank structure only and does not carry the")
    print("  detector's signal, and the defect is genuinely spread out.")


if __name__ == "__main__":
    main()
