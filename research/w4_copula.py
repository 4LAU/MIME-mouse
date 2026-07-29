"""Is the gap in the feature marginals or in how the features hang together?

`w4_whatsees` asked the detector what it uses and got an answer that is not
about any single quantity. All eighteen features together separate at 0.6668.
The best single feature is `movement_duration` at 0.5699 and everything else is
at or under 0.5310, with eight of the eighteen at or BELOW chance on their own.
Yet `angular_velocity_std`, which is 0.4958 alone and therefore carries no
information whatsoever by itself, is the second most expensive feature to
remove and buys the largest greedy gain of any step after the first. Dropping
any one of the eighteen costs at most 0.0244.

That is the signature of signal held in the DEPENDENCE between features rather
than in their distributions. This measures that directly, with no GPU and no
sampling, on feature matrices already on disk.

The instrument is a rank transform. Sorting a generated column and writing a
human column's sorted values into those positions makes the generated marginal
EXACTLY that human marginal while leaving the rank order, and therefore every
dependence between features, exactly as the model produced it.

The donor human sample MUST be disjoint from the scored one. The first version
of this file used the scoring reference as its own donor, which put identical
float values on both sides of the comparison, and out of bag forest predictions
then invert: an out of bag human row lands in a leaf holding only its generated
twins and is called generated, and the reverse. That drove every single feature
AUC to 0.04 to 0.07, far BELOW chance, which is the signature of the artifact
rather than of any real effect, and it biases the headline arm downward. The
donor is now `data/human_ref_features_sir.npy`, an independent 4000 row human
bank with zero rows in common with the scoring reference. The scored side is
unchanged, so these numbers stay comparable with the rest of the programme.

Arms:

  baseline        the model as it is
  marginals fixed every generated feature distribution replaced by the human
                  one, dependence untouched. This is the decisive arm. If the
                  score survives, then no amount of correcting any feature's
                  distribution can help and the target is the coupling. If it
                  collapses, the greedy table was misleading and marginals are
                  the whole story after all.
  human through   the SAME transform applied to a human sample instead of the
                  model. This is the null for the instrument: whatever it
                  scores is what the transform and the finite samples produce
                  on their own, and every other arm must be read against it.
  singles after   every single feature AUC after the fix, which must sit near
                  chance by construction. A sanity check on the transform, not
                  a result. It is what caught the first version's artifact.
  gen shuffled    each generated column independently permuted, destroying all
                  cross feature dependence and preserving all marginals. How
                  much the detector loses when the model's coupling is removed.
  both shuffled   the same to both sides, so both joints become the product of
                  their own marginals. This is marginal only detection in a
                  counterfactual world with no dependence on either side. It
                  does NOT bound what a distribution fix buys in the real
                  world: eighteen independent weak differences compound in a
                  way they cannot when the features are correlated, so this arm
                  can sit ABOVE the baseline. Read `marginals fixed` for the
                  real world answer and read this one only as evidence that the
                  marginals are jointly non trivial.

Then the pair scan names the broken coupling. For every pair the AUC from those
two columns alone, ranked by how far it exceeds the better of its two singles.
A pair whose members are individually useless and jointly informative IS the
defect, stated in the detector's own terms.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402


def auc_cols(H, F, cols):
    n = min(len(H), len(F))
    X = np.vstack([H[:n][:, cols], F[:n][:, cols]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                random_state=42)
    rf.fit(X, y)
    return float(roc_auc_score(y, rf.oob_decision_function_[:, 1]))


def match_marginals(F, H):
    """Give F exactly H's marginals while preserving F's rank structure."""
    G = np.empty_like(F)
    for j in range(F.shape[1]):
        order = np.argsort(F[:, j], kind="stable")
        ranks = np.empty(len(F), dtype=np.int64)
        ranks[order] = np.arange(len(F))
        hs = np.sort(H[:, j])
        idx = (ranks * (len(hs) - 1)) // max(len(F) - 1, 1)
        G[:, j] = hs[idx]
    return G


def shuffle_cols(A, rng):
    B = A.copy()
    for j in range(B.shape[1]):
        B[:, j] = B[rng.permutation(len(B)), j]
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="research/w4_ar_features.npy")
    ap.add_argument("--donor", default="data/human_ref_features_sir.npy",
                    help="human bank supplying marginals; must be disjoint "
                         "from the scoring reference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--out", default="research/w4_copula.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
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
    allc = list(range(len(FEATURE_NAMES)))
    print(f"  {len(F):,} generated, {len(H):,} human, {len(D):,} donor, "
          f"0 shared rows\n")

    out = {}
    base = auc_cols(H, F, allc)
    Fm = match_marginals(F, D)
    fixed = auc_cols(H, Fm, allc)
    half = len(D) // 2
    null = auc_cols(H, match_marginals(D[half:], D[:half]), allc)
    Fs = shuffle_cols(F, rng)
    gshuf = auc_cols(H, Fs, allc)
    Hs = shuffle_cols(H, rng)
    bshuf = auc_cols(Hs, Fs, allc)

    print(f"  {'arm':<26}{'auc':>9}")
    for lbl, v in [("baseline", base), ("marginals fixed", fixed),
                   ("human through transform", null),
                   ("gen shuffled", gshuf), ("both shuffled", bshuf)]:
        out[lbl] = v
        print(f"  {lbl:<26}{v:>9.4f}", flush=True)

    print(f"\n  {'single after fix':<26}{'auc':>9}")
    aft = sorted(((auc_cols(H, Fm, [i]), FEATURE_NAMES[i]) for i in allc),
                 reverse=True)
    out["singles_after_fix"] = [dict(feature=nm, auc=a) for a, nm in aft]
    for a, nm in aft[:5]:
        print(f"  {nm:<26}{a:>9.4f}", flush=True)

    if not args.pairs:
        json.dump(out, open(args.out, "w"), indent=2)
        return

    singles = {i: auc_cols(H, F, [i]) for i in allc}
    pairs = []
    for a in allc:
        for b in allc:
            if b <= a:
                continue
            v = auc_cols(H, F, [a, b])
            pairs.append(dict(a=FEATURE_NAMES[a], b=FEATURE_NAMES[b], auc=v,
                              best_single=max(singles[a], singles[b]),
                              synergy=v - max(singles[a], singles[b])))
        print(f"    pairs done through {FEATURE_NAMES[a]}", flush=True)
    pairs.sort(key=lambda d: -d["synergy"])
    out["pairs_top"] = pairs[:20]
    print(f"\n  {'pair':<50}{'auc':>9}{'best1':>9}{'synergy':>9}")
    for d in pairs[:12]:
        print(f"  {d['a'] + ' + ' + d['b']:<50}{d['auc']:>9.4f}"
              f"{d['best_single']:>9.4f}{d['synergy']:>9.4f}")

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  read every arm against 'human through transform', not against")
    print("  0.5. if 'marginals fixed' stays near baseline the gap is coupling")
    print("  and no distribution fix reaches it. 'both shuffled' bounds nothing")
    print("  and can sit above baseline; it is evidence only that the marginals")
    print("  are jointly non trivial")


if __name__ == "__main__":
    main()
