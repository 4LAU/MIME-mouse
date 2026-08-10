"""Marginals or dependence. What perfect marginal correction can ever buy.

PRE REGISTERED in HANDOFF.md 2026-08-07, "## Marginals or dependence". The
held out human split, the group definitions, the branch thresholds and the
prediction of DEPENDENCE DOMINANT were all fixed before this file existed.

The map: within a column, replace each generated value by the human value at the
same rank. Imposes the human marginal exactly and leaves the generated rank
dependence between columns untouched. What still separates afterwards is what no
marginal correction can remove.

This is a DIAGNOSTIC and not a generation method, but unlike the matched arm of
w4_evprice it resamples nothing. Same rows, same count, no duplicates, so the
random forest out of bag estimate stays valid.

READ ONLY. No generation, no GPU. Imports research/autoloop/scoring.py and never
edits it. Never touches data/human_eval_features.npy.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_joint.py
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w4_featmap import GROUPS  # noqa: E402


def rank_match(G, Hfit, cols):
    """Impose Hfit's marginal on G's columns, preserving G's rank dependence.

    Monotone and within column, so the copula of G is unchanged by
    construction. Ranks are mapped onto Hfit's order statistics by linear
    interpolation, which is the empirical quantile transform.
    """
    out = G.copy()
    n = len(G)
    # AVERAGE ranks, so tied generated rows stay tied. argsort assigns
    # distinct arbitrary ranks inside a tie group, which would send identical
    # generated values to different human values and quietly decorrelate that
    # column from the rest of the matrix. num_direction_changes is 92.7
    # percent tied in the generated sample, so that is not a small effect: it
    # would destroy exactly the dependence this function exists to preserve,
    # and it would do it in the direction that inflates the answer.
    # Midpoint plotting positions, so no row maps to an extreme order statistic.
    u = (rankdata(G[:, cols], method="average", axis=0) - 0.5) / n
    for j, c in enumerate(cols):
        hs = np.sort(Hfit[:, c])
        # NEAREST order statistic, never an interpolation between two. The
        # first version of this function interpolated, and that was an
        # artefact large enough to flip a sign. num_direction_changes is an
        # integer count with 93.5 percent ties in the human sample and
        # movement_duration is discretised at 36.3 percent. Interpolating
        # between two human values produces a number no human row can hold,
        # so the "corrected" column became a free tell and matching a group
        # appeared to make the sample MORE detectable. Landing on an actual
        # human order statistic preserves ties, integrality and support
        # exactly, which is what the empirical quantile transform means for
        # a discrete variable.
        out[:, c] = hs[np.rint(u[:, j] * (len(hs) - 1)).astype(int)]
    return out


def auc(G, H, seeds):
    """The contract's own recipe. Balanced, same forest, out of bag AUC."""
    n = min(len(H), len(G))
    X = np.vstack([H[:n], G[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    v = []
    for s in seeds:
        clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                     oob_score=True, n_jobs=-1, random_state=s)
        clf.fit(X, y)
        v.append(roc_auc_score(y, clf.oob_decision_function_[:, 1]))
    v = np.array(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="research/w4_evprice_cache.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--out", default="research/w4_joint.json")
    args = ap.parse_args()

    G = np.load(args.cache)["F"]
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    seeds = [scoring.RF_SEED + i for i in range(args.seeds)]

    # The half used to fit the marginals is never the half used to score. Any
    # score computed against the fitting half would reward matching that
    # sample's particular order statistics.
    rng = np.random.default_rng(args.split)
    perm = rng.permutation(len(H))
    Hfit, Hscore = H[perm[:len(H) // 2]], H[perm[len(H) // 2:]]
    allc = list(range(len(FEATURE_NAMES)))

    print(f"\n  generated {len(G)}, human {len(H)} split "
          f"{len(Hfit)} fit / {len(Hscore)} score, never overlapping")
    print(f"  {len(seeds)} forest seeds per fit\n")

    base, base_se = auc(G, Hscore, seeds)
    chance_excess = base - 0.5
    print(f"  BASELINE, generated untouched      {base:.4f}  se {base_se:.4f}")
    print(f"  the detector holds {chance_excess:.4f} above chance\n")

    full, full_se = auc(rank_match(G, Hfit, allc), Hscore, seeds)
    print(f"  ALL 18 marginals matched to human  {full:.4f}  se {full_se:.4f}")
    print(f"  removed {base - full:+.4f} of {chance_excess:.4f}, "
          f"{(base - full) / chance_excess:.1%} of the detector's power")
    print(f"  what survives is pure dependence   {full - 0.5:.4f} above chance\n")

    print(f"  {'group matched':<16}{'AUC':>9}{'se':>8}{'bought':>10}{'share':>9}")
    grp = {}
    for name, cols in GROUPS.items():
        a, ase = auc(rank_match(G, Hfit, cols), Hscore, seeds)
        grp[name] = {"auc": a, "se": ase, "bought": base - a}
        print(f"  {name:<16}{a:>9.4f}{ase:>8.4f}{base - a:>+10.4f}"
              f"{(base - a) / chance_excess:>9.1%}", flush=True)

    if full >= 0.62:
        verdict = (f"DEPENDENCE DOMINANT. {full:.4f} survives perfect marginal "
                   "correction. Marginal matching is capped and cannot reach "
                   "chance. The program changes.")
    elif full <= 0.55:
        verdict = (f"MARGINALS DOMINANT. {full:.4f}. Moment matching is the "
                   "right program and should be pushed hard.")
    else:
        verdict = (f"MIXED. {full:.4f}. Marginal correction buys "
                   f"{(base - full) / chance_excess:.1%} and stops.")
    print(f"\n  VERDICT  {verdict}")
    print("  this is an upper bound on marginal correction, not an achieved "
          "score, and the map is not a generation method\n")

    json.dump({"baseline": base, "baseline_se": base_se,
               "all_matched": full, "all_matched_se": full_se,
               "groups": grp, "verdict": verdict,
               "n_gen": int(len(G)), "n_fit": int(len(Hfit)),
               "n_score": int(len(Hscore)), "seeds": seeds},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
