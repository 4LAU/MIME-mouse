"""Where the detector actually looks. Per feature and per group.

PRE REGISTERED in HANDOFF.md 2026-08-07, "## Where the detector actually looks".
The three views, the five group definitions, the branch thresholds and the
prediction of ROUGHNESS DOMINANT were all fixed before this file existed.

READ ONLY. No generation, no GPU. Reuses the generated feature matrix already
cached by w4_evprice. Imports research/autoloop/scoring.py for the human
reference path and never edits it. Never touches data/human_eval_features.npy,
which scoring.py refuses by name anyway.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_featmap.py
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

# Fixed in the registration, before any number was read.
GROUPS = {
    "SPEED": [0, 1, 2, 3],
    "ROUGHNESS": [4, 5, 6, 7, 8],
    "GEOMETRY": [9, 10, 11, 12, 13],
    "TIMING": [14, 15],
    "ANGULAR": [16, 17],
}


def auc_on(cols, H, G, seed=scoring.RF_SEED):
    """The contract's own recipe, restricted to a column subset.

    Same balancing, same forest, same out of bag decision function as
    scoring.score_features. Only the column set changes.
    """
    n = min(len(H), len(G))
    X = np.vstack([H[:n][:, cols], G[:n][:, cols]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 oob_score=True, n_jobs=-1, random_state=seed)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.oob_decision_function_[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="research/w4_evprice_cache.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="research/w4_featmap.json")
    args = ap.parse_args()

    G = np.load(args.cache)["F"]
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    allc = list(range(len(FEATURE_NAMES)))
    seeds = [scoring.RF_SEED + i for i in range(args.seeds)]

    def rep(cols):
        """Mean and standard error over forest seeds. The out of bag AUC has
        real seed to seed noise and a single fit cannot separate a 0.01 drop
        from nothing."""
        v = np.array([auc_on(cols, H, G, s) for s in seeds])
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))

    print(f"\n  generated {len(G)}, human {len(H)}, "
          f"balanced to {min(len(G), len(H))} per class")
    print(f"  {len(seeds)} forest seeds per fit\n")

    full, full_se = rep(allc)
    print(f"  FULL 18 column AUC   {full:.4f}  se {full_se:.4f}\n")

    print(f"  {'feature':<26}{'alone':>9}{'se':>8}{'drop if removed':>18}{'se':>8}")
    single, loo = {}, {}
    for k, name in enumerate(FEATURE_NAMES):
        a, ase = rep([k])
        l, lse = rep([c for c in allc if c != k])
        single[name], loo[name] = (a, ase), (full - l, lse)
        print(f"  {name:<26}{a:>9.4f}{ase:>8.4f}{full - l:>18.4f}{lse:>8.4f}",
              flush=True)

    print(f"\n  {'group':<14}{'alone':>9}{'se':>8}{'drop if removed':>18}{'se':>8}")
    grp = {}
    for name, cols in GROUPS.items():
        a, ase = rep(cols)
        l, lse = rep([c for c in allc if c not in cols])
        grp[name] = {"alone": a, "alone_se": ase, "drop": full - l, "drop_se": lse}
        print(f"  {name:<14}{a:>9.4f}{ase:>8.4f}{full - l:>18.4f}{lse:>8.4f}",
              flush=True)

    top = max(grp, key=lambda g: grp[g]["drop"])
    d = grp[top]["drop"]
    if max(g["drop"] for g in grp.values()) < 0.05:
        verdict = ("DIFFUSE. No group carries 0.05 on its own removal. The "
                   "separation is spread and no kinematic family is the handle.")
    elif top == "ROUGHNESS":
        verdict = (f"ROUGHNESS DOMINANT. {d:+.4f} on removal. The acceleration "
                   "and jerk moments are the handle, as predicted.")
    else:
        verdict = (f"ELSEWHERE. {top} is the largest at {d:+.4f}, not ROUGHNESS. "
                   "The prediction was wrong.")
    print(f"\n  VERDICT  {verdict}")
    print("  a drop measured by deleting a detector column is NOT the gain from "
          "correcting the generator\n")

    json.dump({"full": full, "full_se": full_se,
               "single": {k: v[0] for k, v in single.items()},
               "single_se": {k: v[1] for k, v in single.items()},
               "loo_drop": {k: v[0] for k, v in loo.items()},
               "groups": grp, "verdict": verdict, "seeds": seeds},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
