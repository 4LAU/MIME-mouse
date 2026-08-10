"""Does the defect grow with the length of the rollout.

Six W4 diagnostics agree the gap is DIFFUSE: no feature, no correlation block,
no missing mixture component, no concentrated higher order pair. A model that is
slightly wrong nearly everywhere, at every order, with 61 percent of the excess
above second moments, is the signature of a process whose error ACCUMULATES
rather than one with a named defect.

`EventARModel` is autoregressive and trained by teacher forcing. At sample time
it conditions on its own output, so any per step bias compounds along the
rollout. That mechanism, exposure bias, predicts exactly the diffuse higher
order signature above, and it predicts one thing nothing measured so far has
looked at: the discrepancy should be SMALL for short trajectories and LARGE for
long ones.

The test. Bin both samples by `movement_duration`, which the model is
CONDITIONED on and therefore reproduces by construction, and run the contract
scorer inside each bin. Because duration is matched within a bin it cannot
itself separate the samples there, so a bin to bin trend in AUC is a statement
about what the model does with a longer rollout, not about how long the rollouts
are.

  DRIFT     AUC rises with duration, and the longest bin exceeds the shortest by
            more than the bootstrap spread of that difference. The error
            compounds, exposure bias is the mechanism, and the remedies are
            named and standard: scheduled sampling, rollout consistent training,
            or a non autoregressive decoder.

  UNIFORM   No trend. The per step conditional is wrong in a way that does not
            compound, exposure bias is excluded, and the search moves elsewhere.

DIAGNOSTIC, not a generation method. Resamples nothing, no GPU. Never touches
data/human_eval_features.npy, never modifies scoring code.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_length.py
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

DUR = FEATURE_NAMES.index("movement_duration")
NDC = FEATURE_NAMES.index("num_direction_changes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="research/w4_snap_cache.npz")
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--by", default="movement_duration")
    ap.add_argument("--boot", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_length.json")
    args = ap.parse_args()

    col = FEATURE_NAMES.index(args.by)
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    G = np.load(args.cache)["F"]

    # edges from the HUMAN distribution, so the bins are fixed reference
    # intervals and a difference in how the model populates them shows up as a
    # count difference rather than as a moving definition of "long"
    qs = np.percentile(H[:, col], np.linspace(0, 100, args.bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf

    print(f"\n  binning by {args.by}, edges from the human sample")
    print(f"  human {len(H)}, generated {len(G)}, {args.bins} bins, "
          f"{args.boot} bootstrap resamples per bin")
    print(f"\n  {'bin':>4}{'range':>22}{'n human':>9}{'n gen':>8}"
          f"{'AUC':>9}{'boot sd':>10}{'mean dur':>10}{'mean ndc':>10}")

    rows = []
    for b in range(args.bins):
        lo, hi = qs[b], qs[b + 1]
        hm = (H[:, col] >= lo) & (H[:, col] < hi)
        gm = (G[:, col] >= lo) & (G[:, col] < hi)
        Hb, Gb = H[hm], G[gm]
        if len(Hb) < 50 or len(Gb) < 50:
            print(f"  {b:>4}{f'[{lo:.3g}, {hi:.3g})':>22}{len(Hb):>9}"
                  f"{len(Gb):>8}   too few rows to score")
            continue
        # score_features balances internally to min(n_human, n_synth), but it
        # always loads the FULL human file, so the human side has to be written
        # to a temporary array the scorer will accept. Instead, call the same
        # recipe directly on the two binned matrices.
        a = _auc(Hb, Gb, args.seed)
        sd = _boot(Hb, Gb, args.boot, args.seed)
        rows.append({"bin": b, "lo": float(lo), "hi": float(hi),
                     "n_human": int(len(Hb)), "n_gen": int(len(Gb)),
                     "auc": a, "boot_sd": sd,
                     "mean_dur": float(Gb[:, DUR].mean()),
                     "mean_ndc": float(Gb[:, NDC].mean())})
        print(f"  {b:>4}{f'[{lo:.3g}, {hi:.3g})':>22}{len(Hb):>9}{len(Gb):>8}"
              f"{a:>9.4f}{sd:>10.4f}{Gb[:, DUR].mean():>10.3f}"
              f"{Gb[:, NDC].mean():>10.1f}", flush=True)

    if len(rows) >= 2:
        d = rows[-1]["auc"] - rows[0]["auc"]
        s = np.hypot(rows[-1]["boot_sd"], rows[0]["boot_sd"])
        mono = all(rows[i + 1]["auc"] >= rows[i]["auc"]
                   for i in range(len(rows) - 1))
        print(f"\n  longest bin minus shortest  {d:+.4f}  against a bootstrap "
              f"spread of {s:.4f}")
        print(f"  monotone in duration  {mono}")
        if d > 2 * s and mono:
            verdict = (f"DRIFT. The contract reads {rows[-1]['auc']:.4f} on the "
                       f"longest trajectories against {rows[0]['auc']:.4f} on "
                       f"the shortest, a gap of {d:.4f} against a bootstrap "
                       f"spread of {s:.4f}, monotone across bins. The error "
                       "compounds along the rollout.")
        elif abs(d) <= 2 * s:
            verdict = (f"UNIFORM. Longest minus shortest is {d:+.4f} against a "
                       f"bootstrap spread of {s:.4f}. The per step error does "
                       "not compound and exposure bias is excluded.")
        else:
            verdict = (f"NON MONOTONE. Longest minus shortest {d:+.4f}, spread "
                       f"{s:.4f}, but the trend is not ordered. Reported as "
                       "measured, no verdict.")
        print(f"\n  VERDICT  {verdict}\n")
    else:
        verdict = "NOT ENOUGH POPULATED BINS"

    json.dump({"by": args.by, "bins": rows, "verdict": verdict},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


def _auc(Hb, Gb, seed):
    """The contract recipe, applied to a pair of already selected matrices.
    Identical to scoring.score_features: balance to the smaller side, human 0
    and generated 1, RandomForest(100, oob), AUC of the OOB decision function.
    Kept here rather than calling the scorer because the scorer always loads the
    full human file and these are subsets of it."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    n = min(len(Hb), len(Gb))
    X = np.vstack([Hb[:n], Gb[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 oob_score=True, n_jobs=-1,
                                 random_state=scoring.RF_SEED)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.oob_decision_function_[:, 1]))


def _boot(Hb, Gb, B, seed):
    rng = np.random.default_rng(seed)
    out = [_auc(Hb, Gb[rng.integers(0, len(Gb), len(Gb))], seed)
           for _ in range(B)]
    return float(np.std(out))


if __name__ == "__main__":
    main()
