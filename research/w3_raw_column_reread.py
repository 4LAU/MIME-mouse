"""Re-read the findings that shaped P3's design brief, with a raw-model column.

On 2026-07-26 the endpoint correction turned out to be injecting the defect it
was being blamed for. correct_additive drifts every position and rounds each one
independently; rounding a ramp is a staircase, and a riser in the middle of a
straight run reads as a 45 or 90 degree turn. Measured per speed class, the
sampler was at or below the human almost everywhere and the correction put the
excess in.

Every W3 probe before that date scored a single arm built with correct_additive
and had no raw column to compare it against. w3_stall_pattern.py is the one
exception; it carried the control and it is the reason the operator was caught
at all. The rest are listed in this file's RE_READ table.

Four of them concluded something about TURNING, which is exactly what the
operator manufactures, and one of those four is the finding P3's design brief
currently rests on:

  envelope_ceiling   "the gap lives in turning, worth 0.143 when removed, and
                     the local wobble half carries it, not the whole-path half"
                     That sentence is why the brief asks for a locally
                     dispersion-calibrated architecture. If the operator put
                     the wobble there, the brief is aimed at an artefact.
  turn_floor         "at 100 to 400 px/s the model turns 12.03 deg against the
                     human 4.76, 2.5x"
  style_variance     "angular_velocity_mean is the RF's heaviest feature at
                     weight 0.113 and a per-path style explains almost none
                     of it"
  missing_paths      "the uncovered quarter is smooth low-wobble movement,
                     angular_velocity_mean 15 against the model's unvarying 29"

All four readings are downstream of how much the arm turns, and the arm they
were read on had been dithered. This runs the same three instruments over three
arms instead of one:

  raw        the model's own output, no correction, arrives 0.3% of the time.
             Not servable. It is the control, not a candidate.
  additive   the arm every pre-07-26 finding was actually read on.
  jog        exact arrival by whole-pixel jogs on the longest steps, every other
             step byte identical to the model's own. Servable, and the standing
             single-trajectory number.

Read the raw and jog columns together. A conclusion that holds on raw AND jog
was about the model and survives. One that only holds on additive was about the
operator, and whatever it was used to justify needs re-deriving.

Instruments, all three from the probes being re-read, unchanged except for
looping over arms:

  families   RF-OOB AUC on feature-family column subsets, "alone" and
             "without", the w3_envelope_ceiling recipe. Diagnostics on subsets,
             never contract numbers; the full-18 value is printed beside the
             contract's own so drift is visible.
  weights    contract RF feature importances, which is what style_variance's
             premise rests on.
  turn       mean and median |dtheta| per speed class over motion events, the
             w3_turn_by_class recipe, human beside each arm.

No generation, no GPU, no checkpoint touched. Arms come from the landing cache
(fc_v2, 6000 paths) and the jog cache (resid_v2), so this is CPU and minutes.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_raw_column_reread.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from degeneracy_panel import (features_with_jitter, real_paths,  # noqa: E402
                              _score_against)
from features import FEATURE_NAMES  # noqa: E402
from w3_aiming_price import correct_jog  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402
from w3_turn_by_class import gather, s_to_class  # noqa: E402

OUT = R / "research" / "w3_raw_column_reread_results.json"

CACHES = {
    "fc_v2": R / "research" / "w3_landing_cache.pkl",
    "resid_v2": R / "research" / "w3_jog_cache_event_polar_4m_resid_v2.pkl",
}

# every W3 probe that scored a correct_additive arm, and whether it already had
# a raw column. Recorded here so the audit is part of the artifact rather than
# a claim in a summary.
RE_READ = {
    "w3_stall_pattern": "HAS a raw column already; it is how the operator was found",
    "w3_envelope_ceiling": "single additive arm; conclusion is about turning",
    "w3_turn_floor": "single additive arm; conclusion is about turning",
    "w3_style_variance": "single additive arm; conclusion is about turning",
    "w3_missing_paths": "single additive arm; conclusion is about turning",
    "w3_stall_surgery": "single additive arm; scores a hold-deletion surgery",
    "w3_gap_anatomy": "single additive arm; conclusion is about timestamps",
    "w3_critic_coverage": "single additive arm; trains a critic on it",
    "w3_corpus_coverage": "single additive arm; conclusion is about the corpus "
                          "builder, independent of the arm's motion",
}

GROUPS = {
    "envelope": ["mean_velocity", "std_velocity", "max_velocity",
                 "velocity_skewness", "time_to_peak_velocity",
                 "movement_duration"],
    "turning": ["curvature_mean", "curvature_std", "angular_velocity_mean",
                "angular_velocity_std", "num_direction_changes",
                "path_efficiency", "max_deviation"],
    "derivatives": ["mean_acceleration", "std_acceleration",
                    "max_acceleration", "mean_jerk", "std_jerk"],
    "turning: wobble": ["curvature_mean", "curvature_std",
                        "angular_velocity_mean", "angular_velocity_std",
                        "num_direction_changes"],
    "turning: excursion": ["path_efficiency", "max_deviation"],
}

OPS = {"additive": correct_additive, "jog": correct_jog}
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def subset_auc(Xs, Xh, cols):
    """RF-OOB AUC on a column subset, score_features' own recipe.

    score_features cannot take a subset (its dispersion battery is fixed at 18
    columns), so the AUC half is repeated here with the same estimator, seed,
    balancing and OOB decision function. Copied from w3_envelope_ceiling so the
    numbers are comparable to the run being re-read. Diagnostics, never a
    contract number.
    """
    n = min(len(Xs), len(Xh))
    X = np.vstack([Xh[:n][:, cols], Xs[:n][:, cols]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 oob_score=True, n_jobs=-1,
                                 random_state=scoring.RF_SEED)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.oob_decision_function_[:, 1]))


def load_arms(cache):
    """raw / additive / jog for one cached model arm.

    pickle.load: repo-own artifact written by the landing-price and jog runs on
    this machine, never third-party input.
    """
    with open(cache, "rb") as fh:
        specs, trajs = pickle.load(fh)
    keep = [(tuple(int(v) for v in np.asarray(s)),
             np.asarray(t, dtype=np.float64))
            for s, t in zip(specs, trajs) if t is not None and len(t) >= 3]
    raw = [t for _, t in keep]
    arms = {"raw": raw}
    for op, f in OPS.items():
        arms[op] = [f(t, *s) for s, t in keep]
    return arms


def families(arms, Xr, seed, out):
    print(f"\n{'':<30}" + "".join(f"{a:>22}" for a in arms))
    print(f"{'feature family':<30}" + "".join(
        f"{'alone':>11}{'without':>11}" for _ in arms))
    X = {}
    for a, paths in arms.items():
        M = features_with_jitter(paths, 0.0, seed)
        X[a] = M[np.all(np.isfinite(M), axis=1)]
    all_cols = np.arange(len(FEATURE_NAMES))
    row = {}
    line = f"{'all 18 (contract)':<30}"
    for a in arms:
        c = float(_score_against(X[a], Xr)["auc_rf_oob"])
        f18 = subset_auc(X[a], Xr, all_cols)
        row[a] = {"contract_vs_rebuilt": c, "all18": f18}
        line += f"{f18:>11.4f}{'':>11}"
    print(line)
    fam = {"_all18": row}
    for name, cols in GROUPS.items():
        c = np.array([IDX[f] for f in cols])
        keep = np.setdiff1d(all_cols, c)
        line = f"{name + f' ({len(cols)})':<30}"
        fam[name] = {}
        for a in arms:
            al = subset_auc(X[a], Xr, c)
            wo = subset_auc(X[a], Xr, keep)
            fam[name][a] = {"alone": al, "without": wo,
                            "worth": fam["_all18"][a]["all18"] - wo}
            line += f"{al:>11.4f}{wo:>11.4f}"
        print(line)
    out["families"] = fam

    print(f"\n{'what a family is worth if matched perfectly':<44}"
          + "".join(f"{a:>12}" for a in arms))
    for name in GROUPS:
        print(f"{name:<44}" + "".join(
            f"{fam[name][a]['worth']:>12.4f}" for a in arms))
    return X


def weights(X, Xr, arms, out):
    print(f"\n{'contract RF importance':<28}" + "".join(f"{a:>12}" for a in arms))
    imp = {}
    for a in arms:
        n = min(len(X[a]), len(Xr))
        M = np.vstack([Xr[:n], X[a][:n]])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                     oob_score=True, n_jobs=-1,
                                     random_state=scoring.RF_SEED)
        clf.fit(M, y)
        imp[a] = dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist()))
    order = sorted(FEATURE_NAMES, key=lambda f: -imp["additive"][f])[:8]
    for f in order:
        print(f"{f:<28}" + "".join(f"{imp[a][f]:>12.4f}" for a in arms))
    out["rf_importance"] = imp


def turn_by_class(arms, real, out):
    H = gather(real)
    A = {a: gather(p) for a, p in arms.items()}
    edges = [(1, 5), (6, 11), (12, 21), (22, 31), (32, 99)]
    print(f"\nmean |turn| per motion event, degrees, by speed class")
    print(f"{'speed class':<14}{'px/s':>9}{'human':>9}"
          + "".join(f"{a:>10}" for a in arms) + f"{'human share':>13}")
    rows = []
    for lo, hi in edges:
        hm = (H[:, 0] >= lo) & (H[:, 0] <= hi)
        if hm.sum() < 200:
            continue
        vel = float(np.median(H[hm, 2]))
        h_turn = float(np.mean(H[hm, 1]))
        vals = {}
        for a in arms:
            am = (A[a][:, 0] >= lo) & (A[a][:, 0] <= hi)
            vals[a] = (float(np.mean(A[a][am, 1])) if am.sum() >= 200
                       else float("nan"))
        share = float(hm.mean())
        print(f"{f'{lo} to {hi}':<14}{vel:>9.0f}{h_turn:>9.2f}"
              + "".join(f"{vals[a]:>10.2f}" for a in arms)
              + f"{share:>12.1%}")
        rows.append({"class_lo": lo, "class_hi": hi, "median_velocity": vel,
                     "human_mean_turn": h_turn, "arm_mean_turn": vals,
                     "human_event_share": share,
                     "n_human_events": int(hm.sum())})
    out["turn_by_class"] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--arms", nargs="+", default=["fc_v2", "resid_v2"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    real = [np.asarray(p, dtype=np.float64)
            for p in real_paths(args.n_real, args.seed, "ref")]
    Xr = features_with_jitter(real, 0.0, args.seed)
    Xr = Xr[np.all(np.isfinite(Xr), axis=1)]
    print(f"[reread] {len(Xr)} real reference paths")
    print(f"[reread] probes audited: "
          f"{sum(1 for v in RE_READ.values() if 'HAS' not in v)} of "
          f"{len(RE_READ)} scored a single correct_additive arm with no raw "
          f"column")

    out = {"seed": args.seed, "audit": RE_READ, "arms": {}}
    for name in args.arms:
        cache = CACHES[name]
        if not cache.exists():
            print(f"[reread] MISSING {cache}, skipping {name}")
            continue
        arms = load_arms(cache)
        print(f"\n{'='*74}\n=== {name}: {len(arms['raw'])} paths, "
              f"arms {list(arms)}\n{'='*74}")
        o = {}
        X = families(arms, Xr, args.seed, o)
        weights(X, Xr, arms, o)
        turn_by_class(arms, real, o)
        out["arms"][name] = o

    print(f"\n=== read ===")
    for name, o in out["arms"].items():
        f = o["families"]
        for fam in ("turning", "turning: wobble", "turning: excursion"):
            w = f[fam]
            print(f"{name} {fam:<22} worth: raw {w['raw']['worth']:+.4f}  "
                  f"additive {w['additive']['worth']:+.4f}  "
                  f"jog {w['jog']['worth']:+.4f}")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[reread] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
