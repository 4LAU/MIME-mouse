"""Offline integer-quantization scheme comparison on saved sub-pixel paths.

Loads the raw (un-rounded) trajectories pickled by
research/phase_a_baseline.py --no-round --save-trajectories, applies several
integer-pixel quantization schemes offline (CPU only, no GPU, no model), and
scores each with the identical harness: features.py extract_feature_matrix
-> RF OOB (n_estimators=100, random_state=42) vs
data/human_val_features_grpo.npy at N=2000.

Motivation: adding naive np.round() at decode time raised AUC from ~0.757 to
~0.954 (the "rounding tax"). Hypothesis: the tax comes from pixel-boundary
flip-flop creating a staircase/zigzag texture (inflated num_direction_changes,
jerk, angular velocity). Schemes:

  control-none  no quantization (should reproduce ~0.757)
  naive         np.round per point (should reproduce ~0.954)
  hyst-D        deadband/hysteresis: reported pixel stays fixed until the true
                position moves >= D px from the LAST REPORTED pixel, then
                jumps to round(true). D in {0.5, 0.6, 0.75}. Kills boundary
                flip-flop.
  monotone      within each run where true x (resp y) is monotone, the rounded
                sequence is forced monotone too (regressions clipped to the
                previous reported value); rounds normally at direction changes.

Schemes are applied to x and y independently. First and last points stay
pinned to their spec values, exactly as build_trajectory does (it rounds all
interior samples then overwrites the endpoints with the spec floats).

Uses data/human_val_features_grpo.npy ONLY (never the protected
data/human_eval_features.npy).

Usage:
    .venv/Scripts/python.exe research/quantize_schemes.py \
        --trajectories research/phase_a_trajs_noround.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from features import FEATURE_NAMES, extract_feature_matrix

DATA_DIR = Path("data")
SEED = 42

CURV_MEAN_IDX = FEATURE_NAMES.index("curvature_mean")
CURV_STD_IDX = FEATURE_NAMES.index("curvature_std")
NDC_IDX = FEATURE_NAMES.index("num_direction_changes")
MEAN_JERK_IDX = FEATURE_NAMES.index("mean_jerk")
STD_JERK_IDX = FEATURE_NAMES.index("std_jerk")


# --- detector-space transforms (research/cond_realization_probe.py
#     to_detector_space, the space the RF/checkpoint actually operate in) ---

def det_curv(col):
    return np.log1p(np.clip(col, 0.0, None) * 1e3)


def det_mean_jerk(col):
    return np.asarray(col, dtype=np.float64) / 1e6


def det_std_jerk(col):
    return np.log1p(np.clip(col, 0.0, None))


# --- quantization schemes (1-D, applied to x and y independently) ---

def q_none(v):
    return np.asarray(v, dtype=np.float64).copy()


def q_naive(v):
    return np.round(np.asarray(v, dtype=np.float64))


def q_hysteresis(v, deadband):
    """Reported pixel stays at the last reported value until the true
    position moves >= deadband px away from it, then jumps to round(true).
    Reporting state starts at the true first sample (the pinned start)."""
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    rep = v[0]
    for i in range(len(v)):
        if abs(v[i] - rep) >= deadband:
            rep = np.round(v[i])
        out[i] = rep
    return out


def q_monotone(v):
    """np.round, then clip regressions inside monotone runs of the true
    signal: while true is nondecreasing the reported value may not decrease
    (and vice versa); at a true direction change it rounds normally."""
    v = np.asarray(v, dtype=np.float64)
    r = np.round(v)
    for i in range(1, len(v)):
        dv = v[i] - v[i - 1]
        if dv > 0:
            if r[i] < r[i - 1]:
                r[i] = r[i - 1]
        elif dv < 0:
            if r[i] > r[i - 1]:
                r[i] = r[i - 1]
        else:
            r[i] = r[i - 1]
    return r


SCHEMES = [
    ("control-none", lambda x: q_none(x)),
    ("naive", lambda x: q_naive(x)),
    ("hyst-0.5", lambda x: q_hysteresis(x, 0.5)),
    ("hyst-0.6", lambda x: q_hysteresis(x, 0.6)),
    ("hyst-0.75", lambda x: q_hysteresis(x, 0.75)),
    ("monotone", lambda x: q_monotone(x)),
]


def apply_scheme(traj, fn):
    """Quantize the interior of one trajectory with fn; first and last points
    stay pinned to their spec values (mirrors build_trajectory, which rounds
    every generated sample then overwrites the endpoints)."""
    if len(traj) <= 2:
        return list(traj)
    pts = np.asarray(traj, dtype=np.float64)
    qx = fn(pts[:, 0])
    qy = fn(pts[:, 1])
    out = [tuple(traj[0])]
    for i in range(1, len(traj) - 1):
        out.append((float(qx[i]), float(qy[i]), float(pts[i, 2])))
    out.append(tuple(traj[-1]))
    return out


def score(synth_features, human_features):
    n_use = min(len(human_features), len(synth_features))
    human_bal = human_features[:n_use]
    synth_bal = synth_features[:n_use]
    X = np.vstack([human_bal, synth_bal])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    clf = RandomForestClassifier(
        n_estimators=100, oob_score=True, n_jobs=-1, random_state=SEED,
    )
    clf.fit(X, y)
    auc = roc_auc_score(y, clf.oob_decision_function_[:, 1])
    return auc, clf, human_bal, synth_bal, n_use


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=str,
                    default="research/phase_a_trajs_noround.pkl")
    args = ap.parse_args()

    with open(args.trajectories, "rb") as fh:
        trajectories = pickle.load(fh)
    print(f"[quant] loaded {len(trajectories)} trajectories from "
          f"{args.trajectories}", flush=True)

    human_features = np.load(DATA_DIR / "human_val_features_grpo.npy")
    print(f"[quant] human_val_features_grpo.npy shape={human_features.shape}",
          flush=True)

    for name, fn in SCHEMES:
        qtrajs = [apply_scheme(t, fn) for t in trajectories]
        synth_features = extract_feature_matrix(qtrajs)
        auc, clf, human_bal, synth_bal, n_use = score(
            synth_features, human_features)

        det_cm_s = det_curv(synth_bal[:, CURV_MEAN_IDX])
        det_cm_h = det_curv(human_bal[:, CURV_MEAN_IDX])
        det_cs_s = det_curv(synth_bal[:, CURV_STD_IDX])
        det_cs_h = det_curv(human_bal[:, CURV_STD_IDX])
        ratio_cm = float(np.std(det_cm_s)) / max(float(np.std(det_cm_h)), 1e-12)
        ratio_cs = float(np.std(det_cs_s)) / max(float(np.std(det_cs_h)), 1e-12)

        ndc_s = float(np.mean(synth_bal[:, NDC_IDX]))
        ndc_h = float(np.mean(human_bal[:, NDC_IDX]))

        mj_s = float(np.mean(det_mean_jerk(synth_bal[:, MEAN_JERK_IDX])))
        mj_h = float(np.mean(det_mean_jerk(human_bal[:, MEAN_JERK_IDX])))
        sj_s = float(np.mean(det_std_jerk(synth_bal[:, STD_JERK_IDX])))
        sj_h = float(np.mean(det_std_jerk(human_bal[:, STD_JERK_IDX])))

        imp = clf.feature_importances_
        top5 = np.argsort(imp)[::-1][:5]

        print(f"\n=== SCHEME: {name} ===")
        print(f"N per class: {n_use} (valid synth: {len(synth_features)})")
        print(f"AUC (RF OOB): {auc:.4f}")
        print(f"detector-space variety ratio curvature_mean: {ratio_cm:.4f}")
        print(f"detector-space variety ratio curvature_std:  {ratio_cs:.4f}")
        print(f"num_direction_changes mean (raw): synth={ndc_s:.2f} "
              f"human={ndc_h:.2f}")
        print(f"mean_jerk detector-space mean (raw/1e6): synth={mj_s:.4f} "
              f"human={mj_h:.4f}")
        print(f"std_jerk detector-space mean (log1p): synth={sj_s:.4f} "
              f"human={sj_h:.4f}")
        print("top-5 RF importances:")
        for idx in top5:
            print(f"  {FEATURE_NAMES[idx]}: {imp[idx]:.4f}")
        import sys
        sys.stdout.flush()


if __name__ == "__main__":
    main()
