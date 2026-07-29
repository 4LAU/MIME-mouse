"""Upsample-then-quantize experiment on saved sub-pixel paths (CPU only).

Hypothesis: the +0.20 rounding tax exists because our synthetic paths are
emitted AT exactly 125 Hz, so features.py's resample_trajectory (np.interp
onto a 125 Hz grid) is an identity map and the integer staircase survives
verbatim. Human data is integer coords at its NATIVE (higher/irregular) event
rate, and the same 125 Hz resample linearly interpolates between events,
smoothing the staircase into sub-pixel values. If we serve integer coords at
a higher rate (or as sparse hardware-style events), the harness resample
should smooth them human-style.

Variants (all from research/phase_a_trajs_noround.pkl, the un-rounded
continuous paths):
  upsample-control  linear interp of positions to 250 Hz timestamps, NO
                    rounding (sanity: should score ~= the 0.76 control)
  up250-round       interp to 250 Hz, then np.round positions to integers
  up500-round       interp to 500 Hz, then np.round positions to integers
  event-sim         emulate a mouse sensor at 500 Hz along the continuous
                    path: emit a point only on ticks where the rounded
                    integer position differs from the last reported one
                    (sparse, irregular-timestamp integer event stream),
                    plus always the first and last point

First and last points stay pinned to their spec values in every variant.
Scoring harness is identical to research/quantize_schemes.py: features.py
extract_feature_matrix (does the 125 Hz resample internally) -> RF OOB
(n_estimators=100, random_state=42) vs data/human_val_features_grpo.npy,
N=2000 per class. Uses human_val ONLY, never the protected eval file.

Usage:
    .venv/Scripts/python.exe research/quantize_upsample.py \
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


def det_curv(col):
    """Detector-space transform for the curvature columns (see
    research/cond_realization_probe.py to_detector_space)."""
    return np.log1p(np.clip(col, 0.0, None) * 1e3)


def _interp_grid(pts, hz):
    """Linearly interpolate a (n,3) trajectory onto an hz-rate time grid,
    always ending exactly at the original final timestamp."""
    x, y, t = pts[:, 0], pts[:, 1], np.maximum.accumulate(pts[:, 2])
    step = 1.0 / hz
    tt = np.arange(t[0], t[-1], step, dtype=np.float64)
    if tt.size == 0 or tt[-1] < t[-1]:
        tt = np.append(tt, t[-1])
    return np.interp(tt, t, x), np.interp(tt, t, y), tt


def upsample(traj, hz, do_round):
    """Variant 1-3: interp to hz, optionally round positions, pin endpoints."""
    if len(traj) <= 2:
        return list(traj)
    pts = np.asarray(traj, dtype=np.float64)
    xx, yy, tt = _interp_grid(pts, hz)
    if do_round:
        xx = np.round(xx)
        yy = np.round(yy)
    out = [tuple(traj[0])]
    for i in range(1, len(tt) - 1):
        out.append((float(xx[i]), float(yy[i]), float(tt[i])))
    out.append(tuple(traj[-1]))
    return out


def event_sim(traj, hz=500.0):
    """Variant 4: walk the continuous path at hz ticks, maintain a reported
    integer position, emit a point only when round(true) differs from the
    last reported position (in x or y). Sparse irregular-timestamp integer
    events, like real hardware. First and last points pinned to spec."""
    if len(traj) <= 2:
        return list(traj)
    pts = np.asarray(traj, dtype=np.float64)
    xx, yy, tt = _interp_grid(pts, hz)
    out = [tuple(traj[0])]
    rep_x = float(np.round(pts[0, 0]))
    rep_y = float(np.round(pts[0, 1]))
    for i in range(1, len(tt) - 1):
        rx = float(np.round(xx[i]))
        ry = float(np.round(yy[i]))
        if rx != rep_x or ry != rep_y:
            out.append((rx, ry, float(tt[i])))
            rep_x, rep_y = rx, ry
    out.append(tuple(traj[-1]))
    return out


VARIANTS = [
    ("upsample-control", lambda t: upsample(t, 250.0, do_round=False)),
    ("up250-round", lambda t: upsample(t, 250.0, do_round=True)),
    ("up500-round", lambda t: upsample(t, 500.0, do_round=True)),
    ("event-sim", lambda t: event_sim(t, 500.0)),
]


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
    print(f"[upsample] loaded {len(trajectories)} trajectories from "
          f"{args.trajectories}", flush=True)

    human_features = np.load(DATA_DIR / "human_val_features_grpo.npy")
    print(f"[upsample] human_val_features_grpo.npy shape="
          f"{human_features.shape}", flush=True)

    for name, fn in VARIANTS:
        vtrajs = [fn(t) for t in trajectories]
        n_pts = np.array([len(t) for t in vtrajs])
        synth_features = extract_feature_matrix(vtrajs)
        auc, clf, human_bal, synth_bal, n_use = score(
            synth_features, human_features)

        ratio_cm = (float(np.std(det_curv(synth_bal[:, CURV_MEAN_IDX])))
                    / max(float(np.std(det_curv(human_bal[:, CURV_MEAN_IDX]))), 1e-12))
        ratio_cs = (float(np.std(det_curv(synth_bal[:, CURV_STD_IDX])))
                    / max(float(np.std(det_curv(human_bal[:, CURV_STD_IDX]))), 1e-12))

        ndc_s = float(np.mean(synth_bal[:, NDC_IDX]))
        ndc_h = float(np.mean(human_bal[:, NDC_IDX]))

        imp = clf.feature_importances_
        top5 = np.argsort(imp)[::-1][:5]

        print(f"\n=== VARIANT: {name} ===")
        print(f"N per class: {n_use} (valid synth: {len(synth_features)}; "
              f"points/traj mean={n_pts.mean():.1f} median={np.median(n_pts):.0f})")
        print(f"AUC (RF OOB): {auc:.4f}")
        print(f"detector-space variety ratio curvature_mean: {ratio_cm:.4f}")
        print(f"detector-space variety ratio curvature_std:  {ratio_cs:.4f}")
        print(f"num_direction_changes mean (raw): synth={ndc_s:.2f} "
              f"human={ndc_h:.2f}")
        print("top-5 RF importances:")
        for idx in top5:
            print(f"  {FEATURE_NAMES[idx]}: {imp[idx]:.4f}")


if __name__ == "__main__":
    main()
