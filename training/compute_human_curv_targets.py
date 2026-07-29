"""One-off precompute for DIFFUSION_PILOT_V2.md burst 2 (retarget spec item 1)
and burst 3 (widened multi-feature spec item 2).

Computes fixed scalar targets the chain2/chain3 trainers penalize against:
mean/std, in DETECTOR SPACE, of every "covered" (smooth, differentiable)
feature burst 3 fits -- mean/std velocity, mean/std acceleration, mean/std
jerk, path_efficiency, angular_velocity mean/std, plus the original burst 2
pair curvature_mean/curvature_std -- measured with the EXACT features.py
recipe (extract_features, native un-padded lengths) on a random sample of
the FULL training pool.

Detector-space transforms per feature follow research/cond_realization_probe
.py's to_detector_space (the only place in the repo with a full-feature-
vector smooth transform spec), NOT phase_a_baseline.py's curvature-only
helper (which only covers the two curvature columns):
  mean_velocity, std_velocity, std_acceleration, std_jerk,
  angular_velocity_mean, angular_velocity_std: log1p(clip(x, 0, None))
  mean_acceleration: x / 1e4
  mean_jerk: x / 1e6
  path_efficiency: identity
  curvature_mean, curvature_std: log1p(clip(x, 0, None) * 1e3)

Data source: training/zimt_dxdy.npy + training/zimt_lengths.npy, restricted
to the TRAINING split only via the identical permutation/split convention
train_candi.py and train_candi_chain.py already use (perm = default_rng(42)
.permutation(N); n_val = min(N//10, 30000); tr_idx = perm[n_val:]). This is
the FULL training pool -- never data/human_val_features_grpo.npy (post-hoc
scoring only) and never data/human_eval_features.npy (never touched).

zimt_dxdy.npy is (N, 256, 2) float32 padded per-step pixel displacements at
the native 125Hz grid (training/prepare_zimt_data.py), so cumsum(dxdy[:T])
prepended with a (0,0) start point reproduces the same point sequence
features.py's extract_features expects, with no resampling needed.

Output: training/human_curv_targets.json (burst 2, curvature-only, kept for
continuity/backward-compat) AND training/human_feature_targets.json (burst
3, all 11 covered features).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from features import FEATURE_NAMES, extract_features

DATA_DIR = Path("training")
OUT_PATH_CURV = DATA_DIR / "human_curv_targets.json"
OUT_PATH_FEAT = DATA_DIR / "human_feature_targets.json"
N_SAMPLE = 5000
SEED = 20260719
HZ = 125.0

CURV_MEAN_IDX = FEATURE_NAMES.index("curvature_mean")
CURV_STD_IDX = FEATURE_NAMES.index("curvature_std")

# Burst 3: all covered (smooth, differentiable) features and their
# detector-space transform, per research/cond_realization_probe.py's
# to_detector_space column spec.
COVERED_FEATURES = [
    "mean_velocity",
    "std_velocity",
    "mean_acceleration",
    "std_acceleration",
    "mean_jerk",
    "std_jerk",
    "path_efficiency",
    "curvature_mean",
    "curvature_std",
    "angular_velocity_mean",
    "angular_velocity_std",
]


def _lg(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0.0, None))


DET_TRANSFORMS = {
    "mean_velocity": _lg,
    "std_velocity": _lg,
    "mean_acceleration": lambda x: x / 1e4,
    "std_acceleration": _lg,
    "mean_jerk": lambda x: x / 1e6,
    "std_jerk": _lg,
    "path_efficiency": lambda x: x,
    "curvature_mean": lambda x: np.log1p(np.clip(x, 0.0, None) * 1e3),
    "curvature_std": lambda x: np.log1p(np.clip(x, 0.0, None) * 1e3),
    "angular_velocity_mean": _lg,
    "angular_velocity_std": _lg,
}


def to_detector_space(col: np.ndarray) -> np.ndarray:
    """research/phase_a_baseline.py:to_detector_space_curv, verbatim (kept
    for the burst-2 curvature-only output)."""
    return np.log1p(np.clip(col, 0.0, None) * 1e3)


def trajectory_from_dxdy(dxdy_row: np.ndarray, length: int):
    """Reconstruct a (x, y, t) point list from a native-length dxdy slice,
    the same cumsum-from-zero convention the chain trainers' differentiable
    decode uses (translation-invariant, so starting at (0, 0) is equivalent
    to any real start position for every feature this module measures)."""
    dx = dxdy_row[:length, 0].astype(np.float64)
    dy = dxdy_row[:length, 1].astype(np.float64)
    x = np.concatenate([[0.0], np.cumsum(dx)])
    y = np.concatenate([[0.0], np.cumsum(dy)])
    t = np.arange(length + 1, dtype=np.float64) / HZ
    return list(zip(x.tolist(), y.tolist(), t.tolist()))


def main():
    print("[targets] loading training pool...", flush=True)
    dxdy = np.load(DATA_DIR / "zimt_dxdy.npy", mmap_mode="r")
    lengths = np.load(DATA_DIR / "zimt_lengths.npy")
    N = len(lengths)

    # identical train/val split convention to train_candi.py / train_candi_chain.py
    n_val = min(N // 10, 30000)
    perm = np.random.default_rng(42).permutation(N)
    tr_idx = perm[n_val:]

    eligible = tr_idx[lengths[tr_idx] >= 5]  # extract_features requires len(pts) >= 5
    print(f"[targets] training pool: {len(tr_idx):,} trajectories, "
          f"{len(eligible):,} with length >= 5", flush=True)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(eligible, size=min(N_SAMPLE, len(eligible)), replace=False)
    print(f"[targets] sampling {len(sample_idx):,} trajectories at native lengths "
          f"(seed={SEED})", flush=True)

    feat_cols = {name: [] for name in COVERED_FEATURES}
    name_to_idx = {name: FEATURE_NAMES.index(name) for name in COVERED_FEATURES}
    n_skipped = 0
    for idx in sample_idx:
        L = int(lengths[idx])
        row = np.asarray(dxdy[idx])
        traj = trajectory_from_dxdy(row, L)
        feats = extract_features(traj)
        if feats is None:
            n_skipped += 1
            continue
        for name, fidx in name_to_idx.items():
            feat_cols[name].append(feats[fidx])

    n_usable = len(feat_cols["curvature_mean"])
    print(f"[targets] usable: {n_usable:,} (skipped {n_skipped})", flush=True)

    for name in feat_cols:
        feat_cols[name] = np.asarray(feat_cols[name], dtype=np.float64)

    # --- burst 2 output (curvature-only, kept for continuity) ---
    curv_mean = feat_cols["curvature_mean"]
    curv_std = feat_cols["curvature_std"]
    det_mean = to_detector_space(curv_mean)
    det_std = to_detector_space(curv_std)
    targets_curv = {
        "n_sample": int(n_usable),
        "seed": SEED,
        "source": "training/zimt_dxdy.npy training split (perm seed 42, tr_idx = perm[n_val:])",
        "recipe": "features.py extract_features native lengths, log1p(clip(x,0)*1e3) detector transform",
        "target_mean_curvature_mean_det": float(det_mean.mean()),
        "target_std_curvature_mean_det": float(det_mean.std()),
        "target_mean_curvature_std_det": float(det_std.mean()),
        "target_std_curvature_std_det": float(det_std.std()),
    }
    print("[targets] burst2 curvature-only: " + json.dumps(targets_curv, indent=2), flush=True)
    OUT_PATH_CURV.write_text(json.dumps(targets_curv, indent=2))
    print(f"[targets] wrote {OUT_PATH_CURV}", flush=True)

    # --- burst 3 output (all covered features) ---
    targets_feat = {
        "n_sample": int(n_usable),
        "seed": SEED,
        "source": "training/zimt_dxdy.npy training split (perm seed 42, tr_idx = perm[n_val:])",
        "recipe": "features.py extract_features native lengths, per-feature "
                  "detector-space transform per research/cond_realization_probe.py "
                  "to_detector_space",
        "covered_features": COVERED_FEATURES,
    }
    for name in COVERED_FEATURES:
        det = DET_TRANSFORMS[name](feat_cols[name])
        targets_feat[f"target_mean_{name}_det"] = float(det.mean())
        targets_feat[f"target_std_{name}_det"] = float(det.std())
    print("[targets] burst3 all-feature: " + json.dumps(targets_feat, indent=2), flush=True)
    OUT_PATH_FEAT.write_text(json.dumps(targets_feat, indent=2))
    print(f"[targets] wrote {OUT_PATH_FEAT}", flush=True)


if __name__ == "__main__":
    main()
