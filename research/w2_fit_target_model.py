"""W2 stat-guided probe, step 1: fit an explicit CONDITIONAL model of the
human per-path feature distribution in detector space.

Per PLAN.md "W2 PROBE IS NEXT": the key delta vs burst 3/3b (which matched
BATCH MOMENTS -- one global mean/std per feature -- and died by estimator
bias + collapse) is fitting the JOINT distribution, conditioned on path
distance, so a target vector sampled for one spec is a plausible human
DRAW (full covariance, not independent per-feature marginals) instead of a
population average every path is pulled toward.

Method:
  1. Draw N_SAMPLE (>=5000) trajectories from the FULL TRAINING POOL split
     (training/zimt_dxdy.npy + zimt_lengths.npy + zimt_conditions.npy,
     identical perm(42)/tr_idx convention as compute_human_curv_targets.py
     and every chain*.py trainer -- never data/human_eval_features.npy,
     never data/human_val_features_grpo.npy).
  2. For each, compute the 11 COVERED_FEATURES (training/train_candi_chain3
     .py's smooth/differentiable feature set) via features.py's real
     extract_features on the reconstructed point trajectory, then transform
     to detector space with the SAME per-feature transform
     training/compute_human_curv_targets.py uses (research/
     cond_realization_probe.py's to_detector_space column spec).
  3. Bucket by path distance (zimt_conditions[:,0] is log_dist, already
     computed at data-prep time -- no need to reconstruct distance from the
     dxdy cumsum) into N_BUCKETS quantile buckets.
  4. Per bucket: fit a full-covariance multivariate Gaussian (mean vector +
     11x11 covariance, ridge-regularized for numerical PD) in detector
     space, plus the empirical 1st/99th percentile box per feature (for
     clipping sampled targets away from tail nonsense) and a median-capped
     effective weight per feature (chain3b's WEIGHT_CAP_RATIO=10 rule, reused
     verbatim: raw_weight = 1/std^2, clipped to
     [median/sqrt(10), median*sqrt(10)]) for the guided optimizer's loss.

Output: research/w2_target_model.npz (bucket_edges, bucket_mean, bucket_cov,
bucket_std, bucket_p1, bucket_p99, bucket_eff_weight, all shape-documented in
the npz itself via a companion research/w2_target_model_meta.json) --
read-only input to research/w2_stat_guided_probe.py, never touches any
training checkpoint.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from training.compute_human_curv_targets import (
    COVERED_FEATURES, DET_TRANSFORMS, trajectory_from_dxdy,
)
from features import extract_features

DATA_DIR = Path("training")
OUT_PATH = Path("research/w2_target_model.npz")
OUT_META_PATH = Path("research/w2_target_model_meta.json")

N_SAMPLE = 6000
N_BUCKETS = 6
SEED = 20260719
RIDGE = 1e-4
WEIGHT_CAP_RATIO = 10.0
_EPS = 1e-6


def compute_effective_weights(std_vec: np.ndarray, cap_ratio: float = WEIGHT_CAP_RATIO,
                               eps: float = _EPS) -> np.ndarray:
    """training/train_candi_chain3b.py's compute_effective_weights, reused
    verbatim (median-based capping so no near-zero-variance feature, e.g.
    mean_acceleration/mean_jerk, explodes the loss): raw_weight = 1/std^2,
    clipped into [median/sqrt(cap_ratio), median*sqrt(cap_ratio)]."""
    raw_std = np.maximum(std_vec, eps)
    raw_weight = 1.0 / (raw_std * raw_std)
    median_w = float(np.median(raw_weight))
    half_width = math.sqrt(cap_ratio)
    lo, hi = median_w / half_width, median_w * half_width
    return np.clip(raw_weight, lo, hi)


def main():
    print("[w2_fit] loading training pool...", flush=True)
    dxdy = np.load(DATA_DIR / "zimt_dxdy.npy", mmap_mode="r")
    lengths = np.load(DATA_DIR / "zimt_lengths.npy")
    conditions = np.load(DATA_DIR / "zimt_conditions.npy")
    N = len(lengths)

    n_val = min(N // 10, 30000)
    perm = np.random.default_rng(42).permutation(N)
    tr_idx = perm[n_val:]

    eligible = tr_idx[lengths[tr_idx] >= 5]
    print(f"[w2_fit] training pool: {len(tr_idx):,} trajectories, "
          f"{len(eligible):,} with length >= 5", flush=True)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(eligible, size=min(N_SAMPLE, len(eligible)), replace=False)
    print(f"[w2_fit] sampling {len(sample_idx):,} trajectories at native "
          f"lengths (seed={SEED})", flush=True)

    log_dist = conditions[sample_idx, 0].astype(np.float64)
    dist = np.exp(log_dist)

    feat_rows = []
    dist_kept = []
    n_skipped = 0
    for k, idx in enumerate(sample_idx):
        L = int(lengths[idx])
        # zimt_dxdy rows are NORMALIZED (net displacement == 1.0), despite the
        # legacy docstring calling them pixel displacements -- verified
        # empirically 2026-07-20 (cumsum endpoint 1.0 vs cond dist 84-352 px).
        # The detector measures PIXEL trajectories, so scale by the spec's
        # true distance or every time-based feature (velocity/accel/jerk) and
        # curvature lands in the wrong units by a per-path factor of ~dist.
        row = np.asarray(dxdy[idx], dtype=np.float64) * dist[k]
        traj = trajectory_from_dxdy(row, L)
        feats = extract_features(traj)
        if feats is None:
            n_skipped += 1
            continue
        from features import FEATURE_NAMES
        vec = np.array([feats[FEATURE_NAMES.index(name)] for name in COVERED_FEATURES],
                        dtype=np.float64)
        feat_rows.append(vec)
        dist_kept.append(dist[k])

    n_usable = len(feat_rows)
    print(f"[w2_fit] usable: {n_usable:,} (skipped {n_skipped})", flush=True)
    assert n_usable >= 5000, f"only {n_usable} usable samples, need >= 5000"

    feat_raw = np.stack(feat_rows, axis=0)  # (n_usable, 11)
    dist_kept = np.asarray(dist_kept, dtype=np.float64)

    # detector-space transform, per-feature (matches
    # compute_human_curv_targets.py / train_candi_chain3.py exactly)
    feat_det = np.empty_like(feat_raw)
    for j, name in enumerate(COVERED_FEATURES):
        feat_det[:, j] = DET_TRANSFORMS[name](feat_raw[:, j])

    # --- quantile buckets of distance ---
    quantiles = np.linspace(0.0, 1.0, N_BUCKETS + 1)
    bucket_edges = np.quantile(dist_kept, quantiles)
    bucket_edges[0] = -np.inf
    bucket_edges[-1] = np.inf
    bucket_idx = np.digitize(dist_kept, bucket_edges[1:-1], right=False)
    print(f"[w2_fit] bucket edges (finite part): "
          f"{np.round(bucket_edges[1:-1], 1).tolist()}", flush=True)

    n_feat = len(COVERED_FEATURES)
    bucket_mean = np.zeros((N_BUCKETS, n_feat))
    bucket_cov = np.zeros((N_BUCKETS, n_feat, n_feat))
    bucket_std = np.zeros((N_BUCKETS, n_feat))
    bucket_p1 = np.zeros((N_BUCKETS, n_feat))
    bucket_p99 = np.zeros((N_BUCKETS, n_feat))
    bucket_eff_weight = np.zeros((N_BUCKETS, n_feat))
    bucket_n = np.zeros(N_BUCKETS, dtype=np.int64)

    meta = {
        "covered_features": COVERED_FEATURES,
        "n_buckets": N_BUCKETS,
        "n_sample_usable": int(n_usable),
        "seed": SEED,
        "ridge": RIDGE,
        "weight_cap_ratio": WEIGHT_CAP_RATIO,
        "bucket_edges": bucket_edges.tolist(),
        "buckets": [],
    }

    for b in range(N_BUCKETS):
        m = bucket_idx == b
        n_b = int(m.sum())
        bucket_n[b] = n_b
        sub = feat_det[m]
        mean_b = sub.mean(axis=0)
        cov_b = np.cov(sub, rowvar=False) + RIDGE * np.eye(n_feat)
        std_b = np.sqrt(np.diag(cov_b))
        p1_b = np.percentile(sub, 1, axis=0)
        p99_b = np.percentile(sub, 99, axis=0)
        eff_w_b = compute_effective_weights(std_b)

        bucket_mean[b] = mean_b
        bucket_cov[b] = cov_b
        bucket_std[b] = std_b
        bucket_p1[b] = p1_b
        bucket_p99[b] = p99_b
        bucket_eff_weight[b] = eff_w_b

        lo_d = bucket_edges[b]
        hi_d = bucket_edges[b + 1]
        print(f"[w2_fit] bucket {b}: n={n_b} dist=[{lo_d:.1f},{hi_d:.1f}) "
              f"mean_dist={dist_kept[m].mean():.1f}", flush=True)
        for j, name in enumerate(COVERED_FEATURES):
            print(f"    {name:24s} mean={mean_b[j]:8.4f} std={std_b[j]:8.4f} "
                  f"eff_weight={eff_w_b[j]:10.4f} p1={p1_b[j]:8.4f} p99={p99_b[j]:8.4f}",
                  flush=True)
        max_w, min_w = eff_w_b.max(), eff_w_b.min()
        assert max_w / min_w <= WEIGHT_CAP_RATIO + 1e-6, \
            f"bucket {b}: weight cap violated ({max_w/min_w:.4f})"
        meta["buckets"].append({
            "bucket": b, "n": n_b, "dist_lo": float(lo_d), "dist_hi": float(hi_d),
            "mean_dist": float(dist_kept[m].mean()),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        bucket_edges=bucket_edges,
        bucket_mean=bucket_mean,
        bucket_cov=bucket_cov,
        bucket_std=bucket_std,
        bucket_p1=bucket_p1,
        bucket_p99=bucket_p99,
        bucket_eff_weight=bucket_eff_weight,
        bucket_n=bucket_n,
    )
    OUT_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[w2_fit] wrote {OUT_PATH} and {OUT_META_PATH}", flush=True)


if __name__ == "__main__":
    main()
