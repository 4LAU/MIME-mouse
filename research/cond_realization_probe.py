"""Conditioning-input correction probe for event_polar_4m_fc_v2.

Hypothesis under test: the feature-conditioned event-stream model realizes
its commanded 18-dim "movement character" (drawn from a KDE over a bank of
real human feature vectors, see experiments/event_stream_polar.py) with a
SYSTEMATIC bias, and correcting the conditioning INPUT (aiming off-center so
the realization lands on-center) lowers the pure-model detector AUC vs its
~0.65 baseline toward 0.50 -- without shrinking conditional variance (the
distribution-collapse failure of the earlier EVENT_BESTOF experiment).

Two independent research hooks in experiments/event_stream_polar.py make
this measurable, both inert unless their env var is set:
  EVENT_FEAT_LOG=<path>   logs every commanded (pristine KDE draw) feat
                          vector and its spec log-distance to an .npz.
  EVENT_FEAT_CORR=<path>  applies feat <- feat @ M.T + v to the draw before
                          it reaches the model.

Realized features are computed from the DECODED trajectory via features.py
(extract_features on resample_trajectory output, per repo protocol), then
z-scored with the checkpoint's own feat_mu/feat_sd and clamped to +/-10 --
the exact convention the checkpoint's bank and the Best-of-N code path use.
This is equivalent to (but simpler than) re-deriving detector_features on
GPU tensors, since features.py's extract_features implements the same 18
statistics as training/train_events_polar_dm.py:detector_features, and the
decode already applied the same EVENT_SNAP lattice-snap the bank was built
from.

Protocol (non-negotiable, see task):
  - N=2000 synthetics per stage.
  - VALIDATION humans only: data/human_val_features_grpo.npy. NEVER
    data/human_eval_features.npy.
  - RF detector: RandomForestClassifier(n_estimators=100, oob_score=True,
    n_jobs=-1, random_state=42), humans=0/synth=1, roc_auc_score on
    oob_decision_function_[:, 1].
  - Serving env matches the documented 0.6470 (seed 42, N=2000) pure-model
    control config.

Each stage is its own process invocation (a bounded, minutes-scale GPU
burst); env vars are read at experiments.event_stream_polar import time, so
the env must be set before that import happens.

Usage:
    .venv/Scripts/python.exe research/cond_realization_probe.py --stage measure
    .venv/Scripts/python.exe research/cond_realization_probe.py --stage corrected --lam 1.0
    .venv/Scripts/python.exe research/cond_realization_probe.py --stage corrected --lam 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART_DIR = REPO / "research" / "cond_probe_artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

CKPT_NAME = "event_polar_4m_fc_v2.pt"
SERVING_ENV = {
    "EVENT_CKPT": CKPT_NAME,
    "EVENT_ORDER": "gumbel",
    "EVENT_CHOICE_TEMP": "10",
    "EVENT_SNAP": "2.5",
    "EVENT_DUR_STD": "1.0",
    "DUR_EMPIRICAL": "1",
}

N_SYNTH = 2000
SEED = 42
RIDGE_ALPHA = 1.0
CORR_EPS = 1e-2          # Tikhonov regularization for the inverse
W_ABSORB_THRESHOLD = 0.10  # |w| below this -> ignore log-distance in the correction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["measure", "corrected"])
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--n", type=int, default=N_SYNTH)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def build_specs(n: int, seed: int):
    """Mirror evaluate.py's generate_synthetic_trajectories spec sampling
    exactly: one rng, center screen, distance drawn from the human empirical
    distribution, uniform angle, drawn dist-then-angle per spec in order."""
    import numpy as np
    rng = np.random.default_rng(seed)
    distances = np.load(REPO / "data" / "human_distances.npy")
    cx, cy = 960.0, 540.0
    specs = []
    for _ in range(n):
        dist = float(rng.choice(distances))
        angle = float(rng.uniform(0, 2 * np.pi))
        specs.append((cx, cy, cx + dist * math.cos(angle), cy + dist * math.sin(angle)))
    return specs


def load_feat_stats():
    import torch
    ckpt = torch.load(REPO / "training" / CKPT_NAME, map_location="cpu",
                      weights_only=False)
    return ckpt["feat_mu"].numpy(), ckpt["feat_sd"].numpy()


def generate(env_overrides: dict, n: int, seed: int):
    """Set env, import the experiment module fresh, generate N trajectories.
    Must be called at most once per process (module-level env is read at
    import time)."""
    for k, v in {**SERVING_ENV, **env_overrides}.items():
        os.environ[k] = v
    sys.path.insert(0, str(REPO))
    from experiments import event_stream_polar as m
    specs = build_specs(n, seed)
    t0 = time.perf_counter()
    trajs = m.generate_paths(specs)
    elapsed = time.perf_counter() - t0
    n_valid = sum(t is not None for t in trajs)
    print(f"[probe] generated {n_valid}/{len(specs)} trajectories in "
          f"{elapsed:.1f}s", flush=True)
    return specs, trajs


def to_detector_space(raw):
    """Reproduce training/train_events_polar_dm.py:detector_features' output
    scale from features.py's RAW extract_features vector, in FEATURE_NAMES
    order. The checkpoint's feat_bank / feat_mu / feat_sd were built by
    z-scoring detector_features output (which log1p/rescales several dims),
    NOT raw feature units -- z-scoring raw values against feat_mu/feat_sd
    directly saturates the +/-10 clamp on every log-transformed dimension.
    Column order: mean_v, std_v, max_v, skew_v, mean_acc, std_acc, max_acc,
    mean_jerk, std_jerk, path_eff, max_dev, curv_mean, curv_std,
    n_dir_changes, duration, time_to_peak, omega_mean, omega_std."""
    import numpy as np
    r = np.asarray(raw, dtype=np.float64)

    def lg(x):
        return np.log1p(np.clip(x, 0.0, None))

    out = np.empty_like(r)
    out[:, 0] = lg(r[:, 0])                  # mean_velocity
    out[:, 1] = lg(r[:, 1])                  # std_velocity
    out[:, 2] = lg(r[:, 2])                  # max_velocity
    out[:, 3] = np.clip(r[:, 3], -8.0, 8.0)   # velocity_skewness
    out[:, 4] = r[:, 4] / 1e4                 # mean_acceleration
    out[:, 5] = lg(r[:, 5])                  # std_acceleration
    out[:, 6] = lg(r[:, 6])                  # max_acceleration
    out[:, 7] = r[:, 7] / 1e6                 # mean_jerk
    out[:, 8] = lg(r[:, 8])                  # std_jerk
    out[:, 9] = r[:, 9]                       # path_efficiency
    out[:, 10] = lg(r[:, 10])                 # max_deviation
    out[:, 11] = lg(r[:, 11] * 1e3)           # curvature_mean
    out[:, 12] = lg(r[:, 12] * 1e3)           # curvature_std
    out[:, 13] = lg(r[:, 13])                 # num_direction_changes
    out[:, 14] = lg(r[:, 14] * 10.0)          # movement_duration
    out[:, 15] = r[:, 15]                     # time_to_peak_velocity
    out[:, 16] = lg(r[:, 16])                 # angular_velocity_mean
    out[:, 17] = lg(r[:, 17])                 # angular_velocity_std
    return out


def realized_features(trajs, feat_mu, feat_sd):
    """features.py extraction (resample-first, per protocol) -> (valid_idx,
    X_raw, X_z). X_z is z-scored in the SAME transformed space the
    checkpoint's bank/feat_mu/feat_sd were built in (see to_detector_space)."""
    import numpy as np
    from features import extract_features, resample_trajectory
    valid_idx, raw = [], []
    for i, t in enumerate(trajs):
        if t is None or len(t) < 2:
            continue
        f = extract_features(resample_trajectory(t))
        if f is not None and np.all(np.isfinite(f)):
            valid_idx.append(i)
            raw.append(f)
    X_raw = np.asarray(raw)
    X_transformed = to_detector_space(X_raw)
    X_z = np.clip((X_transformed - feat_mu) / feat_sd, -10.0, 10.0)
    return np.asarray(valid_idx), X_raw, X_z


def run_auc(X_synth_raw, X_human_raw, seed=SEED):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    n = min(len(X_synth_raw), len(X_human_raw))
    X = np.vstack([X_human_raw[:n], X_synth_raw[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                 random_state=seed)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.oob_decision_function_[:, 1])), n


def variance_check(X_z, human_z, feature_names):
    import numpy as np
    synth_std = X_z.std(0)
    human_std = human_z.std(0)
    ratio = synth_std / np.clip(human_std, 1e-6, None)
    rows = []
    for i, name in enumerate(feature_names):
        flag = ratio[i] < 0.85
        rows.append((name, float(synth_std[i]), float(human_std[i]),
                     float(ratio[i]), flag))
    return rows


def print_variance_table(rows):
    print("\n--- Variance check: realized std vs val-human std (z-scored) ---")
    for name, s_std, h_std, ratio, flag in rows:
        mark = "  *** COLLAPSED >15% ***" if flag else ""
        print(f"  {name:26s} synth={s_std:.3f}  human={h_std:.3f}  "
              f"ratio={ratio:.3f}{mark}")


def stage_measure(args):
    from features import FEATURE_NAMES
    import numpy as np

    feat_mu, feat_sd = load_feat_stats()
    feat_log_path = ART_DIR / "measure_feat_log.npz"
    specs, trajs = generate({"EVENT_FEAT_LOG": str(feat_log_path)},
                            args.n, args.seed)

    valid_idx, X_raw, X_z = realized_features(trajs, feat_mu, feat_sd)
    d = np.load(feat_log_path)
    commanded_all, log_dist_all = d["commanded"], d["log_dist"]
    assert commanded_all.shape[0] == len(specs), (
        f"feat log length {commanded_all.shape[0]} != spec count {len(specs)} "
        "(some specs had zero distance or a candidate-selection path was "
        "active); this probe assumes one commanded draw per spec")
    commanded = commanded_all[valid_idx]
    log_dist = log_dist_all[valid_idx]

    # --- realization bias ---------------------------------------------
    bias = (X_z - commanded).mean(axis=0)
    realized_std = X_z.std(axis=0)
    commanded_std = commanded.std(axis=0)
    r = np.array([np.corrcoef(X_z[:, i], commanded[:, i])[0, 1]
                 for i in range(18)])

    order = np.argsort(-np.abs(bias))
    print("\n--- Realization bias (realized_z - commanded_z), sorted by |bias| ---")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:26s} bias={bias[i]:+.3f}  "
              f"realized_std={realized_std[i]:.3f}  "
              f"commanded_std={commanded_std[i]:.3f}  r={r[i]:+.3f}")

    # --- forward map: realized ~ [commanded, log_dist] ------------------
    from sklearn.linear_model import Ridge
    X_design = np.hstack([commanded, log_dist.reshape(-1, 1)])
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_design, X_z)
    A = ridge.coef_[:, :18]
    w = ridge.coef_[:, 18]
    b = ridge.intercept_
    pred = ridge.predict(X_design)
    ss_res = ((X_z - pred) ** 2).sum(axis=0)
    ss_tot = ((X_z - X_z.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-9, None)

    print("\n--- Forward fit realized ~ A*commanded + w*log_dist + b ---")
    for i in range(18):
        print(f"  {FEATURE_NAMES[i]:26s} R2={r2[i]:+.3f}  w={w[i]:+.4f}  "
              f"b={b[i]:+.3f}  diag(A)={A[i, i]:+.3f}")
    print(f"\n  mean R2 = {r2.mean():.3f}  |  max |w| = {np.abs(w).max():.4f}  "
          f"mean log_dist = {log_dist.mean():.3f}")

    if r2.mean() < 0.10:
        print("\n  *** FORWARD FIT IS NEAR-ZERO: the model barely listens to "
              "its own conditioning. The conditioning pathway is too weak to "
              "steer -- this alone would kill the correction lever. ***")

    # --- baseline AUC vs val humans -------------------------------------
    human_raw = np.load(REPO / "data" / "human_val_features_grpo.npy")
    human_z = np.clip((to_detector_space(human_raw) - feat_mu) / feat_sd, -10.0, 10.0)
    auc, n_used = run_auc(X_raw, human_raw, seed=args.seed)
    print(f"\n--- Baseline (uncorrected) AUC vs val humans: {auc:.4f} "
          f"(n={n_used} per class) ---")

    var_rows = variance_check(X_z, human_z, FEATURE_NAMES)
    print_variance_table(var_rows)

    # --- build corrections ----------------------------------------------
    absorb_distance = np.abs(w).max() >= W_ABSORB_THRESHOLD
    d_used = float(log_dist.mean()) if absorb_distance else 0.0
    print(f"\n  distance term: max|w|={np.abs(w).max():.4f} "
          f"{'>=' if absorb_distance else '<'} {W_ABSORB_THRESHOLD} -> "
          f"{'absorbing mean log_dist' if absorb_distance else 'ignoring log_dist'} "
          f"in the correction (d_used={d_used:.3f})")

    Ainv = np.linalg.solve(A.T @ A + CORR_EPS * np.eye(18), A.T)
    eye = np.eye(18)
    for lam, tag in [(1.0, "l1"), (0.5, "l05")]:
        M = (1 - lam) * eye + lam * Ainv
        v = -lam * ((w * d_used + b) @ Ainv.T)
        out_path = ART_DIR / f"corr_{tag}.npz"
        np.savez(out_path, M=M.astype("float32"), v=v.astype("float32"))
        print(f"  wrote correction lambda={lam} -> {out_path}")

    summary = {
        "bias": bias.tolist(), "realized_std": realized_std.tolist(),
        "commanded_std": commanded_std.tolist(), "r": r.tolist(),
        "r2": r2.tolist(), "w": w.tolist(), "b": b.tolist(),
        "auc_uncorrected": auc, "n_used": n_used,
        "absorb_distance": bool(absorb_distance), "d_used": d_used,
        "feature_names": FEATURE_NAMES,
    }
    (ART_DIR / "measure_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {ART_DIR / 'measure_summary.json'}")


def stage_corrected(args):
    from features import FEATURE_NAMES
    import numpy as np

    feat_mu, feat_sd = load_feat_stats()
    tag = "l1" if args.lam == 1.0 else "l05"
    corr_path = ART_DIR / f"corr_{tag}.npz"
    if not corr_path.exists():
        print(f"ERROR: {corr_path} missing -- run --stage measure first",
              file=sys.stderr)
        sys.exit(1)
    feat_log_path = ART_DIR / f"corrected_{tag}_feat_log.npz"
    specs, trajs = generate(
        {"EVENT_FEAT_CORR": str(corr_path), "EVENT_FEAT_LOG": str(feat_log_path)},
        args.n, args.seed)

    valid_idx, X_raw, X_z = realized_features(trajs, feat_mu, feat_sd)
    d = np.load(feat_log_path)
    commanded_all, log_dist_all = d["commanded"], d["log_dist"]
    assert commanded_all.shape[0] == len(specs)
    human_target = commanded_all[valid_idx]  # pristine KDE draw h, pre-correction

    bias = (X_z - human_target).mean(axis=0)
    realized_std = X_z.std(axis=0)
    order = np.argsort(-np.abs(bias))
    print(f"\n--- Corrected (lambda={args.lam}) bias vs human target ---")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:26s} bias={bias[i]:+.3f}  "
              f"realized_std={realized_std[i]:.3f}")

    human_raw = np.load(REPO / "data" / "human_val_features_grpo.npy")
    human_z = np.clip((to_detector_space(human_raw) - feat_mu) / feat_sd, -10.0, 10.0)
    auc, n_used = run_auc(X_raw, human_raw, seed=args.seed)
    print(f"\n--- Corrected (lambda={args.lam}) AUC vs val humans: {auc:.4f} "
          f"(n={n_used} per class) ---")

    var_rows = variance_check(X_z, human_z, FEATURE_NAMES)
    print_variance_table(var_rows)

    summary = {
        "lam": args.lam, "bias": bias.tolist(), "realized_std": realized_std.tolist(),
        "auc": auc, "n_used": n_used, "feature_names": FEATURE_NAMES,
    }
    out = ART_DIR / f"corrected_{tag}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {out}")


def main():
    args = parse_args()
    if args.stage == "measure":
        stage_measure(args)
    else:
        stage_corrected(args)


if __name__ == "__main__":
    main()
