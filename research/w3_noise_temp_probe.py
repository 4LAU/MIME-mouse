"""W3 groundwork: does initial-noise temperature fix conditional under-dispersion?

The tail-ceiling probe (w3_tail_ceiling_results.json) showed the human feature
tails are inside the generator's support but drawn too rarely. Cheapest
possible density lever: inflate the flow sampler's initial noise (z_scale > 1)
at pure sampling time - no training, serving-compatible, zero latency cost.

Per z_scale in --scales: generate N unguided paths (standard make_specs
distribution), compute the 18 real features, report (a) RF-OOB AUC vs
data/human_val_features_grpo.npy at N per class (the single project metric),
(b) per-feature raw dispersion ratios for the W2-diagnosed tell features.
z_scale=1.0 is the control arm and should reproduce ~0.757.

Usage:
    python research/w3_noise_temp_probe.py --n 2000 --scales 1.0,1.15,1.3,1.5 \
        --pid-file research/w3_temp.pid.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.phase_a_baseline import (  # noqa: E402
    load_model, generate_paths, make_specs, DUR_STD,
)
from experiments._common import DurationModel  # noqa: E402
from features import extract_features, FEATURE_NAMES  # noqa: E402

TRAIN_DIR = Path("training")
SRC_CKPT_NAME = "candi_polar_flow_best.pt"
EXPECTED_SRC_MD5 = "91326a29750789f3167055324ef377c5"
HUMAN_REF_PATH = Path("data/human_val_features_grpo.npy")
OUT_JSON = Path("research/w3_noise_temp_results.json")
RF_SEED = 42

TELL_FEATURES = ["max_velocity", "max_acceleration", "mean_jerk", "mean_acceleration",
                 "std_velocity", "curvature_mean", "curvature_std", "movement_duration"]


def md5_file(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rf_oob_auc(synth, human, seed=RF_SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    n_use = min(len(human), len(synth))
    X = np.vstack([human[:n_use], synth[:n_use]])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    return float(roc_auc_score(y, rf.oob_decision_function_[:, 1])), n_use


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--scales", type=str, default="1.0,1.15,1.3,1.5")
    ap.add_argument("--seed", type=int, default=888)
    ap.add_argument("--pid-file", type=str, default=None)
    args = ap.parse_args()

    if args.pid_file:
        import os
        with open(args.pid_file, "w") as fh:
            fh.write(str(os.getpid()))

    scales = [float(s) for s in args.scales.split(",")]
    md5_before = md5_file(TRAIN_DIR / SRC_CKPT_NAME)
    assert md5_before == EXPECTED_SRC_MD5, "source checkpoint MD5 mismatch -- STOP"

    model, data_scale, device, max_seq_len_cfg = load_model(SRC_CKPT_NAME)
    model.max_seq_len_cfg = max_seq_len_cfg
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    duration_model = DurationModel(TRAIN_DIR, std_mult=DUR_STD)

    human = np.load(HUMAN_REF_PATH)
    human = human[np.all(np.isfinite(human), axis=1)]
    specs = make_specs(args.n, args.seed)

    arms = {}
    for zs in scales:
        print(f"[w3_temp] z_scale={zs}: generating {args.n} paths...", flush=True)
        t0 = time.perf_counter()
        trajs = generate_paths(model, data_scale, device, duration_model, specs,
                               no_round=True, z_scale=zs)
        trajs = [t for t in trajs if t is not None and len(t) >= 5]
        rows = []
        for t in trajs:
            f = extract_features(t)
            if f is not None and np.all(np.isfinite(f)):
                rows.append(f)
        synth = np.stack(rows)
        auc, n_use = rf_oob_auc(synth, human)
        disp = {}
        for name in TELL_FEATURES:
            j = FEATURE_NAMES.index(name)
            disp[name] = float(np.std(synth[:, j]) / max(np.std(human[:, j]), 1e-9))
        arms[str(zs)] = {
            "n_usable": len(rows), "auc_rf_oob": auc, "n_per_class": n_use,
            "gen_seconds": time.perf_counter() - t0,
            "raw_dispersion_ratio": disp,
        }
        print(f"[w3_temp] z_scale={zs}: AUC={auc:.4f} n={n_use} "
              f"max_vel_disp={disp['max_velocity']:.3f} "
              f"max_acc_disp={disp['max_acceleration']:.3f}", flush=True)

    md5_after = md5_file(TRAIN_DIR / SRC_CKPT_NAME)
    out = {
        "status": "COMPLETE", "n": args.n, "seed": args.seed, "scales": scales,
        "md5_before": md5_before, "md5_after": md5_after,
        "md5_unchanged": md5_before == md5_after,
        "arms": arms,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[w3_temp] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
