"""Phase 1 scoring: did fooling the frozen Phase 0b critic (research/
phase0b_critic.py, checkpoint research/phase0b_critic.pt) transfer to the
REAL detector?

Measures, on a FRESH N=2000 sample of generated trajectories from
training/candi_polar_flow_phase1.pt (the Phase 1 fine-tune, step 617):
  (a) the real RF-OOB AUC vs data/human_val_features_grpo.npy, using the
      EXACT scoring path that produced the natural 0.757 baseline for
      candi_polar_flow_best.pt (research/phase_a_gpu_run3_noround.log):
      research/phase_a_baseline.py's generation config (steps=200,
      guide=0.15, perp=0.85, correct=rotate, no_round=True) and RF recipe
      (RandomForestClassifier(n_estimators=100, oob_score=True,
      random_state=42), OOB decision function AUC).
  (b) the frozen Phase 0b critic's AUC on the SAME fresh sample vs the SAME
      2000 human paths the critic was trained/evaluated on
      (research/phase0_critic_data.npz, y==0 rows) -- i.e. was the critic
      actually fooled out of sample, or only in-loop during training.

Reused verbatim, never reimplemented:
  - research/phase_a_baseline.py: load_model, generate_paths, make_specs,
    decode_polar, build_trajectory, sample_guided_flow (the published
    0.752/0.757 generation convention). --ckpt selects the checkpoint, so
    this same path also produces the same-day paired baseline for
    candi_polar_flow_best.pt.
  - features.py: extract_feature_matrix (RF input features).
  - research/phase0_critic.py: paths_to_dxdy (dx,dy = np.diff(path, axis=0),
    left-padded to MAX_LEN=256) -- the exact per-step raw-delta construction
    that feeds the critic's channel pipeline.
  - research/phase0b_critic.py: GeoPathCritic, build_all_channels (per-step
    geometric channel construction: dx,dy,speed,acc,jerk,curvature,
    angular_velocity under the constant dt=1/125 assumption).
  - research/phase1_channel_stats.npz: the FROZEN median/IQR
    standardization stats training/train_candi_phase1.py trained against
    (never refit on this new sample -- refitting would silently change the
    input space the frozen critic sees).
  - research/phase0_critic_data.npz: the original 2000 human dxdy/pad_mask
    rows (y==0), reused as the human reference for the critic-AUC
    measurement so the comparison is apples-to-apples with Phase 0b's own
    numbers.

Safety:
  - Never reads data/human_eval_features.npy.
  - Only reads (torch.load) checkpoints; never writes to
    training/candi_polar_flow_best.pt or training/candi_polar_flow_phase1.pt.
  - Generation is short (a few minutes of GPU inference, no backward pass),
    not sustained training; nvidia-smi temperature is checked before/after.

Usage:
    .venv/Scripts/python.exe research/phase1_score.py \\
        --ckpt candi_polar_flow_phase1.pt --n 2000 --tag phase1

    .venv/Scripts/python.exe research/phase1_score.py \\
        --ckpt candi_polar_flow_best.pt --n 2000 --tag base --skip-critic
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments._common import DurationModel, get_device  # noqa: E402
from features import extract_feature_matrix  # noqa: E402
from phase_a_baseline import (  # noqa: E402
    load_model, generate_paths, make_specs, DUR_STD,
)
from phase0_critic import paths_to_dxdy, MAX_LEN  # noqa: E402
from phase0b_critic import GeoPathCritic, build_all_channels  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
TRAIN_DIR = REPO_ROOT / "training"
RESEARCH_DIR = REPO_ROOT / "research"

RF_SEED = 42
CRITIC_CKPT_PATH = RESEARCH_DIR / "phase0b_critic.pt"
STATS_PATH = RESEARCH_DIR / "phase1_channel_stats.npz"
CRITIC_DATA_PATH = RESEARCH_DIR / "phase0_critic_data.npz"


def gpu_temp() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<nvidia-smi failed: {exc}>"


def generate_fresh_sample(ckpt_name: str, n: int, seed: int):
    print(f"[phase1_score] loading {ckpt_name} ...", flush=True)
    model, data_scale, device, max_seq_len_cfg = load_model(ckpt_name)
    model.max_seq_len_cfg = max_seq_len_cfg
    duration_model = DurationModel(TRAIN_DIR, std_mult=DUR_STD)
    specs = make_specs(n, seed)

    print(f"[phase1_score] GPU temp before generation: {gpu_temp()}", flush=True)
    t0 = time.perf_counter()
    trajectories = generate_paths(model, data_scale, device, duration_model,
                                   specs, no_round=True)
    trajectories = [t for t in trajectories if t is not None and len(t) >= 2]
    elapsed = time.perf_counter() - t0
    print(f"[phase1_score] generated {len(trajectories)}/{n} trajectories in "
          f"{elapsed:.1f}s (no_round=True)", flush=True)
    print(f"[phase1_score] GPU temp after generation: {gpu_temp()}", flush=True)
    return trajectories, elapsed


def rf_oob_auc(trajectories, out_features_path: Path):
    synth_features = extract_feature_matrix(trajectories)
    np.save(out_features_path, synth_features)
    print(f"[phase1_score] saved synthetic feature matrix to "
          f"{out_features_path} shape={synth_features.shape}", flush=True)

    human_features = np.load(DATA_DIR / "human_val_features_grpo.npy")
    n_use = min(len(human_features), len(synth_features))
    human_bal = human_features[:n_use]
    synth_bal = synth_features[:n_use]
    print(f"[phase1_score] RF-OOB: N used for scoring: {n_use} per class "
          f"(synth valid {len(synth_features)}/{len(trajectories)})", flush=True)

    X = np.vstack([human_bal, synth_bal])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                  random_state=RF_SEED)
    clf.fit(X, y)
    oob_proba = clf.oob_decision_function_[:, 1]
    auc = roc_auc_score(y, oob_proba)
    return float(auc), n_use, len(synth_features)


def load_frozen_critic(device):
    ckpt = torch.load(CRITIC_CKPT_PATH, map_location=device, weights_only=False)
    critic = GeoPathCritic(
        n_channels=len(ckpt["channel_names"]), d_model=ckpt["d_model"],
        n_layers=ckpt["n_layers"], n_head=ckpt["n_head"], d_ff=ckpt["d_ff"],
        dropout=ckpt["dropout"], max_len=ckpt["max_len"],
    ).to(device)
    critic.load_state_dict(ckpt["model_state_dict"])
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    return critic


def critic_score_channels(dxdy: np.ndarray, pad_mask: np.ndarray, scales: np.ndarray,
                           critic: nn.Module, device: torch.device) -> np.ndarray:
    """dxdy/pad_mask -> geometric channels (research/phase0b_critic.py's own
    build_all_channels, numpy) -> FROZEN median/IQR + signed-log
    standardization (frozen scales, never refit) -> critic forward. Returns
    sigmoid probabilities (1 == predicted synthetic)."""
    channels, n_nonfinite, n_valid_elems = build_all_channels(dxdy, pad_mask)
    if n_nonfinite:
        print(f"[phase1_score] WARNING: {n_nonfinite}/{n_valid_elems} "
              f"non-finite channel values clamped", flush=True)
    med = scales[:, 0]
    iqr = scales[:, 1]
    z = (channels - med) / iqr
    standardized = np.sign(z) * np.log1p(np.abs(z))
    standardized = standardized * pad_mask[:, :, None]

    logits = np.zeros(len(dxdy), dtype=np.float64)
    batch = 256
    with torch.no_grad():
        for b0 in range(0, len(dxdy), batch):
            x = torch.from_numpy(standardized[b0:b0 + batch]).float().to(device)
            m = torch.from_numpy(pad_mask[b0:b0 + batch]).to(device)
            logits[b0:b0 + batch] = critic(x, m).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs


def critic_auc(trajectories, device):
    print("[phase1_score] loading frozen critic + frozen channel stats...",
          flush=True)
    critic = load_frozen_critic(device)
    d = np.load(STATS_PATH, allow_pickle=True)
    scales = d["scales"]

    synth_dxdy, synth_pad_mask, synth_lengths, n_truncated = paths_to_dxdy(
        trajectories, max_len=MAX_LEN)
    if n_truncated:
        print(f"[phase1_score] WARNING: {n_truncated} fresh synth paths "
              f"truncated to MAX_LEN={MAX_LEN}", flush=True)
    synth_probs = critic_score_channels(synth_dxdy, synth_pad_mask, scales,
                                         critic, device)

    dcache = np.load(CRITIC_DATA_PATH, allow_pickle=True)
    y_cache = dcache["y"]
    human_idx = np.flatnonzero(y_cache == 0)
    human_dxdy = dcache["dxdy"][human_idx]
    human_pad_mask = dcache["pad_mask"][human_idx]
    print(f"[phase1_score] critic AUC: reusing {len(human_idx)} cached human "
          f"paths from {CRITIC_DATA_PATH.name} (y==0) as the human reference",
          flush=True)
    human_probs = critic_score_channels(human_dxdy, human_pad_mask, scales,
                                         critic, device)

    n_use = min(len(human_probs), len(synth_probs))
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    probs = np.concatenate([human_probs[:n_use], synth_probs[:n_use]])
    auc = roc_auc_score(y, probs)
    return float(auc), n_use, len(synth_probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="candi_polar_flow_phase1.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="phase1",
                     help="Prefix for output files (research/phase1_score_<tag>_*)")
    ap.add_argument("--skip-critic", action="store_true",
                     help="Skip the frozen-critic AUC (e.g. for the base-model "
                          "paired control, which the critic was never meant "
                          "to score meaningfully either way)")
    args = ap.parse_args()

    device = get_device()
    print(f"[phase1_score] device={device} ckpt={args.ckpt} n={args.n} "
          f"seed={args.seed} tag={args.tag}", flush=True)

    trajectories, gen_elapsed = generate_fresh_sample(args.ckpt, args.n, args.seed)

    traj_path = RESEARCH_DIR / f"phase1_score_{args.tag}_trajectories.pkl"
    with open(traj_path, "wb") as fh:
        pickle.dump(trajectories, fh)
    print(f"[phase1_score] pickled {len(trajectories)} raw trajectories to "
          f"{traj_path}", flush=True)

    feat_path = RESEARCH_DIR / f"phase1_score_{args.tag}_features.npy"
    rf_auc, rf_n_use, rf_n_valid = rf_oob_auc(trajectories, feat_path)
    print(f"[phase1_score] === RF-OOB AUC ({args.tag}): {rf_auc:.4f} "
          f"(N={rf_n_use} per class) ===", flush=True)

    results = {
        "ckpt": args.ckpt,
        "tag": args.tag,
        "n_requested": args.n,
        "n_valid_trajectories": rf_n_valid,
        "seed": args.seed,
        "generation_elapsed_sec": gen_elapsed,
        "rf_oob_auc": rf_auc,
        "rf_oob_n_per_class": rf_n_use,
        "rf_oob_reference_baseline_best_pt_noround": 0.7573,
        "human_reference_file": "data/human_val_features_grpo.npy",
        "pipeline": "research/phase_a_baseline.py generate_paths/RF-OOB recipe "
                    "(steps=200, guide=0.15, perp=0.85, correct=rotate, "
                    "no_round=True); RandomForestClassifier(n_estimators=100, "
                    "oob_score=True, random_state=42)",
    }

    if not args.skip_critic:
        c_auc, c_n_use, c_n_valid = critic_auc(trajectories, device)
        print(f"[phase1_score] === Frozen Phase0b critic AUC ({args.tag}): "
              f"{c_auc:.4f} (N={c_n_use} per class) ===", flush=True)
        results.update({
            "critic_auc": c_auc,
            "critic_n_per_class": c_n_use,
            "critic_ckpt": str(CRITIC_CKPT_PATH),
            "critic_stats_path": str(STATS_PATH),
            "critic_human_reference": f"{CRITIC_DATA_PATH.name} (y==0 rows, "
                                       "same 2000 human paths used to "
                                       "train/evaluate the critic)",
            "critic_phase0b_oof_reference": 0.632,
            "critic_note": "phase0b_critic_oof_auc=0.632 is the critic's OWN "
                            "5-fold OOF AUC on its ORIGINAL 2000v2000 sample "
                            "(base-model synth); this run's critic_auc is a "
                            "fresh out-of-sample measurement on Phase 1's "
                            "post-fine-tune output, so lower means the fool "
                            "loss transferred to the critic itself.",
        })

    out_json = RESEARCH_DIR / f"phase1_score_{args.tag}_results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[phase1_score] wrote {out_json}", flush=True)
    print(f"[phase1_score] GPU temp at end: {gpu_temp()}", flush=True)


if __name__ == "__main__":
    main()
