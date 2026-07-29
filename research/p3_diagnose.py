"""Diagnose the P3 probe result: which features separate, and why.

Regenerates a small sample with the trained inpainting model and compares
every feature against the human validation set, so the 0.98 AUC can be
attributed to specific failures rather than guessed at.

Usage:
  python research/p3_diagnose.py --ckpt training/inpaint_flow_v1.pt --n 500
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

os.environ.setdefault("DUR_EMPIRICAL", "1")

from features import FEATURE_NAMES, extract_feature_matrix
from models.inpaint_flow import InpaintFlowModel
from phase_a_baseline import make_specs
from experiments._common import DurationModel
from p3_inpaint_eval import build_batch, decode, MAX_DUR, HZ, N_SLOTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(R / "training" / "inpaint_flow_v1.pt"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flow-steps", type=int, default=64)
    ap.add_argument("--cfg", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InpaintFlowModel().to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    specs = [tuple(np.round(s).astype(int)) for s in make_specs(args.n, args.seed)]
    dur_model = DurationModel(R / "training")
    torch.manual_seed(args.seed)

    durs = []
    for (sx, sy, ex, ey) in specs:
        d = dur_model.sample(math.log(math.hypot(ex - sx, ey - sy)))
        durs.append(min(d, MAX_DUR))
    cond, kv, kf, pad, ns = build_batch(specs, durs, device)

    raw_stall_prob = []
    trajs = []
    for lo in range(0, len(specs), 250):
        sl_ = slice(lo, lo + 250)
        xt, st = model.flow_sample(cond[sl_], kv[sl_], kf[sl_],
                                   pad_mask=pad[sl_], n_steps=args.flow_steps,
                                   cfg_scale=args.cfg)
        xt = xt.cpu().numpy()
        st = st.cpu().numpy()
        trajs.extend(decode(xt, st, ns[sl_], specs[sl_]))

    # How often does the decoded pixel path actually repeat a pixel?
    rep = [float(np.mean(np.all(np.diff(t[:, :2], axis=0) == 0, axis=1)))
           for t in trajs]

    X = extract_feature_matrix(trajs)
    human = np.load(R / "data" / "human_val_features_grpo.npy")

    print(f"[diag] n={len(X)} human={len(human)}")
    print(f"[diag] decoded pixel-repeat frac: mean {np.mean(rep):.3f} "
          f"median {np.median(rep):.3f}")
    print()
    print(f"{'feature':<24}{'syn_mean':>11}{'hum_mean':>11}"
          f"{'syn_std':>10}{'hum_std':>10}{'std_ratio':>10}{'z_shift':>9}")
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        sm, hm = float(np.mean(X[:, j])), float(np.mean(human[:, j]))
        ss, hs = float(np.std(X[:, j])), float(np.std(human[:, j]))
        z = (sm - hm) / hs if hs > 0 else 0.0
        rows.append((abs(z), name, sm, hm, ss, hs, ss / hs if hs > 0 else 0.0, z))
    for _, name, sm, hm, ss, hs, r, z in sorted(rows, reverse=True):
        print(f"{name:<24}{sm:>11.3f}{hm:>11.3f}{ss:>10.3f}{hs:>10.3f}"
              f"{r:>10.2f}{z:>9.2f}")


if __name__ == "__main__":
    main()
