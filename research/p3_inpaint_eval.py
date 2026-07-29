"""P3 probe eval: one-shot scoring of the inpainting flow model.

Generates one path per spec on the standing eval stream (make_specs,
n=2000, seed 42), no selection, no correction step (arrival is pinned by
construction), then scores with the frozen RF-OOB recipe. Also reports
the tell-feature spread ratios (synthetic std over human std) for the
gate registered in W3_PROPOSAL.md.

Never touches evaluate.py or any forbidden human feature file.

Usage:
  python research/p3_inpaint_eval.py --ckpt training/inpaint_flow_v1.pt
"""
from __future__ import annotations

import argparse
import json
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

import scoring
from features import FEATURE_NAMES, extract_feature_matrix
from models.inpaint_flow import InpaintFlowModel
from phase_a_baseline import make_specs
from experiments._common import DurationModel

N_SLOTS = 192
HZ = 125.0
MAX_DUR = (N_SLOTS - 1 - 1) / HZ  # 191 slots max, uniform 8 ms steps
TELL_FEATURES = ["max_velocity", "max_acceleration", "mean_jerk", "std_jerk"]


def build_batch(specs, durs, device):
    B = len(specs)
    cond = np.zeros((B, 4), dtype=np.float32)
    known_val = np.zeros((B, N_SLOTS, 2), dtype=np.float32)
    known_flag = np.zeros((B, N_SLOTS), dtype=np.float32)
    pad = np.zeros((B, N_SLOTS), dtype=bool)
    ns = np.zeros(B, dtype=int)
    for i, ((sx, sy, ex, ey), dur) in enumerate(zip(specs, durs)):
        dist = math.hypot(ex - sx, ey - sy)
        ca, sa = (ex - sx) / dist, (ey - sy) / dist
        n = int(np.clip(round(dur * HZ) + 1, 8, N_SLOTS - 1))
        cond[i] = [math.log(dist), math.log(dur), ca, sa]
        known_val[i, n - 1:] = [ca, sa]
        known_flag[i, 0] = 1.0
        known_flag[i, n - 1:] = 1.0
        pad[i, n:] = True
        ns[i] = n
    to = lambda a, dt: torch.from_numpy(a).to(device=device, dtype=dt)
    return (to(cond, torch.float32), to(known_val, torch.float32),
            to(known_flag, torch.float32), to(pad, torch.bool), ns)


def decode(xt, stall, ns, specs):
    """Apply stalls, denormalize to pixels, pin endpoints, add timestamps."""
    trajs = []
    for row, st, n, (sx, sy, ex, ey) in zip(xt, stall, ns, specs):
        p = row[:n].copy()
        for i in range(1, n - 1):
            if st[i] > 0.5:
                p[i] = p[i - 1]
        dist = math.hypot(ex - sx, ey - sy)
        px = np.round(np.array([sx, sy]) + p * dist)
        px[0] = [sx, sy]
        px[-1] = [ex, ey]
        t = np.arange(n) / HZ
        trajs.append(np.c_[px, t])
    return trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(R / "training" / "inpaint_flow_v1.pt"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flow-steps", type=int, default=64)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--out", default=str(R / "research" / "p3_eval_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InpaintFlowModel().to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"[p3eval] ckpt {args.ckpt} step {ck.get('step')} "
          f"val_loss {ck.get('val_loss')}", flush=True)

    specs = [tuple(np.round(s).astype(int)) for s in make_specs(args.n, args.seed)]
    dur_model = DurationModel(R / "training")
    torch.manual_seed(args.seed)

    n_clipped = 0
    trajs = []
    stall_used = []
    for lo in range(0, len(specs), args.batch):
        chunk = specs[lo:lo + args.batch]
        durs = []
        for (sx, sy, ex, ey) in chunk:
            d = dur_model.sample(math.log(math.hypot(ex - sx, ey - sy)))
            if d > MAX_DUR:
                n_clipped += 1
                d = MAX_DUR
            durs.append(d)
        cond, kv, kf, pad, ns = build_batch(chunk, durs, device)
        xt, st = model.flow_sample(cond, kv, kf, pad_mask=pad,
                                   n_steps=args.flow_steps, cfg_scale=args.cfg)
        xt = xt.cpu().numpy()
        st = st.cpu().numpy()
        for row, s_row, n in zip(xt, st, ns):
            stall_used.append(float(np.mean(s_row[1:n - 1] > 0.5)))
        trajs.extend(decode(xt, st, ns, chunk))
        print(f"[p3eval] generated {len(trajs)}/{len(specs)}", flush=True)

    X = extract_feature_matrix(trajs)
    print(f"[p3eval] features {X.shape} (dropped {len(trajs) - len(X)})", flush=True)
    res = scoring.score_features(X)

    human = np.load(R / "data" / "human_val_features_grpo.npy")
    spread = {}
    for name in TELL_FEATURES:
        j = FEATURE_NAMES.index(name)
        spread[name] = float(np.std(X[:, j]) / np.std(human[:, j]))

    out = {
        "ckpt": args.ckpt, "n": args.n, "seed": args.seed,
        "flow_steps": args.flow_steps, "cfg": args.cfg,
        "auc_rf_oob": res["auc_rf_oob"], "n_per_class": res["n_per_class"],
        "tell_spread": spread,
        "stall_frac_mean": float(np.mean(stall_used)),
        "dur_clip_frac": n_clipped / len(specs),
        "n_features_valid": int(len(X)),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[p3eval] AUC {res['auc_rf_oob']:.4f} spread " +
          " ".join(f"{k}={v:.2f}" for k, v in spread.items()) +
          f" stall_frac {out['stall_frac_mean']:.3f} "
          f"dur_clip {out['dur_clip_frac']:.3f}", flush=True)


if __name__ == "__main__":
    main()
