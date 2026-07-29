"""P2 gate evaluation: character latent from the learned prior.

Pre-registered success criteria (W3_PROPOSAL.md, adapted after P1's FAIL):
one-shot exact-arrival AUC materially below the base model's 0.728, with
the spread of the tell features (max velocity, max acceleration, jerk)
moving toward human levels rather than past them.

Generates N one-shot paths with a fresh z drawn from the checkpoint's
conditional prior per path, reports miss stats, raw and corrected AUC, and
the synthetic-to-human standard deviation ratio of the tell features.
Scoring goes through research/autoloop/scoring.py only.

Usage:
    env PYTHONPATH=. ~/venvs/mime/bin/python research/w3_p2_eval.py \
        --ckpt event_polar_4m_char_v1.pt --n 2000 --seed 42 --feat zero
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

TELL = {"max_velocity": 2, "max_acceleration": 6,
        "mean_jerk": 7, "std_jerk": 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--feat", choices=("bank", "zero"), default="zero",
                    help="bank keeps the KDE feature command alongside z; "
                         "zero serves on the learned character alone")
    ap.add_argument("--z-temp", type=float, default=1.0)
    ap.add_argument("--cfg-w", type=float, default=0.0,
                    help="classifier-free guidance weight on the character "
                         "channels; 0 off, 1 plain conditional")
    ap.add_argument("--z-off", action="store_true",
                    help="ablation: no prior draw, resid slot unused")
    args = ap.parse_args()

    # locked one-shot decode recipe, plus the character prior
    os.environ.update(EVENT_CKPT=args.ckpt, EVENT_ORDER="gumbel",
                      EVENT_SNAP="2.5", EVENT_DUR_STD="1.0", DUR_EMPIRICAL="1",
                      EVENT_CHOICE_TEMP="10", EVENT_BESTOF="1", EVENT_SIR="1",
                      EVENT_Z_TEMP=str(args.z_temp),
                      EVENT_CFG_W=str(args.cfg_w),
                      EVENT_FEAT="1" if args.feat == "bank" else "0")
    if not args.z_off:
        os.environ["EVENT_Z_PRIOR"] = "1"
    for k in ("EVENT_POOL_TOKENS", "EVENT_POOL_SAVE", "EVENT_POOL_LOAD"):
        os.environ.pop(k, None)

    import scoring
    from features import extract_feature_matrix
    from phase_a_baseline import make_specs
    from experiments import event_stream_polar
    from w3_fallback_arrival import correct_additive

    specs = [(round(a), round(b), round(c), round(d))
             for a, b, c, d in make_specs(args.n, args.seed)]
    t0 = time.time()
    raw = event_stream_polar.generate_paths(specs)
    print(f"[gen] {len(raw)} paths in {time.time()-t0:.0f}s", flush=True)

    pairs = [(s, t) for s, t in zip(specs, raw) if t is not None and len(t) >= 3]
    miss = np.array([math.hypot(t[-1][0] - s[2], t[-1][1] - s[3])
                     for s, t in pairs])
    travel = np.array([math.hypot(s[2] - s[0], s[3] - s[1]) for s, _ in pairs])
    print(f"[miss] median={np.median(miss):.1f}px mean={miss.mean():.1f} "
          f"p90={np.percentile(miss, 90):.1f} "
          f"within15px={100 * (miss <= 15).mean():.1f}%", flush=True)

    human = np.load(R / "data" / "human_val_features_grpo.npy")
    res = {}
    for mode in ("raw", "corrected"):
        if mode == "raw":
            trajs = [t for _, t in pairs]
        else:
            trajs = [correct_additive(np.asarray(t), *s) for s, t in pairs]
        feats = extract_feature_matrix(trajs)
        r = scoring.score_features(feats)
        spread = {name: float(feats[:, i].std() / human[:, i].std())
                  for name, i in TELL.items()}
        res[mode] = {"auc": float(r["auc_rf_oob"]),
                     "n_per_class": int(r["n_per_class"]),
                     "tell_std_ratio": spread}
        print(f"[auc] {mode:9s} {r['auc_rf_oob']:.4f} "
              f"(n={r['n_per_class']}/class) spread "
              + " ".join(f"{k}={v:.2f}" for k, v in spread.items()),
              flush=True)

    out = {"ckpt": args.ckpt, "n": args.n, "seed": args.seed,
           "feat": args.feat, "z_temp": args.z_temp,
           "z_off": bool(args.z_off), "cfg_w": args.cfg_w,
           "miss_median_px": float(np.median(miss)),
           "miss_p90_px": float(np.percentile(miss, 90)),
           "within_15px_pct": float(100 * (miss <= 15).mean()),
           "usable": len(pairs), **res}
    out_path = args.out or str(
        R / "research" / f"w3_p2_eval_{Path(args.ckpt).stem}_{args.feat}.json")
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
