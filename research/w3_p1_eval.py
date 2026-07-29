"""P1 gate evaluation: native miss and one-shot AUC, raw and with arrival.

Pre-registered success criteria (W3_PROPOSAL.md): median native miss 15px or
less, AND one-shot AUC with exact arrival enforced materially below 0.728
(the base model's corrected number from the landing-price run).

Generates N one-shot paths at the locked recipe for the given checkpoint,
then reports: native miss stats, raw AUC (comparability with the 0.6544
baseline), and corrected AUC using the same magnitude-weighted additive
correction as every arrival number this week. Scoring goes through
research/autoloop/scoring.py only.

Usage:
    env PYTHONPATH=. ~/venvs/mime/bin/python research/w3_p1_eval.py \
        --ckpt event_polar_4m_resid_v1.pt --n 2000 --seed 42
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--order", default="gumbel",
                    help="reveal order; l2r for prefix-grow closed-loop ckpts")
    ap.add_argument("--resid-only", action="store_true",
                    help="withhold static displacement cond, steer on resid")
    ap.add_argument("--resid-mode", default="prefix",
                    help="prefix (partial-state) or draft (endpoint error "
                         "of the model's own provisional draft)")
    args = ap.parse_args()

    # locked one-shot decode recipe (see HANDOFF_W3.md, Evaluation section)
    os.environ.update(EVENT_CKPT=args.ckpt, EVENT_ORDER=args.order,
                      EVENT_RESID_MODE=args.resid_mode,
                      EVENT_SNAP="2.5", EVENT_DUR_STD="1.0", DUR_EMPIRICAL="1",
                      EVENT_CHOICE_TEMP="10", EVENT_BESTOF="1", EVENT_SIR="1")
    if args.resid_only:
        os.environ["EVENT_RESID_ONLY"] = "1"
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
          f"pct_travel_median={np.median(100 * miss / travel):.1f}% "
          f"within15px={100 * (miss <= 15).mean():.1f}%", flush=True)

    res = {}
    for mode in ("raw", "corrected"):
        if mode == "raw":
            trajs = [t for _, t in pairs]
        else:
            trajs = [correct_additive(np.asarray(t), *s) for s, t in pairs]
        feats = extract_feature_matrix(trajs)
        r = scoring.score_features(feats)
        res[mode] = {"auc": float(r["auc_rf_oob"]),
                     "n_per_class": int(r["n_per_class"])}
        print(f"[auc] {mode:9s} {r['auc_rf_oob']:.4f} "
              f"(n={r['n_per_class']}/class)", flush=True)

    out = {"ckpt": args.ckpt, "n": args.n, "seed": args.seed,
           "order": args.order, "resid_only": bool(args.resid_only),
           "resid_mode": args.resid_mode,
           "miss_median_px": float(np.median(miss)),
           "miss_mean_px": float(miss.mean()),
           "miss_p90_px": float(np.percentile(miss, 90)),
           "miss_pct_travel_median": float(np.median(100 * miss / travel)),
           "within_15px_pct": float(100 * (miss <= 15).mean()),
           "usable": len(pairs), **res}
    out_path = args.out or str(
        R / "research" / f"w3_p1_eval_{Path(args.ckpt).stem}.json")
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
