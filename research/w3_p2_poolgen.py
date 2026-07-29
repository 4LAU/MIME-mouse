"""Regenerate the K=32 candidate pool with the P2 character generator.

Same spec stream as pool_s42_k32.npz (make_specs, n=2000, seed 42) but the
candidates come from the char_v3 checkpoint served with the KDE feature bank,
the learned character prior, and classifier-free guidance at weight 2. The
pool feeds research/w3_fallback_arrival.py, replacing the base-model pool
that produced the 0.58 corr_corr product number.

Generation is driven directly through generate_paths, not evaluate.py, so no
holdout data is ever loaded here.

Usage:
    env PYTHONPATH=. ~/venvs/mime/bin/python research/w3_p2_poolgen.py \
        --ckpt event_polar_4m_char_v3.pt --out pool_char_v3_cfg2_s42_k32.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg-w", type=float, default=2.0)
    ap.add_argument("--sir-k", type=int, default=32)
    args = ap.parse_args()

    os.environ.update(
        EVENT_CKPT=args.ckpt, EVENT_ORDER="gumbel", EVENT_SNAP="2.5",
        EVENT_DUR_STD="1.0", DUR_EMPIRICAL="1", EVENT_CHOICE_TEMP="10",
        EVENT_FEAT="1", EVENT_Z_PRIOR="1", EVENT_CFG_W=str(args.cfg_w),
        EVENT_BESTOF="1", EVENT_SIR=str(args.sir_k), EVENT_SIR_TEMP="0.7",
        EVENT_SIR_DUR_DIVERSE="1", EVENT_POOL_SAVE=args.out)
    for k in ("EVENT_POOL_TOKENS", "EVENT_POOL_LOAD", "EVENT_POOL_PICKS"):
        os.environ.pop(k, None)

    from phase_a_baseline import make_specs
    from experiments import event_stream_polar

    specs = [(round(a), round(b), round(c), round(d))
             for a, b, c, d in make_specs(args.n, args.seed)]
    t0 = time.time()
    paths = event_stream_polar.generate_paths(specs)
    n_ok = sum(1 for p in paths if p is not None and len(p) >= 3)
    print(f"[poolgen] {n_ok}/{len(specs)} winners decoded in "
          f"{time.time()-t0:.0f}s; pool at {args.out}", flush=True)


if __name__ == "__main__":
    main()
