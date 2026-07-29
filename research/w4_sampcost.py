"""What does one AR sample actually cost, and is the 8GB card the reason?

`w4_arcurve` needs 9 checkpoints scored, and the first one took over 30 minutes
at the default `--batch 500`, which puts a single curve at four and a half
hours. That measurement gets repeated after every scaling run, so the cost
compounds and is worth pricing once.

Two candidate causes, and they call for opposite fixes:

  memory        `EventARModel.sample` keeps no KV cache and re-runs the trunk
                over the prefix at every step, so attention materialises a
                B x heads x T x T matrix. That term is LINEAR in batch, and
                `w4_arbench` already established that this card silently spills
                to host memory instead of raising, turning a 5.7x parameter
                increase into a 44x slowdown. If this is it, a SMALLER batch is
                FASTER, which is the opposite of the usual intuition.
  launch bound  256 strictly sequential steps, each a small kernel. If this is
                it, a smaller batch is strictly worse and the only real fix is
                a KV cache, which is a change to generation itself and must not
                be made casually while it is the thing under measurement.

The two are told apart by the shape of traj/s against batch. Memory bound gives
a peak at some batch below the default with a cliff above it. Launch bound
gives a curve that rises monotonically with batch and flattens.

Timing only. This deliberately does not score anything, because a timing probe
that also touches the contract scorer invites reading a score off a run that
was not built to produce one.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop:training \
        ~/venvs/mime/bin/python research/w4_sampcost.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

# the locked serving recipe, set exactly as `w4_ar_eval` sets it so the cond
# vectors this times are the cond vectors the real evaluation generates from
for _k, _v in (("EVENT_CHOICE_TEMP", "10"), ("EVENT_SNAP", "2.5"),
               ("EVENT_DUR_STD", "1.0"), ("DUR_EMPIRICAL", "1"),
               ("EVENT_BESTOF", "1"), ("EVENT_SIR", "1"),
               ("EVENT_ORDER", "gumbel"), ("EVENT_STEPS", "100"),
               ("EVENT_CFG_W", "0")):
    os.environ.setdefault(_k, _v)

for p in (".", "research", "research/autoloop", "training"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from phase_a_baseline import make_specs  # noqa: E402


def cond_rows(n, seed):
    """The same cond vectors `w4_ar_eval` builds, so the timing is honest."""
    rows = []
    for sx, sy, ex, ey in make_specs(n, seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ld = math.log(dist)
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([ld, math.log(esp._duration.sample(ld)),
                     math.cos(ang), math.sin(ang)])
    return np.asarray(rows, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v1.pt")
    ap.add_argument("--batches", default="500,250,125,64,32")
    ap.add_argument("--n", type=int, default=250,
                    help="trajectories timed per batch size; one full batch is "
                         "used when the batch exceeds this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_sampcost.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    n_par = sum(p.numel() for p in model.parameters())
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20

    batches = [int(b) for b in args.batches.split(",")]
    rows = cond_rows(max(max(batches), args.n) + 64, args.seed)
    print(f"  {args.ckpt}, {n_par/1e6:.1f}M params, card {total:,.0f} MiB, "
          f"{len(rows)} cond rows available\n")
    print(f"  {'batch':>7}{'s/call':>9}{'traj/s':>9}{'alloc':>9}{'reserv':>9}"
          f"{'hrs/9ckpt@2000':>16}")

    out = []
    for B in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cond = torch.tensor(rows[:B], device=dev)
        try:
            t0 = time.time()
            model.sample(cond, temperature=1.0)
            torch.cuda.synchronize()
            sec = time.time() - t0
        except torch.cuda.OutOfMemoryError:
            print(f"  {B:>7}{'OOM':>9}")
            out.append(dict(batch=B, oom=True))
            torch.cuda.empty_cache()
            continue
        tps = B / sec
        rec = dict(batch=B, s_per_call=sec, traj_per_s=tps,
                   alloc_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                   reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
                   hours_9ckpt_2000=9 * 2000 / tps / 3600)
        out.append(rec)
        print(f"  {B:>7}{sec:>9.1f}{tps:>9.1f}{rec['alloc_mib']:>9.0f}"
              f"{rec['reserved_mib']:>9.0f}{rec['hours_9ckpt_2000']:>16.2f}",
              flush=True)
        del cond
        torch.cuda.empty_cache()

    json.dump(dict(ckpt=args.ckpt, params=int(n_par), card_mib=total,
                   rows=out), open(args.out, "w"), indent=2)
    print("\n  traj/s peaking below the largest batch means the card is the")
    print("  limit and a smaller batch is the whole fix. traj/s rising all the")
    print("  way to the largest batch means the cost is sequential decode and")
    print("  only a KV cache touches it.")


if __name__ == "__main__":
    main()
