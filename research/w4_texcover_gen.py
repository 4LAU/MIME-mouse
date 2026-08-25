"""Dump model rollout TOKEN STREAMS at the closed serving optimum.

Registered in /home/aaronadmin/w4_arms/texcover_prereg.md. Read that first.

This is the only GPU this arm spends. It generates trajectories exactly the way
`w4_ar_eval.py` does, at the same specs from the same `make_specs`, at the
optimum the three head registration closed (s 0.95, th 0.90, dt 1.00), and
saves the raw class streams plus the conditioning instead of scoring them.

`research/w4_typpos_streams.npz` is the earlier dump of this kind and RESUME
records it as STALE, predating the sampling clock fix, so it cannot be reused
and this file does not read it.

Everything downstream of this is CPU and lives in `research/w4_texcover.py`.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

# Identical to w4_ar_eval.py's block. The rollouts must come from the same
# serving configuration as every scored number on this checkpoint or the
# coverage question is being asked about a different sampler.
os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_SNAP", "2.5")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_BESTOF", "1")
os.environ.setdefault("EVENT_SIR", "1")
os.environ.setdefault("EVENT_ORDER", "gumbel")
os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from phase_a_baseline import make_specs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--s-temp", type=float, default=0.95)
    ap.add_argument("--th-temp", type=float, default=0.90)
    ap.add_argument("--dt-temp", type=float, default=1.00)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {args.ckpt} step {ck.get('step')}", flush=True)

    # Same spec construction as w4_ar_eval.py, including the duration draw,
    # so the conditioning population is identical.
    specs = make_specs(args.n, args.seed)
    rows = []
    for sx, sy, ex, ey in specs:
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ld = math.log(dist)
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([ld, math.log(esp._duration.sample(ld)),
                     math.cos(ang), math.sin(ang)])

    S, TH, DT, C = [], [], [], []
    for c0 in range(0, len(rows), args.batch):
        cond = torch.tensor(rows[c0:c0 + args.batch], dtype=torch.float32,
                            device=dev)
        s_cls, th_cls, dt_cls = model.sample(
            cond, temperature=args.s_temp, th_temperature=args.th_temp,
            dt_temperature=args.dt_temp)
        S.append(s_cls.cpu().numpy().astype(np.int16))
        TH.append(th_cls.cpu().numpy().astype(np.int16))
        DT.append(dt_cls.cpu().numpy().astype(np.int16))
        C.append(cond.cpu().numpy().astype(np.float32))
        if c0 % (args.batch * 20) == 0:
            print(f"    {c0}/{len(rows)}", flush=True)

    # sample() pads every batch to the same length, but different batches can
    # stop at different lengths, so pad to the widest before stacking.
    T = max(a.shape[1] for a in S)

    def pad_to(a, T, fill):
        if a.shape[1] == T:
            return a
        return np.pad(a, ((0, 0), (0, T - a.shape[1])), constant_values=fill)

    from models.event_ar import DT_PAD_CLASS
    from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS
    S = np.concatenate([pad_to(a, T, S_PAD_CLASS) for a in S])
    TH = np.concatenate([pad_to(a, T, TH_NULL_CLASS) for a in TH])
    DT = np.concatenate([pad_to(a, T, DT_PAD_CLASS) for a in DT])
    C = np.concatenate(C)

    np.savez_compressed(args.out, s=S, th=TH, dt=DT, cond=C,
                        s_temp=args.s_temp, th_temp=args.th_temp,
                        dt_temp=args.dt_temp, seed=args.seed,
                        ckpt=args.ckpt)
    n_ev = (S > 0) & (S < S_PAD_CLASS)
    print(f"  -> {args.out}  {S.shape}  median events "
          f"{np.median(n_ev.sum(1)):.0f}", flush=True)


if __name__ == "__main__":
    main()
