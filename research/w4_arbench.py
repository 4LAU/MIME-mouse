"""What size of AR model actually fits this card, and what does it cost?

`w4_arfit` settled the direction: the 7.95M model memorised nothing, its held
out loss is within 0.57 percent of its training loss on all three streams, so
it is capacity and optimisation limited and scaling is the lever.

The first attempt to price a scaled run failed informatively. At d_model 512,
12 layers, 45.55M parameters, the card sat at 7,923 of 8,188 MiB with 100
percent reported utilisation and only 34W of draw, and took over six seconds a
step. A 5.7x parameter increase producing a roughly 30x slowdown is not compute
scaling, it is thrashing against an 8GB limit. At six seconds a step a 40,000
step run is 66 hours, which does not fit a supervised session and is not
something this machine should attempt anyway.

So the question is not "how big can we go" but "what is the largest model that
still runs at a sane speed on an RTX 4070 Laptop with 8GB", and that has to be
measured rather than guessed. This times the real training step on synthetic
batches of the correct shape, which removes data loading from the measurement
entirely and isolates the thing being priced.

Reported per config:

  s/step      steady state, after warmup steps that are discarded because the
              first includes allocator growth and kernel autotuning
  alloc       peak tensor memory. What the model actually needs.
  reserved    peak the allocator held. Close to the card limit means the run is
              one fragmentation event away from thrashing even if alloc looks
              comfortable, which is exactly what the 512x12 config did.
  hours/40k   the honest wall clock for a run matching v1's step count, which
              is the number that decides whether a config is reachable at all

`--configs` takes `d_model:layers:d_ff:heads:batch` triples so the sweep is
visible in the command rather than buried here.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:training \
        ~/venvs/mime/bin/python research/w4_arbench.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

for p in (".", "research", "training"):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.event_ar import (  # noqa: E402
    DT_MAX_MS, EventARModel, S_BOS_CLASS, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    N_S_CLASSES, N_TH_CLASSES, S_PAD_CLASS, TH_NULL_CLASS,
)
from train_event_ar import batch_losses  # noqa: E402

DEFAULT = "256:8:1024:4:128,384:10:1536:6:128,512:12:2048:8:128,512:12:2048:8:64"


def synth(B, T, device, gen):
    """A batch with the shapes and value ranges the real loader produces."""
    s_cls = torch.randint(0, S_PAD_CLASS, (B, T), device=device, generator=gen)
    th_cls = torch.randint(0, TH_NULL_CLASS, (B, T), device=device,
                           generator=gen)
    dt_cls = torch.randint(0, DT_MAX_MS + 1, (B, T), device=device,
                           generator=gen)
    n_sup = torch.randint(T // 2, T, (B,), device=device, generator=gen)
    cond = torch.randn(B, 4, device=device, generator=gen)
    return s_cls, th_cls, dt_cls, n_sup, cond


def bench(spec, T, warmup, iters, device):
    d_model, n_layers, d_ff, n_heads, B = (int(x) for x in spec.split(":"))
    cfg = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
               max_seq_len=T, cond_dim=4, dropout=0.1, cond_dropout=0.1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = EventARModel(**cfg).to(device)
    model.train()
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01,
                            betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    gen = torch.Generator(device=device).manual_seed(0)
    batch = synth(B, T, device, gen)

    def step():
        s, th, dt = batch_losses(model, batch, device, True)
        loss = s + th + dt
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    sec = (time.time() - t0) / iters

    rec = dict(spec=spec, params=int(n_par), batch=B, s_per_step=sec,
               alloc_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
               reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
               traj_per_s=B / sec, hours_40k=sec * 40000 / 3600)
    del model, opt, scaler, batch
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=DEFAULT,
                    help="comma separated d_model:layers:d_ff:heads:batch")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--out", default="research/w4_arbench.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    print(f"  {torch.cuda.get_device_name(0)}, {total:,.0f} MiB, "
          f"seq {args.seq}, {args.iters} timed steps after {args.warmup} "
          f"warmup\n")
    print(f"  {'config':<22}{'params':>10}{'s/step':>9}{'traj/s':>9}"
          f"{'alloc':>9}{'reserv':>9}{'hrs/40k':>9}")

    out = []
    for spec in args.configs.split(","):
        try:
            r = bench(spec, args.seq, args.warmup, args.iters, device)
        except torch.cuda.OutOfMemoryError:
            print(f"  {spec:<22}{'OOM':>10}")
            out.append(dict(spec=spec, oom=True))
            torch.cuda.empty_cache()
            continue
        out.append(r)
        print(f"  {spec:<22}{r['params']/1e6:>9.1f}M{r['s_per_step']:>9.3f}"
              f"{r['traj_per_s']:>9.0f}{r['alloc_mib']:>9.0f}"
              f"{r['reserved_mib']:>9.0f}{r['hours_40k']:>9.1f}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  reserved close to {total:,.0f} means the config is one")
    print("  fragmentation event from thrashing even if alloc looks fine.")
    print("  hrs/40k is the wall clock that decides reachability: this machine")
    print("  runs supervised sessions only and has bluescreened under load.")


if __name__ == "__main__":
    main()
