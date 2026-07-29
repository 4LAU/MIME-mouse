"""W4: train the left-to-right autoregressive event model from scratch.

Rationale and the prior failure this is distinguished from are in
models/event_ar.py. In one line: `w4_refine` showed the masked model's local
conditionals are mutually inconsistent, a chain-rule factorization cannot have
that defect, and the three earlier suffix-mask attempts were short fine-tunes
of a bidirectional checkpoint rather than a from-scratch causal model.

Loss is plain teacher-forced cross entropy on all three streams, supervised at
every real position plus the single terminating PAD, which is what teaches the
model to stop. Nothing after that PAD is supervised, because the decoder never
reads it and weighting it only dilutes the gradient.

Run:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=. ~/venvs/mime/bin/python \
        training/train_event_ar.py --save-name event_ar_v1.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.event_ar import (  # noqa: E402
    DT_MAX_MS, EventARModel, N_DT_CLASSES, STATE_DIM, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    N_S_CLASSES, N_TH_CLASSES, S_PAD_CLASS, TH_NULL_CLASS,
    dth_lattice_to_class, s2_to_class,
)


class ARDataset(Dataset):
    """Same corpus as PolarEventDataset, but dt stays in whole milliseconds
    instead of being z-scored, and `n_sup` marks the supervised span."""

    def __init__(self, s2, dth, dt, lengths, conditions, max_len):
        self.s2, self.dth, self.dt = s2, dth, dt
        self.lengths, self.conditions = lengths, conditions
        self.max_len = max_len

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        T = self.max_len
        L = min(int(self.lengths[idx]), T)

        s2 = torch.from_numpy(self.s2[idx, :L].astype(np.int64))
        dth = torch.from_numpy(self.dth[idx, :L].astype(np.int64))

        s_cls = torch.full((T,), S_PAD_CLASS, dtype=torch.long)
        s_cls[:L] = s2_to_class(s2)
        th_cls = torch.full((T,), TH_NULL_CLASS, dtype=torch.long)
        th_cls[:L] = torch.where(s2 > 0, dth_lattice_to_class(dth),
                                 torch.full_like(dth, TH_NULL_CLASS))
        dt_cls = torch.zeros(T, dtype=torch.long)
        dt_ms = torch.from_numpy(self.dt[idx, :L].astype(np.float32))
        dt_cls[:L] = torch.round(dt_ms).long().clamp(0, DT_MAX_MS)

        # supervise the L real events plus the one PAD that terminates them
        n_sup = min(L + 1, T)
        return (s_cls, th_cls, dt_cls, torch.tensor(n_sup),
                torch.from_numpy(self.conditions[idx].copy()))


def batch_losses(model, batch, device, amp):
    """Teacher forced CE on the three streams.

    Shared by the training step and the held out evaluation so the two can
    never drift apart, which is the only way a validation number is worth
    logging at all.
    """
    s_cls, th_cls, dt_cls, n_sup, cond = (x.to(device, non_blocking=True)
                                          for x in batch)
    B, T = s_cls.shape

    s_prev, th_prev, dt_prev = model.shift_inputs(s_cls, th_cls, dt_cls)
    with torch.no_grad():
        state = prefix_state(s_cls, th_cls, dt_cls, cond)

    with torch.amp.autocast("cuda", enabled=amp):
        s_logits, th_logits, dt_logits = model(
            s_prev, th_prev, dt_prev, state, cond, s_cls, th_cls, dt_cls)

        ar = torch.arange(T, device=device).unsqueeze(0)
        sup = (ar < n_sup.unsqueeze(1)).float()
        # dtheta is only defined where there is motion; ticks and the
        # terminating PAD carry NULL and are excluded rather than taught
        motion = (s_cls > 0) & (s_cls < S_PAD_CLASS)
        sup_th = sup * motion.float()

        ce_s = F.cross_entropy(s_logits.reshape(-1, N_S_CLASSES),
                               s_cls.reshape(-1),
                               reduction="none").view(B, T)
        ce_th = F.cross_entropy(th_logits.reshape(-1, N_TH_CLASSES),
                                th_cls.reshape(-1),
                                reduction="none").view(B, T)
        ce_dt = F.cross_entropy(dt_logits.reshape(-1, N_DT_CLASSES),
                                dt_cls.reshape(-1),
                                reduction="none").view(B, T)
        s_loss = (ce_s * sup).sum() / sup.sum().clamp(1)
        th_loss = (ce_th * sup_th).sum() / sup_th.sum().clamp(1)
        dt_loss = (ce_dt * sup).sum() / sup.sum().clamp(1)
    return s_loss, th_loss, dt_loss


@torch.no_grad()
def validate(model, val_dl, device, amp):
    model.eval()
    tot = np.zeros(3)
    for nb, b in enumerate(val_dl, 1):
        tot += [float(x) for x in batch_losses(model, b, device, amp)]
    model.train()
    return tot / max(nb, 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    torch.manual_seed(args.seed)

    print("Loading polar event data...", flush=True)
    s2 = np.load(data_dir / "events_s2.npy", mmap_mode="r")
    dth = np.load(data_dir / "events_dth.npy", mmap_mode="r")
    dt = np.load(data_dir / "events_dt.npy", mmap_mode="r")
    lengths = np.load(data_dir / "events_len.npy")
    conditions = np.load(data_dir / "events_cond.npy")
    N = len(lengths)
    rng = np.random.default_rng(123)
    idx = np.sort(rng.choice(N, min(N, args.n_train), replace=False))
    ds = ARDataset(s2[idx], dth[idx], dt[idx], lengths[idx], conditions[idx],
                   args.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True,
                    drop_last=True, persistent_workers=args.num_workers > 0)
    print(f"  {len(ds):,} trajectories", flush=True)

    # `w4_arfit` had to recover this split after the fact, because v1 recorded
    # only a training loss EMA and the underfit question could not be answered
    # from the checkpoint. Held out loss is logged from the start now.
    val_dl = None
    if args.val_every:
        held = np.setdiff1d(np.arange(N), idx)
        if len(held) < args.val_n:
            raise SystemExit(f"only {len(held):,} unseen rows for a validation "
                             f"set of {args.val_n:,}; lower --n-train or "
                             f"--val-n, or pass --val-every 0")
        vi = np.sort(np.random.default_rng(7).choice(held, args.val_n,
                                                     replace=False))
        val_dl = DataLoader(
            ARDataset(s2[vi], dth[vi], dt[vi], lengths[vi], conditions[vi],
                      args.max_seq_len),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        print(f"  {args.val_n:,} held out for validation", flush=True)

    cfg = dict(d_model=args.d_model, n_heads=args.n_heads,
               n_layers=args.n_layers, d_ff=args.d_ff,
               max_seq_len=args.max_seq_len, cond_dim=4,
               dropout=args.dropout, cond_dropout=args.cond_dropout,
               emit_order=args.emit_order)
    model = EventARModel(**cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {n_par/1e6:.2f}M parameters, {args.n_layers} causal layers",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03,
        anneal_strategy="cos", div_factor=25.0, final_div_factor=20.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    save_path = data_dir / args.save_name
    latest_path = save_path.with_stem(save_path.stem + "_latest")
    start_step = 0
    if args.auto_resume and latest_path.exists():
        rck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(rck["model_state_dict"])
        opt.load_state_dict(rck["opt_state_dict"])
        sched.load_state_dict(rck["sched_state_dict"])
        start_step = rck["step"]
        print(f"  Resumed at step {start_step}", flush=True)

    model.train()
    step_i = start_step
    t0 = time.time()
    ema = None
    val_hist = []
    data_iter = iter(dl)
    while step_i < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        s_loss, th_loss, dt_loss = batch_losses(model, batch, device, args.amp)
        loss = s_loss + th_loss + dt_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        step_i += 1
        if step_i % args.log_every == 0 or step_i == 1:
            print(f"  step {step_i:6d}/{args.steps} | loss {ema:.4f} "
                  f"(s {s_loss.item():.3f} th {th_loss.item():.3f} "
                  f"dt {dt_loss.item():.3f}) | lr {sched.get_last_lr()[0]:.2e} "
                  f"| {time.time() - t0:.0f}s", flush=True)
        if val_dl is not None and (step_i % args.val_every == 0
                                   or step_i == args.steps):
            v = validate(model, val_dl, device, args.amp)
            val_hist.append(dict(step=step_i, train_ema=ema, s=float(v[0]),
                                 th=float(v[1]), dt=float(v[2]),
                                 total=float(v.sum())))
            print(f"  step {step_i:6d} | HELD OUT {v.sum():.4f} "
                  f"(s {v[0]:.3f} th {v[1]:.3f} dt {v[2]:.3f}) "
                  f"| train ema {ema:.4f} | gap {v.sum() - ema:+.4f}",
                  flush=True)

        if step_i % args.save_every == 0 or step_i == args.steps:
            out = {"model_state_dict": model.state_dict(),
                   "opt_state_dict": opt.state_dict(),
                   "sched_state_dict": sched.state_dict(),
                   "config": cfg, "step": step_i, "loss_ema": ema,
                   "val_hist": val_hist, "n_train": len(ds)}
            torch.save(out, latest_path)
            torch.save(out, save_path)
        # numbered snapshots so the loss against AUC slope can be measured
        # afterwards; without these a scaling run yields exactly one point
        if args.snapshot_every and step_i % args.snapshot_every == 0:
            torch.save({"model_state_dict": model.state_dict(),
                        "config": cfg, "step": step_i, "loss_ema": ema,
                        "val_hist": val_hist},
                       save_path.with_stem(f"{save_path.stem}_s{step_i}"))

    print(f"Done. Final loss (ema): {ema:.4f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="training")
    p.add_argument("--save-name", default="event_ar_v1.pt")
    p.add_argument("--n-train", type=int, default=1_500_000)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cond-dropout", type=float, default=0.1)
    p.add_argument("--emit-order", default="s_th_dt",
                   choices=["s_th_dt", "dt_s_th"],
                   help="within-step factorization; see models/event_ar.py")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=0,
                   help="held out loss every N steps; 0 disables. v1 recorded "
                        "no validation number and w4_arfit had to recover the "
                        "split after the fact to answer the underfit question")
    p.add_argument("--val-n", type=int, default=20_000)
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="keep a numbered checkpoint every N steps so the loss "
                        "against contract AUC slope can be measured; 0 keeps "
                        "only the overwritten latest")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--auto-resume", action="store_true")
    train(p.parse_args())
