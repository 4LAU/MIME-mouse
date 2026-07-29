"""Stage 4: teach the polar event model to OBEY its character command.

The diagnosis this run acts on, in full, from the HANDOFF section dated
2026-07-27:

The trunk is handed an 18-dim character vector drawn from a bank of real human
feature vectors, and the contract measures those same 18 numbers. If the model
realized its command exactly the score would be 0.50 by construction. It does
not. research/cond_realization_probe.py measures commanded-to-realized
correlation at 0.41 on average and at or below 0.31 across the whole shape
family, with mean_jerk at -0.005, which is no obedience at all.

Two inference-time repairs of that are already closed. Affine pre-distortion
inflates realized spread to 2.3x and scores 0.812. Guidance goes 0.6546 ->
0.6904 -> 0.8049 over w=0,2,4 and cannot reach curvature. Both failed for the
same reason: a command the model never learned to obey cannot be enforced at
sampling time. So it has to be trained in, which is what this does.

Three changes, all serving that one hypothesis, none of them tried before:

1. The command gets its own per-layer FiLM (models/event_stream_polar.py,
   feat_film=True). Previously it was SUMMED into a single d_model vector
   together with the diffusion timestep embedding and the displacement
   conditioning, and only that sum reached the block FiLM. The timestep signal
   varies over 1000 steps and owns most of the reconstruction loss, so the
   command was competing with it for one channel and losing.

2. feat_dropout=0.0. The parent was trained with the command deleted 10% of the
   time. That is the classifier-free guidance recipe, and it only pays off if
   guidance is used at serving. w3_guidance_capacity says guidance must stay at
   0 here, so the repo was paying the cost of the technique and taking none of
   the benefit, while actively teaching the trunk that the channel is
   unreliable and can be ignored.

3. Actual training. The parent's character channel was fine-tuned for 4000
   steps at lr 2e-5 from a zero init. That is close to no training at all, and
   it is the simplest available explanation for 0.41. This run gives the
   command pathway its own higher learning rate and runs 4x longer.

Everything else is held fixed on purpose. Same data, same losses, same frozen
dt_head, and the parent's OWN feat_mu, feat_sd and sampling bank are carried
over rather than recomputed, so the serving distribution is byte-identical to
fc_v2's and any movement in the score is attributable to obedience alone.

feat_film is zero-initialized, so training starts at the parent's behaviour to
the digit (verified: max abs trunk difference 0.0).

What this run does NOT attempt, deliberately, so the result stays readable:
an explicit rollout adherence loss, and the missing curvature variety
(w4_variety_vs_steering measures the model's curvature spread at 0.10 of human,
which is a capability gap, not a steering one). Both are next if this works.

Run:
    env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
      AVX512BW AVX512DQ AVX512VL" ~/venvs/mime/bin/python \
      training/train_events_polar_fc_v3.py
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.event_stream_polar import (  # noqa: E402
    EventStreamPolarModel, TICK_CLASS, S_PAD_CLASS, TH_BINS)
from training.train_events_polar import PolarEventDataset  # noqa: E402
from training.train_events_polar_dm import (  # noqa: E402
    build_value_tables, detector_features, real_batch_values, stream_to_frames,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    ckpt = torch.load(data_dir / args.load_from, map_location=device,
                      weights_only=False)
    cfg = dict(ckpt["config"])
    assert cfg.get("feat_dim") == args.n_feat, cfg.get("feat_dim")
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    cfg["feat_film"] = True
    cfg["feat_dropout"] = args.feat_dropout
    model = EventStreamPolarModel(**cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"],
                                                strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("feat_film") for k in missing), missing
    print(f"Loaded {args.load_from} (step {ckpt.get('step')}), "
          f"feat_film fresh ({len(missing)} tensors, zero-init: exact no-op)",
          flush=True)

    # the parent's own standardization and bank, carried over verbatim so the
    # serving distribution is identical and the comparison is clean
    f_mu = ckpt["feat_mu"].to(device)
    f_sd = ckpt["feat_sd"].to(device)
    bank = ckpt["feat_bank"]
    bank_log_dist = ckpt["feat_bank_log_dist"]
    print(f"  carried parent feat_mu/feat_sd and bank {tuple(bank.shape)}",
          flush=True)

    print("Loading polar event data...", flush=True)
    s2 = np.load(data_dir / "events_s2.npy", mmap_mode="r")
    dth = np.load(data_dir / "events_dth.npy", mmap_mode="r")
    dt = np.load(data_dir / "events_dt.npy", mmap_mode="r")
    lengths = np.load(data_dir / "events_len.npy")
    conditions = np.load(data_dir / "events_cond.npy")
    N = len(lengths)
    rng = np.random.default_rng(123)
    idx = np.sort(rng.choice(N, min(N, 400_000), replace=False))
    ds = PolarEventDataset(s2[idx], dth[idx], dt[idx], lengths[idx],
                           conditions[idx], cfg["max_seq_len"], dt_mean, dt_std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True,
                    drop_last=True, persistent_workers=args.num_workers > 0)
    print(f"  {len(ds):,} trajectories", flush=True)

    tables = build_value_tables(device)

    def batch_features(s_cls, th_cls, dt_z, real, cond):
        dt_s = torch.exp(dt_z * dt_std + dt_mean).clamp(0.1, 1000.0) / 1000.0
        speed, motion, tick, cos_th, sin_th = real_batch_values(
            s_cls, th_cls, tables)
        x, y, fmask = stream_to_frames(speed, motion, cos_th, sin_th,
                                       dt_s, real, cond, args.n_frames)
        return detector_features(x, y, fmask)

    for p in model.dt_head.parameters():
        p.requires_grad_(False)

    # the command pathway is what this run exists to train, so it gets its own
    # learning rate. The rest of the trunk moves slowly, since it is already
    # correct at the plan level and on every step-level marginal that was
    # measured and there is nothing to gain by disturbing it.
    cmd_names = ("feat_film", "feat_embed")
    cmd, base = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (cmd if n.startswith(cmd_names) else base).append(p)
    g_params = cmd + base
    opt = torch.optim.AdamW(
        [{"params": cmd, "lr": args.cmd_lr},
         {"params": base, "lr": args.lr}], weight_decay=0.0)
    print(f"  command pathway {sum(p.numel() for p in cmd):,} params at lr "
          f"{args.cmd_lr}, trunk {sum(p.numel() for p in base):,} at {args.lr}",
          flush=True)

    def lr_at(step):
        """Linear warmup then cosine decay, on both groups."""
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    save_path = data_dir / args.save_name
    latest_path = save_path.with_stem(save_path.stem + "_latest")
    start_step = 0
    if args.auto_resume and latest_path.exists():
        rck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(rck["model_state_dict"])
        opt.load_state_dict(rck["opt_state_dict"])
        start_step = rck["step"]
        print(f"  Resumed at step {start_step}", flush=True)

    model.train()
    step_i = start_step
    t0 = time.time()
    ema = None
    data_iter = iter(dl)
    while step_i < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        dt_z, s_cls, th_cls, real, cond = (x.to(device) for x in batch)
        B = dt_z.shape[0]

        scale = lr_at(step_i)
        for grp, base_lr in zip(opt.param_groups, (args.cmd_lr, args.lr)):
            grp["lr"] = base_lr * scale

        with torch.no_grad():
            feat = ((batch_features(s_cls, th_cls, dt_z, real, cond) - f_mu)
                    / f_sd).clamp(-10.0, 10.0)

        # t_power > 1 skews the diffusion time toward the heavily masked end.
        # That is where local context is unavailable and the character command
        # is the only information the trunk has, so it is where obedience is
        # actually taught. Default 1.0 leaves the parent's uniform schedule.
        u = torch.rand(B, device=device)
        t_cont = u ** (1.0 / args.t_power) if args.t_power != 1.0 else u
        t_int = (t_cont * (model.n_steps - 1)).long()
        dt_noisy, _, velocity = model.q_flow(dt_z, t_cont)
        s_m, th_m, mask = model.q_mask_joint(s_cls, th_cls, t_int)
        v_pred, s_logits, th_logits = model(
            dt_noisy, s_m, th_m, t_cont * (model.n_steps - 1), cond, s_cls,
            feat=feat,
        )
        w_flow = real + (1.0 - real) * 0.1
        flow_loss = ((v_pred - velocity) ** 2 * w_flow).sum() / w_flow.sum().clamp(1)
        ce_s = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]),
                               s_cls.reshape(-1), reduction="none").view(B, -1)
        ws = mask.float() * (real + (1.0 - real) * 0.15)
        s_loss = (ce_s * ws).sum() / ws.sum().clamp(1)
        motion = (s_cls > TICK_CLASS) & (s_cls < S_PAD_CLASS)
        ce_th = F.cross_entropy(th_logits.reshape(-1, th_logits.shape[-1]),
                                th_cls.clamp(max=TH_BINS - 1).reshape(-1),
                                reduction="none").view(B, -1)
        wt = (mask & motion).float()
        th_loss = (ce_th * wt).sum() / wt.sum().clamp(1)
        loss = flow_loss + s_loss + th_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(g_params, 1.0)
        opt.step()

        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        step_i += 1
        if step_i % 100 == 0 or step_i == 1:
            print(f"  step {step_i:5d}/{args.steps} | loss {ema:.4f} "
                  f"(flow {flow_loss.item():.3f} s {s_loss.item():.3f} "
                  f"th {th_loss.item():.3f}) | lr x{scale:.3f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step_i % args.save_every == 0 or step_i == args.steps:
            out = {
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "config": cfg, "dt_mean": dt_mean, "dt_std": dt_std,
                "feat_mu": f_mu.cpu(), "feat_sd": f_sd.cpu(),
                "feat_bank": bank, "feat_bank_log_dist": bank_log_dist,
                "step": step_i, "epoch": ckpt.get("epoch"),
            }
            torch.save(out, latest_path)
            torch.save(out, save_path)

    print(f"Done. Final loss (ema): {ema:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--load-from", default="event_polar_4m_fc_v2.pt")
    parser.add_argument("--save-name", default="event_polar_4m_fc_v3.pt")
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="trunk lr, the parent's value")
    parser.add_argument("--cmd-lr", type=float, default=3e-4,
                        help="lr for feat_film and feat_embed, the pathway "
                             "this run exists to train")
    parser.add_argument("--feat-dropout", type=float, default=0.0)
    parser.add_argument("--t-power", type=float, default=1.0)
    parser.add_argument("--n-feat", type=int, default=18)
    parser.add_argument("--n-frames", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--auto-resume", action="store_true")
    train(parser.parse_args())
