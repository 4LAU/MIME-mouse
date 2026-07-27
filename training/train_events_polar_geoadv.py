"""Phase 1: close the loop against the whole-path geometric critic.

ADVERSARIAL_CRITIC.md Phase 1. Not the full build: this is the smoke test that
decides whether Phase 2 is worth weeks. Two questions only. Can the generator
actually reduce this critic's score, and when it does, does the real detector
number move the right way on a proper N=2000 measurement.

Why a new script rather than a flag on train_events_polar_adv.py. That script's
critic is an MLP over the same 18 hand-measured numbers the detector uses, fed
through the differentiable rig. Its own header records how it failed: "the
critic found a real gap (D gap grew 0.03 -> 1.80 and never closed)" that the
generator could not act on. The successor, train_events_polar_advfc.py, blamed
the missing pathway and routed the criticism through the global character
vector instead. research/cond_realization_probe.py has since measured that the
model barely listens to that vector (commanded-to-realized coefficient about
0.1), so that attempt was speaking into a channel the generator ignores.

This one replaces the critic outright. It reads the whole resampled path as a
sequence, in the geometric vocabulary research/phase0b_critic.py established it
needs (speed, acceleration, jerk, curvature, angular velocity as explicit
channels, because a transformer cannot recover a second derivative from raw
deltas on this much data). research/w3_critic_coverage.py then confirmed the
thing that makes it worth coupling: it reads 0.843 against the quarter of human
movement the model cannot produce and 0.544 against the rest, so its gradient
points at the actual deficit rather than at the part already indistinguishable.

Everything else is inherited unchanged from the adv harness: partial no-grad
MaskGIT reveal, straight-through Gumbel completion, lattice-snapped integer
decode, 125Hz resample, dt head frozen, pretraining anchor, hinge loss, critic
lr an order above the generator's.

The geometry is computed with train_events_polar_dm.detector_features' clamps,
not phase0b's numpy ones. Those clamps (speed floor 30 px/s under the curvature
cube, segment length floor 0.5 px under the heading normalization) were bought
by a divergence in an earlier build; the numpy critic never needed them because
it never had to backpropagate. Values are unchanged, gradients are bounded.

Run (supervised, 90-minute bursts, never unattended -- see the GPU rules in
ADVERSARIAL_CRITIC.md):
    ~/venvs/mime/bin/python training/train_events_polar_geoadv.py \
        --steps 400 --load-from event_polar_4m_fc_v2.pt \
        --save-name event_polar_4m_geoadv_v1.pt
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
from models.event_stream_polar import (EventStreamPolarModel, S_PAD_CLASS,  # noqa: E402
                                       TH_BINS, TICK_CLASS)
from phase0b_critic import GeoPathCritic, N_CHANNELS  # noqa: E402
from training.train_events_polar import PolarEventDataset  # noqa: E402
from training.train_events_polar_dm import (  # noqa: E402
    build_value_tables, partial_reveal, real_batch_values, st_complete,
    stream_to_frames,
)

HZ = 125.0


def geo_channels(x, y, fmask, hz=HZ):
    """(B, L, 7) geometric channels from resampled frames, differentiable.

    Mirrors research/phase0b_critic.compute_channels_for_path channel for
    channel, including its left-edge padding so every channel lines up on the
    same step index as dx/dy and one mask covers them all. Differs only in the
    clamps, which are train_events_polar_dm.detector_features': the value is
    the same, the gradient is bounded.
    """
    dt = 1.0 / hz
    m1 = fmask[:, 1:] * fmask[:, :-1]                    # (B, L) segment valid
    dx = (x[:, 1:] - x[:, :-1]) * m1
    dy = (y[:, 1:] - y[:, :-1]) * m1
    speed = (dx ** 2 + dy ** 2 + 1e-8).sqrt() / dt
    vx, vy = dx / dt, dy / dt

    def lpad(a, n):
        """Left-pad by repeating the first computable step, as phase0b does."""
        return torch.cat([a[:, :1].expand(-1, n), a], dim=1) if n > 0 else a

    acc = (speed[:, 1:] - speed[:, :-1]) / dt            # L-1
    jerk = (acc[:, 1:] - acc[:, :-1]) / dt               # L-2

    ax = (vx[:, 1:] - vx[:, :-1]) / dt
    ay = (vy[:, 1:] - vy[:, :-1]) / dt
    # 1/speed^3 on sub-pixel frames is the dominant source of destabilizing
    # gradient noise; an earlier build diverged on it. Floor, do not detach.
    speed_mid = speed[:, :-1].clamp(min=30.0)
    curv = ((vx[:, :-1] * ay - vy[:, :-1] * ax).abs()
            / speed_mid ** 3).clamp(max=1e4)             # L-1

    # heading change via atan2(cross, dot) on segments normalized by a CLAMPED
    # length: identical value (atan2 is scale-invariant), but the 1/|seg|^2
    # gradient blowup at slow frames is capped. Slow frames are exactly where
    # the angular-velocity gap lives, so a hard detach there would freeze it.
    r = (dx ** 2 + dy ** 2 + 1e-12).sqrt().clamp(min=0.5)
    ux, uy = dx / r, dy / r
    cross_seg = ux[:, :-1] * uy[:, 1:] - uy[:, :-1] * ux[:, 1:]
    dot_seg = ux[:, :-1] * ux[:, 1:] + uy[:, :-1] * uy[:, 1:]
    omega = torch.atan2(cross_seg, dot_seg + 1e-9) / dt  # L-1

    ch = torch.stack([dx, dy, speed, lpad(acc, 1), lpad(jerk, 2),
                      lpad(curv, 1), lpad(omega, 1)], dim=-1)
    return ch * m1.unsqueeze(-1), m1


def geo_standardize(ch, m1, med, iqr):
    """Median/IQR then signed log1p, phase0b's robust_standardize.

    The compression is not cosmetic. Curvature and jerk are power-law tailed
    (a single sharp reversal lands orders of magnitude above the bulk), and
    feeding that raw into attention produced NaN losses inside one epoch.
    log1p is monotonic and near-linear in the bulk, so ordering survives.
    """
    z = (ch - med) / iqr
    return (torch.sign(z) * torch.log1p(z.abs())) * m1.unsqueeze(-1)


def measure_scales(dl, model_bits, device, args, n_batches=12):
    """Per-channel median and IQR over real resampled paths, computed once."""
    tables, dt_mean, dt_std = model_bits
    vals = []
    with torch.no_grad():
        for bi, (dt_z, s_cls, th_cls, real, cond) in enumerate(dl):
            dt_z, s_cls, th_cls, real, cond = (
                t.to(device) for t in (dt_z, s_cls, th_cls, real, cond))
            dt_s = torch.exp(dt_z * dt_std + dt_mean).clamp(0.1, 1000.0) / 1000.0
            rv = real_batch_values(s_cls, th_cls, tables)
            x, y, fm = stream_to_frames(rv[0], rv[1], rv[3], rv[4], dt_s, real,
                                        cond, args.n_frames)
            ch, m1 = geo_channels(x, y, fm)
            sel = m1.bool()
            vals.append(torch.stack([ch[..., c][sel] for c in range(N_CHANNELS)],
                                    dim=-1).cpu())
            if bi >= n_batches:
                break
    v = torch.cat(vals)
    med = v.median(dim=0).values
    q = torch.quantile(v, torch.tensor([0.25, 0.75]), dim=0)
    iqr = (q[1] - q[0]).clamp(min=1e-6)
    print(f"  channel scales over {len(v):,} real steps", flush=True)
    for c, nm in enumerate(["dx", "dy", "speed", "acc", "jerk", "curv", "omega"]):
        print(f"    {nm:<6} median {med[c]:>12.4f}  iqr {iqr[c]:>12.4f}")
    return med.to(device), iqr.to(device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    # weights_only=False: the checkpoint carries the config dict alongside the
    # tensors. These are this repo's own checkpoints on this machine, written
    # by training/train_events_polar*.py, never third-party downloads.
    ckpt = torch.load(data_dir / args.load_from, map_location=device,
                      weights_only=False)
    cfg = ckpt["config"]
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    model = EventStreamPolarModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {args.load_from} (epoch {ckpt.get('epoch')})", flush=True)

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
    dl = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True,
                    drop_last=True, persistent_workers=args.num_workers > 0)
    print(f"  {len(ds):,} trajectories", flush=True)

    tables = build_value_tables(device)
    med, iqr = measure_scales(dl, (tables, dt_mean, dt_std), device, args)
    med, iqr = med.to(device), iqr.to(device)

    critic = GeoPathCritic(n_channels=N_CHANNELS, max_len=args.n_frames).to(device)

    for p in model.dt_head.parameters():
        p.requires_grad_(False)
    g_params = [p for p in model.parameters() if p.requires_grad]
    opt_g = torch.optim.AdamW(g_params, lr=args.lr, weight_decay=0.0)
    opt_d = torch.optim.Adam(critic.parameters(), lr=args.critic_lr,
                             betas=(0.5, 0.999))

    save_path = data_dir / args.save_name
    latest_path = save_path.with_stem(save_path.stem + "_latest")
    start_step = 0
    if args.auto_resume and latest_path.exists():
        rck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(rck["model_state_dict"])
        critic.load_state_dict(rck["critic_state_dict"])
        opt_g.load_state_dict(rck["opt_g_state_dict"])
        opt_d.load_state_dict(rck["opt_d_state_dict"])
        start_step = rck["step"]
        print(f"  Resumed at step {start_step}", flush=True)

    def to_critic(vals, dt_s, real, cond):
        x, y, fm = stream_to_frames(vals[0], vals[1], vals[3], vals[4], dt_s,
                                    real, cond, args.n_frames)
        ch, m1 = geo_channels(x, y, fm)
        return geo_standardize(ch, m1, med, iqr), m1.bool()

    model.train()
    step_i = start_step
    t0 = time.time()
    ema_gap, ema_anchor = None, None
    data_iter = iter(dl)
    while step_i < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        dt_z, s_cls, th_cls, real, cond = (t.to(device) for t in batch)
        B2 = dt_z.shape[0]
        h = B2 // 2  # first half generates, second half is the real reference

        dt_s = torch.exp(dt_z * dt_std + dt_mean).clamp(0.1, 1000.0) / 1000.0

        r = float(np.random.default_rng(step_i).uniform(args.reveal_min,
                                                        args.reveal_max))
        s_tok, th_tok, masked = partial_reveal(
            model, dt_z[:h], cond[:h], real[:h], r,
            args.reveal_steps, args.choice_temp, device,
        )
        gen_vals = st_complete(model, dt_z[:h], s_tok, th_tok, masked,
                               cond[:h], real[:h], tables, args.tau)
        gen_c, gen_m = to_critic(gen_vals, dt_s[:h], real[:h], cond[:h])

        with torch.no_grad():
            ref_vals = real_batch_values(s_cls[h:], th_cls[h:], tables)
            ref_c, ref_m = to_critic(ref_vals, dt_s[h:], real[h:], cond[h:])

        # critic update(s) on detached generator paths
        gen_c_d, gen_m_d = gen_c.detach(), gen_m
        for _ in range(args.critic_iters):
            d_real = critic(ref_c, ref_m)
            d_fake = critic(gen_c_d, gen_m_d)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        gap = (d_real.mean() - d_fake.mean()).item()

        # generator update: fool the critic, stay anchored to pretraining
        warm = step_i < args.critic_warmup
        adv = (-critic(gen_c, gen_m).mean() if not warm
               else gen_c.new_zeros(()))

        t_cont = torch.rand(B2 - h, device=device)
        t_int = (t_cont * (model.n_steps - 1)).long()
        dt_noisy, _, velocity = model.q_flow(dt_z[h:], t_cont)
        s_m, th_m, mask_a = model.q_mask_joint(s_cls[h:], th_cls[h:], t_int)
        v_pred, s_logits, th_logits = model(
            dt_noisy, s_m, th_m, t_cont * (model.n_steps - 1), cond[h:],
            s_cls[h:],
        )
        w_flow = real[h:] + (1.0 - real[h:]) * 0.1
        flow_loss = ((v_pred - velocity) ** 2 * w_flow).sum() / w_flow.sum().clamp(1)
        ce_s = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]),
                               s_cls[h:].reshape(-1),
                               reduction="none").view(B2 - h, -1)
        ws = mask_a.float() * (real[h:] + (1.0 - real[h:]) * 0.15)
        s_loss = (ce_s * ws).sum() / ws.sum().clamp(1)
        motion_a = (s_cls[h:] > TICK_CLASS) & (s_cls[h:] < S_PAD_CLASS)
        ce_th = F.cross_entropy(th_logits.reshape(-1, th_logits.shape[-1]),
                                th_cls[h:].clamp(max=TH_BINS - 1).reshape(-1),
                                reduction="none").view(B2 - h, -1)
        wt = (mask_a & motion_a).float()
        th_loss = (ce_th * wt).sum() / wt.sum().clamp(1)
        anchor = flow_loss + s_loss + th_loss

        loss = args.adv_weight * adv + args.anchor_weight * anchor
        opt_g.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(g_params, 1.0)
        opt_g.step()

        ema_gap = gap if ema_gap is None else 0.95 * ema_gap + 0.05 * gap
        ema_anchor = (anchor.item() if ema_anchor is None
                      else 0.95 * ema_anchor + 0.05 * anchor.item())
        step_i += 1

        if step_i % 20 == 0 or step_i == 1:
            print(f"  step {step_i:4d}/{args.steps} | D gap {ema_gap:.4f} "
                  f"(real {d_real.mean().item():+.3f} "
                  f"fake {d_fake.mean().item():+.3f}) | "
                  f"anchor {ema_anchor:.3f} (flow {flow_loss.item():.3f} "
                  f"s {s_loss.item():.3f} th {th_loss.item():.3f}) | "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step_i % args.save_every == 0 or step_i == args.steps:
            out = {
                "model_state_dict": model.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "opt_g_state_dict": opt_g.state_dict(),
                "opt_d_state_dict": opt_d.state_dict(),
                "geo_median": med.cpu(), "geo_iqr": iqr.cpu(),
                "config": cfg, "dt_mean": dt_mean, "dt_std": dt_std,
                "step": step_i, "epoch": ckpt.get("epoch"),
            }
            torch.save(out, latest_path)
            torch.save(out, save_path)
            torch.save(out, save_path.with_stem(save_path.stem + f"_s{step_i}"))

    print(f"Done. Final D gap (ema): {ema_gap:.4f}", flush=True)
    print("Phase 1 reads on the score, not on this number: generate N=2000 "
          "and run research/autoloop/scoring.py against the checkpoints.",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="training")
    p.add_argument("--load-from", default="event_polar_4m_fc_v2.pt")
    p.add_argument("--save-name", default="event_polar_4m_geoadv_v1.pt")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32,
                   help="halved from the adv harness: the sequence critic "
                        "backpropagates through 255 frames, not 18 numbers")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--critic-iters", type=int, default=1)
    p.add_argument("--critic-warmup", type=int, default=50,
                   help="steps of critic-only training before adversarial "
                        "gradients reach the generator")
    p.add_argument("--adv-weight", type=float, default=1.0)
    p.add_argument("--anchor-weight", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--reveal-min", type=float, default=0.2)
    p.add_argument("--reveal-max", type=float, default=0.9)
    p.add_argument("--reveal-steps", type=int, default=12)
    p.add_argument("--choice-temp", type=float, default=7.0)
    p.add_argument("--n-frames", type=int, default=256)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--auto-resume", action="store_true")
    train(p.parse_args())
