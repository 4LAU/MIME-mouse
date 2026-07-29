"""W3 P2: learned movement character latent, replacing the hand-picked bank.

W2 closed the door on hand-picked statistics: steering toward detector
feature values hits the values without fooling the detector. This probe
replaces the 18-dim feature command with a small latent the model infers
from real paths while training. A per-path encoder reads the full event
sequence and produces z; the trunk receives z through the same additive
global slot the residual experiments used; a conditional prior learns to
draw z fresh from (distance, duration, angle) at serving time. Path-level
variety then lives in z by construction, which is aimed at the measured
failure: conditional under-dispersion, the model explaining all human
variety with per-step noise and averaging it away.

Trained as a VAE: reconstruction is the usual masked-token loss, plus a KL
between the encoder posterior and the conditional prior. The bank feature
command is dropped for half the batch so character information is forced
through z rather than staying redundant (the P1 v1 lesson).

Run:
    env PYTHONPATH=. ~/venvs/mime/bin/python training/train_events_polar_char.py \
        --save-name event_polar_4m_char_v1.pt
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
from models.event_stream_polar import (  # noqa: E402
    EventStreamPolarModel, TICK_CLASS, S_PAD_CLASS, TH_BINS,
    N_S_CLASSES, N_TH_CLASSES,
)
from training.train_events_polar import PolarEventDataset  # noqa: E402
from training.train_events_polar_dm import (  # noqa: E402
    build_value_tables, detector_features, real_batch_values, stream_to_frames,
)


class CharEncoder(nn.Module):
    """Reads a full real event sequence, returns a posterior over z."""

    def __init__(self, z_dim: int, d: int = 128, max_seq_len: int = 256):
        super().__init__()
        self.s_embed = nn.Embedding(N_S_CLASSES, d)
        self.th_embed = nn.Embedding(N_TH_CLASSES, d)
        self.dt_proj = nn.Linear(1, d)
        self.pos = nn.Embedding(max_seq_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=4 * d,
            dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * z_dim))

    def forward(self, s_cls, th_cls, dt_z, real):
        T = s_cls.shape[1]
        x = (self.s_embed(s_cls.clamp(max=N_S_CLASSES - 1))
             + self.th_embed(th_cls.clamp(max=N_TH_CLASSES - 1))
             + self.dt_proj(dt_z.unsqueeze(-1))
             + self.pos(torch.arange(T, device=s_cls.device)))
        pad = real < 0.5
        h = self.enc(x, src_key_padding_mask=pad)
        h = (h * real.unsqueeze(-1)).sum(1) / real.sum(1, keepdim=True).clamp(1)
        mu, logvar = self.head(h).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)


def build_prior(cond_dim: int, z_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(cond_dim, 128), nn.GELU(), nn.Linear(128, 2 * z_dim))


def kl_diag(mu_q, lv_q, mu_p, lv_p, free_bits: float = 0.0):
    """KL(q || p) for diagonal Gaussians, summed over z, mean over batch.

    free_bits floors each dimension's batch-mean KL: below the floor the
    term is constant, so there is no gradient reward for emptying that
    dimension. v1 collapsed to KL 0.00 within 300 steps because shrinking
    KL pays immediately while the zero-init z pathway pays nothing yet.
    """
    var_q, var_p = torch.exp(lv_q), torch.exp(lv_p)
    kl = 0.5 * (lv_p - lv_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    if free_bits > 0:
        return kl.mean(0).clamp(min=free_bits).sum()
    return kl.sum(-1).mean()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    ckpt = torch.load(data_dir / args.load_from, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    cfg["resid_dim"] = args.z_dim
    model = EventStreamPolarModel(**cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("resid_embed") for k in missing), missing
    print(f"Loaded {args.load_from}, z rides the resid slot "
          f"(dim {args.z_dim}, zero-init output)", flush=True)

    encoder = CharEncoder(args.z_dim, max_seq_len=cfg["max_seq_len"]).to(device)
    prior = build_prior(cfg["cond_dim"], args.z_dim).to(device)

    f_mu = ckpt["feat_mu"].to(device)
    f_sd = ckpt["feat_sd"].to(device)

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
                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.num_workers > 0)
    print(f"  {len(ds):,} trajectories", flush=True)

    tables = build_value_tables(device)

    def batch_features(s_cls, th_cls, dt_z, real, cond):
        dt_s = torch.exp(dt_z * dt_std + dt_mean).clamp(0.1, 1000.0) / 1000.0
        speed, motion, tick, cos_th, sin_th = real_batch_values(s_cls, th_cls, tables)
        x, y, fmask = stream_to_frames(speed, motion, cos_th, sin_th,
                                       dt_s, real, cond, args.n_frames)
        return detector_features(x, y, fmask)

    for p in model.dt_head.parameters():
        p.requires_grad_(False)
    trunk_params = [p for p in model.parameters() if p.requires_grad]
    fresh_params = list(encoder.parameters()) + list(prior.parameters())
    opt = torch.optim.AdamW(
        [{"params": trunk_params, "lr": args.lr},
         {"params": fresh_params, "lr": args.lr_fresh}], weight_decay=0.0)

    save_path = data_dir / args.save_name
    latest_path = save_path.with_stem(save_path.stem + "_latest")
    start_step = 0
    if args.auto_resume and latest_path.exists():
        rck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(rck["model_state_dict"])
        encoder.load_state_dict(rck["encoder_state_dict"])
        prior.load_state_dict(rck["prior_state_dict"])
        opt.load_state_dict(rck["opt_state_dict"])
        start_step = rck["step"]
        print(f"  Resumed at step {start_step}", flush=True)

    model.train()
    encoder.train()
    prior.train()
    step_i = start_step
    t0 = time.time()
    ema = None
    kl_ema = None
    data_iter = iter(dl)
    while step_i < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        dt_z, s_cls, th_cls, real, cond = (x.to(device) for x in batch)
        B = dt_z.shape[0]

        with torch.no_grad():
            feat = ((batch_features(s_cls, th_cls, dt_z, real, cond) - f_mu)
                    / f_sd).clamp(-10.0, 10.0)
        # drop the bank command for half the batch so character information
        # is forced through z instead of staying redundant with feat
        if args.feat_drop > 0:
            fdrop = (torch.rand(B, 1, device=device) < args.feat_drop).float()
            feat = feat * (1.0 - fdrop)

        mu_q, lv_q = encoder(s_cls, th_cls, dt_z, real)
        z = mu_q + torch.exp(0.5 * lv_q) * torch.randn_like(mu_q)
        mu_p, lv_p = prior(cond).chunk(2, dim=-1)
        kl = kl_diag(mu_q, lv_q, mu_p, lv_p.clamp(-8.0, 8.0),
                     free_bits=args.free_bits)

        t_cont = torch.rand(B, device=device)
        t_int = (t_cont * (model.n_steps - 1)).long()
        dt_noisy, _, velocity = model.q_flow(dt_z, t_cont)
        s_m, th_m, mask = model.q_mask_joint(s_cls, th_cls, t_int)
        v_pred, s_logits, th_logits = model(
            dt_noisy, s_m, th_m, t_cont * (model.n_steps - 1), cond, s_cls,
            feat=feat, resid=z,
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
        loss = flow_loss + s_loss + th_loss + args.beta * kl

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(trunk_params + fresh_params, 1.0)
        opt.step()

        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        kl_ema = kl.item() if kl_ema is None else 0.95 * kl_ema + 0.05 * kl.item()
        step_i += 1
        if step_i % 50 == 0 or step_i == 1:
            print(f"  step {step_i:5d}/{args.steps} | loss {ema:.4f} "
                  f"(flow {flow_loss.item():.3f} s {s_loss.item():.3f} "
                  f"th {th_loss.item():.3f} kl {kl_ema:.2f}) "
                  f"| {time.time() - t0:.0f}s", flush=True)
        if step_i % args.save_every == 0 or step_i == args.steps:
            out = {
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "prior_state_dict": prior.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "config": cfg, "z_dim": args.z_dim,
                "dt_mean": dt_mean, "dt_std": dt_std,
                "feat_mu": ckpt["feat_mu"], "feat_sd": ckpt["feat_sd"],
                "feat_bank": ckpt["feat_bank"],
                "feat_bank_log_dist": ckpt["feat_bank_log_dist"],
                "step": step_i, "epoch": ckpt.get("epoch"),
            }
            torch.save(out, latest_path)
            torch.save(out, save_path)

    print(f"Done. Final loss (ema): {ema:.4f}, kl (ema): {kl_ema:.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--load-from", default="event_polar_4m_fc_v2.pt")
    parser.add_argument("--save-name", default="event_polar_4m_char_v1.pt")
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--free-bits", type=float, default=0.0,
                        help="per-dim KL floor in nats; 0 disables")
    parser.add_argument("--feat-drop", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-fresh", type=float, default=1e-4)
    parser.add_argument("--n-frames", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--auto-resume", action="store_true")
    train(parser.parse_args())
