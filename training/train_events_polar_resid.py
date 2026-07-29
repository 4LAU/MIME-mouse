"""W3 P1: closed-loop arrival fine-tune of the feature-conditioned model.

The model is told the endpoint (cond encodes the exact net displacement) but
draws the whole path against that one fixed instruction, so step errors
accumulate to a median 58px miss and forcing arrival afterwards costs +0.078
AUC (research/w3_landing_price_results.json). This fine-tune adds a feedback
channel: a per-sequence vector describing the displacement the still-masked
part of the sequence has yet to cover, measured from the longest fully
revealed prefix (model.prefix_resid). During training that is computed from
the masking pattern on real sequences, so recorded human closing behaviour
supervises it for free. During sampling the same function runs on the
sampler's partial state at every reveal iteration, which closes the loop.

The projection is zero-initialized, so training starts exactly at the
pretrained model. Everything else (losses, frozen dt head, feature
conditioning with the checkpoint's own stats and bank) matches
train_events_polar_featcond.py.

Run:
    env PYTHONPATH=. ~/venvs/mime/bin/python training/train_events_polar_resid.py \
        --load-from event_polar_4m_fc_v2.pt --save-name event_polar_4m_resid_v1.pt
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
    S_MASK_TOKEN, TH_MASK_TOKEN, N_S_CLASSES, TH_NULL_CLASS,
)
from training.train_events_polar import PolarEventDataset  # noqa: E402
from training.train_events_polar_dm import (  # noqa: E402
    build_value_tables, detector_features, real_batch_values, stream_to_frames,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    ckpt = torch.load(data_dir / args.load_from, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    cfg["resid_dim"] = args.resid_dim
    model = EventStreamPolarModel(**cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("resid_embed") for k in missing), missing
    print(f"Loaded {args.load_from} (epoch {ckpt.get('epoch')}), "
          f"resid_embed fresh ({len(missing)} tensors, zero-init output)", flush=True)

    # the base checkpoint already carries feature stats and the sampling bank;
    # reuse them so the feat pathway sees exactly the inputs it was tuned on.
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
    g_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(g_params, lr=args.lr, weight_decay=0.0)

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

        with torch.no_grad():
            feat = ((batch_features(s_cls, th_cls, dt_z, real, cond) - f_mu)
                    / f_sd).clamp(-10.0, 10.0)

        t_cont = torch.rand(B, device=device)
        t_int = (t_cont * (model.n_steps - 1)).long()
        dt_noisy, _, velocity = model.q_flow(dt_z, t_cont)
        if args.suffix_mask:
            # contiguous suffix masking: reveal a prefix, hide everything
            # from there to the end (pad tail included), so prefix_resid
            # carries a live signal instead of duplicating cond.
            # The prefix is a fraction of the FULL buffer, because that is
            # what the sampler reveals at time t (n_target = sqrt_ab[t] *
            # seq_len). v3 used a fraction of the real length instead, a
            # 6.5x timestep mismatch at the median length that collapsed
            # generation quality (ledger row ...fd8265eb).
            mask_prob = 1.0 - model.sqrt_ab[t_int].view(-1, 1)
            prefix = (s_cls.shape[1] * (1.0 - mask_prob)).round().long()
            idx_t = torch.arange(s_cls.shape[1], device=device).unsqueeze(0)
            mask = idx_t >= prefix
            s_m = s_cls.masked_fill(mask, S_MASK_TOKEN)
            th_m = th_cls.masked_fill(mask, TH_MASK_TOKEN)
            if args.frontier_k > 0:
                # the l2r decoder only ever consumes predictions at the
                # frontier (the next few positions after the prefix), but
                # uniform suffix CE is dominated by far-future positions
                # whose conditional entropy is irreducible - v4 plateaued
                # there. Concentrate the loss where decoding happens.
                front_w = torch.exp(-(idx_t - prefix).clamp(min=0).float()
                                    / args.frontier_k)
            else:
                front_w = torch.ones_like(mask, dtype=torch.float32)
        else:
            s_m, th_m, mask = model.q_mask_joint(s_cls, th_cls, t_int)
            front_w = torch.ones_like(mask, dtype=torch.float32)
        # the residual always comes from the true cond; the trunk's static
        # cond has its displacement dims withheld for a random subset so
        # endpoint information is forced through the residual channel (v1
        # left the pathway at zero because the signal was redundant).
        if args.draft_resid:
            # closed loop on the model's own draft (v6): sample provisional
            # tokens for the masked positions with a no-grad pass, complete
            # the sequence, and hand the trunk the resulting endpoint error.
            # The real tokens it is then trained on DO land on the target,
            # so the model learns to correct exactly the miss the residual
            # reports - the same signal the sampler feeds back at decode
            # time from the previous iteration's draft, under any order.
            with torch.no_grad():
                resid0 = model.prefix_resid(s_m, th_m, cond)
                x0 = model.trunk(dt_noisy, s_m, th_m,
                                 t_cont * (model.n_steps - 1), cond,
                                 feat=feat, resid=resid0)
                s_probs = torch.softmax(model.s_head(x0), dim=-1)
                s_prov = torch.multinomial(
                    s_probs.view(-1, s_probs.shape[-1]), 1).view(B, -1)
                s_draft = torch.where(mask, s_prov, s_cls)
                th_probs = torch.softmax(
                    model.th_logits(x0, s_draft.clamp(max=N_S_CLASSES - 1)),
                    dim=-1)
                th_prov = torch.multinomial(
                    th_probs.view(-1, th_probs.shape[-1]), 1).view(B, -1)
                prov_motion = (s_draft > TICK_CLASS) & (s_draft < S_PAD_CLASS)
                th_prov = torch.where(prov_motion, th_prov,
                                      torch.full_like(th_prov, TH_NULL_CLASS))
                th_draft = torch.where(mask, th_prov, th_cls)
                resid = model.prefix_resid(s_draft, th_draft, cond)
        else:
            resid = model.prefix_resid(s_m, th_m, cond)
        cond_in = cond
        if args.cond_disp_drop > 0:
            drop = (torch.rand(B, 1, device=device) < args.cond_disp_drop).float()
            cond_in = cond.clone()
            cond_in[:, [0, 2, 3]] = cond_in[:, [0, 2, 3]] * (1.0 - drop)
        v_pred, s_logits, th_logits = model(
            dt_noisy, s_m, th_m, t_cont * (model.n_steps - 1), cond_in, s_cls,
            feat=feat, resid=resid,
        )
        w_flow = real + (1.0 - real) * 0.1
        flow_loss = ((v_pred - velocity) ** 2 * w_flow).sum() / w_flow.sum().clamp(1)
        ce_s = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]),
                               s_cls.reshape(-1), reduction="none").view(B, -1)
        ws = mask.float() * (real + (1.0 - real) * 0.15) * front_w
        s_loss = (ce_s * ws).sum() / ws.sum().clamp(1)
        motion = (s_cls > TICK_CLASS) & (s_cls < S_PAD_CLASS)
        ce_th = F.cross_entropy(th_logits.reshape(-1, th_logits.shape[-1]),
                                th_cls.clamp(max=TH_BINS - 1).reshape(-1),
                                reduction="none").view(B, -1)
        wt = (mask & motion).float() * front_w
        th_loss = (ce_th * wt).sum() / wt.sum().clamp(1)
        loss = flow_loss + s_loss + th_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(g_params, 1.0)
        opt.step()

        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        step_i += 1
        if step_i % 50 == 0 or step_i == 1:
            print(f"  step {step_i:5d}/{args.steps} | loss {ema:.4f} "
                  f"(flow {flow_loss.item():.3f} s {s_loss.item():.3f} "
                  f"th {th_loss.item():.3f}) | {time.time() - t0:.0f}s", flush=True)
        if step_i % args.save_every == 0 or step_i == args.steps:
            out = {
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "config": cfg, "dt_mean": dt_mean, "dt_std": dt_std,
                "feat_mu": ckpt["feat_mu"], "feat_sd": ckpt["feat_sd"],
                "feat_bank": ckpt["feat_bank"],
                "feat_bank_log_dist": ckpt["feat_bank_log_dist"],
                "step": step_i, "epoch": ckpt.get("epoch"),
            }
            torch.save(out, latest_path)
            torch.save(out, save_path)

    print(f"Done. Final loss (ema): {ema:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--load-from", default="event_polar_4m_fc_v2.pt")
    parser.add_argument("--save-name", default="event_polar_4m_resid_v2.pt")
    parser.add_argument("--cond-disp-drop", type=float, default=0.5)
    parser.add_argument("--suffix-mask", action="store_true")
    parser.add_argument("--draft-resid", action="store_true")
    parser.add_argument("--frontier-k", type=float, default=0.0,
                        help="exp loss-weight decay length past the prefix "
                             "frontier; 0 disables (suffix-mask only)")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--resid-dim", type=int, default=4)
    parser.add_argument("--n-frames", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--auto-resume", action="store_true")
    train(parser.parse_args())
