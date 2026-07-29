"""Score-function MMD fine-tune: train through the SERVED sampler.

Why this exists rather than train_events_polar_dm.py.

That trainer computes its match loss on `partial_reveal` plus `st_complete`
output run through the differentiable `stream_to_frames`. Two measurements this
session showed both halves of that are wrong, in the same direction.

  w4_train_serve_gap  the object being graded is not the object being served.
                      The training path scores 0.88 to 0.95 by the contract on
                      the same checkpoint that serves at 0.6459. Supplying the
                      character command is worth 0.02 of that and matching the
                      sampler settings another 0.02, so the shortfall is the
                      generation procedure itself, not its knobs.
  w4_token_ceiling    the renderer is not the shipped one. Real human tokens
                      through `stream_to_frames` read 0.5751 where the serving
                      decoder reads 0.5118, so roughly 0.06 of the gap the loss
                      chases is manufactured and closable by no parameter.

Both defects exist for one reason: a pathwise gradient has to be able to flow
back through generation, so generation had to be made differentiable, and every
step of making it differentiable moved it further from what is served. The
straight-through Gumbel completion, the frame-grid renderer and the held-real
timings are all consequences of that single constraint.

The score-function estimator drops the constraint. For a cost c(x) evaluated on
sampled sequences,

    grad E[c] = E[ c(x) * grad log p_theta(x) ]

so nothing between the model's logits and the cost has to be differentiable.
Generation can then be the exact serving sampler, the renderer can be the exact
serving decoder, and the features can come from `features.extract_feature_matrix`,
the contract itself, rather than a GPU analogue of it.

The price is variance, and that is where the queue earns its place. c(x) here is
the MMD witness measured against a ring buffer of 1024 past generated rows and
1024 real rows, which `w4_mmd_queue` measured at 100 percent separability and
Cohen's d 7.77 from 32 fresh samples, against 63 percent for the 32-by-32
estimator the old trainer used. A low variance per sample cost is exactly what a
score-function estimator needs.

This is policy-gradient machinery and the GRPO pilot is closed, so the
difference is worth stating plainly. GRPO optimised a scalar reward per sample
produced by a learned critic, and it failed the way reward-model methods fail,
by finding the critic's blind spots. There is no critic and no reward here. The
per-sample cost is the MMD witness against real data, a fixed statistic of the
two empirical distributions, and it cannot be gamed without actually moving the
generated distribution toward the real one.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from features import extract_feature_matrix
from models.event_stream_polar import (
    EventStreamPolarModel, class_to_speed, class_to_dtheta,
    S_MASK_TOKEN, TH_MASK_TOKEN, S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS,
    N_S_CLASSES, TH_BINS,
)
from training.train_events_polar import PolarEventDataset
from training.train_events_polar_dm import FeatQueue, mmd_rbf


# ---------------------------------------------------------------- rollout


@torch.no_grad()
def rollout(model, cond, feat, seq_len, n_steps, choice_temp, temp, device):
    """The serving sampler, recording what it needs for a replay with grad.

    Mirrors EventStreamPolarModel.sample at order="gumbel" with guidance off.
    Every position starts masked, including the tail, so the model emits its
    own PAD and chooses its own length exactly as it does when served. The dt
    flow runs alongside the reveal rather than being held at the real timings.

    Returns (dt_z, s_tok, th_tok, tape) where tape is one entry per reveal
    step holding the trunk inputs at that step and which tokens were taken.
    """
    B = cond.shape[0]
    dt_z = torch.randn(B, seq_len, device=device)
    s_tok = torch.full((B, seq_len), S_MASK_TOKEN, dtype=torch.long, device=device)
    th_tok = torch.full((B, seq_len), TH_MASK_TOKEN, dtype=torch.long, device=device)
    step = 1.0 / n_steps
    tape = []
    for i in range(n_steps):
        t_cont = 1.0 - i * step
        t_scaled = torch.full((B,), t_cont * (model.n_steps - 1), device=device)
        # the replay needs the trunk inputs as they were BEFORE this step wrote
        # its tokens, so snapshot first
        pre = (dt_z.clone(), s_tok.clone(), th_tok.clone(), t_scaled)

        x_feat = model.trunk(dt_z, s_tok, th_tok, t_scaled, cond, feat)
        dt_z = dt_z - step * model.dt_head(x_feat).squeeze(-1)
        s_logits = model.s_head(x_feat)

        t_next = max(t_cont - step, 0.0)
        n_target = int(round(float(model.sqrt_ab[int(t_next * (model.n_steps - 1))])
                             * seq_len))
        masked = s_tok == S_MASK_TOKEN
        n_new = n_target - int(seq_len - masked[0].sum().item())
        if n_new <= 0:
            continue

        s_probs = torch.softmax(s_logits / temp, dim=-1)
        s_new = torch.multinomial(s_probs.view(-1, s_probs.shape[-1]), 1).view(B, seq_len)
        s_for_th = torch.where(masked, s_new, s_tok.clamp(max=N_S_CLASSES - 1))
        th_l = model.th_logits(x_feat, s_for_th)
        th_probs = torch.softmax(th_l / temp, dim=-1)
        th_new = torch.multinomial(th_probs.view(-1, th_probs.shape[-1]), 1).view(B, seq_len)
        motion = (s_new > TICK_CLASS) & (s_new < S_PAD_CLASS)
        conf = s_probs.gather(-1, s_new.unsqueeze(-1)).squeeze(-1)
        th_conf = th_probs.gather(-1, th_new.clamp(max=TH_BINS - 1).unsqueeze(-1)).squeeze(-1)
        conf = torch.where(motion, conf * th_conf, conf)
        th_new = torch.where(motion, th_new, torch.full_like(th_new, TH_NULL_CLASS))

        g = -torch.log(-torch.log(torch.rand_like(conf).clamp(1e-9, 1.0)))
        score = torch.log(conf.clamp(min=1e-9)) + choice_temp * (1.0 - i / n_steps) * g
        score = torch.where(masked, score, torch.full_like(score, -1e9))
        rank = score.argsort(dim=-1, descending=True)
        ar = torch.arange(seq_len, device=device).unsqueeze(0)
        rev = torch.zeros_like(masked)
        rev.scatter_(1, rank, (ar < n_new).expand(B, -1))
        rev &= masked

        s_tok = torch.where(rev, s_new, s_tok)
        th_tok = torch.where(rev, th_new, th_tok)
        tape.append((*pre, rev, s_new, th_new, motion))
    return dt_z, s_tok, th_tok, tape


def replay_logp(model, cond, feat, entry, temp):
    """Differentiable log p of the tokens this step took, per sequence.

    One trunk forward with grad, reproducing the rollout's state at that step.
    The heading token only counts where the speed token it is conditioned on is
    a motion class, because the sampler forces TH_NULL everywhere else and a
    forced token carries no choice.
    """
    dt_z, s_tok, th_tok, t_scaled, rev, s_new, th_new, motion = entry
    x_feat = model.trunk(dt_z, s_tok, th_tok, t_scaled, cond, feat)
    s_lp = torch.log_softmax(model.s_head(x_feat) / temp, dim=-1)
    lp = s_lp.gather(-1, s_new.unsqueeze(-1)).squeeze(-1)
    masked = s_tok == S_MASK_TOKEN
    s_for_th = torch.where(masked, s_new, s_tok.clamp(max=N_S_CLASSES - 1))
    th_lp = torch.log_softmax(model.th_logits(x_feat, s_for_th) / temp, dim=-1)
    lp = lp + torch.where(
        motion,
        th_lp.gather(-1, th_new.clamp(max=TH_BINS - 1).unsqueeze(-1)).squeeze(-1),
        torch.zeros_like(lp),
    )
    return (lp * rev.float()).sum(dim=1)


# ---------------------------------------------------------------- decode


def decode_contract(s_tok, th_tok, dt_z, cond, dt_mean, dt_std, snap, do_round):
    """experiments/event_stream_polar._decode, batched, returning paths ready
    for features.extract_feature_matrix. Deliberately the serving arithmetic in
    float64 rather than the differentiable analogue: under a score-function
    gradient nothing here has to be differentiable, which is the whole point."""
    s_np = s_tok.cpu().numpy(); th_np = th_tok.cpu().numpy()
    dt_np = dt_z.cpu().numpy(); c_np = cond.cpu().numpy()
    spd_tab = class_to_speed(torch.arange(N_S_CLASSES)).numpy().astype(np.float64)
    dth_tab = class_to_dtheta(torch.arange(TH_BINS)).numpy().astype(np.float64)
    out = []
    for i in range(len(s_np)):
        sc = s_np[i]
        pad = np.flatnonzero(sc >= S_PAD_CLASS)
        n = int(pad[0]) if len(pad) else len(sc)
        if n < 2:
            out.append(None)
            continue
        s = spd_tab[np.clip(sc[:n], 0, N_S_CLASSES - 1)]
        d = dth_tab[np.clip(th_np[i, :n], 0, TH_BINS - 1)]
        m = sc[:n] > TICK_CLASS
        hd = math.atan2(float(c_np[i, 3]), float(c_np[i, 2])) + np.cumsum(np.where(m, d, 0.0))
        dx = np.where(m, s * np.cos(hd), 0.0)
        dy = np.where(m, s * np.sin(hd), 0.0)
        if snap > 0:
            sl = m & (s > 0) & (s < snap)
            dx = np.where(sl, np.round(dx), dx)
            dy = np.where(sl, np.round(dy), dy)
        x = np.concatenate([[0.0], np.cumsum(dx)])
        y = np.concatenate([[0.0], np.cumsum(dy)])
        if do_round:
            x, y = np.round(x), np.round(y)
        t = np.concatenate([[0.0], np.cumsum(
            np.clip(np.exp(dt_np[i, :n] * dt_std + dt_mean), 0.1, 1000.0) / 1000.0)])
        out.append(np.stack([x, y, t], 1))
    return out


def contract_features(paths):
    """(features, kept_index). Rows that the contract rejects are dropped, and
    the index says which sequences survived so their log p can be dropped too."""
    keep, rows = [], []
    for i, p in enumerate(paths):
        if p is None or len(p) < 5:
            continue
        F_ = extract_feature_matrix([p])
        if len(F_) and np.all(np.isfinite(F_[0])):
            rows.append(F_[0]); keep.append(i)
    if not rows:
        return np.empty((0, 18)), np.empty((0,), dtype=np.int64)
    return np.asarray(rows), np.asarray(keep, dtype=np.int64)


# ---------------------------------------------------------------- witness


def mmd_witness(fresh, gen_pool, ref_pool, bandwidths=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Per-sample cost whose batch mean is the gradient-bearing part of MMD^2.

    MMD^2 = E_gg k - 2 E_gr k + E_rr k, and only the first two terms depend on
    the generator, so the contribution of one fresh row is
    2 * (mean_j k(x, g_j) - mean_j k(x, r_j)). Lower is closer to real. Pools
    are the detached ring buffers, so this is the low variance estimator from
    w4_mmd_queue rather than the 32-by-32 one, which matters far more here than
    it did under a pathwise gradient.
    """
    dg = torch.cdist(fresh, gen_pool) ** 2
    dr = torch.cdist(fresh, ref_pool) ** 2
    c = fresh.new_zeros(fresh.shape[0])
    for bw in bandwidths:
        c = c + 2.0 * (torch.exp(-dg / (2 * bw)).mean(dim=1)
                       - torch.exp(-dr / (2 * bw)).mean(dim=1))
    return c


# ---------------------------------------------------------------- train


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    ckpt = torch.load(data_dir / args.load_from, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    dt_mean, dt_std = float(ckpt["dt_mean"]), float(ckpt["dt_std"])
    model = EventStreamPolarModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    seq_len = cfg["max_seq_len"]
    print(f"Loaded {args.load_from} (epoch {ckpt.get('epoch')}), seq_len={seq_len}", flush=True)

    feat_bank = fb_order = fb_sorted = None
    if cfg.get("feat_dim", 0) > 0 and "feat_bank" in ckpt:
        feat_bank = ckpt["feat_bank"].to(device)
        ld = ckpt["feat_bank_log_dist"]
        fb_order = torch.argsort(ld).to(device)
        fb_sorted = ld.sort().values.to(device)
        print(f"  character bank {tuple(feat_bank.shape)}, drawn as served", flush=True)

    s2 = np.load(data_dir / "events_s2.npy", mmap_mode="r")
    dth = np.load(data_dir / "events_dth.npy", mmap_mode="r")
    dt = np.load(data_dir / "events_dt.npy", mmap_mode="r")
    lengths = np.load(data_dir / "events_len.npy")
    conditions = np.load(data_dir / "events_cond.npy")
    rng = np.random.default_rng(123)
    idx = np.sort(rng.choice(len(lengths), min(len(lengths), 400_000), replace=False))
    ds = PolarEventDataset(s2[idx], dth[idx], dt[idx], lengths[idx], conditions[idx],
                           seq_len, dt_mean, dt_std)
    dl = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.num_workers > 0)
    print(f"  {len(ds):,} trajectories", flush=True)

    def real_contract(s_cls, th_cls, dt_z, cond):
        p = decode_contract(s_cls, th_cls, dt_z, cond, dt_mean, dt_std,
                            args.snap, args.round_pos)
        return contract_features(p)[0]

    # feature standardization in the CONTRACT space, from real data
    acc = []
    for bi, (dt_z, s_cls, th_cls, real, cond) in enumerate(dl):
        acc.append(real_contract(s_cls, th_cls, dt_z, cond))
        if bi >= args.stat_batches:
            break
    sf = np.concatenate(acc)
    f_mu = torch.tensor(sf.mean(0), dtype=torch.float32, device=device)
    f_sd = torch.tensor(sf.std(0), dtype=torch.float32, device=device).clamp(min=1e-4)
    zs = ((torch.tensor(sf, dtype=torch.float32, device=device) - f_mu) / f_sd).clamp(-10, 10)
    g = torch.Generator(device="cpu").manual_seed(0)
    fl = [mmd_rbf(zs[p[:args.batch_size]], zs[p[args.batch_size:2 * args.batch_size]]).item()
          for p in (torch.randperm(len(zs), generator=g) for _ in range(20))]
    print(f"  contract-space stats over {len(sf)} real trajectories; "
          f"real-vs-real mmd floor at n={args.batch_size}: {np.mean(fl):.4f}", flush=True)

    for p in model.dt_head.parameters():
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    gen_q = FeatQueue(args.mmd_queue, f_mu.shape[0], device)
    ref_q = FeatQueue(args.mmd_queue, f_mu.shape[0], device)

    save_path = data_dir / args.save_name
    model.train()
    t0 = time.time()
    ema_c = ema_anchor = None
    data_iter = iter(dl)
    for step_i in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        dt_z, s_cls, th_cls, real, cond = (x.to(device) for x in batch)
        B2 = dt_z.shape[0]
        h = B2 // 2

        # character command, drawn the way serving draws it
        feat = None
        if feat_bank is not None:
            pos = torch.searchsorted(fb_sorted, cond[:h, 0].contiguous())
            jit = torch.randint(-args.feat_win, args.feat_win + 1, (h,), device=device)
            pos = (pos + jit).clamp(0, len(fb_order) - 1)
            feat = feat_bank[fb_order[pos]] + args.feat_bw * torch.randn(
                h, feat_bank.shape[1], device=device)

        g_dt, g_s, g_th, tape = rollout(model, cond[:h], feat, seq_len,
                                        args.n_steps, args.choice_temp,
                                        args.temp, device)
        gf, keep = contract_features(
            decode_contract(g_s, g_th, g_dt, cond[:h], dt_mean, dt_std,
                            args.snap, args.round_pos))
        rf = real_contract(s_cls[h:], th_cls[h:], dt_z[h:], cond[h:])
        if len(gf) < 4 or len(rf) < 4:
            print(f"  step {step_i}: only {len(gf)} valid generated rows, skipped", flush=True)
            continue
        zg = ((torch.tensor(gf, dtype=torch.float32, device=device) - f_mu) / f_sd).clamp(-10, 10)
        zr = ((torch.tensor(rf, dtype=torch.float32, device=device) - f_mu) / f_sd).clamp(-10, 10)

        if gen_q.n >= gen_q.size:
            cost = mmd_witness(zg, torch.cat([zg, gen_q.get()]), ref_q.get())
        else:
            cost = mmd_witness(zg, zg, zr)
        gen_q.push(zg); ref_q.push(zr)

        adv = cost - cost.mean()
        if args.norm_adv:
            adv = adv / adv.std().clamp(min=1e-6)
        adv = adv.detach()
        keep_t = torch.tensor(keep, device=device)

        optimizer.zero_grad()
        # replay one step at a time and free it: the surrogate is a sum over
        # steps, so its gradient accumulates and peak memory stays flat in
        # n_steps rather than growing with it
        # log p is a sum over every revealed position, so it runs to hundreds
        # and would swamp the anchor once clip_grad_norm renormalises the two
        # together. Dividing by seq_len makes the surrogate a mean per-token
        # log probability instead: same gradient direction, O(1) scale, and
        # sf_weight then means what it says.
        for entry in tape:
            lp = replay_logp(model, cond[:h], feat, entry, args.temp)
            ((args.sf_weight / seq_len) * (adv * lp[keep_t]).mean()).backward()

        t_cont = torch.rand(B2 - h, device=device)
        t_int = (t_cont * (model.n_steps - 1)).long()
        dt_noisy, _, velocity = model.q_flow(dt_z[h:], t_cont)
        s_m, th_m, mask_a = model.q_mask_joint(s_cls[h:], th_cls[h:], t_int)
        v_pred, s_logits, th_logits = model(
            dt_noisy, s_m, th_m, t_cont * (model.n_steps - 1), cond[h:], s_cls[h:])
        w_flow = real[h:] + (1.0 - real[h:]) * 0.1
        flow_loss = ((v_pred - velocity) ** 2 * w_flow).sum() / w_flow.sum().clamp(1)
        ce_s = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]),
                               s_cls[h:].reshape(-1), reduction="none").view(B2 - h, -1)
        ws = mask_a.float() * (real[h:] + (1.0 - real[h:]) * 0.15)
        s_loss = (ce_s * ws).sum() / ws.sum().clamp(1)
        motion_a = (s_cls[h:] > TICK_CLASS) & (s_cls[h:] < S_PAD_CLASS)
        ce_th = F.cross_entropy(th_logits.reshape(-1, th_logits.shape[-1]),
                                th_cls[h:].clamp(max=TH_BINS - 1).reshape(-1),
                                reduction="none").view(B2 - h, -1)
        wt = (mask_a & motion_a).float()
        th_loss = (ce_th * wt).sum() / wt.sum().clamp(1)
        anchor = flow_loss + s_loss + th_loss
        (args.anchor_weight * anchor).backward()

        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        cm = cost.mean().item()
        ema_c = cm if ema_c is None else 0.95 * ema_c + 0.05 * cm
        ema_anchor = anchor.item() if ema_anchor is None else 0.95 * ema_anchor + 0.05 * anchor.item()
        if (step_i + 1) % args.log_every == 0:
            print(f"  step {step_i + 1}/{args.steps} | witness {ema_c:.4f} "
                  f"(raw {cm:.4f}, kept {len(gf)}/{h}) | anchor {ema_anchor:.3f} "
                  f"(flow {flow_loss.item():.3f} s {s_loss.item():.3f} th {th_loss.item():.3f}) "
                  f"| {time.time() - t0:.0f}s", flush=True)
        if args.save_every and (step_i + 1) % args.save_every == 0:
            save(model, optimizer, ckpt, cfg, dt_mean, dt_std, step_i + 1,
                 save_path.with_stem(save_path.stem + f"_s{step_i + 1}"))
    save(model, optimizer, ckpt, cfg, dt_mean, dt_std, args.steps, save_path)
    print(f"Done. Final witness (ema): {ema_c:.4f}", flush=True)


def save(model, optimizer, ckpt, cfg, dt_mean, dt_std, step, path):
    out = {"model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
           "config": cfg, "dt_mean": dt_mean, "dt_std": dt_std,
           "step": step, "epoch": ckpt.get("epoch")}
    for k in ("feat_mu", "feat_sd", "feat_bank", "feat_bank_log_dist"):
        if k in ckpt:
            out[k] = ckpt[k]
    torch.save(out, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="training")
    p.add_argument("--load-from", default="event_polar_4m_fc_v3.pt")
    p.add_argument("--save-name", default="event_polar_4m_sfmmd.pt")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32,
                   help="per side; the generated half is what gets rolled out")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--n-steps", type=int, default=24,
                   help="reveal steps in the rollout. Serving uses 100, but "
                        "w4_train_serve_gap measured 12 against 100 as worth "
                        "0.015 of AUC, and cost here is linear in this")
    p.add_argument("--choice-temp", type=float, default=10.0)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--snap", type=float, default=2.5)
    p.add_argument("--round-pos", type=int, default=1)
    p.add_argument("--mmd-queue", type=int, default=1024)
    p.add_argument("--sf-weight", type=float, default=1.0)
    p.add_argument("--anchor-weight", type=float, default=1.0)
    p.add_argument("--norm-adv", type=int, default=1)
    p.add_argument("--feat-bw", type=float, default=0.25)
    p.add_argument("--feat-win", type=int, default=256)
    p.add_argument("--stat-batches", type=int, default=30)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    args.round_pos = bool(args.round_pos)
    args.norm_adv = bool(args.norm_adv)
    train(args)


if __name__ == "__main__":
    main()
