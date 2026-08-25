"""w4_deepq. q past event 0: q4(e_k | cond, e_0..k-1) for k in 1, 2, 3.
AMENDMENT 21 in /home/aaronadmin/w4_arms/step0_prereg.md, registered before
this file was written.

Event 0 stays with the served w4_firsthead_q, untouched. This file trains
the opening model on positions 1 to 3 (subcommand train) and runs the
STAGE A gate (subcommand nll): held out CE at positions 1 to 3, q4 versus
the AR teacher forced on the same rows and examples, summed over three
heads and three positions. Bar: q4 better by >= 0.08 nats summed ->
PROCEED to the gen stage; less -> STOP the line.

Reads training/events_*.npy and checkpoints, never the protected eval
file, never candi_polar_flow_best.pt. q4 is saved to training/w4_deepq.pt.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_SNAP", "2.5")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_BESTOF", "1")
os.environ.setdefault("EVENT_SIR", "1")
os.environ.setdefault("EVENT_ORDER", "gumbel")
os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                      # noqa: E402
from models.event_ar import (DT_MAX_MS, N_DT_CLASSES, EventARModel,  # noqa: E402
                             dt_ms_to_class, prefix_state)
from models.event_stream_polar import (N_S_CLASSES, N_TH_CLASSES,  # noqa: E402
                                       S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS,
                                       dth_lattice_to_class, s2_to_class)

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
K = 4                     # positions 0..3 in the slab; q4 trains on 1..3
Q4_PATH = "training/w4_deepq.pt"
NLL_ROWS_SEED, VAL_ROWS_SEED = 2024, 2025
N_NLL, N_VAL = 200_000, 200_000


def splits():
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    return lengths, trained, held


def slab_tokens():
    """Class tokens at positions 0..3 for every corpus row, plus a validity
    mask. Position p is valid when p <= length: a real event for p < length,
    the PAD terminator (S_PAD, TH_NULL, dt 0) for p == length, exactly the
    AR's supervision. Positions past the terminator are invalid."""
    t0 = time.time()
    lengths = np.load("training/events_len.npy")
    s2 = np.asarray(np.load("training/events_s2.npy", mmap_mode="r")[:, :K], dtype=np.int64)
    dth = np.asarray(np.load("training/events_dth.npy", mmap_mode="r")[:, :K], dtype=np.int64)
    dtm = np.asarray(np.load("training/events_dt.npy", mmap_mode="r")[:, :K], dtype=np.float64)
    s = s2_to_class(torch.from_numpy(s2)).numpy()
    th = np.where(s2 > 0, dth_lattice_to_class(torch.from_numpy(dth)).numpy(), TH_NULL_CLASS)
    d = dt_ms_to_class(torch.from_numpy(dtm)).numpy().clip(0, DT_MAX_MS)
    pos = np.arange(K)[None, :]
    term = pos == lengths[:, None]
    s[term], th[term], d[term] = S_PAD_CLASS, TH_NULL_CLASS, 0
    valid = pos <= lengths[:, None]
    print(f"  slab built, {time.time() - t0:.0f}s  valid per position "
          + " ".join(f"{int(valid[:, p].sum()):,}" for p in range(K)), flush=True)
    return s, th, d, valid


class DeepQ(nn.Module):
    """q4(e_k | cond, e_0..k-1), k in 1..3. FirstHead's within step chain
    p(s) p(th | s) p(dt | s, th) on a trunk that sees the Fourier featured
    condition, the exact prefix_state at position k, class embeddings of the
    up to 3 previous events (absolute slots, zeroed when absent), and a
    learned position embedding."""

    E = 32

    def __init__(self, d=512, n_freq=6):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float() * np.pi / 4)
        inp = 4 + 4 * 2 * n_freq + 6 + (K - 1) * 3 * self.E + 16
        self.inp = nn.Sequential(nn.Linear(inp, d), nn.GELU(), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.slot_s = nn.Embedding(N_S_CLASSES, self.E)
        self.slot_th = nn.Embedding(N_TH_CLASSES, self.E)
        self.slot_dt = nn.Embedding(N_DT_CLASSES, self.E)
        self.pos_emb = nn.Embedding(K - 1, 16)
        self.s_head = nn.Linear(d, N_S_CLASSES)
        self.s_emb = nn.Embedding(N_S_CLASSES, d)
        self.th_norm = nn.LayerNorm(d)
        self.th_head = nn.Linear(d, N_TH_CLASSES)
        self.th_emb = nn.Embedding(N_TH_CLASSES, d)
        self.dt_norm = nn.LayerNorm(d)
        self.dt_head = nn.Linear(d, N_DT_CLASSES)

    def feat(self, cond, state, ps, pth, pdt, pmask, k):
        x = cond.unsqueeze(-1) * self.freqs
        four = torch.cat([cond, torch.sin(x).flatten(1), torch.cos(x).flatten(1)], -1)
        m = pmask.unsqueeze(-1).float()
        prev = torch.cat([self.slot_s(ps) * m, self.slot_th(pth) * m,
                          self.slot_dt(pdt) * m], -1).flatten(1)
        return torch.cat([four, state, prev, self.pos_emb(k - 1)], -1)

    def trunk(self, cond, state, ps, pth, pdt, pmask, k):
        return self.inp(self.feat(cond, state, ps, pth, pdt, pmask, k))

    def forward(self, cond, state, ps, pth, pdt, pmask, k, s, th):
        h = self.trunk(cond, state, ps, pth, pdt, pmask, k)
        zs = self.s_head(h)
        zth = self.th_head(self.th_norm(h + self.s_emb(s)))
        zdt = self.dt_head(self.dt_norm(h + self.s_emb(s) + self.th_emb(th)))
        return zs, zth, zdt

    @torch.no_grad()
    def sample(self, cond, state, ps, pth, pdt, pmask, k, s_temp=1.0,
               th_temp=1.0, dt_temp=1.0):
        h = self.trunk(cond, state, ps, pth, pdt, pmask, k)
        s = torch.multinomial(torch.softmax(self.s_head(h) / s_temp, -1), 1).squeeze(-1)
        th = torch.multinomial(torch.softmax(
            self.th_head(self.th_norm(h + self.s_emb(s))) / th_temp, -1), 1).squeeze(-1)
        motion = (s > TICK_CLASS) & (s < S_PAD_CLASS)
        th = torch.where(motion, th, torch.full_like(th, TH_NULL_CLASS))
        dt = torch.multinomial(torch.softmax(
            self.dt_head(self.dt_norm(h + self.s_emb(s) + self.th_emb(th))) / dt_temp, -1),
            1).squeeze(-1)
        return s, th, dt


def ce_triplet(zs, zth, zdt, s, th, dt):
    ce_s = F.cross_entropy(zs, s, reduction="none")
    ce_th = F.cross_entropy(zth, th, reduction="none")
    ce_dt = F.cross_entropy(zdt, dt, reduction="none")
    motion = ((s > 0) & (s < S_PAD_CLASS)).float()
    return (ce_s.sum(), ce_th.mul(motion).sum(), ce_dt.sum(),
            float(len(s)), motion.sum())


def load_ar(ckpt, dev):
    ck = torch.load(f"training/{ckpt}", map_location=dev, weights_only=False)
    m = EventARModel(**ck["config"]).to(dev).eval()
    m.load_state_dict(ck["model_state_dict"])
    return m, ck


def batch_inputs(S, TH, D, C, rows, ks, dev):
    """Per example inputs for rows (np array) at positions ks (np array of
    equal length, values 1..3). Returns everything DeepQ.forward needs plus
    the targets. State is computed with the exact AR prefix_state on the
    4 slab and gathered at each example's position."""
    Sb = torch.from_numpy(S[rows]).to(dev)
    THb = torch.from_numpy(TH[rows]).to(dev)
    Db = torch.from_numpy(D[rows]).to(dev)
    Cb = torch.from_numpy(C[rows]).to(dev)
    kb = torch.from_numpy(ks).to(dev)
    st = prefix_state(Sb, THb, Db, Cb)                    # (B, K, 6)
    stk = st.gather(1, kb.view(-1, 1, 1).expand(-1, 1, st.shape[-1])).squeeze(1)
    slots = torch.arange(K - 1, device=dev).unsqueeze(0)  # absolute slots 0..2
    pmask = slots < kb.unsqueeze(1)
    ps, pth, pdt = Sb[:, :K - 1], THb[:, :K - 1], Db[:, :K - 1]
    ar_idx = torch.arange(len(rows), device=dev)
    tgt = (Sb[ar_idx, kb], THb[ar_idx, kb], Db[ar_idx, kb])
    return Cb, stk, ps, pth, pdt, pmask, kb, tgt


def cmd_train(a):
    dev = esp._DEVICE
    lengths, trained, held = splits()
    S, TH, D, valid = slab_tokens()
    C = np.load("training/events_cond.npy")[:, :4].astype(np.float32)
    val = np.sort(np.random.default_rng(VAL_ROWS_SEED).choice(held, N_VAL, replace=False))

    def examples(rows):
        rr, kk = [], []
        for k in range(1, K):
            r = rows[lengths[rows] >= k]
            rr.append(r); kk.append(np.full(len(r), k, dtype=np.int64))
        return np.concatenate(rr), np.concatenate(kk)

    tr_r, tr_k = examples(trained)
    va_r, va_k = examples(val)
    torch.manual_seed(a.seed)
    q = DeepQ(d=a.d).to(dev)
    opt = torch.optim.AdamW(q.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(tr_r) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    print(f"  q4 params {sum(p.numel() for p in q.parameters()) / 1e6:.2f}M"
          f"  train examples {len(tr_r):,}  val examples {len(va_r):,}"
          f"  epochs {a.epochs}  steps {steps}", flush=True)

    def evaluate():
        q.eval()
        per = {k: np.zeros(5) for k in range(1, K)}
        with torch.no_grad():
            for c0 in range(0, len(va_r), 16384):
                r, k = va_r[c0:c0 + 16384], va_k[c0:c0 + 16384]
                Cb, stk, ps, pth, pdt, pm, kb, (ts, tth, td) = batch_inputs(S, TH, D, C, r, k, dev)
                zs, zth, zdt = q(Cb, stk, ps, pth, pdt, pm, kb, ts, tth)
                for kk in range(1, K):
                    m = kb == kk
                    if m.any():
                        per[kk] += [float(x) for x in ce_triplet(
                            zs[m], zth[m], zdt[m], ts[m], tth[m], td[m])]
        q.train()
        out = {}
        for kk in range(1, K):
            t = per[kk]
            out[kk] = (t[0] / t[3], t[1] / max(t[4], 1), t[2] / t[3])
        return out

    rng = np.random.default_rng(a.seed)
    best, hist, step = None, [], 0
    for ep in range(a.epochs):
        perm = rng.permutation(len(tr_r))
        for c0 in range(0, len(perm) - a.batch + 1, a.batch):
            i = perm[c0:c0 + a.batch]
            Cb, stk, ps, pth, pdt, pm, kb, (ts, tth, td) = batch_inputs(
                S, TH, D, C, tr_r[i], tr_k[i], dev)
            zs, zth, zdt = q(Cb, stk, ps, pth, pdt, pm, kb, ts, tth)
            ls, lth, ldt, n, nm = ce_triplet(zs, zth, zdt, ts, tth, td)
            loss = ls / n + lth / nm.clamp(min=1) + ldt / n
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
        v = evaluate()
        vsum = sum(sum(v[kk]) for kk in range(1, K))
        hist.append(dict(epoch=ep + 1, step=step, val_sum=vsum,
                         per_pos={kk: list(v[kk]) for kk in v}))
        print(f"  epoch {ep + 1:2d}  step {step}  val sum {vsum:.4f}  "
              + "  ".join(f"p{kk} {sum(v[kk]):.4f}" for kk in range(1, K)), flush=True)
        if best is None or vsum < best["val_sum"]:
            best = hist[-1]
            torch.save(dict(config=dict(d=a.d), model_state_dict=q.state_dict(),
                            hist=hist, best=best, seed=a.seed), Q4_PATH)
    print(f"  best epoch {best['epoch']} val sum {best['val_sum']:.4f}  saved {Q4_PATH}")


def cmd_nll(a):
    dev = esp._DEVICE
    lengths, trained, held = splits()
    S, TH, D, valid = slab_tokens()
    C = np.load("training/events_cond.npy")[:, :4].astype(np.float32)
    rows = np.sort(np.random.default_rng(NLL_ROWS_SEED).choice(held, N_NLL, replace=False))
    ar, ck = load_ar(a.ckpt, dev)
    print(f"  {a.ckpt} step {ck.get('step')}   nll rows {len(rows):,} held out, seed {NLL_ROWS_SEED}")

    # AR teacher forced on the FULL 256 wide layout, the training layout.
    # A truncated slab poisons prefix_state's idx / T feature (position 3
    # reads 0.75 on a 4 slab against 0.012 in training) and inflated the AR
    # CE by five nats on the first run of this stage; found 2026-08-22 by
    # checking the number against the AMENDMENT 14 context table before
    # trusting the verdict. CE collected at positions 1..3, valid only.
    T_FULL = 256
    s2m = np.load("training/events_s2.npy", mmap_mode="r")
    dthm = np.load("training/events_dth.npy", mmap_mode="r")
    dtmm = np.load("training/events_dt.npy", mmap_mode="r")
    per_ar = {k: np.zeros(5) for k in range(1, K)}
    with torch.no_grad():
        for c0 in range(0, len(rows), 256):
            i = rows[c0:c0 + 256]
            S2 = torch.from_numpy(np.asarray(s2m[i], dtype=np.int64))
            DH = torch.from_numpy(np.asarray(dthm[i], dtype=np.int64))
            DM = torch.from_numpy(np.asarray(dtmm[i], dtype=np.float64))
            Sb = s2_to_class(S2)
            THb = torch.where(S2 > 0, dth_lattice_to_class(DH),
                              torch.full_like(DH, TH_NULL_CLASS))
            Db = dt_ms_to_class(DM).clamp(0, DT_MAX_MS)
            ln = torch.from_numpy(lengths[i].astype(np.int64))
            pos = torch.arange(T_FULL).unsqueeze(0)
            term = pos == ln.unsqueeze(1)
            pad = pos > ln.unsqueeze(1)
            Sb[term | pad] = S_PAD_CLASS
            THb[term | pad] = TH_NULL_CLASS
            Db[term | pad] = 0
            Sb, THb, Db = Sb.to(dev), THb.to(dev), Db.to(dev)
            Cb = torch.from_numpy(C[i]).to(dev)
            sp, tp, dp = ar.shift_inputs(Sb, THb, Db)
            st = prefix_state(Sb, THb, Db, Cb)
            zs, zth, zdt = ar(sp, tp, dp, st, Cb, Sb, THb, Db)
            vb = torch.from_numpy(valid[i]).to(dev)
            for p in range(1, K):
                m = vb[:, p]
                if m.any():
                    per_ar[p] += [float(x) for x in ce_triplet(
                        zs[m, p], zth[m, p], zdt[m, p], Sb[m, p], THb[m, p], Db[m, p])]

    qk = torch.load(Q4_PATH, map_location=dev, weights_only=False)
    q = DeepQ(**qk["config"]).to(dev).eval()
    q.load_state_dict(qk["model_state_dict"])
    per_q = {k: np.zeros(5) for k in range(1, K)}
    with torch.no_grad():
        for k in range(1, K):
            r = rows[lengths[rows] >= k]
            for c0 in range(0, len(r), 16384):
                rr = r[c0:c0 + 16384]
                kk = np.full(len(rr), k, dtype=np.int64)
                Cb, stk, ps, pth, pdt, pm, kb, (ts, tth, td) = batch_inputs(
                    S, TH, D, C, rr, kk, dev)
                zs, zth, zdt = q(Cb, stk, ps, pth, pdt, pm, kb, ts, tth)
                per_q[k] += [float(x) for x in ce_triplet(zs, zth, zdt, ts, tth, td)]

    def unpack(t):
        return t[0] / t[3], t[1] / max(t[4], 1), t[2] / t[3]

    print(f"\n  held out CE, positions 1 to 3, {len(rows):,} rows, valid examples only")
    print(f"   pos          s                th                dt")
    tot_ar = tot_q = 0.0
    for p in range(1, K):
        va, vq = unpack(per_ar[p]), unpack(per_q[p])
        tot_ar += sum(va); tot_q += sum(vq)
        print(f"   {p}   AR {va[0]:.4f} q4 {vq[0]:.4f}   AR {va[1]:.4f} q4 {vq[1]:.4f}"
              f"   AR {va[2]:.4f} q4 {vq[2]:.4f}   n {int(per_ar[p][3]):,}")
    adv = tot_ar - tot_q
    print(f"\n  AR sum {tot_ar:.4f}  q4 sum {tot_q:.4f}  advantage {adv:+.4f} nats")
    verdict = "PROCEED" if adv >= 0.08 else "STOP"
    print(f"  STAGE A: advantage {adv:+.4f} (bar 0.08)  -> {verdict}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--d", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)
    n = sub.add_parser("nll")
    n.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    n.set_defaults(fn=cmd_nll)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
