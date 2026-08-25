"""w4_pairq. Train q2(e0, e1 | cond), the pair model. AMENDMENT 27 in
/home/aaronadmin/w4_arms/step0_prereg.md, registered before this file
was written.

Six head chain on the FirstHead trunk:
  p(s0) p(th0 | s0) p(dt0 | s0, th0)
  p(s1 | e0) p(th1 | s1, e0) p(dt1 | s1, th1, e0)
with e0 entering position 1 as summed embeddings of (s0, th0, dt0).
Training mirrors w4_firsthead cmd_train: rng(123) 1.5M train split
filtered to length >= 2, val = rng(2025) 200k held rows filtered the
same way, d 512, 12 epochs, batch 4096, AdamW lr 1e-3 wd 0.01,
OneCycleLR, clip 1.0, torch seed 0, best val sum checkpoint.

Gate (a) printed at the end: position 0 val CE vs FirstHead's best,
must be within 0.02 nats per term.

Reads training/events_*.npy, never the protected eval file, never
candi_polar_flow_best.pt. Saved to training/w4_pairq.pt.
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                      # noqa: E402
from models.event_ar import DT_MAX_MS, N_DT_CLASSES, dt_ms_to_class  # noqa: E402
from models.event_stream_polar import (N_S_CLASSES, N_TH_CLASSES,  # noqa: E402
                                       S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS,
                                       dth_lattice_to_class, s2_to_class)
from w4_firsthead import Q_PATH, ce_triplet                       # noqa: E402

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
PAIR_CACHE = "/home/aaronadmin/w4_arms/pairq_pos01.npz"
P_PATH = "training/w4_pairq.pt"
VAL_ROWS_SEED = 2025
N_VAL = 200_000


def splits():
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    return lengths, trained, held


def pair_tokens():
    """Positions 0 and 1 classes for every corpus row, cached once. Rows of
    length < 2 are junk here and get filtered by the caller."""
    if os.path.exists(PAIR_CACHE):
        z = np.load(PAIR_CACHE)
        return tuple(z[k] for k in ("s0", "th0", "dt0", "s1", "th1", "dt1"))
    t0 = time.time()
    s2 = np.ascontiguousarray(np.load("training/events_s2.npy", mmap_mode="r")[:, :2]).astype(np.int64)
    dth = np.ascontiguousarray(np.load("training/events_dth.npy", mmap_mode="r")[:, :2]).astype(np.int64)
    dt = np.ascontiguousarray(np.load("training/events_dt.npy", mmap_mode="r")[:, :2]).astype(np.float64)
    s = s2_to_class(torch.from_numpy(s2)).numpy()
    th = np.where(s2 > 0, dth_lattice_to_class(torch.from_numpy(dth)).numpy(), TH_NULL_CLASS)
    d = dt_ms_to_class(torch.from_numpy(dt)).numpy().clip(0, DT_MAX_MS)
    out = dict(s0=s[:, 0], th0=th[:, 0], dt0=d[:, 0],
               s1=s[:, 1], th1=th[:, 1], dt1=d[:, 1])
    np.savez(PAIR_CACHE, **out)
    print(f"  pair tokens cached, {time.time() - t0:.0f}s")
    return tuple(out[k] for k in ("s0", "th0", "dt0", "s1", "th1", "dt1"))


class PairHead(nn.Module):
    """q2(e0, e1 | cond), the FirstHead chain twice, position 1 conditioned
    on e0 through summed token embeddings."""

    def __init__(self, d=512, n_freq=6):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float() * np.pi / 4)
        inp = 4 + 4 * 2 * n_freq
        self.inp = nn.Sequential(nn.Linear(inp, d), nn.GELU(), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.s_emb = nn.Embedding(N_S_CLASSES, d)
        self.th_emb = nn.Embedding(N_TH_CLASSES, d)
        self.dt_emb = nn.Embedding(N_DT_CLASSES, d)
        self.s0_head = nn.Linear(d, N_S_CLASSES)
        self.th0_norm = nn.LayerNorm(d)
        self.th0_head = nn.Linear(d, N_TH_CLASSES)
        self.dt0_norm = nn.LayerNorm(d)
        self.dt0_head = nn.Linear(d, N_DT_CLASSES)
        self.s1_norm = nn.LayerNorm(d)
        self.s1_head = nn.Linear(d, N_S_CLASSES)
        self.th1_norm = nn.LayerNorm(d)
        self.th1_head = nn.Linear(d, N_TH_CLASSES)
        self.dt1_norm = nn.LayerNorm(d)
        self.dt1_head = nn.Linear(d, N_DT_CLASSES)

    def feat(self, cond):
        x = cond.unsqueeze(-1) * self.freqs
        return torch.cat([cond, torch.sin(x).flatten(1), torch.cos(x).flatten(1)], -1)

    def forward(self, cond, s0, th0, dt0, s1, th1):
        h = self.inp(self.feat(cond))
        zs0 = self.s0_head(h)
        zth0 = self.th0_head(self.th0_norm(h + self.s_emb(s0)))
        zdt0 = self.dt0_head(self.dt0_norm(h + self.s_emb(s0) + self.th_emb(th0)))
        e0 = self.s_emb(s0) + self.th_emb(th0) + self.dt_emb(dt0)
        zs1 = self.s1_head(self.s1_norm(h + e0))
        zth1 = self.th1_head(self.th1_norm(h + e0 + self.s_emb(s1)))
        zdt1 = self.dt1_head(self.dt1_norm(h + e0 + self.s_emb(s1) + self.th_emb(th1)))
        return zs0, zth0, zdt0, zs1, zth1, zdt1

    @torch.no_grad()
    def sample(self, cond, s_temp, th_temp, dt_temp):
        h = self.inp(self.feat(cond))

        def draw(zs_fn, zth_fn, zdt_fn):
            s = torch.multinomial(torch.softmax(zs_fn() / s_temp, -1), 1).squeeze(-1)
            th = torch.multinomial(torch.softmax(zth_fn(s) / th_temp, -1), 1).squeeze(-1)
            motion = (s > TICK_CLASS) & (s < S_PAD_CLASS)
            th = torch.where(motion, th, torch.full_like(th, TH_NULL_CLASS))
            dt = torch.multinomial(torch.softmax(zdt_fn(s, th) / dt_temp, -1), 1).squeeze(-1)
            return s, th, dt

        s0, th0, dt0 = draw(
            lambda: self.s0_head(h),
            lambda s: self.th0_head(self.th0_norm(h + self.s_emb(s))),
            lambda s, th: self.dt0_head(self.dt0_norm(h + self.s_emb(s) + self.th_emb(th))))
        e0 = self.s_emb(s0) + self.th_emb(th0) + self.dt_emb(dt0)
        s1, th1, dt1 = draw(
            lambda: self.s1_head(self.s1_norm(h + e0)),
            lambda s: self.th1_head(self.th1_norm(h + e0 + self.s_emb(s))),
            lambda s, th: self.dt1_head(self.dt1_norm(h + e0 + self.s_emb(s) + self.th_emb(th))))
        return s0, th0, dt0, s1, th1, dt1


P1_PATH = "training/w4_pairq1.pt"


class Pair1(nn.Module):
    """q1g0(e1 | e0, cond), the factorized conditional (AMENDMENT 27
    REVISION). Nothing shared with FirstHead or across positions: e0 enters
    as summed input embeddings concatenated to the Fourier featured cond,
    the chain heads have their own private within chain embeddings."""

    def __init__(self, d=512, n_freq=6):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float() * np.pi / 4)
        inp = 4 + 4 * 2 * n_freq + d
        self.e_s = nn.Embedding(N_S_CLASSES, d)
        self.e_th = nn.Embedding(N_TH_CLASSES, d)
        self.e_dt = nn.Embedding(N_DT_CLASSES, d)
        self.inp = nn.Sequential(nn.Linear(inp, d), nn.GELU(), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.s_head = nn.Linear(d, N_S_CLASSES)
        self.s_emb = nn.Embedding(N_S_CLASSES, d)
        self.th_norm = nn.LayerNorm(d)
        self.th_head = nn.Linear(d, N_TH_CLASSES)
        self.th_emb = nn.Embedding(N_TH_CLASSES, d)
        self.dt_norm = nn.LayerNorm(d)
        self.dt_head = nn.Linear(d, N_DT_CLASSES)

    def feat(self, cond):
        x = cond.unsqueeze(-1) * self.freqs
        return torch.cat([cond, torch.sin(x).flatten(1), torch.cos(x).flatten(1)], -1)

    def trunk(self, cond, s0, th0, dt0):
        e0 = self.e_s(s0) + self.e_th(th0) + self.e_dt(dt0)
        return self.inp(torch.cat([self.feat(cond), e0], -1))

    def forward(self, cond, s0, th0, dt0, s1, th1):
        h = self.trunk(cond, s0, th0, dt0)
        zs = self.s_head(h)
        zth = self.th_head(self.th_norm(h + self.s_emb(s1)))
        zdt = self.dt_head(self.dt_norm(h + self.s_emb(s1) + self.th_emb(th1)))
        return zs, zth, zdt

    @torch.no_grad()
    def sample(self, cond, s0, th0, dt0, s_temp, th_temp, dt_temp):
        h = self.trunk(cond, s0, th0, dt0)
        s = torch.multinomial(torch.softmax(self.s_head(h) / s_temp, -1), 1).squeeze(-1)
        th = torch.multinomial(torch.softmax(
            self.th_head(self.th_norm(h + self.s_emb(s))) / th_temp, -1), 1).squeeze(-1)
        motion = (s > TICK_CLASS) & (s < S_PAD_CLASS)
        th = torch.where(motion, th, torch.full_like(th, TH_NULL_CLASS))
        dt = torch.multinomial(torch.softmax(
            self.dt_head(self.dt_norm(h + self.s_emb(s) + self.th_emb(th))) / dt_temp, -1),
            1).squeeze(-1)
        return s, th, dt

    @torch.no_grad()
    def sample_completing(self, cond, s0, th0, dt0, s1=None, th1=None,
                          s_temp=1.0, th_temp=1.0, dt_temp=1.0):
        """AMENDMENT 29. The sample() chain, but any provided prefix of
        (s1, th1) is taken as given and only the rest is drawn, so the
        completion is coherent with the forced channels. The motion rule
        uses the effective s1."""
        h = self.trunk(cond, s0, th0, dt0)
        if s1 is None:
            s1 = torch.multinomial(torch.softmax(self.s_head(h) / s_temp, -1),
                                   1).squeeze(-1)
        if th1 is None:
            th1 = torch.multinomial(torch.softmax(
                self.th_head(self.th_norm(h + self.s_emb(s1))) / th_temp, -1),
                1).squeeze(-1)
            motion = (s1 > TICK_CLASS) & (s1 < S_PAD_CLASS)
            th1 = torch.where(motion, th1, torch.full_like(th1, TH_NULL_CLASS))
        dt1 = torch.multinomial(torch.softmax(
            self.dt_head(self.dt_norm(h + self.s_emb(s1) + self.th_emb(th1))) / dt_temp, -1),
            1).squeeze(-1)
        return s1, th1, dt1


def cmd_train1(a):
    dev = esp._DEVICE
    lengths, trained, held = splits()
    s0, th0, d0, s1, th1, d1 = pair_tokens()
    cond = np.load("training/events_cond.npy")[:, :4].astype(np.float32)
    trained = trained[lengths[trained] >= 2]
    val = np.sort(np.random.default_rng(VAL_ROWS_SEED).choice(held, N_VAL, replace=False))
    val = val[lengths[val] >= 2]
    torch.manual_seed(a.seed)
    q = Pair1(d=a.d).to(dev)
    opt = torch.optim.AdamW(q.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(trained) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    C = torch.from_numpy(cond).to(dev)
    S0, TH0, D0, S1, TH1, D1 = (torch.from_numpy(np.asarray(x, dtype=np.int64)).to(dev)
                                for x in (s0, th0, d0, s1, th1, d1))
    tr = torch.from_numpy(trained).to(dev)
    va = torch.from_numpy(val).to(dev)
    print(f"  q1g0 params {sum(p.numel() for p in q.parameters()) / 1e6:.2f}M  train rows "
          f"{len(trained):,} (length >= 2)  val rows {len(val):,}  epochs {a.epochs}"
          f"  steps {steps}", flush=True)

    def batch_terms(i):
        z = q(C[i], S0[i], TH0[i], D0[i], S1[i], TH1[i])
        return ce_triplet(z[0], z[1], z[2], S1[i], TH1[i], D1[i])

    def evaluate(idx):
        q.eval(); tot = np.zeros(5)
        with torch.no_grad():
            for c0 in range(0, len(idx), 65536):
                tot += [float(x) for x in batch_terms(idx[c0:c0 + 65536])]
        q.train()
        return tot[0] / tot[3], tot[1] / max(tot[4], 1), tot[2] / tot[3]

    best, hist, step = None, [], 0
    g = torch.Generator(device=dev).manual_seed(a.seed)
    for ep in range(a.epochs):
        perm = tr[torch.randperm(len(tr), generator=g, device=dev)]
        for c0 in range(0, len(perm) - a.batch + 1, a.batch):
            ls, lth, ldt, n, nm = batch_terms(perm[c0:c0 + a.batch])
            loss = ls / n + lth / nm.clamp(min=1) + ldt / n
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
        v = evaluate(va)
        hist.append(dict(epoch=ep + 1, step=step, val_s1=v[0], val_th1=v[1],
                         val_dt1=v[2], val_sum=sum(v)))
        print(f"  epoch {ep + 1:2d}  step {step}  pos1 s {v[0]:.4f} th {v[1]:.4f}"
              f" dt {v[2]:.4f}  sum {sum(v):.4f}", flush=True)
        if best is None or hist[-1]["val_sum"] < best["val_sum"]:
            best = hist[-1]
            torch.save(dict(config=dict(d=a.d), model_state_dict=q.state_dict(),
                            hist=hist, best=best, seed=a.seed), P1_PATH)
    print(f"  best epoch {best['epoch']} val sum {best['val_sum']:.4f}  saved {P1_PATH}")

    # GATE (a'): dedicated conditional must not be worse than the joint
    # PairHead's position 1 terms.
    jk = torch.load(P_PATH, map_location="cpu", weights_only=False)
    jb = jk["best"]
    print("  GATE (a'), position 1 val CE vs the joint PairHead best:")
    for name, pv, jv in (("s", best["val_s1"], jb["val_s1"]),
                         ("th", best["val_th1"], jb["val_th1"]),
                         ("dt", best["val_dt1"], jb["val_dt1"])):
        dlt = pv - jv
        print(f"    {name:>3}: q1g0 {pv:.4f}  joint {jv:.4f}"
              f"  delta {dlt:+.4f}  {'OK' if dlt <= 0.02 else 'FAIL'}")


def cmd_train(a):
    dev = esp._DEVICE
    lengths, trained, held = splits()
    s0, th0, d0, s1, th1, d1 = pair_tokens()
    cond = np.load("training/events_cond.npy")[:, :4].astype(np.float32)
    trained = trained[lengths[trained] >= 2]
    val = np.sort(np.random.default_rng(VAL_ROWS_SEED).choice(held, N_VAL, replace=False))
    val = val[lengths[val] >= 2]
    torch.manual_seed(a.seed)
    q = PairHead(d=a.d).to(dev)
    opt = torch.optim.AdamW(q.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(trained) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    C = torch.from_numpy(cond).to(dev)
    S0, TH0, D0, S1, TH1, D1 = (torch.from_numpy(np.asarray(x, dtype=np.int64)).to(dev)
                                for x in (s0, th0, d0, s1, th1, d1))
    tr = torch.from_numpy(trained).to(dev)
    va = torch.from_numpy(val).to(dev)
    print(f"  q2 params {sum(p.numel() for p in q.parameters()) / 1e6:.2f}M  train rows "
          f"{len(trained):,} (length >= 2)  val rows {len(val):,}  epochs {a.epochs}"
          f"  steps {steps}", flush=True)

    def batch_loss(i):
        z = q(C[i], S0[i], TH0[i], D0[i], S1[i], TH1[i])
        l0 = ce_triplet(z[0], z[1], z[2], S0[i], TH0[i], D0[i])
        l1 = ce_triplet(z[3], z[4], z[5], S1[i], TH1[i], D1[i])
        return l0, l1

    def evaluate(idx):
        q.eval(); tot = np.zeros(10)
        with torch.no_grad():
            for c0 in range(0, len(idx), 65536):
                l0, l1 = batch_loss(idx[c0:c0 + 65536])
                tot += [float(x) for x in l0 + l1]
        q.train()
        return (tot[0] / tot[3], tot[1] / max(tot[4], 1), tot[2] / tot[3],
                tot[5] / tot[8], tot[6] / max(tot[9], 1), tot[7] / tot[8])

    best, hist, step = None, [], 0
    g = torch.Generator(device=dev).manual_seed(a.seed)
    for ep in range(a.epochs):
        perm = tr[torch.randperm(len(tr), generator=g, device=dev)]
        for c0 in range(0, len(perm) - a.batch + 1, a.batch):
            l0, l1 = batch_loss(perm[c0:c0 + a.batch])
            loss = (l0[0] / l0[3] + l0[1] / l0[4].clamp(min=1) + l0[2] / l0[3]
                    + l1[0] / l1[3] + l1[1] / l1[4].clamp(min=1) + l1[2] / l1[3])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
        v = evaluate(va)
        hist.append(dict(epoch=ep + 1, step=step,
                         val_s0=v[0], val_th0=v[1], val_dt0=v[2],
                         val_s1=v[3], val_th1=v[4], val_dt1=v[5],
                         val_sum=sum(v)))
        print(f"  epoch {ep + 1:2d}  step {step}  pos0 s {v[0]:.4f} th {v[1]:.4f} dt {v[2]:.4f}"
              f"  pos1 s {v[3]:.4f} th {v[4]:.4f} dt {v[5]:.4f}  sum {sum(v):.4f}", flush=True)
        if best is None or hist[-1]["val_sum"] < best["val_sum"]:
            best = hist[-1]
            torch.save(dict(config=dict(d=a.d), model_state_dict=q.state_dict(),
                            hist=hist, best=best, seed=a.seed), P_PATH)
    print(f"  best epoch {best['epoch']} val sum {best['val_sum']:.4f}  saved {P_PATH}")

    # GATE (a): FirstHead's stored best was measured on the UNFILTERED val
    # rows; re-evaluate it on the length >= 2 rows used here so the
    # comparison is on identical data.
    from w4_firsthead import FirstHead
    qk = torch.load(Q_PATH, map_location=dev, weights_only=False)
    fh = FirstHead(**qk["config"]).to(dev).eval()
    fh.load_state_dict(qk["model_state_dict"])
    tot = np.zeros(5)
    with torch.no_grad():
        for c0 in range(0, len(va), 65536):
            i = va[c0:c0 + 65536]
            zs, zth, zdt = fh(C[i], S0[i], TH0[i])
            tot += [float(x) for x in ce_triplet(zs, zth, zdt, S0[i], TH0[i], D0[i])]
    fv = (tot[0] / tot[3], tot[1] / max(tot[4], 1), tot[2] / tot[3])
    print("  GATE (a), position 0 val CE vs FirstHead on the same filtered rows:")
    for name, pv, f in (("s", best["val_s0"], fv[0]), ("th", best["val_th0"], fv[1]),
                        ("dt", best["val_dt0"], fv[2])):
        dlt = pv - f
        print(f"    {name:>3}: pairq {pv:.4f}  firsthead {f:.4f}"
              f"  delta {dlt:+.4f}  {'OK' if abs(dlt) <= 0.02 else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "train1"])
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (cmd_train1 if a.cmd == "train1" else cmd_train)(a)


if __name__ == "__main__":
    main()
