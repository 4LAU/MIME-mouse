"""How detectable is the token stream to something that learns what to look at.

PRE REGISTERED in HANDOFF.md 2026-08-09, "## The learned critic bound".
Architecture, controls, thresholds and the prediction were fixed before the run.

`w4_views` put nine blocks of hand written statistics at 0.6201 against a 0.5095
split half floor, and the eighteen contract features on the same two samples at
0.6139 against 0.5050. Those agree, which says the decoder amplifies nothing and
that this file's descriptive vocabulary sees everything the contract sees. It
does not say what a discriminator that chooses its own statistics would find.

This measures that. A small bidirectional transformer reads the raw class
streams, a speed, a turn and a millisecond wait per event, together with the
four dimensional command the trajectory was generated for, and is trained to
say human or model. The held out AUC is an estimate of how far the model's
sequence distribution really is from the human one, and it bounds from above
everything any summary statistic can find.

The floor. The identical critic, identical hyperparameters, identical rows per
class, trained to separate the A half of the human pool from the B half. There
is nothing there to find, so anything it reads above 0.5 is capacity leaking
into memorisation and the reading that matters cannot be trusted above it.

The halves are assigned by a random permutation and NEVER by position. The
corpus is ordered by session, so the first and second halves of a sorted sample
are different people, and a positional floor measures whose mouse it is. A 7000
row smoke run made exactly that mistake and read 0.828, higher than the reading
it was supposed to be the floor for. Above 0.60 the run is VOID.

Both critics are conditional: the command is an input, so this is a test of the
CONDITIONAL distribution, which is the one that matters. A critic denied the
command could separate on the marginal mix of commands alone.

Safety. Reads the training corpus and one checkpoint, writes neither. Touches no
evaluation data and no scoring code, never `training/candi_polar_flow_best.pt`.
Paces on GPU temperature through `w4_latent`. Generation is one trajectory per
command, nothing selected, though nothing here is a deliverable in any case.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python -u research/w4_critic.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

os.environ.setdefault("EVENT_SNAP", "2.5")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import (DT_MAX_MS, EventARModel)  # noqa: E402
from models.event_stream_polar import (N_S_CLASSES, N_TH_CLASSES,  # noqa: E402
                                       S_PAD_CLASS, TH_NULL_CLASS,
                                       dth_lattice_to_class, s2_to_class)
from w4_latent import cooldown, gpu_c, throttle  # noqa: E402

DATA = Path("training")
N_DT_CLASSES = DT_MAX_MS + 1

# CLOSE means the critic finds little more than the summary statistics already
# did, so the model's sequence distribution is genuinely near the human one and
# the remaining work is calibration. WIDE means it finds a great deal more, so
# the model is wrong in ways no statistic in this file can see and descriptive
# work should stop. The band between them is the uninformative outcome.
STAT_AUC = 0.6201
CLOSE_AT = 0.68
WIDE_AT = 0.85
FLOOR_MAX = 0.60


class Critic(nn.Module):
    """Three class streams plus the command, pooled at a command initialised
    CLS position. Deliberately small: the question is how far apart the two
    distributions are, not how well this can be made to overfit."""

    def __init__(self, d=192, layers=4, heads=4, ff=384, drop=0.1):
        super().__init__()
        self.s_emb = nn.Embedding(N_S_CLASSES + 1, d)
        self.th_emb = nn.Embedding(N_TH_CLASSES + 1, d)
        self.dt_emb = nn.Embedding(N_DT_CLASSES + 1, d)
        self.pos = nn.Embedding(257, d)
        self.cls = nn.Linear(4, d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d),
                                  nn.GELU(), nn.Linear(d, 1))

    def forward(self, s, th, dt, mask, cond, dt_vec=None):
        """`dt_vec` replaces the timing embedding lookup with an already built
        (B, T, d) tensor. It exists so `w4_advtime` can push a straight through
        gradient into the generator's timing head. Left at None every line runs
        exactly as it did for `w4_critic` and `w4_critic_ablate`."""
        B, T = s.shape
        x = self.s_emb(s) + self.th_emb(th) + (
            self.dt_emb(dt) if dt_vec is None else dt_vec)
        x = x + self.pos(torch.arange(1, T + 1, device=s.device))[None]
        x = torch.cat([self.cls(cond)[:, None, :], x], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool,
                                     device=s.device), ~mask], 1)
        return self.head(self.enc(x, src_key_padding_mask=pad)[:, 0]).squeeze(-1)


def pack(rows, T):
    """A list of (s, th, dt) into padded arrays plus a validity mask."""
    n = len(rows)
    s = np.full((n, T), S_PAD_CLASS, dtype=np.int64)
    th = np.full((n, T), TH_NULL_CLASS, dtype=np.int64)
    dt = np.zeros((n, T), dtype=np.int64)
    m = np.zeros((n, T), dtype=bool)
    for i, (a, b, c) in enumerate(rows):
        L = min(len(a), T)
        s[i, :L], th[i, :L], dt[i, :L], m[i, :L] = (a[:L], b[:L], c[:L], True)
    return s, th, dt, m


def human_rows(arr, idx, min_len):
    out = []
    for i in idx:
        L = int(arr["len"][i])
        if L < min_len:
            out.append(None)
            continue
        s2 = torch.from_numpy(arr["s2"][i, :L].astype(np.int64))
        dth = torch.from_numpy(arr["dth"][i, :L].astype(np.int64))
        out.append((s2_to_class(s2).numpy(),
                    torch.where(s2 > 0, dth_lattice_to_class(dth),
                                torch.full_like(dth, TH_NULL_CLASS)).numpy(),
                    torch.round(torch.from_numpy(
                        arr["dt"][i, :L].astype(np.float32))
                    ).long().clamp(0, DT_MAX_MS).numpy()))
    return out


def generate(model, cond, batch, dev, seed, min_len):
    out, t0 = [], time.time()
    with torch.no_grad():
        for c0 in range(0, len(cond), batch):
            throttle()
            torch.manual_seed(seed + c0)
            blk = torch.tensor(cond[c0:c0 + batch], dtype=torch.float32,
                               device=dev)
            s_cls, th_cls, dt_cls = model.sample(blk, temperature=1.0)
            s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
            dt_np = dt_cls.cpu().numpy()
            pad = s_np >= S_PAD_CLASS
            for j in range(s_np.shape[0]):
                L = int(pad[j].argmax()) if pad[j].any() else s_np.shape[1]
                out.append(None if L < min_len
                           else (s_np[j, :L], th_np[j, :L], dt_np[j, :L]))
            print(f"    generated {len(out)}/{len(cond)} "
                  f"in {time.time() - t0:.0f}s, {gpu_c()}C", flush=True)
    return out


def train_critic(A, B, cond_a, cond_b, dev, epochs, bs, lr, seed, tag):
    """A is class 0, B is class 1. Returns the best held out AUC.

    The held out split is by trajectory and is made before any training, so a
    sequence the critic has seen never appears in the number reported.
    """
    torch.manual_seed(seed)
    n = min(len(A[0]), len(B[0]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1000, n // 6)
    vi, ti = perm[:n_val], perm[n_val:]

    def tens(X, cond, idx):
        return [torch.from_numpy(X[k][idx]) for k in range(3)] + \
               [torch.from_numpy(X[3][idx]), torch.from_numpy(cond[idx])]

    tr = [tens(A, cond_a, ti), tens(B, cond_b, ti)]
    va = [tens(A, cond_a, vi), tens(B, cond_b, vi)]
    net = Critic().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss()
    best, n_tr = 0.5, len(ti)

    for ep in range(epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(n_tr))
        for c0 in range(0, n_tr - bs + 1, bs):
            if (c0 // bs) % 20 == 0:
                throttle()
            k = order[c0:c0 + bs]
            xa = [t[k].to(dev) for t in tr[0]]
            xb = [t[k].to(dev) for t in tr[1]]
            logit = torch.cat([net(*xa), net(*xb)])
            y = torch.cat([torch.zeros(len(k)), torch.ones(len(k))]).to(dev)
            opt.zero_grad()
            lossf(logit, y).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        net.eval()
        with torch.no_grad():
            p = []
            for side in va:
                q = []
                for c0 in range(0, len(vi), 256):
                    q.append(net(*[t[c0:c0 + 256].to(dev)
                                   for t in side]).cpu())
                p.append(torch.cat(q))
        y = np.concatenate([np.zeros(len(vi)), np.ones(len(vi))])
        auc = float(roc_auc_score(y, torch.cat(p).numpy()))
        best = max(best, auc)
        print(f"    {tag} epoch {ep + 1}/{epochs} held out AUC {auc:.4f} "
              f"(best {best:.4f}) {gpu_c()}C", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="research/w4_critic.json")
    args = ap.parse_args()

    arr = {k: np.load(DATA / f"events_{k}.npy", mmap_mode="r")
           for k in ("s2", "dth", "dt", "len", "cond")}
    ok = np.load(DATA / "events_feat18_ok.npy")

    # Twice the per class count, split into an A half and a B half by a random
    # permutation and NOT by position. The corpus is ordered by session, so the
    # first half and the second half of any sorted sample are different people,
    # and a floor built that way measures whose mouse it is rather than what the
    # critic memorised. A 7000 row smoke run read 0.828 on exactly that mistake.
    # Generation runs on the A half only; the B half exists to give the floor
    # the same number of rows per class that the reading gets.
    rng = np.random.default_rng(args.seed)
    pool = np.sort(rng.choice(np.flatnonzero(ok), 2 * args.n, replace=False))
    assign = rng.permutation(2 * args.n)
    ai, bi = np.sort(assign[:args.n]), np.sort(assign[args.n:])
    cond = np.asarray(arr["cond"][pool], dtype=np.float32)

    dev = esp._DEVICE
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"\n  {len(pool)} random corpus rows, {args.n} of them generated for")
    print(f"  ckpt {args.ckpt} at step {ck.get('step', '?')}")
    print(f"  statistic based reading to beat {STAT_AUC}\n")

    # The cache holds fixed width integer arrays and a length, never a pickled
    # object array, so reloading it cannot execute anything. Generation is the
    # expensive half of this file and rerunning the critic alone is common.
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        z = np.load(cache)
        gl, gs, gt, gd = z["len"], z["s"], z["th"], z["dt"]
        gen = [None if gl[j] < args.min_len
               else (gs[j, :gl[j]].astype(np.int64),
                     gt[j, :gl[j]].astype(np.int64),
                     gd[j, :gl[j]].astype(np.int64)) for j in range(len(gl))]
        print(f"  loaded {len(gen)} generated rows from {cache}")
    else:
        cooldown()
        gen = generate(model, cond[ai], args.batch, dev, args.seed + 1,
                       args.min_len)
        if cache is not None:
            gl = np.array([0 if g is None else len(g[0]) for g in gen],
                          dtype=np.int32)
            gs, gt, gd, _ = pack([g if g is not None
                                  else (np.zeros(1, np.int64),) * 3
                                  for g in gen], 256)
            np.savez_compressed(cache, len=gl, s=gs.astype(np.int16),
                                th=gt.astype(np.int16), dt=gd.astype(np.int16))

    humA = human_rows(arr, pool[ai], args.min_len)
    humB = human_rows(arr, pool[bi], args.min_len)
    ka = [j for j in range(args.n) if humA[j] is not None
          and gen[j] is not None]
    kb = [j for j in range(args.n) if humB[j] is not None]
    m = min(len(ka), len(kb))
    ka, kb = ka[:m], kb[:m]
    print(f"  {m} rows per class after losses\n")
    if m < 5000:
        print("  ABORT, too few rows for a critic")
        return

    T = 256
    HA = pack([humA[j] for j in ka], T)
    G = pack([gen[j] for j in ka], T)
    HB = pack([humB[j] for j in kb], T)
    CA, CB = cond[ai][ka], cond[bi][kb]

    print("  the reading: human against its own paired generated rows")
    signal = train_critic(HA, G, CA, CA, dev, args.epochs, args.bs, args.lr,
                          args.seed + 2, "signal")

    # The floor. Same critic, same hyperparameters, same rows per class, human
    # against human across a random split of the pool. Anything above 0.5 here
    # is memorisation or session identity, not a difference the model made.
    print("\n  the floor: the A half of the human rows against the B half")
    floor = train_critic(HA, HB, CA, CB, dev, args.epochs, args.bs, args.lr,
                         args.seed + 3, "floor")

    excess = signal - floor
    print(f"\n  learned critic   {signal:.4f}")
    print(f"  memorisation floor {floor:.4f}")
    print(f"  excess           {excess:+.4f}")
    print(f"  statistics read  {STAT_AUC:.4f}")

    if floor > FLOOR_MAX:
        verdict = (f"VOID. The human against human floor reads {floor:.4f}, "
                   f"above {FLOOR_MAX}, so the critic is separating something "
                   f"that is present between two samples of humans and the "
                   f"reading of {signal:.4f} cannot be attributed to the "
                   f"model.")
    elif signal <= CLOSE_AT:
        verdict = (f"CLOSE. A learned critic reads {signal:.4f} against "
                   f"{STAT_AUC} for hand chosen statistics and a {floor:.4f} "
                   f"memorisation floor. The model's conditional sequence "
                   f"distribution is genuinely near the human one, there is no "
                   f"large hidden structure, and the remaining work is "
                   f"calibration rather than a new objective.")
    elif signal >= WIDE_AT:
        verdict = (f"WIDE. A learned critic reads {signal:.4f} against "
                   f"{STAT_AUC} for hand chosen statistics. The model is wrong "
                   f"in ways no summary statistic in this file can see. "
                   f"Descriptive work stops here and the next arm trains "
                   f"against a learned signal.")
    else:
        verdict = (f"MIDDLE. {signal:.4f} sits between {CLOSE_AT} and "
                   f"{WIDE_AT}, so the critic finds real structure the "
                   f"statistics miss without finding a great deal of it. "
                   f"Neither branch is claimed.")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"n_per_class": m, "seed": args.seed, "ckpt": args.ckpt,
               "epochs": args.epochs, "signal": signal, "floor": floor,
               "excess": excess, "statistic_auc": STAT_AUC,
               "verdict": verdict, "gpu_c": gpu_c()},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
