"""w4_firsthead. A dedicated first event conditional, and whether it moves the
contract. AMENDMENT 14 in /home/aaronadmin/w4_arms/step0_prereg.md.

w4_firstev (AMENDMENT 13, OUTCOME 11) found that replacing the model's first
event with the human row's own first event drops the contract by 0.023 at 3.7
paired se, a third of the residue, from one event in about forty. If the
model's p(e0 | cond) were the true conditional, a human draw from that same
conditional could not move the contract in expectation, so the model's first
event conditional is wrong as a distribution. Position 0 is one thirty ninth
of the training signal and the only position whose context is the four number
condition alone, which cond dropout zeroes ten percent of the time.

Three stages, each its own subcommand so the registration can read the first
before the third runs.

  nll    held out cross entropy at position 0, per head, for the AR optimum
         (teacher forced, T=1 forward, the same loss as training) and for a
         dedicated model q(e0 | cond). Also the AR's CE at positions 1 to 7
         for context, never compared to position 0 directly.
  train  fit q on the position 0 tokens of the AR model's own 1.5M training
         rows (same default_rng(123) pick), validate on 200k held out rows
         disjoint from the 200k the nll stage reports on.
  gen    the w4_firstev machinery with the forced first event drawn from q
         instead of the human row. Arms k0 (free running), q1 (q at
         temperature 1 on all heads), qT (q at the served AR temperatures
         s 0.95 th 0.90 dt 1.00). Same rows, same seeds, same eligibility
         (length > 4) as w4_firstev so the human k=1 ceiling pairs exactly.

One trajectory per row, no selection, every arm is a complete configuration.
Reads training/events_*.npy and checkpoints, never the protected eval file,
never candi_polar_flow_best.pt. q is saved to training/w4_firsthead_q.pt.
"""
from __future__ import annotations

import argparse
import json
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
import scoring                                                    # noqa: E402
from features import extract_feature_matrix                       # noqa: E402
from models.event_ar import (DT_MAX_MS, N_DT_CLASSES, EventARModel,  # noqa: E402
                             class_to_dt_ms, dt_ms_to_class, prefix_state)
from models.event_stream_polar import (N_S_CLASSES, N_TH_CLASSES,  # noqa: E402
                                       S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS,
                                       dth_lattice_to_class, s2_to_class)

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256
POS0_CACHE = "/home/aaronadmin/w4_arms/firsthead_pos0.npz"
Q_PATH = "training/w4_firsthead_q.pt"
NLL_ROWS_SEED, VAL_ROWS_SEED = 2024, 2025
N_NLL, N_VAL = 200_000, 200_000


def splits():
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    return lengths, trained, held


def pos0_tokens():
    """Position 0 classes for every corpus row, cached once. Rows of length
    0 carry PAD at position 0, the same supervised terminator the AR model
    sees, and are kept."""
    if os.path.exists(POS0_CACHE):
        z = np.load(POS0_CACHE)
        return z["s"], z["th"], z["dt"]
    t0 = time.time()
    s2 = np.ascontiguousarray(np.load("training/events_s2.npy", mmap_mode="r")[:, 0]).astype(np.int64)
    dth = np.ascontiguousarray(np.load("training/events_dth.npy", mmap_mode="r")[:, 0]).astype(np.int64)
    dt = np.ascontiguousarray(np.load("training/events_dt.npy", mmap_mode="r")[:, 0]).astype(np.float64)
    lengths = np.load("training/events_len.npy")
    s = s2_to_class(torch.from_numpy(s2)).numpy()
    th = np.where(s2 > 0, dth_lattice_to_class(torch.from_numpy(dth)).numpy(), TH_NULL_CLASS)
    d = dt_ms_to_class(torch.from_numpy(dt)).numpy().clip(0, DT_MAX_MS)
    empty = lengths < 1
    s[empty], th[empty], d[empty] = S_PAD_CLASS, TH_NULL_CLASS, 0
    np.savez(POS0_CACHE, s=s, th=th, dt=d)
    print(f"  position 0 tokens cached, {time.time() - t0:.0f}s, empty rows {int(empty.sum())}")
    return s, th, d


class FirstHead(nn.Module):
    """q(e0 | cond) = p(s) p(th | s) p(dt | s, th), the AR model's own within
    step chain, on a Fourier featured condition."""

    def __init__(self, d=512, n_freq=6):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float() * np.pi / 4)
        inp = 4 + 4 * 2 * n_freq
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

    def forward(self, cond, s, th):
        h = self.inp(self.feat(cond))
        zs = self.s_head(h)
        zth = self.th_head(self.th_norm(h + self.s_emb(s)))
        zdt = self.dt_head(self.dt_norm(h + self.s_emb(s) + self.th_emb(th)))
        return zs, zth, zdt

    @torch.no_grad()
    def sample(self, cond, s_temp, th_temp, dt_temp):
        h = self.inp(self.feat(cond))
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
    """Per row CE on the three heads, th only where there is motion, the
    training loss convention. Returns sums and counts."""
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


def cmd_train(a):
    dev = esp._DEVICE
    lengths, trained, held = splits()
    s, th, d = pos0_tokens()
    cond = np.load("training/events_cond.npy")[:, :4].astype(np.float32)
    val = np.sort(np.random.default_rng(VAL_ROWS_SEED).choice(held, N_VAL, replace=False))
    torch.manual_seed(a.seed)
    q = FirstHead(d=a.d).to(dev)
    opt = torch.optim.AdamW(q.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(trained) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    C = torch.from_numpy(cond).to(dev)
    S, TH, D = (torch.from_numpy(x).to(dev) for x in (s, th, d))
    tr = torch.from_numpy(trained).to(dev)
    va = torch.from_numpy(val).to(dev)
    print(f"  q params {sum(p.numel() for p in q.parameters()) / 1e6:.2f}M  train rows {len(trained):,}"
          f"  val rows {len(val):,}  epochs {a.epochs}  steps {steps}", flush=True)

    def evaluate(idx):
        q.eval(); tot = np.zeros(5)
        with torch.no_grad():
            for c0 in range(0, len(idx), 65536):
                i = idx[c0:c0 + 65536]
                zs, zth, zdt = q(C[i], S[i], TH[i])
                tot += [float(x) for x in ce_triplet(zs, zth, zdt, S[i], TH[i], D[i])]
        q.train()
        return tot[0] / tot[3], tot[1] / max(tot[4], 1), tot[2] / tot[3]

    best, hist, step = None, [], 0
    g = torch.Generator(device=dev).manual_seed(a.seed)
    for ep in range(a.epochs):
        perm = tr[torch.randperm(len(tr), generator=g, device=dev)]
        for c0 in range(0, len(perm) - a.batch + 1, a.batch):
            i = perm[c0:c0 + a.batch]
            zs, zth, zdt = q(C[i], S[i], TH[i])
            ls, lth, ldt, n, nm = ce_triplet(zs, zth, zdt, S[i], TH[i], D[i])
            loss = ls / n + lth / nm.clamp(min=1) + ldt / n
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
        v = evaluate(va)
        hist.append(dict(epoch=ep + 1, step=step, val_s=v[0], val_th=v[1], val_dt=v[2], val_sum=sum(v)))
        print(f"  epoch {ep + 1:2d}  step {step}  val s {v[0]:.4f}  th {v[1]:.4f}  dt {v[2]:.4f}"
              f"  sum {sum(v):.4f}", flush=True)
        if best is None or sum(v) < best["val_sum"]:
            best = hist[-1]
            torch.save(dict(config=dict(d=a.d), model_state_dict=q.state_dict(), hist=hist,
                            best=best, seed=a.seed), Q_PATH)
    print(f"  best epoch {best['epoch']} val sum {best['val_sum']:.4f}  saved {Q_PATH}")


def cmd_nll(a):
    dev = esp._DEVICE
    lengths, trained, held = splits()
    s, th, d = pos0_tokens()
    cond = np.load("training/events_cond.npy")[:, :4].astype(np.float32)
    rows = np.sort(np.random.default_rng(NLL_ROWS_SEED).choice(held, N_NLL, replace=False))
    ar, ck = load_ar(a.ckpt, dev)
    print(f"  {a.ckpt} step {ck.get('step')}   nll rows {len(rows):,} held out, seed {NLL_ROWS_SEED}")

    # AR at position 0, teacher forced with T=1, exactly the training loss.
    tot = np.zeros(5)
    with torch.no_grad():
        for c0 in range(0, len(rows), 4096):
            i = rows[c0:c0 + 4096]
            S = torch.from_numpy(s[i]).to(dev).unsqueeze(1)
            TH = torch.from_numpy(th[i]).to(dev).unsqueeze(1)
            D = torch.from_numpy(d[i]).to(dev).unsqueeze(1)
            Cn = torch.from_numpy(cond[i]).to(dev)
            sp, tp, dp = ar.shift_inputs(S, TH, D)
            st = prefix_state(S, TH, D, Cn)
            zs, zth, zdt = ar(sp, tp, dp, st, Cn, S, TH, D)
            tot += [float(x) for x in ce_triplet(zs[:, 0], zth[:, 0], zdt[:, 0],
                                                 S[:, 0], TH[:, 0], D[:, 0])]
    ar0 = dict(s=tot[0] / tot[3], th=tot[1] / max(tot[4], 1), dt=tot[2] / tot[3],
               motion_frac=tot[4] / tot[3])
    print(f"  AR position 0   s {ar0['s']:.4f}  th {ar0['th']:.4f}  dt {ar0['dt']:.4f}"
          f"  sum {ar0['s'] + ar0['th'] + ar0['dt']:.4f}   motion frac {ar0['motion_frac']:.3f}")

    # AR at positions 0..7 on rows with at least 8 events, for context only.
    KP = 8
    r8 = rows[lengths[rows] >= KP][:50_000]
    s2 = np.load("training/events_s2.npy", mmap_mode="r")
    dth = np.load("training/events_dth.npy", mmap_mode="r")
    dtm = np.load("training/events_dt.npy", mmap_mode="r")
    per = np.zeros((KP, 5))
    with torch.no_grad():
        for c0 in range(0, len(r8), 2048):
            i = r8[c0:c0 + 2048]
            S2 = torch.from_numpy(np.asarray(s2[i, :KP], dtype=np.int64))
            DH = torch.from_numpy(np.asarray(dth[i, :KP], dtype=np.int64))
            DM = torch.from_numpy(np.asarray(dtm[i, :KP], dtype=np.float64))
            S = s2_to_class(S2).to(dev)
            TH = torch.where(S2 > 0, dth_lattice_to_class(DH), torch.full_like(DH, TH_NULL_CLASS)).to(dev)
            D = dt_ms_to_class(DM).clamp(0, DT_MAX_MS).to(dev)
            Cn = torch.from_numpy(cond[i]).to(dev)
            sp, tp, dp = ar.shift_inputs(S, TH, D)
            st = prefix_state(S, TH, D, Cn)
            zs, zth, zdt = ar(sp, tp, dp, st, Cn, S, TH, D)
            for p in range(KP):
                per[p] += [float(x) for x in ce_triplet(zs[:, p], zth[:, p], zdt[:, p],
                                                        S[:, p], TH[:, p], D[:, p])]
    print(f"  AR per position, {len(r8):,} rows of length >= {KP}, context only")
    print(f"     pos      s      th      dt")
    ar_pos = []
    for p in range(KP):
        v = (per[p, 0] / per[p, 3], per[p, 1] / max(per[p, 4], 1), per[p, 2] / per[p, 3])
        ar_pos.append(v)
        print(f"     {p:3d}  {v[0]:.4f}  {v[1]:.4f}  {v[2]:.4f}")

    out = dict(ckpt=a.ckpt, n_rows=int(len(rows)), ar_pos0=ar0, ar_per_position=ar_pos)
    if os.path.exists(Q_PATH):
        qk = torch.load(Q_PATH, map_location=dev, weights_only=False)
        q = FirstHead(**qk["config"]).to(dev).eval()
        q.load_state_dict(qk["model_state_dict"])
        tot = np.zeros(5)
        with torch.no_grad():
            for c0 in range(0, len(rows), 16384):
                i = rows[c0:c0 + 16384]
                S = torch.from_numpy(s[i]).to(dev); TH = torch.from_numpy(th[i]).to(dev)
                D = torch.from_numpy(d[i]).to(dev); Cn = torch.from_numpy(cond[i]).to(dev)
                zs, zth, zdt = q(Cn, S, TH)
                tot += [float(x) for x in ce_triplet(zs, zth, zdt, S, TH, D)]
        q0 = dict(s=tot[0] / tot[3], th=tot[1] / max(tot[4], 1), dt=tot[2] / tot[3])
        print(f"  q  position 0   s {q0['s']:.4f}  th {q0['th']:.4f}  dt {q0['dt']:.4f}"
              f"  sum {q0['s'] + q0['th'] + q0['dt']:.4f}")
        print(f"  q minus AR      s {q0['s'] - ar0['s']:+.4f}  th {q0['th'] - ar0['th']:+.4f}"
              f"  dt {q0['dt'] - ar0['dt']:+.4f}  sum "
              f"{(q0['s'] + q0['th'] + q0['dt']) - (ar0['s'] + ar0['th'] + ar0['dt']):+.4f}"
              "   (negative means q is the better conditional)")
        out["q_pos0"] = q0
        out["q_best"] = qk["best"]
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  wrote {a.out}")


def cmd_gen(a):
    dev = esp._DEVICE
    lengths, trained, held = splits()
    rng = np.random.default_rng(1000 + a.seed)
    kmax = 4                                  # w4_firstev's eligibility, so rows pair
    elig = held[lengths[held] > kmax]
    pick = np.sort(rng.choice(elig, a.n, replace=False))
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    B = len(L)
    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))
    angs = np.arctan2(conds[:, 3].astype(np.float64), conds[:, 2].astype(np.float64))
    print(f"  held out {len(held):,}, eligible {len(elig):,}, using {a.n:,}, seed {a.seed}", flush=True)

    model, ck = load_ar(a.ckpt, dev)
    qk = torch.load(Q_PATH, map_location=dev, weights_only=False)
    q = FirstHead(**qk["config"]).to(dev).eval()
    q.load_state_dict(qk["model_state_dict"])
    print(f"  {a.ckpt} step {ck.get('step')}  q best epoch {qk['best']['epoch']}"
          f"  AR temps s {a.s_temp} th {a.th_temp} dt {a.dt_temp}", flush=True)

    def decode_all(s_np, th_np, dtc_np):
        paths = []
        for i in range(len(s_np)):
            dd = class_to_dt_ms(torch.from_numpy(dtc_np[i])).numpy()
            dz = (np.log(np.maximum(dd, 0.05)) - esp._DT_MEAN) / esp._DT_STD
            p = esp._decode(dz, s_np[i], th_np[i], 0.0, 0.0, float(angs[i]))
            if p is not None and len(p) >= 4:
                paths.append(np.asarray(p, dtype=np.float64))
        return paths

    ARMS = {"k0": None, "q1": (1.0, 1.0, 1.0), "qT": (a.s_temp, a.th_temp, a.dt_temp)}

    def generate(arm):
        qt = ARMS[arm]
        g_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
        g_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
        g_dt = np.zeros((B, MAX_T), dtype=np.int64)
        q0 = {"s": [], "th": [], "dt": []}
        for c0 in range(0, B, a.batch):
            sl = slice(c0, min(c0 + a.batch, B))
            cb = cond_t[sl].to(dev)
            f = None
            if qt is not None:
                torch.manual_seed(a.seed * 100003 + c0 + 7)
                qs, qth, qdt = q.sample(cb, *qt)
                nb = cb.shape[0]
                fs = torch.full((nb, MAX_T), S_PAD_CLASS, device=dev, dtype=torch.long)
                fth = torch.full((nb, MAX_T), TH_NULL_CLASS, device=dev, dtype=torch.long)
                fdt = torch.zeros((nb, MAX_T), device=dev, dtype=torch.long)
                fs[:, 0], fth[:, 0], fdt[:, 0] = qs, qth, qdt
                mask = torch.zeros((nb, MAX_T), device=dev, dtype=torch.bool)
                mask[:, 0] = True
                f = (fs, fth, fdt, mask)
                q0["s"] += qs.tolist(); q0["th"] += qth.tolist(); q0["dt"] += qdt.tolist()
            # same AR sampling seed per batch across arms, as in w4_firstev
            torch.manual_seed(a.seed * 100003 + c0)
            with torch.no_grad():
                s_o, th_o, dt_o = model.sample(cb, temperature=a.s_temp,
                                               th_temperature=a.th_temp,
                                               dt_temperature=a.dt_temp, force=f)
            n_got = s_o.shape[1]
            g_s[sl, :n_got] = s_o.cpu().numpy()
            g_th[sl, :n_got] = th_o.cpu().numpy()
            g_dt[sl, :n_got] = dt_o.cpu().numpy()
            if c0 % (a.batch * 20) == 0:
                print(f"    {arm}  {c0}/{B}", flush=True)
        info = {}
        if qt is not None:
            qs = np.array(q0["s"])
            info = dict(q_pad_frac=float((qs >= S_PAD_CLASS).mean()),
                        q_tick_frac=float((qs == TICK_CLASS).mean()))
        return decode_all(g_s, g_th, g_dt), info

    out = {"ckpt": a.ckpt, "seed": a.seed, "n_rows": int(B),
           "temps": [a.s_temp, a.th_temp, a.dt_temp], "arms": {}}
    print(f"\n  {'arm':>4}{'contract':>10}{'n':>7}{'collapse':>10}", flush=True)
    for arm in a.arms.split(","):
        paths, info = generate(arm)
        Fm = extract_feature_matrix(paths)
        Fm = Fm[np.all(np.isfinite(Fm), 1)]
        Fm = Fm[np.random.default_rng(a.seed).permutation(len(Fm))]
        r = scoring.score_features(Fm)
        out["arms"][arm] = dict(contract=float(r["auc_rf_oob"]), n=int(len(Fm)),
                                collapse=bool(r["collapse_flag"]),
                                collapse_features=list(r["collapse_features"]),
                                len_p50=float(np.median([len(p) for p in paths])), **info)
        print(f"  {arm:>4}    {r['auc_rf_oob']:.4f} {len(Fm):6d}     {str(bool(r['collapse_flag'])):>6}"
              f"  {info}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  wrote {a.out}")
    print("  one trajectory per row, no selection")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "nll", "gen"])
    ap.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--arms", default="k0,q1,qT")
    ap.add_argument("--s-temp", type=float, default=0.95)
    ap.add_argument("--th-temp", type=float, default=0.90)
    ap.add_argument("--dt-temp", type=float, default=1.00)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "gen" and a.batch == 4096:
        a.batch = 200
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "nll":
        a.out = a.out or "research/w4_firsthead_nll.json"
        cmd_nll(a)
    else:
        assert a.out, "--out required for gen"
        cmd_gen(a)


if __name__ == "__main__":
    main()
