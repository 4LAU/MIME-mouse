"""w4_nodur, AMENDMENT 17 in step0_prereg.md. Read that first.

FirstHeadND: the FirstHead architecture on a 3 number condition
([log dist, cos ang, sin ang], duration dropped), because under the
headline protocol the duration is a draw given distance and carries nothing
about e0 that distance does not. Train recipe identical to w4_firsthead.

  train    same splits, cache, val rows, optimizer, epochs, seed
  gen      the AMENDMENT 15 confirm protocol byte for byte, arms
           k0 / n1 / nT; the AR gets the full 4-number cond, q_nodur
           sees only the 3 it was trained on

Stage A read (registered): retained = 0.88 - (nodur val sum - 8.7426),
proceed to gen only if retained >= 0.44. No headline from this arm.
"""
from __future__ import annotations

import argparse
import math
import json
import os
import sys

import numpy as np
import torch

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

import torch.nn as nn                                              # noqa: E402

import experiments.event_stream_polar as esp                       # noqa: E402
import scoring                                                     # noqa: E402
from features import extract_feature_matrix                        # noqa: E402
from models.event_ar import (DT_MAX_MS, EventARModel,              # noqa: E402
                             class_to_dt_ms)
from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS   # noqa: E402
from phase_a_baseline import make_specs                            # noqa: E402
from w4_firsthead import (MAX_T, VAL_ROWS_SEED, N_VAL, FirstHead,  # noqa: E402
                          ce_triplet, load_ar, pos0_tokens, splits)

ND_PATH = "training/w4_nodur_q.pt"
COND_COLS = [0, 2, 3]  # log dist, cos ang, sin ang


class FirstHeadND(FirstHead):
    """FirstHead with the duration column dropped from the condition."""

    def __init__(self, d=512, n_freq=6, cond_dim=3):
        super().__init__(d=d, n_freq=n_freq)
        inp = cond_dim + cond_dim * 2 * n_freq
        self.inp = nn.Sequential(nn.Linear(inp, d), nn.GELU(), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, d), nn.GELU())


def cmd_train(a):
    dev = esp._DEVICE
    lengths, trained, held = splits()
    s, th, d = pos0_tokens()
    cond = np.load("training/events_cond.npy")[:, :4].astype(np.float32)[:, COND_COLS]
    val = np.sort(np.random.default_rng(VAL_ROWS_SEED).choice(held, N_VAL, replace=False))
    torch.manual_seed(a.seed)
    q = FirstHeadND(d=a.d).to(dev)
    opt = torch.optim.AdamW(q.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(trained) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    C = torch.from_numpy(cond).to(dev)
    S, TH, D = (torch.from_numpy(x).to(dev) for x in (s, th, d))
    tr = torch.from_numpy(trained).to(dev)
    va = torch.from_numpy(val).to(dev)
    print(f"  q_nodur params {sum(p.numel() for p in q.parameters()) / 1e6:.2f}M"
          f"  train rows {len(trained):,}  val rows {len(val):,}"
          f"  epochs {a.epochs}  steps {steps}", flush=True)

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
                            best=best, seed=a.seed), ND_PATH)
    retained = 0.88 - (best["val_sum"] - 8.7426)
    print(f"  best epoch {best['epoch']} val sum {best['val_sum']:.4f}  saved {ND_PATH}")
    print(f"  STAGE A: retained advantage {retained:.3f} of 0.88 nats"
          f"  (bar 0.44)  -> {'PROCEED' if retained >= 0.44 else 'STOP'}")


def cmd_gen(a):
    dev = esp._DEVICE
    model, ck = load_ar(a.ckpt, dev)
    qk = torch.load(ND_PATH, map_location=dev, weights_only=False)
    q = FirstHeadND(**qk["config"]).to(dev).eval()
    q.load_state_dict(qk["model_state_dict"])
    print(f"  {a.ckpt} step {ck.get('step')}  q_nodur best epoch {qk['best']['epoch']}"
          f"  AR temps s {a.s_temp} th {a.th_temp} dt {a.dt_temp}", flush=True)

    # The AMENDMENT 15 confirm's spec and duration construction, byte for byte.
    rows, meta = [], []
    for sx, sy, ex, ey in make_specs(a.n, a.seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([math.log(dist), math.log(esp._duration.sample(math.log(dist))),
                     math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ang))
    B = len(rows)
    cond_t = torch.tensor(rows, dtype=torch.float32)
    print(f"  specs {B} of {a.n}, seed {a.seed}", flush=True)

    ARMS = {"k0": None, "n1": (1.0, 1.0, 1.0), "nT": (a.s_temp, a.th_temp, a.dt_temp)}

    def decode_all(s_np, th_np, dtc_np):
        paths = []
        for i in range(len(s_np)):
            dd = class_to_dt_ms(torch.from_numpy(dtc_np[i])).numpy()
            dz = (np.log(np.maximum(dd, 0.05)) - esp._DT_MEAN) / esp._DT_STD
            sx, sy, ang = meta[i]
            p = esp._decode(dz, s_np[i], th_np[i], sx, sy, ang)
            if p is not None:
                paths.append(np.asarray(p, dtype=np.float64))
        return paths

    def generate(arm):
        qt = ARMS[arm]
        g_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
        g_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
        g_dt = np.zeros((B, MAX_T), dtype=np.int64)
        info = {"q_tick_frac": 0.0, "q_pad_frac": 0.0}
        nq = 0
        for c0 in range(0, B, a.batch):
            sl = slice(c0, min(c0 + a.batch, B))
            cb = cond_t[sl].to(dev)
            f = None
            if qt is not None:
                torch.manual_seed(a.seed * 100003 + c0 + 7)
                qs, qth, qdt = q.sample(cb[:, COND_COLS], *qt)
                nb = cb.shape[0]
                fs = torch.full((nb, MAX_T), S_PAD_CLASS, device=dev, dtype=torch.long)
                fth = torch.full((nb, MAX_T), TH_NULL_CLASS, device=dev, dtype=torch.long)
                fdt = torch.zeros((nb, MAX_T), device=dev, dtype=torch.long)
                fs[:, 0], fth[:, 0], fdt[:, 0] = qs, qth, qdt.clamp(max=DT_MAX_MS)
                mask = torch.zeros((nb, MAX_T), device=dev, dtype=torch.bool)
                mask[:, 0] = True
                f = (fs, fth, fdt, mask)
                info["q_tick_frac"] += float((qs == 0).sum())
                info["q_pad_frac"] += float((qs >= S_PAD_CLASS).sum())
                nq += nb
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
        if nq:
            info = {k: v / nq for k, v in info.items()}
        else:
            info = {}
        return decode_all(g_s, g_th, g_dt), info

    out = {"ckpt": a.ckpt, "seed": a.seed, "n_rows": int(B),
           "temps": [a.s_temp, a.th_temp, a.dt_temp], "q": "nodur", "arms": {}}
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
        print(f"  {arm:>4}    {r['auc_rf_oob']:.4f} {len(Fm):6d}     "
              f"{str(bool(r['collapse_flag'])):>6}  {info}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  wrote {a.out}")
    print("  one trajectory per spec, no selection, no headline from this arm")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--d", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)
    g = sub.add_parser("gen")
    g.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    g.add_argument("--n", type=int, default=2000)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--batch", type=int, default=200)
    g.add_argument("--arms", default="k0,n1,nT")
    g.add_argument("--s-temp", type=float, default=0.95)
    g.add_argument("--th-temp", type=float, default=0.90)
    g.add_argument("--dt-temp", type=float, default=1.00)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_gen)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
