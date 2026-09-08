"""w4_q2serve, AMENDMENT 68 in step0_prereg.md. Read that first.

Per seed: one spec set from make_specs, matched durations, TWO arms on
identical specs, identical durations, identical per batch AR seeds:
mq1 = the serve, event 0 from training/w4_firsthead_q.pt, mq2 = event 0
from training/w4_firsthead_q2.pt, the coupled head. Contract scorer on
the official row order and on ten row orders. One trajectory per spec,
no selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

import experiments.event_stream_polar as esp                      # noqa: E402
import scoring                                                    # noqa: E402
from features import extract_feature_matrix                       # noqa: E402
from models.event_ar import (DT_MAX_MS, EventARModel,             # noqa: E402
                             class_to_dt_ms)
from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS  # noqa: E402
from phase_a_baseline import make_specs                           # noqa: E402
from w4_firsthead import (MAX_T, FirstHead, Q_PATH,               # noqa: E402
                          splits)

Q2_PATH = "training/w4_firsthead_q2.pt"


def matched_log_durations(spec_log_d, seed, k=64):
    """The durmatch matcher, verbatim. Empirical p(log dur | log dist) from
    held out human rows: for each spec, sample one of the k nearest pool
    rows by |log dist difference| and take its log duration verbatim. One
    draw per spec, shared across arms."""
    lengths, _, held = splits()
    pool = held[lengths[held] > 4]
    cond = np.load("training/events_cond.npy")[:, :4].astype(np.float32)
    pd, pdur = cond[pool, 0], cond[pool, 1]
    order = np.argsort(pd, kind="stable")
    pd, pdur = pd[order], pdur[order]
    rng = np.random.default_rng(9000 + seed)
    out = np.empty(len(spec_log_d), dtype=np.float64)
    for i, ld in enumerate(spec_log_d):
        j = np.searchsorted(pd, ld)
        lo, hi = max(0, j - k), min(len(pd), j + k)
        cand = np.arange(lo, hi)
        near = cand[np.argsort(np.abs(pd[cand] - ld), kind="stable")[:k]]
        out[i] = pdur[near[rng.integers(len(near))]]
    print(f"  M duration pool {len(pd):,} held out rows (lengths > 4), "
          f"k={k}, rng 9000+{seed}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--s-temp", type=float, default=0.95)
    ap.add_argument("--th-temp", type=float, default=0.90)
    ap.add_argument("--dt-temp", type=float, default=1.00)
    ap.add_argument("--out", default=None)
    ap.add_argument("--npz", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = f"research/w4_q2serve_s{a.seed}.json"
    if a.npz is None:
        a.npz = f"research/w4_q2serve_s{a.seed}.npz"

    dev = esp._DEVICE
    m_ar = hashlib.md5(open(f"training/{a.ckpt}", "rb").read()).hexdigest()
    m_q = hashlib.md5(open(Q_PATH, "rb").read()).hexdigest()
    m_q2 = hashlib.md5(open(Q2_PATH, "rb").read()).hexdigest()
    print(f"  md5 {a.ckpt} {m_ar}  w4_firsthead_q.pt {m_q}  "
          f"w4_firsthead_q2.pt {m_q2}", flush=True)
    ck = torch.load(f"training/{a.ckpt}", map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    qk = torch.load(Q_PATH, map_location=dev, weights_only=False)
    q = FirstHead(**qk["config"]).to(dev).eval()
    q.load_state_dict(qk["model_state_dict"])
    qk2 = torch.load(Q2_PATH, map_location=dev, weights_only=False)
    q2 = FirstHead(**qk2["config"]).to(dev).eval()
    q2.load_state_dict(qk2["model_state_dict"])
    print(f"  {a.ckpt} step {ck.get('step')}  q best epoch {qk['best']['epoch']}"
          f"  q2 best epoch {qk2['best']['epoch']}"
          f"  AR temps s {a.s_temp} th {a.th_temp} dt {a.dt_temp}", flush=True)

    # One spec set, matched durations only. No protocol duration draw: both
    # arms condition on the same M cond.
    geo, meta = [], []
    for sx, sy, ex, ey in make_specs(a.n, a.seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        geo.append([math.log(dist), math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ang))
    geo = np.asarray(geo, dtype=np.float64)
    mdur = matched_log_durations(geo[:, 0], a.seed)
    B = len(geo)
    query = np.stack([geo[:, 0], mdur, geo[:, 1], geo[:, 2]], 1).astype(np.float64)
    cond = torch.tensor(query, dtype=torch.float32)
    print(f"  specs {B} of {a.n}, seed {a.seed}", flush=True)

    def decode_all(s_np, th_np, dtc_np):
        paths = []
        for i in range(len(s_np)):
            dd = class_to_dt_ms(torch.from_numpy(dtc_np[i])).numpy()
            dz = (np.log(np.maximum(dd, 0.05)) - esp._DT_MEAN) / esp._DT_STD
            sx, sy, ang = meta[i]
            p = esp._decode(dz, s_np[i], th_np[i], sx, sy, ang)
            paths.append(np.asarray(p, dtype=np.float64) if p is not None
                         else None)
        return paths

    def generate(label, e0):
        g_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
        g_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
        g_dt = np.zeros((B, MAX_T), dtype=np.int64)
        info = {"e0_tick_frac": 0.0, "e0_pad_frac": 0.0}
        for c0 in range(0, B, a.batch):
            sl = slice(c0, min(c0 + a.batch, B))
            cb = cond[sl].to(dev)
            nb = cb.shape[0]
            fs = torch.full((nb, MAX_T), S_PAD_CLASS, device=dev, dtype=torch.long)
            fth = torch.full((nb, MAX_T), TH_NULL_CLASS, device=dev, dtype=torch.long)
            fdt = torch.zeros((nb, MAX_T), device=dev, dtype=torch.long)
            qh = q if e0 == "q" else q2
            torch.manual_seed(a.seed * 100003 + c0 + 7)
            qs, qth, qdt = qh.sample(cb, 1.0, 1.0, 1.0)
            fs[:, 0], fth[:, 0], fdt[:, 0] = qs, qth, qdt.clamp(max=DT_MAX_MS)
            mask = torch.zeros((nb, MAX_T), device=dev, dtype=torch.bool)
            mask[:, 0] = True
            f = (fs, fth, fdt, mask)
            info["e0_tick_frac"] += float((fs[:, 0] == 0).sum())
            info["e0_pad_frac"] += float((fs[:, 0] >= S_PAD_CLASS).sum())
            torch.manual_seed(a.seed * 100003 + c0)
            with torch.no_grad():
                s_o, th_o, dt_o = model.sample(cb, temperature=a.s_temp,
                                               th_temperature=a.th_temp,
                                               dt_temperature=a.dt_temp,
                                               force=f, kv_cache=True)
            n_got = s_o.shape[1]
            g_s[sl, :n_got] = s_o.cpu().numpy()
            g_th[sl, :n_got] = th_o.cpu().numpy()
            g_dt[sl, :n_got] = dt_o.cpu().numpy()
            if c0 % (a.batch * 20) == 0:
                print(f"    {label}  {c0}/{B}", flush=True)
        info = {k: v / B for k, v in info.items()}
        return g_s, g_th, g_dt, info

    arms, kept_all = {}, {}
    print(f"\n  {'arm':>5}{'contract':>10}{'ten mean':>10}{'n':>7}{'collapse':>10}",
          flush=True)
    for label, e0 in (("mq1", "q"), ("mq2", "q2")):
        g_s, g_th, g_dt, info = generate(label, e0)
        paths = decode_all(g_s, g_th, g_dt)
        kept = [i for i, p in enumerate(paths) if p is not None]
        Fk = extract_feature_matrix([paths[i] for i in kept])
        F_full = np.full((B, Fk.shape[1]), np.nan)
        F_full[kept] = Fk
        Fm = F_full[np.all(np.isfinite(F_full), 1)]
        r = scoring.score_features(Fm[np.random.default_rng(a.seed)
                                      .permutation(len(Fm))])
        ten_order = [float(scoring.score_features(
            Fm[np.random.default_rng(a.seed * 7919 + j)
               .permutation(len(Fm))])["auc_rf_oob"]) for j in range(10)]
        arms[label] = dict(
            contract=float(r["auc_rf_oob"]), ten_order=ten_order,
            ten_mean=float(np.mean(ten_order)), n=int(len(Fm)),
            collapse=bool(r["collapse_flag"]),
            collapse_features=list(r["collapse_features"]),
            len_p50=float(np.median([len(paths[i]) for i in kept])),
            e0_tick_frac=info["e0_tick_frac"], e0_pad_frac=info["e0_pad_frac"])
        kept_all[label] = (g_s, g_th, g_dt, F_full)
        print(f"  {label:>5}  {arms[label]['contract']:.4f}   "
              f"{arms[label]['ten_mean']:.4f} {arms[label]['n']:6d}     "
              f"{str(arms[label]['collapse']):>6}  {info}", flush=True)

    out = {"ckpt": a.ckpt, "seed": a.seed, "n_rows": int(B),
           "temps": [a.s_temp, a.th_temp, a.dt_temp],
           "md5": {"event_ar_hm_mlp.pt": m_ar, "w4_firsthead_q.pt": m_q,
                   "w4_firsthead_q2.pt": m_q2},
           "kv_cache": True,
           "q2": {"path": Q2_PATH, "md5": m_q2,
                  "best_epoch": qk2["best"]["epoch"], "config": qk2["config"]},
           "arms": arms,
           "diff": {"contract": arms["mq2"]["contract"] - arms["mq1"]["contract"],
                    "ten_mean": arms["mq2"]["ten_mean"] - arms["mq1"]["ten_mean"]}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  wrote {a.out}")
    np.savez_compressed(
        a.npz, cond=query,
        s_mq1=kept_all["mq1"][0], th_mq1=kept_all["mq1"][1],
        dt_mq1=kept_all["mq1"][2], F_mq1=kept_all["mq1"][3],
        s_mq2=kept_all["mq2"][0], th_mq2=kept_all["mq2"][1],
        dt_mq2=kept_all["mq2"][2], F_mq2=kept_all["mq2"][3])
    print(f"  wrote {a.npz}")
    print(f"  mq2 minus mq1  contract {out['diff']['contract']:+.4f}  "
          f"ten order {out['diff']['ten_mean']:+.4f}")
    print(f"  md5 {a.ckpt} {m_ar}  w4_firsthead_q.pt {m_q}  "
          f"w4_firsthead_q2.pt {m_q2}")
    print("  one trajectory per spec, no selection, no headline from this arm")


if __name__ == "__main__":
    main()
