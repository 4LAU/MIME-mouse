"""w4_durmatch, AMENDMENT 16 in step0_prereg.md. Read that first.

The AMENDMENT 15 confirm repeated exactly, with ONE ingredient changed: each
spec's log duration comes from the empirical conditional of held out human
rows (64 nearest by |log dist difference|, one sampled uniformly,
default_rng(9000 + seed), drawn once and shared across arms) instead of the
fitted marginal esp._duration.sample(log d). Everything else is byte for
byte the confirm: seeds 10 to 14, n 2000, batch 200, arms k0 / q1 / qT, the
same per-batch torch seeds, decode, shuffle, contract scorer.

THIS ARM PRODUCES NO HEADLINE AND NO SERVE DECISION. Reads, registered in
AMENDMENT 16: (1) descriptive k0 shift vs the confirm's k0 per seed;
(2) PRIMARY paired q1 minus k0, <= -0.008 at 2 paired se -> H_dur, within
0.004 -> H_small, else BETWEEN; (3) qT minus k0 informational.
"""
from __future__ import annotations

import argparse
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


def matched_log_durations(spec_log_d, seed, k=64):
    """Empirical p(log dur | log dist) from held out human rows: for each
    spec, sample one of the k nearest pool rows by |log dist difference| and
    take its log duration verbatim. One draw per spec, shared across arms."""
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
    print(f"  duration pool {len(pd):,} held out rows (lengths > 4), "
          f"k={k}, rng 9000+{seed}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--arms", default="k0,q1,qT")
    ap.add_argument("--s-temp", type=float, default=0.95)
    ap.add_argument("--th-temp", type=float, default=0.90)
    ap.add_argument("--dt-temp", type=float, default=1.00)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{a.ckpt}", map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    qk = torch.load(Q_PATH, map_location=dev, weights_only=False)
    q = FirstHead(**qk["config"]).to(dev).eval()
    q.load_state_dict(qk["model_state_dict"])
    print(f"  {a.ckpt} step {ck.get('step')}  q best epoch {qk['best']['epoch']}"
          f"  AR temps s {a.s_temp} th {a.th_temp} dt {a.dt_temp}", flush=True)

    # Same specs as the confirm (make_specs is deterministic in n, seed).
    # THE ONE CHANGE: log duration from matched held out rows, not from
    # esp._duration.sample.
    geo, meta = [], []
    for sx, sy, ex, ey in make_specs(a.n, a.seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        geo.append([math.log(dist), math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ang))
    geo = np.asarray(geo, dtype=np.float64)
    ldur = matched_log_durations(geo[:, 0], a.seed)
    rows = np.stack([geo[:, 0], ldur, geo[:, 1], geo[:, 2]], axis=1)
    B = len(rows)
    cond_t = torch.tensor(rows, dtype=torch.float32)
    print(f"  specs {B} of {a.n}, seed {a.seed}", flush=True)

    ARMS = {"k0": None, "q1": (1.0, 1.0, 1.0), "qT": (a.s_temp, a.th_temp, a.dt_temp)}

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
                qs, qth, qdt = q.sample(cb, *qt)
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
           "temps": [a.s_temp, a.th_temp, a.dt_temp], "durations": "matched",
           "arms": {}}
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


if __name__ == "__main__":
    main()
