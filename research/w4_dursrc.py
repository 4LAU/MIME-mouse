"""w4_dursrc, AMENDMENT 18 in step0_prereg.md. Read that first.

Duration source and the q effect, inside one run. Per seed: one spec set
from make_specs; TWO log duration vectors drawn once each, P = protocol
draw (esp._duration.sample per spec, exactly the confirm) and M = NN
matched draw (the durmatch 64 NN matcher over held out rows, rng
9000+seed); FOUR arms on identical specs: P-k0, P-q1, M-k0, M-q1 (full q
from training/w4_firsthead_q.pt, qT dropped). Same per batch torch seeds
across arms and across duration sources, so P-q1 and M-q1 share q draws
where conds allow. Seeds 15 to 19, n 2000, batch 200. Contract scorer
after shuffle, per arm.

THIS ARM PRODUCES NO HEADLINE AND NO SERVE DECISION. Reads, registered in
AMENDMENT 18: (1) PRIMARY interaction D = (M-q1 minus M-k0) minus (P-q1
minus P-k0), D <= -0.010 at 2 se MODULATION REAL, |D| <= 0.005 A DRAW,
else BETWEEN; (2) do the per source q effects reproduce durmatch -0.0215
and confirm +0.0012, descriptive; (3) P-k0 minus M-k0 paired.
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

    # One spec set. TWO duration vectors, each drawn once, shared by every
    # arm that uses that source. P consumes esp._duration's own rng inside
    # the loop, exactly as the confirm did.
    geo, meta, pdur = [], [], []
    for sx, sy, ex, ey in make_specs(a.n, a.seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        geo.append([math.log(dist), math.cos(ang), math.sin(ang)])
        pdur.append(math.log(esp._duration.sample(math.log(dist))))
        meta.append((sx, sy, ang))
    geo = np.asarray(geo, dtype=np.float64)
    pdur = np.asarray(pdur, dtype=np.float64)
    mdur = matched_log_durations(geo[:, 0], a.seed)
    B = len(geo)
    conds = {
        "P": torch.tensor(np.stack([geo[:, 0], pdur, geo[:, 1], geo[:, 2]], 1),
                          dtype=torch.float32),
        "M": torch.tensor(np.stack([geo[:, 0], mdur, geo[:, 1], geo[:, 2]], 1),
                          dtype=torch.float32),
    }
    print(f"  specs {B} of {a.n}, seed {a.seed}", flush=True)
    print(f"  log dur medians  P {np.median(pdur):+.4f}  M {np.median(mdur):+.4f}",
          flush=True)

    QT = {"k0": None, "q1": (1.0, 1.0, 1.0)}

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

    def generate(cond_t, qt, label):
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
                print(f"    {label}  {c0}/{B}", flush=True)
        if nq:
            info = {k: v / nq for k, v in info.items()}
        else:
            info = {}
        return decode_all(g_s, g_th, g_dt), info

    out = {"ckpt": a.ckpt, "seed": a.seed, "n_rows": int(B),
           "temps": [a.s_temp, a.th_temp, a.dt_temp], "arms": {}}
    print(f"\n  {'arm':>6}{'contract':>10}{'n':>7}{'collapse':>10}", flush=True)
    for src in ("P", "M"):
        for qarm in ("k0", "q1"):
            label = f"{src}-{qarm}"
            paths, info = generate(conds[src], QT[qarm], label)
            Fm = extract_feature_matrix(paths)
            Fm = Fm[np.all(np.isfinite(Fm), 1)]
            Fm = Fm[np.random.default_rng(a.seed).permutation(len(Fm))]
            r = scoring.score_features(Fm)
            out["arms"][label] = dict(
                contract=float(r["auc_rf_oob"]), n=int(len(Fm)),
                collapse=bool(r["collapse_flag"]),
                collapse_features=list(r["collapse_features"]),
                len_p50=float(np.median([len(p) for p in paths])), **info)
            print(f"  {label:>6}    {r['auc_rf_oob']:.4f} {len(Fm):6d}     "
                  f"{str(bool(r['collapse_flag'])):>6}  {info}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  wrote {a.out}")
    print("  one trajectory per spec, no selection, no headline from this arm")


if __name__ == "__main__":
    main()
