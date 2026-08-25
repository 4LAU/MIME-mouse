"""w4_qladder. Is the k=4 ladder already cashed by the served q event 0.
AMENDMENT 22 in /home/aaronadmin/w4_arms/step0_prereg.md, registered before
this file was written.

The w4_firstev row machinery byte for byte (held out rows, rng(1000+seed)
pick, eligibility length > 4, row's own condition, served temps) with four
arms: k0 free running, q0 (q event 0 forced, temps 1,1,1, torch offset +7),
q0h13 (q event 0 plus the row's human events 1 to 3), h04 (human events 0
to 3, the ladder's k=4 arm in run). Same per batch AR torch seed across
arms; the q draw is shared by both q arms. One trajectory per row, no
selection, shuffle before score_features. No headline, no serve decision.

Reads training/events_*.npy and checkpoints, never the protected eval
file, never candi_polar_flow_best.pt.
"""
from __future__ import annotations

import argparse
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

import experiments.event_stream_polar as esp                      # noqa: E402
import scoring                                                    # noqa: E402
from features import extract_feature_matrix                       # noqa: E402
from models.event_ar import (DT_MAX_MS, EventARModel,             # noqa: E402
                             class_to_dt_ms, dt_ms_to_class)
from models.event_stream_polar import (S_PAD_CLASS, TH_NULL_CLASS,  # noqa: E402
                                       dth_lattice_to_class, s2_to_class)
from w4_firsthead import FirstHead, Q_PATH                        # noqa: E402
from w4_pairq import Pair1, P1_PATH                               # noqa: E402

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256
KMAX = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--s-temp", type=float, default=0.95)
    ap.add_argument("--th-temp", type=float, default=0.90)
    ap.add_argument("--dt-temp", type=float, default=1.00)
    ap.add_argument("--arms", default="k0,q0,q0h13,h04")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = esp._DEVICE
    lengths = np.load("training/events_len.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    rng = np.random.default_rng(1000 + a.seed)
    elig = held[lengths[held] > KMAX]
    pick = np.sort(rng.choice(elig, a.n, replace=False))
    print(f"  corpus {N:,}, held out {len(held):,}, eligible {len(elig):,}, "
          f"using {a.n:,}, seed {a.seed}", flush=True)

    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    dt_ms = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    B = len(L)

    real_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
    real_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
    real_dt = np.zeros((B, MAX_T), dtype=np.int64)
    sc = s2_to_class(torch.from_numpy(np.asarray(s2, dtype=np.int64))).numpy()
    tc = np.where(np.asarray(s2) > 0,
                  dth_lattice_to_class(
                      torch.from_numpy(np.asarray(dth, dtype=np.int64))).numpy(),
                  TH_NULL_CLASS)
    dc = dt_ms_to_class(torch.from_numpy(dt_ms)).numpy()
    for i in range(B):
        n = int(L[i])
        real_s[i, :n] = sc[i, :n]
        real_th[i, :n] = tc[i, :n]
        real_dt[i, :n] = dc[i, :n].clip(0, DT_MAX_MS)

    cond_t = torch.from_numpy(conds[:, :4].astype(np.float32))
    angs = np.arctan2(conds[:, 3].astype(np.float64),
                      conds[:, 2].astype(np.float64))

    ck = torch.load(f"training/{a.ckpt}", map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    qk = torch.load(Q_PATH, map_location=dev, weights_only=False)
    q = FirstHead(**qk["config"]).to(dev).eval()
    q.load_state_dict(qk["model_state_dict"])
    print(f"  {a.ckpt} step {ck.get('step')}  q best epoch {qk['best']['epoch']}"
          f"  AR temps s {a.s_temp} th {a.th_temp} dt {a.dt_temp}", flush=True)

    def decode_all(s_np, th_np, dtc_np):
        paths = []
        for i in range(len(s_np)):
            d = class_to_dt_ms(torch.from_numpy(dtc_np[i])).numpy()
            dz = (np.log(np.maximum(d, 0.05)) - esp._DT_MEAN) / esp._DT_STD
            p = esp._decode(dz, s_np[i], th_np[i], 0.0, 0.0, float(angs[i]))
            if p is not None and len(p) >= 4:
                paths.append(np.asarray(p, dtype=np.float64))
        return paths

    # Arms: (label, e0src, human_positions). e0src None = free at 0 unless
    # 0 in human_positions, "q" = q draw, "wrong" = donor row (i+1) mod n
    # human event 0 (AMENDMENT 23). Eligibility guarantees the human
    # tokens at positions 0..3 all exist.
    ALL_ARMS = (("k0", None, ()),
                ("q0", "q", ()),
                ("q0h13", "q", (1, 2, 3)),
                ("h04", None, (0, 1, 2, 3)),
                ("wr0h13", "wrong", (1, 2, 3)),
                ("warm4", "warm", ()),
                ("h02", None, (0, 1)),
                ("h03", None, (0, 1, 2)),
                ("q0w1", "qwarm", ()),
                ("p01", "pair", ()),
                ("h01", None, (0,)),
                ("h0p1", "hpair", ()),
                ("h0s1p", "hchan_s", ()),
                ("h0st1p", "hchan_st", ()),
                ("r01", "rpair", ()))
    want = a.arms.split(",")
    assert all(w in {x[0] for x in ALL_ARMS} for w in want), want
    ARMS = tuple(x for x in ALL_ARMS if x[0] in want)
    donor = (np.arange(B) + 1) % B

    pair = None
    if any(x[1] in ("pair", "hpair", "hchan_s", "hchan_st") for x in ARMS):
        pk = torch.load(P1_PATH, map_location=dev, weights_only=False)
        pair = Pair1(**pk["config"]).to(dev).eval()
        pair.load_state_dict(pk["model_state_dict"])
        print(f"  q1g0 best epoch {pk['best']['epoch']}", flush=True)
    rpair = None
    if any(x[1] == "rpair" for x in ARMS):
        rk = torch.load("training/w4_pairadv.pt", map_location=dev, weights_only=False)
        rpair = Pair1(**rk["config"]).to(dev).eval()
        rpair.load_state_dict(rk["model_state_dict"])
        print(f"  refined pair best epoch {rk['best']['epoch']} lam {rk['lam']}",
              flush=True)

    def generate(label, e0src, hpos):
        g_s = np.full((B, MAX_T), S_PAD_CLASS, dtype=np.int64)
        g_th = np.full((B, MAX_T), TH_NULL_CLASS, dtype=np.int64)
        g_dt = np.zeros((B, MAX_T), dtype=np.int64)
        for c0 in range(0, B, a.batch):
            sl = slice(c0, min(c0 + a.batch, B))
            cb = cond_t[sl].to(dev)
            nb = cb.shape[0]
            f = None
            if e0src in ("q", "qwarm", "wrong", "pair", "rpair", "hpair",
                         "hchan_s", "hchan_st") or hpos:
                fs = torch.from_numpy(real_s[sl]).to(dev).clone()
                fth = torch.from_numpy(real_th[sl]).to(dev).clone()
                fdt = torch.from_numpy(real_dt[sl]).to(dev).clone()
                mask = torch.zeros((nb, MAX_T), device=dev, dtype=torch.bool)
                for p in hpos:
                    mask[:, p] = True
                if e0src in ("q", "qwarm"):
                    torch.manual_seed(a.seed * 100003 + c0 + 7)
                    qs, qth, qdt = q.sample(cb, 1.0, 1.0, 1.0)
                    fs[:, 0], fth[:, 0] = qs, qth
                    fdt[:, 0] = qdt.clamp(max=DT_MAX_MS)
                    mask[:, 0] = True
                elif e0src in ("pair", "rpair"):
                    pm = pair if e0src == "pair" else rpair
                    torch.manual_seed(a.seed * 100003 + c0 + 7)
                    qs, qth, qdt = q.sample(cb, 1.0, 1.0, 1.0)
                    qdt = qdt.clamp(max=DT_MAX_MS)
                    torch.manual_seed(a.seed * 100003 + c0 + 13)
                    ps1, pth1, pdt1 = pm.sample(cb, qs, qth, qdt, 1.0, 1.0, 1.0)
                    fs[:, 0], fth[:, 0], fdt[:, 0] = qs, qth, qdt
                    fs[:, 1], fth[:, 1] = ps1, pth1
                    fdt[:, 1] = pdt1.clamp(max=DT_MAX_MS)
                    mask[:, 0], mask[:, 1] = True, True
                elif e0src == "hpair":
                    mask[:, 0] = True
                    torch.manual_seed(a.seed * 100003 + c0 + 13)
                    ps1, pth1, pdt1 = pair.sample(cb, fs[:, 0], fth[:, 0],
                                                  fdt[:, 0], 1.0, 1.0, 1.0)
                    fs[:, 1], fth[:, 1] = ps1, pth1
                    fdt[:, 1] = pdt1.clamp(max=DT_MAX_MS)
                    mask[:, 1] = True
                elif e0src in ("hchan_s", "hchan_st"):
                    mask[:, 0] = True
                    torch.manual_seed(a.seed * 100003 + c0 + 13)
                    kw = dict(s1=fs[:, 1].clone())
                    if e0src == "hchan_st":
                        kw["th1"] = fth[:, 1].clone()
                    ps1, pth1, pdt1 = pair.sample_completing(
                        cb, fs[:, 0], fth[:, 0], fdt[:, 0], **kw)
                    fs[:, 1], fth[:, 1] = ps1, pth1
                    fdt[:, 1] = pdt1.clamp(max=DT_MAX_MS)
                    mask[:, 1] = True
                elif e0src == "wrong":
                    d_idx = donor[np.arange(c0, min(c0 + a.batch, B))]
                    fs[:, 0] = torch.from_numpy(real_s[d_idx, 0]).to(dev)
                    fth[:, 0] = torch.from_numpy(real_th[d_idx, 0]).to(dev)
                    fdt[:, 0] = torch.from_numpy(real_dt[d_idx, 0]).to(dev)
                    mask[:, 0] = True
                f = (fs, fth, fdt, mask)
            if e0src in ("warm", "qwarm"):
                W = 4 if e0src == "warm" else 2
                tmp_s = [1.0] * W + [a.s_temp] * (MAX_T - W)
                tmp_th = [1.0] * W + [a.th_temp] * (MAX_T - W)
                tmp_dt = [1.0] * W + [a.dt_temp] * (MAX_T - W)
            else:
                tmp_s, tmp_th, tmp_dt = a.s_temp, a.th_temp, a.dt_temp
            torch.manual_seed(a.seed * 100003 + c0)
            with torch.no_grad():
                s_o, th_o, dt_o = model.sample(cb, temperature=tmp_s,
                                               th_temperature=tmp_th,
                                               dt_temperature=tmp_dt, force=f)
            n_got = s_o.shape[1]
            g_s[sl, :n_got] = s_o.cpu().numpy()
            g_th[sl, :n_got] = th_o.cpu().numpy()
            g_dt[sl, :n_got] = dt_o.cpu().numpy()
            if c0 % (a.batch * 5) == 0:
                print(f"    {label}  {c0}/{B}", flush=True)
        return decode_all(g_s, g_th, g_dt)

    out = {"ckpt": a.ckpt, "seed": a.seed, "n_rows": int(B),
           "temps": [a.s_temp, a.th_temp, a.dt_temp], "arms": {}}
    print(f"\n  {'arm':>6}{'contract':>10}{'n':>7}{'collapse':>10}", flush=True)
    for label, e0src, hpos in ARMS:
        paths = generate(label, e0src, hpos)
        F = extract_feature_matrix(paths)
        F = F[np.all(np.isfinite(F), 1)]
        F = F[np.random.default_rng(a.seed).permutation(len(F))]
        r = scoring.score_features(F)
        out["arms"][label] = dict(contract=float(r["auc_rf_oob"]),
                                  n=int(len(F)),
                                  collapse=bool(r["collapse_flag"]),
                                  collapse_features=list(r["collapse_features"]))
        print(f"  {label:>6}    {r['auc_rf_oob']:.4f} {len(F):6d}     "
              f"{str(bool(r['collapse_flag'])):>6}  "
              f"{ {k: round(v, 4) for k, v in r.get('collapse_features', {}).items()} if isinstance(r.get('collapse_features'), dict) else list(r['collapse_features']) }",
              flush=True)

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {a.out}")
    print("  one trajectory per row, no selection, no headline from this arm")


if __name__ == "__main__":
    main()
