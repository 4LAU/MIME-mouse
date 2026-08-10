"""A duration conditional tilt on the timing head, at generation.

PRE REGISTERED in HANDOFF.md 2026-08-07, "## The generation time timing tilt".
The calibrate then test order, the reversed sign placebo, the branch thresholds
and the prediction of NO EFFECT or under 0.02 were all fixed before this ran.

    dt logit bias = lambda * (log D - mean log D) * (class index / max - 0.5)

Reads only `cond`, emits one trajectory per spec, selects nothing. lambda zero
is exactly the served path.

Safety. Scores through research/autoloop/scoring.py only. Touches no evaluation
data directly, never modifies scoring code, never
training/candi_polar_flow_best.pt.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_dttilt.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import extract_feature_matrix  # noqa: E402
from models.event_ar import EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS  # noqa: E402
from w4_evprice import build_specs  # noqa: E402

HUMAN_DT_SLOPE = 0.6023      # w4_evcount, real held out streams
HUMAN_CNT_SLOPE = 0.3974
LAMBDAS = (-4.0, -2.0, 0.0, 2.0, 4.0)


def run_one(model, rows, meta, mu, lam, batch, temp, dev):
    """One trajectory per spec at this lambda. No selection."""
    paths, nev, mdt, cn = [], [], [], []
    for c0 in range(0, len(rows), batch):
        blk = rows[c0:c0 + batch]
        cond = torch.tensor(blk, dtype=torch.float32, device=dev)
        tilt = None
        if lam != 0.0:
            tilt = lam * (cond[:, 1] - mu)
        s_cls, th_cls, dt_cls = model.sample(cond, temperature=temp,
                                             dt_tilt=tilt)
        pad = (s_cls >= S_PAD_CLASS).cpu().numpy()
        dt_ms = class_to_dt_ms(dt_cls)
        dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
                / esp._DT_STD).cpu().numpy()
        dtn = dt_ms.cpu().numpy()
        s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
        for j in range(s_np.shape[0]):
            sx, sy, ang = meta[c0 + j]
            p = esp._decode(dt_z[j], s_np[j], th_np[j], sx, sy, ang)
            L = int(pad[j].argmax()) if pad[j].any() else s_np.shape[1]
            if p is None or L < 2:
                continue
            paths.append(np.asarray(p, dtype=np.float64))
            nev.append(L)
            mdt.append(float(dtn[j, :L].mean()) / 1000.0)
            cn.append(blk[j])
    return paths, np.array(nev, float), np.array(mdt, float), np.array(cn, float)


def slope(y, cnd):
    X = np.column_stack([np.ones(len(cnd)), cnd[:, 0], cnd[:, 1]])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambdas", default=None)
    ap.add_argument("--out", default="research/w4_dttilt.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    rows, meta = build_specs(args.n, args.seed)
    mu = float(np.mean([r[1] for r in rows]))
    lams = ([float(x) for x in args.lambdas.split(",")] if args.lambdas
            else list(LAMBDAS))
    print(f"\n  {len(rows)} specs, mean log duration {mu:.4f}")
    print(f"  human dt slope {HUMAN_DT_SLOPE:.4f}, count slope "
          f"{HUMAN_CNT_SLOPE:.4f}\n")
    print(f"  {'lambda':>8}{'dt slope':>11}{'cnt slope':>11}{'contract':>11}"
          f"{'n_ev p50':>10}{'n':>7}")

    out = {}
    with torch.no_grad():
        for lam in lams:
            paths, nev, mdt, cnd = run_one(model, rows, meta, mu, lam,
                                           args.batch, args.temp, dev)
            F = extract_feature_matrix(paths)
            F = F[np.all(np.isfinite(F), 1)]
            auc = float(scoring.score_features(F)["auc_rf_oob"])
            sd = slope(np.log(np.maximum(mdt, 1e-6)), cnd)
            sc = slope(np.log(np.maximum(nev, 1)), cnd)
            out[f"lam{lam}"] = {"lambda": lam, "dt_slope": sd,
                               "count_slope": sc, "contract": auc,
                               "n_ev_p50": float(np.median(nev)), "n": len(F)}
            print(f"  {lam:>8.2f}{sd:>11.4f}{sc:>11.4f}{auc:>11.4f}"
                  f"{np.median(nev):>10.0f}{len(F):>7}", flush=True)

    ls = np.array([out[k]["lambda"] for k in out])
    sds = np.array([out[k]["dt_slope"] for k in out])
    aucs = np.array([out[k]["contract"] for k in out])
    o = np.argsort(ls)
    lam_star = float(np.interp(HUMAN_DT_SLOPE, sds[o], ls[o]))
    base = float(aucs[ls == 0.0][0]) if (ls == 0.0).any() else float("nan")
    print(f"\n  calibrated lambda, dt slope {HUMAN_DT_SLOPE:.4f} reached at "
          f"lambda {lam_star:+.3f}")
    print(f"  chosen WITHOUT looking at the contract column, as registered")
    print(f"  baseline at lambda 0   {base:.4f}")
    out["lambda_star"] = lam_star
    out["baseline"] = base
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")
    print(f"  next: confirm at lambda {lam_star:+.3f} and at "
          f"{-lam_star:+.3f} as the placebo\n")


if __name__ == "__main__":
    main()
