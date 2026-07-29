"""P2 diagnostic: does the character latent CONTROL realized character?

Bypasses the prior entirely. Encode real paths whose max velocity is in the
top decile, and real paths from the middle of the distribution, with the
trained CharEncoder. Generate one path per latent (same displacement spec
as the source path, feature command off) through the normal serving path,
and compare the realized tell features of the two groups.

If the extreme-z group's realized features match the median-z group's, the
latent does not control character and no amount of prior work can help. If
they separate, the bottleneck is the prior draw, which is fixable.

Usage:
    env PYTHONPATH=. ~/venvs/mime/bin/python research/w3_p2_probe_zcontrol.py \
        --ckpt event_polar_4m_char_v3.pt --per-group 200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))

TELL = {"max_velocity": 2, "max_acceleration": 6,
        "mean_jerk": 7, "std_jerk": 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--per-group", type=int, default=200)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    from training.train_events_polar_char import CharEncoder
    from training.train_events_polar import PolarEventDataset
    from training.train_events_polar_dm import (
        build_value_tables, detector_features, real_batch_values,
        stream_to_frames)
    from torch.utils.data import DataLoader

    ck = torch.load(R / "training" / args.ckpt, map_location=device,
                    weights_only=False)
    cfg = ck["config"]
    enc = CharEncoder(ck["z_dim"], max_seq_len=cfg["max_seq_len"]).to(device)
    enc.load_state_dict(ck["encoder_state_dict"])
    enc.eval()

    tdir = R / "training"
    s2 = np.load(tdir / "events_s2.npy", mmap_mode="r")
    dth = np.load(tdir / "events_dth.npy", mmap_mode="r")
    dt = np.load(tdir / "events_dt.npy", mmap_mode="r")
    ln = np.load(tdir / "events_len.npy")
    cd = np.load(tdir / "events_cond.npy")
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(ln), args.pool, replace=False))
    ds = PolarEventDataset(s2[idx], dth[idx], dt[idx], ln[idx], cd[idx],
                           cfg["max_seq_len"], float(ck["dt_mean"]),
                           float(ck["dt_std"]))
    dl = DataLoader(ds, batch_size=512)
    tables = build_value_tables(device)

    feats_l, mus_l, conds_l = [], [], []
    with torch.no_grad():
        for b in dl:
            dt_z, s_cls, th_cls, real, cond = (x.to(device) for x in b)
            dt_s = torch.exp(dt_z * float(ck["dt_std"])
                             + float(ck["dt_mean"])).clamp(0.1, 1000.0) / 1000.0
            speed, motion, tick, cos_th, sin_th = real_batch_values(
                s_cls, th_cls, tables)
            x, y, fmask = stream_to_frames(speed, motion, cos_th, sin_th,
                                           dt_s, real, cond, 256)
            feats_l.append(detector_features(x, y, fmask).cpu())
            mu_q, _ = enc(s_cls, th_cls, dt_z, real)
            mus_l.append(mu_q.cpu())
            conds_l.append(cond.cpu())
    feats = torch.cat(feats_l).numpy()
    mus = torch.cat(mus_l).numpy()
    conds = torch.cat(conds_l).numpy()

    # keep paths long enough that character is meaningful
    dist = np.exp(conds[:, 0])
    ok = dist > 100
    mv = feats[:, TELL["max_velocity"]]
    hi_cut = np.quantile(mv[ok], 0.90)
    mid_lo, mid_hi = np.quantile(mv[ok], [0.40, 0.60])
    hi_idx = np.where(ok & (mv >= hi_cut))[0][:args.per_group]
    mid_idx = np.where(ok & (mv >= mid_lo) & (mv <= mid_hi))[0][:args.per_group]
    print(f"[select] extreme {len(hi_idx)} (mv>={hi_cut:.0f}), "
          f"median {len(mid_idx)} ({mid_lo:.0f}..{mid_hi:.0f})", flush=True)

    sel = np.concatenate([hi_idx, mid_idx])
    z = mus[sel]
    specs = []
    for i in sel:
        d = float(np.exp(conds[i, 0]))
        dx = round(d * float(conds[i, 2]))
        dy = round(d * float(conds[i, 3]))
        specs.append((0, 0, dx if dx or dy else 1, dy))

    zfile = str(R / "research" / "w3_p2_zcontrol_z.npz")
    np.savez(zfile, z=z.astype(np.float32))

    os.environ.update(EVENT_CKPT=args.ckpt, EVENT_ORDER="gumbel",
                      EVENT_SNAP="2.5", EVENT_DUR_STD="1.0", DUR_EMPIRICAL="1",
                      EVENT_CHOICE_TEMP="10", EVENT_BESTOF="1", EVENT_SIR="1",
                      EVENT_FEAT="0", EVENT_Z_FILE=zfile)
    for k in ("EVENT_Z_PRIOR", "EVENT_POOL_TOKENS", "EVENT_POOL_SAVE",
              "EVENT_POOL_LOAD"):
        os.environ.pop(k, None)

    from features import extract_feature_matrix
    from experiments import event_stream_polar
    raw = event_stream_polar.generate_paths(specs)
    good = [(g, t) for g, t in zip(
        ["hi"] * len(hi_idx) + ["mid"] * len(mid_idx), raw)
        if t is not None and len(t) >= 3]
    gen_feats = extract_feature_matrix([t for _, t in good])
    groups = np.array([g for g, _ in good])

    out = {"ckpt": args.ckpt, "per_group": args.per_group}
    for name, fi in TELL.items():
        hi_v = gen_feats[groups == "hi", fi]
        mid_v = gen_feats[groups == "mid", fi]
        src_hi = feats[hi_idx, fi]
        src_mid = feats[mid_idx, fi]
        # simple separation: P(hi draw > mid draw), 0.5 = no control
        a = (hi_v[:, None] > mid_v[None, :]).mean()
        out[name] = {
            "source_hi_mean": float(src_hi.mean()),
            "source_mid_mean": float(src_mid.mean()),
            "gen_hi_mean": float(hi_v.mean()),
            "gen_mid_mean": float(mid_v.mean()),
            "gen_separation_auc": float(a),
        }
        print(f"[{name}] source hi/mid {src_hi.mean():.0f}/{src_mid.mean():.0f}"
              f" -> generated hi/mid {hi_v.mean():.0f}/{mid_v.mean():.0f}"
              f" (sep AUC {a:.3f})", flush=True)

    out_path = args.out or str(R / "research" / "w3_p2_zcontrol_results.json")
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
