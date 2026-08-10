"""Which feature of the token history carries the state effect.

PRE REGISTERED in HANDOFF.md 2026-08-06, "## Which feature of the token history
carries the excess". Features, branch thresholds, the overlap warning and the
tick tension clause were all fixed before this file existed.

Same exact split as `w4_statecoord`, aimed at the only trunk input that run left
standing:

    C - B = sum_b (wgen[b] - wreal[b]) * Hbar_real[b]      BETWEEN
          + sum_b  wgen[b] * (Hbar_gen[b] - Hbar_real[b])  WITHIN

No generation. Reuses the streams saved by `w4_typpos`.

Safety. Reads the saved streams and one checkpoint. Touches no evaluation data,
never scoring.py, never training/candi_polar_flow_best.pt.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_histfeat.py
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
from models.event_ar import EventARModel, prefix_state, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    N_S_CLASSES, N_TH_CLASSES, S_PAD_CLASS, TICK_CLASS,
    class_to_dtheta, class_to_speed,
)
from w4_beta_curve import MAX_T  # noqa: E402
from w4_launch import N_REAL  # noqa: E402
from w4_statecoord import boot_between, decompose  # noqa: E402

TH_NULL_CLASS = N_REAL["th"]
FEATS = ("last speed", "last turn magnitude", "last inter event time",
         "last speed change", "mean speed last 5", "tick frac last 8",
         "events since tick", "turn persistence")


def _shift(x, k):
    """x delayed by k positions along time, zero filled. Strictly causal."""
    z = torch.zeros_like(x[:, :k])
    return torch.cat([z, x[:, :-k]], dim=1)


def hist_feats(s_cls, th_cls, dt_cls):
    """Eight causal history features, using only events strictly before t."""
    s = class_to_speed(s_cls.clamp(max=N_S_CLASSES - 1))
    dth = class_to_dtheta(th_cls.clamp(max=N_TH_CLASSES - 1))
    dts = class_to_dt_ms(dt_cls) / 1000.0
    motion = ((s_cls > TICK_CLASS) & (s_cls < S_PAD_CLASS)).float()
    tick = (s_cls == TICK_CLASS).float()
    sm = s * motion

    f0 = _shift(sm, 1)
    f1 = _shift(dth.abs() * motion, 1)
    f2 = _shift(dts, 1)
    f3 = (_shift(sm, 1) - _shift(sm, 2)).abs()
    f4 = sum(_shift(sm, k) for k in range(1, 6)) / 5.0
    f5 = sum(_shift(tick, k) for k in range(1, 9)) / 8.0
    # events since the last tick, causally, capped at the window used elsewhere
    run = torch.zeros_like(f0)
    seen = torch.zeros_like(f0)
    for k in range(1, 17):
        t = _shift(tick, k)
        first = (seen == 0) & (t > 0)
        run = torch.where(first, torch.full_like(run, float(k)), run)
        seen = seen + t
    f6 = torch.where(seen > 0, run, torch.full_like(run, 17.0))
    d1, d2 = _shift(dth * motion, 1), _shift(dth * motion, 2)
    f7 = torch.sign(d1) * torch.sign(d2)
    return torch.stack([f0, f1, f2, f3, f4, f5, f6, f7], dim=-1)


def entropy_and_feats(model, s, th, dt, cond, lens, batch, dev):
    """Per position entropy pooled over three heads, with its history features."""
    B = len(lens)
    H, F, R = [], [], []
    with torch.no_grad():
        for c0 in range(0, B, batch):
            sl = slice(c0, min(c0 + batch, B))
            s_b, th_b = s[sl].to(dev), th[sl].to(dev)
            dt_b, cnd = dt[sl].to(dev), cond[sl].to(dev)
            n = s_b.shape[0]
            s_p, th_p, dt_p = model.shift_inputs(s_b, th_b, dt_b)
            st = prefix_state(s_b, th_b, dt_b, cnd)
            lg_s, lg_th, lg_dt = model.forward(s_p, th_p, dt_p, st, cnd,
                                               s_b, th_b, dt_b)
            pos = torch.arange(MAX_T, device=dev).unsqueeze(0)
            Lb = torch.from_numpy(lens[sl]).to(dev).unsqueeze(1)
            live = pos < Lb
            live_th = live & (th_b < TH_NULL_CLASS)
            h = torch.zeros_like(live, dtype=torch.float64)
            for lg, msk in ((lg_s, live), (lg_th, live_th), (lg_dt, live)):
                ll = torch.log_softmax(lg.float(), dim=-1).double()
                e = -(ll.exp() * ll).sum(-1)
                h += torch.where(msk, e, torch.zeros_like(e))
            ft = hist_feats(s_b, th_b, dt_b)
            idx = torch.arange(c0, c0 + n, device=dev).unsqueeze(1).expand(-1, MAX_T)
            H.append(h[live].cpu().numpy())
            F.append(ft[live].float().cpu().numpy())
            R.append(idx[live].cpu().numpy())
    return np.concatenate(H), np.concatenate(F, axis=0), np.concatenate(R)


def joint_code(a, b, nbin=8):
    """Two features collapsed to one label on an 8 by 8 real quantile grid."""
    def cut(x, ref):
        e = np.unique(np.quantile(ref, np.linspace(0, 1, nbin + 1)))[1:-1]
        return np.digitize(x, e)
    return cut(a[0], a[1]) * nbin + cut(b[0], b[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--streams", default="research/w4_typpos_streams.npz")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_histfeat.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    z = np.load(args.streams)
    cond = torch.from_numpy(z["cond"])
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    rL = z["real_L"].astype(np.int64)
    gL = z["gen_L"].astype(np.int64)
    ok = np.flatnonzero(gL >= 12)

    hr, fr, rr = entropy_and_feats(
        model, torch.from_numpy(z["real_s"].astype(np.int64)),
        torch.from_numpy(z["real_th"].astype(np.int64)),
        torch.from_numpy(z["real_dt"].astype(np.int64)),
        cond, rL, args.batch, dev)
    hg, fg, rg = entropy_and_feats(
        model, torch.from_numpy(z["gen_s"][ok].astype(np.int64)),
        torch.from_numpy(z["gen_th"][ok].astype(np.int64)),
        torch.from_numpy(z["gen_dt"][ok].astype(np.int64)),
        cond[ok], gL[ok], args.batch, dev)

    gap = float(hg.mean() - hr.mean())
    print(f"\n  {len(hr):,} real positions, {len(hg):,} generated")
    print(f"  C - B {gap:+.4f}   (w4_statecoord had +0.9596)\n")
    print("  shares OVERLAP, these features are correlated, they do not sum\n")
    print(f"  {'history feature':<26}{'BETWEEN':>10}{'se':>8}{'share':>9}"
          f"{'WITHIN':>10}{'gen mean':>11}{'real mean':>11}")
    rows = []
    for k, name in enumerate(FEATS):
        b, w, _, _, _, _ = decompose(hr, fr[:, k], hg, fg[:, k])
        se = boot_between(hr, fr[:, k], rr, hg, fg[:, k], rg, args.seed + k)
        print(f"  {k} {name:<24}{b:>+10.4f}{se:>8.4f}{b / gap:>9.1%}"
              f"{w:>+10.4f}{fg[:, k].mean():>11.4f}{fr[:, k].mean():>11.4f}")
        rows.append({"k": k, "name": name, "between": b, "se": se,
                     "share": b / gap, "within": w,
                     "gen_mean": float(fg[:, k].mean()),
                     "real_mean": float(fr[:, k].mean())})

    order = sorted(rows, key=lambda r: -abs(r["share"]))
    i, j = order[0]["k"], order[1]["k"]
    cr = joint_code((fr[:, i], fr[:, i]), (fr[:, j], fr[:, j]))
    cg = joint_code((fg[:, i], fr[:, i]), (fg[:, j], fr[:, j]))
    jb, jw, _, _, _, _ = decompose(hr, cr.astype(float), hg, cg.astype(float),
                                   nbin=1000)
    print(f"\n  JOINT on the top two, {order[0]['name']} and "
          f"{order[1]['name']}, 8 by 8 grid")
    print(f"    BETWEEN {jb:+.4f}   share {jb / gap:.1%}   WITHIN {jw:+.4f}")

    top = order[0]
    if abs(jb / gap) >= 0.50:
        verdict = (f"NAMED, jointly. {order[0]['name']} and {order[1]['name']} "
                   f"together carry {jb / gap:.1%} of the state effect.")
    elif abs(top["share"]) >= 0.50:
        verdict = (f"NAMED. {top['name']} carries {top['share']:.1%} "
                   "of the state effect.")
    elif abs(top["share"]) < 0.25 and abs(jb / gap) < 0.25:
        verdict = ("DIFFUSE AGAIN. No short window history summary carries the "
                   "effect either. What remains is long range sequence "
                   "structure that only the attention sees.")
    else:
        verdict = (f"PARTIAL. Largest single is {top['name']} at "
                   f"{top['share']:.1%}, joint {jb / gap:.1%}. Claim the table.")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"gap": gap, "feats": rows, "verdict": verdict,
               "joint": {"a": order[0]["name"], "b": order[1]["name"],
                         "between": jb, "within": jw, "share": jb / gap},
               "n_real_pos": int(len(hr)), "n_gen_pos": int(len(hg))},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
