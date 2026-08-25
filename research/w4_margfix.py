"""w4_margfix. Correct the model's free running token marginals directly.

Registered in /home/aaronadmin/w4_arms/margfix_prereg.md.

`w4_detcap` put most of what a detector can extract into MARGINAL token rates.
`w4_poskl` showed that rate shift is position stationary, the same size at event
1 as at event 40. The matching correction is a per class additive bias on each
head's logits, fitted to the gap between the corpus token rates and the model's
own free running rates, applied identically at every step.

ONE APPLICATION WILL NOT WORK. The free running marginal is the stationary
distribution of a feedback loop, not the average of the conditionals, so this
runs a fixed point iteration and reports whether it converges.

The human target comes from the CORPUS, which is training data.
`data/human_eval_features.npy` is never read here.
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

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                      # noqa: E402
import scoring                                                    # noqa: E402
from features import extract_feature_matrix                       # noqa: E402
from models.event_ar import (DT_MAX_MS, EventARModel,             # noqa: E402
                             N_DT_CLASSES, class_to_dt_ms)
from models.event_stream_polar import (S_PAD_CLASS, TH_BINS,      # noqa: E402
                                       TH_NULL_CLASS, TICK_CLASS,
                                       N_S_CLASSES, N_TH_CLASSES)
from phase_a_baseline import make_specs                           # noqa: E402
from w4_detcap import corpus_ids, corpus_tokens                   # noqa: E402

AR_OPTIMUM = 0.5792
CONTRACT_SD = 0.0073
EPS = 1e-6


def marginals(s, th, dt, L=None):
    """Free running token rates over the population each bias actually acts on.

    s   live positions, every class below PAD including TICK
    th  MOTION positions only. NULL is the no motion marker, not a turn, and
        the served path overwrites it from the speed token anyway.
    dt  live positions
    """
    if L is None:
        pad = s >= S_PAD_CLASS
        L = np.where(pad.any(1), pad.argmax(1), s.shape[1])
    live = np.arange(s.shape[1])[None] < np.asarray(L)[:, None]
    motion = live & (s > TICK_CLASS) & (s < S_PAD_CLASS)
    ps = np.bincount(s[live], minlength=N_S_CLASSES)[:S_PAD_CLASS]
    pth = np.bincount(th[motion], minlength=N_TH_CLASSES)[:TH_BINS]
    pdt = np.bincount(np.clip(dt[live], 0, DT_MAX_MS),
                      minlength=N_DT_CLASSES)[:DT_MAX_MS + 1]
    return tuple(x / max(x.sum(), 1) for x in (ps, pth, pdt))


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def update(bias, p_h, p_m, lam, clip=2.0):
    """One fixed point step, centred so the correction does not move the length
    distribution to first order.

    Centring matters: raising every real speed class also lowers PAD through
    the softmax denominator, which would change the event count. Weighting the
    centre by the model's own current rates makes that first order change zero.
    """
    step = np.clip(lam * (np.log(p_h + EPS) - np.log(p_m + EPS)), -clip, clip)
    b = bias + step
    b = b - float((p_m * b).sum() / max(p_m.sum(), EPS))
    return b


def to_dev(b, n_classes, dev):
    """Pad a bias over the acted-on classes up to the head's full width. The
    classes left out, PAD on speed and NULL on turn, get exactly zero."""
    full = np.zeros(n_classes, dtype=np.float32)
    full[:len(b)] = b
    return torch.tensor(full, device=dev)


def build_specs(n, seed):
    rows, meta = [], []
    for sx, sy, ex, ey in make_specs(n, seed):
        d = math.hypot(ex - sx, ey - sy)
        if d < 1e-6:
            continue
        ld, ang = math.log(d), math.atan2(ey - sy, ex - sx)
        rows.append([ld, math.log(esp._duration.sample(ld)),
                     math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ang, ex, ey))
    return rows, meta


def generate(model, rows, meta, dev, batch, temps, biases):
    """One trajectory per specification. No selection, no best of N."""
    st, tt, dtt = temps
    sb, tb, db = biases
    S, TH, DT, paths = [], [], [], []
    for c0 in range(0, len(rows), batch):
        cond = torch.tensor(rows[c0:c0 + batch], dtype=torch.float32,
                            device=dev)
        s_cls, th_cls, dt_cls = model.sample(
            cond, temperature=st, th_temperature=tt, dt_temperature=dtt,
            s_bias=sb, th_bias=tb, dt_bias=db)
        s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
        dt_ms = class_to_dt_ms(dt_cls)
        dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
                / esp._DT_STD).cpu().numpy()
        S.append(s_np); TH.append(th_np); DT.append(dt_cls.cpu().numpy())
        for j in range(len(s_np)):
            sx, sy, ang, ex, ey = meta[c0 + j]
            p = esp._decode(dt_z[j], s_np[j], th_np[j], sx, sy, ang)
            if p is not None:
                paths.append(np.asarray(p, dtype=np.float64))
    return np.vstack(S), np.vstack(TH), np.vstack(DT), paths


def score(paths, seed):
    F = extract_feature_matrix(paths)
    F = F[np.all(np.isfinite(F), 1)]
    F = F[np.random.default_rng(seed).permutation(len(F))]   # shuffle, always
    r = scoring.score_features(F)
    return (float(r["auc_rf_oob"]), bool(r["collapse_flag"]),
            list(r["collapse_features"]), int(len(F)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_hm_mlp.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--human-rows", type=int, default=200000)
    ap.add_argument("--out", default="research/w4_margfix_results.json")
    a = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{a.ckpt}", map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {a.ckpt} step {ck.get('step')}  device {dev}", flush=True)

    # ---- human target, from the CORPUS. training data, never the eval set --
    hs, hth, hdt, _, hL = corpus_tokens(
        corpus_ids(np.random.default_rng(3), a.human_rows))
    PH = marginals(hs, hth, hdt, hL)
    print(f"  human target from {a.human_rows} corpus rows, "
          f"support s {int((PH[0] > 0).sum())}  th {int((PH[1] > 0).sum())}  "
          f"dt {int((PH[2] > 0).sum())}\n", flush=True)

    temps = (0.95, 0.90, 1.00)          # the closed three head optimum
    seeds = [int(x) for x in a.seeds.split(",")]
    specs = {sd: build_specs(a.n, sd) for sd in seeds}
    bias = [np.zeros(S_PAD_CLASS), np.zeros(TH_BINS), np.zeros(DT_MAX_MS + 1)]
    widths = (N_S_CLASSES, N_TH_CLASSES, N_DT_CLASSES)

    out = {"config": dict(ckpt=a.ckpt, n=a.n, seeds=seeds, rounds=a.rounds,
                          lam=a.lam, temps=temps), "rounds": []}
    print(f"  {'round':>6}{'tv_s':>9}{'tv_th':>9}{'tv_dt':>9}"
          f"{'contract':>10}{'se':>8}{'collapse':>10}")
    for rd in range(a.rounds + 1):
        bt = [None if rd == 0 else to_dev(b, w, dev)
              for b, w in zip(bias, widths)]
        aucs, coll, cfs, pm_acc = [], [], set(), []
        for sd in seeds:
            rows, meta = specs[sd]
            torch.manual_seed(5000 + sd)
            S, TH, DT, paths = generate(model, rows, meta, dev, a.batch,
                                        temps, bt)
            pm_acc.append(marginals(S, TH, DT))
            au, cl, cf, n = score(paths, sd)
            aucs.append(au); coll.append(cl); cfs |= set(cf)
        PM = tuple(np.mean([p[i] for p in pm_acc], axis=0) for i in range(3))
        tvs = [tv(PH[i], PM[i]) for i in range(3)]
        m = float(np.mean(aucs))
        se = float(np.std(aucs, ddof=1) / math.sqrt(len(aucs)))
        print(f"  {rd:>6}{tvs[0]:>9.4f}{tvs[1]:>9.4f}{tvs[2]:>9.4f}"
              f"{m:>10.4f}{se:>8.4f}{str(any(coll)):>10}"
              + (f"  {sorted(cfs)}" if cfs else ""), flush=True)
        out["rounds"].append(dict(round=rd, tv=tvs, aucs=aucs, mean=m, se=se,
                                  collapse=bool(any(coll)),
                                  collapse_features=sorted(cfs)))
        if rd < a.rounds:
            bias = [update(bias[i], PH[i], PM[i], a.lam) for i in range(3)]

    r0, rl = out["rounds"][0], out["rounds"][-1]
    print("\n  GATES, read before the primary")
    g1 = all(rl["tv"][i] < r0["tv"][i] for i in range(3))
    g2 = not rl["collapse"]
    print(f"    G1  marginal tv fell on all three channels, "
          f"{[round(x, 4) for x in r0['tv']]} -> "
          f"{[round(x, 4) for x in rl['tv']]}  -> {'PASS' if g1 else 'FAIL'}")
    print(f"    G2  no collapse flag  -> {'PASS' if g2 else 'FAIL'}"
          + (f"   {rl['collapse_features']}" if rl["collapse_features"] else ""))
    print("    G3  one trajectory per spec, no selection, by construction")

    best = min(out["rounds"][1:], key=lambda r: r["mean"])
    v = ("STRONG" if best["mean"] < 0.5646
         else "LIVE" if best["mean"] <= AR_OPTIMUM else "DEAD")
    print(f"\n  PRIMARY. baseline round 0 {r0['mean']:.4f} "
          f"(record {AR_OPTIMUM}), best corrected round {best['round']} "
          f"{best['mean']:.4f} se {best['se']:.4f}")
    print(f"    change vs this session's own baseline "
          f"{best['mean'] - r0['mean']:+.4f}, contract sd {CONTRACT_SD}")
    print(f"\n  VERDICT  {v}")
    if not (g1 and g2):
        print("  A GATE FAILED. the verdict above is NOT readable.")
    out["primary"] = dict(baseline=r0["mean"], best_round=best["round"],
                          best=best["mean"], verdict=v, g1=g1, g2=g2)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
