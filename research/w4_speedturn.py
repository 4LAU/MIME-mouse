"""Can the model turn hard while moving fast? Humans can.

`w4_copula` showed that giving the model perfect one dimensional statistics on
all eighteen scored features would move the detector only from 0.6668 to 0.6208
against an instrument null of 0.5148, so roughly seventy percent of the excess
is cross feature coupling and not marginals.

`w4_couplemap` then found that coupling error is not diffuse. All ten of the
largest rank correlation errors pair a speed or acceleration feature with a
turning feature, all with the same sign, the model always more negative:

  max_velocity     vs angular_velocity_std     human -0.024  model -0.210
  max_velocity     vs angular_velocity_mean    human -0.081  model -0.265
  std_velocity     vs angular_velocity_mean    human -0.096  model -0.279
  max_deviation    vs angular_velocity_std     human +0.172  model +0.004
  max_acceleration vs num_direction_changes    human +0.248  model +0.093

In a person, speed and turning are close to independent, because a real
correction is a fast sharp movement. In the model they trade off: fast paths
come out too straight and curved paths too slow. The last two rows are the same
defect seen twice more. Human wiggle accumulates into real displacement from
the straight line and the model's cancels out, and human direction changes
arrive with acceleration bursts while the model's do not.

Those are all trajectory SUMMARY statistics. This measures the same thing where
the model actually makes its decisions, on single events, so the finding either
survives at the level a loss function could reach or it does not.

  turn given speed  P(|dtheta| > 0.5 rad) inside each speed decile, and the
                    median and p90 of |dtheta| there. If the model cannot turn
                    at speed this collapses in the top deciles.
  pooled rho        Spearman of step size against |dtheta| over every moving
                    event. One number for the whole corpus.
  per path rho      the same correlation computed inside each trajectory, then
                    its distribution. This is the quantity that becomes the
                    feature level coupling, so it is the one that has to match.
  burst             |dtheta| against the change in step size between
                    consecutive moving events, which is acceleration at event
                    resolution. Tests the direction changes without
                    acceleration bursts row directly.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_speedturn.py --masked \
        --ar event_ar_v1.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_SNAP", "2.5")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_BESTOF", "1")
os.environ.setdefault("EVENT_SIR", "1")
os.environ.setdefault("EVENT_ORDER", "gumbel")
os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")
os.environ["EVENT_TICKMERGE"] = "0"

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, class_to_dtheta, class_to_speed,
    dth_lattice_to_class, s2_to_class,
)
from phase_a_baseline import make_specs  # noqa: E402

BIG_TURN = 0.5


def _measure(label, streams, out):
    S, T, dS, dT, per = [], [], [], [], []
    for s_cls, th_cls, n in streams:
        if n < 12:
            continue
        mo = np.flatnonzero(s_cls[:n] > TICK_CLASS)
        if len(mo) < 8:
            continue
        s = class_to_speed(torch.from_numpy(s_cls[mo].astype(np.int64))).numpy()
        t = np.abs(class_to_dtheta(
            torch.from_numpy(th_cls[mo].astype(np.int64))).numpy())
        S.append(s)
        T.append(t)
        dS.append(np.abs(np.diff(s)))
        dT.append(t[1:])
        if s.std() > 1e-9 and t.std() > 1e-9:
            per.append(float(spearmanr(s, t).statistic))
    S, T = np.concatenate(S), np.concatenate(T)
    dS, dT = np.concatenate(dS), np.concatenate(dT)
    edges = np.quantile(S, np.linspace(0, 1, 11))
    edges[-1] += 1e-6
    big, med, p90 = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (S >= a) & (S < b)
        if m.sum() < 20:
            big.append(float("nan"))
            med.append(float("nan"))
            p90.append(float("nan"))
            continue
        big.append(float((T[m] > BIG_TURN).mean()))
        med.append(float(np.median(T[m])))
        p90.append(float(np.percentile(T[m], 90)))
    per = np.asarray(per)
    rec = dict(n_paths=len(per), n_events=int(len(S)),
               pooled_rho=float(spearmanr(S, T).statistic),
               per_path_rho_p50=float(np.median(per)),
               per_path_rho_p10=float(np.percentile(per, 10)),
               per_path_rho_p90=float(np.percentile(per, 90)),
               burst_rho=float(spearmanr(dS, dT).statistic),
               speed_edges=[float(e) for e in edges],
               big_turn_share=big, turn_p50=med, turn_p90=p90)
    out[label] = rec
    print(f"  {label:<20}{rec['pooled_rho']:>9.3f}"
          f"{rec['per_path_rho_p50']:>9.3f}{rec['per_path_rho_p10']:>9.3f}"
          f"{rec['per_path_rho_p90']:>9.3f}{rec['burst_rho']:>9.3f}"
          f"{len(per):>7}", flush=True)
    print("      bigturn " + "".join(f"{v:>7.3f}" for v in big), flush=True)
    print("      turnp90 " + "".join(f"{v:>7.3f}" for v in p90), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--ar", default="")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="research/w4_speedturn.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    out = {}
    print(f"  {'arm':<20}{'pooled':>9}{'pathP50':>9}{'pathP10':>9}"
          f"{'pathP90':>9}{'burst':>9}{'n':>7}")

    rng = np.random.default_rng(args.seed)
    lengths = np.load("training/events_len.npy")
    pick = np.sort(rng.choice(len(lengths), args.n * 2, replace=False))
    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    L = np.minimum(lengths[pick], 256)
    hs = s2_to_class(torch.from_numpy(s2.astype(np.int64))).numpy()
    hth = np.where(s2 > 0,
                   dth_lattice_to_class(torch.from_numpy(dth.astype(np.int64))).numpy(),
                   TH_NULL_CLASS)
    _measure("human", ((hs[i], hth[i], int(L[i]))
                       for i in range(min(args.n, len(L)))), out)

    specs = make_specs(args.n, args.seed)
    rows = []
    for sx, sy, ex, ey in specs:
        dd = math.hypot(ex - sx, ey - sy)
        if dd < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([math.log(dd), math.log(esp._duration.sample(math.log(dd))),
                     math.cos(ang), math.sin(ang)])

    def collect(fn, label):
        S, T, N = [], [], []
        for c0 in range(0, len(rows), args.batch):
            cond = torch.tensor(rows[c0:c0 + args.batch], dtype=torch.float32,
                                device=dev)
            s_cls, th_cls = fn(cond)
            s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
            pad = s_np >= S_PAD_CLASS
            S.append(s_np)
            T.append(th_np)
            N.append(np.where(pad.any(1), pad.argmax(1), s_np.shape[1]))
        s_np, th_np, n = np.concatenate(S), np.concatenate(T), np.concatenate(N)
        _measure(label, ((s_np[i], th_np[i], int(n[i])) for i in range(len(n))),
                 out)

    if args.masked:
        m, seq_len = esp._model, esp._cfg["max_seq_len"]

        def masked_fn(cond):
            with torch.no_grad():
                _, s_cls, th_cls = m.sample(
                    cond, seq_len, n_steps=100, temperature=args.temp,
                    order="gumbel", choice_temp=10.0,
                    feat=torch.zeros(cond.shape[0], esp._FEAT_BANK.shape[1],
                                     device=dev) if esp._FEAT_BANK is not None else None)
            return s_cls, th_cls
        collect(masked_fn, "masked served")

    for ck_name in [c for c in args.ar.split(",") if c]:
        ck = torch.load(f"training/{ck_name}", map_location=dev,
                        weights_only=False)
        model = EventARModel(**ck["config"]).to(dev).eval()
        model.load_state_dict(ck["model_state_dict"])

        def ar_fn(cond, _m=model):
            a, b, _c = _m.sample(cond, temperature=args.temp)
            return a, b
        collect(ar_fn, f"ar {ck_name.replace('.pt', '')}")

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  bigturn is P(|dtheta| > 0.5 rad) inside each speed decile,")
    print("  left to right from the slowest events to the fastest. A model")
    print("  that cannot turn at speed collapses on the right of that row.")
    print("  pathP50 is the median within trajectory speed against turn")
    print("  correlation, which is what becomes the feature level coupling.")


if __name__ == "__main__":
    main()
