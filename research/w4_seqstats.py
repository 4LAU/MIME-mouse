"""Does the arrangement in time actually move?

The contract score alone cannot say WHY an arm scored what it scored, and the
diagnosed defect is local temporal arrangement, not any marginal. This measures
arrangement directly, in the model's own units, so a new architecture can be
judged on the mechanism it claims rather than only on the headline number.

Three statistics, all invariant to the marginals and sensitive only to order:

- lag-1 to lag-8 autocorrelation of per event speed and of absolute turn.
  `w4_arrangement` showed that permuting speeds inside a window of sixteen,
  which leaves every marginal exactly unchanged, takes real human data from
  0.5576 to 0.8595. Autocorrelation is what that permutation destroys.
- dispersion of the gaps between turning events, as the variance-to-mean ratio.
  `w4_joint2d` and `w4_straight` together said the model's turning is CLUMPED
  where human turning is spread: gen no-turn share 0.3940 against a human
  0.3238 and gen straight runs median 6 / p90 20 against human 4 / 14. A
  clumped point process has a gap dispersion above 1; an evenly spread one
  sits at or below it.
- run-length distribution of consecutive no-turn events, reported at the
  median and p90 so it lines up with the numbers already recorded.

Arms are human, the served masked model, and whatever checkpoints are passed
with --ar. Everything is computed on the token streams, before decoding, so
no part of the serving pipeline can mask a difference.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_seqstats.py --ar event_ar_v1.pt
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

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TICK_CLASS, class_to_dtheta, class_to_speed,
    dth_lattice_to_class, s2_to_class,
)
from phase_a_baseline import make_specs  # noqa: E402

MAXLAG = 8
TURN_EPS = 1e-9


def _acf(seqs, maxlag=MAXLAG):
    """Pooled autocorrelation: standardize within each sequence, then pool the
    lagged products. Standardizing per sequence is what makes this blind to
    between-trajectory differences in level and spread, which are marginal
    properties and already known to be correct."""
    num = np.zeros(maxlag)
    den = np.zeros(maxlag)
    for v in seqs:
        if len(v) < maxlag + 3:
            continue
        v = np.asarray(v, dtype=np.float64)
        sd = v.std()
        if sd < 1e-12:
            continue
        z = (v - v.mean()) / sd
        for k in range(1, maxlag + 1):
            num[k - 1] += float(np.dot(z[:-k], z[k:]))
            den[k - 1] += len(z) - k
    return (num / np.maximum(den, 1)).tolist()


def _gap_stats(turn_flags):
    """Gaps between turning events. Variance-to-mean ratio above 1 means
    clumped, at 1 means Poisson-like, below 1 means spread out."""
    gaps, runs = [], []
    for f in turn_flags:
        idx = np.flatnonzero(f)
        if len(idx) >= 3:
            gaps.extend(np.diff(idx).tolist())
        r, c = [], 0
        for b in f:
            if b:
                if c:
                    r.append(c)
                c = 0
            else:
                c += 1
        if c:
            r.append(c)
        runs.extend(r)
    g = np.asarray(gaps, dtype=np.float64)
    r = np.asarray(runs, dtype=np.float64)
    return dict(
        gap_mean=float(g.mean()) if len(g) else float("nan"),
        gap_vmr=float(g.var() / g.mean()) if len(g) and g.mean() > 0 else float("nan"),
        run_p50=float(np.median(r)) if len(r) else float("nan"),
        run_p90=float(np.percentile(r, 90)) if len(r) else float("nan"),
        no_turn_share=float(np.mean([1.0 - f.mean() for f in turn_flags
                                     if len(f)])),
    )


def _streams(s_cls, th_cls, lens):
    """Token streams -> (per event speed, per event absolute turn, turn flag),
    one entry per trajectory, truncated at the real length."""
    speeds, turns, flags = [], [], []
    for i in range(len(lens)):
        n = int(lens[i])
        if n < MAXLAG + 3:
            continue
        s = class_to_speed(torch.from_numpy(s_cls[i, :n].astype(np.int64))).numpy()
        d = class_to_dtheta(torch.from_numpy(th_cls[i, :n].astype(np.int64))).numpy()
        motion = s_cls[i, :n] > TICK_CLASS
        a = np.abs(np.where(motion, d, 0.0))
        speeds.append(s)
        turns.append(a)
        flags.append(a > TURN_EPS)
    return speeds, turns, flags


def _report(name, speeds, turns, flags, out):
    acs = _acf(speeds)
    act = _acf(turns)
    gs = _gap_stats(flags)
    out[name] = dict(acf_speed=acs, acf_turn=act, n=len(speeds), **gs)
    print(f"  {name:<22}{acs[0]:>9.4f}{acs[1]:>9.4f}{acs[3]:>9.4f}"
          f"{act[0]:>9.4f}{act[1]:>9.4f}{gs['gap_vmr']:>9.4f}"
          f"{gs['run_p50']:>8.1f}{gs['run_p90']:>8.1f}{len(speeds):>7}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--ar", default="", help="comma separated AR checkpoints")
    ap.add_argument("--masked", action="store_true",
                    help="also run the served masked model")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="research/w4_seqstats.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    out = {}
    print(f"  {'arm':<22}{'s_ac1':>9}{'s_ac2':>9}{'s_ac4':>9}{'t_ac1':>9}"
          f"{'t_ac2':>9}{'gapVMR':>9}{'run50':>8}{'run90':>8}{'n':>7}")

    # human, straight off the recorded corpus
    rng = np.random.default_rng(args.seed)
    lengths = np.load("training/events_len.npy")
    pick = np.sort(rng.choice(len(lengths), args.n * 3, replace=False))
    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    L = np.minimum(lengths[pick], 256)
    hs = s2_to_class(torch.from_numpy(s2.astype(np.int64))).numpy()
    hth = np.where(s2 > 0,
                   dth_lattice_to_class(torch.from_numpy(dth.astype(np.int64))).numpy(),
                   256)
    sp, tu, fl = _streams(hs, hth, L)
    _report("human", sp[:args.n], tu[:args.n], fl[:args.n], out)

    specs = make_specs(args.n, args.seed)
    rows, lens_hint = [], []
    for sx, sy, ex, ey in specs:
        d = math.hypot(ex - sx, ey - sy)
        if d < 1e-6:
            continue
        ld = math.log(d)
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([ld, math.log(esp._duration.sample(ld)),
                     math.cos(ang), math.sin(ang)])

    def collect(sample_fn, label):
        S, T, Ln = [], [], []
        for c0 in range(0, len(rows), args.batch):
            cond = torch.tensor(rows[c0:c0 + args.batch], dtype=torch.float32,
                                device=dev)
            s_cls, th_cls = sample_fn(cond)
            s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
            pad = s_np >= S_PAD_CLASS
            n = np.where(pad.any(1), pad.argmax(1), s_np.shape[1])
            S.append(s_np)
            T.append(th_np)
            Ln.append(n)
        s_np, th_np, n = np.concatenate(S), np.concatenate(T), np.concatenate(Ln)
        a, b, c = _streams(s_np, th_np, n)
        _report(label, a, b, c, out)

    if args.masked:
        m = esp._model
        seq_len = esp._cfg["max_seq_len"]

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
            s_cls, th_cls, _ = _m.sample(cond, temperature=args.temp)
            return s_cls, th_cls
        collect(ar_fn, f"ar {ck_name.replace('.pt', '')}")

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  gapVMR above 1 is clumped turning, at 1 Poisson, below spread")
    print("  human is the target on every column")


if __name__ == "__main__":
    main()
