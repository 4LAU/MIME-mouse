"""The speed alternation is tick placement, not speed.

`w4_dtstruct` settled two things on the recorded corpus alone. Dividing the
interval out of displacement leaves the lag-2 over lag-1 ratio unchanged, 1.048
against 1.049, so the alternation is not carried by the interval. And once the
stream is restricted to events that actually move, the ratio falls to 0.932 and
the decay becomes monotone, so there is no alternation among moving events at
all.

What is left is the only other thing in the stream: events that report zero
displacement. Interleaving zeros with non-zeros is what makes a large value
tend to be followed by a small one. So the statistic `w4_seqstats` flagged as a
speed defect is really a statement about where the still events sit.

That matters because it changes what has to be fixed. This measures the still
events directly, on token streams before decoding, for human and for any AR
checkpoint:

  share            fraction of events that are still
  gap VMR          dispersion of the gaps between still events, above 1 clumped
  run p50 / p90    consecutive moving events between stills
  motion acf       speed autocorrelation with the still events removed

The last column is the one that says whether speed itself is wrong. If the
model matches human there, the whole discrepancy is arrangement of stills.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_tickstruct.py --ar event_ar_v1.pt
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
    S_PAD_CLASS, TICK_CLASS, class_to_speed, s2_to_class,
)
from phase_a_baseline import make_specs  # noqa: E402
from w4_seqstats import _acf, _gap_stats  # noqa: E402

MAXLAG = 8


def _measure(name, s_cls, lens, out):
    still_flags, motion_speeds, all_speeds, shares = [], [], [], []
    for i in range(len(lens)):
        n = int(lens[i])
        if n < MAXLAG + 3:
            continue
        cls = s_cls[i, :n].astype(np.int64)
        sp = class_to_speed(torch.from_numpy(cls)).numpy()
        still = cls <= TICK_CLASS
        all_speeds.append(sp)
        shares.append(float(still.mean()))
        if (~still).sum() >= MAXLAG + 3:
            motion_speeds.append(sp[~still])
        still_flags.append(still)

    a_all = _acf(all_speeds)
    a_mot = _acf(motion_speeds)
    gs = _gap_stats(still_flags)
    rec = dict(share=float(np.mean(shares)), gap_vmr=gs["gap_vmr"],
               run_p50=gs["run_p50"], run_p90=gs["run_p90"],
               acf_all=a_all, acf_motion=a_mot, n=len(all_speeds),
               n_motion=len(motion_speeds))
    out[name] = rec
    print(f"  {name:<22}{rec['share']:>8.4f}{gs['gap_vmr']:>9.4f}"
          f"{gs['run_p50']:>8.1f}{gs['run_p90']:>8.1f}"
          f"{a_all[0]:>9.4f}{a_all[1] / a_all[0]:>8.3f}"
          f"{a_mot[0]:>9.4f}{a_mot[1] / a_mot[0]:>8.3f}{len(all_speeds):>7}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--ar", default="", help="comma separated AR checkpoints")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="research/w4_tickstruct.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    out = {}
    print(f"  {'arm':<22}{'share':>8}{'gapVMR':>9}{'run50':>8}{'run90':>8}"
          f"{'allAc1':>9}{'ratio':>8}{'motAc1':>9}{'ratio':>8}{'n':>7}")

    rng = np.random.default_rng(args.seed)
    lengths = np.load("training/events_len.npy")
    pick = np.sort(rng.choice(len(lengths), args.n * 2, replace=False))
    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    L = np.minimum(lengths[pick], 256)
    hs = s2_to_class(torch.from_numpy(s2.astype(np.int64))).numpy()
    _measure("human", hs[:args.n], L[:args.n], out)

    specs = make_specs(args.n, args.seed)
    rows = []
    for sx, sy, ex, ey in specs:
        d = math.hypot(ex - sx, ey - sy)
        if d < 1e-6:
            continue
        ld = math.log(d)
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([ld, math.log(esp._duration.sample(ld)),
                     math.cos(ang), math.sin(ang)])

    def collect(fn, label):
        S, N = [], []
        for c0 in range(0, len(rows), args.batch):
            cond = torch.tensor(rows[c0:c0 + args.batch], dtype=torch.float32,
                                device=dev)
            s_cls = fn(cond).cpu().numpy()
            pad = s_cls >= S_PAD_CLASS
            S.append(s_cls)
            N.append(np.where(pad.any(1), pad.argmax(1), s_cls.shape[1]))
        _measure(label, np.concatenate(S), np.concatenate(N), out)

    if args.masked:
        m, seq_len = esp._model, esp._cfg["max_seq_len"]

        def masked_fn(cond):
            with torch.no_grad():
                _, s_cls, _ = m.sample(
                    cond, seq_len, n_steps=100, temperature=args.temp,
                    order="gumbel", choice_temp=10.0,
                    feat=torch.zeros(cond.shape[0], esp._FEAT_BANK.shape[1],
                                     device=dev) if esp._FEAT_BANK is not None else None)
            return s_cls
        collect(masked_fn, "masked served")

    for ck_name in [c for c in args.ar.split(",") if c]:
        ck = torch.load(f"training/{ck_name}", map_location=dev,
                        weights_only=False)
        model = EventARModel(**ck["config"]).to(dev).eval()
        model.load_state_dict(ck["model_state_dict"])
        collect(lambda cond, _m=model: _m.sample(cond, temperature=args.temp)[0],
                f"ar {ck_name.replace('.pt', '')}")

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  motAc1 is speed autocorrelation with still events removed.")
    print("  If it matches human, speed is right and only still placement is "
          "wrong.")


if __name__ == "__main__":
    main()
