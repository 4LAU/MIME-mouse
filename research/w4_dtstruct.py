"""Where does the alternation in human speed actually live?

`w4_seqstats` found that human per event displacement carries a lag-2
autocorrelation ABOVE its lag-1, 0.6220 against 0.5952. That is an alternating
component: a large step tends to be followed by a small one. `event_ar_v1` lost
it, ratio 1.010 against a human 1.045.

The proposed cause is the within step factorization. `event_ar_v1` emits
p(s) p(th | s) p(dt | s, th), so it commits to a displacement before it knows
the interval that displacement covers. A mouse reports on a fixed poll cadence,
so displacement in one sample is roughly velocity times interval. If the
alternation lives in the interval rather than in the velocity, then choosing
displacement first forces the network to average over the interval, and
averaging is exactly what removes an alternation.

That is a claim about the recorded data, not about any model, so it can be
settled on the corpus alone with no GPU and no sampling. Three streams, the
same pooled autocorrelation as `w4_seqstats` so the numbers line up:

  displacement   s_i                what the model emits first today
  interval       dt_i               what it emits last today
  velocity       s_i / dt_i         displacement with the interval divided out

If the alternation is in the interval, velocity loses it and interval has it.
If velocity keeps it, the factorization is not the cause and the defect is
elsewhere.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=. ~/venvs/mime/bin/python \
        research/w4_dtstruct.py
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from models.event_stream_polar import TICK_CLASS, class_to_speed, s2_to_class  # noqa: E402

MAXLAG = 8
DT_FLOOR_MS = 0.5


def _acf(seqs, maxlag=MAXLAG):
    """Identical to w4_seqstats._acf so the numbers are directly comparable."""
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


def _pooled_corr(xs, ys):
    """Correlation between two streams, standardized within each trajectory
    first so between-trajectory scale cannot create it."""
    num, n = 0.0, 0
    for x, y in zip(xs, ys):
        if len(x) < MAXLAG + 3:
            continue
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        sx, sy = x.std(), y.std()
        if sx < 1e-12 or sy < 1e-12:
            continue
        num += float(np.dot((x - x.mean()) / sx, (y - y.mean()) / sy))
        n += len(x)
    return num / max(n, 1)


def _row(name, seqs, out):
    a = _acf(seqs)
    out[name] = dict(acf=a, n=len(seqs))
    print(f"  {name:<26}{a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>9.4f}{a[3]:>9.4f}"
          f"{a[1] / a[0] if a[0] else float('nan'):>10.3f}{len(seqs):>7}",
          flush=True)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="research/w4_dtstruct.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lengths = np.load("training/events_len.npy")
    pick = np.sort(rng.choice(len(lengths), args.n, replace=False))
    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dt = np.load("training/events_dt.npy", mmap_mode="r")[pick]
    L = np.minimum(lengths[pick], 256)

    s_cls = s2_to_class(torch.from_numpy(s2.astype(np.int64))).numpy()

    disp_all, dt_all, vel_all = [], [], []
    disp_mot, dt_mot, vel_mot = [], [], []
    for i in range(len(L)):
        n = int(L[i])
        if n < MAXLAG + 3:
            continue
        s = class_to_speed(torch.from_numpy(s_cls[i, :n].astype(np.int64))).numpy()
        d = np.maximum(dt[i, :n].astype(np.float64), DT_FLOOR_MS)
        disp_all.append(s)
        dt_all.append(d)
        vel_all.append(s / d)
        m = s_cls[i, :n] > TICK_CLASS
        if m.sum() >= MAXLAG + 3:
            disp_mot.append(s[m])
            dt_mot.append(d[m])
            vel_mot.append((s / d)[m])

    out = {}
    print(f"  {'stream':<26}{'lag1':>9}{'lag2':>9}{'lag3':>9}{'lag4':>9}"
          f"{'lag2/lag1':>10}{'n':>7}")
    print("  all events, ticks included")
    a_disp = _row("displacement s", disp_all, out)
    _row("interval dt", dt_all, out)
    a_vel = _row("velocity s/dt", vel_all, out)
    print("  motion events only")
    _row("displacement s (motion)", disp_mot, out)
    _row("interval dt (motion)", dt_mot, out)
    _row("velocity s/dt (motion)", vel_mot, out)

    out["corr_disp_dt_all"] = _pooled_corr(disp_all, dt_all)
    out["corr_disp_dt_motion"] = _pooled_corr(disp_mot, dt_mot)
    print(f"\n  within trajectory corr(s_i, dt_i): "
          f"all {out['corr_disp_dt_all']:.4f}, "
          f"motion only {out['corr_disp_dt_motion']:.4f}")

    r_d = a_disp[1] / a_disp[0] if a_disp[0] else float("nan")
    r_v = a_vel[1] / a_vel[0] if a_vel[0] else float("nan")
    print(f"  lag2/lag1 ratio: displacement {r_d:.3f}, velocity {r_v:.3f}")
    print("  a ratio above 1 is the alternation. If it is present in "
          "displacement\n  and absent in velocity, the alternation is carried "
          "by the interval.")
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
