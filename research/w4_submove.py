"""Are the pauses corrective submovements?

L's account, which is also the standard one in motor control (Woodworth's two
component model, Meyer's optimized submovement model): a person launches a fast
ballistic movement aimed from peripheral or remembered information, that launch
systematically misses, they pause, and then one or more slower visually guided
corrections bring the pointer onto the element. Sometimes the launch overshoots
and the correction comes back.

`w4_tickstruct` established WHERE the models are wrong about still events: human
stills are 9.48 percent of events in clumps of gap dispersion 21.6, the served
masked model emits 15.67 percent, and `event_ar_v1` emits 6.21 percent with
moving runs of p90 37 against a human 21. It said nothing about why the stills
sit where they do, and a statistic with no mechanism is a hard thing to hand a
model as a target.

A pause is measured in MILLISECONDS, not in events. The first version of this
probe counted every still run, and most still runs are a single 8ms sample where
the pointer did not move far enough to register a lattice step. Those are
quantization, not decisions, they outnumber real pauses heavily, and averaging
them in drives every conditional signal to its unconditional value. `--pause-ms`
sets the floor; the default 40ms is below any plausible visual correction
latency and well above one poll interval.

If the pauses are corrective, these must hold of the recorded data:

  1. pauses sit near the target, so the remaining fraction of the straight line
     distance at a pause is small, and the LAST pause more so than the first
  2. the movement after a pause is slower than the movement before it
  3. the heading changes more across a pause than across a matched stretch of
     uninterrupted motion, because a correction is a re-aim
  4. the launch misses, so the path has a closest approach and then leaves it
  5. longer movements take more pauses

Everything is computed from the token streams with the same geometry the serving
decoder uses. EVENT_TICKMERGE is forced off because it deletes isolated stills
and would erase the thing being measured.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_submove.py --ar event_ar_v1.pt \
        --masked --pause-ms 40
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
os.environ["EVENT_TICKMERGE"] = "0"

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
from models.event_ar import EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import (  # noqa: E402
    S_PAD_CLASS, TH_NULL_CLASS, TICK_CLASS, class_to_dtheta, class_to_speed,
    dth_lattice_to_class, s2_to_class,
)
from phase_a_baseline import make_specs  # noqa: E402

CTX = 5          # motion events averaged either side of a pause
NEAR = 0.10      # "closest approach" threshold as a fraction of total distance


def _geom(s_cls, th_cls, angle):
    """Per event displacement and cumulative position, as _decode builds them."""
    s = class_to_speed(torch.from_numpy(s_cls.astype(np.int64))).numpy()
    dth = class_to_dtheta(torch.from_numpy(th_cls.astype(np.int64))).numpy()
    motion = s_cls > TICK_CLASS
    heading = angle + np.cumsum(np.where(motion, dth, 0.0))
    dx = np.where(motion, s * np.cos(heading), 0.0)
    dy = np.where(motion, s * np.sin(heading), 0.0)
    x = np.concatenate([[0.0], np.cumsum(dx)])
    y = np.concatenate([[0.0], np.cumsum(dy)])
    return s, motion, heading, x, y


def _runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _measure(label, streams, pause_ms, out):
    """streams yields (s_cls, th_cls, dt_ms, n, angle) per trajectory."""
    acc = dict(rem=[], rem_last=[], idxf=[], ratio=[], turn=[], dur=[])
    base_turn, npause, nstill, dists = [], [], [], []
    over_exc, over_hit = [], []
    for s_cls, th_cls, dt_ms, n, angle in streams:
        if n < 12:
            continue
        s, motion, heading, x, y = _geom(s_cls[:n], th_cls[:n], angle)
        d = np.asarray(dt_ms[:n], dtype=np.float64)
        ex, ey = x[-1], y[-1]
        D = math.hypot(ex - x[0], ey - y[0])
        if D < 20:
            continue
        d_end = np.hypot(x - ex, y - ey)

        rs = _runs(~motion)
        nstill.append(len(rs))
        kept = []
        for a, b in rs:
            if d[a:b].sum() < pause_ms:
                continue
            kept.append((a, b))
        npause.append(len(kept))
        dists.append(D)

        per_traj_rem = []
        for a, b in kept:
            acc["dur"].append(float(d[a:b].sum()))
            if a < CTX or b > n - CTX:
                continue
            pre = s[max(0, a - CTX):a][motion[max(0, a - CTX):a]]
            post = s[b:b + CTX][motion[b:b + CTX]]
            if len(pre) < 2 or len(post) < 2:
                continue
            r = float(d_end[a] / D)
            acc["rem"].append(r)
            per_traj_rem.append(r)
            acc["idxf"].append(float(a / n))
            acc["ratio"].append(float(post.mean() / max(pre.mean(), 1e-6)))
            h_pre = heading[max(0, a - CTX):a][motion[max(0, a - CTX):a]].mean()
            h_post = heading[b:b + CTX][motion[b:b + CTX]].mean()
            acc["turn"].append(abs(float(np.arctan2(np.sin(h_post - h_pre),
                                                    np.cos(h_post - h_pre)))))
        if per_traj_rem:
            acc["rem_last"].append(per_traj_rem[-1])

        mi = np.flatnonzero(motion)
        for k in range(CTX, len(mi) - 2 * CTX, max(1, len(mi) // 4)):
            h_pre = heading[mi[k - CTX:k]].mean()
            h_post = heading[mi[k:k + CTX]].mean()
            base_turn.append(abs(float(np.arctan2(np.sin(h_post - h_pre),
                                                  np.cos(h_post - h_pre)))))

        near = np.flatnonzero(d_end <= NEAR * D)
        if len(near):
            k = int(near[0])
            exc = float(d_end[k:].max())
            over_exc.append(exc / D)
            over_hit.append(1.0 if exc > 0.05 * D else 0.0)

    def med(v):
        return float(np.median(v)) if len(v) else float("nan")

    rec = dict(
        n=len(npause), n_pause_events=len(acc["rem"]),
        pause_rem_p50=med(acc["rem"]), pause_rem_last_p50=med(acc["rem_last"]),
        pause_idx_p50=med(acc["idxf"]), spd_ratio_p50=med(acc["ratio"]),
        pause_dur_p50=med(acc["dur"]),
        pause_dur_p90=float(np.percentile(acc["dur"], 90)) if acc["dur"] else float("nan"),
        turn_at_pause_p50=med(acc["turn"]), turn_baseline_p50=med(base_turn),
        n_pauses_p50=med(npause), n_pauses_mean=float(np.mean(npause)) if npause else float("nan"),
        n_stillruns_mean=float(np.mean(nstill)) if nstill else float("nan"),
        corr_pauses_dist=float(np.corrcoef(np.log(dists), npause)[0, 1])
        if len(dists) > 10 and np.std(npause) > 0 else float("nan"),
        overshoot_share=float(np.mean(over_hit)) if over_hit else float("nan"),
        overshoot_exc_p50=med(over_exc),
    )
    out[label] = rec
    print(f"  {label:<20}{rec['n_pauses_mean']:>8.2f}{rec['pause_dur_p50']:>9.0f}"
          f"{rec['pause_rem_p50']:>9.3f}{rec['pause_rem_last_p50']:>9.3f}"
          f"{rec['spd_ratio_p50']:>10.3f}{rec['turn_at_pause_p50']:>10.3f}"
          f"{rec['turn_baseline_p50']:>10.3f}{rec['corr_pauses_dist']:>8.3f}"
          f"{rec['overshoot_share']:>10.3f}{rec['n']:>7}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--ar", default="")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--pause-ms", type=float, default=40.0)
    ap.add_argument("--out", default="research/w4_submove.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    out = {"pause_ms": args.pause_ms}
    print(f"  pause floor {args.pause_ms:.0f}ms\n")
    print(f"  {'arm':<20}{'nPause':>8}{'durMs':>9}{'remFrac':>9}{'remLast':>9}"
          f"{'spdRatio':>10}{'turnPause':>10}{'turnBase':>10}{'corrD':>8}"
          f"{'overshoot':>10}{'n':>7}")

    rng = np.random.default_rng(args.seed)
    lengths = np.load("training/events_len.npy")
    pick = np.sort(rng.choice(len(lengths), args.n * 2, replace=False))
    s2 = np.load("training/events_s2.npy", mmap_mode="r")[pick]
    dth = np.load("training/events_dth.npy", mmap_mode="r")[pick]
    hdt = np.load("training/events_dt.npy", mmap_mode="r")[pick].astype(np.float64)
    conds = np.load("training/events_cond.npy")[pick]
    L = np.minimum(lengths[pick], 256)
    hs = s2_to_class(torch.from_numpy(s2.astype(np.int64))).numpy()
    hth = np.where(s2 > 0,
                   dth_lattice_to_class(torch.from_numpy(dth.astype(np.int64))).numpy(),
                   TH_NULL_CLASS)
    _measure("human",
             ((hs[i], hth[i], hdt[i], int(L[i]),
               math.atan2(float(conds[i, 3]), float(conds[i, 2])))
              for i in range(min(args.n, len(L)))), args.pause_ms, out)

    specs = make_specs(args.n, args.seed)
    rows, angs = [], []
    for sx, sy, ex, ey in specs:
        dd = math.hypot(ex - sx, ey - sy)
        if dd < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([math.log(dd), math.log(esp._duration.sample(math.log(dd))),
                     math.cos(ang), math.sin(ang)])
        angs.append(ang)

    def collect(fn, label):
        S, T, DT, N, A = [], [], [], [], []
        for c0 in range(0, len(rows), args.batch):
            cond = torch.tensor(rows[c0:c0 + args.batch], dtype=torch.float32,
                                device=dev)
            s_cls, th_cls, dt_ms = fn(cond)
            s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
            pad = s_np >= S_PAD_CLASS
            S.append(s_np)
            T.append(th_np)
            DT.append(dt_ms.cpu().numpy())
            N.append(np.where(pad.any(1), pad.argmax(1), s_np.shape[1]))
            A.extend(angs[c0:c0 + args.batch])
        s_np, th_np = np.concatenate(S), np.concatenate(T)
        dt_np, n = np.concatenate(DT), np.concatenate(N)
        _measure(label, ((s_np[i], th_np[i], dt_np[i], int(n[i]), A[i])
                         for i in range(len(n))), args.pause_ms, out)

    if args.masked:
        m, seq_len = esp._model, esp._cfg["max_seq_len"]

        def masked_fn(cond):
            with torch.no_grad():
                dt_z, s_cls, th_cls = m.sample(
                    cond, seq_len, n_steps=100, temperature=args.temp,
                    order="gumbel", choice_temp=10.0,
                    feat=torch.zeros(cond.shape[0], esp._FEAT_BANK.shape[1],
                                     device=dev) if esp._FEAT_BANK is not None else None)
            return s_cls, th_cls, torch.exp(dt_z * esp._DT_STD + esp._DT_MEAN)
        collect(masked_fn, "masked served")

    for ck_name in [c for c in args.ar.split(",") if c]:
        ck = torch.load(f"training/{ck_name}", map_location=dev,
                        weights_only=False)
        model = EventARModel(**ck["config"]).to(dev).eval()
        model.load_state_dict(ck["model_state_dict"])

        def ar_fn(cond, _m=model):
            a, b, c = _m.sample(cond, temperature=args.temp)
            return a, b, class_to_dt_ms(c)
        collect(ar_fn, f"ar {ck_name.replace('.pt', '')}")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  nPause   pauses per trajectory lasting at least "
          f"{args.pause_ms:.0f}ms")
    print("  remFrac  straight line distance still to go at a pause, over the")
    print("           total. Corrective pauses sit near the target, so small.")
    print("  remLast  the same for the LAST pause in each trajectory.")
    print("  spdRatio speed after over speed before. Corrections are slower.")
    print("  turnPause against turnBase: re-aiming at a pause against the same")
    print("           heading change across uninterrupted motion.")
    print("  overshoot share coming within 10 percent of the target and then")
    print("           leaving by more than 5 percent of the distance.")


if __name__ == "__main__":
    main()
