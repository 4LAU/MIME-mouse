"""Which speed classes does the model over-turn in?

w3_turn_floor.py found the defect and localized it on the detector's own view:
per-step turning at 125Hz matches the human at 1000+ px/s (2.61 against 2.63),
runs 1.5x hot at 400 to 1000, and 2.5x hot at 100 to 400, which is the single
largest band on both sides. It also showed the alphabet can spell a human path,
so this is a sampling problem.

The intended lever is the dtheta head's temperature, which is already
conditioned on the speed class at the same position. But the band above is in
px/s on the resample and the head's condition is a speed CLASS, which is pixels
per event. Those are not the same axis: velocity is s/dt and dt varies. So the
band cannot be translated by arithmetic, it has to be measured.

This measures, per speed class, on human events and on the arm's events:

  turn        mean |dtheta| per event, and the share of events that continue
              perfectly straight
  velocity    median s/dt, so a class can be placed against the 125Hz bands
  share       how much of each side's event mass sits in the class

The median is the wrong statistic here and the first version of this used it.
Displacements are integers, so at 1 px per event the only reachable headings are
multiples of 45 degrees and the only reachable turns are 0, 45, 90 and so on.
The median then collapses onto one lattice value and a human 0.00 against an arm
26.57 reports a ratio in the billions while saying nothing about how far apart
the two distributions are. Mean and straight-share both survive the lattice.

If the excess concentrates in a contiguous run of classes, a class-indexed
temperature has something to aim at and the sweep knows its own support. If the
excess is flat across classes, the defect is not expressible as a function of
the head's condition and this lever is the wrong one, whatever it scores.

The arm side is read off the landing cache, which stores decoded paths, not
tokens. So both sides are re-encoded through the codec: same instrument, same
merging and splitting rules, no privileged access on either side. The codec
round trip costs 0.0101 AUC (w3_turn_floor.py), which is small against the
effect and, more to the point, applies equally to both.

The arm appears twice, before and after correct_additive. That correction moves
every sample and rounds to the lattice, and w3_stall_pattern.py already caught
it manufacturing a turn signature the raw model does not have: excess turn at a
hold is 0.4 degrees raw and 2.8 after. So the correction has to be a column
here, not an assumption, or a defect in the endpoint fix gets attributed to the
sampler and the sweep chases the wrong knob.

No GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_turn_by_class.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))

from degeneracy_panel import real_paths  # noqa: E402
from event_codec import encode_events, to_polar  # noqa: E402
from models.event_stream_polar import S_BINS, S_LOG_W, S_MAX  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_turn_by_class_results.json"


def s_to_class(s):
    """Speed value -> speed class, the numpy twin of s2_to_class.

    s2_to_class rounds 0.5*log(s^2)/S_LOG_W, which is log(s)/S_LOG_W for
    positive s. Ticks are class 0 and carry no dtheta target.
    """
    k = np.round(np.log(np.maximum(s, 1.0)) / S_LOG_W).astype(np.int64)
    return np.where(s > 0, np.clip(k, 0, S_BINS) + 1, 0)


def events(traj):
    """(speed class, |dtheta| deg, velocity px/s) per motion event."""
    p = np.asarray(traj, dtype=np.float64)
    if p.ndim != 2 or len(p) < 5:
        return None
    enc = encode_events(np.round(p[:, :2]).astype(np.int64), p[:, 2])
    if enc is None:
        return None
    dt, dx, dy = enc
    s, dth, _ = to_polar(dx, dy)
    m = s > 0
    if m.sum() < 2:
        return None
    v = s[m] / np.maximum(dt[m], 1e-6)
    return np.c_[s_to_class(s[m]), np.abs(np.degrees(dth[m])), v]


def gather(paths):
    out = [events(p) for p in paths]
    a = [v for v in out if v is not None and len(v)]
    return np.concatenate(a) if a else np.empty((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    raw = [np.asarray(t) for t in trajs]
    arm = [correct_additive(np.asarray(t), *(int(v) for v in s)) if len(t) >= 3
           else np.asarray(t) for s, t in zip(specs, trajs)]
    real = [np.asarray(p, dtype=np.float64) for p in
            real_paths(args.n_real, args.seed, "ref")]
    print(f"[class] {len(arm)} arm paths, {len(real)} real paths", flush=True)

    H, A, W = gather(real), gather(arm), gather(raw)
    print(f"[class] {len(H)} human motion events, {len(A)} arm events, "
          f"{len(W)} raw events")
    print(f"[class] classes are pixels per event: class c -> "
          f"{np.exp(0 * S_LOG_W):.2f} to {S_MAX:g} px over {S_BINS} bins\n")

    # group classes so every row has enough events to read a median from
    lo = int(min(H[:, 0].min(), A[:, 0].min()))
    hi = int(max(H[:, 0].max(), A[:, 0].max()))
    print(f"{'class':<12}{'px/event':>10}{'px/s':>7}"
          f"{'mean |turn| deg':>27}{'straight share':>29}{'share':>14}")
    print(f"{'':<12}{'':>10}{'':>7}{'human':>9}{'raw':>9}{'corr':>9}"
          f"{'human':>9}{'raw':>10}{'corr':>10}{'human':>7}{'corr':>7}")
    rows = []
    c = lo
    while c <= hi:
        end = c
        while end <= hi:
            hm = (H[:, 0] >= c) & (H[:, 0] <= end)
            am = (A[:, 0] >= c) & (A[:, 0] <= end)
            wm = (W[:, 0] >= c) & (W[:, 0] <= end)
            if min(hm.sum(), am.sum(), wm.sum()) >= args.min_n:
                break
            end += 1
        if end > hi:
            break
        h, a, w = (float(H[hm, 1].mean()), float(A[am, 1].mean()),
                   float(W[wm, 1].mean()))
        hz = float((H[hm, 1] < 1e-9).mean())
        az = float((A[am, 1] < 1e-9).mean())
        wz = float((W[wm, 1] < 1e-9).mean())
        px = float(np.exp((c - 1) * S_LOG_W)), float(np.exp((end - 1) * S_LOG_W))
        vel = float(np.median(H[hm, 2]))
        hs, as_ = float(hm.mean()), float(am.mean())
        label = f"{c}" if c == end else f"{c} to {end}"
        print(f"{label:<12}{px[0]:>4.1f}-{px[1]:<5.1f}{vel:>7.0f}"
              f"{h:>9.2f}{w:>9.2f}{a:>9.2f}"
              f"{hz:>9.1%}{wz:>10.1%}{az:>10.1%}{hs:>7.1%}{as_:>7.1%}")
        rows.append({"class_lo": c, "class_hi": end,
                     "px_per_event": px, "human_median_vel": vel,
                     "human_mean_turn_deg": h, "arm_mean_turn_deg": a,
                     "raw_mean_turn_deg": w,
                     "human_straight_share": hz, "arm_straight_share": az,
                     "raw_straight_share": wz,
                     "human_share": hs, "arm_share": as_,
                     "human_n": int(hm.sum()), "arm_n": int(am.sum())})
        c = end + 1

    # where the excess mass actually is: a per-event excess of 20 degrees on
    # 0.2% of events is not the defect, whatever its ratio looks like
    print("\nexcess turning weighted by arm event share")
    for r in sorted(rows, key=lambda r: -(r["arm_mean_turn_deg"]
                                          - r["human_mean_turn_deg"])
                    * r["arm_share"])[:6]:
        lab = (f"{r['class_lo']}" if r["class_lo"] == r["class_hi"]
               else f"{r['class_lo']} to {r['class_hi']}")
        print(f"  class {lab:<10} {r['px_per_event'][0]:.1f}-"
              f"{r['px_per_event'][1]:.1f} px/event, "
              f"{r['human_median_vel']:.0f} px/s, excess "
              f"{r['arm_mean_turn_deg'] - r['human_mean_turn_deg']:+.2f} deg "
              f"on {r['arm_share']:.1%} of events, straight share "
              f"{r['human_straight_share']:.1%} -> {r['arm_straight_share']:.1%}")

    Path(args.out).write_text(json.dumps(
        {"seed": args.seed, "min_n": args.min_n, "bands": rows,
         "human_events": int(len(H)), "arm_events": int(len(A)),
         "wall_sec": time.time() - t0}, indent=2))
    print(f"\n[class] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
