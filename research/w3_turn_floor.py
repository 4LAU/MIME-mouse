"""Can the model's representation express a human path's turning at all?

w3_envelope_ceiling.py located the remaining gap in step-scale turning texture,
worth 0.102, and showed the whole-path features are worth almost nothing. That
inverts the recorded explanation for why three adversarial runs and a
fixed-statistic run all failed to move curvature: the blocker on record is that
per-position heads cannot coordinate global outcomes, and the target turns out
not to be global. So something else is stopping it.

The first candidate is the representation. The model does not emit a path, it
emits events, each one a quantized speed and a quantized turn. If that alphabet
cannot spell a human path, no amount of training pressure was ever going to
move curvature, and every one of those failures has a single cheap explanation.

Inspection alone does not settle it. TH_BINS is 256 over a full turn, so 1.4
degrees a step, which is not obviously coarse, and S_BINS is 128 log-uniform
bins over [1, 90], about 3.6 percent apart. But the alphabet is not only the bin
widths: displacements are integers, headings are re-derived by accumulating
turns, and the decoder rounds. The floor is the whole round trip, not one
constant.

So run real human paths through the model's own encode and decode and score what
comes out. Three arms, each adding one layer, so the damage is attributed rather
than lumped:

  human untouched          the 0.4922 floor
  codec round trip         encode to events, decode back, no quantization. This
                           is what merging, splitting and heading accumulation
                           cost by themselves.
  round trip + vocabulary  the same with speed and turn snapped to the model's
                           actual bins. The difference from the row above is
                           what the alphabet costs.

If the last row is detectable, the ceiling on this model is set by its alphabet
and no fine-tune can cross it. If it stays near the floor, the representation is
innocent and the failure to move curvature is a training problem, not a
expressive one.

Two distribution reads alongside, for whichever answer comes back: per-event
turning and per-step turning after the 125Hz resample, human against the arm.

No GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_turn_floor.py
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
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
from event_codec import (encode_events, from_polar, quantize_dtheta,  # noqa: E402
                         quantize_speed, to_polar)
from features import resample_trajectory  # noqa: E402
from models.event_stream_polar import S_BINS, S_MAX, TH_BINS  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_turn_floor_results.json"
HZ = 125.0


def roundtrip(traj, quantize):
    """Encode a path to events, optionally snap to the model's bins, decode.

    Mirrors the decode contract in experiments/event_stream_polar.py: heading
    accumulates through motion events and persists across ticks, positions are
    the running sum, and the result is rounded to the lattice because
    EVENT_ROUND is on in serving.
    """
    p = np.asarray(traj, dtype=np.float64)
    if p.ndim != 2 or len(p) < 5:
        return None
    xy = np.round(p[:, :2]).astype(np.int64)
    enc = encode_events(xy, p[:, 2])
    if enc is None:
        return None
    dt, dx, dy = enc
    s, dth, th0 = to_polar(dx, dy)
    if quantize:
        s = quantize_speed(s, S_BINS, S_MAX)
        dth = quantize_dtheta(dth, TH_BINS)
    ndx, ndy = from_polar(s, dth, th0)
    x = np.round(p[0, 0] + np.concatenate([[0.0], np.cumsum(ndx)]))
    y = np.round(p[0, 1] + np.concatenate([[0.0], np.cumsum(ndy)]))
    t = p[0, 2] + np.concatenate([[0.0], np.cumsum(dt)])
    return np.c_[x, y, t] if len(x) >= 5 else None


def event_turns(traj):
    """Per-event turn angles in degrees, motion events only."""
    p = np.asarray(traj, dtype=np.float64)
    if p.ndim != 2 or len(p) < 5:
        return None
    enc = encode_events(np.round(p[:, :2]).astype(np.int64), p[:, 2])
    if enc is None:
        return None
    _, dx, dy = enc
    s, dth, _ = to_polar(dx, dy)
    return np.degrees(dth[s > 0])


def step_turns(traj):
    """Per-step turn angles in degrees on the 125Hz path, the detector's view."""
    q = np.asarray(resample_trajectory(traj, hz=HZ), dtype=np.float64)
    if len(q) < 6:
        return None
    d = np.diff(q[:, :2], axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    m = n > 1e-9
    u = d[m] / n[m, None]
    if len(u) < 2:
        return None
    cross = u[:-1, 0] * u[1:, 1] - u[:-1, 1] * u[1:, 0]
    dot = (u[:-1] * u[1:]).sum(1)
    return np.degrees(np.arctan2(cross, dot))


def turn_by_speed(traj):
    """(speed, |turn|) per 125Hz step, so turning can be read against speed.

    Pooled turning is misleading here. EXPERIMENTS.md:233 records that all human
    curvature comes from moments under 5 px/s, and the model's dtheta head is
    already conditioned on the speed class at the same position. So the question
    is not whether the model turns too much overall but whether it turns at the
    right speeds.
    """
    q = np.asarray(resample_trajectory(traj, hz=HZ), dtype=np.float64)
    if len(q) < 6:
        return None
    d = np.diff(q[:, :2], axis=0)
    dt = np.maximum(np.diff(q[:, 2]), 1e-6)
    n = np.hypot(d[:, 0], d[:, 1])
    m = n > 1e-9
    if m.sum() < 2:
        return None
    u = d[m] / n[m, None]
    sp = (n[m] / dt[m])[:-1]
    cross = u[:-1, 0] * u[1:, 1] - u[:-1, 1] * u[1:, 0]
    dot = (u[:-1] * u[1:]).sum(1)
    return np.c_[sp, np.abs(np.degrees(np.arctan2(cross, dot)))]


def gather(paths, fn):
    out = [fn(p) for p in paths]
    a = [v for v in out if v is not None and len(v)]
    return np.concatenate(a) if a else np.array([])


def describe(a):
    if not len(a):
        return {}
    b = np.abs(a)
    return {"mean_abs_deg": float(b.mean()),
            "p50_abs_deg": float(np.percentile(b, 50)),
            "p90_abs_deg": float(np.percentile(b, 90)),
            "p99_abs_deg": float(np.percentile(b, 99)),
            "frac_under_1deg": float((b < 1.0).mean()),
            "frac_over_45deg": float((b > 45.0).mean())}


def score(paths, seed):
    X = features_with_jitter(paths, 0.0, seed)
    X = X[np.all(np.isfinite(X), axis=1)]
    return float(scoring.score_features(X)["auc_rf_oob"]), int(len(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    arm = [correct_additive(np.asarray(t), *(int(v) for v in s)) if len(t) >= 3
           else np.asarray(t) for s, t in zip(specs, trajs)]
    real = [np.asarray(p, dtype=np.float64) for p in
            real_paths(args.n_real, args.seed, "ref")]
    print(f"[turn] {len(arm)} arm paths, {len(real)} real paths", flush=True)
    print(f"[turn] alphabet: TH_BINS {TH_BINS} "
          f"({360.0 / TH_BINS:.3f} deg per bin), S_BINS {S_BINS} over "
          f"[1, {S_MAX:g}]", flush=True)

    rt = [q for q in (roundtrip(p, False) for p in real) if q is not None]
    rq = [q for q in (roundtrip(p, True) for p in real) if q is not None]

    print(f"\n{'':<30}{'n':>7}{'contract AUC':>15}")
    res = {}
    for name, paths in (("human untouched", real),
                        ("codec round trip", rt),
                        ("round trip + vocabulary", rq),
                        ("arm as scored", arm)):
        auc, n = score(paths, args.seed)
        res[name] = {"auc_rf_oob": auc, "n": n}
        print(f"{name:<30}{n:>7}{auc:>15.4f}")

    floor = res["human untouched"]["auc_rf_oob"]
    print(f"\ncodec costs      {res['codec round trip']['auc_rf_oob'] - floor:+.4f}")
    print(f"vocabulary costs "
          f"{res['round trip + vocabulary']['auc_rf_oob'] - res['codec round trip']['auc_rf_oob']:+.4f}")

    turns = {}
    for level, fn in (("per event", event_turns), ("per 125Hz step", step_turns)):
        h, a = describe(gather(real, fn)), describe(gather(arm, fn))
        turns[level] = {"human": h, "arm": a}
        print(f"\nturn angles, {level}")
        print(f"{'':<22}{'human':>10}{'arm':>10}{'ratio':>9}")
        for k in h:
            r = a[k] / h[k] if abs(h[k]) > 1e-12 else float("nan")
            print(f"{k:<22}{h[k]:>10.4f}{a[k]:>10.4f}{r:>9.2f}")

    # turning against speed: the marginal above cannot distinguish "turns too
    # much" from "turns at the wrong moments", and only the second matches the
    # curvature story on record
    edges = [0, 5, 25, 100, 400, 1000, np.inf]
    H = gather(real, turn_by_speed)
    A = gather(arm, turn_by_speed)
    print(f"\nmedian |turn| by step speed, 125Hz")
    print(f"{'speed px/s':<16}{'human':>9}{'arm':>9}{'ratio':>8}"
          f"{'h share':>10}{'a share':>10}")
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hm = (H[:, 0] >= lo) & (H[:, 0] < hi)
        am = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if hm.sum() < 50 or am.sum() < 50:
            continue
        h = float(np.median(H[hm, 1])); a = float(np.median(A[am, 1]))
        hs, as_ = float(hm.mean()), float(am.mean())
        bands.append({"lo": lo, "hi": None if hi == np.inf else hi,
                      "human_median_deg": h, "arm_median_deg": a,
                      "human_share": hs, "arm_share": as_})
        label = f"{lo:g} to {hi:g}" if hi != np.inf else f"{lo:g}+"
        print(f"{label:<16}{h:>9.2f}{a:>9.2f}{a/max(h,1e-9):>8.2f}"
              f"{hs:>10.1%}{as_:>10.1%}")

    Path(args.out).write_text(json.dumps(
        {"seed": args.seed, "th_bins": TH_BINS, "s_bins": S_BINS,
         "cases": res, "turns": turns, "by_speed": bands,
         "wall_sec": time.time() - t0}, indent=2))
    print(f"\n[turn] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
