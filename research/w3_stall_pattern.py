"""Which part of the stop-and-go pattern is our model getting wrong?

The 2026-05-15 synthesis in EXPERIMENTS.md says the detector's heaviest tell,
angular velocity, does not come from smooth curves. It comes from a four-part
pattern at a full stop: decelerate, hold, change heading, accelerate. WS4
measured one number off that pattern, exact-zero steps at 0.025 against a human
0.048 to 0.060, and nothing has looked at the rest of it. w3_style_variance.py
then showed angular_velocity_mean is the RF's heaviest feature at weight 0.113
and that a per-path style explains 0.026 of it, so whatever produces it is
local, which is what sent this here.

This measures all four parts on the model's paths against the humans', so the
answer is which part is broken rather than "stalls, somehow".

Everything is measured on the raw integer-pixel path at its own timestamps,
never the 125Hz resample, on both sides. Human pool recordings are whole pixels
already and the model's decode rounds (EVENT_ROUND defaults on), so the two
sides are the same kind of object. A hold is a maximal run of consecutive
samples at an identical pixel. That is the same event the codec calls a tick,
read off the output rather than the tokens, which is the form the detector
eventually sees.

Two controls, because the first version of this would have been wrong:

  turn null    paths change direction anyway. The heading change across a hold
               is reported next to the heading change across a randomly placed
               non-hold stretch of the same width in the same path. Without
               that, any path that curves at all looks like it turns at stops.
  correction   the scored arm has correct_additive applied, which rounds to the
               lattice and can create or destroy holds. Both the raw model
               output and the corrected arm are reported, so the correction
               cannot be blamed or credited by assumption.

No GPU, no checkpoint touched: the arm comes from the landing cache.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_stall_pattern.py
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

from degeneracy_panel import real_paths  # noqa: E402
from features import resample_trajectory  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_stall_pattern_results.json"
W = 3            # samples either side of a hold used to read heading and speed
HZ = 125.0


def holds(p):
    """Maximal runs of consecutive samples at an identical pixel.

    Returns (starts, ends) as half-open index ranges into p, so a run covering
    samples i..j inclusive is (i, j+1) and the pointer is stationary from t[i]
    to t[j].
    """
    xy = p[:, :2]
    same = np.all(np.diff(xy, axis=0) == 0, axis=1)
    if not same.any():
        return np.empty(0, int), np.empty(0, int)
    edge = np.diff(np.concatenate([[0], same.astype(int), [0]]))
    return np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) + 1


def heading(p, i, j):
    """Unit direction of net displacement from sample i to sample j, or None
    if the pointer did not move over that span."""
    d = p[j, :2] - p[i, :2]
    n = float(np.hypot(*d))
    return (d / n) if n > 1e-9 else None


def turn_at(p, a, b):
    """Absolute heading change in degrees across the span [a, b), read from W
    samples before a and W samples after b. None if either side is stationary
    or runs off the path."""
    if a - W < 0 or b + W >= len(p):
        return None
    h0, h1 = heading(p, a - W, a), heading(p, b, b + W)
    if h0 is None or h1 is None:
        return None
    c = float(np.clip(h0 @ h1, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def speed_over(p, i, j):
    """Mean speed in px/s from sample i to sample j along the path."""
    if j <= i:
        return None
    seg = p[i:j + 1]
    dist = float(np.hypot(*np.diff(seg[:, :2], axis=0).T).sum())
    dur = float(seg[-1, 2] - seg[0, 2])
    return dist / dur if dur > 1e-9 else None


def measure(paths, rng):
    """Per-path and per-hold statistics for one arm."""
    per_path, hold_ms, hold_len, turns, turns_null = [], [], [], [], []
    ramp_in, ramp_out = [], []
    zero_frac = []
    for tr in paths:
        p = np.asarray(tr, dtype=np.float64)
        if p.ndim != 2 or len(p) < 4 * W + 4:
            continue
        dur = float(p[-1, 2] - p[0, 2])
        if dur <= 1e-6:
            continue
        st, en = holds(p)
        n_samp = len(p) - 1
        in_hold = int(sum(en - st))
        t_hold = float(sum(p[e - 1, 2] - p[s, 2] for s, e in zip(st, en)))
        base = speed_over(p, 0, len(p) - 1)

        for s, e in zip(st, en):
            hold_len.append(int(e - s))
            hold_ms.append(float(p[e - 1, 2] - p[s, 2]) * 1000.0)
            t = turn_at(p, s, e - 1)
            if t is not None:
                turns.append(t)
            # matched null: same width, placed anywhere the pointer is moving
            for _ in range(4):
                a = int(rng.integers(W, max(len(p) - (e - s) - W - 1, W + 1)))
                b = a + int(e - s) - 1
                if b + W < len(p) and not np.any(
                        np.all(np.diff(p[a:b + 1, :2], axis=0) == 0, axis=1)
                        if b > a else False):
                    tn = turn_at(p, a, b)
                    if tn is not None:
                        turns_null.append(tn)
                    break
            v_in, v_out = speed_over(p, max(s - W, 0), s), speed_over(p, e - 1,
                                                                     min(e - 1 + W, len(p) - 1))
            if base and base > 1e-6:
                if v_in is not None:
                    ramp_in.append(v_in / base)
                if v_out is not None:
                    ramp_out.append(v_out / base)

        per_path.append({"n_holds": len(st), "samples": n_samp,
                         "hold_sample_frac": in_hold / max(n_samp, 1),
                         "hold_time_frac": t_hold / dur,
                         "holds_per_sec": len(st) / dur})

        q = np.asarray(resample_trajectory(tr, hz=HZ), dtype=np.float64)
        if len(q) > 2:
            d = np.diff(q[:, :2], axis=0)
            zero_frac.append(float(np.all(d == 0, axis=1).mean()))

    def pct(a, qs=(50, 90)):
        a = np.asarray(a, dtype=np.float64)
        return {f"p{q}": float(np.percentile(a, q)) for q in qs} if len(a) else {}

    return {
        "n_paths": len(per_path),
        "holds_per_path": float(np.mean([d["n_holds"] for d in per_path])),
        "holds_per_sec": float(np.mean([d["holds_per_sec"] for d in per_path])),
        "hold_sample_frac": float(np.mean([d["hold_sample_frac"] for d in per_path])),
        "hold_time_frac": float(np.mean([d["hold_time_frac"] for d in per_path])),
        "paths_with_no_hold": float(np.mean([d["n_holds"] == 0 for d in per_path])),
        "hold_ms_mean": float(np.mean(hold_ms)) if hold_ms else 0.0,
        "hold_ms": pct(hold_ms),
        "hold_len_mean": float(np.mean(hold_len)) if hold_len else 0.0,
        "hold_len": pct(hold_len),
        "turn_deg_mean": float(np.mean(turns)) if turns else 0.0,
        "turn_deg": pct(turns),
        "turn_null_deg_mean": float(np.mean(turns_null)) if turns_null else 0.0,
        "turn_excess_deg": (float(np.mean(turns)) - float(np.mean(turns_null))
                            if turns and turns_null else 0.0),
        "ramp_in_mean": float(np.mean(ramp_in)) if ramp_in else 0.0,
        "ramp_out_mean": float(np.mean(ramp_out)) if ramp_out else 0.0,
        "resampled_zero_step_frac": float(np.mean(zero_frac)) if zero_frac else 0.0,
        "n_holds_total": len(hold_ms),
    }


ROWS = [
    ("holds per path", "holds_per_path", "{:.2f}"),
    ("holds per second", "holds_per_sec", "{:.2f}"),
    ("paths with no hold at all", "paths_with_no_hold", "{:.1%}"),
    ("share of samples in a hold", "hold_sample_frac", "{:.1%}"),
    ("share of time in a hold", "hold_time_frac", "{:.1%}"),
    ("hold length, samples", "hold_len_mean", "{:.2f}"),
    ("hold duration, ms", "hold_ms_mean", "{:.1f}"),
    ("heading change across a hold", "turn_deg_mean", "{:.1f}"),
    ("  same, matched non-hold null", "turn_null_deg_mean", "{:.1f}"),
    ("  excess turn at a hold", "turn_excess_deg", "{:.1f}"),
    ("speed into a hold, x path mean", "ramp_in_mean", "{:.3f}"),
    ("speed out of a hold, x path mean", "ramp_out_mean", "{:.3f}"),
    ("exact-zero steps after resample", "resampled_zero_step_frac", "{:.3%}"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    raw = [np.asarray(t) for t in trajs]
    corrected = [correct_additive(np.asarray(t), *(int(v) for v in s))
                 if len(t) >= 3 else np.asarray(t) for s, t in zip(specs, trajs)]
    real = real_paths(args.n_real, args.seed, "ref")

    arms = {"human": real, "model raw": raw, "model as scored": corrected}
    res = {k: measure(v, np.random.default_rng(args.seed)) for k, v in arms.items()}
    counts = ", ".join(f"{k} {res[k]['n_paths']}" for k in arms)
    print(f"[stall] {counts} paths\n")

    print(f"{'':<34}{'human':>12}{'model raw':>12}{'as scored':>12}{'ratio':>9}")
    for label, key, fmt in ROWS:
        h, mr, mc = (res["human"][key], res["model raw"][key],
                     res["model as scored"][key])
        ratio = mc / h if abs(h) > 1e-12 else float("nan")
        print(f"{label:<34}{fmt.format(h):>12}{fmt.format(mr):>12}"
              f"{fmt.format(mc):>12}{ratio:>9.2f}")

    print(f"\nhold length and duration percentiles")
    print(f"{'':<20}{'human p50':>11}{'p90':>8}{'model p50':>11}{'p90':>8}")
    for key in ("hold_len", "hold_ms"):
        h, m = res["human"][key], res["model as scored"][key]
        print(f"{key:<20}{h.get('p50', 0):>11.1f}{h.get('p90', 0):>8.1f}"
              f"{m.get('p50', 0):>11.1f}{m.get('p90', 0):>8.1f}")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\n[stall] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
