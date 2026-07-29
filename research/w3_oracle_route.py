"""Does route-shape tail coverage account for the gap? The gate before P3.

w3_route_variance established that the model's detour_ratio p90 is 1.69 against
the human's 3.85 in the quarter of real movement it does not cover, and that
the human's route is a free choice (R squared 0.003 on request distance). The
P3 brief now rests on that. P1 burned six fine-tunes on a prize nobody had
priced, so this prices it first, on CPU, before any epoch is spent.

Two instruments pointing in opposite directions, because each one's weakness is
the other's strength.

  ablation    THE CLEAN ONE. Real paths on both sides, nothing synthetic
              anywhere. Thin the human arm by rejection sampling until its
              detour distribution matches the MODEL's, which removes the
              wandering tail and changes nothing else, then score that arm
              through the ordinary contract. If real human movement with its
              route tail clipped scores like the model does, the tail accounts
              for the gap, and it does so without asking anyone to believe a
              synthetic excursion looks human. Ships with a null control: thin
              on duration_s instead, a descriptor the model already matches at
              0.94, which must NOT move the score. If the null control moves,
              thinning itself is doing the work and the ablation is void.

  injection   THE FORWARD ONE. Take the model's own raw paths and add a smooth
              excursion that vanishes at both endpoints, sized per path to hit
              a detour target drawn from the human distribution. Endpoints stay
              exact by construction so no correction operator is involved. The
              original speed profile is preserved step by step and duration is
              allowed to grow, because a person who wanders takes longer and
              re-timing is what stops this from becoming a speed test.

Read them together. A null on the injection alone is weak evidence, because it
cannot separate "the tail is not the gap" from "this excursion does not look
like a human one". The ablation has no such escape route: it is real data on
both sides and the only thing that changed is the route distribution.

No generation, no GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_oracle_route.py
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

from degeneracy_panel import panel, print_panel, real_paths  # noqa: E402
from w3_missing_paths import describe  # noqa: E402

OUT = R / "research" / "w3_oracle_route_results.json"
CACHE = R / "research" / "w3_landing_cache.pkl"
CACHE_RESID = R / "research" / "w3_jog_cache_event_polar_4m_resid_v2.pkl"
# what the arms are being read against, from w3_route_variance / w3_p3_fork
RAW_AUC = {"fc_v2": 0.6451, "resid_v2": 0.6534}
FLOOR = 0.50


def load_raw(cache):
    """Raw model paths from a cache this repo wrote on this machine.

    pickle.load: repo-own artifact from the landing-price and jog runs, never
    third-party input.
    """
    with open(cache, "rb") as fh:
        _, trajs = pickle.load(fh)
    return [np.asarray(t, dtype=np.float64) for t in trajs
            if t is not None and len(t) >= 5]


def raw_detour(p):
    """travelled / straight on the path as emitted, no resampling.

    Used for the bisection only, because it is called thousands of times. Every
    reported number goes through describe, which is what the record uses.
    """
    step = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    straight = float(np.hypot(p[-1, 0] - p[0, 0], p[-1, 1] - p[0, 1]))
    return (float(step.sum()) / straight if straight > 1e-9 else np.nan), straight


def excursion(p, coef):
    """A smooth detour that is exactly zero at both endpoints.

    Sine harmonics vanish at s=0 and s=1, so start and end survive untouched
    and the corrected-arrival question never arises. Applied in the frame of
    the straight line, so one axis wanders sideways and the other runs the path
    past its target and back, which is where overshoot comes from.
    """
    n = len(p)
    s = np.linspace(0.0, 1.0, n)
    sx, sy, ex, ey = p[0, 0], p[0, 1], p[-1, 0], p[-1, 1]
    d = float(np.hypot(ex - sx, ey - sy))
    if d < 1e-9:
        return np.zeros((n, 2))
    ux, uy = (ex - sx) / d, (ey - sy) / d          # along, then perpendicular
    b = np.stack([np.sin((k + 1) * np.pi * s) for k in range(len(coef[0]))])
    along, perp = coef[0] @ b, coef[1] @ b
    return np.stack([along * ux - perp * uy, along * uy + perp * ux], axis=1)


def retime(p, q):
    """Give the perturbed path the original path's speed profile.

    Without this the path travels further in the same time and the arm becomes
    a speed manipulation rather than a route one. Steps that were stationary
    stay stationary, so pauses are preserved rather than stretched.
    """
    dt = np.diff(p[:, 2])
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.008
    s0 = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    s1 = np.hypot(np.diff(q[:, 0]), np.diff(q[:, 1]))
    moving = s0 > 1e-6
    new = dt.copy()
    new[moving] = s1[moving] * dt[moving] / s0[moving]
    out = q.copy()
    out[:, 2] = p[0, 2] + np.concatenate([[0.0], np.cumsum(new)])
    return out


def inject(p, target, rng, n_harm=4, iters=24):
    """Bend one path until its detour hits `target`, endpoints untouched."""
    cur, straight = raw_detour(p)
    if not np.isfinite(cur) or target <= cur:
        return p, cur
    coef = [rng.normal(size=n_harm) / (np.arange(n_harm) + 1.0),
            rng.normal(size=n_harm) / (np.arange(n_harm) + 1.0)]
    lo, hi = 0.0, 0.5 * straight + 1.0
    for _ in range(8):                       # grow the bracket if it is short
        q = p.copy()
        q[:, :2] = p[:, :2] + hi * excursion(p, coef)
        if raw_detour(q)[0] >= target:
            break
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q = p.copy()
        q[:, :2] = p[:, :2] + mid * excursion(p, coef)
        if raw_detour(q)[0] < target:
            lo = mid
        else:
            hi = mid
    q = p.copy()
    q[:, :2] = p[:, :2] + hi * excursion(p, coef)
    return retime(p, q), raw_detour(q)[0]


def descriptor(paths, key):
    v = []
    for p in paths:
        d = describe(p)
        v.append(d[key] if d is not None and np.isfinite(d[key]) else np.nan)
    return np.array(v, dtype=np.float64)


def thin_to(src, src_v, tgt_v, bins, rng):
    """Rejection-sample `src` so its `key` distribution matches the target's.

    Rejection rather than resampling with replacement: duplicated rows would
    give the forest an easy handle and inflate the very number being read.
    """
    ok = np.isfinite(src_v) & (src_v > 0)
    tv = tgt_v[np.isfinite(tgt_v) & (tgt_v > 0)]
    edges = np.unique(np.quantile(np.concatenate([src_v[ok], tv]),
                                  np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    hs, _ = np.histogram(src_v[ok], edges)
    ht, _ = np.histogram(tv, edges)
    ps, pt = hs / max(hs.sum(), 1), ht / max(ht.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(ps > 0, pt / np.maximum(ps, 1e-12), 0.0)
    w = w / max(w.max(), 1e-12)                      # keep the richest bin whole
    idx = np.clip(np.digitize(src_v, edges) - 1, 0, len(w) - 1)
    keep = ok & (rng.random(len(src_v)) < w[idx])
    return [p for p, k in zip(src, keep) if k], int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=6000)
    ap.add_argument("--n-panel", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--model", default="fc_v2", choices=["fc_v2", "resid_v2"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    real = [np.asarray(p, dtype=np.float64)
            for p in real_paths(args.n_real, args.seed, "ref")]
    raw = load_raw(CACHE if args.model == "fc_v2" else CACHE_RESID)
    print(f"[oracle] {len(real)} human paths, {len(raw)} raw {args.model} paths")

    hd, md = descriptor(real, "detour_ratio"), descriptor(raw, "detour_ratio")
    hu, mu = descriptor(real, "duration_s"), descriptor(raw, "duration_s")
    print(f"[oracle] detour p50/p90  human {np.nanpercentile(hd,50):.2f}/"
          f"{np.nanpercentile(hd,90):.2f}   model {np.nanpercentile(md,50):.2f}/"
          f"{np.nanpercentile(md,90):.2f}")

    clipped, n_clip = thin_to(real, hd, md, args.bins, rng)
    control, n_ctrl = thin_to(real, hu, mu, args.bins, rng)
    cd = descriptor(clipped, "detour_ratio")
    print(f"[oracle] tail-clipped humans: {n_clip} kept, detour p50/p90 "
          f"{np.nanpercentile(cd,50):.2f}/{np.nanpercentile(cd,90):.2f}")
    print(f"[oracle] duration-clipped control: {n_ctrl} kept")

    finite = hd[np.isfinite(hd) & (hd > 0)]
    inj, ach = [], []
    for p in raw:
        q, d = inject(p, float(rng.choice(finite)), rng)
        inj.append(q)
        ach.append(d)
    ach = np.array(ach)
    print(f"[oracle] injected {args.model}: detour p50/p90 "
          f"{np.nanpercentile(ach,50):.2f}/{np.nanpercentile(ach,90):.2f}")
    moved = sum(1 for p, q in zip(raw, inj) if not np.allclose(p[:, :2], q[:, :2]))
    print(f"[oracle] {moved}/{len(raw)} paths bent ({moved/len(raw):.1%}); "
          f"endpoint drift max "
          f"{max(float(np.hypot(*(q[-1,:2]-p[-1,:2]))) for p, q in zip(raw, inj)):.2e} px")

    arms = {f"{args.model} raw": raw,
            f"{args.model} + injected routes": inj,
            "human, route tail clipped to model": clipped,
            "human, duration clipped (null control)": control}
    # the panel compares arms path for path, and rejection sampling leaves the
    # two human arms shorter. Truncate rather than pad: every arm is already in
    # its own random order, so the head of each is an unbiased sample of it.
    m = min(len(v) for v in arms.values())
    arms = {k: v[:m] for k, v in arms.items()}
    print(f"[oracle] panel arms truncated to {m} paths each")
    res = panel(arms, n_paths=args.n_panel, seed=args.seed)
    print_panel(res, f"Oracle: is the route tail the gap? ({args.model})")

    raw_auc = res[f"{args.model} raw"]["contract"]
    inj_auc = res[f"{args.model} + injected routes"]["contract"]
    clip_auc = res["human, route tail clipped to model"]["contract"]
    ctrl_auc = res["human, duration clipped (null control)"]["contract"]
    floor = res["real (holdout)"]["contract"]

    print(f"\n=== read ===")
    print(f"ablation: real paths with the route tail removed score "
          f"{clip_auc:.4f}, against a {floor:.4f} floor and a {raw_auc:.4f} "
          f"model. That is "
          f"{(clip_auc-floor)/max(raw_auc-floor,1e-9):.0%} of the gap "
          f"reproduced from real data alone.")
    print(f"null control: clipping duration instead moves it to {ctrl_auc:.4f} "
          f"({ctrl_auc-floor:+.4f} off the floor). If that is not small, the "
          f"ablation is thinning, not route shape.")
    print(f"injection: {raw_auc:.4f} -> {inj_auc:.4f} ({inj_auc-raw_auc:+.4f}), "
          f"{(raw_auc-inj_auc)/max(raw_auc-floor,1e-9):+.0%} of the gap closed.")

    out = {"model": args.model, "seed": args.seed, "n_real": len(real),
           "n_raw": len(raw), "n_clipped": n_clip, "n_control": n_ctrl,
           "bent_fraction": moved / len(raw), "panel": res,
           "auc": {"raw": raw_auc, "injected": inj_auc, "clipped": clip_auc,
                   "control": ctrl_auc, "floor": floor},
           "gap_share_reproduced_by_ablation":
               (clip_auc - floor) / max(raw_auc - floor, 1e-9),
           "gap_share_closed_by_injection":
               (raw_auc - inj_auc) / max(raw_auc - floor, 1e-9),
           "detour": {
               "human_p50": float(np.nanpercentile(hd, 50)),
               "human_p90": float(np.nanpercentile(hd, 90)),
               "model_p50": float(np.nanpercentile(md, 50)),
               "model_p90": float(np.nanpercentile(md, 90)),
               "clipped_p50": float(np.nanpercentile(cd, 50)),
               "clipped_p90": float(np.nanpercentile(cd, 90)),
               "injected_p50": float(np.nanpercentile(ach, 50)),
               "injected_p90": float(np.nanpercentile(ach, 90))},
           "wall_sec": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[oracle] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
