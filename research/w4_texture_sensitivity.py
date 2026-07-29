"""What scale does the contract actually read, and can texture be added on top?

Written after w4_submovement_ceiling read 0.83 on a plan that tracks a real
human path to 1 px. That number is uninterpretable without knowing how much a
1 px error is worth on ANY path, so this measures it on real human paths and
nothing else, then asks whether the missing texture can be supplied separately
from the movement it belongs to.

Part one, sensitivity. Perturb real paths, put them back on the lattice, score
them against a disjoint half of real paths. Nothing about models is involved.

Part two, composition. Fit a plan to a real path, take the leftover, and try to
give it back in ways that preserve everything except its exact alignment. If a
reversed copy of a path's OWN leftover fails, the leftover is not a process that
can be added to a smooth curve, and a two layer design that renders a continuous
plan and adds noise cannot work no matter how good the plan is.

  real, re-emitted        error diffusion is the identity on an already integer
                          path. Must be the floor or the harness is broken.
  real + s px, rounded    the sensitivity curve.
  real + s px, diffused   carrying the rounding error forward instead of
                          discarding it, which is what a device does. Says
                          whether independent rounding is the problem.
  real, k-pt smoothed     changes the actual shape by much more than a pixel.
                          The contrast that shows the metric is not reading
                          shape.
  plan + own              exact reconstruction. Control.
  plan + own reversed     same magnitude, same spectrum, same lag structure,
                          wrong alignment.
  plan + own rolled       second destruction, same question.
  plan + swap             another path's leftover, tiled or cut, NEVER
                          stretched. Stretching changes the frequency content
                          relative to the sampling grid and cannot answer this.
  plan + swap rescaled    the same at the recipient's own leftover size.
  plan + speed-scaled     white noise tracking the plan's local speed, the
                          cheapest heteroscedastic model.

No generation, no GPU, no checkpoint read or written.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w4_texture_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

OUT = R / "research" / "w4_texture_sensitivity_results.json"
NSUB = 6
NOISE = [0.25, 0.5, 1.0, 2.0]


def diffuse(xy):
    """Continuous positions to an integer path, carrying the error forward.

    Independent rounding gives every sample its own error and turns a slow ramp
    into a staircase with risers in the middle of straight runs, which this repo
    already met on 2026-07-26 inside correct_additive without naming the general
    form. This is the alternative a device actually implements.
    """
    out = np.empty_like(xy)
    cur = np.rint(xy[0])
    out[0] = cur
    e = xy[0] - cur
    for k in range(1, len(xy)):
        d = xy[k] - xy[k - 1] + e
        s = np.rint(d)
        cur = cur + s
        out[k] = cur
        e = d - s
    return out


def smooth(p, k):
    w = np.ones(k) / k
    xy = np.c_[np.convolve(p[:, 0], w, "same"), np.convolve(p[:, 1], w, "same")]
    xy[:k], xy[-k:] = p[:k, :2], p[-k:, :2]
    return np.c_[np.rint(xy), p[:, 2]]


def tile_to(r, k):
    """Another path's leftover at k samples, tiled or cut. Never stretched."""
    if len(r) >= k:
        return r[:k]
    return np.vstack([r] * (k // len(r) + 1))[:k]


def _fit(p):
    from w4_submovement_ceiling import fit_one, plan_xy
    try:
        params, rms = fit_one(p, NSUB)
        return plan_xy(p, params), float(rms)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t_start = time.time()
    from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
    import scoring  # noqa: E402
    from w3_raw_column_reread import subset_auc  # noqa: E402

    rng = np.random.default_rng(args.seed)
    ref = [np.asarray(p) for p in real_paths(args.n_paths, args.seed, "ref")
           if len(p) >= 8]
    hold = real_paths(args.n_paths, args.seed, "holdout")
    cols = list(range(18))
    Xh = features_with_jitter(hold, 0.0, args.seed)
    Xh = Xh[np.all(np.isfinite(Xh), axis=1)]

    def score(paths):
        X = features_with_jitter(paths, 0.0, args.seed)
        X = X[np.all(np.isfinite(X), axis=1)]
        return (float(subset_auc(X, Xh, cols)),
                float(scoring.score_features(X)["auc_rf_oob"]))

    arms = {"real (floor)": ref,
            "real, re-emitted": [np.c_[diffuse(p[:, :2]), p[:, 2]] for p in ref]}
    for s in NOISE:
        q = [p[:, :2] + rng.normal(0, s, size=(len(p), 2)) for p in ref]
        arms[f"real + {s} px, rounded"] = [np.c_[np.rint(a), p[:, 2]]
                                           for a, p in zip(q, ref)]
        arms[f"real + {s} px, diffused"] = [np.c_[diffuse(a), p[:, 2]]
                                            for a, p in zip(q, ref)]
    for k in (3, 5):
        arms[f"real, {k}-pt smoothed"] = [smooth(p, k) for p in ref]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        got = list(ex.map(_fit, ref, chunksize=8))
    keep = [(p, g) for p, g in zip(ref, got) if g is not None]
    pl = [g[0] for _, g in keep]
    obs = [p for p, _ in keep]
    res = [o[:, :2] - q for o, q in zip(obs, pl)]
    sd = np.array([r.std() for r in res])
    spd = [np.r_[0.0, np.hypot(*np.diff(q, axis=0).T)] for q in pl]
    perm = rng.permutation(len(keep))
    print(f"[tex] {len(keep)} plans at N={NSUB}, fit rms median "
          f"{np.median([g[1] for _, g in keep]):.2f} px, leftover sd median "
          f"{np.median(sd):.2f} px (pure rounding would be 0.29)", flush=True)

    arms.update({
        "plan only": [np.c_[np.rint(q), o[:, 2]] for o, q in zip(obs, pl)],
        "plan + own": [np.c_[np.rint(q + r), o[:, 2]]
                       for o, q, r in zip(obs, pl, res)],
        "plan + own reversed": [np.c_[np.rint(q + r[::-1]), o[:, 2]]
                                for o, q, r in zip(obs, pl, res)],
        "plan + own rolled": [np.c_[np.rint(q + np.roll(r, len(r) // 2, 0)),
                                    o[:, 2]] for o, q, r in zip(obs, pl, res)],
        "plan + swap": [np.c_[np.rint(q + tile_to(res[j], len(q))), o[:, 2]]
                        for o, q, j in zip(obs, pl, perm)],
        "plan + swap rescaled": [
            np.c_[np.rint(q + tile_to(res[j], len(q))
                          * (sd[i] / max(sd[j], 1e-6))), o[:, 2]]
            for i, (o, q, j) in enumerate(zip(obs, pl, perm))],
        "plan + speed-scaled": [
            np.c_[np.rint(q + rng.normal(0, 1, size=q.shape)
                          * (s[:, None] * float(np.median(sd))
                             / max(s.mean(), 1e-6))), o[:, 2]]
            for o, q, s in zip(obs, pl, spd)],
    })

    out = {"n_paths": len(ref), "seed": args.seed, "n_sub": NSUB,
           "leftover_sd_median": float(np.median(sd)), "arms": {}}
    print(f"\n{'arm':<26}{'internal':>10}{'contract':>10}")
    for k, v in arms.items():
        a, b = score(v)
        out["arms"][k] = {"auc_internal": a, "auc_contract": b}
        print(f"{k:<26}{a:>10.4f}{b:>10.4f}", flush=True)

    out["wall_sec"] = time.time() - t_start
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[tex] wrote {args.out} ({out['wall_sec']:.0f}s)")


if __name__ == "__main__":
    main()
