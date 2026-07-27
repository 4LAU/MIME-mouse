"""Does the jog correction's 0.014 compose on top of a better base model?

w3_jog_verify.py replaced the endpoint correction and moved the standing arm
from 0.7283 to 0.7144, one path per spec, exact arrival, no selection. That was
measured on cached event_polar_4m_fc_v2 paths, the model behind every W3 number.

Separately, the P1 gate run of 2026-07-21 evaluated six checkpoints carrying the
resid aiming channel and none of them met its 15 px miss target. But two of them
score BELOW fc_v2 under the correction that was in service at the time:

  event_polar_4m_resid_v1   miss p50 57.3 px, additive 0.7185
  event_polar_4m_resid_v2   miss p50 55.3 px, additive 0.7203
  event_polar_4m_fc_v2      miss p50 58.0 px, additive 0.7283

Those runs saved summary JSON and not the paths, so the jog correction has never
been applied to them. Two things get tested here, and they are separable:

  compose   jog minus additive on a base model it was not developed against. A
            correction that only helps the model it was tuned on is a fit to
            that model, not a fix to the operator.
  best      the lowest honest single-trajectory number available from anything
            already on disk, with no training and no candidate selection.

The decode recipe is copied from w3_p1_eval.py verbatim so the additive column
here is comparable to the numbers above. If it is not, the run says so: the
additive AUC is printed next to the recorded one for the same checkpoint and
seed, and a drift means the recipe moved and nothing else should be read.

The paths are cached on the way out so the next probe does not need the GPU.

Generation only. The checkpoint is read, never written, and its MD5 is printed
before and after.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_jog_on_resid.py \
      --ckpt event_polar_4m_resid_v2.pt --n 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

# recorded additive AUC per checkpoint from the 2026-07-21 P1 gate, seed 42,
# n=2000, order=gumbel, resid_mode=prefix. Used only as a recipe tripwire.
RECORDED = {"event_polar_4m_resid_v1.pt": 0.7185,
            "event_polar_4m_resid_v2.pt": 0.7203,
            "event_polar_4m_resid_v6.pt": 0.7472}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_polar_4m_resid_v2.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--order", default="gumbel")
    ap.add_argument("--resid-mode", default="prefix")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = R / "training" / args.ckpt
    before = md5(ck)
    print(f"[jog] {args.ckpt} md5 {before}", flush=True)

    # locked one-shot decode recipe, copied from research/w3_p1_eval.py
    os.environ.update(EVENT_CKPT=args.ckpt, EVENT_ORDER=args.order,
                      EVENT_RESID_MODE=args.resid_mode,
                      EVENT_SNAP="2.5", EVENT_DUR_STD="1.0", DUR_EMPIRICAL="1",
                      EVENT_CHOICE_TEMP="10", EVENT_BESTOF="1", EVENT_SIR="1")
    for k in ("EVENT_POOL_TOKENS", "EVENT_POOL_SAVE", "EVENT_POOL_LOAD",
              "EVENT_RESID_ONLY"):
        os.environ.pop(k, None)

    import scoring
    from features import extract_feature_matrix
    from phase_a_baseline import make_specs
    from experiments import event_stream_polar
    from w3_aiming_price import correct_jog
    from w3_fallback_arrival import correct_additive

    specs = [(round(a), round(b), round(c), round(d))
             for a, b, c, d in make_specs(args.n, args.seed)]
    t0 = time.time()
    raw = event_stream_polar.generate_paths(specs)
    print(f"[gen] {len(raw)} paths in {time.time()-t0:.0f}s", flush=True)
    after = md5(ck)
    assert before == after, f"checkpoint changed: {before} -> {after}"
    print(f"[jog] checkpoint unchanged after generation", flush=True)

    pairs = [(s, np.asarray(t)) for s, t in zip(specs, raw)
             if t is not None and len(t) >= 3]
    miss = np.array([math.hypot(t[-1][0] - s[2], t[-1][1] - s[3])
                     for s, t in pairs])
    travel = np.array([math.hypot(s[2] - s[0], s[3] - s[1]) for s, _ in pairs])
    print(f"[miss] median {np.median(miss):.1f}px  p90 "
          f"{np.percentile(miss, 90):.1f}  "
          f"{np.median(100 * miss / travel):.1f}% of travel  "
          f"within 15px {100 * (miss <= 15).mean():.1f}%", flush=True)

    arms = {"raw": [t for _, t in pairs],
            "additive": [correct_additive(t, *s) for s, t in pairs],
            "jog": [correct_jog(t, *s) for s, t in pairs]}

    print(f"\n{'':<12}{'arrives':>10}{'n':>7}{'contract AUC':>15}")
    res = {}
    for name, trajs in arms.items():
        hit = sum(1 for p, (s, _) in zip(trajs, pairs)
                  if p[0][0] == s[0] and p[0][1] == s[1]
                  and p[-1][0] == s[2] and p[-1][1] == s[3])
        arr = hit / len(trajs)
        r = scoring.score_features(extract_feature_matrix(trajs))
        res[name] = {"auc": float(r["auc_rf_oob"]),
                     "n_per_class": int(r["n_per_class"]),
                     "exact_arrival": arr,
                     "collapse_flag": bool(r["collapse_flag"]),
                     "collapse_features": list(r["collapse_features"])}
        print(f"{name:<12}{arr:>10.1%}{r['n_per_class']:>7}"
              f"{r['auc_rf_oob']:>15.4f}")

    gap = res["jog"]["auc"] - res["additive"]["auc"]
    print(f"\njog minus additive {gap:+.4f}  "
          f"(fc_v2 arm moved -0.0139 on the same change)")

    rec = RECORDED.get(args.ckpt)
    if rec is not None:
        drift = res["additive"]["auc"] - rec
        print(f"recipe tripwire: additive {res['additive']['auc']:.4f} against "
              f"{rec:.4f} on record, drift {drift:+.4f}"
              + ("" if abs(drift) < 0.005 else "   RECIPE MOVED, stop here"))

    print(f"\ncollapse flag: additive {res['additive']['collapse_flag']}, "
          f"jog {res['jog']['collapse_flag']}, same features "
          f"{res['additive']['collapse_features'] == res['jog']['collapse_features']}")

    stem = Path(args.ckpt).stem
    cache = R / "research" / f"w3_jog_cache_{stem}.pkl"
    with open(cache, "wb") as fh:
        pickle.dump(([np.asarray(s) for s, _ in pairs],
                     [t for _, t in pairs]), fh)
    print(f"[jog] cached {len(pairs)} paths to {cache.name}")

    out = args.out or str(R / "research" / f"w3_jog_on_resid_{stem}.json")
    json.dump({"ckpt": args.ckpt, "n": args.n, "seed": args.seed,
               "order": args.order, "resid_mode": args.resid_mode,
               "ckpt_md5": before, "usable": len(pairs),
               "miss_median_px": float(np.median(miss)),
               "miss_p90_px": float(np.percentile(miss, 90)),
               "miss_pct_travel_median": float(np.median(100 * miss / travel)),
               "within_15px_pct": float(100 * (miss <= 15).mean()),
               "arms": res, "jog_minus_additive": gap,
               "recorded_additive": rec,
               "wall_sec": time.time() - t0}, open(out, "w"), indent=1)
    print(f"[jog] wrote {out}")


if __name__ == "__main__":
    main()
