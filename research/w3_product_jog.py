"""Re-read the 0.58 selection product with the repaired arrival operator.

The packageable answer on record is the corrected-then-judged K filter: draw 32
candidates per request, force every one onto the requested pixel, judge the
corrected candidates, serve the winner. It reads 0.5833 at K=32 (row
...7957c1d4), and w3_product_control.py confirmed that number survives the
degeneracy control (contract 0.5833, control 0.5889).

Every one of those readings used correct_additive, which the 2026-07-26 finding
showed injects the defect it was being blamed for: it drifts every position and
rounds each one independently, so a straight run becomes a staircase and each
riser reads as a 45 or 90 degree turn. correct_jog spends the same error as
whole-pixel changes on the longest steps and leaves every other step byte
identical to the model's own. On one-shot paths that was worth -0.0139 on fc_v2
and -0.0223 on resid_v2.

Whether it is worth anything HERE is a separate question, because selection sits
on top. Best-of-32 already recovers the tail of the candidate distribution, and
the P2 routing test established that one-shot gains of this size do not compound
through it. So the honest prior is that this moves nothing. The run is worth one
scoring pass anyway: the number gets quoted, and it is currently quoted off a
defective operator.

Three arms, because the operator enters the product twice and the two entries
are separable:

  additive     additive-judged, additive-served. Reproduces the number on record.
  jog          jog-judged, jog-served. The product as it would now be built.
  jog-served   additive-judged, jog-served. Same winners as the first arm, better
               serving. The gap between this and "jog" is what re-judging buys
               on top of re-serving, which says whether the old judge was being
               misled by the operator or merely handed damaged paths.

All three are restricted to the specs that have a valid candidate under both
operators, so the panel compares the same requests across arms.

Selection is left exactly as the product does it: the judge sees plain features,
because that is what a shipped judge would see. No generation, no GPU, no
checkpoint touched. Reuses pool_s42_k32.npz and selection_lab.pick_sir.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_product_jog.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

from degeneracy_panel import panel, print_panel  # noqa: E402
from features import extract_features, resample_trajectory  # noqa: E402
from selection_lab import pick_sir  # noqa: E402
from w3_aiming_price import correct_jog  # noqa: E402
from w3_fallback_arrival import SubPool, correct_additive  # noqa: E402

POOL = R / "pool_s42_k32.npz"
REF_SIR = R / "data" / "human_ref_features_sir.npy"
OUT = R / "research" / "w3_product_jog_results.json"
K = 32
SIR_TEMP = 0.7
ORIGINAL_K32 = 0.5833

OPS = {"additive": correct_additive, "jog": correct_jog}


def corrected_features(trajs, tgt, owner, op):
    """Correct every candidate with `op` and extract its selection features.

    Returns (X, spec_rows) where X has one row per candidate (NaN where the
    candidate is unusable) and spec_rows maps a spec index to the rows of its
    valid candidates, in original draw order, which is what SubPool slices K
    from.
    """
    X = np.full((len(trajs), 18), np.nan)
    for ci in range(len(trajs)):
        sx, sy, ex, ey = tgt[owner[ci]]
        t = trajs[ci]
        if t is None or len(t) < 3:
            continue
        f = extract_features(resample_trajectory(
            op(np.asarray(t, dtype=np.float64), sx, sy, ex, ey)))
        if f is not None and np.all(np.isfinite(f)):
            X[ci] = f
    valid = np.flatnonzero(np.all(np.isfinite(X), axis=1))
    rows = {}
    for new_ci, ci in enumerate(valid):
        rows.setdefault(int(owner[ci]), []).append(new_ci)
    return X, valid, {i: np.asarray(r) for i, r in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # allow_pickle: repo-own poolgen output (object-dtype trajs array), the same
    # file the original fallback run scored, never third-party input.
    d = np.load(args.pool, allow_pickle=True)
    specs, trajs, owner = d["specs"], d["trajs"], d["owner_idx"].astype(int)
    tgt = np.round(specs).astype(int)
    print(f"[product] pool {Path(args.pool).name}: {len(trajs):,} candidates, "
          f"{len(specs)} specs", flush=True)

    built = {}
    for name, op in OPS.items():
        print(f"[product] correcting with {name}, extracting features...",
              flush=True)
        Xf, valid, rows = corrected_features(trajs, tgt, owner, op)
        built[name] = {"X": Xf, "valid": valid, "rows": rows}
        print(f"[product]   {len(valid):,} valid candidates, "
              f"{len(rows)} specs covered  ({time.time()-t0:.0f}s)", flush=True)

    shared = sorted(set(built["additive"]["rows"]) & set(built["jog"]["rows"]))
    print(f"[product] {len(shared)} specs valid under both operators", flush=True)

    ref = np.load(REF_SIR)
    ref_a = ref[np.random.default_rng(0).permutation(len(ref))[:len(ref) // 2]]

    def serve(picks_by_spec, source_op, serve_op):
        """Rebuild the winners' served paths, in shared-spec order."""
        b = built[source_op]
        out = []
        for si in shared:
            ci = b["valid"][picks_by_spec[si]]
            sx, sy, ex, ey = tgt[owner[ci]]
            out.append(OPS[serve_op](np.asarray(trajs[ci], dtype=np.float64),
                                     sx, sy, ex, ey))
        return out

    out = {"pool": Path(args.pool).name, "K": K, "original_k32": ORIGINAL_K32,
           "n_specs_shared": len(shared), "seeds": {}}
    for seed in args.seeds:
        picks = {name: pick_sir(SubPool(built[name]["X"][built[name]["valid"]],
                                        built[name]["rows"], K),
                                ref_a, temp=SIR_TEMP, seed=seed)
                 for name in OPS}
        arms = {"additive": serve(picks["additive"], "additive", "additive"),
                "jog": serve(picks["jog"], "jog", "jog"),
                "jog-served": serve(picks["additive"], "additive", "jog")}

        same = sum(1 for si in shared
                   if built["additive"]["valid"][picks["additive"][si]]
                   == built["jog"]["valid"][picks["jog"][si]])
        arrive = {k: sum(1 for p, si in zip(v, shared)
                         if p[-1][0] == tgt[si][2] and p[-1][1] == tgt[si][3])
                     / len(v) for k, v in arms.items()}

        res = panel(arms, n_paths=args.n, seed=42)
        print_panel(res, f"Product K=32, judge seed {seed}, "
                         f"{len(shared)} served paths")
        print(f"[product] the two judges pick the same candidate on "
              f"{same}/{len(shared)} specs ({same/len(shared):.1%})")
        print("[product] exact arrival: " + "  ".join(
            f"{k} {v:.1%}" for k, v in arrive.items()))
        res["_picks_agree"] = same / len(shared)
        res["_exact_arrival"] = arrive
        out["seeds"][str(seed)] = res

    cols = ("contract", "rebuilt", "control")
    names = list(arms)
    print(f"\n{'arm':<14}" + "".join(f"{c:>10}" for c in cols)
          + f"{'sd(contract)':>14}")
    out["mean"], out["sd"] = {}, {}
    for name in names:
        per = {c: [out["seeds"][str(s)][name][c] for s in args.seeds]
               for c in cols}
        out["mean"][name] = {c: float(np.mean(per[c])) for c in cols}
        out["sd"][name] = {c: float(np.std(per[c])) for c in cols}
        print(f"{name:<14}" + "".join(f"{np.mean(per[c]):>10.4f}" for c in cols)
              + f"{np.std(per['contract']):>14.4f}")
    fl = out["seeds"][str(args.seeds[0])]["real (holdout)"]
    print(f"{'real (floor)':<14}" + "".join(f"{fl[c]:>10.4f}" for c in cols))

    ja = out["mean"]["jog"]["contract"] - out["mean"]["additive"]["contract"]
    js = (out["mean"]["jog-served"]["contract"]
          - out["mean"]["additive"]["contract"])
    out["jog_minus_additive"] = ja
    out["jogserved_minus_additive"] = js
    print(f"\njog minus additive          {ja:+.4f}  "
          f"(one-shot moved -0.0139 on fc_v2)")
    print(f"jog-served minus additive   {js:+.4f}  "
          f"(re-serving only, same winners)")
    print(f"seed sd on the additive arm {out['sd']['additive']['contract']:.4f}"
          f"   drift vs recorded {ORIGINAL_K32}: "
          f"{out['mean']['additive']['contract'] - ORIGINAL_K32:+.4f}")
    out["wall_sec"] = time.time() - t0

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[product] wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
