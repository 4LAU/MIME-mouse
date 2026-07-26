"""Re-read the 0.58 product number with the degeneracy control.

HANDOFF_W3.md's packageable answer is the corrected-then-judged K filter: draw
32 candidates per request, force every one onto the requested pixel with the
additive correction, judge the corrected candidates, serve the winner. Base pool
reads 0.5833 at K=32 (row ...7957c1d4), and the handoff quotes 0.58 to 0.59
after seed noise.

That number is measured on paths that were all shifted and re-rounded by the
correction, which research/p3_ceiling_probe.py showed erases the exact
collinearity the contract scorer reads. research/w3_arrival_tax_control.py found
the arrival tax itself survives that control on one-shot paths, but the product
adds best-of-32 selection on top, and a judge fitted to a human reference could
be picking winners partly on the same arithmetic. This checks the number we
would actually quote.

Selection is left exactly as the product does it: the shipped judge sees plain
features, because that is what a shipped judge would see. Only the reading of
the winners changes. So the control answers "is the number we would quote
honest", not "would a smarter judge pick differently".

No generation. Reuses pool_s42_k32.npz and selection_lab.pick_sir, the same
pool and judge as the original run.

Usage:
  env PYTHONPATH=. \
    ~/venvs/mime/bin/python research/w3_product_control.py
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
from w3_fallback_arrival import SubPool, correct_additive  # noqa: E402

POOL = R / "pool_s42_k32.npz"
REF_SIR = R / "data" / "human_ref_features_sir.npy"
OUT = R / "research" / "w3_product_control_results.json"
K = 32
SIR_TEMP = 0.7
ORIGINAL_K32 = 0.5833


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

    print("[product] correcting every candidate, extracting selection "
          "features...", flush=True)
    X = np.full_like(d["X"], np.nan)
    for ci in range(len(trajs)):
        sx, sy, ex, ey = tgt[owner[ci]]
        t = trajs[ci]
        if t is None or len(t) < 3:
            continue
        f = extract_features(resample_trajectory(
            correct_additive(np.asarray(t), sx, sy, ex, ey)))
        if f is not None and np.all(np.isfinite(f)):
            X[ci] = f
    valid = np.flatnonzero(np.all(np.isfinite(X), axis=1))
    Xv = X[valid]
    spec_rows = {}
    for new_ci, ci in enumerate(valid):
        spec_rows.setdefault(int(owner[ci]), []).append(new_ci)
    spec_rows = {i: np.asarray(r) for i, r in spec_rows.items()}
    print(f"[product] {len(valid):,} valid corrected candidates in "
          f"{time.time()-t0:.0f}s", flush=True)

    ref = np.load(REF_SIR)
    ref_a = ref[np.random.default_rng(0).permutation(len(ref))[:len(ref) // 2]]

    out = {"pool": Path(args.pool).name, "K": K, "original_k32": ORIGINAL_K32,
           "seeds": {}}
    for seed in args.seeds:
        picks = pick_sir(SubPool(Xv, spec_rows, K), ref_a, temp=SIR_TEMP,
                         seed=seed)
        rows = np.asarray(sorted(picks.values()))
        # rebuild only the winners' corrected paths; the panel needs paths, not
        # the feature rows the judge worked from
        won = []
        for r in rows:
            ci = valid[r]
            sx, sy, ex, ey = tgt[owner[ci]]
            won.append(correct_additive(np.asarray(trajs[ci]), sx, sy, ex, ey))
        res = panel({"product K=32": won}, n_paths=args.n, seed=42)
        print_panel(res, f"Product K=32, judge seed {seed}, "
                         f"{len(won)} served paths")
        out["seeds"][str(seed)] = res

    print(f"\n{'judge seed':<14}{'contract':>10}{'rebuilt':>10}{'control':>10}")
    cols = ("contract", "rebuilt", "control")
    per = {c: [] for c in cols}
    for seed, res in out["seeds"].items():
        v = res["product K=32"]
        for c in cols:
            per[c].append(v[c])
        print(f"{seed:<14}{v['contract']:>10.4f}{v['rebuilt']:>10.4f}"
              f"{v['control']:>10.4f}")
    print(f"{'mean':<14}" + "".join(f"{np.mean(per[c]):>10.4f}" for c in cols))
    print(f"{'sd':<14}" + "".join(f"{np.std(per[c]):>10.4f}" for c in cols))
    out["mean"] = {c: float(np.mean(per[c])) for c in cols}
    out["sd"] = {c: float(np.std(per[c])) for c in cols}
    out["wall_sec"] = time.time() - t0

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[product] wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
