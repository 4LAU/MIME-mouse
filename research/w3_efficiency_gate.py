"""How much of the duration-banded win is the speed-to-straightness coupling?

research/w3_conditional_gate.py priced duration-banded repair at about 0.03,
8 replications out of 8. research/w3_duration_response.py then named which part
of that structure is actually wrong: in the longest movements every one of the
six largest misses involves path_efficiency, where humans couple speed to
inefficiency at about -0.65 and both checkpoints reach only -0.16 to -0.26. In
people a long fast movement bows away from the straight line; in the model it
stays nearly straight however fast it goes.

Named is not priced. So this splits the 0.03 in two and finds out which half
carries it. Everything is arithmetic on output already generated.

Six arms per checkpoint, all sharing the same duration terciles, the same human
band marginals, and the same held-out human half to score against, so the only
thing that varies is WHICH couplings get repaired.

  raw                  where the model is.
  banded marginals     human band marginals, the arm's own band couplings. The
                       base every coupling arm below is read against, so the
                       marginal effect is charged once and not attributed to
                       any coupling.
  + path_efficiency    repair only path_efficiency's row and column, inside
                       each band. This is the price of the finding.
  + everything else    repair every coupling EXCEPT path_efficiency's row and
                       column. The decisive complement: if this arm carries the
                       whole win then path_efficiency was a correlate and the
                       finding is the same shape as the detour result that died
                       this morning.
  + all couplings      the full banded repair, reproducing w3_conditional_gate.
                       The two arms above have to add up to roughly this or the
                       decomposition is not a decomposition.
  random bands         the targeted repair over randomly assigned bands. Three
                       correlation matrices estimated from a third of the rows
                       each are noisier than one from all of them and noise
                       alone moves an AUC, so this carries the identical
                       estimation noise with no conditioning information. It is
                       the baseline, not the global arm.

Arms are capped at the human fit size throughout: marginal matching reads
values off the sorted human half, and an arm with more rows lands several on
every human value and leaves a grid the forest can see (worth a free 0.018 on
fc_v2 before this was caught).

No generation, no GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_efficiency_gate.py
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

from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w3_conditional_gate import NBAND, bands  # noqa: E402
from w3_coupling_gate import CACHES, normal_scores, recouple  # noqa: E402
from w3_joint_structure import load_raw  # noqa: E402
from w3_raw_column_reread import subset_auc  # noqa: E402

OUT = R / "research" / "w3_efficiency_gate_results.json"
ALLC = list(range(len(FEATURE_NAMES)))
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
TARGET = "path_efficiency"
# max_deviation measures the same thing on a different scale, so a repair that
# leaves it alone is not really leaving the arm's straightness structure alone.
# Reported separately rather than folded in, since the finding named only one.
KIN = "max_deviation"


def splice(Ca, Ch, cols):
    """Human structure in `cols` rows and columns, the arm's everywhere else."""
    C = Ca.copy()
    for c in cols:
        C[c, :], C[:, c] = Ch[c, :], Ch[:, c]
    for c in cols:
        C[c, c] = 1.0
    return C


def banded(Xa, Xh, ba, bh, mode, cols):
    """Per-band repair with a per-band target chosen by `mode`.

    Every mode remaps onto the human band's marginals, so the marginal effect
    is identical across arms and only the coupling target differs. `none` is
    therefore the base, not a no-op.
    """
    out = np.empty_like(Xa)
    for j in range(NBAND):
        ia, ih = np.flatnonzero(ba == j), np.flatnonzero(bh == j)
        if len(ia) < 40 or len(ih) < 40:
            out[ia] = Xa[ia]
            continue
        Ca = np.corrcoef(normal_scores(Xa[ia]), rowvar=False)
        Ch = np.corrcoef(normal_scores(Xh[ih]), rowvar=False)
        if mode == "none":
            C = Ca
        elif mode == "all":
            C = Ch
        elif mode == "only":
            C = splice(Ca, Ch, cols)
        elif mode == "except":
            C = splice(Ch, Ca, cols)
        else:
            raise ValueError(mode)
        out[ia] = recouple(Xa[ia], C, Xh[ih])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1234, 99])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    raw_arms = {}
    for name, cache in CACHES.items():
        if not cache.exists():
            print(f"[eff] MISSING {cache}, skipping {name}")
            continue
        X = features_with_jitter(load_raw(cache), 0.0, 42)
        raw_arms[name] = X[np.all(np.isfinite(X), axis=1)]

    cols = [IDX[TARGET]]
    cols2 = [IDX[TARGET], IDX[KIN]]
    out = {"seeds": args.seeds, "target": TARGET, "nband": NBAND, "runs": []}
    acc: dict = {}

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        Xr = features_with_jitter(real_paths(args.n_real, seed, "ref"), 0.0, seed)
        Xr = Xr[np.all(np.isfinite(Xr), axis=1)]
        h = rng.permutation(len(Xr))
        fit, hold = Xr[h[:len(Xr) // 2]], Xr[h[len(Xr) // 2:]]
        floor = subset_auc(fit, hold, ALLC)
        D = IDX["movement_duration"]
        bh = bands(fit[:, D])
        for name, X in raw_arms.items():
            Xa = X[:len(fit)]
            ba = bands(Xa[:, D])
            rb_a = np.asarray(rng.permutation(len(Xa)) % NBAND)
            rb_h = np.asarray(rng.permutation(len(fit)) % NBAND)
            arms = {
                "raw": Xa,
                "banded marginals (base)": banded(Xa, fit, ba, bh, "none", cols),
                f"+ {TARGET} only": banded(Xa, fit, ba, bh, "only", cols),
                f"+ {TARGET} and {KIN}": banded(Xa, fit, ba, bh, "only", cols2),
                "+ everything else": banded(Xa, fit, ba, bh, "except", cols),
                "+ all couplings": banded(Xa, fit, ba, bh, "all", cols),
                "random bands, target only": banded(Xa, fit, rb_a, rb_h, "only",
                                                    cols),
            }
            rec = {k: float(subset_auc(v, hold, ALLC)) for k, v in arms.items()}
            rec["floor"] = float(floor)
            out["runs"].append({"seed": seed, "arm": name, "auc": rec})
            for k, v in rec.items():
                acc.setdefault((name, k), []).append(v)
            print(f"[eff] seed {seed:>5} {name:<9} "
                  + "  ".join(f"{k.split(' ')[0][:9]}={v:.4f}"
                              for k, v in list(rec.items())[:3]), flush=True)

    print(f"\n{'':<28}" + "".join(f"{n:>14}" for n in raw_arms))
    keys = ["raw", "banded marginals (base)", f"+ {TARGET} only",
            f"+ {TARGET} and {KIN}", "+ everything else", "+ all couplings",
            "random bands, target only", "floor"]
    summary = {}
    for k in keys:
        row = {n: float(np.mean(acc[(n, k)])) for n in raw_arms}
        sd = {n: float(np.std(acc[(n, k)])) for n in raw_arms}
        summary[k] = {"mean": row, "sd": sd}
        print(f"{k:<28}" + "".join(f"{row[n]:>9.4f} sd{sd[n]:.3f}"[:14]
                                   for n in raw_arms))
    out["summary"] = summary

    print(f"\n=== read, against the random-band control ===")
    for n in raw_arms:
        base = np.mean(acc[(n, "random bands, target only")])
        for k in (f"+ {TARGET} only", "+ everything else", "+ all couplings"):
            d = np.array(acc[(n, k)]) - np.array(
                acc[(n, "random bands, target only")])
            print(f"{n:<10}{k:<26}{np.mean(d):+.4f}  "
                  f"({np.sum(d < 0)}/{len(d)} seeds helped)")
        out.setdefault("vs_control", {})[n] = {
            k: float(np.mean(np.array(acc[(n, k)]) - np.array(
                acc[(n, "random bands, target only")])))
            for k in keys if k not in ("floor", "random bands, target only")}
        del base

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[eff] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
