"""What is the forest still using once averages and pairwise links are fixed?

research/w3_coupling_gate.py left the arm at 0.6083 (fc_v2) and 0.5811
(resid_v2) against a 0.5039 floor with every marginal and every pairwise rank
correlation forced to human. The forest only ever sees these 18 features, so
the survivor is not outside the feature table: it is structure inside it that a
single correlation number per pair cannot represent. The obvious candidate is
CONDITIONAL structure, where the way two features move together in people
depends on where a third one sits, and the model keeps one fixed relationship
everywhere.

Two instruments. The first asks the forest, the second tries to fix what it
names.

  what survives   leave-one-out worth and alone-AUC per feature, measured on
                  the fully repaired arm rather than the raw one. On the raw
                  arm this reads the model's obvious defects. On the repaired
                  arm those are gone by construction, so whatever still carries
                  weight is the higher-order part and nothing else.

  banded repair   redo the copula repair separately inside each tercile of a
                  conditioning feature, so the arm receives the human's
                  correlation structure AS IT VARIES with that feature instead
                  of one global average of it. Strictly richer than the global
                  repair and reduces to it when nothing varies. Scored on the
                  whole arm against the whole held-out human half, so no
                  subsetting enters the comparison and the narrowing rule in
                  HANDOFF.md does not apply.

  random bands    the same banded repair over randomly assigned bands. Three
                  18x18 correlation matrices estimated from a third of the rows
                  each are noisier than one from all of them, and noise alone
                  can move an AUC. This arm carries the identical noise and no
                  conditioning information, so it is the only honest baseline
                  for the banded arms. Anything the banded arms do not beat it
                  by is estimation noise, not structure.

  conditional r   descriptive support: for each conditioning feature, the pairs
                  whose human rank correlation changes most across its terciles,
                  next to the arm's change on the same pairs. A pair that swings
                  in people and holds still in the model is exactly the defect
                  the banded repair is built to price, and this names them.

Both checkpoints. No generation, no GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_conditional_gate.py
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
from w3_coupling_gate import CACHES, normal_scores, recouple  # noqa: E402
from w3_joint_structure import _ranks, load_raw  # noqa: E402
from w3_raw_column_reread import subset_auc  # noqa: E402

OUT = R / "research" / "w3_conditional_gate_results.json"
ALLC = list(range(len(FEATURE_NAMES)))
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
NBAND = 3
# the features worth conditioning on: the two the pairwise gaps pointed at, the
# forest's strongest single handles from the raw column re-read, and duration,
# which is the one dimension w3_p3_fork found the model already matches and so
# acts as a soft null among the real candidates.
CONDS = ["path_efficiency", "angular_velocity_mean", "angular_velocity_std",
         "velocity_skewness", "movement_duration"]


def bands(v, k=NBAND):
    """Equal-count band labels by rank, so bands are the same size everywhere."""
    o = np.argsort(v, kind="stable")
    lab = np.empty(len(v), dtype=int)
    lab[o] = (np.arange(len(v)) * k) // len(v)
    return lab


def banded_repair(Xa, Xh, ba, bh, k=NBAND):
    """Copula repair done inside each band instead of once over everything.

    Band j of the arm is repaired towards band j of the humans, which imposes
    the human dependence structure conditional on the band rather than averaged
    over it. Bands are matched by rank position, so band j means the same
    quantile range on both sides even when the two distributions differ.
    """
    out = np.empty_like(Xa)
    for j in range(k):
        ia, ih = np.flatnonzero(ba == j), np.flatnonzero(bh == j)
        if len(ia) < 40 or len(ih) < 40:
            out[ia] = Xa[ia]
            continue
        C = np.corrcoef(normal_scores(Xh[ih]), rowvar=False)
        out[ia] = recouple(Xa[ia], C, Xh[ih])
    return out


def what_survives(Xa, Xh, k=8):
    """Leave-one-out worth and alone-AUC per feature, on the repaired arm."""
    base = subset_auc(Xa, Xh, ALLC)
    rows = []
    for c in ALLC:
        rest = [x for x in ALLC if x != c]
        rows.append({"feature": FEATURE_NAMES[c],
                     "worth": float(base - subset_auc(Xa, Xh, rest)),
                     "alone": float(subset_auc(Xa, Xh, [c]))})
    rows.sort(key=lambda r: -r["worth"])
    print(f"\nwhat the forest still uses after the repair (all 18: {base:.4f})")
    print(f"{'feature':<26}{'worth':>9}{'alone':>9}")
    for r in rows[:k]:
        print(f"{r['feature']:<26}{r['worth']:>9.4f}{r['alone']:>9.4f}")
    return base, rows


def cond_swing(X, cond, k=NBAND, top=5):
    """Pairs whose rank correlation moves most across the bands of `cond`."""
    b = bands(X[:, IDX[cond]], k)
    mats = [np.corrcoef(_ranks(X[b == j]), rowvar=False) for j in range(k)]
    swing = np.max(mats, axis=0) - np.min(mats, axis=0)
    iu = np.triu_indices(len(FEATURE_NAMES), 1)
    order = np.argsort(-swing[iu])[:top]
    return [{"a": FEATURE_NAMES[iu[0][o]], "b": FEATURE_NAMES[iu[1][o]],
             "swing": float(swing[iu[0][o], iu[1][o]]),
             "by_band": [float(m[iu[0][o], iu[1][o]]) for m in mats]}
            for o in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    Xr = features_with_jitter(real_paths(args.n_real, args.seed, "ref"), 0.0,
                              args.seed)
    Xr = Xr[np.all(np.isfinite(Xr), axis=1)]
    half = rng.permutation(len(Xr))
    fit, hold = Xr[half[:len(Xr) // 2]], Xr[half[len(Xr) // 2:]]
    floor = subset_auc(fit, hold, ALLC)
    print(f"[cond] {len(fit)} human rows to fit, {len(hold)} to score against")
    print(f"[cond] real against real floor: {floor:.4f}")

    Ch = np.corrcoef(normal_scores(fit), rowvar=False)
    out = {"seed": args.seed, "floor": float(floor), "nband": NBAND,
           "conds": CONDS, "human_swing": {}, "arms": {}}

    print(f"\nwhere human pairwise structure actually varies, by conditioner")
    for c in CONDS:
        sw = cond_swing(fit, c)
        out["human_swing"][c] = sw
        print(f"  {c:<24} biggest swing {sw[0]['swing']:.3f} on "
              f"{sw[0]['a']} / {sw[0]['b']}  bands "
              + " ".join(f"{v:+.2f}" for v in sw[0]["by_band"]))

    for name, cache in CACHES.items():
        if not cache.exists():
            print(f"[cond] MISSING {cache}, skipping {name}")
            continue
        Xa = features_with_jitter(load_raw(cache), 0.0, args.seed)
        Xa = Xa[np.all(np.isfinite(Xa), axis=1)]
        # cap at the human fit size, and it matters more here than in
        # w3_coupling_gate: banded repair reads each band's values off a THIRD
        # of the human rows, so an uncapped arm lands nine rows on every human
        # value and the grid the forest can see is nine times coarser than the
        # global arm's. The first run of this probe had fc_v2 at 5999 rows and
        # every banded arm scored ABOVE its global arm, which is the artefact
        # and not a result.
        Xa = Xa[:len(fit)]
        raw = subset_auc(Xa, hold, ALLC)
        glob = recouple(Xa, Ch, fit)
        gauc = subset_auc(glob, hold, ALLC)
        print(f"\n{'='*74}\n=== {name}: {len(Xa)} paths, raw {raw:.4f}, "
              f"globally repaired {gauc:.4f}\n{'='*74}")

        arm = {"n": len(Xa), "raw": float(raw), "global_repair": float(gauc)}
        arm["survivors"] = what_survives(glob, hold)[1]

        # random bands first: it is the baseline every real conditioner is read
        # against, and printing it above them keeps the comparison honest.
        print(f"\nrepair done inside bands of one feature instead of globally")
        print(f"{'conditioner':<26}{'AUC':>9}{'vs global':>11}"
              f"{'vs random bands':>17}")
        rl = [np.asarray(rng.permutation(len(Xa)) % NBAND),
              np.asarray(rng.permutation(len(fit)) % NBAND)]
        rnd = subset_auc(banded_repair(Xa, fit, rl[0], rl[1]), hold, ALLC)
        print(f"{'RANDOM bands (control)':<26}{rnd:>9.4f}{rnd-gauc:>11.4f}"
              f"{'':>17}")
        arm["random_bands"] = float(rnd)
        arm["banded"] = {}
        for c in CONDS:
            a = subset_auc(banded_repair(Xa, fit, bands(Xa[:, IDX[c]]),
                                         bands(fit[:, IDX[c]])), hold, ALLC)
            arm["banded"][c] = {"auc": float(a), "vs_global": float(a - gauc),
                                "vs_random": float(a - rnd)}
            print(f"{c:<26}{a:>9.4f}{a-gauc:>11.4f}{a-rnd:>17.4f}")
        arm["arm_swing"] = {c: cond_swing(Xa, c) for c in CONDS}
        out["arms"][name] = arm

    print(f"\n=== read ===")
    for name, a in out["arms"].items():
        best = min(a["banded"], key=lambda c: a["banded"][c]["auc"])
        b = a["banded"][best]
        print(f"{name}: best conditioner {best}, {b['auc']:.4f}, beating the "
              f"random-band control by {-b['vs_random']:.4f} and the global "
              f"repair by {-b['vs_global']:.4f}; floor is {out['floor']:.4f}")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[cond] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
