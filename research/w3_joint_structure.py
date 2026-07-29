"""Is the remaining gap in the marginals, or in how the features move together?

Today's controlled result is that the deficit is diffuse: no subpopulation of
human movement is missing, and every apparent localisation was a narrowing
artefact (research/w3_narrowing_audit.py). EXPERIMENTS.md reached the same
place from a completely different direction two months earlier, on the
continuous models:

  "In human data, mean_acceleration and mean_jerk are uncorrelated (r=-0.025).
   In all synthetic approaches, they're near-perfectly correlated (r=0.999).
   The RF detects this joint distribution mismatch, not individual feature
   gaps."

  "Feature importance is spread out. Top feature (angular_velocity_std) is only
   10.8%. Top 5 features = 41%. Cannot fix by targeting 1-2 features."

That was measured on DDPM and CFM arms scoring 0.92 to 0.99. Nobody has checked
whether it still holds on the discrete event-stream model at 0.645, and it
decides what a new architecture has to be good at. If the gap is marginal, each
feature can be attacked on its own and a better-calibrated head might do it. If
the gap is joint, no amount of per-feature calibration helps and the
architecture has to model the dependence structure.

One decisive instrument, and it needs no subsetting of either side, so the
narrowing artefact cannot reach it.

  marginal match   rank-transform every feature of the arm onto the human's
                   own quantiles for that feature, independently. Afterwards
                   all 18 one-dimensional marginals are exactly human by
                   construction. Score that. Whatever the forest still finds is
                   pure joint structure. The drop from the raw AUC is what the
                   marginals were worth.

Two supports.

  correlation      the 18x18 RANK correlation matrix of each arm against the
                   human's, ranked by the pairs that disagree most, with
                   mean_acceleration / mean_jerk called out by name since it is
                   the pair EXPERIMENTS.md convicted. Rank, not Pearson: these
                   features carry extreme outliers and Pearson is decided by a
                   handful of paths. Human mean_acceleration against mean_jerk
                   reads 0.9999 on Pearson, 0.193 on Spearman and -0.138 inside
                   the slowest decile. The quoted -0.025 is a Pearson figure on
                   a different sample and is not reproducible as stated.

  shuffle control  score an arm built by shuffling each human feature column
                   independently. That destroys all dependence while keeping
                   every marginal exactly right, so it is the pure joint-only
                   arm and it prices what the forest can see from dependence
                   alone on this feature set.

Raw output, both checkpoints. No generation, no GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_joint_structure.py
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

from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w3_raw_column_reread import subset_auc  # noqa: E402

OUT = R / "research" / "w3_joint_structure_results.json"
CACHES = {
    "fc_v2": R / "research" / "w3_landing_cache.pkl",
    "resid_v2": R / "research" / "w3_jog_cache_event_polar_4m_resid_v2.pkl",
}
ALLC = list(range(len(FEATURE_NAMES)))
CONVICTED = ("mean_acceleration", "mean_jerk")
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def load_raw(cache):
    """Raw model paths from a cache this repo wrote on this machine.

    pickle.load: repo-own artifact from the landing-price and jog runs, never
    third-party input.
    """
    with open(cache, "rb") as fh:
        _, trajs = pickle.load(fh)
    return [np.asarray(t, dtype=np.float64) for t in trajs
            if t is not None and len(t) >= 5]


def match_marginals(Xa, Xr):
    """Rewrite each arm column onto the human column's own quantiles.

    Rank within the arm, read the human value at that same rank. Column order
    is untouched, so every row keeps its position in every feature's ordering
    and the arm's DEPENDENCE structure survives intact while its marginals
    become exactly human. That is the whole point: it isolates one from the
    other without a model of either.
    """
    out = np.empty_like(Xa)
    hs = np.sort(Xr, axis=0)
    n, m = Xa.shape
    q = (np.arange(n) + 0.5) / n
    pos = np.clip((q * len(hs)).astype(int), 0, len(hs) - 1)
    for j in range(m):
        rank = np.empty(n, dtype=int)
        rank[np.argsort(Xa[:, j], kind="stable")] = np.arange(n)
        out[:, j] = hs[pos, j][rank]
    return out


def shuffled_human(Xr, rng):
    """Human marginals, dependence destroyed. The joint-only reference arm."""
    out = Xr.copy()
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(len(out)), j]
    return out


def _ranks(X):
    """Column ranks. These features carry extreme outliers (mean_acceleration
    spans -2.3e5 to 5.1e7 on the human reference), and Pearson on raw values is
    then decided by a handful of paths: human mean_acceleration against
    mean_jerk reads r=0.9999 on Pearson and 0.193 on Spearman, and -0.138
    inside the slowest decile. Rank correlation is the only readable version.
    """
    out = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        r = np.empty(len(X), dtype=np.float64)
        r[np.argsort(X[:, j], kind="stable")] = np.arange(len(X))
        out[:, j] = r
    return out


def corr_gap(Xa, Xr, k=8):
    """Which feature pairs move together in one and not the other, by rank."""
    ca = np.corrcoef(_ranks(Xa), rowvar=False)
    cr = np.corrcoef(_ranks(Xr), rowvar=False)
    d = np.abs(ca - cr)
    iu = np.triu_indices(len(FEATURE_NAMES), 1)
    order = np.argsort(-d[iu])[:k]
    rows = []
    for o in order:
        i, j = iu[0][o], iu[1][o]
        rows.append({"a": FEATURE_NAMES[i], "b": FEATURE_NAMES[j],
                     "r_arm": float(ca[i, j]), "r_human": float(cr[i, j]),
                     "gap": float(d[i, j])})
    fro = float(np.sqrt(np.sum((ca - cr) ** 2) / 2))
    return rows, fro, ca, cr


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
    print(f"[joint] {len(Xr)} real reference paths")

    shuf = subset_auc(shuffled_human(Xr, rng), Xr, ALLC)
    print(f"[joint] joint-only reference arm (human marginals, dependence "
          f"destroyed): {shuf:.4f}")
    print(f"[joint]   that is the ceiling on what dependence alone can be "
          f"worth on these 18 features")

    out = {"seed": args.seed, "n_real": len(Xr),
           "shuffled_human_auc": float(shuf), "arms": {}}
    for name, cache in CACHES.items():
        if not cache.exists():
            print(f"[joint] MISSING {cache}, skipping {name}")
            continue
        Xa = features_with_jitter(load_raw(cache), 0.0, args.seed)
        Xa = Xa[np.all(np.isfinite(Xa), axis=1)]
        n = min(len(Xa), len(Xr))
        Xa = Xa[:n]

        raw = subset_auc(Xa, Xr, ALLC)
        # match to one half of the humans and score against the other. Matching
        # to the same rows the forest is scored against would hand the arm the
        # reference's exact order statistics and overstate what the marginals
        # are worth.
        half = rng.permutation(len(Xr))
        fit, hold = Xr[half[:len(Xr) // 2]], Xr[half[len(Xr) // 2:]]
        Xm = match_marginals(Xa, fit)
        matched = subset_auc(Xm, hold, ALLC)
        raw_hold = subset_auc(Xa, hold, ALLC)
        print(f"[joint] against the held-out human half, before matching: "
              f"{raw_hold:.4f}")
        print(f"\n{'='*76}\n=== {name}: {n} paths\n{'='*76}")
        print(f"as it is                                  {raw:.4f}")
        print(f"with all 18 marginals forced to human     {matched:.4f}")
        print(f"  the marginals were worth                {raw-matched:+.4f}")
        print(f"  what is left is joint structure         "
              f"{matched-0.5:+.4f} above chance")

        rows, fro, ca, cr = corr_gap(Xa, Xr[:n])
        i, j = IDX[CONVICTED[0]], IDX[CONVICTED[1]]
        print(f"\nRANK correlation structure, Frobenius gap {fro:.3f}")
        print(f"{'pair':<46}{'arm':>8}{'human':>8}{'gap':>8}")
        for r in rows:
            print(f"{r['a'] + ' / ' + r['b']:<46}{r['r_arm']:>8.3f}"
                  f"{r['r_human']:>8.3f}{r['gap']:>8.3f}")
        print(f"\nthe pair EXPERIMENTS.md convicted on the continuous models:")
        print(f"{CONVICTED[0]} / {CONVICTED[1]}: arm {ca[i,j]:+.3f}, "
              f"human {cr[i,j]:+.3f}   (it was +0.999 vs -0.025 on DDPM/CFM)")

        out["arms"][name] = {
            "n": int(n), "auc_raw": float(raw),
            "auc_marginals_matched": float(matched),
            "marginals_worth": float(raw - matched),
            "joint_residual_above_chance": float(matched - 0.5),
            "corr_frobenius_gap": fro, "worst_pairs": rows,
            "convicted_pair": {"arm": float(ca[i, j]), "human": float(cr[i, j])}}

    print(f"\n=== read ===")
    for name, a in out["arms"].items():
        share = (a["auc_raw"] - a["auc_marginals_matched"]) / max(
            a["auc_raw"] - 0.5, 1e-9)
        print(f"{name}: matching every marginal exactly removes {share:.0%} of "
              f"the gap; {a['auc_marginals_matched']:.4f} survives on joint "
              f"structure alone")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[joint] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
