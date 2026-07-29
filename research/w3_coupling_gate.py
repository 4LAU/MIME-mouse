"""What is the missing coupling actually worth? The gate before P3, take two.

research/w3_joint_structure.py established that forcing all 18 marginals to
human removes only about a third of the gap (fc_v2 0.6801 to 0.6221, resid_v2
0.6684 to 0.6059), so roughly two thirds of what remains is dependence rather
than calibration. It also named the shape of the dependence gap: on fc_v2 the
worst rank-correlation disagreements are all something against path_efficiency,
on resid_v2 all something against angular_velocity_mean, and the theme both
times is that in people how hard you move is coupled to how directly you move
and how much you turn, while in the model those couplings sit near zero.

Naming a gap is not pricing it. The gaps are modest, 0.13 to 0.18 in rank
correlation, and w3_oracle_route already showed once today that a gap this
plausible can be worth nothing. So this prices it the same way, before any
architecture is committed to.

The instrument is a Gaussian copula rewrite. Map the arm to normal scores by
rank, linearly transform so its correlation matrix matches the human's, map
back. That repairs every pairwise rank dependence while leaving the marginals
exactly as the arm produced them, so marginal and dependence effects stay
separable and the arm is never handed a value it did not generate.

Six arms, and the last two are the ones that decide anything.

  raw               where the model is.
  marginals only    every marginal forced to human, dependence left as-is.
                    Reproduces the w3_joint_structure number.
  couplings only    every pairwise dependence forced to human, marginals left
                    as the arm's. This is the price of the finding.
  both              marginals AND pairwise dependence human. If this does not
                    reach chance, the remainder is higher-order structure that
                    no pairwise account describes, which is itself a fact an
                    architecture has to face.
  one feature       dependence repaired only in the row and column of the
                    single feature the rank gaps point at. Says whether the
                    theme is the whole story or a fraction of it.
  scrambled target  the same transform aimed at a SHUFFLED human correlation
                    matrix. The transform itself must not help. If this arm
                    moves, the machinery is doing the work and nothing here is
                    readable.

Marginals are fitted on one half of the humans and every arm is scored against
the other half, because fitting and scoring on the same rows overstated the
marginals by nearly two to one in w3_joint_structure.

No generation, no GPU, no checkpoint touched.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_coupling_gate.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from w3_joint_structure import load_raw, match_marginals  # noqa: E402
from w3_raw_column_reread import subset_auc  # noqa: E402

OUT = R / "research" / "w3_coupling_gate_results.json"
CACHES = {
    "fc_v2": R / "research" / "w3_landing_cache.pkl",
    "resid_v2": R / "research" / "w3_jog_cache_event_polar_4m_resid_v2.pkl",
}
ALLC = list(range(len(FEATURE_NAMES)))
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
# the feature each checkpoint's worst rank gaps all involve, from
# w3_joint_structure's ranked pair table
THEME = {"fc_v2": "path_efficiency", "resid_v2": "angular_velocity_mean"}


def normal_scores(X):
    """Rank each column and push it through the normal quantile function."""
    n = len(X)
    q = (np.arange(n) + 0.5) / n
    z = ndtri(q)
    out = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        rank = np.empty(n, dtype=int)
        rank[np.argsort(X[:, j], kind="stable")] = np.arange(n)
        out[:, j] = z[rank]
    return out


def psd_chol(C, eps=1e-8):
    """Cholesky of the nearest PSD matrix, for targets built by splicing.

    Splicing one row and column of a human correlation matrix into an arm's can
    leave the result indefinite, which is not a numerical nuisance to be
    ignored: it means the requested dependence structure does not exist. The
    eigenvalue clip returns the closest one that does.
    """
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    if w.min() < eps:
        C = V @ np.diag(np.maximum(w, eps)) @ V.T
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
    return np.linalg.cholesky(C + eps * np.eye(len(C)))


def recouple(Xa, target_corr, marginals):
    """Give the arm `target_corr`'s pairwise dependence, keep `marginals`.

    The arm's normal scores are whitened by its own Cholesky factor and
    recoloured by the target's, which is an exact fix on the correlation matrix
    and leaves everything else about the arm alone. Values are then read back
    off whichever marginal source was asked for, so no arm ever receives a
    number that did not occur in the distribution it is supposed to have.
    """
    Za = normal_scores(Xa)
    La = psd_chol(np.corrcoef(Za, rowvar=False))
    Lt = psd_chol(target_corr)
    Zn = np.linalg.solve(La, Za.T).T @ Lt.T
    src = np.sort(marginals, axis=0)
    n = len(Xa)
    pos = np.clip(((np.arange(n) + 0.5) / n * len(src)).astype(int), 0,
                  len(src) - 1)
    out = np.empty_like(Xa)
    for j in range(Xa.shape[1]):
        rank = np.empty(n, dtype=int)
        rank[np.argsort(Zn[:, j], kind="stable")] = np.arange(n)
        out[:, j] = src[pos, j][rank]
    return out


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
    Ch = np.corrcoef(normal_scores(fit), rowvar=False)
    print(f"[gate] {len(Xr)} real paths, {len(fit)} to fit, {len(hold)} to "
          f"score against")
    floor = subset_auc(fit, hold, ALLC)
    print(f"[gate] real against real floor: {floor:.4f}")

    # the transform aimed at nonsense: a human correlation matrix with its
    # features relabelled. Same machinery, no information about this arm.
    perm = rng.permutation(len(FEATURE_NAMES))
    Cs = Ch[np.ix_(perm, perm)]

    out = {"seed": args.seed, "floor": float(floor), "theme": THEME, "arms": {}}
    for name, cache in CACHES.items():
        if not cache.exists():
            print(f"[gate] MISSING {cache}, skipping {name}")
            continue
        Xa = features_with_jitter(load_raw(cache), 0.0, args.seed)
        Xa = Xa[np.all(np.isfinite(Xa), axis=1)]
        # cap the arm at the number of human rows it is being matched to.
        # Marginal matching reads values off the sorted human half, so an arm
        # with 3x the rows lands 3 rows on every human value and the forest can
        # see the grid: fc_v2 read 0.6083 on the fully repaired arm at 5999
        # rows and 0.5899 at 2000, a free 0.018 that is pure row count. It also
        # buys nothing, since the scorer balances to the smaller side anyway.
        Xa = Xa[:len(fit)]
        Za = normal_scores(Xa)
        Ca = np.corrcoef(Za, rowvar=False)

        t = IDX[THEME[name]]
        Cone = Ca.copy()
        Cone[t, :], Cone[:, t] = Ch[t, :], Ch[:, t]
        Cone[t, t] = 1.0

        arms = {
            "raw": Xa,
            "marginals only": match_marginals(Xa, fit),
            "couplings only": recouple(Xa, Ch, Xa),
            "both": recouple(Xa, Ch, fit),
            f"couplings: {THEME[name]} only": recouple(Xa, Cone, Xa),
            "scrambled target (control)": recouple(Xa, Cs, Xa),
        }
        print(f"\n{'='*74}\n=== {name}: {len(Xa)} paths, theme feature "
              f"{THEME[name]}\n{'='*74}")
        print(f"{'arm':<34}{'AUC':>9}{'vs raw':>10}{'gap closed':>13}")
        base = subset_auc(arms["raw"], hold, ALLC)
        rec = {}
        for k, X in arms.items():
            a = subset_auc(X, hold, ALLC)
            share = (base - a) / max(base - floor, 1e-9)
            rec[k] = {"auc": float(a), "delta": float(a - base),
                      "gap_closed": float(share)}
            print(f"{k:<34}{a:>9.4f}{a-base:>10.4f}"
                  + (f"{share:>12.0%}" if k != "raw" else f"{'-':>13}"))
        out["arms"][name] = {"n": len(Xa), "base": float(base), "arms": rec}

    print(f"\n=== read ===")
    for name, a in out["arms"].items():
        c = a["arms"]["couplings only"]["gap_closed"]
        b = a["arms"]["both"]["gap_closed"]
        s = a["arms"]["scrambled target (control)"]["gap_closed"]
        print(f"{name}: repairing every pairwise coupling closes {c:.0%} of "
              f"the gap, marginals and couplings together {b:.0%}, and the "
              f"scrambled control {s:.0%}")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[gate] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
