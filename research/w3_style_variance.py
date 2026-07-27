"""Is a per-path style a real axis, or a cosmetic one?

research/w3_missing_paths.py named three kinds of movement the model never
produces. The proposal that follows from that is a style drawn once per path
that the generator cannot ignore. Before any of that is built, one question
decides whether the idea has anything under it:

  of the variation in the features the detector actually reads, how much sits
  BETWEEN styles and how much sits WITHIN one?

If most of it is between, a discrete per-path label can carry it and the
direction is worth the training time. If most of it is within, styles are a
description rather than a mechanism and the latent buys nothing.

Two traps this is built to avoid, both of which would hand back a flattering
number for free.

  circular space   k-means maximizes between-cluster spread in whatever space
                   it clustered in. Clustering the 18 detector features and
                   then reporting between-cluster variance of the 18 detector
                   features would be measuring the fitting, not the data. So
                   the clustering runs on w3_missing_paths.describe's eight
                   path descriptors and the decomposition runs on the 18
                   features. Correlated, but not maximized by construction.

  no null          ANY partition of 2000 paths into k groups explains some
                   variance. Every number here is reported next to the same
                   number computed from random labels with matched group
                   sizes, so the reader can see the excess rather than the
                   raw value.

Clusters are fit on one half of the real paths and every number is computed on
the other half, assigned by nearest centroid, so nothing reports the fit.

Three readings, in order of how much they decide:

  variance    eta^2 of the style label over the 18 features, per feature and
              weighted by what the contract RF leans on, against the null.
  style mix   the arm's distribution over styles against the humans'. The
              claim under the proposal is that the model concentrates where
              humans spread. This measures it directly.
  within      the arm scored against real paths of its OWN style. If the arm
              is already close inside a style and only the mix is wrong, a
              style latent is the fix. If it is far inside every style, it is
              not. Also reported against the random-partition null, because
              conditioning on any grouping moves this number.

Nothing here trains a generator, touches a checkpoint, or uses the GPU.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_style_variance.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from degeneracy_panel import (_score_against, features_with_jitter,  # noqa: E402
                              real_paths)
from features import FEATURE_NAMES, resample_trajectory  # noqa: E402
from w3_fallback_arrival import correct_additive  # noqa: E402
from w3_missing_paths import DESCRIPTORS, describe  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_style_variance_results.json"
MIN_PER_STYLE = 60      # below this an AUC on the subset is noise, not a reading


def prep(trajs, seed):
    """Descriptor matrix, feature matrix and the surviving paths, one subset.

    Both matrices are filtered by the same mask so row i of D and row i of X
    are the same recording. Doing this separately is how the coverage probe
    desynced its split indices earlier in this program.
    """
    X = features_with_jitter(trajs, 0.0, seed)
    desc = [describe(t) for t in trajs]
    ok = np.array([d is not None for d in desc]) & np.all(np.isfinite(X), axis=1)
    D = np.array([[desc[i][c] for c in DESCRIPTORS] for i in np.flatnonzero(ok)])
    return D, X[ok], [t for t, k in zip(trajs, ok) if k]


def eta2(X, labels):
    """Share of total variance in X explained by the labels, per column.

    Standardization is irrelevant per column and cancels, so this is computed
    on the raw columns and the multivariate total is taken on standardized
    ones, where each feature counts equally rather than by its own units.
    """
    tot = ((X - X.mean(0)) ** 2).sum(0)
    within = np.zeros(X.shape[1])
    for c in np.unique(labels):
        m = labels == c
        if m.sum() > 0:
            within += ((X[m] - X[m].mean(0)) ** 2).sum(0)
    return 1.0 - within / np.maximum(tot, 1e-12)


def null_labels(labels, rng):
    """A random partition with exactly the same group sizes."""
    out = labels.copy()
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-sweep", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--k", type=int, default=0,
                    help="k for the expensive readings, 0 = best of the sweep")
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    arm_in = [correct_additive(np.asarray(t), *(int(v) for v in s)) if len(t) >= 3
              else np.asarray(t) for s, t in zip(specs, trajs)]

    Dr, Xr, real = prep(real_paths(args.n_real, args.seed, "ref"), args.seed)
    Da, Xa, arm = prep(arm_in, args.seed)
    print(f"[style] {len(Xr)} real paths, {len(Xa)} arm paths, "
          f"{len(DESCRIPTORS)} descriptors, {Xr.shape[1]} detector features",
          flush=True)

    # which features the contract RF actually leans on, so the headline is
    # weighted by what moves the score rather than treating all 18 alike
    n = min(len(Xr), len(Xa))
    clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                 oob_score=True, n_jobs=-1,
                                 random_state=scoring.RF_SEED)
    clf.fit(np.vstack([Xr[:n], Xa[:n]]),
            np.concatenate([np.zeros(n), np.ones(n)]))
    imp = clf.feature_importances_

    # clusters fit on half the real paths, every number read on the other half
    perm = rng.permutation(len(Dr))
    fit_i, ev_i = perm[:len(perm) // 2], perm[len(perm) // 2:]
    mu, sd = Dr[fit_i].mean(0), np.maximum(Dr[fit_i].std(0), 1e-12)
    Zr_fit, Zr_ev = (Dr[fit_i] - mu) / sd, (Dr[ev_i] - mu) / sd

    Xs = (Xr - Xr.mean(0)) / np.maximum(Xr.std(0), 1e-12)   # equal-weight total

    print(f"\n{'k':>3}{'eta2 total':>12}{'null':>8}{'eta2 weighted':>16}"
          f"{'null':>8}")
    sweep = []
    best_k, best_excess = args.k_sweep[0], -1.0
    for k in args.k_sweep:
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(Zr_fit)
        lab = km.predict(Zr_ev)
        e_all = eta2(Xr[ev_i], lab)
        tot = float(eta2(Xs[ev_i], lab).mean())
        wgt = float((e_all * imp).sum() / imp.sum())
        nt, nw = [], []
        for _ in range(args.n_null):
            nl = null_labels(lab, rng)
            nt.append(float(eta2(Xs[ev_i], nl).mean()))
            nw.append(float((eta2(Xr[ev_i], nl) * imp).sum() / imp.sum()))
        nt_m, nw_m = float(np.mean(nt)), float(np.mean(nw))
        print(f"{k:>3}{tot:>12.4f}{nt_m:>8.4f}{wgt:>16.4f}{nw_m:>8.4f}")
        sweep.append({"k": k, "eta2_total": tot, "eta2_total_null": nt_m,
                      "eta2_weighted": wgt, "eta2_weighted_null": nw_m,
                      "sizes": [int((lab == c).sum()) for c in range(k)]})
        if wgt - nw_m > best_excess:
            best_k, best_excess = k, wgt - nw_m

    k = args.k or best_k
    print(f"\nk = {k} for the rest "
          f"({'chosen by largest excess over null' if not args.k else 'given'})")

    km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(Zr_fit)
    lab_ev = km.predict(Zr_ev)
    e_all = eta2(Xr[ev_i], lab_ev)
    nulls = np.array([eta2(Xr[ev_i], null_labels(lab_ev, rng))
                      for _ in range(args.n_null)])
    print(f"\nper feature, held-out real paths only")
    print(f"{'feature':<24}{'RF weight':>11}{'eta2':>8}{'null':>8}{'excess':>9}")
    per_feature = {}
    for j in np.argsort(-imp):
        ex = float(e_all[j] - nulls[:, j].mean())
        per_feature[FEATURE_NAMES[j]] = {"importance": float(imp[j]),
                                         "eta2": float(e_all[j]),
                                         "null": float(nulls[:, j].mean()),
                                         "excess": ex}
        print(f"{FEATURE_NAMES[j]:<24}{imp[j]:>11.4f}{e_all[j]:>8.4f}"
              f"{nulls[:, j].mean():>8.4f}{ex:>9.4f}")

    # style mix: where the arm sits against where humans sit, on centroids
    # fit from real paths alone so the arm is measured, never accommodated
    lab_real_all = km.predict((Dr - mu) / sd)
    lab_arm = km.predict((Da - mu) / sd)
    ph = np.array([(lab_real_all == c).mean() for c in range(k)])
    pa = np.array([(lab_arm == c).mean() for c in range(k)])
    tvd = float(0.5 * np.abs(ph - pa).sum())
    print(f"\nstyle mix, share of paths in each style")
    print(f"{'style':>6}{'human':>9}{'arm':>9}{'ratio':>9}")
    for c in range(k):
        print(f"{c + 1:>6}{ph[c]:>9.3f}{pa[c]:>9.3f}"
              f"{(pa[c] / ph[c] if ph[c] > 1e-9 else float('nan')):>9.2f}")
    print(f"total variation distance {tvd:.3f}  "
          f"(0 = same mix, 1 = disjoint)")

    # within style: the arm against real paths of its own style, and the same
    # against a random partition, because ANY grouping moves this number
    print(f"\nthe arm scored against real paths of the same style")
    print(f"{'style':>6}{'arm n':>8}{'real n':>8}{'AUC':>9}{'null AUC':>10}")
    rand_real = null_labels(lab_real_all, rng)
    rand_arm = null_labels(lab_arm, rng)
    within = []
    for c in range(k):
        ia, ir = np.flatnonzero(lab_arm == c), np.flatnonzero(lab_real_all == c)
        if len(ia) < MIN_PER_STYLE or len(ir) < MIN_PER_STYLE:
            print(f"{c + 1:>6}{len(ia):>8}{len(ir):>8}"
                  f"{'too few':>9}{'':>10}")
            within.append({"style": c + 1, "n_arm": int(len(ia)),
                           "n_real": int(len(ir)), "auc": None})
            continue
        mm = min(len(ia), len(ir))
        auc = float(_score_against(Xa[ia[:mm]], Xr[ir[:mm]])["auc_rf_oob"])
        ja = np.flatnonzero(rand_arm == c)[:mm]
        jr = np.flatnonzero(rand_real == c)[:mm]
        nauc = (float(_score_against(Xa[ja], Xr[jr])["auc_rf_oob"])
                if min(len(ja), len(jr)) >= MIN_PER_STYLE else float("nan"))
        print(f"{c + 1:>6}{len(ia):>8}{len(ir):>8}{auc:>9.4f}{nauc:>10.4f}")
        within.append({"style": c + 1, "n_arm": int(len(ia)),
                       "n_real": int(len(ir)), "auc": auc, "null_auc": nauc})
    pooled = float(_score_against(Xa[:n], Xr[:n])["auc_rf_oob"])
    print(f"pooled, no styles: {pooled:.4f}")

    # can the style be decided from what the generator knows before it starts?
    # displacement and duration are the conditioning it already receives
    j_dur = DESCRIPTORS.index("duration_s")
    j_dist = DESCRIPTORS.index("straight_dist_px")
    C = Dr[:, [j_dur, j_dist]]
    base = float(max(ph))
    acc = float(cross_val_score(
        RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        C, lab_real_all, cv=5, scoring="accuracy").mean())
    print(f"\nstyle predicted from distance and duration alone: "
          f"{acc:.3f} against a {base:.3f} base rate")

    out = {"n_real": int(len(Xr)), "n_arm": int(len(Xa)), "seed": args.seed,
           "k": int(k), "sweep": sweep, "per_feature": per_feature,
           "style_mix": {"human": ph.tolist(), "arm": pa.tolist(), "tvd": tvd},
           "within_style": within, "pooled_auc": pooled,
           "style_from_conditioning": {"accuracy": acc, "base_rate": base},
           "wall_sec": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[style] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
