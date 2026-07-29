"""Read-only, CPU-only feasibility diagnostic for distillation.

Question: are the SELECTED trajectories a distinct, learnable region of the
model's output distribution, or statistically indistinguishable from generic
(unselected) outputs? Uses only cached features (no re-extraction, no GPU).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from features import FEATURE_NAMES

POOL_PATH = "pool_s42_k16.npz"
PICKS_PATH = "pool_s42_k16_picks_trust33_f20d85_r30_rf.npy"
HUMAN_PATH = "data/human_eval_features.npy"


def oob_auc(x0: np.ndarray, x1: np.ndarray, n_estimators: int) -> float:
    """RF OOB AUC between two feature matrices (label 0 vs 1)."""
    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))])
    rf = RandomForestClassifier(
        n_estimators=n_estimators, oob_score=True, n_jobs=-1, random_state=42
    )
    rf.fit(X, y)
    return float(roc_auc_score(y, rf.oob_decision_function_[:, 1]))


def n_duplicate_rows(x: np.ndarray) -> int:
    return len(x) - len(np.unique(x, axis=0))


def main() -> None:
    pool = np.load(POOL_PATH, allow_pickle=True)
    print(f"pool keys: {pool.files}")
    X = pool["X"]
    trajs = pool["trajs"]
    owner_idx = pool["owner_idx"]
    specs = pool["specs"]
    picks = np.load(PICKS_PATH)
    human = np.load(HUMAN_PATH)
    print(f"pool X shape: {X.shape}")
    print(f"trajs shape/dtype: {trajs.shape}, {trajs.dtype}")
    print(f"owner_idx shape/dtype: {owner_idx.shape}, {owner_idx.dtype}, "
          f"range [{owner_idx.min()}, {owner_idx.max()}], "
          f"unique conditions: {len(np.unique(owner_idx))}")
    print(f"specs shape: {specs.shape}")
    print(f"picks shape: {picks.shape}, dtype: {picks.dtype}")
    print(f"human features shape: {human.shape}")

    print()
    print("=" * 70)
    print("TEST 0: LAYOUT VERIFICATION")
    print("=" * 70)
    # Pool is NOT a clean flat c*16+k block: 7 of 2000 conditions only have
    # 15 candidates (one dropped), so owner_idx must be used, not arithmetic.
    block_sizes = np.bincount(owner_idx)
    uniq_sizes, uniq_counts = np.unique(block_sizes, return_counts=True)
    print("block-size histogram (candidates per condition):",
          dict(zip(uniq_sizes.tolist(), uniq_counts.tolist())))
    for owner in [0, 1, 2, 999, 1999]:
        rows = np.where(owner_idx == owner)[0]
        starts = np.array([trajs[i][0, :2] for i in rows])
        ends = np.array([trajs[i][-1, :2] for i in rows])
        print(f"  owner {owner}: n={len(rows)} start_std={starts.std(axis=0)} "
              f"end_std={ends.std(axis=0)} spec={specs[owner]}")
    owners_of_picks = owner_idx[picks]
    one_per_condition = np.array_equal(np.sort(owners_of_picks),
                                        np.arange(len(specs)))
    print(f"picks cover exactly one row per condition, all "
          f"{len(specs)} conditions: {one_per_condition}")

    n_pool = X.shape[0]
    mask = np.zeros(n_pool, dtype=bool)
    mask[picks] = True
    sel = X[mask]
    unsel = X[~mask]
    n_unique_picks = len(np.unique(picks))
    print(f"unique pick indices: {n_unique_picks} (raw picks: {len(picks)})")

    print()
    print("=" * 70)
    print("TEST 3: SELECTION FRACTION")
    print("=" * 70)
    print(f"pool size N          : {n_pool}")
    print(f"selected (unique)    : {n_unique_picks}")
    print(f"selection fraction   : {n_unique_picks / n_pool:.4f} "
          f"({100.0 * n_unique_picks / n_pool:.2f}%)")

    print()
    print("=" * 70)
    print("TEST 1: SEPARABILITY - SELECTED vs UNSELECTED (core test)")
    print("=" * 70)
    rng = np.random.RandomState(42)
    n_sel = len(sel)
    idx_unsel = rng.choice(len(unsel), size=min(n_sel, len(unsel)), replace=False)
    unsel_bal = unsel[idx_unsel]
    print(f"balanced set: {len(sel)} selected vs {len(unsel_bal)} unselected")

    Xb = np.vstack([unsel_bal, sel])
    yb = np.concatenate([np.zeros(len(unsel_bal)), np.ones(len(sel))])

    rf = RandomForestClassifier(
        n_estimators=200, oob_score=True, n_jobs=-1, random_state=42
    )
    rf.fit(Xb, yb)
    auc_oob = roc_auc_score(yb, rf.oob_decision_function_[:, 1])
    print(f"RF OOB AUC (sel vs unsel)    : {auc_oob:.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    proba = cross_val_predict(rf_cv, Xb, yb, cv=cv, method="predict_proba")
    auc_cv = roc_auc_score(yb, proba[:, 1])
    print(f"RF 5-fold CV AUC (sel vs unsel): {auc_cv:.4f}")

    if auc_cv >= 0.65:
        regime = "DISTINCT learnable region (good for distillation)"
    elif auc_cv <= 0.55:
        regime = "NOT distinguishable from generic output (bad for distillation)"
    else:
        regime = "AMBIGUOUS / weakly separable"
    print(f"regime: {regime}")

    print()
    print("=" * 70)
    print("TEST 2: FEATURE-DISTRIBUTION SHIFT (Cohen's d, sel vs unsel)")
    print("=" * 70)
    n_feat = X.shape[1]
    if n_feat == len(FEATURE_NAMES):
        names = list(FEATURE_NAMES)
    else:
        names = [f"col_{i}" for i in range(n_feat)]
        print(f"note: {n_feat} columns != {len(FEATURE_NAMES)} FEATURE_NAMES; "
              "using column indices")
    ds = []
    for j in range(n_feat):
        a, b = sel[:, j], unsel[:, j]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        d = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
        ds.append(d)
    order = np.argsort(np.abs(ds))[::-1]
    print("top 6 features by |Cohen's d|:")
    for j in order[:6]:
        print(f"  {names[j]:<24s} d = {ds[j]:+.4f}")
    print(f"max |d| = {np.abs(ds).max():.4f}, mean |d| = {np.abs(ds).mean():.4f}")

    print()
    print("=" * 70)
    print("TEST 4: INDIVIDUAL HUMAN-LIKENESS (crowd AUC vs human, N=2000)")
    print("=" * 70)
    auc_sel_h = oob_auc(human, sel, n_estimators=100)
    print(f"RF OOB AUC SELECTED   vs human: {auc_sel_h:.4f}  "
          f"(n_synth={len(sel)})")
    unsel_same = unsel[idx_unsel]
    auc_unsel_h = oob_auc(human, unsel_same, n_estimators=100)
    print(f"RF OOB AUC UNSELECTED vs human: {auc_unsel_h:.4f}  "
          f"(n_synth={len(unsel_same)}, same-size random sample)")

    dup_sel = n_duplicate_rows(sel)
    dup_unsel = n_duplicate_rows(unsel_same)
    print(f"exact-duplicate rows in SELECTED           : {dup_sel} / {len(sel)}")
    print(f"exact-duplicate rows in UNSELECTED sample  : {dup_unsel} / {len(unsel_same)}")

    print()
    print("=" * 70)
    print("TEST 5: CROWD AUC NULL BASELINE (random pick per condition, N=2000)")
    print("=" * 70)
    print("Is the picked-set crowd AUC actually better than just grabbing one "
          "random candidate per condition, at the same N=2000 scale?")
    n_conditions = len(specs)
    n_repeats = 30
    rng2 = np.random.default_rng(42)
    random_aucs = []
    for r in range(n_repeats):
        chosen = np.empty(n_conditions, dtype=np.int64)
        for c in range(n_conditions):
            rows = np.where(owner_idx == c)[0]
            chosen[c] = rows[rng2.integers(len(rows))]
        random_aucs.append(oob_auc(human, X[chosen], n_estimators=100))
    random_aucs = np.array(random_aucs)
    print(f"random-pick-per-condition null over {n_repeats} repeats: "
          f"mean={random_aucs.mean():.4f} std={random_aucs.std():.4f} "
          f"min={random_aucs.min():.4f} max={random_aucs.max():.4f}")
    dist_picked = abs(auc_sel_h - 0.5)
    dist_random = np.abs(random_aucs - 0.5)
    # fraction of random draws that the picked set BEATS, i.e. picked is
    # closer to 0.5 (more human-indistinguishable) than that random draw.
    percentile = (dist_picked < dist_random).mean() * 100
    z = (dist_picked - dist_random.mean()) / (dist_random.std(ddof=1) + 1e-12)
    print(f"picked-set |AUC-0.5| = {dist_picked:.4f} vs random-null "
          f"|AUC-0.5| mean={dist_random.mean():.4f} std={dist_random.std():.4f}")
    print(f"picked set is closer to 0.5 than {percentile:.1f}% of random "
          f"per-condition draws (z={z:+.2f} std below random-null mean distance)")
    if percentile >= 90:
        print("VERDICT T5: picked-set AUC is a clear outlier vs random draws -- "
              "selection is doing real work at the whole-set level.")
    elif percentile <= 60:
        print("VERDICT T5: picked-set AUC sits inside the random-draw null -- "
              "selection is not clearly beating chance at the whole-set level "
              "for this pool/seed.")
    else:
        print("VERDICT T5: picked-set AUC is somewhat better than typical "
              "random draws but not a dramatic outlier.")

    print()
    print("=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    print(f"T1 separability (sel vs unsel), OOB AUC={auc_oob:.4f}, "
          f"5-fold CV AUC={auc_cv:.4f} -> regime: {regime}")
    print(f"T4 crowd AUC: selected={auc_sel_h:.4f}, same-size random "
          f"unselected sample={auc_unsel_h:.4f}")
    print(f"T5 crowd AUC vs random-per-condition null: picked percentile "
          f"{percentile:.1f}, random mean {random_aucs.mean():.4f}")
    if auc_cv >= 0.6 and percentile >= 80:
        print("\nOVERALL: evidence supports a DISTINCT, LEARNABLE sub-style -- "
              "picks are separable from the rest of the pool in feature space "
              "AND the picked set clearly beats random per-condition draws. "
              "Fine-tuning/distillation toward this region is reasonable.")
    elif auc_cv < 0.55:
        print("\nOVERALL: evidence supports COMBINATORICS/CONDITION-MATCHING, "
              "not a distinct global sub-style -- a classifier cannot tell "
              "picks apart from the rest of the pool using cached features "
              "(CV AUC near 0.5). Whatever value the picked set has comes from "
              "choosing the right candidate PER CONDITION, not from favoring "
              "any consistent region of feature space. Distilling the model "
              "toward a fixed 'picked style' is unlikely to reproduce this "
              "effect on its own.")
    else:
        print("\nOVERALL: mixed/inconclusive -- see individual verdicts above.")

    print()
    print("done.")


if __name__ == "__main__":
    main()
