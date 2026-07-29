"""Selection-seed noise band for the fallback corr_corr arm, both pools."""
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring
from features import extract_features, resample_trajectory
from selection_lab import pick_sir
from w3_fallback_arrival import correct_additive, SubPool

for pool_name in ("pool_s42_k32.npz", "pool_char_v3_cfg2_s42_k32.npz"):
    # allow_pickle: both pool files are produced by this repo's own poolgen
    # (object-dtype trajs array), never third-party input.
    d = np.load(R / pool_name, allow_pickle=True)
    specs, trajs, owner = d["specs"], d["trajs"], d["owner_idx"].astype(int)
    tgt = np.round(specs).astype(int)
    X_corr = np.full_like(d["X"], np.nan)
    for ci in range(len(trajs)):
        sx, sy, ex, ey = tgt[owner[ci]]
        t = trajs[ci]
        if t is None or len(t) < 3:
            continue
        f = extract_features(resample_trajectory(
            correct_additive(np.asarray(t), sx, sy, ex, ey)))
        if f is not None and np.all(np.isfinite(f)):
            X_corr[ci] = f
    valid = np.flatnonzero(np.all(np.isfinite(X_corr), axis=1))
    X_corr = X_corr[valid]
    spec_rows = {}
    for new_ci, ci in enumerate(valid):
        spec_rows.setdefault(int(owner[ci]), []).append(new_ci)
    spec_rows = {i: np.asarray(r) for i, r in spec_rows.items()}
    ref = np.load(R / "data" / "human_ref_features_sir.npy")
    ref_a = ref[np.random.default_rng(0).permutation(len(ref))[:len(ref) // 2]]
    for k in (8, 32):
        aucs = []
        for seed in (0, 1, 2):
            picks = pick_sir(SubPool(X_corr, spec_rows, k), ref_a,
                             temp=0.7, seed=seed)
            rows = np.asarray(sorted(picks.values()))
            aucs.append(scoring.score_features(X_corr[rows])["auc_rf_oob"])
        print(f"[seedvar] {pool_name} K={k} corr_corr "
              f"aucs={['%.4f' % a for a in aucs]} "
              f"mean={np.mean(aucs):.4f} std={np.std(aucs):.4f}", flush=True)
