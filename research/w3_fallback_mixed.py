"""Mixed-pool fallback probe: base and char_v3 candidates judged together.

Both pools share the same 2000-spec stream, so per spec the judge sees 64
candidates, 32 from each generator. Tests whether generator diversity moves
the corr_corr product number when either pool alone sits at 0.58-0.59.
"""
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


def load_corrected(pool_name):
    # allow_pickle: repo-own poolgen output (object-dtype trajs), not
    # third-party input.
    d = np.load(R / pool_name, allow_pickle=True)
    specs, trajs, owner = d["specs"], d["trajs"], d["owner_idx"].astype(int)
    tgt = np.round(specs).astype(int)
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
    return X[valid], owner[valid]


Xa, oa = load_corrected("pool_s42_k32.npz")
Xb, ob = load_corrected("pool_char_v3_cfg2_s42_k32.npz")
X = np.concatenate([Xa, Xb])
owner = np.concatenate([oa, ob])
print(f"[mixed] {len(Xa)} base + {len(Xb)} char = {len(X)} candidates",
      flush=True)

# interleave the two generators in draw order so a K-prefix takes K/2 from
# each; SubPool truncates rows[:k]
spec_rows = {}
rows_a, rows_b = {}, {}
for ci, idx in enumerate(oa):
    rows_a.setdefault(int(idx), []).append(ci)
for ci, idx in enumerate(ob):
    rows_b.setdefault(int(idx), []).append(len(Xa) + ci)
for idx in set(rows_a) | set(rows_b):
    a, b = rows_a.get(idx, []), rows_b.get(idx, [])
    inter = [r for pair in zip(a, b) for r in pair]
    longer = a if len(a) > len(b) else b
    inter += longer[min(len(a), len(b)):]
    spec_rows[idx] = np.asarray(inter)

ref = np.load(R / "data" / "human_ref_features_sir.npy")
ref_a = ref[np.random.default_rng(0).permutation(len(ref))[:len(ref) // 2]]
for k in (32, 64):
    aucs = []
    for seed in (0, 1, 2):
        picks = pick_sir(SubPool(X, spec_rows, k), ref_a, temp=0.7, seed=seed)
        rows = np.asarray(sorted(picks.values()))
        aucs.append(scoring.score_features(X[rows])["auc_rf_oob"])
    print(f"[mixed] K={k} corr_corr aucs={['%.4f' % a for a in aucs]} "
          f"mean={np.mean(aucs):.4f} std={np.std(aucs):.4f}", flush=True)
