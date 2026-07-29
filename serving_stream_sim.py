"""Serve-on-demand simulation: if each request retrieves the nearest library
member by (direction, distance) and serves it integer-translated, does the
served STREAM stay near AUC 0.50?

Key simplification: translation is free and the selected trajectories are
exactly integer-valued, so an integer-translated member has IDENTICAL features
to the member itself. A served stream is therefore just a multiset of library
members, and its AUC is computable from the cached features (pool["X"][picks])
with no re-extraction. Repeats are kept (finite library, that's the point).

Nearest-match metric (stated): score = |delta log-distance| / 0.10
                                     + |circular angle diff in degrees| / 2.0
i.e. a 10% distance mismatch costs the same as a 2 degree angle mismatch,
matching the tolerances measured earlier (scale within +/-10% was ~free;
snap keeps rotation under ~0.52 only within ~2 deg).

R = 2000 requests per regime. RNG = default_rng(0), RF random_state=42.
"""
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")

RNG = np.random.default_rng(0)
R = 2000  # requests per regime

pool = np.load("pool_s42_k16.npz", allow_pickle=True)
picks = np.load("pool_s42_k16_picks_trust33_f20d85_r30_rf.npy")
human = np.load("data/human_eval_features.npy")
human_dist = np.load("data/human_distances.npy")

sel_trajs = [np.asarray(pool["trajs"][i], dtype=np.float64) for i in picks]
X_pool = pool["X"][picks]


def auc(synth_feats):
    good = np.array([f is not None and np.all(np.isfinite(f)) for f in synth_feats])
    S = np.array([f for f, g in zip(synth_feats, good) if g])
    X = np.vstack([human, S])
    y = np.concatenate([np.zeros(len(human)), np.ones(len(S))])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    return roc_auc_score(y, clf.oob_decision_function_[:, 1]), len(S)


# --- library index: native (distance, angle) of each member ----------------
lib_dist = np.empty(len(sel_trajs))
lib_ang = np.empty(len(sel_trajs))   # radians, [-pi, pi)
for i, t in enumerate(sel_trajs):
    dx = t[-1, 0] - t[0, 0]
    dy = t[-1, 1] - t[0, 1]
    lib_dist[i] = np.hypot(dx, dy)
    lib_ang[i] = np.arctan2(dy, dx)

usable = lib_dist > 1e-6
n_dropped = int(np.sum(~usable))
print(f"library: {len(sel_trajs)} members, {n_dropped} degenerate (zero net distance) excluded from index")
lib_idx = np.where(usable)[0]
L_logd = np.log(lib_dist[usable])
L_ang = lib_ang[usable]
print(f"library native distance range: {lib_dist[usable].min():.1f} .. {lib_dist[usable].max():.1f} px")
print(f"library native angle coverage: {len(lib_idx)} members over 360 deg "
      f"(~{360.0 / len(lib_idx):.2f} deg mean spacing if uniform)")
print()

W_LOGD = 0.10          # 10% distance mismatch ...
W_ANG_DEG = 2.0        # ... costs the same as 2 deg angle mismatch


def serve(req_dist, req_ang):
    """Retrieve nearest library member for each (distance, angle) request.
    Returns (member indices into picks-order, angle gaps in degrees)."""
    q_logd = np.log(req_dist)
    served = np.empty(len(req_dist), dtype=int)
    gaps = np.empty(len(req_dist))
    for k in range(len(req_dist)):
        dlog = np.abs(L_logd - q_logd[k])
        dang = np.abs(L_ang - req_ang[k])
        dang = np.minimum(dang, 2 * np.pi - dang)          # circular
        dang_deg = np.rad2deg(dang)
        score = dlog / W_LOGD + dang_deg / W_ANG_DEG
        j = int(np.argmin(score))
        served[k] = lib_idx[j]
        gaps[k] = dang_deg[j]
    return served, gaps


results = []


def run_regime(label, req_dist, req_ang):
    served, gaps = serve(req_dist, req_ang)
    a, n = auc([X_pool[i] for i in served])
    uniq = len(np.unique(served))
    med, p90 = np.median(gaps), np.percentile(gaps, 90)
    results.append((label, a, n, uniq, med, p90))
    print(f"{label:<38} AUC {a:.4f}   N {n:5d}   unique {uniq:4d}/{R}   "
          f"angle-gap median {med:.3f} deg, p90 {p90:.3f} deg")


# --- regime 1: requests ~ library's own (dist, angle) distribution ---------
idx = RNG.integers(0, len(lib_idx), size=R)
r1_dist = np.exp(L_logd[idx] + RNG.normal(0, 0.02, size=R))       # ~2% dist noise
r1_ang = L_ang[idx] + RNG.normal(0, np.deg2rad(1.0), size=R)      # ~1 deg noise
run_regime("1 requests ~ library distribution", r1_dist, r1_ang)

# --- regime 2: uniform angle, distance ~ human empirical -------------------
r2_dist = RNG.choice(human_dist, size=R).astype(np.float64)
r2_dist = np.maximum(r2_dist, 1e-6)
r2_ang = RNG.uniform(-np.pi, np.pi, size=R)
run_regime("2 uniform angle, human distances", r2_dist, r2_ang)

# --- regime 3: uniform angle, log-uniform distance over library range ------
r3_dist = np.exp(RNG.uniform(L_logd.min(), L_logd.max(), size=R))
r3_ang = RNG.uniform(-np.pi, np.pi, size=R)
run_regime("3 uniform angle, log-uniform distance", r3_dist, r3_ang)

# --- reference: the library itself, once each (the 0.508 anchor) -----------
a_ref, n_ref = auc([f for f in X_pool])
print(f"{'ref  library served exactly once each':<38} AUC {a_ref:.4f}   N {n_ref:5d}")
print()

print("=== summary table ===")
print(f"{'regime':<38} {'AUC':>8} {'N':>6} {'unique':>7} {'med gap':>9} {'p90 gap':>9}")
for label, a, n, uniq, med, p90 in results:
    print(f"{label:<38} {a:8.4f} {n:6d} {uniq:7d} {med:8.3f}d {p90:8.3f}d")
print(f"{'ref  library once each':<38} {a_ref:8.4f} {n_ref:6d} {2000:7d} {0.0:8.3f}d {0.0:8.3f}d")
