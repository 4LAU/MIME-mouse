"""Beat the duplication wall cheaply: break exact-duplicate fingerprints with
a SMALL ON-GRID (integer) per-serve perturbation, instead of a huge serve-once
pool.

Constraint: sub-pixel jitter is fatal (+/-0.1px -> 0.69), so the perturbation
MUST keep coordinates on the integer lattice -> integer offsets only. Endpoints
(first and last raw points) are held fixed; only interior points move.

For each served instance (regime-1 multiset, nearest-match with replacement,
same seed as serving_stream_sim.py), apply an INDEPENDENT integer dither:
each interior raw point, with probability p, gets +dx,+dy each drawn from
{-1,0,+1}. Fresh resample-125Hz + extract per perturbed copy, so two serves of
the same base member now yield different feature rows.

Sweep p in {0.02, 0.05, 0.10, 0.20}, plus p=0 control (should reproduce ~0.815).

Diagnostics: unique-feature fraction (is duplication broken?), and two feature
means vs the untouched set -- direction changes (feat idx 13) and std_jerk
(feat idx 8) -- to see if the dither pushes those out of distribution.
"""
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from features import extract_features, resample_trajectory

R = 2000
IDX_STD_JERK = 8
IDX_DIRCHANGES = 13

pool = np.load("pool_s42_k16.npz", allow_pickle=True)
picks = np.load("pool_s42_k16_picks_trust33_f20d85_r30_rf.npy")
human = np.load("data/human_eval_features.npy")

sel_trajs = [np.asarray(pool["trajs"][i], dtype=np.float64) for i in picks]
X_pool = pool["X"][picks]
N_LIB = len(X_pool)


def auc_rows(rows):
    S = np.asarray(rows)
    good = np.all(np.isfinite(S), axis=1)
    S = S[good]
    X = np.vstack([human, S])
    y = np.concatenate([np.zeros(len(human)), np.ones(len(S))])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    return roc_auc_score(y, clf.oob_decision_function_[:, 1]), len(S)


# --- library (dist, angle) index -------------------------------------------
lib_dist = np.empty(N_LIB); lib_ang = np.empty(N_LIB)
for i, t in enumerate(sel_trajs):
    dx = t[-1, 0] - t[0, 0]; dy = t[-1, 1] - t[0, 1]
    lib_dist[i] = np.hypot(dx, dy); lib_ang[i] = np.arctan2(dy, dx)
usable = lib_dist > 1e-6
lib_idx = np.where(usable)[0]
L_logd = np.log(lib_dist[usable]); L_ang = lib_ang[usable]
W_LOGD, W_ANG_DEG = 0.10, 2.0


def serve(req_dist, req_ang):
    q_logd = np.log(req_dist)
    served = np.empty(len(req_dist), dtype=int)
    for k in range(len(req_dist)):
        dlog = np.abs(L_logd - q_logd[k])
        dang = np.abs(L_ang - req_ang[k]); dang = np.minimum(dang, 2 * np.pi - dang)
        score = dlog / W_LOGD + np.rad2deg(dang) / W_ANG_DEG
        served[k] = lib_idx[int(np.argmin(score))]
    return served


# --- reproduce regime-1 served multiset EXACTLY -----------------------------
RNG = np.random.default_rng(0)
idx = RNG.integers(0, len(lib_idx), size=R)
r1_dist = np.exp(L_logd[idx] + RNG.normal(0, 0.02, size=R))
r1_ang = L_ang[idx] + RNG.normal(0, np.deg2rad(1.0), size=R)
served = serve(r1_dist, r1_ang)
print(f"regime-1 served multiset: {R} rows, {len(np.unique(served))} unique base members")

# untouched-set diagnostic baselines (served multiset, no perturbation)
base_feats = X_pool[served]
base_dirchg = float(np.mean(base_feats[:, IDX_DIRCHANGES]))
base_jerk = float(np.mean(base_feats[:, IDX_STD_JERK]))
print(f"untouched served set: mean dir-changes {base_dirchg:.3f}, mean std_jerk {base_jerk:.4g}")
print()


def perturb(traj, p, rng):
    """Add integer {-1,0,+1} offsets (independent x,y) to interior points with
    probability p each. Endpoints fixed. Stays on the integer lattice."""
    out = traj.copy()
    n = len(out)
    if n <= 2:
        return out
    interior = np.arange(1, n - 1)
    hit = rng.random(len(interior)) < p
    sel = interior[hit]
    if len(sel):
        out[sel, 0] += rng.integers(-1, 2, size=len(sel))
        out[sel, 1] += rng.integers(-1, 2, size=len(sel))
    return out


def unique_fraction(feats_arr):
    finite = feats_arr[np.all(np.isfinite(feats_arr), axis=1)]
    # round to defeat float noise before uniqueness test
    keyed = np.round(finite, 6)
    _, counts = np.unique(keyed, axis=0, return_counts=True)
    # a row is "unique" if no other row shares its exact vector
    singleton_rows = int(np.sum(counts[counts == 1]))
    return singleton_rows / len(finite), len(finite)


results = []
for p in (0.0, 0.02, 0.05, 0.10, 0.20):
    pert_rng = np.random.default_rng(1234)   # deterministic dither stream per sweep
    rows = []
    for m in served:
        if p == 0.0:
            t = sel_trajs[m]
        else:
            t = perturb(sel_trajs[m], p, pert_rng)
        f = extract_features(resample_trajectory(t, hz=125.0))
        rows.append(f if f is not None else np.full(18, np.nan))
    rows = np.asarray(rows)
    a, n = auc_rows(rows)
    ufrac, nfin = unique_fraction(rows)
    finite = rows[np.all(np.isfinite(rows), axis=1)]
    mdir = float(np.mean(finite[:, IDX_DIRCHANGES]))
    mjerk = float(np.mean(finite[:, IDX_STD_JERK]))
    results.append((p, a, n, ufrac, mdir, mjerk))
    print(f"p={p:.2f}   AUC {a:.4f}   rows {n:5d}   unique-frac {ufrac:.4f}   "
          f"mean dir-chg {mdir:.3f}   mean std_jerk {mjerk:.4g}")

print()
print("=== summary table ===")
print(f"{'p':>5} {'AUC':>8} {'rows':>6} {'uniq-frac':>10} {'dir-chg':>9} {'std_jerk':>12}")
print(f"{'--':>5} {'--':>8} {'--':>6} {'--':>10} "
      f"{base_dirchg:9.3f} {base_jerk:12.4g}   <- untouched served baseline")
for p, a, n, ufrac, mdir, mjerk in results:
    print(f"{p:5.2f} {a:8.4f} {n:6d} {ufrac:10.4f} {mdir:9.3f} {mjerk:12.4g}")
