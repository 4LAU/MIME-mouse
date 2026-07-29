"""Decompose the regime-1 jump to 0.815: is it the OOB-with-duplicates
artifact (the v53 "with-replacement OOB fingerprint"), or does nearest-match
retrieval itself bias the served crowd above 0.50?

Regime 1 = requests ~ library's own (dist,angle) distribution, nearest-match
with replacement. Reproduced identically here (same RNG(0), same draw order as
serving_stream_sim.py) so the served multiset matches.

TEST 1  unique-subset AUC: the distinct members served, each once. Removes
        duplicate rows -> is the retrieved SUBSET still a ~0.50 crowd?
TEST 2  duplication control: reproduce regime 1's COUNT profile but assign the
        counts to RANDOM members (no retrieval). If ~0.82, duplication alone
        explains the jump.
TEST 3  frequency-flattened: cap serves per member at 1 (==test1) and at 2.

Cached data only, no re-extraction, no new generation.
"""
import sys
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")

RNG = np.random.default_rng(0)
R = 2000

pool = np.load("pool_s42_k16.npz", allow_pickle=True)
picks = np.load("pool_s42_k16_picks_trust33_f20d85_r30_rf.npy")
human = np.load("data/human_eval_features.npy")

sel_trajs = [np.asarray(pool["trajs"][i], dtype=np.float64) for i in picks]
X_pool = pool["X"][picks]                                  # (2000, 18) cached feats
N_LIB = len(X_pool)


def auc_rows(rows, human_set=None):
    """AUC separating human_set (default full human) from a list/array of
    synthetic feature rows (duplicates allowed)."""
    H = human if human_set is None else human_set
    S = np.asarray(rows)
    good = np.all(np.isfinite(S), axis=1)
    S = S[good]
    X = np.vstack([H, S])
    y = np.concatenate([np.zeros(len(H)), np.ones(len(S))])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    return roc_auc_score(y, clf.oob_decision_function_[:, 1]), len(S)


# --- library (dist, angle) index -------------------------------------------
lib_dist = np.empty(N_LIB)
lib_ang = np.empty(N_LIB)
for i, t in enumerate(sel_trajs):
    dx = t[-1, 0] - t[0, 0]
    dy = t[-1, 1] - t[0, 1]
    lib_dist[i] = np.hypot(dx, dy)
    lib_ang[i] = np.arctan2(dy, dx)
usable = lib_dist > 1e-6
lib_idx = np.where(usable)[0]
L_logd = np.log(lib_dist[usable])
L_ang = lib_ang[usable]

W_LOGD = 0.10
W_ANG_DEG = 2.0


def serve(req_dist, req_ang):
    q_logd = np.log(req_dist)
    served = np.empty(len(req_dist), dtype=int)
    for k in range(len(req_dist)):
        dlog = np.abs(L_logd - q_logd[k])
        dang = np.abs(L_ang - req_ang[k])
        dang = np.minimum(dang, 2 * np.pi - dang)
        score = dlog / W_LOGD + np.rad2deg(dang) / W_ANG_DEG
        served[k] = lib_idx[int(np.argmin(score))]
    return served


# --- reproduce regime 1 EXACTLY (same RNG draw order as serving_stream_sim) --
idx = RNG.integers(0, len(lib_idx), size=R)
r1_dist = np.exp(L_logd[idx] + RNG.normal(0, 0.02, size=R))
r1_ang = L_ang[idx] + RNG.normal(0, np.deg2rad(1.0), size=R)
served = serve(r1_dist, r1_ang)                            # member indices (into picks order)

counts = Counter(served.tolist())
unique_members = np.array(sorted(counts.keys()))
count_vals = np.array([counts[m] for m in unique_members])
print(f"regime-1 served multiset: {R} rows, {len(unique_members)} unique members")
print(f"  count profile: served-once {np.sum(count_vals == 1)}, x2 {np.sum(count_vals == 2)}, "
      f"x3 {np.sum(count_vals == 3)}, x4+ {np.sum(count_vals >= 4)}, "
      f"never-served {N_LIB - len(unique_members)}")
print()

results = []


def report(label, rows, note=""):
    a, n = auc_rows(rows)
    uniq = len(np.unique(np.asarray(rows), axis=0))
    results.append((label, a, n, uniq, note))
    print(f"{label:<40} AUC {a:.4f}   rows {n:5d}   unique {uniq:4d}   {note}")


# --- regime-1 full reference (should match 0.8153) -------------------------
report("regime-1 full (ref)", X_pool[served], "(expect ~0.8153)")

# --- TEST 1: unique subset, each once --------------------------------------
uniq_rows = X_pool[unique_members]
report("test1 unique-subset, once each", uniq_rows, "(retrieved SET as a crowd)")

# balanced variant: subsample human to same count, random_state via RNG(42)
rng42 = np.random.default_rng(42)
hsub_idx = rng42.choice(len(human), size=len(unique_members), replace=False)
a_bal, n_bal = auc_rows(uniq_rows, human_set=human[hsub_idx])
print(f"{'test1b unique-subset, balanced human':<40} AUC {a_bal:.4f}   "
      f"rows {n_bal:5d}   (human subsampled to {len(hsub_idx)}, rng42)")

# --- TEST 2: same COUNT profile, RANDOM members (no retrieval) --------------
# Reproduce the exact histogram of per-member counts, but assign those counts
# to randomly chosen library members instead of nearest-match ones.
rng_ctrl = np.random.default_rng(0)
profile = np.sort(count_vals)[::-1]                        # multiplicities to place
rand_members = rng_ctrl.choice(N_LIB, size=len(profile), replace=False)
rand_multiset = np.repeat(rand_members, profile)
report("test2 random-duplication (no retrieval)", X_pool[rand_multiset],
       f"(same {len(profile)}-member count profile, random members)")

# --- TEST 3: frequency-flattened caps --------------------------------------
def capped(served_arr, cap):
    out = []
    seen = Counter()
    for m in served_arr:
        if seen[m] < cap:
            out.append(m)
            seen[m] += 1
    return np.array(out)

cap1 = capped(served, 1)
cap2 = capped(served, 2)
report("test3 cap=1 (==test1, dedup)", X_pool[cap1], "")
report("test3 cap=2", X_pool[cap2], "")

# --- serve-once anchor ------------------------------------------------------
report("anchor library once each (all 2000)", X_pool, "(the 0.509 crowd)")

print()
print("=== summary table ===")
print(f"{'label':<40} {'AUC':>8} {'rows':>6} {'unique':>7}  note")
for label, a, n, uniq, note in results:
    print(f"{label:<40} {a:8.4f} {n:6d} {uniq:7d}  {note}")
print(f"{'test1b balanced-human variant':<40} {a_bal:8.4f} {n_bal:6d} {n_bal:7d}  (human->{len(hsub_idx)})")
