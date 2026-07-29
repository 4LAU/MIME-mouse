"""Does re-snapping to the integer pixel grid AFTER rotating recover the
rotation tax measured in serving_tax.py?

Hypothesis: rotation is expensive (~0.705 AUC) because it knocks trajectory
coordinates off the integer pixel lattice that real (and selected synthetic)
mouse data sits on. If we round rotated coordinates back to whole pixels
before feature extraction, AUC should drop back toward ~0.51.

N=2000 throughout (the reliable size). Same picks set, same RNG seeding
convention as serving_tax.py.
"""
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from features import extract_features, resample_trajectory

# The cached pool features were built via extract_feature_matrix, which
# resamples each trajectory to 125 Hz BEFORE extracting. Re-extraction must
# do the same or the numbers are meaningless (raw extraction reads ~0.95).

RNG = np.random.default_rng(0)

pool = np.load("pool_s42_k16.npz", allow_pickle=True)
picks = np.load("pool_s42_k16_picks_trust33_f20d85_r30_rf.npy")
human = np.load("data/human_eval_features.npy")          # (2000, 18)
human_dist = np.load("data/human_distances.npy")         # empirical distances

sel_trajs = [np.asarray(pool["trajs"][i], dtype=np.float64) for i in picks]
X_pool = pool["X"][picks]                                 # cached features of the set


def auc(synth_feats):
    """RF OOB AUC separating the human eval set from a synthetic feature set."""
    good = np.array([f is not None and np.all(np.isfinite(f)) for f in synth_feats])
    S = np.array([f for f, g in zip(synth_feats, good) if g])
    X = np.vstack([human, S])
    y = np.concatenate([np.zeros(len(human)), np.ones(len(S))])
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    return roc_auc_score(y, clf.oob_decision_function_[:, 1]), len(S)


def reaim(traj, ratio):
    """Similarity transform of a trajectory's (x, y) by complex `ratio`,
    anchored at its start. Timing column is left untouched (a replay keeps
    its recorded delays), so scaling distance scales speed."""
    xy = traj[:, :2]
    start = xy[0]
    off = (xy[:, 0] - start[0]) + 1j * (xy[:, 1] - start[1])
    moved = off * ratio
    out = traj.copy()
    out[:, 0] = start[0] + moved.real
    out[:, 1] = start[1] + moved.imag
    return out


def snap(traj):
    """Round the (x, y) columns to the nearest integer pixel. Timing column
    (and any further columns) left untouched. Accepts an ndarray or a list
    of (x, y, t) tuples (resample_trajectory's return type) and always
    returns an ndarray."""
    out = np.asarray(traj, dtype=np.float64).copy()
    out[:, 0] = np.round(out[:, 0])
    out[:, 1] = np.round(out[:, 1])
    return out


def stall_fraction(trajs):
    """Fraction of consecutive-point pairs that become identical (x AND y
    unchanged from previous sample) -- a side effect of rounding points that
    were <1px apart."""
    total = 0
    stalled = 0
    for t in trajs:
        xy = t[:, :2]
        if len(xy) < 2:
            continue
        d = np.diff(xy, axis=0)
        stalled += int(np.sum((d[:, 0] == 0) & (d[:, 1] == 0)))
        total += len(d)
    return stalled / total if total else float("nan")


def feats(trajs):
    return [extract_features(resample_trajectory(t, hz=125.0)) for t in trajs]


results = []


def report(label, feature_list, note=""):
    a, n = auc(feature_list)
    results.append((label, a, n, note))
    print(f"{label:<45} AUC {a:.4f}   N {n:5d}   {note}")


# --- sanity anchors: reproduce A and D from serving_tax.py -----------------
report("A  selected set, cached features", [f for f in X_pool],
       "(expect ~0.508)")
report("A' selected set, re-extracted (sanity)", feats(sel_trajs),
       "(should match A)")

RNG = np.random.default_rng(0)  # reset to match serving_tax.py's transform RNG state exactly
d_ratios = [np.exp(1j * RNG.uniform(0, 2 * np.pi)) for _ in sel_trajs]
d_raw = [reaim(t, r) for t, r in zip(sel_trajs, d_ratios)]
report("D  rotate only, random angle, no snap", feats(d_raw),
       "(expect ~0.705, rotation tax)")

# --- D_snap: same rotation, round raw rotated points to int pixel first ----
d_snapped = [snap(t) for t in d_raw]
report("D_snap rotate + snap-raw-before-resample", feats(d_snapped),
       f"(stall frac {stall_fraction(d_snapped):.4f} vs raw {stall_fraction(d_raw):.4f})")

# secondary variant: snap AFTER resample instead of before
d_snap_after = [snap(resample_trajectory(t, hz=125.0)) for t in d_raw]
d_snap_after_feats = [extract_features(t) for t in d_snap_after]
report("D_snap_post rotate, snap AFTER resample", d_snap_after_feats,
       f"(stall frac {stall_fraction(d_snap_after):.4f}; secondary variant)")

# --- small-angle bands: +/- band degrees, no-snap vs snap ------------------
for deg in (2, 5, 10, 20):
    rad = np.deg2rad(deg)
    ratios = [np.exp(1j * RNG.uniform(-rad, rad)) for _ in sel_trajs]
    raw = [reaim(t, r) for t, r in zip(sel_trajs, ratios)]
    report(f"band +/-{deg}deg no-snap", feats(raw), "")
    snapped = [snap(t) for t in raw]
    report(f"band +/-{deg}deg snap", feats(snapped),
           f"(stall frac {stall_fraction(snapped):.4f})")

# --- control: translate only, WITH snap -------------------------------------
c_raw = [reaim(t, 1.0 + 0j) for t in sel_trajs]
c_snapped = [snap(t) for t in c_raw]
report("C_snap translate only + snap (control)", feats(c_snapped),
       f"(stall frac {stall_fraction(c_snapped):.4f}; expect ~A, snap shouldn't hurt free case)")

print()
print("=== summary table ===")
print(f"{'label':<45} {'AUC':>8} {'N':>6}  note")
for label, a, n, note in results:
    print(f"{label:<45} {a:8.4f} {n:6d}  {note}")
