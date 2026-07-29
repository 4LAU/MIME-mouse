"""Is the rotation tax fundamentally a pixel-quantization-fingerprint effect?

The ±2deg = near-full-tax result from serving_tax_snap.py suggests the RF may
be reading an exact integer-pixel quantization fingerprint, not rotation per se.
This nails it down:

  STEP 0  verify the pool coords are actually integer-valued.
  STEP 1  fine fixed-angle rotation sweep, NO snap -> find the onset.
  STEP 2  sub-pixel jitter probe, NO rotation -> does going off-lattice alone
          reproduce the tax?
  STEP 3  jitter-then-snap -> does re-snapping recover it?
  STEP 4  round the untransformed set (identity control).

Same harness/loaders/seeds as serving_tax_snap.py. N=2000. Always resample to
125 Hz before extract. No new trajectory generation.
"""
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from features import extract_features, resample_trajectory

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


def reaim(traj, ratio):
    """Similarity transform of (x, y) by complex `ratio`, anchored at start."""
    xy = traj[:, :2]
    start = xy[0]
    off = (xy[:, 0] - start[0]) + 1j * (xy[:, 1] - start[1])
    moved = off * ratio
    out = traj.copy()
    out[:, 0] = start[0] + moved.real
    out[:, 1] = start[1] + moved.imag
    return out


def snap(traj):
    out = np.asarray(traj, dtype=np.float64).copy()
    out[:, 0] = np.round(out[:, 0])
    out[:, 1] = np.round(out[:, 1])
    return out


def feats(trajs):
    return [extract_features(resample_trajectory(t, hz=125.0)) for t in trajs]


results = []


def report(label, feature_list, note=""):
    a, n = auc(feature_list)
    results.append((label, a, n, note))
    print(f"{label:<42} AUC {a:.4f}   N {n:5d}   {note}")


# =========================================================================
# STEP 0: are the raw pool coords integer-valued?
# =========================================================================
print("=== STEP 0: integer-lattice check on raw selected trajectories ===")
all_dev = []
all_frac = []
for i in range(len(sel_trajs)):
    xy = sel_trajs[i][:, :2]
    dev = np.abs(xy - np.round(xy))
    all_dev.append(dev.max())
    all_frac.append(float(np.mean(dev < 1e-9)))
    if i < 5:
        print(f"  traj {i:4d}: n={len(xy):4d}  max|dev from int|={dev.max():.6g}  "
              f"frac exact int={np.mean(dev < 1e-9):.4f}")
all_dev = np.array(all_dev)
all_frac = np.array(all_frac)
print(f"  ACROSS ALL {len(sel_trajs)} selected trajs:")
print(f"    max abs deviation from nearest int (worst traj): {all_dev.max():.6g}")
print(f"    mean per-traj max deviation:                     {all_dev.mean():.6g}")
print(f"    mean fraction of points exactly integer:         {all_frac.mean():.4f}")
print(f"    fraction of trajs that are FULLY integer:        {np.mean(all_frac == 1.0):.4f}")
print()

# baseline anchors
print("=== anchors ===")
report("A  cached features", [f for f in X_pool], "(expect ~0.508)")
report("A' re-extracted", feats(sel_trajs), "(should match A)")
print()

# =========================================================================
# STEP 1: fine fixed-angle rotation sweep, NO snap
# =========================================================================
print("=== STEP 1: fixed-angle rotation, single deterministic theta, NO snap ===")
for theta in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
    ratio = np.exp(1j * np.deg2rad(theta))
    rotated = [reaim(t, ratio) for t in sel_trajs]
    report(f"rot {theta:>5.2f}deg no-snap", feats(rotated), "")
print()

# =========================================================================
# STEP 2: sub-pixel jitter probe, NO rotation
# =========================================================================
print("=== STEP 2: sub-pixel jitter, NO rotation (off-lattice only) ===")
RNG = np.random.default_rng(0)
for eps in (0.1, 0.25, 0.5, 1.0):
    jittered = []
    for t in sel_trajs:
        out = t.copy()
        out[:, 0] = out[:, 0] + RNG.uniform(-eps, eps, size=len(out))
        out[:, 1] = out[:, 1] + RNG.uniform(-eps, eps, size=len(out))
        jittered.append(out)
    report(f"jitter uniform +/-{eps}px", feats(jittered), "")

# Gaussian variant sd=0.3px
gauss = []
for t in sel_trajs:
    out = t.copy()
    out[:, 0] = out[:, 0] + RNG.normal(0.0, 0.3, size=len(out))
    out[:, 1] = out[:, 1] + RNG.normal(0.0, 0.3, size=len(out))
    gauss.append(out)
report("jitter gaussian sd=0.3px", feats(gauss), "")
print()

# =========================================================================
# STEP 3: jitter-then-snap (reuse eps=0.5 uniform set) -- recompute that set
# with a fresh RNG(0) so it is deterministic and independent of step-2 order.
# =========================================================================
print("=== STEP 3: eps=0.5px uniform jitter, THEN snap back to int ===")
RNG3 = np.random.default_rng(0)
jit05 = []
for t in sel_trajs:
    out = t.copy()
    out[:, 0] = out[:, 0] + RNG3.uniform(-0.5, 0.5, size=len(out))
    out[:, 1] = out[:, 1] + RNG3.uniform(-0.5, 0.5, size=len(out))
    jit05.append(out)
report("jitter +/-0.5px no-snap (ref)", feats(jit05), "(should match step-2 eps=0.5)")
report("jitter +/-0.5px THEN snap", feats([snap(t) for t in jit05]),
       "(expect ~0.509 if lattice is the whole story)")
print()

# =========================================================================
# STEP 4: round the untransformed set (identity control)
# =========================================================================
print("=== STEP 4: snap untransformed set (identity control) ===")
report("untransformed + snap", feats([snap(t) for t in sel_trajs]),
       "(expect ~0.509; ~identity if already integer)")
print()

print("=== summary table ===")
print(f"{'label':<42} {'AUC':>8} {'N':>6}  note")
for label, a, n, note in results:
    print(f"{label:<42} {a:8.4f} {n:6d}  {note}")
