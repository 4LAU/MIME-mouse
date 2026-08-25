"""Is the 0.067 carried by a MINORITY of trajectories or by ALL of them?

Registered in /home/aaronadmin/w4_arms/rowmap_prereg.md, before this file
existed. Read that first: the sanity gate, the null arm and the thresholds were
all fixed in advance, and the primary is defined as a DIFFERENCE against the
null rather than as the model's own curve.

WHY. Every arm in this record slices the gap by FEATURE or by MECHANISM.
Nothing has sliced it along the ROW axis. An aggregate over rows is blind to a
five percent tail by construction, so a tail could have survived thirty arms
without being seen.

THE PEEL. Sort each class by how confidently the forest classifies it, drop the
top q percent from BOTH classes, refit, rescore. Peeling removes exactly the
rows the forest could classify, so AUC falls with q even under the null. The
model curve alone is therefore meaningless and is never read on its own.

Safety. CPU only, read only. Replicates score_features' recipe here rather than
importing and mutating it; scoring.py is never edited. Never reads
data/human_eval_features.npy. Writes no checkpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
DATA = Path.home() / "mts_data"

N_USE = 2000
QS = (0, 2, 5, 10, 25)
NULL_SEEDS = (1001, 1002, 1003)
NOISE_SD = 0.0073            # the record's pooled within arm contract noise
RECORDED_OPTIMUM = 0.5792    # what q=0 on the model arm must reproduce, G1
RECORDED_FLOOR = 0.512       # what q=0 on the null arm must reproduce, G2

# score_features' recipe, replicated rather than imported-and-mutated.
RF = dict(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)


def rf_oob(X, y):
    clf = RandomForestClassifier(**RF).fit(X, y)
    p = clf.oob_decision_function_[:, 1]
    return float(roc_auc_score(y, p)), p


def peel(human, synth):
    """AUC after dropping the q percent most confidently classified rows.

    Dropped from BOTH classes so the balance score_features enforces is kept:
    for synthetic rows that is the highest p(synth), for human rows the lowest.
    """
    n = min(len(human), len(synth))
    H, S = np.asarray(human[:n], float), np.asarray(synth[:n], float)
    y = np.r_[np.zeros(n), np.ones(n)]
    auc0, p = rf_oob(np.vstack([H, S]), y)
    ph, ps = p[:n], p[n:]
    ord_h, ord_s = np.argsort(ph), np.argsort(-ps)

    curve = {}
    for q in QS:
        k = int(round(q / 100 * n))
        if k == 0:
            curve[q] = auc0
            continue
        Hq, Sq = H[ord_h[k:]], S[ord_s[k:]]
        m = min(len(Hq), len(Sq))
        curve[q], _ = rf_oob(np.vstack([Hq[:m], Sq[:m]]),
                             np.r_[np.zeros(m), np.ones(m)])
    # the plain language readout: share of generated rows that look LESS
    # synthetic than a typical real one
    overlap = float((ps < np.median(ph)).mean())
    return curve, overlap, ps, ph


def corpus_draw(rng, n):
    ok = np.flatnonzero(np.load(DATA / "events_feat18_ok.npy"))
    mm = np.load(DATA / "events_feat18.npy", mmap_mode="r")
    idx = np.sort(ok[rng.choice(len(ok), n, replace=False)])
    return np.asarray(mm[idx], dtype=np.float64)[rng.permutation(n)]


def main():
    ref = np.load(REPO / "data" / "human_val_features_grpo.npy")
    print(f"reference rows {len(ref)}")

    out = {"qs": list(QS), "model": {}, "null": {}}

    # ---- the NULL arm, three independent realisations -------------------
    print("\n  NULL, real human population against the same reference")
    print(f"    {'seed':<8}" + "".join(f"{q:>9}%" for q in QS) + f"{'overlap':>10}")
    null_curves = []
    for sd in NULL_SEEDS:
        c, ov, _, _ = peel(ref, corpus_draw(np.random.default_rng(sd), N_USE))
        null_curves.append([c[q] for q in QS])
        out["null"][str(sd)] = {"curve": c, "overlap": ov}
        print(f"    {sd:<8}" + "".join(f"{c[q]:>10.4f}" for q in QS)
              + f"{ov:>10.3f}")
    null_mean = np.mean(null_curves, 0)
    null_sd = np.std(null_curves, 0, ddof=1)
    out["null_mean"] = null_mean.tolist()
    out["null_sd"] = null_sd.tolist()
    print(f"    {'mean':<8}" + "".join(f"{v:>10.4f}" for v in null_mean))
    print(f"    {'sd':<8}" + "".join(f"{v:>10.4f}" for v in null_sd))

    # ---- the MODEL arm, both rollout seeds ------------------------------
    print("\n  MODEL at the recorded optimum, s 0.95 th 0.90 dt 1.00")
    print(f"    {'seed':<8}" + "".join(f"{q:>9}%" for q in QS) + f"{'overlap':>10}")
    mod_curves, overlaps = [], []
    for sd in (0, 1):
        f = REPO / "research" / f"w4_rowmap_feats_s{sd}.npz"
        if not f.exists():
            print(f"    seed {sd} MISSING, {f}")
            continue
        F = np.load(f)["F"]
        c, ov, ps, ph = peel(ref, F)
        mod_curves.append([c[q] for q in QS])
        overlaps.append(ov)
        out["model"][str(sd)] = {"curve": c, "overlap": ov, "n": int(len(F))}
        print(f"    {sd:<8}" + "".join(f"{c[q]:>10.4f}" for q in QS)
              + f"{ov:>10.3f}")
    if not mod_curves:
        print("\n  NO MODEL FEATURES. Generation has not finished.")
        return
    mod_mean = np.mean(mod_curves, 0)
    out["model_mean"] = mod_mean.tolist()
    print(f"    {'mean':<8}" + "".join(f"{v:>10.4f}" for v in mod_mean))

    # ---- THE SANITY GATE, read BEFORE the primary -----------------------
    g1 = abs(mod_mean[0] - RECORDED_OPTIMUM) <= 2 * NOISE_SD
    g2 = abs(null_mean[0] - RECORDED_FLOOR) <= 2 * NOISE_SD
    print(f"\n  SANITY GATE, read before anything else")
    print(f"    G1  model q=0 {mod_mean[0]:.4f} vs recorded {RECORDED_OPTIMUM} "
          f"-> {'PASS' if g1 else 'FAIL'}")
    print(f"    G2  null  q=0 {null_mean[0]:.4f} vs floor    {RECORDED_FLOOR} "
          f"-> {'PASS' if g2 else 'FAIL'}")
    out["gate"] = {"G1": bool(g1), "G2": bool(g2),
                   "model_q0": float(mod_mean[0]), "null_q0": float(null_mean[0])}
    if not (g1 and g2):
        print("\n  GATE FAILED. This is a broken instrument and NOTHING below "
              "is read, including anything that looks like a finding.")
        json.dump(out, open(REPO / "research" / "w4_rowmap_results.json", "w"),
                  indent=1)
        return

    # ---- THE PRIMARY, the difference between the curves ------------------
    diff = mod_mean - null_mean
    out["diff"] = diff.tolist()
    print(f"\n  PRIMARY. model minus null at the same q. The model curve alone "
          f"is NOT a result.")
    print(f"    {'q':<8}" + "".join(f"{q:>9}%" for q in QS))
    print(f"    {'diff':<8}" + "".join(f"{v:>10.4f}" for v in diff))

    d10 = diff[QS.index(10)]
    d25 = diff[QS.index(25)]
    if abs(d10) <= 2 * NOISE_SD:
        verdict = "TAIL"
    elif d25 > 0.03:
        verdict = "UNIFORM"
    else:
        verdict = "MIXED"
    out["verdict"] = verdict
    print(f"\n  q=10 difference {d10:+.4f} against the TAIL line {2*NOISE_SD:.4f}")
    print(f"  q=25 difference {d25:+.4f} against the UNIFORM line 0.0300")
    print(f"\n  VERDICT  {verdict}")

    ov = float(np.mean(overlaps))
    out["overlap_mean"] = ov
    print(f"\n  SECOND READOUT. {100*ov:.1f} percent of generated trajectories "
          f"look LESS synthetic\n  than a typical real one. Chance is 50 percent.")

    json.dump(out, open(REPO / "research" / "w4_rowmap_results.json", "w"),
              indent=1)
    print("\n  wrote research/w4_rowmap_results.json")


if __name__ == "__main__":
    main()
