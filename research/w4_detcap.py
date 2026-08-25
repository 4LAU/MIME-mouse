"""w4_detcap. How much of the divergence does the 18 feature detector see?

Registered in /home/aaronadmin/w4_arms/detcap_prereg.md before any code existed.

The contract AUC is a LOWER BOUND on the divergence. This arm measures how loose
that bound is, by running a ladder of detectors of increasing strength over the
SAME trajectories, and reading every rung against a human vs human null.

CPU only. Nothing is generated. No checkpoint is touched. `human_eval_features`
is never read.
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path

from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from models.event_stream_polar import (s2_to_class, dth_lattice_to_class,
                                       S_PAD_CLASS, TH_NULL_CLASS,
                                       N_S_CLASSES, N_TH_CLASSES)
from models.event_ar import DT_MAX_MS, N_DT_CLASSES
from w4_views import decode_features

REPO = Path("/mnt/c/Users/aaron/Code/mouse-trajectory-synthesis")
CORPUS = Path("/home/aaronadmin/mts_data")
TBUF = 256
NOISE_SD = 0.0073
TODAY_CONTRACT = 0.5895          # w4_rowmap model q=0, OOB, same streams config
NULL_SEEDS = (2001, 2002, 2003)
FOLDS = KFold(5, shuffle=True, random_state=42)

# coarsening. rates, not identities: the loaded die story is about how often
# each KIND of event fires, so the alphabets are binned before counting.
NS_B, NTH_B, NDT_B = 16, 16, 16

# Exact block layout of rate_features, used by the low dimensional rungs added
# in AMENDMENT 2. Asserted against rate_names() at run time so a change to the
# alphabets cannot silently desynchronise these slices.
_U = 49                      # u_s 16 + u_th 17 + u_dt 16
_J0 = _U + NS_B**2 + (NTH_B + 1)**2 + NDT_B**2      # start of the joints
_LEN = 1                     # trailing len_frac
IDX_UNI = np.r_[np.arange(_U), [-1]]
IDX_UNIJOINT = np.r_[np.arange(_U), np.arange(_J0, _J0 + NS_B * (NTH_B + 1)
                                              + NS_B * NDT_B), [-1]]


# --------------------------------------------------------------- sources ----
def model_tokens(path):
    z = np.load(path)
    s, th, dt = (z[k].astype(np.int64) for k in ("s", "th", "dt"))
    cond = z["cond"].astype(np.float32)
    pad = s >= S_PAD_CLASS
    L = np.where(pad.any(1), pad.argmax(1), s.shape[1])
    return s, th, dt, cond, L


def corpus_tokens(ids):
    """Corpus rows -> class streams, converted exactly as ARDataset converts."""
    s2m = np.load(CORPUS / "events_s2.npy", mmap_mode="r")
    dthm = np.load(CORPUS / "events_dth.npy", mmap_mode="r")
    dtm = np.load(CORPUS / "events_dt.npy", mmap_mode="r")
    Lall = np.load(CORPUS / "events_len.npy")
    Call = np.load(CORPUS / "events_cond.npy")

    ids = np.sort(ids)
    L = np.minimum(Lall[ids], TBUF)
    s2 = torch.from_numpy(np.asarray(s2m[ids, :TBUF], dtype=np.int64))
    dth = torch.from_numpy(np.asarray(dthm[ids, :TBUF], dtype=np.int64))
    dt_ms = torch.from_numpy(np.asarray(dtm[ids, :TBUF], dtype=np.float32))
    valid = torch.from_numpy(np.arange(TBUF)[None, :] < L[:, None])

    s_cls = torch.where(valid, s2_to_class(s2), torch.full_like(s2, S_PAD_CLASS))
    th_cls = torch.where(valid & (s2 > 0), dth_lattice_to_class(dth),
                         torch.full_like(dth, TH_NULL_CLASS))
    dt_cls = torch.where(valid, torch.round(dt_ms).long().clamp(0, DT_MAX_MS),
                         torch.zeros_like(s2))
    return (s_cls.numpy(), th_cls.numpy(), dt_cls.numpy(),
            np.asarray(Call[ids], dtype=np.float32), L)


def corpus_ids(rng, n):
    Lall = np.load(CORPUS / "events_len.npy")
    ok = np.flatnonzero(Lall >= 5)
    return rng.choice(ok, n, replace=False)


# -------------------------------------------------------------- features ----
def bin_streams(s, th, dt, L):
    """Coarsen to rate alphabets, per row, trimmed to the row's own length."""
    out = []
    for j in range(len(s)):
        n = int(L[j])
        if n < 3:
            out.append(None)
            continue
        sj, tj, dj = s[j, :n], th[j, :n], dt[j, :n]
        sb = np.minimum(sj, S_PAD_CLASS - 1) * NS_B // S_PAD_CLASS
        tb = np.where(tj >= TH_NULL_CLASS, NTH_B,
                      np.minimum(tj, TH_NULL_CLASS - 1) * NTH_B // TH_NULL_CLASS)
        db = np.minimum(dj, N_DT_CLASSES - 1) * NDT_B // N_DT_CLASSES
        out.append((sb, tb, db))
    return out


def rate_features(binned):
    """Normalised unigram, bigram and within step joint COUNTS. Rates."""
    rows = []
    for b in binned:
        if b is None:
            rows.append(None)
            continue
        sb, tb, db = b
        n = len(sb)
        u_s = np.bincount(sb, minlength=NS_B) / n
        u_t = np.bincount(tb, minlength=NTH_B + 1) / n
        u_d = np.bincount(db, minlength=NDT_B) / n
        m = max(n - 1, 1)
        b_s = np.bincount(sb[:-1] * NS_B + sb[1:], minlength=NS_B ** 2) / m
        b_t = np.bincount(tb[:-1] * (NTH_B + 1) + tb[1:],
                          minlength=(NTH_B + 1) ** 2) / m
        b_d = np.bincount(db[:-1] * NDT_B + db[1:], minlength=NDT_B ** 2) / m
        j_st = np.bincount(sb * (NTH_B + 1) + tb,
                           minlength=NS_B * (NTH_B + 1)) / n
        j_sd = np.bincount(sb * NDT_B + db, minlength=NS_B * NDT_B) / n
        rows.append(np.concatenate([u_s, u_t, u_d, b_s, b_t, b_d, j_st, j_sd,
                                    [n / TBUF]]))
    return rows


def rate_names():
    nm = ([f"u_s{i}" for i in range(NS_B)]
          + [f"u_th{i}" for i in range(NTH_B + 1)]
          + [f"u_dt{i}" for i in range(NDT_B)]
          + [f"bi_s{i}_{j}" for i in range(NS_B) for j in range(NS_B)]
          + [f"bi_th{i}_{j}" for i in range(NTH_B + 1)
             for j in range(NTH_B + 1)]
          + [f"bi_dt{i}_{j}" for i in range(NDT_B) for j in range(NDT_B)]
          + [f"j_s{i}_th{j}" for i in range(NS_B) for j in range(NTH_B + 1)]
          + [f"j_s{i}_dt{j}" for i in range(NS_B) for j in range(NDT_B)]
          + ["len_frac"])
    return nm


def build(s, th, dt, cond, L):
    """Both views of the SAME rows. A row survives only if both views do."""
    ang = np.arctan2(cond[:, 3], cond[:, 2]).astype(np.float64)
    binned = bin_streams(s, th, dt, L)
    rates = rate_features(binned)
    F18, FR = [], []
    for j in range(len(s)):
        if rates[j] is None:
            continue
        n = int(L[j])
        f = decode_features(s[j, :n], th[j, :n], dt[j, :n], float(ang[j]))
        if f is None:
            continue
        F18.append(f)
        FR.append(rates[j])
    return np.asarray(F18, float), np.asarray(FR, float)


# --------------------------------------------------------------- detectors --
def cv_auc(make, X, y):
    p = np.zeros(len(y))
    for tr, te in FOLDS.split(X):
        m = make().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p)), p


def rf_oob(X, y):
    c = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                               random_state=42).fit(X, y)
    return float(roc_auc_score(y, c.oob_decision_function_[:, 1]))


MK_LOGIT = lambda: make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
MK_RF = lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                       random_state=42)
MK_GBM = lambda: HistGradientBoostingClassifier(max_iter=200, random_state=42)


def ladder(A18, AR, B18, BR, seed=0):
    """A is class 0, B is class 1. Balanced. Every rung, one fold assignment.

    CORRECTION 2026-08-18. Both arms are SHUFFLED before the balancing
    truncation. `corpus_tokens` sorts the ids it is handed, so taking the first
    n rows of the longer arm took the n LOWEST ids, which is a subpopulation
    and not a subsample. The human arm here is 1.25x the model arm, so it was
    the only arm truncated, and the null arms are equal sized so they never
    carried the same bias and could not cancel it. Measured cost of the bug on
    the unigram rung, +0.0215, about three contract noise sds. The 18 feature
    and rate views are two views of the SAME rows, so they take the SAME
    permutation.
    """
    ra = np.random.default_rng(6100 + seed)
    pa, pb = ra.permutation(len(A18)), ra.permutation(len(B18))
    A18, AR, B18, BR = A18[pa], AR[pa], B18[pb], BR[pb]
    n = min(len(A18), len(B18))
    X18 = np.vstack([A18[:n], B18[:n]])
    XR = np.vstack([AR[:n], BR[:n]])
    y = np.r_[np.zeros(n), np.ones(n)]
    r = {}
    r["L1_logit_18"], _ = cv_auc(MK_LOGIT, X18, y)
    r["L2_rf_18_cv"], _ = cv_auc(MK_RF, X18, y)
    r["L2_rf_18_oob"] = rf_oob(X18, y)
    r["L3_gbm_18"], _ = cv_auc(MK_GBM, X18, y)
    r["L4_logit_rate"], _ = cv_auc(MK_LOGIT, XR, y)
    r["L5_gbm_rate"], _ = cv_auc(MK_GBM, XR, y)
    # AMENDMENT 2, part 1. Same information, fewer dimensions. If the 1379
    # vector is diluting a real signal these read HIGHER, not lower.
    r["L4b_logit_uni"], _ = cv_auc(MK_LOGIT, XR[:, IDX_UNI], y)
    r["L5b_gbm_uni"], _ = cv_auc(MK_GBM, XR[:, IDX_UNI], y)
    r["L5c_gbm_unijoint"], _ = cv_auc(MK_GBM, XR[:, IDX_UNIJOINT], y)
    r["n_per_side"] = int(n)
    return r, X18, XR, y


RUNGS = ("L1_logit_18", "L2_rf_18_cv", "L3_gbm_18", "L4_logit_rate",
         "L5_gbm_rate", "L4b_logit_uni", "L5b_gbm_uni", "L5c_gbm_unijoint")
RATE_RUNGS = ("L4_logit_rate", "L5_gbm_rate", "L4b_logit_uni", "L5b_gbm_uni",
              "L5c_gbm_unijoint")
NULL_ANCHOR = 0.500          # AMENDMENT 1. Not w4_floor's 0.512, see prereg.


def planted(rng, n_side, alphas=(0.4, 0.8, 1.4)):
    """AMENDMENT 2, part 2. THE POWER CONTROL.

    Two human halves. Half B is resampled with weight exp(alpha * z(stat)) on a
    real trajectory statistic, so a difference of KNOWN size is planted in a
    currency BOTH detector families can read. The tilt statistic is mean dt,
    which the 18 features already summarise, so this deliberately favours the
    contract rung and makes POWERED the harder verdict to reach.

    Asks the only question that makes a low rate reading interpretable: when a
    difference of comparable size IS present, does the rate detector see it?
    """
    ids = corpus_ids(rng, int(n_side * 5.2))
    half = len(ids) // 2
    tokA = corpus_tokens(ids[:half])
    tokB = corpus_tokens(ids[half:])
    A18, AR = build(*tokA)
    # same correction as in ladder(). A is 2.6x the picked B arm, so without
    # this the alpha selection loop below compared the LOWEST ids against a
    # random pick and chose the planted size against a biased reference.
    pa = np.random.default_rng(6001).permutation(len(A18))
    A18, AR = A18[pa], AR[pa]
    # tilt statistic on B, computed BEFORE the feature build so the resample
    # picks rows, not features
    sB, thB, dtB, condB, LB = tokB
    stat = np.array([dtB[j, :int(LB[j])].mean() if LB[j] > 0 else 0.0
                     for j in range(len(sB))])
    z = (stat - stat.mean()) / (stat.std() + 1e-9)
    best = None
    for a in alphas:
        w = np.exp(a * z); w /= w.sum()
        pick = rng.choice(len(sB), min(n_side, (w > 0).sum()), replace=False,
                          p=w)
        B18, BR = build(sB[pick], thB[pick], dtB[pick], condB[pick], LB[pick])
        m = min(len(A18), len(B18))
        X18 = np.vstack([A18[:m], B18[:m]])
        y = np.r_[np.zeros(m), np.ones(m)]
        auc18, _ = cv_auc(MK_RF, X18, y)
        print(f"    alpha {a:<5} contract rung {auc18:.4f}  n {m}")
        if best is None or abs(auc18 - 0.59) < abs(best[1] - 0.59):
            best = (a, auc18, B18, BR)
    a, auc18, B18, BR = best
    print(f"    using alpha {a}, contract rung {auc18:.4f}")
    r, _, _, _ = ladder(A18, AR, B18, BR, seed=99)
    r["alpha"] = a
    return r


DEFAULT_STREAMS = [f"research/w4_texcover_streams_s{sd}.npz" for sd in (0, 1)]


def main(argv=None):
    # Strictly ADDITIVE. Called with no arguments this is byte for byte the run
    # recorded in detcap_prereg.md, including the optimum assertion, so that
    # result stays reproducible. The flags exist so the same ladder can be
    # pointed at a different generator's streams, which is the second readout
    # nardiff_prereg.md registers.
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", default=None)
    ap.add_argument("--out", default="research/w4_detcap_results.json")
    ap.add_argument("--label", default="event_ar rollout at the optimum")
    ap.add_argument("--contract", type=float, default=TODAY_CONTRACT,
                    help="the arm's own contract AUC. G1 checks that this "
                         "ladder's 18 feature OOB rung reproduces it, which "
                         "is meaningless against another arm's number.")
    opt = ap.parse_args(argv)
    streams = opt.streams or DEFAULT_STREAMS
    is_default = opt.streams is None

    out = {"null": {}, "rungs": list(RUNGS), "label": opt.label,
           "streams": list(streams)}
    print("w4_detcap. detector capacity ladder, CPU only, nothing generated.")
    print(f"  arm: {opt.label}\n")

    # ---- model arm, both seeds pooled ----------------------------------
    M18, MR = [], []
    for sd, sp in enumerate(streams):
        p = REPO / sp
        z = np.load(p)
        if is_default:
            assert float(z["s_temp"]) == 0.95 and float(z["th_temp"]) == 0.90 \
                and float(z["dt_temp"]) == 1.00, "streams are not at the optimum"
        a, b = build(*model_tokens(p))
        print(f"  model seed {sd}  {len(a)} rows survived, "
              f"18f {a.shape[1]}  rate {b.shape[1]}")
        M18.append(a); MR.append(b)
    M18, MR = np.vstack(M18), np.vstack(MR)

    # ---- matched human reference --------------------------------------
    rng = np.random.default_rng(11)
    H18, HR = build(*corpus_tokens(corpus_ids(rng, int(len(M18) * 1.25))))
    print(f"  human reference {len(H18)} rows\n")

    res, X18, XR, y = ladder(H18, HR, M18, MR)
    out["model"] = res
    print("  MODEL arm")
    for k in RUNGS:
        print(f"    {k:<16}{res[k]:.4f}")
    print(f"    {'L2 OOB':<16}{res['L2_rf_18_oob']:.4f}   "
          f"n per side {res['n_per_side']}")

    # ---- NULL arm, human vs human, three seeds ------------------------
    print("\n  NULL, human against human, same rungs, same folds")
    nulls = {k: [] for k in RUNGS}
    n_side = res["n_per_side"]
    for sd in NULL_SEEDS:
        r = np.random.default_rng(sd)
        ids = corpus_ids(r, int(n_side * 2.6))
        A18, AR = build(*corpus_tokens(ids[:len(ids) // 2]))
        B18, BR = build(*corpus_tokens(ids[len(ids) // 2:]))
        nr, _, _, _ = ladder(A18, AR, B18, BR, seed=sd)
        out["null"][str(sd)] = nr
        for k in RUNGS:
            nulls[k].append(nr[k])
        print(f"    {sd:<8}" + "".join(f"{nr[k]:>10.4f}" for k in RUNGS))
    nmean = {k: float(np.mean(v)) for k, v in nulls.items()}
    out["null_mean"] = nmean
    print(f"    {'mean':<8}" + "".join(f"{nmean[k]:>10.4f}" for k in RUNGS))

    # ---- GATES, read first --------------------------------------------
    g1 = abs(res["L2_rf_18_oob"] - opt.contract) <= 2 * NOISE_SD
    bad = [k for k in RUNGS if abs(nmean[k] - NULL_ANCHOR) > 2 * NOISE_SD]
    g2 = not bad
    g3 = abs(res["L2_rf_18_cv"] - res["L2_rf_18_oob"]) <= 0.03
    print("\n  SANITY GATE, read before the primary")
    print(f"    G1  L2 OOB {res['L2_rf_18_oob']:.4f} vs today's "
          f"{opt.contract}  -> {'PASS' if g1 else 'FAIL'}")
    print(f"    G2  every null within {2*NOISE_SD:.4f} of {NULL_ANCHOR}  -> "
          f"{'PASS' if g2 else 'FAIL ' + ','.join(bad)}")
    print(f"    G3  L2 cv {res['L2_rf_18_cv']:.4f} vs oob "
          f"{res['L2_rf_18_oob']:.4f}  -> {'PASS' if g3 else 'FAIL'}")
    out["gate"] = {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3),
                   "null_out_of_band": bad}

    # ---- PRIMARY -------------------------------------------------------
    best = max(RATE_RUNGS, key=lambda k: res[k])
    lift = (res[best] - nmean[best]) - (res["L2_rf_18_cv"] - nmean["L2_rf_18_cv"])
    out["best_rate_rung"] = best
    out["lift"] = float(lift)
    verdict = ("MATERIAL" if lift > 0.05 else
               "NEGLIGIBLE" if lift <= 0.015 else "MIXED")
    out["verdict"] = verdict
    print(f"\n  PRIMARY. null corrected lift of {best} over the contract")
    print(f"    {best:<16}{res[best]:.4f}  null {nmean[best]:.4f}  "
          f"corrected {res[best]-nmean[best]:+.4f}")
    print(f"    {'L2_rf_18_cv':<16}{res['L2_rf_18_cv']:.4f}  null "
          f"{nmean['L2_rf_18_cv']:.4f}  corrected "
          f"{res['L2_rf_18_cv']-nmean['L2_rf_18_cv']:+.4f}")
    print(f"    lift {lift:+.4f}   MATERIAL >0.05, NEGLIGIBLE <=0.015")
    if not (g1 and g2 and g3):
        print("\n  GATE FAILED. The ladder is void and the verdict above is "
              "NOT read.")
        out["verdict"] = "VOID"
    else:
        print(f"\n  VERDICT  {verdict}")

    # ---- AMENDMENT 2 part 2. POWER CONTROL, read before NEGLIGIBLE stands --
    print("\n  POWER CONTROL. planted difference of known comparable size")
    pr = planted(np.random.default_rng(77), n_side)
    out["planted"] = pr
    print(f"    {'rung':<20}{'planted':>10}{'model arm':>12}")
    for k in RUNGS:
        print(f"    {k:<20}{pr[k]:>10.4f}{res[k]:>12.4f}")
    best_rate_planted = max(RATE_RUNGS, key=lambda k: pr[k])
    powered = pr[best_rate_planted] >= pr["L2_rf_18_cv"] - 2 * NOISE_SD
    out["power"] = {"best_rate_rung_planted": best_rate_planted,
                    "rate": pr[best_rate_planted],
                    "contract": pr["L2_rf_18_cv"],
                    "POWERED": bool(powered)}
    print(f"\n    best rate rung on the planted difference "
          f"{best_rate_planted} {pr[best_rate_planted]:.4f} vs contract "
          f"{pr['L2_rf_18_cv']:.4f}")
    print(f"    -> {'POWERED' if powered else 'UNDERPOWERED'}")
    if out["verdict"] == "NEGLIGIBLE" and not powered:
        out["verdict"] = "UNDERPOWERED"
        print("\n  The rate detector cannot see a difference of this size even "
              "when one is planted. NEGLIGIBLE is WITHDRAWN and this arm has "
              "measured its own instrument, not the model.")
    print(f"\n  FINAL VERDICT  {out['verdict']}")

    json.dump(out, open(REPO / opt.out, "w"), indent=1)
    print(f"\n  wrote {opt.out}")


if __name__ == "__main__":
    main()
