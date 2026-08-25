"""w4_e1proj. AMENDMENT 35, registered in step0_prereg.md before this
file existed.

Anchored paired projection: build a corpus human vs h02 discriminant
in the contract's 18 feature space, cross fitted by seed, and take
the row paired difference of P(human) between h0p1 and h02. CPU only,
no generation. Diagnostic only, never a training signal, no
selection. Reads training/events_*.npy and the A34 feature dumps;
never reads the protected eval file (score_features is called once,
on corpus human rows, as the registered plausibility gate).
"""
import json
import os
import sys

import numpy as np
import torch

# AMENDMENT 35 REVISION: the identical decoder environment qladder
# pins, so the human decode is bit compatible with the A34 dumps.
os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_SNAP", "2.5")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_BESTOF", "1")
os.environ.setdefault("EVENT_SIR", "1")
os.environ.setdefault("EVENT_ORDER", "gumbel")
os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                       # noqa: E402
import scoring                                                     # noqa: E402
from features import (FEATURE_NAMES, extract_features,             # noqa: E402
                      resample_trajectory)
from models.event_ar import DT_MAX_MS, class_to_dt_ms, dt_ms_to_class  # noqa: E402
from models.event_stream_polar import (S_PAD_CLASS, TH_NULL_CLASS,  # noqa: E402
                                       dth_lattice_to_class, s2_to_class)
import ledger                                                      # noqa: E402

SEEDS = [40, 41, 42, 43, 44, 45]
RF_SEED = 3203
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256
KMAX = 4
N = 2000


def main():
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    s2_all = np.load("training/events_s2.npy", mmap_mode="r")
    dth_all = np.load("training/events_dth.npy", mmap_mode="r")
    dt_all = np.load("training/events_dt.npy", mmap_mode="r")
    cond_all = np.load("training/events_cond.npy", mmap_mode="r")

    def human_features(seed):
        """The exact qladder row pick and decode path, real tokens only."""
        pick = np.sort(np.random.default_rng(1000 + seed).choice(elig, N, replace=False))
        s2 = np.asarray(s2_all[pick])
        dth = np.asarray(dth_all[pick])
        dt_ms = np.asarray(dt_all[pick]).astype(np.float64)
        conds = np.asarray(cond_all[pick])
        L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
        sc = s2_to_class(torch.from_numpy(np.asarray(s2, dtype=np.int64))).numpy()
        tc = np.where(np.asarray(s2) > 0,
                      dth_lattice_to_class(
                          torch.from_numpy(np.asarray(dth, dtype=np.int64))).numpy(),
                      TH_NULL_CLASS)
        dc = dt_ms_to_class(torch.from_numpy(dt_ms)).numpy()
        angs = np.arctan2(conds[:, 3].astype(np.float64),
                          conds[:, 2].astype(np.float64))
        F = np.full((N, len(FEATURE_NAMES)), np.nan)
        for i in range(N):
            n = int(L[i])
            s_row = np.full(MAX_T, S_PAD_CLASS, dtype=np.int64)
            th_row = np.full(MAX_T, TH_NULL_CLASS, dtype=np.int64)
            dt_row = np.zeros(MAX_T, dtype=np.int64)
            s_row[:n] = sc[i, :n]
            th_row[:n] = tc[i, :n]
            dt_row[:n] = dc[i, :n].clip(0, DT_MAX_MS)
            d = class_to_dt_ms(torch.from_numpy(dt_row)).numpy()
            dz = (np.log(np.maximum(d, 0.05)) - esp._DT_MEAN) / esp._DT_STD
            p = esp._decode(dz, s_row, th_row, 0.0, 0.0, float(angs[i]))
            if p is not None and len(p) >= 4:
                fv = extract_features(
                    resample_trajectory(np.asarray(p, dtype=np.float64)))
                if fv is not None and np.all(np.isfinite(fv)):
                    F[i] = fv
        return pick, F

    hum, picks = {}, {}
    for s in SEEDS:
        picks[s], hum[s] = human_features(s)
        print(f"  seed {s}: corpus human valid {int(np.isfinite(hum[s]).all(1).sum())}/{N}",
              flush=True)

    hp = {s: np.load(f"research/w4_e1feat_F_h0p1_s{s}.npy") for s in SEEDS}
    h2 = {s: np.load(f"research/w4_e1feat_F_h02_s{s}.npy") for s in SEEDS}

    print("\n  STEP 2, plausibility gate: contract AUC of corpus human rows:")
    pool = np.concatenate([hum[s] for s in SEEDS])
    pool = pool[np.isfinite(pool).all(1)]
    pool = pool[np.random.default_rng(RF_SEED).permutation(len(pool))]
    r = scoring.score_features(pool)
    gate_auc = float(r["auc_rf_oob"])
    print(f"  corpus human contract AUC {gate_auc:.4f} (bar 0.52)", flush=True)
    if gate_auc > 0.52:
        print("  GATE FAILED, decode path too far from the anchor, STOP (registered)")
        ledger.append_row(
            "w4_e1proj", {"seeds": SEEDS, "rf_seed": RF_SEED}, "failed",
            metrics={"gate_auc": gate_auc},
            notes="AMENDMENT 35 plausibility gate failed: corpus human rows "
                  "read above 0.52 against the anchor, direction untrusted, "
                  "no read taken (registered).", tier=1)
        ledger.regenerate_leaderboard()
        sys.exit(2)

    from sklearn.ensemble import RandomForestClassifier

    res = {"seeds": SEEDS, "gate_auc": gate_auc}
    deltas, seps, imps = {}, {}, []
    feat_deltas = {}
    for s in SEEDS:
        tr_h, tr_s = [], []
        banned = set(picks[s].tolist())
        for o in SEEDS:
            if o == s:
                continue
            keep = np.array([ix not in banned for ix in picks[o]])
            v = np.isfinite(hum[o]).all(1) & keep
            tr_h.append(hum[o][v])
            v2 = np.isfinite(h2[o]).all(1) & keep
            tr_s.append(h2[o][v2])
        X = np.concatenate(tr_h + tr_s)
        y = np.concatenate([np.ones(sum(len(a) for a in tr_h)),
                            np.zeros(sum(len(a) for a in tr_s))])
        perm = np.random.default_rng(RF_SEED + s).permutation(len(X))
        rf = RandomForestClassifier(n_estimators=300, n_jobs=28,
                                    random_state=RF_SEED)
        rf.fit(X[perm], y[perm])
        imps.append(rf.feature_importances_)

        valid = (np.isfinite(hp[s]).all(1) & np.isfinite(h2[s]).all(1))
        p_hp = rf.predict_proba(hp[s][valid])[:, 1]
        p_h2 = rf.predict_proba(h2[s][valid])[:, 1]
        deltas[s] = p_hp - p_h2
        vh = np.isfinite(hum[s]).all(1)
        seps[s] = (float(rf.predict_proba(hum[s][vh])[:, 1].mean()),
                   float(p_h2.mean()))
        feat_deltas[s] = hp[s][valid] - h2[s][valid]
        print(f"  fold {s}: n {valid.sum()}  mean proj_delta "
              f"{deltas[s].mean():+.5f}  anchor P(human): human "
              f"{seps[s][0]:.4f} h02 {seps[s][1]:.4f}", flush=True)

    d = np.array([deltas[s].mean() for s in SEEDS])
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t = m / se if se > 0 else float("inf")
    if t <= -3.0:
        v = "AWAY FROM HUMAN"
    elif t >= 3.0:
        v = "TOWARD HUMAN"
    elif abs(t) < 2.0:
        v = "NULL"
    else:
        v = "BETWEEN"
    print(f"\n  READ 1 (PRIMARY): mean {m:+.5f}  se {se:.5f}  t {t:+.2f}"
          f"  per seed " + " ".join(f"{x:+.5f}" for x in d))
    print(f"  VERDICT: {v}")
    res["read1"] = dict(mean=float(m), se=float(se), t=float(t), verdict=v,
                        per_seed={s: float(deltas[s].mean()) for s in SEEDS})

    mi = np.mean(imps, 0)
    print("\n  READ 2, direction definition (mean importances, top 8): "
          + "  ".join(f"{k} {vv:.3f}" for k, vv in
                      sorted(zip(FEATURE_NAMES, mi), key=lambda kv: -kv[1])[:8]))
    res["read2"] = dict(importances=dict(zip(FEATURE_NAMES, [float(x) for x in mi])),
                        anchor_sep={s: seps[s] for s in SEEDS})

    from scipy.stats import spearmanr
    print("\n  READ 3, which feature deltas move with the projection:")
    fd = np.concatenate([feat_deltas[s] for s in SEEDS])
    pd = np.concatenate([deltas[s] for s in SEEDS])
    read3 = {}
    for j, nm in enumerate(FEATURE_NAMES):
        rho = spearmanr(fd[:, j], pd).statistic
        read3[nm] = float(rho)
    for nm, rho in sorted(read3.items(), key=lambda kv: -abs(kv[1]))[:8]:
        print(f"  {nm:>22}: rho {rho:+.3f}")
    res["read3"] = read3

    with open("research/w4_e1proj.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1proj.json")
    print("  diagnostic only, never a training signal, no selection, no headline")

    rid = ledger.append_row(
        "w4_e1proj",
        {"seeds": SEEDS, "n": N, "rf_seed": RF_SEED,
         "anchor": "corpus human rows, same picks, token decoded",
         "paired_against": ["w4_e1feat_2026-08-25T083832+0000_1e8ddc20"]},
        "ok",
        metrics={"gate_auc": gate_auc, "read1_mean": float(m),
                 "read1_se": float(se), "read1_t": float(t)},
        artifacts=["research/w4_e1proj.json"],
        notes=f"AMENDMENT 35 anchored paired projection of the drawn e1"
              f" damage. Gate {gate_auc:.4f}. READ 1 {v} mean {m:+.5f} t"
              f" {t:+.2f}. Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
