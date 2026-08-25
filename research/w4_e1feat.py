"""w4_e1feat. AMENDMENT 34, registered in step0_prereg.md before this
file existed.

Where in the contract's 18 feature space does the drawn e1's damage
live. Paired contrast of h0p1 (human e0, q1g0 drawn e1, AR continues)
against h02 (human e0, human e1, AR continues) on identical rows,
from the row aligned feature dumps written by w4_qladder.py
--dump-features. Measurement only; nothing here may ever become a
training signal (the w4_audit rule extends to this instrument).
Never reads the protected human eval file, never touches scoring.
"""
import glob
import json
import os
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from features import FEATURE_NAMES, normalized_wasserstein_by_feature  # noqa: E402
import ledger                                                          # noqa: E402

SEEDS = [40, 41, 42, 43, 44, 45]
RF_SEED = 3202
SPLIT_SEED = 3107


def main():
    hp, h2, valid = {}, {}, {}
    for s in SEEDS:
        hp[s] = np.load(f"research/w4_e1feat_F_h0p1_s{s}.npy")
        h2[s] = np.load(f"research/w4_e1feat_F_h02_s{s}.npy")
        valid[s] = np.isfinite(hp[s]).all(1) & np.isfinite(h2[s]).all(1)
        print(f"  seed {s}: valid in both {valid[s].sum()}/{len(valid[s])}",
              flush=True)

    print("\n  INTEGRITY GATE, regenerated contract AUC vs stored, 4 dp:")
    gate_ok = True
    for s in SEEDS:
        ef = json.load(open(f"research/w4_qladder_ef_s{s}.json"))["arms"]
        stored_h02 = json.load(open(f"research/w4_qladder_kf_s{s}.json"))["arms"]["h02"]["contract"]
        stored_hp = json.load(open(f"research/w4_qladder_ps_s{s}.json"))["arms"]["h0p1"]["contract"]
        ok2 = round(ef["h02"]["contract"], 4) == round(stored_h02, 4)
        okp = round(ef["h0p1"]["contract"], 4) == round(stored_hp, 4)
        print(f"  seed {s}: h02 {ef['h02']['contract']:.4f} vs {stored_h02:.4f}"
              f" {'ok' if ok2 else 'MISMATCH'}   h0p1 {ef['h0p1']['contract']:.4f}"
              f" vs {stored_hp:.4f} {'ok' if okp else 'MISMATCH'}", flush=True)
        gate_ok = gate_ok and ok2 and okp
    if not gate_ok:
        print("\n  INTEGRITY GATE FAILED, no read may be taken (registered)")
        sys.exit(2)

    res = {"seeds": SEEDS, "valid": {s: int(valid[s].sum()) for s in SEEDS}}

    pool_h2 = np.concatenate([h2[s][valid[s]] for s in SEEDS])
    pool_hp = np.concatenate([hp[s][valid[s]] for s in SEEDS])
    sd = pool_h2.std(0, ddof=1)

    print("\n  READ 1 (PRIMARY), paired standardized shift per feature,"
          " h0p1 minus h02, t over 6 seeds. CARRIES |t| >= 4.0,"
          " SUGGESTIVE 2.5 to 4.0:")
    read1 = {}
    for j, nm in enumerate(FEATURE_NAMES):
        d = np.array([(hp[s][valid[s], j] - h2[s][valid[s], j]).mean() / sd[j]
                      for s in SEEDS])
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        v = ("CARRIES" if abs(t) >= 4.0 else
             "SUGGESTIVE" if abs(t) >= 2.5 else "null")
        print(f"  {nm:>22}: mean {m:+.4f}  se {se:.4f}  t {t:+6.2f}  {v}",
              flush=True)
        read1[nm] = dict(mean=float(m), se=float(se), t=float(t), verdict=v)
    res["read1"] = read1

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    def oob_auc(Xa, Xb, seed, tag):
        X = np.concatenate([Xa, Xb])
        y = np.concatenate([np.ones(len(Xa)), np.zeros(len(Xb))])
        perm = np.random.default_rng(seed).permutation(len(X))
        X, y = X[perm], y[perm]
        rf = RandomForestClassifier(n_estimators=300, n_jobs=28, oob_score=True,
                                    random_state=seed)
        rf.fit(X, y)
        auc = roc_auc_score(y, rf.oob_decision_function_[:, 1])
        print(f"  {tag}: OOB AUC {auc:.4f}", flush=True)
        return auc, rf

    # Disjoint row halves per seed (standing rule, no shared covariate
    # rows across classes), pooled across seeds.
    Xa, Xb, Ca, Cb = [], [], [], []
    for s in SEEDS:
        idx = np.where(valid[s])[0]
        half = np.random.default_rng(SPLIT_SEED + s).permutation(len(idx))
        i1, i2 = idx[half[:len(idx) // 2]], idx[half[len(idx) // 2:]]
        Xa.append(hp[s][i1]); Xb.append(h2[s][i2])
        Ca.append(h2[s][i1]); Cb.append(h2[s][i2])
    Xa, Xb = np.concatenate(Xa), np.concatenate(Xb)
    Ca, Cb = np.concatenate(Ca), np.concatenate(Cb)

    print("\n  READ 2, trajectory level visibility, disjoint halves:")
    auc_main, rf_main = oob_auc(Xa, Xb, RF_SEED, "h0p1(h1) vs h02(h2)")
    auc_ctrl, _ = oob_auc(Ca, Cb, RF_SEED + 1, "control h02(h1) vs h02(h2)")
    imp = dict(zip(FEATURE_NAMES,
                   [float(v) for v in rf_main.feature_importances_]))
    print("  importances: "
          + "  ".join(f"{k} {v:.3f}" for k, v in
                      sorted(imp.items(), key=lambda kv: -kv[1])[:8]))
    ctrl_ok = 0.49 <= auc_ctrl <= 0.51
    if not ctrl_ok:
        v2 = "VOID (control failed)"
    elif auc_main >= 0.520:
        v2 = "VISIBLE"
    elif auc_main >= 0.505:
        v2 = "FAINT"
    else:
        v2 = "INVISIBLE"
    print(f"  READ 2 verdict: {v2}")
    res["read2"] = dict(auc=float(auc_main), control=float(auc_ctrl),
                        verdict=v2, importances=imp)

    print("\n  READ 3 (informational), normalized Wasserstein per feature:")
    w = normalized_wasserstein_by_feature(pool_hp, pool_h2)
    for nm, wv in sorted(zip(FEATURE_NAMES, w), key=lambda kv: -kv[1]):
        print(f"  {nm:>22}: {wv:.4f}")
    res["read3"] = dict(zip(FEATURE_NAMES, [float(v) for v in w]))

    with open("research/w4_e1feat.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1feat.json")
    print("  measurement only, never a training signal, no headline")

    carries = [k for k, r in read1.items() if r["verdict"] == "CARRIES"]
    rid = ledger.append_row(
        "w4_e1feat",
        {"seeds": SEEDS, "n": 2000, "arms": ["h0p1", "h02"],
         "dumps": sorted(glob.glob("research/w4_e1feat_F_*.npy")),
         "rf_seed": RF_SEED, "split_seed": SPLIT_SEED,
         "paired_against": ["w4_pairsplit_2026-08-25T012510+0000_489c9097",
                            "w4_kfill_2026-08-23T013629+0000_dc718032"]},
        "ok",
        metrics={"read2_auc": float(auc_main),
                 "read2_control": float(auc_ctrl),
                 "n_carries": len(carries)},
        artifacts=["research/w4_e1feat.json"],
        notes=f"AMENDMENT 34 feature space localization of the drawn e1"
              f" damage, h0p1 vs h02 paired rows. READ 2 {v2}"
              f" {auc_main:.4f}, CARRIES: {', '.join(carries) or 'none'}."
              f" Diagnostic only, never a training signal, registered in"
              f" advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
