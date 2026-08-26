"""w4_e1repro. AMENDMENT 39, registered in step0_prereg.md before this
file existed.

How much does a contract AUC move when nothing changes except the order
the rows are handed to the scorer? CPU only, no model, no new data. The
protected scorer is imported and used unmodified.
"""
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402

K = 20
SEEDS = [40, 41, 42, 43, 44, 45]
PERM_SEED = 3207
STORED_S40_H02 = 0.5411   # research/w4_qladder_ef_s40.json
A38_S40_H02 = 0.5279      # research/w4_e1atfloor.json


def spread(mat, tag):
    m = mat[np.isfinite(mat).all(1)]
    vals = []
    for k in range(K):
        perm = np.random.default_rng(PERM_SEED + k).permutation(len(m))
        vals.append(float(scoring.score_features(m[perm])["auc_rf_oob"]))
    v = np.array(vals)
    print(f"  {tag:>10}: mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
          f"min {v.min():.4f}  max {v.max():.4f}  range {v.max() - v.min():.4f}"
          f"  (n_rows {len(m)})", flush=True)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                min=float(v.min()), max=float(v.max()),
                n_rows=int(len(m)), values=[float(x) for x in v])


def main():
    print(f"  {K} permutations per matrix, score_features unmodified\n", flush=True)
    res = {}
    res["S1"] = spread(np.load("research/w4_e1feat_F_h02_s40.npy"), "S1 h02 s40")
    res["S2"] = spread(np.load("research/w4_e1floor_F_L0_RAW_s40.npy"), "S2 human s40")
    res["S3"] = spread(np.concatenate(
        [np.load(f"research/w4_e1feat_F_h02_s{s}.npy") for s in SEEDS]), "S3 pooled")

    sd1 = res["S1"]["sd"]
    print(f"\n  READ 1 (PRIMARY): S1 sd {sd1:.4f}  "
          f"{'MATERIAL' if sd1 >= 0.005 else 'not material'} (bar 0.005)")
    print(f"  READ 2: S2 sd {res['S2']['sd']:.4f}   S3 sd {res['S3']['sd']:.4f}  "
          f"{'S3 wider' if res['S3']['sd'] > sd1 else 'S3 not wider'}")
    lo, hi = res["S1"]["min"], res["S1"]["max"]
    ins = lambda x: lo <= x <= hi
    print(f"  READ 3: S1 range [{lo:.4f}, {hi:.4f}]  stored {STORED_S40_H02:.4f} "
          f"{'inside' if ins(STORED_S40_H02) else 'OUTSIDE'}  A38 {A38_S40_H02:.4f} "
          f"{'inside' if ins(A38_S40_H02) else 'OUTSIDE'}")
    res["read1_material"] = bool(sd1 >= 0.005)
    res["read3"] = dict(stored_inside=bool(ins(STORED_S40_H02)),
                        a38_inside=bool(ins(A38_S40_H02)))

    with open("research/w4_e1repro.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1repro.json")
    print("  diagnostic only, the scorer is unmodified, no headline")

    rid = ledger.append_row(
        "w4_e1repro", {"k": K, "perm_seed": PERM_SEED}, "ok",
        metrics={"S1_sd": sd1, "S2_sd": res["S2"]["sd"], "S3_sd": res["S3"]["sd"],
                 "S1_range": res["S1"]["max"] - res["S1"]["min"]},
        artifacts=["research/w4_e1repro.json"],
        notes=f"AMENDMENT 39 scorer reproducibility under row permutation alone."
              f" S1 sd {sd1:.4f} range {res['S1']['max'] - res['S1']['min']:.4f}."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
