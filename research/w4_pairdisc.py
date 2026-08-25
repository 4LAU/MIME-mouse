"""w4_pairdisc. AMENDMENT 31, registered in step0_prereg.md before this
file existed.

Fresh RF discriminator on (cond, e0, e1) tuples: human e1 vs a Pair1
draw behind the same (cond, e0). Not the protected scorer, no contract
features. Control: draw vs draw, must read 0.50.
"""
import json
import sys

import numpy as np
import torch

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                       # noqa: E402
from w4_pairq import (N_VAL, P1_PATH, VAL_ROWS_SEED, Pair1,        # noqa: E402
                      pair_tokens, splits)
import ledger                                                      # noqa: E402

SEED_D1, SEED_D2 = 30001, 30002
RF_SEED = 3100
SPLIT_SEED = 3105


def main():
    dev = esp._DEVICE
    lengths, trained, held = splits()
    val = np.sort(np.random.default_rng(VAL_ROWS_SEED).choice(held, N_VAL, replace=False))
    val = val[lengths[val] >= 2]
    s0, th0, d0, s1, th1, d1 = (x[val] for x in pair_tokens())
    cond = np.load("training/events_cond.npy")[val, :4].astype(np.float64)
    print(f"  val rows {len(val):,}", flush=True)

    pk = torch.load(P1_PATH, map_location=dev, weights_only=False)
    pair = Pair1(**pk["config"]).to(dev).eval()
    pair.load_state_dict(pk["model_state_dict"])

    C = torch.from_numpy(cond.astype(np.float32)).to(dev)
    S0, TH0, D0 = (torch.from_numpy(np.asarray(x, dtype=np.int64)).to(dev)
                   for x in (s0, th0, d0))

    def draw(seed):
        out = []
        with torch.no_grad():
            for c0 in range(0, len(val), 65536):
                torch.manual_seed(seed + c0)
                sl = slice(c0, c0 + 65536)
                out.append(torch.stack(
                    pair.sample(C[sl], S0[sl], TH0[sl], D0[sl], 1.0, 1.0, 1.0),
                    -1).cpu().numpy())
        return np.concatenate(out)          # (n, 3) s, th, dt

    e1_h = np.stack([s1, th1, d1], -1).astype(np.float64)
    e1_a = draw(SEED_D1).astype(np.float64)
    e1_b = draw(SEED_D2).astype(np.float64)

    def feats(e1, resid=True):
        cols = [cond, s0[:, None], th0[:, None], d0[:, None], e1]
        if resid:
            cols += [(e1[:, 0] - s0)[:, None], (e1[:, 2] - d0)[:, None]]
        return np.concatenate([np.asarray(c, dtype=np.float64) for c in cols], -1)

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

    names = ["cond0", "cond1", "cond2", "cond3", "s0", "th0", "dt0",
             "s1", "th1", "dt1", "res_s", "res_dt"]
    # AMENDMENT 31 REVISION: disjoint row halves per class, no row's
    # (cond, e0) appears in both classes (the OOB control rule).
    half = np.random.default_rng(SPLIT_SEED).permutation(len(val))
    h1, h2 = half[:len(val) // 2], half[len(val) // 2:]
    auc_main, rf_main = oob_auc(feats(e1_h)[h1], feats(e1_a)[h2], RF_SEED,
                                "READ 1 human(h1) vs d1(h2)")
    auc_ctrl, _ = oob_auc(feats(e1_a)[h1], feats(e1_b)[h2], RF_SEED + 1,
                          "control d1(h1) vs d2(h2)")
    auc_nores, _ = oob_auc(feats(e1_h, resid=False)[h1],
                           feats(e1_a, resid=False)[h2],
                           RF_SEED + 2, "READ 3 no residual features")

    imp = dict(zip(names, [float(v) for v in rf_main.feature_importances_]))
    print("  READ 2 importances: "
          + "  ".join(f"{k} {v:.3f}" for k, v in
                      sorted(imp.items(), key=lambda kv: -kv[1])))

    ctrl_ok = abs(auc_ctrl - 0.5) <= 0.005
    if auc_main >= 0.510 and ctrl_ok:
        v = "PAIR VISIBLE"
    elif auc_main <= 0.505:
        v = "PAIR INVISIBLE"
    else:
        v = "BETWEEN"
    if not ctrl_ok:
        v += " (CONTROL FAILED, main read invalid)"
    print(f"  VERDICT: {v}")

    res = dict(n_rows=int(len(val)), auc_main=float(auc_main),
               auc_control=float(auc_ctrl), auc_no_resid=float(auc_nores),
               importances=imp, verdict=v, draw_seeds=[SEED_D1, SEED_D2],
               rf_seed=RF_SEED, split_seed=SPLIT_SEED)
    with open("research/w4_pairdisc.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("  wrote research/w4_pairdisc.json")

    rid = ledger.append_row(
        "w4_pairdisc",
        {"n_rows": res["n_rows"], "draw_seeds": [SEED_D1, SEED_D2],
         "rf": "fresh sklearn RF 300 trees OOB, not the protected scorer",
         "split": "disjoint halves rng(3105), REVISION on record",
         "population": "Pair1 val rows rng(2025) 200k of held"},
        "ok",
        metrics={"auc_main": res["auc_main"], "auc_control": res["auc_control"],
                 "auc_no_resid": res["auc_no_resid"]},
        artifacts=["research/w4_pairdisc.json"],
        notes=f"AMENDMENT 31 pair only discriminator, human e1 vs q1g0 draw"
              f" behind the same (cond, e0). {v}. No contract run, registered"
              f" in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
