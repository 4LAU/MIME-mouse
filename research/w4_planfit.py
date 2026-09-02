"""Is the plan model the weak link of the observed plan arm? Diagnostic,
CPU, never a training signal, never selection.

The plan decoder given the REAL plan is 0.13 nats per token better than
the served checkpoint; given ONE drawn plan it is worse, and the rollouts
read worse on the screen. The drawn plan comes from `PlanModel`, a small
per dimension MLP over 32 equal mass bins. This tests that model the way
the contract scorer tests paths: a random forest, out of bag, on
[cond, plan] rows, real plans against plans drawn for the DISJOINT half of
the held out conditions, so it can only separate through p(plan | cond).
A disjoint real against real split is the floor. The two sides must not
share condition values: the condition is in the feature vector and is near
unique per row, so a shared value puts one row of each class on the same
leaf, the out of bag vote goes to the twin, and the AUC reads below 0.5.
The first build made that mistake and its per dimension numbers, which have
one live plan column and are therefore the most exposed, were the worst hit. Per dimension marginal total
variation and a per dimension RF say which plan coordinate is off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "autoloop"))
from models.plan import PLAN_NAMES, PlanModel, PlanQuant, PlanSampler  # noqa: E402
import ledger  # noqa: E402

TRAIN_PICK_SEED = 123
N_TRAIN = 1_500_000


def oob_auc(X0, X1, seed):
    X = np.concatenate([X0, X1]); y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, oob_score=True,
                                n_jobs=12, random_state=seed)
    rf.fit(X, y)
    return float(roc_auc_score(y, rf.oob_decision_function_[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-ckpt", default="plan_model.pt")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/aaronadmin/w4_arms/planfit.json")
    a = ap.parse_args()
    t0 = time.time()
    feats = np.load("training/plan_feats.npy")
    lengths = np.load("training/events_len.npy")
    cond = np.load("training/events_cond.npy")
    N = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    elig = held[lengths[held] >= 5]
    rng = np.random.default_rng(1000 + a.seed)
    pick = rng.choice(elig, 2 * a.n, replace=False)
    A, Bc = pick[:a.n], pick[a.n:]                      # disjoint real pools
    pk = torch.load(f"training/{a.plan_ckpt}", map_location="cpu", weights_only=False)
    pm = PlanModel(**pk["config"]).eval()
    pm.load_state_dict(pk["model_state_dict"])
    quant = PlanQuant(**pk["quant"])
    sampler = PlanSampler(pm, quant)
    g = torch.Generator().manual_seed(a.seed * 100003 + 53)
    with torch.no_grad():
        zB = sampler(torch.as_tensor(cond[Bc]), generator=g).numpy()
    rA = quant.standardise(feats[A]).astype(np.float32)
    rB = quant.standardise(feats[Bc]).astype(np.float32)
    cA, cB = cond[A].astype(np.float32), cond[Bc].astype(np.float32)
    out = dict(args=vars(a), n=a.n)
    out["auc_real_vs_drawn"] = oob_auc(np.c_[cA, rA], np.c_[cB, zB], a.seed)
    out["auc_real_vs_real"] = oob_auc(np.c_[cA, rA], np.c_[cB, rB], a.seed)
    out["auc_plan_only_real_vs_drawn"] = oob_auc(rA, zB, a.seed)
    per = {}
    for j, name in enumerate(PLAN_NAMES):
        hr, e = np.histogram(rA[:, j], bins=32)
        hd, _ = np.histogram(zB[:, j], bins=e)
        tv = 0.5 * np.abs(hr / len(rA) - hd / len(zB)).sum()
        per[name] = dict(tv=round(float(tv), 4),
                         mean_real=round(float(rA[:, j].mean()), 3), mean_drawn=round(float(zB[:, j].mean()), 3),
                         sd_real=round(float(rA[:, j].std()), 3), sd_drawn=round(float(zB[:, j].std()), 3),
                         auc_dim=round(oob_auc(np.c_[cA, rA[:, j]], np.c_[cB, zB[:, j]], a.seed), 4))
    out["per_dim"] = per
    # coupling: correlation matrices real vs drawn (plan dims), max abs gap
    cr, cd = np.corrcoef(rA.T), np.corrcoef(zB.T)
    out["corr_max_gap"] = round(float(np.abs(cr - cd).max()), 4)
    out["corr_real"] = np.round(cr, 3).tolist(); out["corr_drawn"] = np.round(cd, 3).tolist()
    print(f"\n  PLAN MODEL FIT, {a.plan_ckpt}, {a.n} held out rows per side")
    print(f"    RF OOB AUC [cond, plan]  real vs drawn {out['auc_real_vs_drawn']:.4f}"
          f"   real vs real floor {out['auc_real_vs_real']:.4f}   plan only {out['auc_plan_only_real_vs_drawn']:.4f}")
    for name, v in per.items():
        print(f"    {name:16s} tv {v['tv']:.4f}  mean {v['mean_real']:+.3f}/{v['mean_drawn']:+.3f}"
              f"  sd {v['sd_real']:.3f}/{v['sd_drawn']:.3f}  auc dim {v['auc_dim']:.4f}")
    print(f"    corr max gap {out['corr_max_gap']}")
    print(f"    real corr  {out['corr_real']}\n    drawn corr {out['corr_drawn']}")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}, {time.time() - t0:.0f}s")
    rid = ledger.append_row(
        "w4_planfit",
        {"plan_ckpt": a.plan_ckpt, "n": a.n, "seed": a.seed},
        "ok",
        metrics={"real_vs_drawn": out["auc_real_vs_drawn"],
                 "floor": out["auc_real_vs_real"],
                 "plan_only": out["auc_plan_only_real_vs_drawn"]},
        artifacts=[a.out],
        notes="DIAGNOSTIC, never a training signal and never selection. Plans"
              " are drawn at the DISJOINT half of the conditions so no"
              " condition value appears on both sides of a forest.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid['run_id']}")


if __name__ == "__main__":
    main()
