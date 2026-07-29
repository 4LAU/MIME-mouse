"""Stop guessing channels. Ask the detector what it is using.

Every hand picked statistic priced this session came back small. Still placement
fully destroyed is worth 0.052, still count is worth nothing, and `w4_redundancy`
and `w4_coupling` had already shown the eighteen marginals are not the gap. The
joint still separates at 0.61 to 0.64 against a 0.467 to 0.512 floor, so the
signal exists and is not in any single property anybody has proposed.

This asks the forest directly, on the model's own output:

  single      AUC from each feature alone. `w4_redundancy` implies these are all
              near the floor; this re-measures them on THIS model so the rest of
              the table has a baseline that belongs to it.
  greedy      forward selection. Add the feature that buys the most, repeat.
              This is the one that matters. If three features reach the full
              eighteen feature score, the gap is a small conspiracy and can be
              attacked. If it takes twelve, the signal is genuinely distributed
              and no single mechanism will close it.
  drop one    AUC with each feature removed. A feature that costs nothing to
              remove is not carrying the signal even if it looks important.
  impurity    the forest's own ranking, printed for contrast because it is
              known to mislead on correlated features and the first three
              columns are the honest version.

The scorer in scoring.py takes all eighteen by definition and must not be
touched, so every AUC here is built locally with the same recipe: 100 trees,
out of bag decision function, seed 42, roc_auc_score on the OOB scores rather
than oob_score_, which is accuracy.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_whatsees.py --ckpt event_ar_v1.pt \
        --temp 0.9 --n 2500
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import FEATURE_NAMES, extract_feature_matrix  # noqa: E402
from models.event_ar import EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS  # noqa: E402
from phase_a_baseline import make_specs  # noqa: E402


def auc_cols(H, F, cols):
    n = min(len(H), len(F))
    X = np.vstack([H[:n][:, cols], F[:n][:, cols]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                random_state=42)
    rf.fit(X, y)
    return float(roc_auc_score(y, rf.oob_decision_function_[:, 1])), rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v1.pt")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat-out", default="research/w4_ar_features.npy")
    ap.add_argument("--out", default="research/w4_whatsees.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    specs = make_specs(args.n, args.seed)
    rows, meta = [], []
    for sx, sy, ex, ey in specs:
        d = math.hypot(ex - sx, ey - sy)
        if d < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([math.log(d), math.log(esp._duration.sample(math.log(d))),
                     math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ang))

    paths = []
    for c0 in range(0, len(rows), args.batch):
        cond = torch.tensor(rows[c0:c0 + args.batch], dtype=torch.float32,
                            device=dev)
        s_cls, th_cls, dt_cls = model.sample(cond, temperature=args.temp)
        dt_ms = class_to_dt_ms(dt_cls)
        dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
                / esp._DT_STD).cpu().numpy()
        s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
        for j in range(s_np.shape[0]):
            sx, sy, ang = meta[c0 + j]
            p = esp._decode(dt_z[j], s_np[j], th_np[j], sx, sy, ang)
            if p is not None:
                paths.append(np.asarray(p, dtype=np.float64))
        print(f"  generated {len(paths):,}", flush=True)

    F = extract_feature_matrix(paths)
    F = F[np.all(np.isfinite(F), 1)]
    np.save(args.feat_out, F)
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    H = H[np.all(np.isfinite(H), 1)]
    print(f"\n  {len(F):,} generated, {len(H):,} human, temp {args.temp}")

    full, rf_full = auc_cols(H, F, list(range(len(FEATURE_NAMES))))
    contract = float(scoring.score_features(F)["auc_rf_oob"])
    print(f"  all 18 here {full:.4f}, contract scorer {contract:.4f}\n")

    singles = []
    for i, nm in enumerate(FEATURE_NAMES):
        a, _ = auc_cols(H, F, [i])
        singles.append((a, i, nm))
    singles.sort(reverse=True)
    print(f"  {'single feature':<26}{'auc':>8}")
    for a, i, nm in singles:
        print(f"  {nm:<26}{a:>8.4f}", flush=True)

    print(f"\n  {'greedy forward':<26}{'auc':>8}{'gain':>8}")
    chosen, prev, greedy = [], 0.5, []
    remaining = list(range(len(FEATURE_NAMES)))
    while remaining and len(chosen) < 10:
        best = None
        for i in remaining:
            a, _ = auc_cols(H, F, chosen + [i])
            if best is None or a > best[0]:
                best = (a, i)
        a, i = best
        chosen.append(i)
        remaining.remove(i)
        greedy.append(dict(feature=FEATURE_NAMES[i], auc=a, gain=a - prev))
        print(f"  +{FEATURE_NAMES[i]:<25}{a:>8.4f}{a - prev:>8.4f}", flush=True)
        prev = a

    print(f"\n  {'drop one':<26}{'auc':>8}{'cost':>8}")
    drops = []
    for i, nm in enumerate(FEATURE_NAMES):
        cols = [j for j in range(len(FEATURE_NAMES)) if j != i]
        a, _ = auc_cols(H, F, cols)
        drops.append((full - a, a, nm))
    drops.sort(reverse=True)
    for c, a, nm in drops[:8]:
        print(f"  -{nm:<25}{a:>8.4f}{c:>8.4f}")

    out = dict(ckpt=args.ckpt, temp=args.temp, n_gen=int(len(F)),
               auc_all18=full, auc_contract=contract,
               singles=[dict(feature=nm, auc=a) for a, i, nm in singles],
               greedy=greedy,
               drop_one=[dict(feature=nm, auc=a, cost=c) for c, a, nm in drops])
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  if a handful of features reach the full score the gap is a small")
    print("  conspiracy; if it takes most of them the signal is distributed and")
    print("  no single mechanism will close it.")


if __name__ == "__main__":
    main()
