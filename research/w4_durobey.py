"""Does the AR model realize the duration it is commanded, and what would
perfect obedience be worth?

Why this question and not another. `movement_duration` is the largest single
feature tell on this family, 0.5699 in `w4_whatsees`, and it is the only
quantity the model is explicitly told. With DUR_EMPIRICAL=1 the command is not
a parametric fit: `DurationModel.sample` draws an ACTUAL human log duration
from the matching log distance bin and adds 0.02 of jitter, so the commanded
conditional p(duration | distance) is human by construction. Any duration tell
in the output is therefore the model failing to obey a correct instruction, not
a bad instruction.

The masked family's equivalent number is on record: `cond_realization_probe`
measured a commanded-to-realized correlation of 0.172 on movement_duration
across an 18-dim character vector, and 0.41 averaged over all eighteen. That
programme then closed because both inference-time repairs of disobedience
failed, affine pre-distortion and guidance, for the same reason each time: a
model cannot be made to obey at sampling time a command it never learned to
obey at training time. This file does NOT propose either repair. It asks the
prior question that was never asked for the AR family, which is how large the
disobedience is and what closing it would buy, so that a training-time change
is priced before it is run rather than after.

Two arms, one sampling pass, nothing selected or reranked:

  as_served      the locked recipe, unchanged
  time_rescaled  every dt in the sequence multiplied by the single scalar that
                 makes the decoded duration equal the commanded one, then
                 rounded back to whole milliseconds so the lattice the family
                 buys by construction is not thrown away by the diagnostic

`time_rescaled` IS a decode-time repair and decode-time repairs are closed as a
road. It is here as an ORACLE UPPER BOUND, not as a proposal: it uses the
model's own realized duration to compute its own correction factor, which is
information no sampler has. Read it only as the price of the lever. If it moves
the contract score by less than the plus or minus 0.03 run to run noise, then
duration obedience is worth nothing and the next training run must aim
elsewhere.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_durobey.py \
        --ckpt event_ar_v2_s40000.pt --n 2000
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
# `w4_sampcost`: without expandable segments the no-KV-cache sampler fragments
# the caching allocator and spills to host memory, silently, at 1.2 traj/s.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import FEATURE_NAMES, extract_feature_matrix  # noqa: E402
from models.event_ar import DT_MAX_MS, EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS  # noqa: E402
from phase_a_baseline import make_specs  # noqa: E402

MD = FEATURE_NAMES.index("movement_duration")


def _single_feature_auc(F, H, col):
    """Same forest recipe as the contract scorer, one column instead of
    eighteen. NOT comparable to an eighteen feature number: `w4_dvjoint`
    established this instrument is not additive across dimensionality."""
    n = min(len(H), len(F))
    X = np.vstack([H[:n, [col]], F[:n, [col]]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                random_state=42)
    rf.fit(X, y)
    return float(roc_auc_score(y, rf.oob_decision_function_[:, 1]))


def _dt_to_z(dt_ms: np.ndarray) -> np.ndarray:
    return (np.log(np.clip(dt_ms, 0.05, None)) - esp._DT_MEAN) / esp._DT_STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="research/w4_durobey.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    npar = sum(p.numel() for p in model.parameters())
    print(f"  {args.ckpt} step {ck.get('step')} params {npar/1e6:.1f}M",
          flush=True)

    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    H = H[np.all(np.isfinite(H), 1)]

    rows, meta = [], []
    for sx, sy, ex, ey in make_specs(args.n, args.seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ld = math.log(dist)
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([ld, math.log(esp._duration.sample(ld)),
                     math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ang, ex, ey))

    served, rescaled = [], []
    cmd_s, real_s, factors = [], [], []
    for c0 in range(0, len(rows), args.batch):
        cond = torch.tensor(rows[c0:c0 + args.batch], dtype=torch.float32,
                            device=dev)
        s_cls, th_cls, dt_cls = model.sample(cond, temperature=args.temp)
        s_np = s_cls.cpu().numpy()
        th_np = th_cls.cpu().numpy()
        dt_ms = class_to_dt_ms(dt_cls).cpu().numpy()
        dt_z = _dt_to_z(dt_ms)

        for j in range(s_np.shape[0]):
            sx, sy, ang, ex, ey = meta[c0 + j]
            p = esp._decode(dt_z[j], s_np[j], th_np[j], sx, sy, ang)
            if p is None:
                continue
            a = np.asarray(p, dtype=np.float64)
            served.append(a)
            cmd = float(math.exp(rows[c0 + j][1]))
            real = float(a[-1, 2] - a[0, 2])
            if real <= 1e-6:
                continue
            cmd_s.append(cmd)
            real_s.append(real)

            # One scalar per sequence, applied to the whole dt stream, then
            # rounded back onto the whole millisecond lattice. Rounding is the
            # point: an unrounded rescale would remove the one tell this
            # family buys for free and the arm would price two things at once.
            f = cmd / real
            factors.append(f)
            pad = s_np[j] >= S_PAD_CLASS
            k = int(pad.argmax()) if pad.any() else s_np.shape[1]
            scaled = np.rint(dt_ms[j] * f).clip(0.0, float(DT_MAX_MS))
            scaled[k:] = dt_ms[j][k:]
            q = esp._decode(_dt_to_z(scaled), s_np[j], th_np[j], sx, sy, ang)
            if q is not None:
                rescaled.append(np.asarray(q, dtype=np.float64))

    cmd_a = np.asarray(cmd_s)
    real_a = np.asarray(real_s)
    r_log = float(np.corrcoef(np.log(cmd_a), np.log(real_a))[0, 1])
    fac = np.asarray(factors)

    print(f"\n  commanded vs realized duration, n {len(cmd_a)}")
    print(f"    correlation in log space        {r_log:.4f}")
    print(f"    median realized / commanded     {np.median(1.0/fac):.4f}")
    print(f"    p10 / p90 of that ratio         "
          f"{np.percentile(1.0/fac, 10):.4f} / "
          f"{np.percentile(1.0/fac, 90):.4f}")
    print(f"    median commanded / realized s   "
          f"{np.median(cmd_a):.4f} / {np.median(real_a):.4f}")

    out = dict(ckpt=args.ckpt, n=len(cmd_a), r_log_duration=r_log,
               ratio_p50=float(np.median(1.0 / fac)),
               ratio_p10=float(np.percentile(1.0 / fac, 10)),
               ratio_p90=float(np.percentile(1.0 / fac, 90)),
               cmd_p50=float(np.median(cmd_a)),
               real_p50=float(np.median(real_a)), arms={})

    print(f"\n  {'arm':<16}{'contract':>10}{'dur_only':>10}{'n':>7}")
    for name, paths in (("as_served", served), ("time_rescaled", rescaled)):
        F = extract_feature_matrix(paths)
        F = F[np.all(np.isfinite(F), 1)]
        auc = float(scoring.score_features(F)["auc_rf_oob"])
        dur = _single_feature_auc(F, H, MD)
        out["arms"][name] = dict(contract=auc, dur_only=dur, n=int(len(F)))
        print(f"  {name:<16}{auc:>10.4f}{dur:>10.4f}{len(F):>7}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  one trajectory per spec, no selection, no reranking")
    print("  time_rescaled is an ORACLE and is not a proposal: it uses the")
    print("  model's own realized duration to correct itself. Read it as the")
    print("  price of duration obedience and nothing else.")
    print("  run to run contract noise is plus or minus 0.03")
    print("  reference split-half floor 0.467 to 0.512")


if __name__ == "__main__":
    main()
