"""The two environment fixes, measured paired instead of one shot each.

The W4 sections in HANDOFF.md priced the duration prior fix at minus 0.0174 and
the lattice snap fix at minus 0.0425, each from a single run against a single
reference. A replicate of the snap configuration then came back 0.0077 away from
itself, because `DurationModel.__init__` ends with an unseeded `default_rng` and
the sampling loop uses the global torch RNG. Both deltas were read as though the
flag were the only thing that changed between arms.

This runs all three arms at each of several seeds, with the duration draws and
the torch stream seeded identically inside a seed, so an arm to arm difference
at a fixed seed is attributable to the flag and the seed to seed spread is the
noise floor. The model is loaded once and the arms are switched on the module
globals, so nothing depends on process start order.

Arms:
    base   std_mult 0.7 gaussian, snap off   the configuration every W4 section
                                             above the 2026-08-07 entries used
    dur    std_mult 1.0 empirical, snap off
    snap   std_mult 1.0 empirical, snap 2.5  the served configuration

One trajectory per spec. Nothing generated twice, nothing selected.

Safety. Scores through research/autoloop/scoring.py only. Never modifies scoring
code, never training/candi_polar_flow_best.pt. Paces itself on GPU temperature:
this machine crashed on this workload on 2026-08-06, kill line tightened to 79C.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_paired.py --seeds 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from experiments._common import DurationModel  # noqa: E402
from features import extract_feature_matrix  # noqa: E402
from models.event_ar import EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS  # noqa: E402
from phase_a_baseline import make_specs  # noqa: E402
from w4_latent import cooldown, gpu_c, throttle  # noqa: E402

# name -> (std_mult, DUR_EMPIRICAL, EVENT_SNAP)
ARMS = {"base": (0.7, False, 0.0),
        "dur": (1.0, True, 0.0),
        "snap": (1.0, True, 2.5)}


def set_arm(std_mult, empirical, snap):
    """Switch the generative configuration on the module globals. _decode reads
    esp._SNAP at call time, and the duration prior is a plain attribute, so
    rebuilding it here is equivalent to having set the environment before the
    import."""
    os.environ["DUR_EMPIRICAL"] = "1" if empirical else "0"
    esp._duration = DurationModel("./training", std_mult=std_mult)
    esp._SNAP = snap


def specs_for(n, spec_seed, rngseed):
    """Durations are drawn here, so the duration RNG is seeded immediately
    before. Distances and angles come from make_specs and depend only on
    spec_seed, so every arm sees identical endpoints."""
    esp._duration._rng = np.random.default_rng(rngseed)
    rows, meta = [], []
    for sx, sy, ex, ey in make_specs(n, spec_seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ld = math.log(dist)
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([ld, math.log(esp._duration.sample(ld)),
                     math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ang))
    return rows, meta


def gen(model, rows, meta, batch, temp, dev, rngseed):
    F = extract_feature_matrix(gen_paths(model, rows, meta, batch, temp, dev,
                                         rngseed))
    return F[np.all(np.isfinite(F), 1)]


def gen_paths(model, rows, meta, batch, temp, dev, rngseed):
    """The decoded trajectories themselves. `gen` is this plus the feature
    extraction; anything that needs the paths rather than their summaries reads
    this so the two cannot drift apart."""
    torch.manual_seed(rngseed)
    paths = []
    for c0 in range(0, len(rows), batch):
        throttle()
        blk = rows[c0:c0 + batch]
        cond = torch.tensor(blk, dtype=torch.float32, device=dev)
        s_cls, th_cls, dt_cls = model.sample(cond, temperature=temp)
        pad = (s_cls >= S_PAD_CLASS).cpu().numpy()
        dt_ms = class_to_dt_ms(dt_cls)
        dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
                / esp._DT_STD).cpu().numpy()
        s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
        for j in range(s_np.shape[0]):
            sx, sy, ang = meta[c0 + j]
            p = esp._decode(dt_z[j], s_np[j], th_np[j], sx, sy, ang)
            L = int(pad[j].argmax()) if pad[j].any() else s_np.shape[1]
            if p is None or L < 2:
                continue
            paths.append(np.asarray(p, dtype=np.float64))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default="research/w4_paired.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    print(f"\n  {args.n} specs, {args.seeds} seeds, {len(ARMS)} arms, "
          f"one trajectory each, no selection")
    print(f"  specs identical across arms and seeds, "
          f"durations and sampling paired within a seed\n")
    print(f"  {'seed':>6}{'arm':>8}{'contract':>10}{'collapse':>10}"
          f"{'dur sd':>9}{'n':>7}{'gpu':>6}")

    res = {a: [] for a in ARMS}
    col = {a: [] for a in ARMS}
    with torch.no_grad():
        for sd in range(args.seeds):
            for name, (sm, emp, snap) in ARMS.items():
                cooldown()
                set_arm(sm, emp, snap)
                rows, meta = specs_for(args.n, args.seed, sd)
                F = gen(model, rows, meta, args.batch, args.temp, dev, sd)
                r = scoring.score_features(F)
                a = float(r["auc_rf_oob"])
                nfl = len(r["collapse_features"])
                res[name].append(a)
                col[name].append(nfl)
                print(f"  {sd:>6}{name:>8}{a:>10.4f}{nfl:>10}"
                      f"{np.log(F[:, 14]).std():>9.4f}{len(F):>7}"
                      f"{gpu_c():>6}", flush=True)

    print(f"\n  {'arm':>8}{'mean AUC':>11}{'sd':>9}{'collapse':>11}")
    for name in ARMS:
        v = np.array(res[name])
        print(f"  {name:>8}{v.mean():>11.4f}{v.std(ddof=1):>9.4f}"
              f"{np.mean(col[name]):>11.2f}")

    print(f"\n  PAIRED differences, within seed, so the flag is the only change")
    print(f"  {'comparison':<22}{'mean':>9}{'sd':>9}{'seeds won':>11}")
    out = {"auc": res, "collapse": col, "n": args.n, "seeds": args.seeds}
    for a, b in [("base", "dur"), ("dur", "snap"), ("base", "snap")]:
        d = np.array(res[b]) - np.array(res[a])
        out[f"{a}_to_{b}"] = {"mean": float(d.mean()),
                              "sd": float(d.std(ddof=1)),
                              "wins": int((d < 0).sum())}
        print(f"  {a + ' -> ' + b:<22}{d.mean():>+9.4f}{d.std(ddof=1):>9.4f}"
              f"{int((d < 0).sum())}/{len(d):>10}")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  a paired difference is trustworthy when its mean is several "
          f"times its own sd, not the arm sd\n  wrote {args.out}\n")


if __name__ == "__main__":
    main()
