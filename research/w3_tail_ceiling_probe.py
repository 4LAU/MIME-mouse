"""W3 groundwork: tail-expression ceiling of the current generator.

The W2 verdict (units_jd_main, 2026-07-20) localized the detector's signal in
feature tails the guided sampler never reached: peak dynamics (max_velocity,
max_acceleration) and mean_accel/mean_jerk extremes. Before designing W3
(representation/scale), measure whether those tails are OUTSIDE the current
generator's support entirely (nothing in noise space reaches them - a
representation limit) or merely rare (a frequency/objective limit).

Method: generate N_POOL unguided paths from candi_polar_flow_best.pt across
the standard spec distribution (same make_specs convention as every probe),
extract the real 18 features, and compare per-feature upper-tail quantiles
against data/human_val_features_grpo.npy. Key statistic per feature: the
fraction of human probability mass beyond the synth pool's q999 and max
(support-coverage gap). Read-only on the checkpoint; standard watchdog applies.

Usage:
    python research/w3_tail_ceiling_probe.py --n 8000 --pid-file research/w3_tail.pid.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.phase_a_baseline import (  # noqa: E402
    load_model, generate_paths, make_specs, DUR_STD,
)
from experiments._common import DurationModel  # noqa: E402
from features import extract_features, FEATURE_NAMES  # noqa: E402

TRAIN_DIR = Path("training")
SRC_CKPT_NAME = "candi_polar_flow_best.pt"
EXPECTED_SRC_MD5 = "91326a29750789f3167055324ef377c5"
HUMAN_REF_PATH = Path("data/human_val_features_grpo.npy")
OUT_JSON = Path("research/w3_tail_ceiling_results.json")
OUT_FEAT = Path("research/w3_tail_ceiling_features.npy")

QUANTS = [0.5, 0.9, 0.99, 0.999]


def md5_file(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--pid-file", type=str, default=None)
    args = ap.parse_args()

    if args.pid_file:
        import os
        with open(args.pid_file, "w") as fh:
            fh.write(str(os.getpid()))

    md5_before = md5_file(TRAIN_DIR / SRC_CKPT_NAME)
    assert md5_before == EXPECTED_SRC_MD5, "source checkpoint MD5 mismatch -- STOP"

    model, data_scale, device, max_seq_len_cfg = load_model(SRC_CKPT_NAME)
    model.max_seq_len_cfg = max_seq_len_cfg
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    duration_model = DurationModel(TRAIN_DIR, std_mult=DUR_STD)

    specs = make_specs(args.n, args.seed)
    print(f"[w3_tail] generating {args.n} unguided paths (seed={args.seed})...", flush=True)
    t0 = time.perf_counter()
    trajs = generate_paths(model, data_scale, device, duration_model, specs, no_round=True)
    trajs = [t for t in trajs if t is not None and len(t) >= 5]
    gen_s = time.perf_counter() - t0
    print(f"[w3_tail] generated {len(trajs)}/{args.n} usable in {gen_s:.0f}s", flush=True)

    rows = []
    for t in trajs:
        f = extract_features(t)
        if f is not None and np.all(np.isfinite(f)):
            rows.append(f)
    synth = np.stack(rows)
    np.save(OUT_FEAT, synth)
    human = np.load(HUMAN_REF_PATH)
    human = human[np.all(np.isfinite(human), axis=1)]
    print(f"[w3_tail] features: synth {synth.shape}, human {human.shape}", flush=True)

    table = {}
    for j, name in enumerate(FEATURE_NAMES):
        s, h = synth[:, j], human[:, j]
        entry = {
            "synth_q": {str(q): float(np.quantile(s, q)) for q in QUANTS},
            "human_q": {str(q): float(np.quantile(h, q)) for q in QUANTS},
            "synth_max": float(s.max()), "human_max": float(h.max()),
            "synth_min": float(s.min()), "human_min": float(h.min()),
            # support-coverage gaps: human mass the synth pool NEVER reaches
            "human_mass_above_synth_q999": float((h > np.quantile(s, 0.999)).mean()),
            "human_mass_above_synth_max": float((h > s.max()).mean()),
            "human_mass_below_synth_min": float((h < s.min()).mean()),
        }
        table[name] = entry
        print(f"[w3_tail] {name:24s} human_mass_above_synth_max="
              f"{entry['human_mass_above_synth_max']:.4f} "
              f"above_synth_q999={entry['human_mass_above_synth_q999']:.4f}", flush=True)

    md5_after = md5_file(TRAIN_DIR / SRC_CKPT_NAME)
    out = {
        "status": "COMPLETE", "n_requested": args.n, "n_usable": len(rows),
        "seed": args.seed, "gen_seconds": gen_s,
        "md5_before": md5_before, "md5_after": md5_after,
        "md5_unchanged": md5_before == md5_after,
        "quantile_table": table,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[w3_tail] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
