"""What would PERFECT obedience score? The ceiling of the whole conditioning
programme, measured without a model.

The trunk is handed an 18-dim character command and asked to realize it. The
argument for working on obedience is that if it realized the command exactly,
the output feature distribution would BE the command distribution, and since
commands are drawn from real human feature vectors the score would be 0.50 by
construction.

That argument has an unexamined step. The commands are not real human feature
vectors. They are a KERNEL DENSITY DRAW over a bank of them:

    feat = bank[row] + EVENT_FEAT_BW * randn(18)      BW defaults to 0.25

in z-scored detector space (experiments/event_stream_polar.py). Adding
independent noise of sd 0.25 to each of 18 standardized dimensions inflates
every marginal variance by about 6 percent and, more importantly, dilutes every
cross-feature correlation, and w3_joint_structure already established that the
human features are strongly coupled. w4_texture_sensitivity separately measured
how brutally this contract punishes small independent perturbations: real human
paths plus half a pixel of noise read 0.866.

So the ceiling has to be measured before more GPU time is spent chasing it. If
the command distribution itself scores well above 0.50, then obedience work is
capped there no matter how well it succeeds, and the bandwidth is a bug rather
than a detail.

Method: draw commands exactly as serving does, invert the checkpoint's z-score
and the detector-space transform to recover raw feature vectors in features.py
units, and score them against the contract's human reference. Three arms:

  bank rows, no noise     the pure empirical bank. Its own floor.
  bank + BW               what serving actually commands.
  bank + BW, shuffled     an independence control: same marginals, correlations
                          destroyed. Puts the BW arm on a scale.

No model, no GPU, no generation. This is arithmetic on the checkpoint's bank.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w4_command_ceiling.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

OUT = R / "research" / "w4_command_ceiling_results.json"


def from_detector_space(d: np.ndarray) -> np.ndarray:
    """Inverse of cond_realization_probe.to_detector_space.

    That function is the bridge between features.py's raw 18 statistics and the
    scale train_events_polar_dm.detector_features works in, which is what the
    checkpoint's bank was z-scored from. Every column is either identity, a
    fixed divide, or log1p, so the inverse is exact apart from the skewness
    clamp, which only bites on outliers.
    """
    r = np.empty_like(d)

    def ex(x):
        return np.expm1(np.clip(x, 0.0, 50.0))

    r[:, 0] = ex(d[:, 0])
    r[:, 1] = ex(d[:, 1])
    r[:, 2] = ex(d[:, 2])
    r[:, 3] = d[:, 3]
    r[:, 4] = d[:, 4] * 1e4
    r[:, 5] = ex(d[:, 5])
    r[:, 6] = ex(d[:, 6])
    r[:, 7] = d[:, 7] * 1e6
    r[:, 8] = ex(d[:, 8])
    r[:, 9] = d[:, 9]
    r[:, 10] = ex(d[:, 10])
    r[:, 11] = ex(d[:, 11]) / 1e3
    r[:, 12] = ex(d[:, 12]) / 1e3
    r[:, 13] = ex(d[:, 13])
    r[:, 14] = ex(d[:, 14]) / 10.0
    r[:, 15] = d[:, 15]
    r[:, 16] = ex(d[:, 16])
    r[:, 17] = ex(d[:, 17])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_polar_4m_fc_v2.pt")
    ap.add_argument("--bw", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    import scoring  # noqa: E402
    from features import FEATURE_NAMES  # noqa: E402

    ck = torch.load(R / "training" / args.ckpt, map_location="cpu",
                    weights_only=False)
    bank = ck["feat_bank"].numpy().astype(np.float64)
    mu = ck["feat_mu"].numpy().astype(np.float64)
    sd = ck["feat_sd"].numpy().astype(np.float64)
    print(f"[ceiling] {args.ckpt}: bank {bank.shape}", flush=True)

    rng = np.random.default_rng(args.seed)

    def to_raw(z):
        return from_detector_space(z * sd + mu)

    def score(raw):
        X = raw[np.all(np.isfinite(raw), axis=1)]
        return float(scoring.score_features(X)["auc_rf_oob"])

    rows = rng.choice(len(bank), size=min(args.n, len(bank)), replace=False)
    base = bank[rows]

    out = {"ckpt": args.ckpt, "n": len(base), "seed": args.seed, "arms": {}}
    print(f"\n{'arm':<34}{'contract AUC':>14}")
    for bw in args.bw:
        z = base + bw * rng.standard_normal(base.shape) if bw > 0 else base
        a = score(to_raw(z))
        nm = "bank rows, no noise" if bw == 0 else f"bank + BW {bw}"
        out["arms"][nm] = a
        star = "   <- serving default" if abs(bw - 0.25) < 1e-9 else ""
        print(f"{nm:<34}{a:>14.4f}{star}", flush=True)

    # independence control at the serving bandwidth
    z = base + 0.25 * rng.standard_normal(base.shape)
    sh = np.column_stack([rng.permutation(z[:, i]) for i in range(z.shape[1])])
    out["arms"]["bank + BW 0.25, columns shuffled"] = score(to_raw(sh))
    print(f"{'bank + BW 0.25, columns shuffled':<34}"
          f"{out['arms']['bank + BW 0.25, columns shuffled']:>14.4f}")

    # per-feature spread inflation the bandwidth causes, for the record
    z = base + 0.25 * rng.standard_normal(base.shape)
    print(f"\n{'feature':<24}{'bank sd':>9}{'+BW sd':>9}{'ratio':>8}")
    for i, f in enumerate(FEATURE_NAMES):
        a, b = base[:, i].std(), z[:, i].std()
        print(f"{f:<24}{a:>9.3f}{b:>9.3f}{b / max(a, 1e-9):>8.3f}")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[ceiling] wrote {args.out} ({out['wall_sec']:.0f}s)")


if __name__ == "__main__":
    main()
