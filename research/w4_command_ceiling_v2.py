"""What would PERFECT obedience score? Asked again, inside ONE pipeline.

w4_command_ceiling asked this and got 0.9708 for a bank of real human feature
vectors, which is impossible on its face. w4_pipeline_agreement found why: that
version drew its commands from the checkpoint's bank, which lives in
detector_features space, and tried to bridge them into features.py space to
score them. The bridge is broken on time_to_peak_velocity and lossy on
curvature, so the number measured the mismatch and not a ceiling.

This version never crosses the bridge. The bank is built with features.py, the
draw happens in that bank's own z-space, and the score is taken there too. The
transform between draw and score is the identity.

The question it answers is real and it gates the whole obedience programme.
Serving does not command a real human feature vector. It commands a kernel
density draw over a bank of them:

    feat = bank[row] + EVENT_FEAT_BW * randn(18)      BW defaults to 0.25

Adding independent noise to 18 standardized dimensions inflates every marginal
variance by about 3 percent at BW 0.25 and, more importantly, dilutes every
cross-feature correlation, and w3_joint_structure established that the human
features are strongly coupled. If the command distribution itself scores well
above 0.50, then no amount of obedience gets past that, and the bandwidth is a
bug rather than a detail.

Arms:
  bank rows, no noise     the pure empirical bank. Must land near 0.50 or the
                          harness is wrong; this is the control the first
                          version failed.
  bank + BW               what serving actually commands.
  bank + BW, shuffled     independence control. Same marginals, correlations
                          destroyed. Puts the BW arms on a scale.

Caveat kept in view: serving applies the bandwidth in detector z-space, this
applies it in features.py z-space. For the sixteen columns that bridge cleanly
those are the same perturbation to within a few percent. For
time_to_peak_velocity and curvature they are not, so read those two arms as the
best case.

No model, no GPU, no generation.

Usage:
  env PYTHONPATH=.:research NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD \
    AVX512_SKX AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w4_command_ceiling_v2.py --n 2500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))

OUT = R / "research" / "w4_command_ceiling_v2_results.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--bw", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    import scoring  # noqa: E402
    from features import FEATURE_NAMES, extract_features  # noqa: E402
    from phase0_critic import reconstruct_human_val_paths  # noqa: E402

    paths, _ = reconstruct_human_val_paths(args.n, args.seed, verbose=False)
    bank = []
    for p in paths:
        f = extract_features(p)
        if f is not None and np.all(np.isfinite(f)):
            bank.append(f)
    bank = np.asarray(bank, dtype=np.float64)
    print(f"[ceiling2] bank {bank.shape} built with features.py", flush=True)

    mu, sd = bank.mean(0), bank.std(0)
    sd[sd < 1e-12] = 1.0
    z = (bank - mu) / sd
    rng = np.random.default_rng(args.seed)

    def score(zz):
        X = zz * sd + mu
        X = X[np.all(np.isfinite(X), axis=1)]
        return float(scoring.score_features(X)["auc_rf_oob"])

    out = {"n": int(len(bank)), "seed": args.seed, "arms": {}}
    print(f"\n{'arm':<36}{'contract AUC':>14}")
    for bw in args.bw:
        zz = z + bw * rng.standard_normal(z.shape) if bw > 0 else z
        a = score(zz)
        nm = "bank rows, no noise" if bw == 0 else f"bank + BW {bw}"
        out["arms"][nm] = a
        tag = ""
        if bw == 0:
            tag = "   <- control, must be near 0.50"
        elif abs(bw - 0.25) < 1e-9:
            tag = "   <- serving default"
        print(f"{nm:<36}{a:>14.4f}{tag}", flush=True)

    zz = z + 0.25 * rng.standard_normal(z.shape)
    sh = np.column_stack([rng.permutation(zz[:, i]) for i in range(zz.shape[1])])
    a = score(sh)
    out["arms"]["bank + BW 0.25, columns shuffled"] = a
    print(f"{'bank + BW 0.25, columns shuffled':<36}{a:>14.4f}"
          f"   <- correlations destroyed")

    # how much of the coupling the bandwidth actually costs
    zz = z + 0.25 * rng.standard_normal(z.shape)
    c0 = np.corrcoef(z, rowvar=False)
    c1 = np.corrcoef(zz, rowvar=False)
    iu = np.triu_indices(18, 1)
    out["mean_abs_corr_bank"] = float(np.abs(c0[iu]).mean())
    out["mean_abs_corr_bw025"] = float(np.abs(c1[iu]).mean())
    print(f"\nmean |cross-feature correlation|: bank "
          f"{out['mean_abs_corr_bank']:.3f} -> BW 0.25 "
          f"{out['mean_abs_corr_bw025']:.3f}")

    print(f"\n{'feature':<24}{'bank sd':>9}{'+BW sd':>9}{'ratio':>8}")
    for i, f in enumerate(FEATURE_NAMES):
        a_, b_ = z[:, i].std(), zz[:, i].std()
        print(f"{f:<24}{a_:>9.3f}{b_:>9.3f}{b_ / max(a_, 1e-9):>8.3f}")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[ceiling2] wrote {args.out} ({out['wall_sec']:.0f}s)")


if __name__ == "__main__":
    main()
