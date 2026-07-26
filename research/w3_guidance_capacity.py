"""Can guidance reach curvature, or is that variety simply not in the model?

research/cond_realization_probe.py measured how weakly the model follows the
movement character it is handed: the commanded-to-realized coefficient averages
about 0.1, and correcting the input to compensate makes things worse (0.6512 to
0.8123). HANDOFF_W3.md's P2 verdict already recorded the headline effect of
classifier-free guidance on this same weakness, corrected one-shot 0.728 to
about 0.70, so the AUC question is answered and is not what this asks.

What was never asked is whether guidance reaches CURVATURE. Two measurements
from different directions landed on the same deficit: research/w3_missing_paths.py
found that half the human movement the model cannot cover is the smooth kind,
and the realization probe found curvature spread stuck at 0.59 to 0.67 times
human under every correction strength. If turning guidance up widens curvature
toward human, the variety exists in the model and a retrain that strengthens
adherence can reach it. If curvature stays flat while the velocity features
inflate, the variety is absent and no amount of adherence work will produce it.
That is the difference between a fine-tune and a new model, so it is worth four
minutes of GPU per setting.

Three readings per setting, none of them the headline AUC alone:

  spread      realized std over human std, per feature. Curvature is the one
              that matters; the velocity group is the control that shows
              guidance is doing something at all.
  coverage    the arm scored against the three quarters of real paths the
              base model covers, and against the quarter it does not, using
              w3_missing_paths' own split. Guidance that helps should move the
              uncovered number, since the covered one is already at chance.
  auc         the contract scorer, for continuity with the ledger.

One process per setting: experiments/event_stream_polar reads its env at import.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_guidance_capacity.py --cfg-w 2.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

OUT_DIR = R / "research" / "w3_guidance_capacity"
CURV = ("curvature_mean", "curvature_std", "angular_velocity_std",
        "path_efficiency")
VEL = ("mean_velocity", "max_velocity", "std_acceleration", "mean_jerk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-w", type=float, required=True,
                    help="EVENT_CFG_W; 0 disables guidance")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # must precede the experiment import, which reads env at module load
    from cond_realization_probe import SERVING_ENV, build_specs
    for k, v in SERVING_ENV.items():
        os.environ[k] = v
    os.environ["EVENT_FEAT"] = "1"
    os.environ["EVENT_CFG_W"] = str(args.cfg_w)

    import scoring  # noqa: E402  (metric contract, imported never edited)
    from degeneracy_panel import (_score_against,  # noqa: E402
                                  features_with_jitter, real_paths)
    from features import FEATURE_NAMES  # noqa: E402
    from sklearn.ensemble import RandomForestClassifier  # noqa: E402

    from experiments import event_stream_polar as m  # noqa: E402

    t0 = time.time()
    trajs = [t for t in m.generate_paths(build_specs(args.n, args.seed))
             if t is not None]
    print(f"[guidance] w={args.cfg_w}: {len(trajs)} paths in "
          f"{time.time()-t0:.0f}s", flush=True)

    X = features_with_jitter(trajs, 0.0, args.seed)
    X = X[np.all(np.isfinite(X), axis=1)]
    real = real_paths(args.n, 42, "ref")
    Xr = features_with_jitter(real, 0.0, 42)
    Xr = Xr[np.all(np.isfinite(Xr), axis=1)]
    n = min(len(X), len(Xr))

    auc = float(scoring.score_features(X)["auc_rf_oob"])

    # coverage split, defined by the BASE model so every setting is judged
    # against the same partition of real paths rather than its own
    base = np.load(OUT_DIR / "coverage_split.npz") if (
        OUT_DIR / "coverage_split.npz").exists() else None
    if base is None:
        clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                     oob_score=True, n_jobs=-1,
                                     random_state=scoring.RF_SEED)
        clf.fit(np.vstack([Xr[:n], X[:n]]),
                np.concatenate([np.zeros(n), np.ones(n)]))
        p_real = 1.0 - clf.oob_decision_function_[:n, 1]
        order = np.argsort(-p_real)
        k = max(n // 4, 10)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(OUT_DIR / "coverage_split.npz",
                 uncovered=order[:k], covered=order[-k:], defined_by=args.cfg_w)
        base = np.load(OUT_DIR / "coverage_split.npz")
        print(f"[guidance] defined the coverage split at w={args.cfg_w}")
    un, cov = base["uncovered"], base["covered"]

    cover = {
        "covered": float(_score_against(X[:len(cov)], Xr[cov])["auc_rf_oob"]),
        "uncovered": float(_score_against(X[:len(un)], Xr[un])["auc_rf_oob"]),
    }

    ratios = {nm: float(np.std(X[:, j]) / max(np.std(Xr[:, j]), 1e-12))
              for j, nm in enumerate(FEATURE_NAMES)}

    print(f"\nguidance w={args.cfg_w}   contract AUC {auc:.4f}")
    print(f"  vs real paths the base model covers    {cover['covered']:.4f}")
    print(f"  vs the quarter it does not             {cover['uncovered']:.4f}")
    print("  spread against human, curvature group")
    for nm in CURV:
        print(f"    {nm:<24}{ratios[nm]:.3f}")
    print("  spread against human, velocity group (the control)")
    for nm in VEL:
        print(f"    {nm:<24}{ratios[nm]:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"cfg_w": args.cfg_w, "n": len(X), "seed": args.seed, "auc": auc,
           "coverage": cover, "spread_ratios": ratios,
           "wall_sec": time.time() - t0}
    p = OUT_DIR / f"w{str(args.cfg_w).replace('.', 'p')}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[guidance] wrote {p}")


if __name__ == "__main__":
    main()
