"""Phase 1 readout: did fooling the critic move the real detector number?

training/train_events_polar_geoadv.py prints a D gap while it trains. That
number says only that the generator and the critic are coupled. It cannot say
whether the paths got more human, and the whole history of this program is
warnings about exactly that substitution: the fixed-statistic stage drove its
own objective to the estimator floor and made the score WORSE, and greedy
per-request selection improved a picker's opinion while dropping the contract
score from 0.7204 to 0.7494. An adversarial objective is easier to satisfy
dishonestly than either.

So this ignores the D gap and scores the checkpoints the way everything else
in this program is scored: generate N fresh single trajectories per checkpoint,
apply the same additive arrival correction the 0.7283 arm uses, and hand the
features to research/autoloop/scoring.py. The base checkpoint is scored in the
identical harness so the comparison is not against a remembered number.

Two readings besides the headline, because a fine-tune can improve the score
for reasons that are not progress:

  coverage   scored against the real paths the base model already covers and
             against the quarter it does not, on the split from
             research/w3_critic_coverage.py. Phase 1 passing should move the
             uncovered number. Moving only the covered one means the model
             traded away what already worked.
  collapse   score_features' own dispersion battery, reported per checkpoint.
             An adversarial generator that wins by narrowing what it emits has
             found nothing, and this is what catches it.

One process per checkpoint: experiments/event_stream_polar reads its env at
import, so the checkpoint cannot be swapped in-process.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_geoadv_score.py --ckpt event_polar_4m_geoadv_v1_s200.pt
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

OUT_DIR = R / "research" / "w3_geoadv"
SPLIT = R / "research" / "w3_critic_coverage_scores.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint name in training/")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # must precede the experiment import, which reads env at module load
    from cond_realization_probe import SERVING_ENV, build_specs
    for k, v in SERVING_ENV.items():
        os.environ[k] = v
    os.environ["EVENT_CKPT"] = args.ckpt
    os.environ["EVENT_FEAT"] = "1"

    import scoring  # noqa: E402  (metric contract, imported never edited)
    from degeneracy_panel import (_score_against,  # noqa: E402
                                  features_with_jitter, real_paths)
    from w3_fallback_arrival import correct_additive  # noqa: E402

    from experiments import event_stream_polar as m  # noqa: E402

    t0 = time.time()
    specs = build_specs(args.n, args.seed)
    raw = m.generate_paths(specs)
    arm = [correct_additive(np.asarray(t), *(int(v) for v in s))
           if (t is not None and len(t) >= 3) else
           (np.asarray(t) if t is not None else None)
           for s, t in zip(specs, raw)]
    arm = [t for t in arm if t is not None]
    print(f"[geoadv] {args.ckpt}: {len(arm)} paths in {time.time()-t0:.0f}s",
          flush=True)

    X = features_with_jitter(arm, 0.0, args.seed)
    X = X[np.all(np.isfinite(X), axis=1)]
    res = scoring.score_features(X)
    auc = float(res["auc_rf_oob"])
    collapsed = sorted(res.get("collapse_features", []))

    out = {"ckpt": args.ckpt, "n": int(len(X)), "seed": args.seed,
           "auc_rf_oob": auc, "collapse_features": collapsed,
           "wall_sec": time.time() - t0}

    print(f"\n{args.ckpt}")
    print(f"  contract AUC                {auc:.4f}")
    print(f"  collapsed features          {', '.join(collapsed) or 'none'}")

    # coverage split, taken from the critic probe so every checkpoint is judged
    # against the same partition of real paths rather than one of its own
    if SPLIT.exists():
        sp = np.load(SPLIT)
        real = real_paths(args.n, args.seed, "ref")
        Xr = features_with_jitter(real, 0.0, args.seed)
        Xr = Xr[np.all(np.isfinite(Xr), axis=1)]
        cov, unc = sp["rf_covered"], sp["rf_uncovered"]
        cov = cov[cov < len(Xr)]
        unc = unc[unc < len(Xr)]
        c = {"covered": float(_score_against(X[:len(cov)], Xr[cov])["auc_rf_oob"]),
             "uncovered": float(_score_against(X[:len(unc)], Xr[unc])["auc_rf_oob"])}
        out["coverage"] = c
        print(f"  vs real paths base covers   {c['covered']:.4f}")
        print(f"  vs the quarter it does not  {c['uncovered']:.4f}")
    else:
        print(f"  (no coverage split at {SPLIT}, run w3_critic_coverage.py)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{Path(args.ckpt).stem}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[geoadv] wrote {p}")


if __name__ == "__main__":
    main()
