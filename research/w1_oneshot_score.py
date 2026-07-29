"""W1 gate: one-shot RF-OOB AUC for an event-stream checkpoint, scored
against data/human_val_features_grpo.npy.

Why this exists instead of evaluate.py. HANDOFF_W1.md defines the standing
metric as "RF-OOB AUC at N=2000 per class vs data/human_val_features_grpo.npy"
but its step 4 invokes evaluate.py, and evaluate.py loads
data/human_eval_features.npy (line 173) -- the final untouched eval sample.
research/autoloop/scoring.py, which is the metric contract, forbids that file
anywhere that feeds a search-space decision, and the W1 go/no-go is exactly
such a decision. Running evaluate.py for the gate would both measure against
the wrong reference set and spend the held-out sample. So this script keeps
every other convention identical and swaps only the scorer:

  - specs: research/phase_a_baseline.make_specs, which mirrors evaluate.py's
    spec loop exactly (center 960,540, distances from human_distances.npy,
    uniform angle) and touches no human features file.
  - generation: experiments/event_stream_polar.py at its one-shot defaults
    (EVENT_BESTOF=1, EVENT_SIR=1 -- no selection, K=1), with the locked
    decode recipe passed through the env.
  - features: features.extract_feature_matrix, same as evaluate.py.
  - scoring: research/autoloop/scoring.score_features, the tier-1 contract.

The module under experiments/ reads its knobs at import time, so the env is
set before the import below.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))
sys.path.insert(0, str(REPO_ROOT / "research" / "autoloop"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint name under training/")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/w1_oneshot_results.json")
    ap.add_argument("--choice-temp", type=float, default=10.0,
                    help="EVENT_CHOICE_TEMP. Default 10 matches the locked "
                         "recipe recorded in EXPERIMENTS.md; HANDOFF_W1.md's "
                         "step-4 command omits it, which decodes differently.")
    ap.add_argument("--save-feats", default=None,
                    help="write the extracted synthetic feature matrix to this "
                         ".npy path. Lets w4_manifold_projection re-run on live "
                         "output without paying for a second generation pass.")
    args = ap.parse_args()

    # Locked decode recipe, one-shot: no best-of, no SIR.
    os.environ["EVENT_CKPT"] = args.ckpt
    os.environ["EVENT_ORDER"] = "gumbel"
    os.environ["EVENT_SNAP"] = "2.5"
    os.environ["EVENT_DUR_STD"] = "1.0"
    os.environ["DUR_EMPIRICAL"] = "1"
    os.environ["EVENT_CHOICE_TEMP"] = str(args.choice_temp)
    os.environ["EVENT_BESTOF"] = "1"
    os.environ["EVENT_SIR"] = "1"
    os.environ.pop("EVENT_POOL_TOKENS", None)
    os.environ.pop("EVENT_POOL_SAVE", None)
    os.environ.pop("EVENT_POOL_LOAD", None)

    import torch  # noqa: E402
    import scoring  # noqa: E402
    from features import extract_feature_matrix  # noqa: E402
    from phase_a_baseline import make_specs  # noqa: E402
    from experiments import event_stream_polar  # noqa: E402

    specs = make_specs(args.n, args.seed)
    # --seed used to reach make_specs only. event_stream_polar seeds torch once
    # at import with a hard-coded 42, so every --seed drew the SAME model
    # sampling stream and only the start and end points varied. Runs were also
    # not reproducible across invocations, since CUDA generation is not
    # bit-deterministic: identical config measured 0.649649 then 0.661526 on
    # 2026-07-27. Seeding here makes --seed vary the sampling too, which is what
    # anything calling these "independent seeds" already assumed.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[w1] ckpt={args.ckpt} n={args.n} seed={args.seed}", flush=True)

    t0 = time.perf_counter()
    trajs = event_stream_polar.generate_paths(specs)
    trajs = [t for t in trajs if t is not None and len(t) >= 2]
    gen_s = time.perf_counter() - t0
    print(f"[w1] generated {len(trajs)}/{args.n} in {gen_s:.1f}s", flush=True)

    feats = extract_feature_matrix(trajs)
    valid_ratio = len(feats) / max(args.n, 1)
    print(f"[w1] valid feature vectors: {len(feats)}/{args.n} ({valid_ratio:.0%})",
          flush=True)
    if valid_ratio < 0.80:
        print("[w1] REFUSED: valid ratio below 0.80, AUC would be meaningless")
        return 2

    if args.save_feats:
        np.save(REPO_ROOT / args.save_feats, feats)
        print(f"[w1] saved feature matrix to {args.save_feats}", flush=True)

    res = scoring.score_features(feats)
    auc = float(res["auc_rf_oob"])
    print(f"[w1] auc_rf_oob = {auc:.6f}  (n_per_class={res['n_per_class']})",
          flush=True)
    if res.get("collapse_flag"):
        print(f"[w1] COLLAPSE FLAG: {res.get('collapse_features')}", flush=True)

    payload = {
        "ckpt": args.ckpt, "n": args.n, "seed": args.seed,
        "choice_temp": args.choice_temp,
        "auc_rf_oob": auc, "n_per_class": int(res["n_per_class"]),
        "valid_ratio": valid_ratio, "gen_seconds": gen_s,
        "collapse_flag": bool(res.get("collapse_flag", False)),
        "collapse_features": res.get("collapse_features", []),
        "dispersion_ratios": {k: float(v) for k, v in
                              (res.get("dispersion_ratios") or {}).items()},
        "human_ref": "data/human_val_features_grpo.npy",
    }
    out = REPO_ROOT / args.out
    prev = json.loads(out.read_text()) if out.exists() else []
    if isinstance(prev, dict):
        prev = [prev]
    prev.append(payload)
    out.write_text(json.dumps(prev, indent=2))
    print(f"[w1] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
