"""Read CANDI's 0.752 through the degeneracy control.

CANDI is the one family whose numbers the control has not seen. It emits one
coordinate per 8 ms slot (research/phase_a_baseline.py:230, dt = 1.0 / HZ), so
the feature extractor's own resample is a no-op on its output and it can never
carry the exact-collinearity structure real recordings do. The event-stream
family emits sparse events at its own timings, gets resampled like a human
recording, and so does carry it. That difference is invisible in a single AUC.

Two arms, both at the published generation convention (EXPERIMENTS.md 2026-07-01:
steps 200, guide 0.15, perp 0.85, rotate correction, seed 42, n=2000), imported
from research/phase_a_baseline.py rather than reimplemented:

  rounded    whole-pixel output, the documented default
  unrounded  fractional output, which research/phase1_score.py records as the
             path that produced the 0.757 baseline reading

Each is read three ways by research/degeneracy_panel.py: the contract scorer as
it stands, the same arm against a human reference rebuilt from raw recordings,
and both sides nudged by 1e-9 px so only motion is left. A number that moves
between the second and third column was partly arithmetic.

Inference only, no backward pass, no writes to the checkpoint. The checkpoint
MD5 is verified after generation.

Usage:
  env PYTHONPATH=. \
    ~/venvs/mime/bin/python research/w3_candi_control.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

from degeneracy_panel import panel, print_panel  # noqa: E402

OUT = R / "research" / "w3_candi_control_results.json"
CKPT = "candi_polar_flow_best.pt"
CKPT_MD5 = "91326a29750789f3167055324ef377c5"
# EXPERIMENTS.md 2026-07-01, the three confirmation seeds behind the headline.
ORIGINAL = {"rounded": 0.752, "spread": 0.005}


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref", type=int, default=2000,
                    help="reference and holdout size, each")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # imported here, not at module scope: loading phase_a_baseline pulls in
    # torch and the checkpoint, which the --help path has no business doing
    from experiments._common import DurationModel
    from phase_a_baseline import (DUR_STD, TRAIN_DIR, generate_paths,
                                  load_model, make_specs)

    t0 = time.time()
    model, data_scale, device, max_seq_len_cfg = load_model(CKPT)
    model.max_seq_len_cfg = max_seq_len_cfg
    duration_model = DurationModel(TRAIN_DIR, std_mult=DUR_STD)
    specs = make_specs(args.n, args.seed)
    print(f"[candi] {CKPT} on {device}, {len(specs)} specs, published "
          f"convention (steps 200, guide 0.15, perp 0.85, rotate)", flush=True)

    arms = {}
    for name, no_round in (("rounded", False), ("unrounded", True)):
        t1 = time.time()
        paths = generate_paths(model, data_scale, device, duration_model,
                               specs, no_round=no_round)
        paths = [p for p in paths if p is not None and len(p) >= 2]
        arms[name] = paths
        print(f"[candi] {name}: {len(paths)}/{args.n} paths in "
              f"{time.time()-t1:.0f}s", flush=True)

    got = md5_file(Path(TRAIN_DIR) / CKPT)
    print(f"[candi] checkpoint md5 {got} "
          f"({'unchanged' if got == CKPT_MD5 else 'CHANGED, STOP'})",
          flush=True)
    if got != CKPT_MD5:
        raise SystemExit("protected checkpoint changed; refusing to report")

    out = {"ckpt": CKPT, "ckpt_md5_ok": True, "n": args.n, "seed": args.seed,
           "original": ORIGINAL, "arms": {}}
    # one panel per arm: the two arms can drop different paths as invalid, so
    # they are not row-aligned and must not be put in one shared-subset table
    for name, paths in arms.items():
        res = panel({f"candi {name}": paths}, n_paths=args.ref, seed=42)
        print_panel(res, f"CANDI, {name} output, {len(paths)} paths")
        out["arms"][name] = res

    print(f"\n{'arm':<14}{'published':>11}{'contract':>10}{'rebuilt':>10}"
          f"{'control':>10}")
    for name in arms:
        v = out["arms"][name][f"candi {name}"]
        pub = f"{ORIGINAL['rounded']:.4f}" if name == "rounded" else "-"
        print(f"{name:<14}{pub:>11}{v['contract']:>10.4f}"
              f"{v['rebuilt']:>10.4f}{v['control']:>10.4f}")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[candi] wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
