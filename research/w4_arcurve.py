"""Does a falling likelihood loss actually drag the contract AUC down with it?

This is the open question `w4_arfit` deliberately left open. It showed the AR
model memorised nothing, held out loss within 0.57 percent of training loss, so
capacity is the lever. It did NOT show that spending that capacity improves the
score, because loss and AUC are different objectives and nothing in this
programme has ever measured the link between them.

`event_ar_v2` is instrumented for exactly this: eight numbered snapshots across
40,000 steps, at a capacity of 21.7M against v1's 7.95M, with the sample budget,
the batch size and the `default_rng(123)` training subset all held at v1's
values so capacity is the only variable.

This scores every snapshot the same way and puts loss and AUC side by side.
Three disciplines, all of which have burned this programme before:

  same n, seed, temp    AUC moves plus or minus 0.03 run to run and changes with
                        sample size, so every row uses identical settings. The
                        existing `w4_ar_eval.json` has 0.6271 at n 1500 where
                        `w4_whatsees` has 0.6668 at n 2499 on the SAME model,
                        which is what happens without this discipline.
  v1 measured here      the baseline is re-scored in this run rather than quoted
                        from an old row, so the comparison is like for like.
  never reimplement     scoring goes through `w4_ar_eval.py` unchanged, as a
                        subprocess. The contract path is the one thing that must
                        not be subtly re-derived, so it is reused, not copied.

The shape of the answer is the point, not the endpoint:

  AUC falls with loss           capacity is the lever, scale further, and the
                                slope prices how much a further 2x buys
  AUC flat while loss falls     the likelihood objective and the contract have
                                come apart. That is a more important result than
                                any score, and the honest response is to report
                                it, not to reach for another config.
  AUC rises while loss falls    the model is getting better at the corpus and
                                worse at the eval distribution, which would mean
                                the eval specs and the training data disagree

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_arcurve.py \
        --ckpts event_ar_v1.pt,event_ar_v2_s5000.pt,...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import torch

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

ENV = dict(os.environ)
ENV["NPY_DISABLE_CPU_FEATURES"] = ("AVX512F AVX512CD AVX512_SKX AVX512BW "
                                   "AVX512DQ AVX512VL")
ENV["PYTHONPATH"] = ".:research:research/autoloop"
# `w4_sampcost` priced this. `EventARModel.sample` keeps no KV cache and asks
# for a slightly larger workspace at each of the 256 steps, so the caching
# allocator fragments: at batch 500 it reserved 22,936 MiB on an 8,188 MiB card
# while only 2,312 MiB was live, spilled to host memory, and ran at 1.2 traj/s.
# Expandable segments takes that reservation to 2,468 MiB and 11.5 traj/s. The
# cliff is silent on this card, so this is not optional tuning.
ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def discover(pattern):
    """v1 first, then v2 snapshots in step order, then the final v2."""
    got = sorted(glob.glob(f"training/{pattern}"))
    def key(p):
        m = re.search(r"_s(\d+)\.pt$", p)
        return (0 if "v1" in p else 1, int(m.group(1)) if m else 10 ** 9)
    return [os.path.basename(p) for p in sorted(got, key=key)]


def loss_of(ckpt):
    """Held out and training loss recorded in the checkpoint, if any.

    v1 predates `--val-every` and carries only a training EMA, so its held out
    number comes from `w4_arfit` instead and is passed in rather than invented.
    """
    ck = torch.load(f"training/{ckpt}", map_location="cpu", weights_only=False)
    vh = ck.get("val_hist") or []
    step = ck.get("step")
    at = [r for r in vh if r["step"] == step] or (vh[-1:] if vh else [])
    return dict(step=step, train_ema=ck.get("loss_ema"),
                val_total=at[0]["total"] if at else None,
                val_s=at[0]["s"] if at else None,
                val_th=at[0]["th"] if at else None,
                val_dt=at[0]["dt"] if at else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="",
                    help="comma separated; empty means discover by --pattern")
    ap.add_argument("--pattern", default="event_ar_v[12]*.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", default="1.0")
    ap.add_argument("--batch", type=int, default=64,
                    help="sampling batch. 64 measured fastest under expandable "
                         "segments; the default 500 is 12x slower")
    ap.add_argument("--v1-val", type=float, default=4.3326,
                    help="v1 held out total from w4_arfit; v1 logged none")
    ap.add_argument("--out", default="research/w4_arcurve.json")
    args = ap.parse_args()

    ckpts = ([c for c in args.ckpts.split(",") if c] if args.ckpts
             else discover(args.pattern))
    ckpts = [c for c in ckpts if "_latest" not in c]
    if not ckpts:
        raise SystemExit("no checkpoints matched; pass --ckpts explicitly")
    print(f"  {len(ckpts)} checkpoints, n {args.n}, seed {args.seed}, "
          f"temp {args.temp}, identical for every row\n")
    print(f"  {'checkpoint':<26}{'step':>8}{'trainEma':>10}{'heldOut':>10}"
          f"{'contract':>10}{'missP50':>9}{'nEvP50':>8}")

    rows = []
    for ck in ckpts:
        tmp = f"research/.arcurve_{ck.replace('.pt', '')}.json"
        r = subprocess.run(
            [sys.executable, "research/w4_ar_eval.py", "--ckpt", ck,
             "--n", str(args.n), "--seed", str(args.seed),
             "--temps", args.temp, "--batch", str(args.batch), "--out", tmp],
            env=ENV, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {ck:<26}{'FAILED':>8}  {r.stderr.strip()[-160:]}",
                  flush=True)
            rows.append(dict(ckpt=ck, failed=True,
                             stderr=r.stderr.strip()[-2000:]))
            continue
        ev = json.load(open(tmp))[f"t{args.temp}"]
        lo = loss_of(ck)
        if lo["val_total"] is None and "v1" in ck:
            lo["val_total"] = args.v1_val
        rec = dict(ckpt=ck, **lo, contract=ev["contract"],
                   miss_p50=ev["miss_p50"], n_events_p50=ev["n_events_p50"],
                   dur_only=ev["dur_only"], n=ev["n"])
        rows.append(rec)
        te = f"{lo['train_ema']:.4f}" if lo["train_ema"] is not None else "-"
        vt = f"{lo['val_total']:.4f}" if lo["val_total"] is not None else "-"
        print(f"  {ck:<26}{str(lo['step']):>8}{te:>10}{vt:>10}"
              f"{ev['contract']:>10.4f}{ev['miss_p50']:>9.1f}"
              f"{ev['n_events_p50']:>8.0f}", flush=True)

    ok = [r for r in rows if not r.get("failed") and r.get("val_total")]
    out = dict(n=args.n, seed=args.seed, temp=args.temp, rows=rows)
    if len(ok) >= 2:
        dl = ok[-1]["val_total"] - ok[0]["val_total"]
        da = ok[-1]["contract"] - ok[0]["contract"]
        out["delta_loss"] = dl
        out["delta_contract"] = da
        out["auc_per_nat"] = da / dl if abs(dl) > 1e-9 else None
        print(f"\n  held out loss {ok[0]['val_total']:.4f} -> "
              f"{ok[-1]['val_total']:.4f} ({dl:+.4f})")
        print(f"  contract AUC  {ok[0]['contract']:.4f} -> "
              f"{ok[-1]['contract']:.4f} ({da:+.4f})")
        if abs(dl) > 1e-9:
            print(f"  slope {da / dl:+.4f} AUC per nat of held out loss")

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  run to run AUC noise is plus or minus 0.03, so a single step of")
    print("  the curve means nothing and only the trend across snapshots does.")
    print("  reference split-half floor 0.467 to 0.512.")


if __name__ == "__main__":
    main()
