"""Give the timing head the turn token it was actually trained on.

PRE REGISTERED in HANDOFF.md 2026-08-10, "## Giving the timing head the turn
token it was trained on". Arms, controls, thresholds and the prediction were
fixed before this file was written.

The dataset sets the turn token to NULL wherever there is no motion, and the
turn head is not trained at those positions at all because `sup_th` masks them
out of the loss. `EventARModel.sample` made the same substitution AFTER the
timing head had already run, so the served path fed an untrained arbitrary turn
into the timing head at exactly those events. Humans put a one millisecond wait
on 37.2 percent of no motion events and the served model on 5.8 percent, while
teacher forced on a real human history the head predicts 0.294 against a true
0.297. The head is right and the call was wrong.

Paired. Both arms share commands and generation seeds and differ only by the
flag, so run to run spread mostly cancels. Serving still emits one trajectory
per command with no candidates and no selection.

Safety. Reads the corpus and one checkpoint, writes no model, touches no
evaluation data and no scoring code. Adjudicates through
`research/autoloop/scoring.py` against `data/human_val_features_grpo.npy`.
Paces on GPU temperature through `w4_latent`.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python -u research/w4_ticknull.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("EVENT_SNAP", "2.5")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import extract_feature_matrix  # noqa: E402
from models.event_ar import EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS, TICK_CLASS  # noqa: E402
from w4_latent import gpu_c, throttle  # noqa: E402

DATA = Path("training")

PATH_REF = 0.6337   # this same path, w4_prefix unforced arm
BASE_TOL = 0.035
WIN_AT = 0.025
PARTIAL_AT = 0.008
HUMAN_P1 = 0.372    # a hand's P(1 ms wait | no motion event)
MECH_MIN = 0.20     # below this the flag did not do what it claims


def generate(model, cond, batch, dev, seed, tick_th_null):
    """One trajectory per command. Nothing scored, filtered or selected."""
    rows, keep, t0 = [], [], time.time()
    with torch.no_grad():
        for c0 in range(0, len(cond), batch):
            throttle()
            torch.manual_seed(seed + c0)
            blk = torch.tensor(cond[c0:c0 + batch], dtype=torch.float32,
                               device=dev)
            s_c, th_c, dt_c = model.sample(blk, temperature=1.0,
                                           tick_th_null=tick_th_null)
            s_np = s_c.cpu().numpy()
            th_np, dt_np = th_c.cpu().numpy(), dt_c.cpu().numpy()
            pad = s_np >= S_PAD_CLASS
            for j in range(s_np.shape[0]):
                L = int(pad[j].argmax()) if pad[j].any() else s_np.shape[1]
                if L >= 8:
                    rows.append((s_np[j, :L], th_np[j, :L], dt_np[j, :L]))
                    keep.append(c0 + j)
    print(f"      {len(rows)}/{len(cond)} in {time.time() - t0:.0f}s, "
          f"{gpu_c()}C", flush=True)
    return rows, np.asarray(keep, dtype=np.int64)


def score(rows, conds):
    paths = []
    for (s, th, dt), c in zip(rows, conds):
        d = class_to_dt_ms(torch.from_numpy(dt.astype(np.int64))).numpy()
        dz = (np.log(np.maximum(d, 0.05)) - esp._DT_MEAN) / esp._DT_STD
        p = esp._decode(dz, s, th, 0.0, 0.0,
                        float(np.arctan2(float(c[3]), float(c[2]))))
        if p is not None and len(p) >= 4:
            paths.append(np.asarray(p, dtype=np.float64))
    X = extract_feature_matrix(paths)
    X = X[np.all(np.isfinite(X), 1)]
    return float(scoring.score_features(X)["auc_rf_oob"]), int(len(X))


def tick_p1(rows):
    """The mechanism check: how often a no motion event carries a 1 ms wait."""
    hit = tot = 0
    for s, _, dt in rows:
        z = s == TICK_CLASS
        tot += int(z.sum())
        hit += int((dt[z] == 1).sum())
    return hit / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="research/w4_ticknull.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    cond_all = np.load(DATA / "events_cond.npy", mmap_mode="r")
    ok = np.flatnonzero(np.load(DATA / "events_feat18_ok.npy"))
    # The corpus is ordered by session, so draw at random and never a prefix.
    rng = np.random.default_rng(args.seed)
    ok = ok[rng.permutation(len(ok))][:args.seeds * args.n]

    ck = torch.load(DATA / args.ckpt, map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"\n  {args.ckpt} step {ck.get('step')}, {args.seeds} paired seeds "
          f"at {args.n}\n")

    print(f"  {'seed':>7}{'base':>9}{'fixed':>9}{'delta':>9}{'P1 base':>9}"
          f"{'P1 fixed':>10}{'ev base':>9}{'ev fixed':>10}")
    res = []
    for i in range(args.seeds):
        idx = np.sort(ok[i * args.n:(i + 1) * args.n])
        c = np.asarray(cond_all[idx], dtype=np.float32)
        s = args.seed + 4000 * (i + 1)
        br, bk = generate(model, c, args.batch, dev, s, False)
        fr, fk = generate(model, c, args.batch, dev, s, True)
        b_auc, b_n = score(br, c[bk])
        f_auc, f_n = score(fr, c[fk])
        b_p1, b_t = tick_p1(br)
        f_p1, f_t = tick_p1(fr)
        b_ev = float(np.mean([len(r[0]) for r in br]))
        f_ev = float(np.mean([len(r[0]) for r in fr]))
        b_du = float(np.mean([float(r[2].sum()) for r in br]))
        f_du = float(np.mean([float(r[2].sum()) for r in fr]))
        res.append({"seed": s, "base_auc": b_auc, "fixed_auc": f_auc,
                    "delta": f_auc - b_auc, "base_n": b_n, "fixed_n": f_n,
                    "base_p1": b_p1, "fixed_p1": f_p1,
                    "base_ticks": b_t, "fixed_ticks": f_t,
                    "base_events": b_ev, "fixed_events": f_ev,
                    "base_dur_ms": b_du, "fixed_dur_ms": f_du})
        print(f"  {s:>7}{b_auc:>9.4f}{f_auc:>9.4f}{f_auc - b_auc:>9.4f}"
              f"{b_p1:>9.3f}{f_p1:>10.3f}{b_ev:>9.1f}{f_ev:>10.1f}",
              flush=True)

    base = float(np.mean([r["base_auc"] for r in res]))
    fixed = float(np.mean([r["fixed_auc"] for r in res]))
    delta = fixed - base
    spread = float(np.std([r["delta"] for r in res], ddof=1)) \
        if len(res) > 1 else float("nan")
    p1b = float(np.mean([r["base_p1"] for r in res]))
    p1f = float(np.mean([r["fixed_p1"] for r in res]))
    evb = float(np.mean([r["base_events"] for r in res]))
    evf = float(np.mean([r["fixed_events"] for r in res]))
    dub = float(np.mean([r["base_dur_ms"] for r in res]))
    duf = float(np.mean([r["fixed_dur_ms"] for r in res]))

    if abs(base - PATH_REF) > BASE_TOL:
        verdict = (f"VOID. The base arm reads {base:.4f} against {PATH_REF} for "
                   f"this same path, outside the registered {BASE_TOL}.")
    elif p1f < MECH_MIN:
        verdict = (f"VOID. The mechanism check failed. P(1 ms at a no motion "
                   f"event) moved {p1b:.3f} to {p1f:.3f}, short of the "
                   f"registered {MECH_MIN}, so the flag did not do what it "
                   f"claims and no score change may be attributed to it.")
    elif -delta >= WIN_AT:
        verdict = (f"WIN. Fixed beats base by {-delta:.4f}, past the "
                   f"registered {WIN_AT}.")
    elif -delta >= PARTIAL_AT:
        verdict = (f"PARTIAL. Fixed beats base by {-delta:.4f}, past the "
                   f"registered {PARTIAL_AT} but short of {WIN_AT}.")
    else:
        verdict = (f"NULL. Fixed moves base by {delta:+.4f}, short of the "
                   f"registered {PARTIAL_AT}, even though the mechanism check "
                   f"passed at {p1f:.3f} against a human {HUMAN_P1}.")

    print(f"\n  base {base:.4f}  fixed {fixed:.4f}  delta {delta:+.4f} "
          f"(seed spread {spread:.4f})")
    print(f"  P(1 ms at no motion)  base {p1b:.3f}  fixed {p1f:.3f}  "
          f"human {HUMAN_P1}")
    print(f"  events {evb:.1f} to {evf:.1f}, token duration {dub:.0f} to "
          f"{duf:.0f} ms")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"ckpt": args.ckpt, "n": args.n, "seeds": args.seeds,
               "seed": args.seed, "base": base, "fixed": fixed,
               "delta": delta, "seed_spread": spread,
               "p1_base": p1b, "p1_fixed": p1f, "human_p1": HUMAN_P1,
               "events_base": evb, "events_fixed": evf,
               "dur_base_ms": dub, "dur_fixed_ms": duf,
               "runs": res, "verdict": verdict, "gpu_c": gpu_c()},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
