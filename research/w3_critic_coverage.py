"""Does the learned critic see the movements the model cannot produce?

Phase 0b (research/phase0b_critic.py) settled that a whole-path critic beats
the detector we are graded by: handed the geometric vocabulary as explicit
channels it reads 0.7918 against the RF's 0.7570. That result is what makes a
closed adversarial loop worth running at all, since a teacher weaker than the
examiner teaches nothing.

It does not make the loop worth running yet. research/w3_missing_paths.py
showed the gap is not spread across human movement but concentrated: the arm
sits at chance (0.4821) against three quarters of real paths and at 0.9400
against the remaining quarter, which is the smooth, the hesitant, and the long
fast kind. A critic could reach 0.79 overall while being blind to exactly that
quarter, and a Phase 1 loop driven by it would then spend its GPU polishing the
part already indistinguishable. The overall number cannot tell those apart.

So this splits the critic's own score the same way. One critic, trained by
cross-validation over the whole set, then read twice: against the real paths
the model covers, and against the quarter it does not. The RF is scored on the
identical split for reference, so the two teachers are compared on one ruler.

  critic uncovered >> critic covered   the critic has found the real deficit,
                                       Phase 1 is aimed at the right target
  critic uncovered ~ critic covered    it separates on something else, and the
                                       loop would need a different signal

Two differences from Phase 0b, both deliberate. The paths are the current best
single-trajectory arm (the event model plus additive arrival correction, 0.7283)
rather than the flow model's, because that is the checkpoint Phase 1 would
fine-tune. And the per-step deltas come from the 125Hz resampled path, so the
constant dt the geometric channels assume is exactly true here instead of the
approximation Phase 0b had to accept.

The critic machinery is imported from phase0b_critic, never reimplemented.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_critic_coverage.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
from features import resample_trajectory  # noqa: E402
from phase0b_critic import (MAX_LEN, N_CHANNELS, build_all_channels,  # noqa: E402
                            robust_standardize, run_cv)
from w3_fallback_arrival import correct_additive  # noqa: E402

CACHE = R / "research" / "w3_landing_cache.pkl"
OUT = R / "research" / "w3_critic_coverage_results.json"
HZ = 125.0


def to_steps(traj):
    """Per-step (dx, dy) of the 125Hz resampled path, the form the RF measures."""
    p = np.asarray(resample_trajectory(traj, hz=HZ), dtype=np.float64)
    if len(p) < 6:
        return None
    d = np.diff(p[:, :2], axis=0)
    return d[:MAX_LEN]


def pack(trajs):
    """(n, MAX_LEN, 2) deltas and the valid-step mask, dropping unusable paths."""
    steps = [to_steps(t) for t in trajs]
    keep = np.array([s is not None for s in steps])
    steps = [s for s in steps if s is not None]
    dxdy = np.zeros((len(steps), MAX_LEN, 2), dtype=np.float64)
    mask = np.zeros((len(steps), MAX_LEN), dtype=bool)
    for i, s in enumerate(steps):
        dxdy[i, :len(s)] = s
        mask[i, :len(s)] = True
    n_trunc = sum(1 for t in trajs
                  if (r := to_steps(t)) is not None and len(r) == MAX_LEN)
    return dxdy, mask, keep, n_trunc


def subset_auc(scores, y, keep_neg):
    """AUC of scores over all positives against a chosen subset of negatives."""
    sel = (y == 1) | keep_neg
    return float(roc_auc_score(y[sel], scores[sel]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.time()

    # ---- the arm Phase 1 would fine-tune, and the real reference ----------
    # pickle.load: this repo's own artifact from the 2026-07-20 landing-price
    # run on this machine, never third-party input.
    with open(CACHE, "rb") as fh:
        specs, trajs = pickle.load(fh)
    arm = [correct_additive(np.asarray(t), *(int(v) for v in s)) if len(t) >= 3
           else np.asarray(t) for s, t in zip(specs, trajs)]
    real = real_paths(args.n, args.seed, "ref")

    # a path must survive both filters, and it must be dropped from the
    # features too: the coverage split below is positional into these arrays,
    # so a path present for the RF but absent from the critic's input would
    # silently shift every index after it.
    Xa = features_with_jitter(arm, 0.0, args.seed)
    ka = np.all(np.isfinite(Xa), axis=1) & np.array(
        [to_steps(p) is not None for p in arm])
    Xa, arm = Xa[ka], [p for p, k in zip(arm, ka) if k]
    Xr = features_with_jitter(real, 0.0, args.seed)
    kr = np.all(np.isfinite(Xr), axis=1) & np.array(
        [to_steps(p) is not None for p in real])
    Xr, real = Xr[kr], [p for p, k in zip(real, kr) if k]
    print(f"[critic] dropped {int((~ka).sum())} arm, {int((~kr).sum())} real "
          f"as unusable (non-finite features or too few resampled points)")
    n = min(len(Xa), len(Xr))
    Xa, arm, Xr, real = Xa[:n], arm[:n], Xr[:n], real[:n]
    print(f"[critic] {n} arm paths, {n} real paths, device {args.device}")

    # ---- the coverage split, defined exactly as w3_missing_paths does -----
    rf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS,
                                oob_score=True, n_jobs=-1,
                                random_state=scoring.RF_SEED)
    rf.fit(np.vstack([Xr, Xa]), np.concatenate([np.zeros(n), np.ones(n)]))
    rf_oof = rf.oob_decision_function_[:, 1]
    p_real = 1.0 - rf_oof[:n]
    order = np.argsort(-p_real)
    k = max(n // 4, 10)
    uncovered, covered = order[:k], order[-k:]
    print(f"[critic] split: {k} uncovered, {k} covered")

    # ---- pack both classes into the critic's input form -------------------
    Da, Ma, keep_a, tr_a = pack(arm)
    Dr, Mr, keep_r, tr_r = pack(real)
    if not (keep_a.all() and keep_r.all()):
        # a dropped path would desynchronise the RF split indices below
        raise SystemExit(f"[critic] unusable paths: {(~keep_a).sum()} arm, "
                         f"{(~keep_r).sum()} real. Split indices assume none.")
    dxdy = np.concatenate([Dr, Da])
    pad = np.concatenate([Mr, Ma])
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)
    print(f"[critic] {tr_r + tr_a} paths hit the {MAX_LEN}-step ceiling")

    ch, n_nf, n_valid = build_all_channels(dxdy, pad)
    X, scales = robust_standardize(ch, pad)
    print(f"[critic] {N_CHANNELS} channels, "
          f"{n_nf}/{n_valid} non-finite clamped")

    # ---- one critic, cross-validated, read out of fold --------------------
    dev = torch.device(args.device)
    # run_cv early-stops each fold on its own validation AUC, so this number
    # is mildly optimistic. Inherited unchanged so it stays comparable to
    # Phase 0b's 0.7918, and the split below is a within-run comparison anyway.
    oof, critic_all, fold_aucs = run_cv(X.astype(np.float32), pad, y, dev,
                                        n_folds=args.folds, seed=args.seed)
    oof = np.asarray(oof)
    critic_all = float(critic_all)

    # ---- where does each teacher's separation live? -----------------------
    # A coverage split is defined by ranking real paths on one teacher's own
    # score, so that teacher is guaranteed to look extreme on it: the quarter
    # it calls least model-like is by construction where it separates best.
    # Reading only the RF-defined split would therefore flatter the RF and
    # understate the critic. Both splits are built and both teachers read on
    # each, so the honest comparison is the off-diagonal, where neither
    # teacher is being graded on a partition it chose.
    scores = {"critic": oof, "rf": rf_oof}
    res = {"all": {k_: float(roc_auc_score(y, s)) for k_, s in scores.items()},
           "by_split": {}}
    splits = {}
    for definer, s in scores.items():
        p_real_d = 1.0 - s[:n]
        od = np.argsort(-p_real_d)
        splits[definer] = {"uncovered": od[:k], "covered": od[-k:]}

    for definer, sp in splits.items():
        neg_c = np.zeros(2 * n, dtype=bool); neg_c[sp["covered"]] = True
        neg_u = np.zeros(2 * n, dtype=bool); neg_u[sp["uncovered"]] = True
        res["by_split"][definer] = {
            reader: {"covered": subset_auc(s, y, neg_c),
                     "uncovered": subset_auc(s, y, neg_u)}
            for reader, s in scores.items()}

    agree = len(set(splits["rf"]["uncovered"].tolist())
                & set(splits["critic"]["uncovered"].tolist())) / k
    res["uncovered_overlap"] = float(agree)

    print(f"\n{'split defined by':<18}{'read by':<10}{'covered':>10}"
          f"{'uncovered':>11}{'spread':>9}")
    for definer in ("rf", "critic"):
        for reader in ("rf", "critic"):
            d = res["by_split"][definer][reader]
            own = "  (own split)" if definer == reader else ""
            print(f"{definer:<18}{reader:<10}{d['covered']:>10.4f}"
                  f"{d['uncovered']:>11.4f}"
                  f"{d['uncovered'] - d['covered']:>9.4f}{own}")
    print(f"\noverall AUC: critic {res['all']['critic']:.4f}, "
          f"rf {res['all']['rf']:.4f}")
    print(f"the two teachers agree on {agree:.1%} of which quarter is missing")

    # the honest read: each teacher judged on the split it did not define
    c_x = res["by_split"]["rf"]["critic"]
    r_x = res["by_split"]["critic"]["rf"]
    c_sp, r_sp = (c_x["uncovered"] - c_x["covered"],
                  r_x["uncovered"] - r_x["covered"])
    print(f"\non the other teacher's split, the critic separates the two groups "
          f"by {c_sp:.4f} and the RF by {r_sp:.4f}")
    if c_sp < 0.15:
        print("VERDICT: the critic does not distinguish the missing quarter. "
              "A Phase 1 loop driven by it would not aim at the deficit.")
    elif c_sp >= r_sp - 0.05:
        print("VERDICT: the critic tracks the deficit at least as well as the "
              "RF does. Phase 1 is aimed at the right target.")
    else:
        print("VERDICT: the critic tracks the deficit but less sharply than "
              "the RF. Usable, not the stronger teacher where it counts.")

    np.savez(R / "research" / "w3_critic_coverage_scores.npz",
             critic_oof=oof, rf_oob=rf_oof, y=y,
             rf_uncovered=splits["rf"]["uncovered"],
             rf_covered=splits["rf"]["covered"],
             critic_uncovered=splits["critic"]["uncovered"],
             critic_covered=splits["critic"]["covered"])

    out = {"n_per_class": int(n), "k": int(k), "seed": args.seed,
           "folds": args.folds, "device": str(args.device),
           "n_at_length_ceiling": int(tr_r + tr_a),
           "critic_fold_aucs": [float(a) for a in fold_aucs],
           "aucs": res, "wall_sec": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[critic] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
