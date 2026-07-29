"""Does the trunk's motion change with the duration it is handed, or only its length?

research/w3_conditional_gate.py found the one localisation that survives its
controls: rewriting the arm so its dependence structure varies across duration
terciles the way the humans' does closes 65 to 69 percent of the distance to
the floor, 8 replications out of 8. The human structure genuinely swings, rank
correlation between max_velocity and curvature_mean running -0.42, -0.12, +0.07
from short movements to long.

Reading the trunk shows it is ALREADY handed the duration: the conditioning
vector is [log_dist, log_dur, cos(angle), sin(angle)], with log_dur drawn from
the duration prior before a single event is generated. So the brief is not
"give the model duration". It is "the model has duration and does not use it to
decide how to move". This probe checks whether that is true, because if the
model's structure swings too then the brief is wrong and nothing should be
built on it.

Distance is held fixed, which is the whole point. Duration and distance are
strongly related in people, so a swing measured across all durations at once
could be a distance effect wearing a duration costume. Inside a narrow distance
band that confound is gone, and if the human swing does not survive the band
then the conditional gate's RESULT still stands as an AUC but its duration
INTERPRETATION does not, and this probe has to say so.

Four readings.

  human, banded     human paths in the distance band, split into duration
                    terciles. The swing that has to survive for any of this to
                    mean anything.

  model, commanded  the duration prior is replaced so that each tercile's
                    generation is commanded with durations drawn from the
                    HUMAN durations in that same tercile. Same distances, same
                    angles, only the commanded duration differs.

  realized          what duration the paths actually came out at, per band.
                    If the trunk does not honour the command there is nothing
                    to interpret and the probe stops there.

  swing comparison  the same pair correlations, human against model, band by
                    band. Flat model against swinging human is the finding.

Local GPU. Launch gate 75C, watchdog 83C, supervised.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w3_duration_response.py \
      --ckpt event_polar_4m_resid_v2.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

OUT = R / "research" / "w3_duration_response_results.json"
NBAND = 3
PAIRS = [("max_velocity", "curvature_mean"),
         ("max_velocity", "angular_velocity_mean"),
         ("mean_velocity", "curvature_std"),
         ("time_to_peak_velocity", "curvature_mean"),
         ("max_acceleration", "path_efficiency"),
         ("mean_jerk", "mean_acceleration")]


def rank_corr(X, i, j):
    """Spearman on two columns. Rank, not Pearson: these features carry
    outliers spanning eight orders of magnitude and Pearson on them is decided
    by a handful of paths (see w3_joint_structure)."""
    a, b = X[:, i], X[:, j]
    ra, rb = np.empty(len(a)), np.empty(len(b))
    ra[np.argsort(a, kind="stable")] = np.arange(len(a))
    rb[np.argsort(b, kind="stable")] = np.arange(len(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def banded(X, dur_col, k=NBAND):
    """Equal-count duration bands by rank."""
    o = np.argsort(X[:, dur_col], kind="stable")
    lab = np.empty(len(X), dtype=int)
    lab[o] = (np.arange(len(X)) * k) // len(X)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_polar_4m_resid_v2.pt")
    ap.add_argument("--n-per-band", type=int, default=1500)
    ap.add_argument("--n-real", type=int, default=20000)
    ap.add_argument("--dist-lo", type=float, default=0.40,
                    help="lower quantile of the human distance band")
    ap.add_argument("--dist-hi", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    # the locked decode recipe. Any deviation makes the number incomparable to
    # the 0.6986 standing result, so it is set here and not left to the shell.
    os.environ.update(EVENT_CKPT=args.ckpt, EVENT_ORDER="gumbel",
                      EVENT_RESID_MODE="prefix", EVENT_SNAP="2.5",
                      EVENT_DUR_STD="1.0", DUR_EMPIRICAL="1",
                      EVENT_CHOICE_TEMP="10", EVENT_BESTOF="1", EVENT_SIR="1")
    for k in ("EVENT_POOL_TOKENS", "EVENT_POOL_SAVE", "EVENT_POOL_LOAD",
              "EVENT_RESID_ONLY", "EVENT_FEAT_CORR", "EVENT_Z_FILE"):
        os.environ.pop(k, None)

    from degeneracy_panel import features_with_jitter, real_paths  # noqa: E402
    from features import FEATURE_NAMES  # noqa: E402

    IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
    DUR = IDX["movement_duration"]

    real = real_paths(args.n_real, args.seed, "ref")
    dist = np.array([math.hypot(p[-1][0] - p[0][0], p[-1][1] - p[0][1])
                     for p in real])
    lo, hi = np.quantile(dist, [args.dist_lo, args.dist_hi])
    keep = np.flatnonzero((dist >= lo) & (dist <= hi))
    Xr = features_with_jitter([real[i] for i in keep], 0.0, args.seed)
    ok = np.all(np.isfinite(Xr), axis=1)
    Xr, kd = Xr[ok], dist[keep][ok]
    print(f"[dur] distance band {lo:.0f} to {hi:.0f} px, {len(Xr)} human paths")

    hb = banded(Xr, DUR)
    hband_dur = [Xr[hb == j][:, DUR] for j in range(NBAND)]
    print(f"[dur] human duration bands (s): "
          + "  ".join(f"{d.mean():.3f}" for d in hband_dur))

    # replace the duration prior with a draw from one human band. Monkeypatched
    # rather than added to the trunk as an env hook: this is a one-off probe and
    # the trunk is the artefact every other result is measured on.
    import experiments.event_stream_polar as esp  # noqa: E402
    from phase_a_baseline import make_specs  # noqa: E402

    rng = np.random.default_rng(args.seed)
    specs = [(round(a), round(b), round(c), round(d))
             for a, b, c, d in make_specs(args.n_per_band * 4, args.seed)]
    sd = np.array([math.hypot(c - a, d - b) for a, b, c, d in specs])
    specs = [s for s, v in zip(specs, sd) if lo <= v <= hi][:args.n_per_band]
    print(f"[dur] {len(specs)} generation specs inside the same band")

    out = {"ckpt": args.ckpt, "seed": args.seed,
           "dist_band_px": [float(lo), float(hi)],
           "n_human": int(len(Xr)), "n_specs": len(specs),
           "human_band_duration_s": [float(d.mean()) for d in hband_dur],
           "bands": []}

    Xm = []
    for j in range(NBAND):
        pool = hband_dur[j]
        esp._duration.sample = (  # noqa: SLF001
            lambda _ld, _p=pool: float(rng.choice(_p)))
        t1 = time.time()
        raw = esp.generate_paths(specs)
        paths = [np.asarray(t) for t in raw if t is not None and len(t) >= 5]
        X = features_with_jitter(paths, 0.0, args.seed)
        X = X[np.all(np.isfinite(X), axis=1)]
        Xm.append(X)
        print(f"[dur] band {j}: commanded {pool.mean():.3f}s, realized "
              f"{X[:, DUR].mean():.3f}s, {len(X)} paths, {time.time()-t1:.0f}s")
        out["bands"].append({"band": j, "commanded_s": float(pool.mean()),
                             "realized_s": float(X[:, DUR].mean()),
                             "n": int(len(X))})

    # cache the feature matrices. Generation is 4.5 GPU-minutes and every
    # re-reading of this probe below is free arithmetic on these arrays.
    np.savez(R / "research" / f"w3_duration_response_cache_{Path(args.ckpt).stem}.npz",
             human=Xr, human_band=hb, **{f"model{k}": Xm[k] for k in range(NBAND)})

    cmd = np.array([b["commanded_s"] for b in out["bands"]])
    rlz = np.array([b["realized_s"] for b in out["bands"]])
    obeyed = (rlz[-1] - rlz[0]) / max(cmd[-1] - cmd[0], 1e-9)
    out["command_obeyed_fraction"] = float(obeyed)
    print(f"\n[dur] the trunk moved {obeyed:.0%} of the commanded duration "
          f"spread. Below about 50% and the rest of this is unreadable.")

    print(f"\n{'pair':<44}{'band0':>8}{'band1':>8}{'band2':>8}{'swing':>8}")
    rows = []
    for a, b in PAIRS:
        i, j = IDX[a], IDX[b]
        h = [rank_corr(Xr[hb == k], i, j) for k in range(NBAND)]
        m = [rank_corr(Xm[k], i, j) for k in range(NBAND)]
        hs, ms = max(h) - min(h), max(m) - min(m)
        rows.append({"a": a, "b": b, "human": h, "model": m,
                     "human_swing": hs, "model_swing": ms})
        print(f"{'human   ' + a + ' / ' + b:<44}" + "".join(f"{v:>8.3f}" for v in h)
              + f"{hs:>8.3f}")
        print(f"{'model   ' + a + ' / ' + b:<44}" + "".join(f"{v:>8.3f}" for v in m)
              + f"{ms:>8.3f}")
    out["pairs"] = rows

    # and the same question over every pair at once, so the answer does not
    # depend on the five that were picked in advance
    def allswing(X, lab):
        s = []
        for i in range(len(FEATURE_NAMES)):
            for j in range(i + 1, len(FEATURE_NAMES)):
                v = [rank_corr(X[lab == k], i, j) for k in range(NBAND)]
                s.append(max(v) - min(v))
        return np.array(s)

    mb = np.concatenate([np.full(len(Xm[k]), k) for k in range(NBAND)])
    sh, sm = allswing(Xr, hb), allswing(np.vstack(Xm), mb)
    out["all_pairs_swing"] = {"human_mean": float(sh.mean()),
                              "model_mean": float(sm.mean()),
                              "human_p90": float(np.quantile(sh, 0.9)),
                              "model_p90": float(np.quantile(sm, 0.9))}
    print(f"\nacross all 153 pairs, mean swing across the duration bands:")
    print(f"  human {sh.mean():.3f} (p90 {np.quantile(sh,0.9):.3f})    "
          f"model {sm.mean():.3f} (p90 {np.quantile(sm,0.9):.3f})")

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[dur] wrote {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
