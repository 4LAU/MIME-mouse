"""Is the bulk narrowing or are only the tails thinning? Per feature, both.

WHY THIS EXISTS. This workstream has been calling something "collapse" on the
strength of `spread_err` and the collapse flag, and RESUME's own 2026-08-07
bullet 2 says that reading is wrong: on the three features the flag fires on, the
interquartile ratio to human is 1.04 to 1.10 while the standard deviation ratio
is 0.16 to 0.22. The bulk is already the right width. The flag reports missing
extreme tails on features whose human distributions are enormously heavy tailed,
and that entry ends "do not treat the collapse flag as a dispersion target".

It matters right now because `w4_advmoment`'s writeup leans on the word. That arm
matched the whole distribution of eight critic heads by sorted matching, an
objective that is MAXIMAL for a point mass, and the AUC still rose 0.6338 to
0.6860. The argument that this is surprising, and the estimator bias hypothesis
built on top of it, both need the model to actually be moving toward a point
mass. If instead the tails thinned while the bulk held its width, there is no
contradiction to explain and `w4_estimator`'s motivation evaporates.

WHAT IT REPORTS. For each of the eighteen contract features, sampled from a
checkpoint against a human reference:

    iqr     ratio of the interquartile range, the BULK width
    p10_90  ratio of the tenth to ninetieth percentile range
    p1_99   ratio of the first to ninety ninth percentile range
    sd      ratio of the standard deviation, which the tails dominate

A bulk that has narrowed shows it in `iqr`. Tails alone show a ratio that falls
monotonically from `iqr` to `sd` while `iqr` stays near one. The four together
say WHERE in the distribution the mismatch lives, which one number never can.

Two checkpoints are compared so the reading is a CHANGE and not a level: the base
already sits at 0.6338 and already has its own tail deficit, so only the movement
between the two is attributable to training.

No training, no gradient, no scoring decision. Sampling and percentiles.
The protected checkpoint is never written. The eval sample is never read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("EVENT_SNAP", "2.5")
for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from w4_rollout import (  # noqa: E402
    CKPT, COOL_C, COOL_MAX_S, D, GATE_C, HELD_OUT, KILL_C, RESUME_C, TRAINED,
    decode_batch, gpu_temp,
)

OUT_JSON = "research/w4_bulktail.json"


def widths(X):
    """The four width statistics, per column, in one pass."""
    q = np.percentile(X, [1, 10, 25, 75, 90, 99], axis=0)
    return {"iqr": q[3] - q[2], "p10_90": q[4] - q[1],
            "p1_99": q[5] - q[0], "sd": X.std(0)}


def sample_features(path, cond, ang, cap, dev, batch, thermal):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"] if "config" in ck else None
    model = EventARModel(**cfg).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    S, TH, DT = [], [], []
    for c0 in range(0, len(cond), batch):
        if thermal() >= KILL_C:
            raise SystemExit(f"  GPU at or above the {KILL_C}C kill. Stopping.")
        with torch.no_grad():
            s, th, dt = model.sample(cond[c0:c0 + batch].to(dev), seq_len=cap)
        S.append(s.cpu().numpy()); TH.append(th.cpu().numpy())
        DT.append(dt.cpu().numpy())
    X, _, _ = decode_batch(list(np.concatenate(S)), list(np.concatenate(TH)),
                           list(np.concatenate(DT)), ang)
    del model
    torch.cuda.empty_cache()
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--trained-ckpt", type=str,
                    default="research/w4_advmoment.pt")
    a = ap.parse_args()

    t = gpu_temp()
    if t > GATE_C:
        print(f"  GPU at {t}C, above the {GATE_C}C launch gate. Not starting.")
        return 1
    dev = torch.device("cuda")
    peak = {"t": 0}

    def thermal():
        import time
        v = gpu_temp()
        peak["t"] = max(peak["t"], v)
        if v >= COOL_C:
            c0 = time.time()
            while gpu_temp() > RESUME_C and time.time() - c0 < COOL_MAX_S:
                time.sleep(10)
            v = gpu_temp()
            peak["t"] = max(peak["t"], v)
        return v

    base_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cap = int(base_ck["config"]["max_seq_len"])
    del base_ck

    cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(cond_all))
    rows = np.sort(perm[:a.n])
    cond = torch.tensor(np.asarray(cond_all[rows], dtype=np.float32))
    ang = np.arctan2(cond[:, 3].numpy().astype(np.float64),
                     cond[:, 2].numpy().astype(np.float64))

    # The contract's own reference is the headline, because it is the sample the
    # AUC is measured against. sir is carried alongside only because spread_err
    # is reported against it everywhere else in the record and the two disagree
    # by up to eight times on exactly the statistic in question.
    HREF = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    HSIR = np.load("data/human_ref_features_sir.npy")
    wref, wsir = widths(HREF), widths(HSIR)
    print(f"\n  human reference {len(HREF)} rows, sir {len(HSIR)} rows, "
          f"sampling {a.n} rows per checkpoint at buffer {cap}", flush=True)

    out = {"config": vars(a), "features": FEATURE_NAMES, "checkpoints": {}}
    for tag, path in (("base", CKPT), ("advmoment_step100", a.trained_ckpt)):
        X = sample_features(path, cond, ang, cap, dev, a.batch, thermal)
        w = widths(X)
        out["checkpoints"][tag] = {
            "path": path, "n": int(len(X)),
            **{f"{k}_ratio_ref": (w[k] / wref[k]).tolist() for k in w},
            **{f"{k}_ratio_sir": (w[k] / wsir[k]).tolist() for k in w},
        }
        print(f"\n  === {tag}   {len(X)} rows, ratios to the contract reference")
        print(f"  {'feature':<26}{'iqr':>9}{'p10_90':>9}"
              f"{'p1_99':>9}{'sd':>9}   set")
        for i, f in enumerate(FEATURE_NAMES):
            grp = ("held" if f in HELD_OUT
                   else "trained" if f in TRAINED else "other")
            print(f"  {f:<26}{w['iqr'][i] / wref['iqr'][i]:>9.3f}"
                  f"{w['p10_90'][i] / wref['p10_90'][i]:>9.3f}"
                  f"{w['p1_99'][i] / wref['p1_99'][i]:>9.3f}"
                  f"{w['sd'][i] / wref['sd'][i]:>9.3f}   {grp}", flush=True)

    b = out["checkpoints"]["base"]
    t100 = out["checkpoints"]["advmoment_step100"]
    print("\n  === CHANGE, step 100 ratio minus base ratio, contract reference")
    print("  A narrowed BULK shows as a negative iqr column. Tails only shows")
    print("  as iqr near zero with sd clearly negative.")
    print(f"  {'feature':<26}{'d iqr':>9}{'d p10_90':>9}"
          f"{'d p1_99':>9}{'d sd':>9}   set")
    for i, f in enumerate(FEATURE_NAMES):
        grp = ("held" if f in HELD_OUT
               else "trained" if f in TRAINED else "other")
        print(f"  {f:<26}"
              f"{t100['iqr_ratio_ref'][i] - b['iqr_ratio_ref'][i]:>9.3f}"
              f"{t100['p10_90_ratio_ref'][i] - b['p10_90_ratio_ref'][i]:>9.3f}"
              f"{t100['p1_99_ratio_ref'][i] - b['p1_99_ratio_ref'][i]:>9.3f}"
              f"{t100['sd_ratio_ref'][i] - b['sd_ratio_ref'][i]:>9.3f}"
              f"   {grp}", flush=True)
    for k in ("iqr", "p10_90", "p1_99", "sd"):
        d = np.array(t100[f"{k}_ratio_ref"]) - np.array(b[f"{k}_ratio_ref"])
        tr = [FEATURE_NAMES.index(x) for x in TRAINED]
        print(f"  mean change over the 12 TRAINED features, {k:<7}"
              f"{float(d[tr].mean()):>9.3f}")

    out["gpu_peak_c"] = peak["t"]
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {OUT_JSON}, GPU peak {peak['t']}C")
    return 0


if __name__ == "__main__":
    sys.exit(main())
