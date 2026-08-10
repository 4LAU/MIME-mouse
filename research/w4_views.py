"""Which restricted view of the emitted token stream gives the model away.

PRE REGISTERED in HANDOFF.md 2026-08-09, "## Restricted views of the token
stream". Views, controls, thresholds and the prediction were fixed before this
file was run.

Why this and not `w4_channels`. That arm built hybrid sequences and every one of
the six mixtures scored worse than both parents, so the intervention cost more
than the quantity it was measuring, and it is retired. A restricted view never
constructs anything. Each side keeps its own real sequences and the
discriminator is simply shown less: only the speed marginal, only the turn
autocorrelation, only the speed to turn coupling. Nothing that could not exist
is ever scored, and the question is the same one, which aspect of what the
model emits is detectably not human.

The floor, and why it is not the coarse view. The first draft made the coarse
size of the movement a control and scored every view by what it added on top.
A 1200 row smoke run killed that: the coarse view alone reads 0.403, which is
not a weak signal but no signal at all, six standard errors BELOW chance. A
random forest read out of bag sits under 0.5 when its features are pure noise,
so subtracting it inflates every view by about a tenth. The generation is
conditioned on each row's own distance and duration and duration obedience is
0.998, so the sizes genuinely match and there was never a size confound to
remove.

The floor is therefore the split half of the house convention. For each view,
half the human rows are scored against the other half, which is the same test
with no difference to find, and half the human rows are scored against their own
paired generated rows. Both use identical sample sizes, so the null level, the
below chance artifact included, is common to the two and the excess of one over
the other is the readout.

This is a DIAGNOSTIC two sample test on token statistics. It is deliberately not
`scoring.score_features`, which reads the eighteen contract features against
`data/human_val_features_grpo.npy` and is a different feature space and a
different reference sample. The recipe is copied exactly, balanced classes,
RandomForestClassifier at 100 estimators with OOB and random_state 42, so the
numbers are the same KIND of object, but no decision about the deliverable is
taken from them. The contract number for the same generated rows is computed
through `scoring` and reported alongside.

Safety. Reads the training corpus and one checkpoint, writes neither. Touches no
evaluation data and no scoring code, never `training/candi_polar_flow_best.pt`.
Paces on GPU temperature through `w4_latent`.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python -u research/w4_views.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

os.environ.setdefault("EVENT_SNAP", "2.5")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import extract_features, resample_trajectory  # noqa: E402
from models.event_ar import (DT_MAX_MS, EventARModel, class_to_dt_ms)  # noqa: E402
from models.event_stream_polar import (S_PAD_CLASS, TH_NULL_CLASS,  # noqa: E402
                                       class_to_dtheta, class_to_speed,
                                       dth_lattice_to_class, s2_to_class)
from w4_latent import cooldown, gpu_c, throttle  # noqa: E402

DATA = Path("training")
RF_SEED, RF_TREES = 42, 100
LAGS = (1, 2, 3, 4, 5, 6, 7, 8)
XLAGS = (-3, -2, -1, 0, 1, 2, 3)
QS = (10, 20, 30, 40, 50, 60, 70, 80, 90)

# A view has to beat the coarse control by this much to be called the place the
# model gives itself away, and the leader has to beat every rival family by it
# as well. It is roughly four times the replicate spread this file measures on
# unpaired AUC comparisons, which is about 0.0069.
MARGIN = 0.030


def acf(x, lags=LAGS):
    out = []
    for k in lags:
        a, b = x[:-k], x[k:]
        if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


def xcorr(x, y, lags=XLAGS):
    """Correlation of x with y shifted by lag. Positive lag means y later."""
    out = []
    for k in lags:
        a = x[:-k] if k > 0 else (x[-k:] if k < 0 else x)
        b = y[k:] if k > 0 else (y[:k] if k < 0 else y)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if n < 3 or a.std() < 1e-12 or b.std() < 1e-12:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


def stats(speed, theta, dt):
    """Every statistic this file knows how to compute, as named blocks.

    speed is pixels moved on the event, theta the turn in radians, dt the wait
    in milliseconds. All three are per event and the same length.
    """
    absth = np.abs(theta)
    v = speed / np.maximum(dt, 0.5)                 # px per ms
    k = absth / np.maximum(speed, 0.5)              # radians per px, curvature

    # The two thirds power law of human movement writes angular velocity as a
    # power of curvature, equivalently v proportional to k to the minus a third.
    # A model can have both marginals right and still put this slope elsewhere.
    m = (speed > 0.5) & (absth > 1e-3)
    if m.sum() >= 8:
        lk, lv = np.log(k[m]), np.log(v[m])
        A = np.vstack([lk, np.ones_like(lk)]).T
        coef, *_ = np.linalg.lstsq(A, lv, rcond=None)
        resid = float(np.std(lv - A @ coef))
        law = [float(coef[0]), float(coef[1]), resid]
    else:
        law = [0.0, 0.0, 0.0]

    return {
        "coarse": [float(len(speed)), float(dt.sum()), float(speed.sum())],
        "speed_marg": ([float(x) for x in np.percentile(speed, QS)]
                       + [float(speed.mean()), float(speed.std())]),
        "turn_marg": ([float(x) for x in np.percentile(absth, QS)]
                      + [float(absth.mean()), float(theta.std()),
                         float((absth < 1e-3).mean())]),
        "dt_marg": ([float(x) for x in np.percentile(dt, QS)]
                    + [float(dt.mean()), float(dt.std())]),
        "speed_acf": acf(speed),
        "turn_acf": acf(theta) + acf(absth),
        "dt_acf": acf(dt),
        "couple_st": xcorr(speed, absth) + law,
        "couple_sd": xcorr(speed, dt),
        "couple_td": xcorr(absth, dt),
    }


FAMILY = {"speed_marg": "marginal", "turn_marg": "marginal",
          "dt_marg": "marginal", "speed_acf": "temporal",
          "turn_acf": "temporal", "dt_acf": "temporal",
          "couple_st": "coupling", "couple_sd": "coupling",
          "couple_td": "coupling"}


def rf_auc(a, b, seed=RF_SEED):
    """The contract's recipe, on whatever feature space it is handed."""
    n = min(len(a), len(b))
    X = np.vstack([np.asarray(a[:n], dtype=np.float64),
                   np.asarray(b[:n], dtype=np.float64)])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = RandomForestClassifier(n_estimators=RF_TREES, oob_score=True,
                                 n_jobs=-1, random_state=seed)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.oob_decision_function_[:, 1]))


def human_row(arr, i, min_len):
    L = int(arr["len"][i])
    if L < min_len:
        return None
    s2 = torch.from_numpy(arr["s2"][i, :L].astype(np.int64))
    dth = torch.from_numpy(arr["dth"][i, :L].astype(np.int64))
    s_cls = s2_to_class(s2)
    th_cls = torch.where(s2 > 0, dth_lattice_to_class(dth),
                         torch.full_like(dth, TH_NULL_CLASS))
    dt_cls = torch.round(torch.from_numpy(arr["dt"][i, :L].astype(np.float32))
                         ).long().clamp(0, DT_MAX_MS)
    return s_cls.numpy(), th_cls.numpy(), dt_cls.numpy()


def physical(s_cls, th_cls, dt_cls):
    """Classes to pixels, radians and milliseconds, for both sources
    identically, so no statistic below can read a difference of units."""
    speed = class_to_speed(torch.from_numpy(s_cls.astype(np.int64))
                           ).numpy().astype(np.float64)
    th = np.where(th_cls == TH_NULL_CLASS, 0.0,
                  class_to_dtheta(torch.from_numpy(
                      np.where(th_cls == TH_NULL_CLASS, 0,
                               th_cls).astype(np.int64))).numpy()
                  ).astype(np.float64)
    dt = class_to_dt_ms(torch.from_numpy(dt_cls.astype(np.int64))
                        ).numpy().astype(np.float64)
    return speed, th, dt


def decode_features(s_cls, th_cls, dt_cls, angle):
    dt_ms = class_to_dt_ms(torch.from_numpy(dt_cls.astype(np.int64)))
    dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
            / esp._DT_STD).numpy()
    p = esp._decode(dt_z, s_cls, th_cls, 0.0, 0.0, angle)
    if p is None or len(p) < 2:
        return None
    f = extract_features(resample_trajectory(np.asarray(p, dtype=np.float64),
                                             hz=125.0))
    if f is None or not np.all(np.isfinite(f)):
        return None
    return np.asarray(f, dtype=np.float64)


def generate(model, cond, batch, temp, dev, seed, min_len):
    out = []
    with torch.no_grad():
        for c0 in range(0, len(cond), batch):
            throttle()
            torch.manual_seed(seed + c0)
            blk = torch.tensor(cond[c0:c0 + batch], dtype=torch.float32,
                               device=dev)
            s_cls, th_cls, dt_cls = model.sample(blk, temperature=temp)
            s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
            dt_np = dt_cls.cpu().numpy()
            pad = s_np >= S_PAD_CLASS
            for j in range(s_np.shape[0]):
                L = int(pad[j].argmax()) if pad[j].any() else s_np.shape[1]
                out.append(None if L < min_len
                           else (s_np[j, :L], th_np[j, :L], dt_np[j, :L]))
    return out


def collect(rows, arr, gen, ang, min_len):
    """Statistic blocks and contract features for both sources, on the rows
    where both sides produced a usable trajectory."""
    H, G, HF, GF = [], [], [], []
    for j, i in enumerate(rows):
        h, g = human_row(arr, int(i), min_len), gen[j]
        if h is None or g is None:
            continue
        hs, gs = stats(*physical(*h)), stats(*physical(*g))
        if not all(np.all(np.isfinite(v)) for v in hs.values()):
            continue
        if not all(np.all(np.isfinite(v)) for v in gs.values()):
            continue
        hf, gf = (decode_features(*h, float(ang[j])),
                  decode_features(*g, float(ang[j])))
        if hf is None or gf is None:
            continue
        H.append(hs), G.append(gs), HF.append(hf), GF.append(gf)
    return H, G, np.asarray(HF), np.asarray(GF)


def block(rows, keys):
    return np.asarray([np.concatenate([r[k] for k in keys]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--out", default="research/w4_views.json")
    args = ap.parse_args()

    arr = {k: np.load(DATA / f"events_{k}.npy", mmap_mode="r")
           for k in ("s2", "dth", "dt", "len", "cond")}
    ok = np.load(DATA / "events_feat18_ok.npy")

    # Random rows, never a prefix. The corpus is ordered by session.
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(np.flatnonzero(ok), args.n, replace=False))
    cond = np.asarray(arr["cond"][rows], dtype=np.float32)
    ang = np.arctan2(cond[:, 3], cond[:, 2])

    dev = esp._DEVICE
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    print(f"\n  {len(rows)} random corpus rows, one generated trajectory each")
    print(f"  ckpt {args.ckpt} at step {ck.get('step', '?')}")
    print(f"  minimum {args.min_len} events, margin {MARGIN}\n")

    cooldown()
    gen = generate(model, cond, args.batch, args.temp, dev, args.seed + 1,
                   args.min_len)
    H, G, HF, GF = collect(rows, arr, gen, ang, args.min_len)
    print(f"  {len(H)} rows usable on both sides\n")
    if len(H) < 800:
        print("  ABORT, too few paired rows")
        return

    # Half the human rows against the other half is the same test with nothing
    # to find. Half the human rows against their own paired generated rows is
    # the test. Identical sample sizes, so the null level is common to both.
    perm = np.random.default_rng(args.seed + 7).permutation(len(H))
    ia, ib = perm[:len(H) // 2], perm[len(H) // 2:2 * (len(H) // 2)]

    def split_eval(keys):
        HA, HB = block([H[i] for i in ia], keys), block([H[i] for i in ib],
                                                        keys)
        GA = block([G[i] for i in ia], keys)
        floor, sig = rf_auc(HA, HB), rf_auc(HA, GA)
        return floor, sig, sig - floor

    print(f"  {len(ia)} per class after the split half\n")
    print(f"  {'view':>12}{'family':>10}{'floor':>9}{'signal':>9}"
          f"{'excess':>10}")

    views, best = {}, {}
    for k in ["coarse"] + list(FAMILY):
        fl, sg, ex = split_eval([k])
        fam = FAMILY.get(k, "control")
        views[k] = {"floor": fl, "signal": sg, "excess": ex, "family": fam}
        if fam != "control":
            best[fam] = max(best.get(fam, -9), ex)
        print(f"  {k:>12}{fam:>10}{fl:>9.4f}{sg:>9.4f}{ex:>10.4f}")

    a_fl, a_sg, a_ex = split_eval(["coarse"] + list(FAMILY))
    print(f"\n  {'every view':>12}{'':>10}{a_fl:>9.4f}{a_sg:>9.4f}{a_ex:>10.4f}")

    # The apples to apples comparison for the line above. Same two samples,
    # same recipe, the eighteen contract features instead of token statistics.
    # If this is far above every view together, the difference the model makes
    # is small in what it emits and the decoder is amplifying it.
    c_fl = rf_auc(HF[ia], HF[ib])
    c_sg = rf_auc(HF[ia], GF[ia])
    print(f"  {'contract 18':>12}{'':>10}{c_fl:>9.4f}{c_sg:>9.4f}"
          f"{c_sg - c_fl:>10.4f}")
    contract = float(scoring.score_features(GF)["auc_rf_oob"])
    ceiling = float(scoring.score_features(HF)["auc_rf_oob"])
    print(f"\n  for reference only, against the validation human and so not")
    print(f"  comparable to the block above, generated {contract:.4f}, "
          f"human {ceiling:.4f}")

    lead = max(FAMILY, key=lambda k: views[k]["excess"])
    lead_fam = FAMILY[lead]
    rivals = [v for f, v in best.items() if f != lead_fam]
    dmax = views[lead]["excess"]
    if dmax < MARGIN:
        verdict = (f"NULL. No view beats its own split half floor by "
                   f"{MARGIN}, the best being {lead} at {dmax:+.4f}. The token "
                   f"stream is not detectably not human in any view this file "
                   f"knows how to compute, and the separation is entering at "
                   f"decode or resample rather than in what the model emits.")
    elif all(dmax - r > MARGIN for r in rivals):
        verdict = (f"{lead_fam.upper()}. {lead} leads at {dmax:+.4f} over the "
                   f"coarse control and beats every rival family by more than "
                   f"{MARGIN}. The next arm targets the {lead_fam} structure.")
    else:
        verdict = (f"MIXED. {lead} leads at {dmax:+.4f} but does not clear "
                   f"every rival family by {MARGIN}, so no single family owns "
                   f"the separation and an arm aimed at one of them is a "
                   f"guess.")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"n": len(H), "n_per_class": len(ia), "seed": args.seed,
               "temp": args.temp, "ckpt": args.ckpt, "views": views,
               "all_views": {"floor": a_fl, "signal": a_sg, "excess": a_ex},
               "contract_paired": {"floor": c_fl, "signal": c_sg,
                                   "excess": c_sg - c_fl},
               "contract_vs_validation_generated": contract,
               "contract_vs_validation_human": ceiling, "verdict": verdict,
               "gpu_c": gpu_c()}, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
