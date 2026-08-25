"""w4_e1floor. AMENDMENT 36, registered in step0_prereg.md before this
file existed.

Decompose the cost of the model's own representation. Five human only
arms over identical rows: raw corpus geometry, then each token round
trip added, then the full decoder. No generative model is involved.
Diagnostic only, never a training signal, no selection. Never reads
the protected eval file.
"""
import json
import os
import sys

import numpy as np
import torch

# the identical decoder environment qladder pins
os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_SNAP", "2.5")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_BESTOF", "1")
os.environ.setdefault("EVENT_SIR", "1")
os.environ.setdefault("EVENT_ORDER", "gumbel")
os.environ.setdefault("EVENT_STEPS", "100")
os.environ.setdefault("EVENT_CFG_W", "0")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp                       # noqa: E402
import scoring                                                     # noqa: E402
import features as feat                                            # noqa: E402
from features import (FEATURE_NAMES, extract_features,             # noqa: E402
                      resample_trajectory)
from models.event_ar import DT_MAX_MS, class_to_dt_ms, dt_ms_to_class  # noqa: E402
from models.event_stream_polar import (S_PAD_CLASS, TICK_CLASS,     # noqa: E402
                                       TH_NULL_CLASS, class_to_dtheta,
                                       class_to_speed, dth_lattice_to_class,
                                       s2_to_class)
import ledger                                                      # noqa: E402

SEEDS = [40, 41, 42, 43, 44, 45]
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256
KMAX = 4
N = 2000
SHUF_SEED = 3204
ARMS = ["L0_RAW", "L1_DT", "L2_POLAR", "L3_FULL", "L3B_NOSNAP"]
GATE = 0.5250


def _time_axis(dt_ms):
    """The decoder's own time handling, applied identically in every arm so
    the clip is shared preprocessing and not an arm difference."""
    return np.concatenate([[0.0], np.cumsum(np.clip(dt_ms, 0.1, 1000.0) / 1000.0)])


def build(seed, lengths, s2_all, dth_all, dt_all, dx_all, dy_all, cond_all, elig):
    pick = np.sort(np.random.default_rng(1000 + seed).choice(elig, N, replace=False))
    s2 = np.asarray(s2_all[pick])
    dth = np.asarray(dth_all[pick])
    dt_ms_all = np.asarray(dt_all[pick]).astype(np.float64)
    dx_all_r = np.asarray(dx_all[pick]).astype(np.float64)
    dy_all_r = np.asarray(dy_all[pick]).astype(np.float64)
    conds = np.asarray(cond_all[pick])
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    angs = np.arctan2(conds[:, 3].astype(np.float64), conds[:, 2].astype(np.float64))

    sc = s2_to_class(torch.from_numpy(np.asarray(s2, dtype=np.int64))).numpy()
    tc = np.where(np.asarray(s2) > 0,
                  dth_lattice_to_class(
                      torch.from_numpy(np.asarray(dth, dtype=np.int64))).numpy(),
                  TH_NULL_CLASS)
    dc = dt_ms_to_class(torch.from_numpy(dt_ms_all)).numpy()

    out = {a: np.full((N, len(FEATURE_NAMES)), np.nan) for a in ARMS}
    for i in range(N):
        n = int(L[i])
        dt_true = dt_ms_all[i, :n]
        dt_q = class_to_dt_ms(
            torch.from_numpy(dc[i, :n].clip(0, DT_MAX_MS))).numpy().astype(np.float64)

        # L0 RAW, exact corpus geometry
        x = np.concatenate([[0.0], np.cumsum(dx_all_r[i, :n])])
        y = np.concatenate([[0.0], np.cumsum(dy_all_r[i, :n])])
        paths = {"L0_RAW": (x, y, _time_axis(dt_true)),
                 "L1_DT": (x, y, _time_axis(dt_q))}

        # L2 POLAR, geometry from the tokens, true time, no snap or rounding
        s_cls = sc[i, :n]
        s = class_to_speed(torch.from_numpy(s_cls.astype(np.int64))).numpy()
        dth_a = class_to_dtheta(torch.from_numpy(tc[i, :n].astype(np.int64))).numpy()
        motion = s_cls > TICK_CLASS
        heading = angs[i] + np.cumsum(np.where(motion, dth_a, 0.0))
        pdx = np.where(motion, s * np.cos(heading), 0.0)
        pdy = np.where(motion, s * np.sin(heading), 0.0)
        paths["L2_POLAR"] = (np.concatenate([[0.0], np.cumsum(pdx)]),
                             np.concatenate([[0.0], np.cumsum(pdy)]),
                             _time_axis(dt_true))

        for a, (px, py, pt) in paths.items():
            if len(px) >= 5:
                fv = extract_features(resample_trajectory(
                    list(zip(px.tolist(), py.tolist(), pt.tolist()))))
                if fv is not None and np.all(np.isfinite(fv)):
                    out[a][i] = fv

        # L3 and L3B, the real decoder
        s_row = np.full(MAX_T, S_PAD_CLASS, dtype=np.int64)
        th_row = np.full(MAX_T, TH_NULL_CLASS, dtype=np.int64)
        dt_row = np.zeros(MAX_T, dtype=np.int64)
        s_row[:n] = s_cls
        th_row[:n] = tc[i, :n]
        dt_row[:n] = dc[i, :n].clip(0, DT_MAX_MS)
        d = class_to_dt_ms(torch.from_numpy(dt_row)).numpy()
        dz = (np.log(np.maximum(d, 0.05)) - esp._DT_MEAN) / esp._DT_STD
        for a, snap in (("L3_FULL", 2.5), ("L3B_NOSNAP", 0.0)):
            esp._SNAP = snap
            p = esp._decode(dz, s_row, th_row, 0.0, 0.0, float(angs[i]))
            if p is not None and len(p) >= 5:
                fv = extract_features(resample_trajectory(np.asarray(p, dtype=np.float64)))
                if fv is not None and np.all(np.isfinite(fv)):
                    out[a][i] = fv
        esp._SNAP = 2.5
    return pick, out


def auc(mat):
    m = mat[np.isfinite(mat).all(1)]
    m = m[np.random.default_rng(SHUF_SEED).permutation(len(m))]
    return float(scoring.score_features(m)["auc_rf_oob"]), len(m)


def main():
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    arrs = [np.load(f"training/events_{k}.npy", mmap_mode="r")
            for k in ("s2", "dth", "dt", "dx", "dy", "cond")]

    # Cached per seed so a fault costs one seed, not the run. The box takes
    # a random SIGILL under sustained numeric work; the cache is the same
    # dump discipline the qladder arms already use.
    per_seed = {}
    for s in SEEDS:
        paths = {a: f"research/w4_e1floor_F_{a}_s{s}.npy" for a in ARMS}
        if all(os.path.exists(v) for v in paths.values()):
            per_seed[s] = {a: np.load(v) for a, v in paths.items()}
            print(f"  seed {s}: cached", flush=True)
            continue
        _, per_seed[s] = build(s, lengths, *arrs, elig)
        for a, v in paths.items():
            np.save(v, per_seed[s][a])
        ok = {a: int(np.isfinite(per_seed[s][a]).all(1).sum()) for a in ARMS}
        print(f"  seed {s}: valid {ok}", flush=True)

    pooled = {a: np.concatenate([per_seed[s][a] for s in SEEDS]) for a in ARMS}

    print("\n  VALIDITY GATE, L0 raw human pooled against the anchor:", flush=True)
    a0, n0 = auc(pooled["L0_RAW"])
    print(f"  L0_RAW {a0:.4f} on {n0} rows (bar {GATE})", flush=True)
    if a0 > GATE:
        print("  GATE FAILED, this row draw is atypical, STOP (registered)")
        ledger.append_row("w4_e1floor", {"seeds": SEEDS}, "failed",
                          metrics={"L0_RAW": a0},
                          notes="AMENDMENT 36 validity gate failed: raw corpus "
                                "human rows read above 0.5250 against the anchor, "
                                "ladder void, no step costs reported (registered).",
                          tier=1)
        ledger.regenerate_leaderboard()
        sys.exit(2)

    res = {"seeds": SEEDS, "gate": a0, "pooled": {}, "per_seed": {}}
    print("\n  READ 1 (PRIMARY), pooled ladder:", flush=True)
    for a in ARMS:
        v, nn = auc(pooled[a])
        res["pooled"][a] = v
        print(f"  {a:>11} {v:.4f}  n {nn}  cost vs L0 {v - a0:+.4f}", flush=True)
    total = res["pooled"]["L3_FULL"] - a0
    steps = {"dt": res["pooled"]["L1_DT"] - a0,
             "polar": res["pooled"]["L2_POLAR"] - a0,
             "full": total,
             "snap": res["pooled"]["L3_FULL"] - res["pooled"]["L3B_NOSNAP"]}
    res["steps"] = steps
    for k, v in steps.items():
        share = (v / total * 100) if abs(total) > 1e-9 else float("nan")
        tag = "MATERIAL" if abs(v) >= 0.010 else "not material"
        dom = "  DOMINANT" if (k in ("dt", "polar") and total > 0
                               and v / total >= 0.60) else ""
        print(f"  step {k:>6}: {v:+.4f}  {share:5.1f}% of total  {tag}{dom}")

    print("\n  READ 2, per seed at n 2000, shapes only:", flush=True)
    for s in SEEDS:
        row = {}
        for a in ARMS:
            row[a] = auc(per_seed[s][a])[0]
        res["per_seed"][s] = row
        print("  seed %d: " % s + "  ".join(f"{a} {row[a]:.4f}" for a in ARMS), flush=True)
    for k, (hi, lo) in (("dt", ("L1_DT", "L0_RAW")), ("polar", ("L2_POLAR", "L0_RAW")),
                        ("full", ("L3_FULL", "L0_RAW")),
                        ("snap", ("L3_FULL", "L3B_NOSNAP"))):
        d = np.array([res["per_seed"][s][hi] - res["per_seed"][s][lo] for s in SEEDS])
        se = d.std(ddof=1) / np.sqrt(len(d))
        res.setdefault("per_seed_steps", {})[k] = dict(
            mean=float(d.mean()), se=float(se),
            t=float(d.mean() / se) if se > 0 else float("inf"))
        print(f"  step {k:>6}: mean {d.mean():+.4f} se {se:.4f} "
              f"t {d.mean() / se if se > 0 else float('inf'):+.2f}")

    print("\n  READ 3 (informational), normalized Wasserstein vs L0, top 6 each:")
    base = pooled["L0_RAW"][np.isfinite(pooled["L0_RAW"]).all(1)]
    res["read3"] = {}
    for a in ARMS[1:]:
        m = pooled[a][np.isfinite(pooled[a]).all(1)]
        nw = feat.normalized_wasserstein_by_feature(base, m)
        res["read3"][a] = dict(zip(FEATURE_NAMES, [float(x) for x in nw]))
        top = sorted(zip(FEATURE_NAMES, nw), key=lambda kv: -kv[1])[:6]
        print(f"  {a:>11}: " + "  ".join(f"{k} {v:.3f}" for k, v in top))

    with open("research/w4_e1floor.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1floor.json")
    print("  diagnostic only, never a training signal, no selection, no headline")

    rid = ledger.append_row(
        "w4_e1floor",
        {"seeds": SEEDS, "n": N, "arms": ARMS, "shuf_seed": SHUF_SEED,
         "paired_against": ["w4_e1feat_2026-08-25T202633+0000_1e8ddc20"]},
        "ok",
        metrics={**{f"auc_{a}": res["pooled"][a] for a in ARMS},
                 **{f"step_{k}": v for k, v in steps.items()}},
        artifacts=["research/w4_e1floor.json"],
        notes=f"AMENDMENT 36 representation floor ladder, human only. L0 {a0:.4f}"
              f" L3 {res['pooled']['L3_FULL']:.4f} total {total:+.4f}."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
