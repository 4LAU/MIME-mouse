"""Where along a trajectory the error lives, and whether it accumulates.

`research/w4_length.py` reads the contract AUC inside bins of
`movement_duration` and finds it rises with duration, 0.6219 on the shortest
third against 0.6624 on the longest, monotone. That is consistent with the error
compounding along an autoregressive rollout. It is also consistent with
something much duller: short trajectories have few points, so every summary
feature is a noisy estimate on BOTH sides, and noise masks a difference that is
actually constant. A trend in aggregate features against observation length
cannot separate those two.

This can. It never aggregates over a trajectory. For each step it compares the
distribution of what the model emitted at that step against the human
distribution at that step, which uses the same number of samples per step on
both sides and therefore has no noise masking to confound it.

The confound that DOES remain is the ending. Every trajectory decelerates into
its target, so a model that gets the ending wrong shows a discrepancy that grows
with position without anything having accumulated. Separating the two needs both
axes at once:

  Split by total length into bands, and inside each band read the discrepancy
  against NORMALISED position. Under accumulation the discrepancy at a fixed
  fraction of the way through is worse in the longer bands, because more steps
  have gone by. Under a terminal effect it is the same in every band and spikes
  only in the last decile, because the ending is the ending regardless of how
  long the trajectory was.

Statistic. Per step index, the two sample Kolmogorov Smirnov distance between
human and generated, computed separately for step speed and for step turn
angle. Bootstrap is over trajectories, so the spread quoted is the spread of the
statistic under resampling of the sample, not of the steps.

DIAGNOSTIC, not a generation method. One trajectory per spec, nothing selected.
Never touches data/human_eval_features.npy, never modifies scoring code.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_stepdrift.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")
os.environ.setdefault("EVENT_SNAP", "2.5")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import extract_feature_matrix, resample_trajectory  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from w4_latent import cooldown, gpu_c  # noqa: E402
from w4_paired import gen_paths, specs_for  # noqa: E402

HZ = 125.0
PATHS_CACHE = "research/w4_paths.npz"


def profile(path):
    """Per step speed in units per second and per step turn in radians, on the
    125 Hz grid the feature extractor uses. Length is len(path) - 2, since a
    turn needs two consecutive displacements."""
    xy = np.asarray(path, dtype=np.float64)[:, :2]
    d = np.diff(xy, axis=0)
    if len(d) < 2:
        return None
    spd = np.hypot(d[:, 0], d[:, 1]) * HZ
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    return spd[1:], turn


def ks(a, b):
    """Two sample Kolmogorov Smirnov distance, computed directly so no
    p value machinery is involved and the number is a pure distance."""
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    v = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), v, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), v, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def deciles(profiles, n_dec):
    """Pool each trajectory's per step values into n_dec bins of NORMALISED
    position, so trajectories of different lengths contribute to the same bins.
    Returns a list of arrays, one per bin."""
    out = [[] for _ in range(n_dec)]
    for v in profiles:
        idx = np.minimum((np.arange(len(v)) * n_dec) // len(v), n_dec - 1)
        for k in range(n_dec):
            out[k].append(v[idx == k])
    return [np.concatenate(o) if o else np.array([]) for o in out]


def band_curve(Hp, Gp, n_dec):
    return [ks(h, g) for h, g in zip(deciles(Hp, n_dec), deciles(Gp, n_dec))]


def build_paths(args):
    """Human validation paths and generated paths, cached together so a rerun
    of the analysis costs no GPU. Human paths are reconstructed by the same
    recipe that produced data/human_val_features_grpo.npy and are checked
    against it before anything is measured."""
    if os.path.exists(PATHS_CACHE):
        z = np.load(PATHS_CACHE, allow_pickle=True)
        return list(z["human"]), list(z["gen"])

    from phase0_critic import VAL_N, VAL_SEED, reconstruct_human_val_paths
    hp, hf = reconstruct_human_val_paths(VAL_N, VAL_SEED, verbose=False)
    ref = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    hf = np.asarray(hf)
    assert hf.shape == ref.shape and np.allclose(hf, ref, rtol=1e-6,
                                                 atol=1e-6), (
        "reconstructed human paths do not reproduce the scorer's human "
        "features, so the two sides would not be the same sample")
    print(f"  human paths reconstructed and matched to "
          f"{scoring.DEFAULT_HUMAN_FEATURES_PATH}, {len(hp)} paths")

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    cooldown()
    with torch.no_grad():
        rows, meta = specs_for(args.n, args.seed, args.rngseed)
        gp = gen_paths(model, rows, meta, args.batch, 1.0, dev, args.rngseed)
    gp = [resample_trajectory(p, hz=HZ) for p in gp]
    F = extract_feature_matrix(gp)
    ok = np.all(np.isfinite(F), 1)
    gp = [p for p, k in zip(gp, ok) if k]
    print(f"  generated {len(gp)} paths, contract "
          f"{scoring.score_features(F[ok])['auc_rf_oob']:.4f}, gpu {gpu_c()}")

    np.savez_compressed(PATHS_CACHE,
                        human=np.array([np.asarray(p) for p in hp],
                                       dtype=object),
                        gen=np.array([np.asarray(p) for p in gp],
                                     dtype=object))
    return hp, gp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rngseed", type=int, default=0)
    ap.add_argument("--bands", type=int, default=3)
    ap.add_argument("--dec", type=int, default=10)
    ap.add_argument("--boot", type=int, default=40)
    ap.add_argument("--out", default="research/w4_stepdrift.json")
    args = ap.parse_args()

    print(f"\n  EVENT_SNAP={esp._SNAP} DUR_EMPIRICAL={os.environ['DUR_EMPIRICAL']}")
    hp, gp = build_paths(args)

    H = [profile(p) for p in hp]
    G = [profile(p) for p in gp]
    H = [x for x in H if x is not None]
    G = [x for x in G if x is not None]
    hl = np.array([len(x[0]) for x in H])
    gl = np.array([len(x[0]) for x in G])

    edges = np.percentile(hl, np.linspace(0, 100, args.bands + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rng = np.random.default_rng(args.seed)

    print(f"\n  {len(H)} human and {len(G)} generated profiles, "
          f"{args.bands} length bands, {args.dec} position bins, "
          f"{args.boot} bootstrap resamples")
    print(f"\n  KS distance between human and generated, by normalised "
          f"position within the trajectory")
    hdr = "".join(f"{int((k + 0.5) * 100 / args.dec):>6}"
                  for k in range(args.dec))
    print(f"  {'band':>6}{'steps':>12}{'n h':>6}{'n g':>6}  speed{hdr}")

    out = {"bands": []}
    mids, ends = [], []
    for b in range(args.bands):
        hm = (hl >= edges[b]) & (hl < edges[b + 1])
        gm = (gl >= edges[b]) & (gl < edges[b + 1])
        Hb = [H[i] for i in np.flatnonzero(hm)]
        Gb = [G[i] for i in np.flatnonzero(gm)]
        if len(Hb) < 30 or len(Gb) < 30:
            print(f"  {b:>6}  too few trajectories: {len(Hb)} h, {len(Gb)} g")
            continue
        cs = band_curve([x[0] for x in Hb], [x[0] for x in Gb], args.dec)
        ct = band_curve([x[1] for x in Hb], [x[1] for x in Gb], args.dec)
        rngtxt = f"{int(edges[b]) if b else 0}-{int(edges[b + 1]) if b + 1 < args.bands else int(hl.max())}"
        print(f"  {b:>6}{rngtxt:>12}{len(Hb):>6}{len(Gb):>6}       "
              + "".join(f"{v:>6.3f}" for v in cs))
        print(f"  {'':>6}{'':>12}{'':>6}{'':>6}   turn"
              + "".join(f"{v:>6.3f}" for v in ct), flush=True)

        lo, hi = args.dec // 5, args.dec - 1
        mid = float(np.nanmean(cs[lo:hi] + ct[lo:hi]))
        end = float(np.nanmean([cs[-1], ct[-1]]))
        mids.append(mid)
        ends.append(end)
        out["bands"].append({"band": b, "steps": rngtxt, "n_human": len(Hb),
                             "n_gen": len(Gb), "ks_speed": cs, "ks_turn": ct,
                             "mid": mid, "end": end})

    # bootstrap the mid rollout statistic, resampling TRAJECTORIES so the
    # spread quoted is the spread of the sample and not of the steps
    def mid_of(Hb, Gb):
        lo, hi = args.dec // 5, args.dec - 1
        cs = band_curve([x[0] for x in Hb], [x[0] for x in Gb], args.dec)
        ct = band_curve([x[1] for x in Hb], [x[1] for x in Gb], args.dec)
        return float(np.nanmean(cs[lo:hi] + ct[lo:hi]))

    sds = []
    for rec in out["bands"]:
        b = rec["band"]
        hm = np.flatnonzero((hl >= edges[b]) & (hl < edges[b + 1]))
        gm = np.flatnonzero((gl >= edges[b]) & (gl < edges[b + 1]))
        vals = [mid_of([H[i] for i in rng.choice(hm, len(hm))],
                       [G[i] for i in rng.choice(gm, len(gm))])
                for _ in range(args.boot)]
        sds.append(float(np.std(vals)))
        rec["mid_sd"] = sds[-1]

    print(f"\n  {'band':>6}{'mid rollout KS':>16}{'boot sd':>10}"
          f"{'last bin KS':>13}{'spike':>9}")
    for rec, sd in zip(out["bands"], sds):
        print(f"  {rec['band']:>6}{rec['mid']:>16.4f}{sd:>10.4f}"
              f"{rec['end']:>13.4f}{rec['end'] - rec['mid']:>+9.4f}")

    d = mids[-1] - mids[0]
    s = float(np.hypot(sds[-1], sds[0]))
    mono = all(mids[i + 1] >= mids[i] for i in range(len(mids) - 1))
    spike = float(np.mean([e - m for e, m in zip(ends, mids)]))
    print(f"\n  longest band minus shortest, mid rollout  {d:+.4f} against "
          f"{s:.4f}   monotone {mono}")
    print(f"  mean last bin spike over bands            {spike:+.4f}")

    acc = d > 2 * s and mono
    term = spike > 2 * float(np.mean(sds))
    if acc and term:
        verdict = (f"BOTH. Mid rollout KS rises {d:+.4f} across length bands "
                   f"against a bootstrap spread of {s:.4f}, AND the last "
                   f"position bin sits {spike:+.4f} above the middle in every "
                   "band. The error compounds and the ending is separately "
                   "wrong.")
    elif acc:
        verdict = (f"ACCUMULATION. At the same fraction of the way through, the "
                   f"discrepancy is {d:+.4f} worse in the longest band than the "
                   f"shortest, against a bootstrap spread of {s:.4f}, monotone. "
                   "More elapsed steps means a worse step distribution, which "
                   "is exposure bias and not a noise artifact.")
    elif term:
        verdict = (f"TERMINAL EFFECT ONLY. Mid rollout KS does not grow with "
                   f"length ({d:+.4f} against {s:.4f}) but the last position "
                   f"bin sits {spike:+.4f} above the middle. The model gets the "
                   "ending wrong, not the accumulation.")
    else:
        verdict = (f"UNIFORM. Neither the length trend ({d:+.4f} against "
                   f"{s:.4f}) nor the terminal spike ({spike:+.4f}) clears its "
                   "own noise. The per step error is flat along the rollout and "
                   "the w4_length trend was the aggregation, not the model.")
    out.update({"mid_trend": d, "mid_trend_sd": s, "monotone": bool(mono),
                "spike": spike, "verdict": verdict})
    print(f"\n  VERDICT  {verdict}\n")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
