"""Does telling the AR trunk what kind of movement to make change what it makes.

Executes the registration in HANDOFF.md, "Feature conditioning the event AR
trunk", which fixes three readouts and three branches before any number exists.

  1. The contract. Feature conditioned model minus base model, mean across
     seeds, on identical specs and identical duration draws.
  2. Obedience. Spearman correlation between commanded and realised, per
     feature, averaged over the eighteen. The flow family's number is 0.41.
  3. The control. The same model with the flag zeroed. It must land inside the
     replicate noise floor of the base model, or readout 1 is confounded by
     whatever else the fine tune moved and no branch may be claimed.

The command is drawn from the empirical joint CONDITIONAL on the spec: nearest
neighbours in (log distance, log duration) among labelled training rows, one
picked uniformly from the k nearest. Drawing from the unconditional joint would
hand the model a movement_duration that contradicts its own commanded duration.

One trajectory per spec. Nothing generated twice, nothing selected, no candidate
pool anywhere in this file.

Safety. Scores through research/autoloop/scoring.py only. Never reads
data/human_eval_features.npy, never modifies scoring code, never writes any
checkpoint. Paces itself on GPU temperature: this machine crashed on this
workload on 2026-08-06, kill line tightened to 79C.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python -u research/w4_featcond.py --seeds 3
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
from features import FEATURE_NAMES, extract_features, resample_trajectory  # noqa: E402
from models.event_ar import EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS  # noqa: E402
from training.train_event_ar_featcond import COND_DIM, N_FEAT, to_gauss  # noqa: E402
from w4_latent import cooldown, gpu_c, throttle  # noqa: E402
from w4_paired import set_arm, specs_for  # noqa: E402

BANK_N = 400_000


def load_bank(rng):
    """Labelled human rows to draw commands from, with their own geometry so the
    draw can be made conditional on the spec."""
    feat = np.load("training/events_feat18.npy", mmap_mode="r")
    ok = np.load("training/events_feat18_ok.npy")
    cond = np.load("training/events_cond.npy", mmap_mode="r")
    usable = np.flatnonzero(ok)
    pick = np.sort(rng.choice(usable, min(BANK_N, len(usable)), replace=False))
    return np.asarray(feat[pick]), np.asarray(cond[pick])[:, :2]


def draw_commands(bank_f, bank_g, rows, k, rng):
    """One human feature vector per spec, from the k nearest in (log distance,
    log duration). Both axes are standardised first so neither dominates the
    distance."""
    q = np.asarray([[r[0], r[1]] for r in rows], dtype=np.float64)
    mu, sd = bank_g.mean(0), bank_g.std(0)
    B = (bank_g - mu) / sd
    Q = (q - mu) / sd
    from scipy.spatial import cKDTree
    _, nn = cKDTree(B).query(Q, k=k, workers=-1)
    choice = nn[np.arange(len(Q)), rng.integers(0, k, size=len(Q))]
    return bank_f[choice]


@torch.no_grad()
def generate(model, cond_rows, meta, batch, dev, rngseed):
    """Decoded paths and the index of the spec each one came from. The index is
    what makes an obedience number possible: a dropped path must not silently
    shift the alignment between commanded and realised."""
    torch.manual_seed(rngseed)
    paths, idx = [], []
    for c0 in range(0, len(cond_rows), batch):
        throttle()
        cond = torch.tensor(cond_rows[c0:c0 + batch], dtype=torch.float32,
                            device=dev)
        s_cls, th_cls, dt_cls = model.sample(cond, temperature=1.0)
        pad = (s_cls >= S_PAD_CLASS).cpu().numpy()
        dt_ms = class_to_dt_ms(dt_cls)
        dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
                / esp._DT_STD).cpu().numpy()
        s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
        for j in range(s_np.shape[0]):
            sx, sy, ang = meta[c0 + j]
            p = esp._decode(dt_z[j], s_np[j], th_np[j], sx, sy, ang)
            L = int(pad[j].argmax()) if pad[j].any() else s_np.shape[1]
            if p is None or L < 2:
                continue
            paths.append(np.asarray(p, dtype=np.float64))
            idx.append(c0 + j)
    return paths, np.asarray(idx, dtype=np.int64)


def featurise(paths, idx):
    F, keep = [], []
    for p, i in zip(paths, idx):
        f = extract_features(resample_trajectory(p, hz=125.0))
        if f is not None and np.all(np.isfinite(f)):
            F.append(f)
            keep.append(i)
    return np.asarray(F, dtype=np.float64), np.asarray(keep, dtype=np.int64)


def spearman_by_feature(cmd, got):
    """Rank correlation, one number per feature. Rank rather than Pearson
    because curvature_mean spans five orders of magnitude and a single outlying
    pair would otherwise decide the answer."""
    out = np.empty(cmd.shape[1])
    for j in range(cmd.shape[1]):
        a = np.argsort(np.argsort(cmd[:, j])).astype(np.float64)
        b = np.argsort(np.argsort(got[:, j])).astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        out[j] = float((a * b).sum() / d) if d > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="training/event_ar_v2_s40000.pt")
    ap.add_argument("--fc", default="training/event_ar_fc.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=20)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default="research/w4_featcond.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    set_arm(1.0, True, 2.5)

    bck = torch.load(args.base, map_location=dev, weights_only=False)
    base = EventARModel(**bck["config"]).to(dev).eval()
    base.load_state_dict(bck["model_state_dict"])

    fck = torch.load(args.fc, map_location=dev, weights_only=False)
    assert fck["config"]["cond_dim"] == COND_DIM, fck["config"]
    fc = EventARModel(**fck["config"]).to(dev).eval()
    fc.load_state_dict(fck["model_state_dict"])
    knots = fck["feat_knots"]
    print(f"  base {args.base}  fc {args.fc} at step {fck['step']}\n", flush=True)

    bank_f, bank_g = load_bank(np.random.default_rng(4242))
    print(f"  command bank {len(bank_f):,} labelled human rows", flush=True)

    print(f"  {'seed':>5}{'arm':>10}{'contract':>10}{'collapse':>10}{'n':>7}"
          f"{'obedience':>11}{'gpu':>6}")
    res = {a: [] for a in ("base", "fc", "fc_off")}
    obed = []
    for sd in range(args.seed0, args.seed0 + args.seeds):
        rows, meta = specs_for(args.n, args.seed, sd)
        rng = np.random.default_rng(9000 + sd)
        cmd = draw_commands(bank_f, bank_g, rows, args.k, rng)
        z = to_gauss(cmd, knots)

        four = np.asarray(rows, dtype=np.float32)
        wide_on = np.zeros((len(rows), COND_DIM), dtype=np.float32)
        wide_on[:, :4] = four
        wide_on[:, 4:4 + N_FEAT] = z
        wide_on[:, -1] = 1.0
        wide_off = np.zeros_like(wide_on)
        wide_off[:, :4] = four

        for arm, model, cr in (("base", base, four),
                               ("fc", fc, wide_on),
                               ("fc_off", fc, wide_off)):
            cooldown()
            F, keep = featurise(*generate(model, cr, meta, args.batch, dev, sd))
            r = scoring.score_features(F)
            res[arm].append(float(r["auc_rf_oob"]))
            ob = ""
            if arm == "fc":
                s = spearman_by_feature(cmd[keep], F)
                obed.append(s)
                ob = f"{s.mean():.3f}"
            print(f"  {sd:>5}{arm:>10}{r['auc_rf_oob']:>10.4f}"
                  f"{len(r['collapse_features']):>10}{len(F):>7}{ob:>11}"
                  f"{gpu_c():>6}", flush=True)

    b = np.array(res["base"])
    f = np.array(res["fc"])
    o = np.array(res["fc_off"])
    d = f - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    S = np.mean(obed, axis=0)
    print(f"\n  base   {b.mean():.4f}  fc {f.mean():.4f}  fc_off {o.mean():.4f}")
    print(f"  readout 1  {d.mean():+.4f}, se {se:.4f}, "
          f"{int((d < 0).sum())}/{len(d)} seeds improved")
    print(f"  readout 3  fc_off minus base {o.mean() - b.mean():+.4f}")
    print(f"\n  readout 2, commanded to realised Spearman")
    for j in np.argsort(-S):
        print(f"    {FEATURE_NAMES[j]:<26}{S[j]:>8.3f}")
    print(f"    {'MEAN':<26}{S.mean():>8.3f}   flow family 0.41")

    confounded = abs(o.mean() - b.mean()) > 0.0069
    if confounded:
        verdict = (f"CONFOUNDED. The flag zero control reads "
                   f"{o.mean() - b.mean():+.4f} against the base model, outside "
                   "the 0.0069 replicate floor, so the fine tune moved the model "
                   "for reasons unrelated to conditioning and no branch may be "
                   "claimed.")
    elif S.mean() <= 0.70:
        verdict = (f"DOES NOT OBEY. Mean commanded to realised Spearman "
                   f"{S.mean():.3f} against the 0.70 bar. The steering failure "
                   "is a property of feature conditioning in this repo and not "
                   "of the flow model's renderer, and the family is closed.")
    elif d.mean() < -0.030:
        verdict = (f"WORKS. Obedience {S.mean():.3f} and the contract improves "
                   f"{-d.mean():.4f}, more than the registered 0.030.")
    else:
        verdict = (f"OBEYS BUT DOES NOT PAY. Obedience {S.mean():.3f} clears "
                   f"0.70 and the contract moves only {d.mean():+.4f}. Per "
                   "feature obedience is not joint obedience, and the forest "
                   "lives in the joint.")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"base": res["base"], "fc": res["fc"], "fc_off": res["fc_off"],
               "delta_mean": float(d.mean()), "delta_se": float(se),
               "obedience": {FEATURE_NAMES[j]: float(S[j])
                             for j in range(len(FEATURE_NAMES))},
               "obedience_mean": float(S.mean()), "k": args.k, "n": args.n,
               "fc_step": int(fck["step"]), "verdict": verdict},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
