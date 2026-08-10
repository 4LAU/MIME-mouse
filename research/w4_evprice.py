"""What removing the event count defect would buy.

PRE REGISTERED in HANDOFF.md 2026-08-06, "## What removing the event count
defect would buy". The placebo, the branch thresholds and the prediction of
PARTIAL around minus 0.02 to minus 0.03 were all fixed before this file existed.

DIAGNOSTIC ONLY. Resampling the generated set is SELECTION and is DISQUALIFIED
as a generation method under the mandate. The matched arm's AUC is never a
result the model achieved. The only reportable quantity is matched minus the
same size RANDOM arm, which controls for the sample size dependence of the
random forest out of bag AUC.

Safety. Scores through research/autoloop/scoring.py only. Touches no evaluation
data directly, never modifies scoring code, never
training/candi_polar_flow_best.pt.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_evprice.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

os.environ.setdefault("EVENT_CHOICE_TEMP", "10")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import extract_feature_matrix  # noqa: E402
from models.event_ar import EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS  # noqa: E402
from phase_a_baseline import make_specs  # noqa: E402

NDRAW = 20
NSUB = 1000
NBIN = 20


def build_specs(n, seed):
    rows, meta = [], []
    for sx, sy, ex, ey in make_specs(n, seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ld = math.log(dist)
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([ld, math.log(esp._duration.sample(ld)),
                     math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ang))
    return rows, meta


def generate(model, rows, meta, batch, temp, dev):
    """One trajectory per spec, no selection, the served path unchanged."""
    paths, nev, cnd = [], [], []
    for c0 in range(0, len(rows), batch):
        cond = torch.tensor(rows[c0:c0 + batch], dtype=torch.float32, device=dev)
        s_cls, th_cls, dt_cls = model.sample(cond, temperature=temp)
        pad = (s_cls >= S_PAD_CLASS).cpu().numpy()
        dt_ms = class_to_dt_ms(dt_cls)
        dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
                / esp._DT_STD).cpu().numpy()
        s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
        for j in range(s_np.shape[0]):
            sx, sy, ang = meta[c0 + j]
            p = esp._decode(dt_z[j], s_np[j], th_np[j], sx, sy, ang)
            if p is None:
                continue
            L = int(pad[j].argmax()) if pad[j].any() else s_np.shape[1]
            if L < 2:
                continue
            paths.append(np.asarray(p, dtype=np.float64))
            nev.append(L)
            cnd.append(rows[c0 + j])
        print(f"    {min(c0 + batch, len(rows))} / {len(rows)}", flush=True)
    return paths, np.array(nev, float), np.array(cnd, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--streams", default="research/w4_typpos_streams.npz")
    ap.add_argument("--fit", default="research/w4_evcount.json")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--out", default="research/w4_evprice.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    cache = args.out.replace(".json", "_cache.npz")
    if os.path.exists(cache) and not args.regen:
        c = np.load(cache)
        F, nev, cnd = c["F"], c["nev"], c["cnd"]
        print(f"\n  reusing {cache}, {len(F)} movements, no generation")
    else:
        rows, meta = build_specs(args.n, args.seed)
        print(f"\n  generating {len(rows)} at temperature {args.temp}, "
              "one per spec, no selection", flush=True)
        with torch.no_grad():
            paths, nev, cnd = generate(model, rows, meta, args.batch,
                                       args.temp, dev)
        F = extract_feature_matrix(paths)
        fin = np.all(np.isfinite(F), 1)
        F, nev, cnd = F[fin], nev[fin], cnd[fin]
        np.savez(cache, F=F, nev=nev, cnd=cnd)
    base = float(scoring.score_features(F)["auc_rf_oob"])
    print(f"\n  baseline contract AUC {base:.4f}  on {len(F)} movements\n")

    # the human conditional, fitted on real held out event counts in w4_evcount
    br = np.array(json.load(open(args.fit))["human_fit"], float)
    z = np.load(args.streams)
    hc, hL = z["cond"].astype(float), z["real_L"].astype(float)
    hres = np.log(np.maximum(hL, 1)) - (br[0] + br[1] * hc[:, 0]
                                        + br[2] * hc[:, 1])
    gres = np.log(np.maximum(nev, 1)) - (br[0] + br[1] * cnd[:, 0]
                                         + br[2] * cnd[:, 1])
    print(f"  event count residual against the human conditional")
    print(f"    human      mean {hres.mean():+.4f}  sd {hres.std():.4f}")
    print(f"    generated  mean {gres.mean():+.4f}  sd {gres.std():.4f}\n")

    edges = np.unique(np.quantile(hres, np.linspace(0, 1, NBIN + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    hb = np.digitize(hres, edges[1:-1])
    gb = np.digitize(gres, edges[1:-1])
    nb = len(edges) - 1
    tgt = np.bincount(hb, minlength=nb).astype(float)
    pools = [np.flatnonzero(gb == b) for b in range(nb)]
    live = np.array([len(p) > 0 for p in pools])
    tgt = np.where(live, tgt, 0.0)
    tgt = tgt / tgt.sum()
    miss = 1.0 - float(np.bincount(hb, minlength=nb)[live].sum()
                       / len(hres))
    print(f"  target bins with no generated movement to draw from: "
          f"{miss:.2%} of human mass\n")

    # WITHOUT REPLACEMENT, both arms. The first version of this script drew with
    # replacement and was invalid: duplicated rows break the random forest's out
    # of bag estimate, a duplicate of an in bag row lands in the out of bag set
    # and is classified for free. The matched arm concentrates on fewer unique
    # movements so it duplicates more, and the whole matched minus random signal
    # was that difference in duplicate rate. The registered placebo controlled
    # for sample size and not for duplicate rate, so it did not catch it.
    # Thinning: accept each generated movement with probability proportional to
    # target share over generated share, capped at one. No duplicates, and the
    # retained sample is matched in distribution by construction.
    gshare = np.array([len(p) for p in pools], float) / len(F)
    acc = np.divide(tgt, gshare, out=np.zeros(nb), where=gshare > 0)
    acc = acc / acc.max()
    print(f"  thinning acceptance by bin, min {acc[gshare > 0].min():.3f} "
          f"max {acc.max():.3f}, expected retained {int(acc[gb].sum())} "
          f"of {len(F)}")

    rng = np.random.default_rng(args.seed + 7)
    mm, rr, ns = [], [], []
    for d in range(NDRAW):
        keep = np.flatnonzero(rng.random(len(F)) < acc[gb])
        n_k = len(keep)
        if n_k < 200:
            print(f"    draw {d + 1:>2}   retained {n_k}, too few, skipped")
            continue
        ridx = rng.choice(len(F), n_k, replace=False)
        mm.append(float(scoring.score_features(F[keep])["auc_rf_oob"]))
        rr.append(float(scoring.score_features(F[ridx])["auc_rf_oob"]))
        ns.append(n_k)
        print(f"    draw {d + 1:>2}   n {n_k:>4}   matched {mm[-1]:.4f}   "
              f"random {rr[-1]:.4f}   diff {mm[-1] - rr[-1]:+.4f}", flush=True)

    mm, rr = np.array(mm), np.array(rr)
    diff = mm - rr
    se = float(diff.std(ddof=1) / np.sqrt(NDRAW))
    print(f"\n  matched  {mm.mean():.4f}  se {mm.std(ddof=1) / np.sqrt(NDRAW):.4f}")
    print(f"  random   {rr.mean():.4f}  se {rr.std(ddof=1) / np.sqrt(NDRAW):.4f}")
    print(f"  MATCHED MINUS RANDOM  {diff.mean():+.4f}  se {se:.4f}"
          f"   {diff.mean() / se:+.2f} sigma")

    d = float(diff.mean())
    if d <= -0.04:
        verdict = (f"WORTH IT. {d:+.4f}. Build the training change.")
    elif d > -0.015:
        verdict = ("NOT WORTH IT. Event count is detectable but removing it "
                   "does not move the contract. The lower bound was loose. Go "
                   "to the roughness half of the history result.")
    else:
        verdict = f"PARTIAL. {d:+.4f}. Report, claim neither."
    print(f"\n  VERDICT  {verdict}")
    print("  the matched arm is a DIAGNOSTIC, it is selection and is "
          "DISQUALIFIED as a method\n")

    json.dump({"baseline": base, "n": int(len(F)),
               "matched": mm.tolist(), "random": rr.tolist(),
               "n_per_draw": [int(x) for x in ns],
               "matched_mean": float(mm.mean()), "random_mean": float(rr.mean()),
               "diff": d, "se": se, "verdict": verdict,
               "human_res_sd": float(hres.std()),
               "gen_res_sd": float(gres.std()),
               "unreachable_human_mass": miss},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}\n")


if __name__ == "__main__":
    main()
