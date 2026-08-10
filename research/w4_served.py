"""The same generator, under the duration prior this repo actually serves.

PRE REGISTERED in HANDOFF.md 2026-08-07, "## The duration prior the w4
diagnostics ran under was not the served one". The branch thresholds, the
prediction of MATERIAL and the collapse caveat were all fixed before this file
existed.

research/w4_evprice.py sets EVENT_CHOICE_TEMP and nothing else, so the cache
every w4 diagnostic reads was built at std_mult 0.7 Gaussian. Every serving path
in this repo sets EVENT_DUR_STD=1.0 DUR_EMPIRICAL=1. research/w4_durprior.py
priced that on the duration marginal: 0.769 of the human spread against 0.997.

This regenerates on IDENTICAL specs and seed with the served prior and changes
nothing else, so the difference is attributable to the prior alone.

One trajectory per spec. Nothing generated twice, nothing selected, nothing read
that serving would not have.

Safety. Scores through research/autoloop/scoring.py only. Never modifies scoring
code, never training/candi_polar_flow_best.pt. Paces itself on GPU temperature:
this machine crashed on this workload on 2026-08-06, kill line tightened to 79C.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_served.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# these three are read when experiments.event_stream_polar is imported, so they
# have to be set before the import below, not before the call to sample().
os.environ.setdefault("EVENT_CHOICE_TEMP", "10")
os.environ.setdefault("EVENT_DUR_STD", "1.0")
os.environ.setdefault("DUR_EMPIRICAL", "1")

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import FEATURE_NAMES, extract_feature_matrix  # noqa: E402
from models.event_ar import EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS  # noqa: E402
from w4_copula import normal_scores  # noqa: E402
from w4_evprice import build_specs  # noqa: E402
from w4_latent import cooldown, gpu_c, throttle  # noqa: E402

WOBBLE = [10, 11, 12, 13, 16, 17]


def gen(model, rows, meta, batch, temp, dev):
    """One trajectory per spec, no tilts, no selection. The served path."""
    paths = []
    for c0 in range(0, len(rows), batch):
        throttle()
        blk = rows[c0:c0 + batch]
        cond = torch.tensor(blk, dtype=torch.float32, device=dev)
        s_cls, th_cls, dt_cls = model.sample(cond, temperature=temp)
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
        print(f"    {len(paths)} paths, gpu {gpu_c()}C", flush=True)
    F = extract_feature_matrix(paths)
    return F[np.all(np.isfinite(F), 1)]


def gap(G, H):
    """Leading eigenvector of the human minus generated normal score
    correlation gap, and how much of the gap's total mass it carries."""
    D = (np.corrcoef(normal_scores(H), rowvar=False)
         - np.corrcoef(normal_scores(G), rowvar=False))
    w, V = np.linalg.eigh(D)
    i = np.argmax(np.abs(w))
    off = ~np.eye(len(D), dtype=bool)
    return V[:, i], float(np.abs(w[i])), float(np.abs(D)[off].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rngseed", type=int, default=0)
    ap.add_argument("--old", default="research/w4_evprice_cache.npz")
    ap.add_argument("--cache", default="research/w4_served_cache.npz")
    ap.add_argument("--out", default="research/w4_served.json")
    args = ap.parse_args()

    # esp reads all four at import, so this is a record of what the run
    # actually used, not a request for it.
    print(f"\n  EVENT_DUR_STD={os.environ['EVENT_DUR_STD']}  "
          f"DUR_EMPIRICAL={os.environ['DUR_EMPIRICAL']}  "
          f"EVENT_CHOICE_TEMP={os.environ['EVENT_CHOICE_TEMP']}  "
          f"EVENT_SNAP={esp._SNAP}")

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    # DurationModel.__init__ ends with an UNSEEDED default_rng and the sampling
    # loop uses the global torch RNG, so two runs of the same command differ by
    # about 0.008 of contract AUC. Seeding both here makes an arm to arm
    # comparison paired: same specs, same durations, same sampling noise, so the
    # only difference left is the flag under test. Must happen before
    # build_specs, which is where the durations are drawn.
    esp._duration._rng = np.random.default_rng(args.rngseed)
    torch.manual_seed(args.rngseed)

    rows, meta = build_specs(args.n, args.seed)
    dur = np.array([r[1] for r in rows])
    print(f"  {len(rows)} specs, commanded log duration sd {dur.std():.4f}, "
          f"rngseed {args.rngseed}")

    cooldown()
    with torch.no_grad():
        G = gen(model, rows, meta, args.batch, args.temp, dev)
    np.savez_compressed(args.cache, F=G)

    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    Gold = np.load(args.old)["F"]
    r, rold = scoring.score_features(G), scoring.score_features(Gold)
    v, lead, mean_d = gap(G, H)
    vo, lead_o, mean_do = gap(Gold, H)

    print(f"\n  {'':<28}{'w4 cache':>12}{'served':>12}{'delta':>10}")
    print(f"  {'CONTRACT AUC':<28}{rold['auc_rf_oob']:>12.4f}"
          f"{r['auc_rf_oob']:>12.4f}"
          f"{r['auc_rf_oob'] - rold['auc_rf_oob']:>+10.4f}")
    print(f"  {'collapsed features':<28}{len(rold['collapse_features']):>12}"
          f"{len(r['collapse_features']):>12}"
          f"{len(r['collapse_features']) - len(rold['collapse_features']):>+10}")
    print(f"  {'mean |corr gap|':<28}{mean_do:>12.4f}{mean_d:>12.4f}"
          f"{mean_d - mean_do:>+10.4f}")
    print(f"  {'leading gap eigenvalue':<28}{lead_o:>12.4f}{lead:>12.4f}"
          f"{lead - lead_o:>+10.4f}")
    print(f"  {'log duration sd':<28}{np.log(Gold[:, 14]).std():>12.4f}"
          f"{np.log(G[:, 14]).std():>12.4f}"
          f"{np.log(G[:, 14]).std() - np.log(Gold[:, 14]).std():>+10.4f}")
    print(f"  {'human log duration sd':<28}{'':>12}"
          f"{np.log(H[:, 14]).std():>12.4f}")

    print(f"\n  hesitancy eigenvector, served cache, largest six loadings")
    for k in np.argsort(-np.abs(v))[:6]:
        print(f"    {FEATURE_NAMES[k]:<26}{v[k]:>+8.3f}")

    print(f"\n  collapsed now: {r['collapse_features']}")
    print(f"  collapsed before: {rold['collapse_features']}")

    d = r["auc_rf_oob"] - rold["auc_rf_oob"]
    if d <= -0.02:
        verdict = (f"MATERIAL. {d:+.4f}. The duration prior misconfiguration "
                   "was a real contributor. Every mechanism conclusion in the "
                   "four w4 sections has to be re derived on this cache.")
    elif d <= -0.005:
        verdict = (f"MARGINAL ONLY. {d:+.4f}, the order a duration marginal fix "
                   "alone buys. The hesitancy conclusion survives.")
    else:
        verdict = (f"NO EFFECT. {d:+.4f}. The prior was not the cause and the "
                   "w4 conclusions stand as written.")
    if len(r["collapse_features"]) > len(rold["collapse_features"]):
        verdict += (" COLLAPSE WORSENED, so any AUC gain here is a loss under "
                    "the anti Goodhart rule, not a win.")
    print(f"\n  VERDICT  {verdict}\n")

    json.dump({"auc_served": float(r["auc_rf_oob"]),
               "auc_w4_cache": float(rold["auc_rf_oob"]),
               "delta": float(d),
               "collapse_served": r["collapse_features"],
               "collapse_w4_cache": rold["collapse_features"],
               "mean_corr_gap_served": mean_d,
               "mean_corr_gap_w4_cache": mean_do,
               "leading_eig_served": lead, "leading_eig_w4_cache": lead_o,
               "eigenvector_served": v.tolist(),
               "n": int(len(G)), "verdict": verdict},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out} and {args.cache}\n")


if __name__ == "__main__":
    main()
