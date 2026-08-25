"""Score the non-autoregressive diffusion event model on the contract.

Registered in /home/aaronadmin/w4_arms/nardiff_prereg.md.

Everything downstream of the token draw is IDENTICAL to research/w4_ar_eval.py:
the same specs, the same cond construction, the same serving decoder
(experiments/event_stream_polar._decode), the same feature extractor and the
same contract scorer. Only the thing that produces the three token streams
changes. That is what makes the number comparable to the AR rows in the ledger,
and it is the whole point of the arm.

One trajectory per specification. Fixed pass count. No selection, no best of N.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_nardiff_eval.py \
        --ckpt event_nardiff_v1.pt --n 2000 --seeds 0,1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

# The serving decoder reads these at import. They are copied verbatim from
# w4_ar_eval.py and are NOT free parameters here. EVENT_CHOICE_TEMP in
# particular is load bearing: dropping it moves a one shot contract number from
# about 0.65 to about 0.94, which is a decoder artefact and not a model result.
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

import experiments.event_stream_polar as esp  # noqa: E402
import scoring  # noqa: E402
from features import FEATURE_NAMES, extract_feature_matrix  # noqa: E402
from models.event_ar import class_to_dt_ms  # noqa: E402
from models.event_nardiff import EventNARDiff  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS, class_to_speed  # noqa: E402
from phase_a_baseline import make_specs  # noqa: E402
from w4_seqstats import _acf  # noqa: E402

CORPUS_MEDIAN_LEN = 39          # /home/aaronadmin/mts_data, 4028855 rows
AR_OPTIMUM = 0.5792             # closed three head optimum, the STRONG bar
CONTRACT_SD = 0.0073            # pooled within arm sd, w4_floorwidth


def build_specs(n, seed):
    """Identical to w4_ar_eval.py, so the two arms answer the same queries."""
    rows, meta = [], []
    for sx, sy, ex, ey in make_specs(n, seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ld = math.log(dist)
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([ld, math.log(esp._duration.sample(ld)),
                     math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ang, ex, ey))
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_nardiff_v1.pt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--n-steps", type=int, default=32,
                    help="diffusion passes. FIXED for the primary, identical "
                         "for every row. sweeping it is a post primary "
                         "activity and the prereg says so.")
    ap.add_argument("--temps", default="1.0",
                    help="the primary is 1.0. the registration permits a "
                         "sweep only inside the LIVE band and only after the "
                         "primary is written down.")
    ap.add_argument("--th-temp", type=float, default=None)
    ap.add_argument("--dt-temp", type=float, default=None)
    ap.add_argument("--out", default="research/w4_nardiff_eval.json")
    ap.add_argument("--save-streams", action="store_true",
                    help="dump the raw token streams per seed in the same "
                         "layout as w4_texcover_streams_s*.npz, so the "
                         "detcap ladder can be run on this arm without "
                         "regenerating. written AFTER scoring.")
    a = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{a.ckpt}", map_location=dev, weights_only=False)
    model = EventNARDiff(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    n_par = sum(p.numel() for p in model.parameters())
    hist = ck.get("hist") or []
    # hist entries are dicts written by train_event_nardiff.py, not tuples.
    last_elbo = hist[-1]["val_elbo"] if hist else float("nan")
    print(f"  {a.ckpt} step {ck.get('step')}  init_elbo "
          f"{ck.get('init_elbo', float('nan')):.4f} -> {last_elbo:.4f}  "
          f"params {n_par/1e6:.2f}M", flush=True)

    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    H = H[np.all(np.isfinite(H), 1)]
    PE = FEATURE_NAMES.index("path_efficiency")
    CS = FEATURE_NAMES.index("curvature_std")
    print(f"  human path_efficiency median {np.median(H[:, PE]):.4f}  "
          f"curvature_std median {np.median(H[:, CS]):.4f}", flush=True)
    print(f"  bars   STRONG < {AR_OPTIMUM}   LIVE {AR_OPTIMUM} to 0.66   "
          f"DEAD > 0.66\n", flush=True)

    seeds = [int(s) for s in a.seeds.split(",")]
    temps = [float(t) for t in a.temps.split(",")]
    out = {"config": dict(ckpt=a.ckpt, n=a.n, seeds=seeds,
                          n_steps=a.n_steps, temps=temps,
                          th_temp=a.th_temp, dt_temp=a.dt_temp), "runs": {}}

    print(f"  {'seed':>5}{'temp':>7}{'contract':>10}{'path_eff':>10}"
          f"{'curv_std':>10}{'s_ac1':>9}{'s_ac2':>9}{'miss_p50':>10}"
          f"{'len_p50':>9}{'n':>6}")
    for temp in temps:
        for sd in seeds:
            rows, meta = build_specs(a.n, sd)
            torch.manual_seed(1000 + sd)
            paths, n_ev, miss, spd = [], [], [], []
            keep_s, keep_th, keep_dt, keep_c = [], [], [], []
            for c0 in range(0, len(rows), a.batch):
                cond = torch.tensor(rows[c0:c0 + a.batch],
                                    dtype=torch.float32, device=dev)
                s_cls, th_cls, dt_cls = model.sample(
                    cond, n_steps=a.n_steps, temperature=temp,
                    th_temp=a.th_temp, dt_temp=a.dt_temp)
                keep_s.append(s_cls.cpu().numpy().astype(np.int16))
                keep_th.append(th_cls.cpu().numpy().astype(np.int16))
                keep_dt.append(dt_cls.cpu().numpy().astype(np.int16))
                keep_c.append(cond.cpu().numpy())

                pad = (s_cls >= S_PAD_CLASS).cpu().numpy()
                sp_np = s_cls.cpu().numpy()
                for j in range(sp_np.shape[0]):
                    k = int(pad[j].argmax()) if pad[j].any() else sp_np.shape[1]
                    if k >= 12:
                        spd.append(class_to_speed(torch.from_numpy(
                            sp_np[j, :k].astype(np.int64))).numpy())
                dt_ms = class_to_dt_ms(dt_cls)
                dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
                        / esp._DT_STD).cpu().numpy()
                s_np, th_np = sp_np, th_cls.cpu().numpy()
                for j in range(s_np.shape[0]):
                    sx, sy, ang, ex, ey = meta[c0 + j]
                    p = esp._decode(dt_z[j], s_np[j], th_np[j], sx, sy, ang)
                    if p is not None:
                        arr = np.asarray(p, dtype=np.float64)
                        paths.append(arr)
                        n_ev.append(len(p))
                        miss.append(math.hypot(arr[-1, 0] - ex,
                                               arr[-1, 1] - ey))

            Fm = extract_feature_matrix(paths)
            Fm = Fm[np.all(np.isfinite(Fm), 1)]
            rs = np.random.default_rng(sd)
            Fm = Fm[rs.permutation(len(Fm))]     # shuffle before score_features
            res = scoring.score_features(Fm)
            auc = float(res["auc_rf_oob"])
            ac = _acf(spd, maxlag=2)
            lp = float(np.median(n_ev))
            rec = dict(contract=auc,
                       path_eff=float(np.median(Fm[:, PE])),
                       curv_std=float(np.median(Fm[:, CS])),
                       s_ac1=ac[0], s_ac2=ac[1],
                       miss_p50=float(np.median(miss)),
                       miss_p90=float(np.percentile(miss, 90)),
                       len_p50=lp, n=int(len(Fm)),
                       collapse=bool(res["collapse_flag"]),
                       collapse_features=list(res["collapse_features"]))
            out["runs"][f"t{temp}_s{sd}"] = rec
            print(f"  {sd:>5}{temp:>7.2f}{auc:>10.4f}{rec['path_eff']:>10.4f}"
                  f"{rec['curv_std']:>10.4f}{ac[0]:>9.4f}{ac[1]:>9.4f}"
                  f"{rec['miss_p50']:>10.1f}{lp:>9.0f}{len(Fm):>6}",
                  flush=True)

            if a.save_streams:
                sp = f"research/w4_nardiff_streams_s{sd}.npz"
                np.savez_compressed(
                    sp, s=np.concatenate(keep_s), th=np.concatenate(keep_th),
                    dt=np.concatenate(keep_dt),
                    cond=np.concatenate(keep_c).astype(np.float32),
                    s_temp=temp, th_temp=(a.th_temp if a.th_temp else temp),
                    dt_temp=(a.dt_temp if a.dt_temp else temp),
                    seed=sd, ckpt=a.ckpt, n_steps=a.n_steps)
                print(f"    streams -> {sp}", flush=True)

    # ------------------------------------------------------------- gates ----
    prim = [v["contract"] for k, v in out["runs"].items()
            if k.startswith("t1.0_")]
    lens = [v["len_p50"] for k, v in out["runs"].items()
            if k.startswith("t1.0_")]
    coll = [v["collapse"] for k, v in out["runs"].items()
            if k.startswith("t1.0_")]
    print("\n  GATES, read before the primary")
    g1 = all(abs(l - CORPUS_MEDIAN_LEN) / CORPUS_MEDIAN_LEN <= 0.10
             for l in lens)
    g2 = not any(coll)
    print(f"    G1  median length {lens} vs corpus {CORPUS_MEDIAN_LEN}, "
          f"10 pct band  -> {'PASS' if g1 else 'FAIL'}")
    cf = sorted({x for k, v in out["runs"].items()
                 if k.startswith("t1.0_") for x in v["collapse_features"]})
    print(f"    G2  no collapse flag  -> {'PASS' if g2 else 'FAIL'}"
          + (f"   collapsed: {cf}" if cf else ""))
    print("    G3  held out ELBO below init, read from the training log")

    if prim:
        m = float(np.mean(prim))
        se = float(np.std(prim, ddof=1) / math.sqrt(len(prim))) \
            if len(prim) > 1 else float("nan")
        verdict = ("STRONG" if m < AR_OPTIMUM
                   else "LIVE" if m <= 0.66 else "DEAD")
        print(f"\n  PRIMARY, temperature 1.0, {a.n_steps} passes")
        print(f"    seeds {[round(x, 4) for x in prim]}")
        print(f"    mean {m:.4f}  se {se:.4f}  contract sd {CONTRACT_SD}")
        print(f"    AR optimum {AR_OPTIMUM}   diff {m - AR_OPTIMUM:+.4f}")
        print(f"\n  VERDICT  {verdict}")
        if not (g1 and g2):
            print("  A GATE FAILED. the verdict above is NOT readable.")
        out["primary"] = dict(seeds=prim, mean=m, se=se, verdict=verdict,
                              g1=g1, g2=g2)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n  wrote {a.out}")
    print("  one trajectory per spec, fixed pass count, no selection")


if __name__ == "__main__":
    main()
