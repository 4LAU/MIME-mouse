"""Score the left-to-right autoregressive event model on the contract.

One trajectory per requested spec, no candidates, no selection, no reranking.
Decoding reuses the serving decoder (experiments/event_stream_polar._decode)
unchanged, so the number is comparable to every other one-shot row in the
ledger. The only conversion is time: the AR model emits whole milliseconds and
_decode expects z-scored log milliseconds, so the class is mapped back through
the same exp() the decoder applies, which is exact.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_ar_eval.py \
        --ckpt event_ar_v1.pt --n 1500
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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
from models.event_ar import EventARModel, class_to_dt_ms  # noqa: E402
from models.event_stream_polar import S_PAD_CLASS, class_to_speed  # noqa: E402
from phase_a_baseline import make_specs  # noqa: E402
from w4_seqstats import _acf  # noqa: E402


def _single_feature_auc(F, H, col):
    """Duration-alone separability, the instrument `w4_ms_lattice` used. Same
    forest recipe as the contract scorer, one column instead of eighteen, and
    built here rather than in scoring.py because the contract scorer takes all
    eighteen by definition and must not be touched.

    Reference points: the masked model's continuous duration head scores
    0.7240 here because human durations sit on a whole-millisecond lattice and
    its do not, and rounding them takes it to 0.5649. The AR model chooses
    whole milliseconds directly, so it should land near 0.56. A number well
    ABOVE that is the old tell surviving; a number well above it in the other
    direction is a NEW tell, generated timestamps landing on the lattice about
    thirty times more exactly than human ones do, which is just as separable.
    """
    n = min(len(H), len(F))
    X = np.vstack([H[:n, [col]], F[:n, [col]]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
                                random_state=42)
    rf.fit(X, y)
    return float(roc_auc_score(y, rf.oob_decision_function_[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v1.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--temps", default="1.0",
                    help="swept on the SPEED head; th/dt follow unless pinned")
    ap.add_argument("--th-temp", type=float, default=None)
    ap.add_argument("--dt-temp", type=float, default=None)
    ap.add_argument("--out", default="research/w4_ar_eval.json")
    ap.add_argument("--th-beta-table", default=None,
                    help="research/w4_price.json. applies the fitted confidence "
                         "correction to the direction head at generation, "
                         "indexed by the model's own surprise at the speed it "
                         "just emitted. omit for the unchanged served path.")
    ap.add_argument("--th-beta-arm", choices=("fitted", "reverse", "off"),
                    default="fitted",
                    help="'reverse' is the PLACEBO: the same ten temperatures "
                         "in reversed order, so magnitude and marginal "
                         "distribution match and only the mechanism's sign "
                         "flips. it is what makes the fitted arm readable.")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    print(f"  {args.ckpt} step {ck.get('step')} loss_ema "
          f"{ck.get('loss_ema', float('nan')):.4f}", flush=True)

    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    H = H[np.all(np.isfinite(H), 1)]
    PE = FEATURE_NAMES.index("path_efficiency")
    CS = FEATURE_NAMES.index("curvature_std")
    MD = FEATURE_NAMES.index("movement_duration")
    print(f"  human path_efficiency median {np.median(H[:, PE]):.4f}  "
          f"curvature_std median {np.median(H[:, CS]):.4f}\n", flush=True)

    th_beta = None
    if args.th_beta_table and args.th_beta_arm != "off":
        pr = json.load(open(args.th_beta_table))["families"]["s->th"]
        cen = pr["srp_bin_centres"]
        bet = pr["fam2_beta"]
        keep = [i for i, c in enumerate(cen) if c is not None]
        xs = np.array([cen[i] for i in keep], dtype=np.float64)
        ys = np.array([bet[i] for i in keep], dtype=np.float64)
        if args.th_beta_arm == "reverse":
            ys = ys[::-1].copy()
        o = np.argsort(xs)
        xs, ys = xs[o], ys[o]
        xs_t = torch.tensor(xs, dtype=torch.float32, device=dev)
        ys_t = torch.tensor(ys, dtype=torch.float32, device=dev)

        def th_beta(srp, _x=xs_t, _y=ys_t):
            # Linear interpolation on srp VALUE with flat extrapolation, so the
            # table transfers from the real token surprise distribution it was
            # fitted on to the model's own, which is narrower.
            i = torch.clamp(torch.searchsorted(_x, srp.contiguous()), 1,
                            len(_x) - 1)
            x0, x1_ = _x[i - 1], _x[i]
            y0, y1_ = _y[i - 1], _y[i]
            t = ((srp - x0) / (x1_ - x0)).clamp(0.0, 1.0)
            return y0 + t * (y1_ - y0)

        print(f"  th_beta arm={args.th_beta_arm}  srp knots "
              f"{np.round(xs, 3).tolist()}")
        print(f"                     beta  {np.round(ys, 3).tolist()}\n",
              flush=True)

    specs = make_specs(args.n, args.seed)
    rows, meta = [], []
    for sx, sy, ex, ey in specs:
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ld = math.log(dist)
        ang = math.atan2(ey - sy, ex - sx)
        rows.append([ld, math.log(esp._duration.sample(ld)),
                     math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ang, ex, ey))

    out = {}
    # arrival miss is reported because an in-order model accumulates its own
    # step errors with no later pass to fix them, which is the obvious way
    # this architecture could fail. Reference points from W3 P1: the masked
    # base misses a median 58px, the best of six aiming fine-tunes 55.3px,
    # and the gate that closed that programme was 15px.
    # speed autocorrelation is reported inline so the sweep is self
    # diagnosing: w4_seqstats measured human lag1 0.5952 and lag2 0.6220,
    # lag2 ABOVE lag1, which is the alternation a fixed poll rate imposes on a
    # continuous movement. event_ar_v1 gives 0.6849 and 0.6914, too persistent
    # and with the alternation nearly gone. If heating the speed head alone
    # walks lag1 down to human without wrecking the rest, the defect is
    # calibration. If it cannot, the defect is in the factorization.
    print(f"  {'s_temp':>7}{'contract':>10}{'dur_only':>10}{'path_eff':>10}"
          f"{'curv_std':>10}{'s_ac1':>9}{'s_ac2':>9}{'miss_p50':>10}"
          f"{'n_ev_p50':>10}{'n':>6}")
    for temp in [float(t) for t in args.temps.split(",")]:
        paths, n_ev, miss, spd = [], [], [], []
        for c0 in range(0, len(rows), args.batch):
            cond = torch.tensor(rows[c0:c0 + args.batch], dtype=torch.float32,
                                device=dev)
            s_cls, th_cls, dt_cls = model.sample(
                cond, temperature=temp, th_temperature=args.th_temp,
                dt_temperature=args.dt_temp, th_beta=th_beta)
            pad = (s_cls >= S_PAD_CLASS).cpu().numpy()
            sp_np = s_cls.cpu().numpy()
            for j in range(sp_np.shape[0]):
                k = int(pad[j].argmax()) if pad[j].any() else sp_np.shape[1]
                if k >= 12:
                    spd.append(class_to_speed(
                        torch.from_numpy(sp_np[j, :k].astype(np.int64))).numpy())
            dt_ms = class_to_dt_ms(dt_cls)
            dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
                    / esp._DT_STD).cpu().numpy()
            s_np, th_np = s_cls.cpu().numpy(), th_cls.cpu().numpy()
            for j in range(s_np.shape[0]):
                sx, sy, ang, ex, ey = meta[c0 + j]
                p = esp._decode(dt_z[j], s_np[j], th_np[j], sx, sy, ang)
                if p is not None:
                    a = np.asarray(p, dtype=np.float64)
                    paths.append(a)
                    n_ev.append(len(p))
                    miss.append(math.hypot(a[-1, 0] - ex, a[-1, 1] - ey))
        F = extract_feature_matrix(paths)
        F = F[np.all(np.isfinite(F), 1)]
        auc = float(scoring.score_features(F)["auc_rf_oob"])
        dur = _single_feature_auc(F, H, MD)
        pe, cs = float(np.median(F[:, PE])), float(np.median(F[:, CS]))
        mp = float(np.median(miss))
        ac = _acf(spd, maxlag=2)
        out[f"t{temp}"] = dict(contract=auc, dur_only=dur, path_eff=pe,
                               curv_std=cs, s_ac1=ac[0], s_ac2=ac[1],
                               miss_p50=mp,
                               miss_p90=float(np.percentile(miss, 90)),
                               n_events_p50=float(np.median(n_ev)), n=len(F))
        print(f"  {temp:>7.2f}{auc:>10.4f}{dur:>10.4f}{pe:>10.4f}{cs:>10.4f}"
              f"{ac[0]:>9.4f}{ac[1]:>9.4f}{mp:>10.1f}"
              f"{np.median(n_ev):>10.0f}{len(F):>6}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print("\n  one trajectory per spec, no selection")
    print("  reference split-half floor 0.467 to 0.512")


if __name__ == "__main__":
    main()
