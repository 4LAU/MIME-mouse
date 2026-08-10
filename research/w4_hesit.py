"""One per trajectory latent driving all three heads. The hesitancy factor.

Follows w4_latent, which drove the direction head alone. The leading eigenvector
of the human minus generated correlation gap loads on movement_duration and
mean_velocity about as heavily as on the direction features, so a direction only
latent reproduces about half of it.

    z ~ N(0,1) drawn ONCE per trajectory, held for its whole length
    (th_tilt, dt_tilt, s_tilt) = sigma * a * z

`a` is NOT guessed. Phase A measures the Jacobian of the eighteen contract
features against each head's tilt by generating at a small tilt on one head at a
time. Phase B solves for the `a` whose induced feature direction best aligns
with the measured target eigenvector. Phase C sweeps sigma, and the contract is
read only after sigma is chosen.

One trajectory per request, nothing generated twice, nothing selected.

Safety. Scores through research/autoloop/scoring.py only. Never modifies scoring
code, never training/candi_polar_flow_best.pt. Paces itself on GPU temperature:
this machine crashed on this workload on 2026-08-06, kill line tightened to 79C.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_hesit.py
"""
from __future__ import annotations

import argparse
import json
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
from w4_copula import normal_scores  # noqa: E402
from w4_evprice import build_specs  # noqa: E402
from w4_latent import cooldown, gpu_c, throttle  # noqa: E402

HEADS = ("th", "dt", "s")
PROBE = 2.0
SIGMAS = (0.0, 1.0, 2.0, 3.0, 4.0)


def target_direction():
    """Leading eigenvector of the human minus generated correlation gap, in
    normal score space. Measured, not chosen."""
    H = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    G = np.load("research/w4_evprice_cache.npz")["F"]
    D = (np.corrcoef(normal_scores(H), rowvar=False)
         - np.corrcoef(normal_scores(G), rowvar=False))
    w, V = np.linalg.eigh(D)
    return V[:, np.argmax(np.abs(w))], H


def gen(model, rows, meta, tilts, batch, temp, dev, seed, per_step=False):
    """One trajectory per spec, no selection.

    tilts maps head name to a scalar coefficient. z is drawn once per row and
    held, unless per_step, which redraws it every event for the placebo.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    paths = []
    for c0 in range(0, len(rows), batch):
        throttle()
        blk = rows[c0:c0 + batch]
        cond = torch.tensor(blk, dtype=torch.float32, device=dev)
        n = len(blk)
        kw = {}
        if any(tilts.values()):
            if per_step:
                kw = {f"{h}_tilt": _StepDraw(n, c, g, dev)
                      for h, c in tilts.items() if c}
            else:
                z = torch.randn(n, generator=g)
                kw = {f"{h}_tilt": (z * c).to(dev)
                      for h, c in tilts.items() if c}
        s_cls, th_cls, dt_cls = model.sample(cond, temperature=temp, **kw)
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
    F = extract_feature_matrix(paths)
    return F[np.all(np.isfinite(F), 1)]


class _StepDraw:
    """Placebo. Redraws on every read so the factor is never shared."""

    def __init__(self, n, c, gen_, dev):
        self.n, self.c, self.g, self.dev = n, c, gen_, dev

    def unsqueeze(self, dim):
        return (torch.randn(self.n, generator=self.g)
                * self.c).to(self.dev).unsqueeze(dim)


def zmean(F, ref):
    """Mean of F's features expressed in the reference sample's normal score
    scale, so the Jacobian columns are comparable across features whose raw
    units differ by ten orders of magnitude."""
    out = np.empty(F.shape[1])
    for k in range(F.shape[1]):
        s = np.sort(ref[:, k])
        u = np.clip(np.searchsorted(s, F[:, k]) / len(s), 1e-4, 1 - 1e-4)
        out[k] = np.mean(np.sqrt(2) * torch.erfinv(
            torch.tensor(2 * u - 1)).numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=300)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigmas", default=None)
    ap.add_argument("--out", default="research/w4_hesit.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    v, H = target_direction()
    rows, meta = build_specs(args.n, args.seed)
    out = {"target_eigenvector": v.tolist()}
    print(f"\n  {len(rows)} specs per arm, one trajectory each, no selection")
    print(f"  probe tilt {PROBE} per head\n")

    with torch.no_grad():
        # PHASE A. Jacobian of the features against each head's tilt.
        cooldown()
        F0 = gen(model, rows, meta, {h: 0.0 for h in HEADS}, args.batch,
                 args.temp, dev, args.seed + 1)
        m0 = zmean(F0, H)
        print(f"  {'head':>6}{'|df| per unit tilt':>22}{'cos to target':>16}")
        J = []
        for h in HEADS:
            cooldown()
            Fh = gen(model, rows, meta,
                     {k: (PROBE if k == h else 0.0) for k in HEADS},
                     args.batch, args.temp, dev, args.seed + 1)
            d = (zmean(Fh, H) - m0) / PROBE
            J.append(d)
            cos = float(d @ v / (np.linalg.norm(d) * np.linalg.norm(v) + 1e-12))
            print(f"  {h:>6}{np.linalg.norm(d):>22.4f}{cos:>16.4f}", flush=True)
        J = np.array(J).T

        # PHASE B. The combination whose induced direction best aligns.
        a, *_ = np.linalg.lstsq(J, v, rcond=None)
        a = a / (np.linalg.norm(a) + 1e-12)
        fit = J @ a
        cos = float(fit @ v / (np.linalg.norm(fit) * np.linalg.norm(v) + 1e-12))
        print(f"\n  solved head weights  " +
              "  ".join(f"{h} {c:+.4f}" for h, c in zip(HEADS, a)))
        print(f"  alignment of the induced direction to the target  {cos:.4f}")
        out["head_weights"] = dict(zip(HEADS, a.tolist()))
        out["alignment"] = cos

        # PHASE C. Sweep the one free scale.
        sigs = ([float(x) for x in args.sigmas.split(",")] if args.sigmas
                else list(SIGMAS))
        print(f"\n  {'sigma':>7}{'mode':>9}{'contract':>10}{'collapse':>10}"
              f"{'n':>7}{'gpu':>6}")
        for sg in sigs:
            for per_step in ([False, True] if sg > 0 else [False]):
                cooldown()
                F = gen(model, rows, meta, {h: sg * c for h, c in zip(HEADS, a)},
                        args.batch, args.temp, dev, args.seed + 1, per_step)
                r = scoring.score_features(F)
                mode = "perstep" if per_step else "latent"
                out[f"s{sg}_{mode}"] = {
                    "sigma": sg, "per_step": per_step,
                    "contract": float(r["auc_rf_oob"]),
                    "collapse_n": len(r["collapse_features"]),
                    "collapse": r["collapse_features"], "n": int(len(F))}
                print(f"  {sg:>7.2f}{mode:>9}{r['auc_rf_oob']:>10.4f}"
                      f"{len(r['collapse_features']):>10}{len(F):>7}"
                      f"{gpu_c():>6}", flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}\n")


if __name__ == "__main__":
    main()
