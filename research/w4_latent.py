"""A per trajectory latent on the direction head, at generation time.

PRE REGISTERED in HANDOFF.md 2026-08-07, "## A per trajectory latent at
generation time". The calibrate then score order, the per step placebo, the
branch thresholds and the prediction of a 0.01 to 0.04 improvement were all
fixed before this file existed.

    z ~ N(0, sigma^2), drawn ONCE per trajectory, held for its whole length
    th logit bias = z * (normalised turn magnitude - its mean)

Generation becomes a latent variable model. One trajectory per request, nothing
generated twice, nothing selected, nothing read that serving would not have.

Safety. Scores through research/autoloop/scoring.py only. Touches no evaluation
data directly, never modifies scoring code, never
training/candi_polar_flow_best.pt. Paces itself on GPU temperature: this machine
crashed on this workload on 2026-08-06 and the kill line is tightened to 79C.

Usage:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=.:research:research/autoloop \
        ~/venvs/mime/bin/python research/w4_latent.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

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

WOBBLE = [10, 11, 12, 13, 16, 17]
HUMAN_WOBBLE_COUPLING = 0.4985     # w4_copula, on human_val_features_grpo
SIGMAS = (0.0, 0.5, 1.0, 2.0, 3.0)

# The kill line is 79 and the card idles in the high 60s under this workload, so
# cooling to 70 between arms cost more wall clock in waiting than the generation
# itself took. 74 with a per batch throttle at 73 holds a five degree margin and
# keeps the GPU working.
COOL_TO = 74
COOL_MAX_S = 420
THROTTLE_AT = 73
THROTTLE_MAX_S = 120


def gpu_c():
    try:
        return int(subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().split()[0])
    except Exception:
        return -1


def _require_sensor():
    """A watchdog that cannot read the sensor is worse than none, because it
    reports success. `gpu_c` returns -1 when nvidia-smi cannot be reached, and
    on 2026-08-09 a training run launched as a systemd unit inherited no PATH,
    found no nvidia-smi, and duty cycled against -1 for its whole life without
    a word. Refuse to run blind instead."""
    c = gpu_c()
    if c < 0:
        raise RuntimeError(
            "GPU temperature is unreadable, so the thermal watchdog is blind. "
            "nvidia-smi lives in /usr/lib/wsl/lib on this machine; check PATH.")
    return c


def cooldown():
    """Block until the card is back under COOL_TO. The 79C line is not a
    suggestion: this workload crashed the machine on 2026-08-06."""
    t0 = time.time()
    while time.time() - t0 < COOL_MAX_S:
        c = _require_sensor()
        if c <= COOL_TO:
            return c
        time.sleep(10)
    return gpu_c()


def throttle():
    """Duty cycle between batches. The 4070 Laptop's power limit cannot be set
    from WSL2, nvidia-smi reports the scope as unsupported, so shaping the
    workload is the only lever there is. Pausing between batches holds the
    average draw down without slowing any batch that does run."""
    t0 = time.time()
    while time.time() - t0 < THROTTLE_MAX_S and _require_sensor() > THROTTLE_AT:
        time.sleep(5)


def coupling(F):
    """Mean pairwise normal score correlation among the six wobble features.
    The calibration target. Computed on generated output only, never against
    the contract's AUC."""
    C = np.corrcoef(normal_scores(F[:, WOBBLE]), rowvar=False)
    off = ~np.eye(len(WOBBLE), dtype=bool)
    return float(C[off].mean())


def run_one(model, rows, meta, sigma, per_step, batch, temp, dev, seed):
    """One trajectory per spec. No selection.

    per_step=False draws z once per row and holds it, which is the latent.
    per_step=True redraws it every event at the same sigma, which is the
    placebo: same extra turn variability, no shared factor.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    paths = []
    for c0 in range(0, len(rows), batch):
        throttle()
        blk = rows[c0:c0 + batch]
        cond = torch.tensor(blk, dtype=torch.float32, device=dev)
        n = len(blk)
        if sigma == 0.0:
            tilt = None
        elif per_step:
            tilt = _StepDraw(n, sigma, g, dev)
        else:
            tilt = (torch.randn(n, generator=g) * sigma).to(dev)
        s_cls, th_cls, dt_cls = model.sample(cond, temperature=temp,
                                             th_tilt=tilt)
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
    """Placebo tilt. Looks like a (B,) tensor to sample() but redraws on every
    read, so the factor is never shared across a trajectory."""

    def __init__(self, n, sigma, gen, dev):
        self.n, self.sigma, self.gen, self.dev = n, sigma, gen, dev

    def unsqueeze(self, dim):
        return (torch.randn(self.n, generator=self.gen)
                * self.sigma).to(self.dev).unsqueeze(dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigmas", default=None)
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--out", default="research/w4_latent.json")
    args = ap.parse_args()

    dev = esp._DEVICE
    ck = torch.load(f"training/{args.ckpt}", map_location=dev,
                    weights_only=False)
    model = EventARModel(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])

    rows, meta = build_specs(args.n, args.seed)
    sigs = ([float(x) for x in args.sigmas.split(",")] if args.sigmas
            else list(SIGMAS))
    print(f"\n  {len(rows)} specs, one trajectory each, no selection")
    print(f"  human wobble coupling target {HUMAN_WOBBLE_COUPLING:.4f}")
    print(f"  cooldown to {COOL_TO}C between arms\n")
    print(f"  {'sigma':>7}{'mode':>10}{'coupling':>10}{'contract':>10}"
          f"{'collapse':>10}{'n':>7}{'gpu':>6}")

    out = {}
    with torch.no_grad():
        for sg in sigs:
            for per_step in ([False, True] if (args.placebo and sg > 0)
                             else [False]):
                c = cooldown()
                F = run_one(model, rows, meta, sg, per_step, args.batch,
                            args.temp, dev, args.seed + 1)
                r = scoring.score_features(F)
                cp, auc = coupling(F), float(r["auc_rf_oob"])
                nfl = len(r["collapse_features"])
                mode = "perstep" if per_step else "latent"
                out[f"s{sg}_{mode}"] = {"sigma": sg, "per_step": per_step,
                                        "coupling": cp, "contract": auc,
                                        "collapse_n": nfl,
                                        "collapse": r["collapse_features"],
                                        "n": int(len(F))}
                print(f"  {sg:>7.2f}{mode:>10}{cp:>10.4f}{auc:>10.4f}"
                      f"{nfl:>10}{len(F):>7}{gpu_c():>6}", flush=True)

    lat = [v for v in out.values() if not v["per_step"]]
    lat.sort(key=lambda v: v["sigma"])
    cps = np.array([v["coupling"] for v in lat])
    sgs = np.array([v["sigma"] for v in lat])
    o = np.argsort(cps)
    sig_star = float(np.interp(HUMAN_WOBBLE_COUPLING, cps[o], sgs[o]))
    base = next(v["contract"] for v in lat if v["sigma"] == 0.0)
    print(f"\n  coupling {HUMAN_WOBBLE_COUPLING:.4f} reached at sigma "
          f"{sig_star:.3f}, chosen WITHOUT reading the contract column")
    print(f"  baseline at sigma 0   {base:.4f}")
    out["sigma_star"] = sig_star
    out["baseline"] = base
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")
    print(f"  next: confirm at sigma {sig_star:.3f}, latent and per step "
          f"placebo, on a fresh seed\n")


if __name__ == "__main__":
    main()
