"""Check that generate.py serves the recorded headline recipe, one command.

Regenerates the served arm of ledger run w4_mserve (arm "mq1": a commanded
duration matched from the held out human pool, the first event from the q
head, the autoregressive model at temperatures 0.95 / 0.90 / 1.00, one draw
per request, no selection) for one seed and 2000 requests, through the same
functions generate.py uses, and scores the result with the contract scorer
against the human reference. The recorded values live in
research/w4_mserve_s<seed>.json: seed 20 reads 0.5777, seed 21 reads 0.5782,
and the ten seed mean is 0.5795 with standard error 0.0022.

    python verify_serve.py                 # seed 20, 2000 rows
    python verify_serve.py --seed 21

Two numbers come out. The served sampler on raw decoded paths, which is the
recorded quantity; and the same rows after the landing transform generate.py
applies so every path ends on its target pixel, so the cost of landing sits
on the record next to the model. The record was made with the exact, uncached
sampler on a CUDA machine. generate.py serves a cached sampler that matches
the exact one to about 1e-4, close enough that on the recording machine the
two produce identical token streams, and on other hardware the random draws
differ and the score lands within draw noise, about 0.01 to 0.02.

Exits 0 when the served sampler is within 0.025 of the recorded value. Needs
the release assets (python setup_data.py) and a few minutes on a CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import generate                                                   # noqa: E402
import scoring                                                    # noqa: E402
from features import extract_feature_matrix                       # noqa: E402
from models.event_ar import class_to_dt_ms                        # noqa: E402
from models.event_stream_polar import decode_events               # noqa: E402
from phase_a_baseline import make_specs                           # noqa: E402

TOLERANCE = 0.025


def score(paths, seed):
    F = extract_feature_matrix(paths)
    F = F[np.all(np.isfinite(F), 1)]
    F = F[np.random.default_rng(seed).permutation(len(F))]
    return float(scoring.score_features(F)["auc_rf_oob"]), len(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args()

    recorded = None
    rec_path = f"research/w4_mserve_s{a.seed}.json"
    if os.path.exists(rec_path):
        recorded = float(json.load(open(rec_path))["arms"]["mq1"]["contract"])

    serve = generate.load_serve()
    print(f"device {serve.device}, seed {a.seed}, n {a.n}, batch {a.batch}")

    geo, meta = [], []
    for sx, sy, ex, ey in make_specs(a.n, a.seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        ang = math.atan2(ey - sy, ex - sx)
        geo.append([math.log(dist), math.cos(ang), math.sin(ang)])
        meta.append((sx, sy, ex, ey, ang))
    geo = np.asarray(geo, dtype=np.float64)
    mdur = generate.draw_log_durations(geo[:, 0], np.random.default_rng(9000 + a.seed))
    cond = torch.tensor(np.stack([geo[:, 0], mdur, geo[:, 1], geo[:, 2]], 1),
                        dtype=torch.float32)
    B = len(geo)

    t0 = time.time()
    paths = []
    for c0 in range(0, B, a.batch):
        sl = slice(c0, min(c0 + a.batch, B))
        cb = cond[sl].to(serve.device)
        torch.manual_seed(a.seed * 100003 + c0 + 7)
        force = generate.first_event_force(serve.q, cb)
        torch.manual_seed(a.seed * 100003 + c0)
        with torch.no_grad():
            s, th, dt = serve.model.sample(
                cb, temperature=generate.AR_TEMPS[0],
                th_temperature=generate.AR_TEMPS[1],
                dt_temperature=generate.AR_TEMPS[2], force=force,
                kv_cache=True)
        s, th = s.cpu().numpy(), th.cpu().numpy()
        dt_ms = class_to_dt_ms(dt.cpu()).numpy()
        for i in range(s.shape[0]):
            sx, sy, ex, ey, ang = meta[c0 + i]
            p = decode_events(s[i], th[i], dt_ms[i], sx, sy, ang)
            if p is not None:
                paths.append((p, ex, ey))
    t_gen = time.time() - t0

    auc_raw, n_raw = score([p for p, _, _ in paths], a.seed)
    landed = [generate._land_on_target(p, ex, ey) for p, ex, ey in paths]
    auc_land, n_land = score(landed, a.seed)
    miss = np.array([math.hypot(p[-1, 0] - ex, p[-1, 1] - ey) for p, ex, ey in paths])
    dist = np.array([math.hypot(ex - p[0, 0], ey - p[0, 1]) for p, ex, ey in paths])

    print()
    print(f"  {'arm':>34}  {'contract':>8}  {'rows':>5}")
    print(f"  {'served sampler, raw paths':>34}  {auc_raw:8.4f}  {n_raw:5d}")
    print(f"  {'served sampler, landed on target':>34}  {auc_land:8.4f}  {n_land:5d}")
    if recorded is not None:
        print(f"  {'recorded mq1, this seed':>34}  {recorded:8.4f}")
    print(f"  generation {t_gen:.0f} s for {B} requests on {serve.device}")
    print(f"  raw endpoint miss, px: median {np.median(miss):.1f}, "
          f"p90 {np.percentile(miss, 90):.1f}; as a share of the requested "
          f"distance: median {np.median(miss / dist):.3f}")
    print(f"  landing cost, same rows: {auc_land - auc_raw:+.4f}")

    if recorded is None:
        print(f"no record for seed {a.seed}; nothing to check against")
        sys.exit(0)
    print(f"  served minus recorded {auc_raw - recorded:+.4f}, tolerance {TOLERANCE}")
    ok = abs(auc_raw - recorded) <= TOLERANCE
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
