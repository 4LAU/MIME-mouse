"""Check that generate.py serves the recorded headline recipe, one command.

Regenerates the served recipe (a commanded duration matched from the held out
human pool, the first event from the coupled q2 head, the autoregressive model
at temperatures 0.95 / 0.90 / 1.00, one draw per request, no selection, every
motion step rounded to whole pixels) for one seed and 2000 requests, through
the same functions generate.py uses, and scores the result with the contract
scorer against the human reference. The record for the served decoder is
research/w4_snapdecode.json (arm D1 raw, D1L landed; seed 20 reads 0.5653 raw
and 0.5697 landed, seed 21 0.5522 and 0.5506; the ten seed means are 0.5622
se 0.0035 raw and 0.5658 se 0.0030 landed). The token streams behind it are
the ones logged in research/w4_q2serve_s<seed>.json, arm mq2, whose decoder
rounded only steps shorter than 2.5 px and took each dwell through a single
precision round trip; that record is kept here as the proof that the token
streams reproduce.

    python verify_serve.py                 # seed 20, 2000 rows
    python verify_serve.py --seed 21

Four numbers come out, all from one set of sampled token streams.

  served sampler, raw paths      what generate.py decodes, before landing
  served sampler, landed         the same rows after the whole pixel landing
  record decoder, same tokens    the rows decoded the way the A69 record was
                                 made (snap 2.5, the record's dwell arithmetic)
  recorded values                D1 and D1L for this seed, and the A69 mq2 value

On the machine the records were made on, the first two return the D1 and
D1L values to four decimals and the third returns the A69 value to four
decimals, which proves the token streams are the record's. On other
hardware the random draws differ and every arm lands within draw noise: a
CPU only run on the recording machine, a different draw, read seed 20 at
0.5412 raw (0.024 under its record) and seed 21 at 0.5595 (0.007 over).

Exits 0 when the served raw paths are within 0.03 of the recorded D1 value.
Needs the release assets (python setup_data.py) and a few minutes on a CPU.
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

TOLERANCE = 0.03

# dt_mean and dt_std of training/event_polar_best.pt, the constants the
# record's decoder z scored dwell times with. Only their float rounding
# matters here; the z score is undone on the next line.
RECORD_DT_MEAN = 2.1850943565368652
RECORD_DT_STD = 1.1489064693450928


def record_dwell_ms(dt_ms):
    """The record decoder's dwell arithmetic: float32 ms through log, z score,
    exp. Returns float64 ms with the record's nanosecond offsets."""
    z = (np.log(np.maximum(dt_ms, 0.05)) - RECORD_DT_MEAN) / RECORD_DT_STD
    return np.exp(z * RECORD_DT_STD + RECORD_DT_MEAN)


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

    recorded = rec_landed = rec_a69 = None
    rec_path = "research/w4_snapdecode.json"
    if os.path.exists(rec_path):
        cells = json.load(open(rec_path))["cells"]
        if str(a.seed) in cells:
            recorded = float(cells[str(a.seed)]["D1"]["official"])
            rec_landed = float(cells[str(a.seed)]["D1L"]["official"])
    a69_path = f"research/w4_q2serve_s{a.seed}.json"
    if os.path.exists(a69_path):
        rec_a69 = float(json.load(open(a69_path))["arms"]["mq2"]["contract"])

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
    paths, rec_paths = [], []
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
            p = decode_events(s[i], th[i], dt_ms[i], sx, sy, ang, snap=generate.STEP_SNAP)
            if p is not None:
                paths.append((p, ex, ey))
                rec_paths.append(decode_events(s[i], th[i], record_dwell_ms(dt_ms[i]),
                                               sx, sy, ang))
    t_gen = time.time() - t0

    auc_raw, n_raw = score([p for p, _, _ in paths], a.seed)
    # generate() rounds requests to whole pixels before drawing; the recorded
    # run drew on fractional targets, so round only where the landing needs it.
    landed = [generate._land_on_target(p, round(ex), round(ey)) for p, ex, ey in paths]
    auc_land, n_land = score(landed, a.seed)
    auc_rec, n_rec = score(rec_paths, a.seed)
    miss = np.array([math.hypot(p[-1, 0] - ex, p[-1, 1] - ey) for p, ex, ey in paths])
    dist = np.array([math.hypot(ex - p[0, 0], ey - p[0, 1]) for p, ex, ey in paths])

    print()
    print(f"  {'arm':>34}  {'contract':>8}  {'rows':>5}")
    print(f"  {'served sampler, raw paths':>34}  {auc_raw:8.4f}  {n_raw:5d}")
    print(f"  {'served sampler, landed on target':>34}  {auc_land:8.4f}  {n_land:5d}")
    print(f"  {'record decoder, same tokens':>34}  {auc_rec:8.4f}  {n_rec:5d}")
    if recorded is not None:
        print(f"  {'recorded D1 raw, this seed':>34}  {recorded:8.4f}")
        print(f"  {'recorded D1L landed, this seed':>34}  {rec_landed:8.4f}")
    if rec_a69 is not None:
        print(f"  {'recorded A69 mq2, this seed':>34}  {rec_a69:8.4f}")
    print(f"  generation {t_gen:.0f} s for {B} requests on {serve.device}")
    print(f"  raw endpoint miss, px: median {np.median(miss):.1f}, "
          f"p90 {np.percentile(miss, 90):.1f}; as a share of the requested "
          f"distance: median {np.median(miss / dist):.3f}")
    print(f"  landing cost, same rows: {auc_land - auc_raw:+.4f}")

    if recorded is None:
        print(f"no record for seed {a.seed}; nothing to check against")
        sys.exit(0)
    print(f"  served minus recorded {auc_raw - recorded:+.4f}, tolerance {TOLERANCE}"
          + ("  (the record's rows, reproduced)" if abs(auc_raw - recorded) < 5e-5 else ""))
    print(f"  landed minus recorded {auc_land - rec_landed:+.4f}")
    if rec_a69 is not None:
        print(f"  record decoder minus A69 {auc_rec - rec_a69:+.4f}"
              + ("  (the record's token streams, reproduced)" if abs(auc_rec - rec_a69) < 5e-5 else ""))
    ok = abs(auc_raw - recorded) <= TOLERANCE
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
