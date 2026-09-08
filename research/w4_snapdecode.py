"""w4_snapdecode, AMENDMENT 70 in step0_prereg.md. Read that first.

Re decodes the stored A69 mq2 token streams two ways, snap 2.5 (served
today) and snap inf (every motion step rounded to whole pixels), lands
both, features and scores all four arms on eleven row orders, CPU only,
nothing sampled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import generate                                                   # noqa: E402
import scoring                                                    # noqa: E402
from features import extract_feature_matrix                       # noqa: E402
from models.event_ar import class_to_dt_ms                        # noqa: E402
from models.event_stream_polar import decode_events               # noqa: E402
from phase_a_baseline import make_specs                           # noqa: E402

LEDGER = "research/autoloop/ledger.jsonl"
GATE_TOL = 0.0005
N_ORDERS = 10
GEOMETRY = [9, 10, 11, 12, 13, 16, 17]   # the residual map's geometry group
GATE_RAW = {20: 0.5700, 21: 0.5580, 22: 0.5565, 23: 0.5807, 24: 0.5805,
            25: 0.5809, 26: 0.5664, 27: 0.5734, 28: 0.5793, 29: 0.5716}
GATE_LANDED = {20: 0.5808, 21: 0.5628, 22: 0.5737, 23: 0.5738, 24: 0.5836,
               25: 0.5841, 26: 0.5729, 27: 0.5714, 28: 0.5893, 29: 0.5880}
HEADLINE_SEEDS = list(range(20, 30))


def stats(vals):
    v = np.asarray(vals, dtype=np.float64)
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(len(v))
    return mean, sd, se, mean / se


def one_seed(seed):
    npz = f"research/w4_q2serve_s{seed}.npz"
    if not os.path.exists(npz):
        print(f"missing {npz}")
        sys.exit(2)
    z = np.load(npz)
    s_tok, th_tok, dt_tok = z["s_mq2"], z["th_mq2"], z["dt_mq2"]
    if s_tok.shape[0] != 2000:
        print(f"seed {seed}: s_mq2 rows {s_tok.shape[0]}, refusing")
        sys.exit(2)

    meta = []
    for sx, sy, ex, ey in make_specs(2000, seed):
        dist = math.hypot(ex - sx, ey - sy)
        if dist < 1e-6:
            continue
        meta.append((sx, sy, ex, ey, math.atan2(ey - sy, ex - sx)))
    if len(meta) != s_tok.shape[0]:
        print(f"seed {seed}: {len(meta)} requests against "
              f"{s_tok.shape[0]} token rows, refusing")
        sys.exit(2)

    dt_ms = class_to_dt_ms(torch.from_numpy(dt_tok)).numpy()

    raw0, raw1, land0, land1 = [], [], [], []
    n_none = 0
    for i, (sx, sy, ex, ey, ang) in enumerate(meta):
        p0 = decode_events(s_tok[i], th_tok[i], dt_ms[i], sx, sy, ang)
        if p0 is None:
            n_none += 1
            continue
        p1 = decode_events(s_tok[i], th_tok[i], dt_ms[i], sx, sy, ang, snap=math.inf)
        raw0.append(p0)
        raw1.append(p1)
        land0.append(generate._land_on_target(p0, round(ex), round(ey)))
        land1.append(generate._land_on_target(p1, round(ex), round(ey)))

    F = {"D0": extract_feature_matrix(raw0), "D1": extract_feature_matrix(raw1),
         "D0L": extract_feature_matrix(land0), "D1L": extract_feature_matrix(land1)}
    ok = np.all(np.isfinite(F["D0"]), 1)
    for arm in ("D1", "D0L", "D1L"):
        ok = ok & np.all(np.isfinite(F[arm]), 1)
    for arm in F:
        F[arm] = F[arm][ok]
    n = len(F["D0"])
    n_dropped = s_tok.shape[0] - n_none - n

    perm_official = np.random.default_rng(seed).permutation(n)
    perms = [np.random.default_rng(seed * 7919 + j).permutation(n) for j in range(N_ORDERS)]

    cell = {"n_rows": n, "n_dropped": n_dropped, "n_none": n_none}
    for arm in ("D0", "D1", "D0L", "D1L"):
        official = float(scoring.score_features(F[arm][perm_official])["auc_rf_oob"])
        ten = [float(scoring.score_features(F[arm][p])["auc_rf_oob"]) for p in perms]
        cell[arm] = {"official": official, "ten": ten, "ten_mean": float(np.mean(ten))}

    human = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    n_use = min(len(human), n)
    Hg = np.delete(human[:n_use], GEOMETRY, axis=1)
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    for arm in ("D0", "D1"):
        Xg = np.delete(F[arm][perm_official][:n_use], GEOMETRY, axis=1)
        clf = RandomForestClassifier(n_estimators=scoring.RF_N_ESTIMATORS, oob_score=True,
                                     n_jobs=-1, random_state=scoring.RF_SEED)
        clf.fit(np.vstack([Hg, Xg]), y)
        cell[f"nogeo_{arm}"] = float(roc_auc_score(y, clf.oob_decision_function_[:, 1]))
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",
                    default="20 21 22 23 24 25 26 27 28 29 112 113 114 115 116 117 118 "
                            "119 120 121 122 123")
    ap.add_argument("--out", default="research/w4_snapdecode.json")
    ap.add_argument("--write-ledger", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split()]

    cells = {}
    if a.resume and os.path.exists(a.out):
        cells = json.load(open(a.out)).get("cells", {})
    doc = {"amendment": 70, "seeds": seeds, "cells": cells, "reads": {}}
    for seed in seeds:
        if str(seed) in cells:
            print(f"seed {seed}: cached cell, skipping", flush=True)
            continue
        cell = one_seed(seed)
        cells[str(seed)] = cell
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        print(f"seed {seed}: D0 {cell['D0']['official']:.4f}  D1 {cell['D1']['official']:.4f}"
              f"  D0L {cell['D0L']['official']:.4f}  D1L {cell['D1L']['official']:.4f}"
              f"  nogeo D0 {cell['nogeo_D0']:.4f}  nogeo D1 {cell['nogeo_D1']:.4f}"
              f"  rows {cell['n_rows']}  dropped {cell['n_dropped']}"
              f"  none {cell['n_none']}", flush=True)

    print("\nGATE: D0 against GATE_RAW, D0L against GATE_LANDED (seeds 20 to 29)")
    print(f"  {'seed':>5}{'D0':>10}{'gate':>10}{'diff':>10}"
          f"{'D0L':>10}{'gate':>10}{'diff':>10}")
    miss = []
    for s in seeds:
        if s not in GATE_RAW:
            continue
        d0 = cells[str(s)]["D0"]["official"]
        d0l = cells[str(s)]["D0L"]["official"]
        g0, gl = GATE_RAW[s], GATE_LANDED[s]
        print(f"  {s:>5}{d0:>10.4f}{g0:>10.4f}{d0 - g0:>+10.5f}"
              f"{d0l:>10.4f}{gl:>10.4f}{d0l - gl:>+10.5f}")
        if abs(d0 - g0) > GATE_TOL or abs(d0l - gl) > GATE_TOL:
            miss.append(s)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    if miss:
        print(f"  GATE MISS {miss}")
        sys.exit(4)
    print("  gate holds on every gated seed")

    present = [s for s in seeds if str(s) in cells]
    headline = [s for s in HEADLINE_SEEDS if str(s) in cells]

    d0_ten = [cells[str(s)]["D0"]["ten_mean"] for s in present]
    d1_ten = [cells[str(s)]["D1"]["ten_mean"] for s in present]
    d_ten = [b - aa for aa, b in zip(d0_ten, d1_ten)]
    off_d = [cells[str(s)]["D1"]["official"] - cells[str(s)]["D0"]["official"]
             for s in present]
    m, sd, se, t = stats(d_ten)
    verdict = "NULL"
    if abs(t) > 2.2:
        verdict = "REAL TOWARD HUMAN" if m < 0 else "REAL AWAY"
    om, _, ose, _ = stats(off_d)
    n_neg = sum(1 for d in d_ten if d < 0)

    print("\nREAD 1 (PRIMARY): D1 minus D0, ten order mean paired per seed")
    print(f"  {'seed':>5}{'D0 ten':>10}{'D1 ten':>10}{'diff':>10}{'off diff':>10}")
    for s, aa, b, d, od in zip(present, d0_ten, d1_ten, d_ten, off_d):
        print(f"  {s:>5}{aa:>10.4f}{b:>10.4f}{d:>+10.4f}{od:>+10.4f}")
    print(f"  ten order diffs  mean {m:+.4f}  sd {sd:.4f}  se {se:.4f}  t {t:+.2f}"
          f"  {verdict}")
    print(f"  official diffs   mean {om:+.4f}  se {ose:.4f}")
    print(f"  seeds with negative ten order diff: {n_neg} of {len(present)}")

    reads = {"read1": {"mean": m, "sd": sd, "se": se, "t": t, "verdict": verdict,
                       "official_mean": om, "official_se": ose, "n_negative": n_neg,
                       "ten_diff": d_ten, "official_diff": off_d}}

    read2 = {}
    print("\nREAD 2: officials over the headline seeds present")
    for arm in ("D1L", "D0L", "D1", "D0"):
        vals = [cells[str(s)][arm]["official"] for s in headline]
        r2m, _, r2se, _ = stats(vals)
        read2[arm] = {"mean": r2m, "se": r2se}
        print(f"  {arm:>3} official  mean {r2m:.4f}  se {r2se:.4f}")
    reads["read2"] = read2

    read3 = {}
    print("\nREAD 3: landing paired on ten order means, headline seeds")
    for hi, lo in (("D1L", "D1"), ("D0L", "D0")):
        d = [cells[str(s)][hi]["ten_mean"] - cells[str(s)][lo]["ten_mean"]
             for s in headline]
        r3m, r3sd, r3se, r3t = stats(d)
        read3[hi] = {"mean": r3m, "sd": r3sd, "se": r3se, "t": r3t, "diff": d}
        print(f"  {hi} minus {lo}  mean {r3m:+.4f}  sd {r3sd:.4f}  se {r3se:.4f}"
              f"  t {r3t:+.2f}")
    reads["read3"] = read3

    read4 = {}
    print("\nREAD 4: nogeo minus official, headline seeds")
    for arm in ("D0", "D1"):
        d = [cells[str(s)][f"nogeo_{arm}"] - cells[str(s)][arm]["official"]
             for s in headline]
        r4m, r4sd, r4se, r4t = stats(d)
        read4[arm] = {"mean": r4m, "sd": r4sd, "se": r4se, "t": r4t, "diff": d}
        print(f"  {arm}  mean {r4m:+.4f}  sd {r4sd:.4f}  se {r4se:.4f}  t {r4t:+.2f}")
    reads["read4"] = read4

    print("\nP7: READ 1 difference on the two seed blocks")
    p7 = {}
    for lo, hi in ((20, 29), (112, 123)):
        block = [s for s in present if lo <= s <= hi]
        d = [cells[str(s)]["D1"]["ten_mean"] - cells[str(s)]["D0"]["ten_mean"]
             for s in block]
        bm, _, bse, _ = stats(d)
        p7[f"s{lo}_{hi}"] = {"mean": bm, "se": bse, "n": len(block)}
        print(f"  seeds {lo} to {hi}  mean {bm:+.4f}  se {bse:.4f}  n {len(block)}")
    m1, se1 = p7["s20_29"]["mean"], p7["s20_29"]["se"]
    m2, se2 = p7["s112_123"]["mean"], p7["s112_123"]["se"]
    p7["difference"] = {"mean": m1 - m2, "se": math.sqrt(se1 ** 2 + se2 ** 2)}
    print(f"  block difference  mean {m1 - m2:+.4f}"
          f"  se {p7['difference']['se']:.4f}")
    reads["p7"] = p7
    doc["reads"] = reads
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)

    with open(LEDGER) as fh:
        tail = [json.loads(l) for l in fh.read().strip().splitlines()[-3:] if l.strip()]
    schema_keys = list(tail[-1].keys())
    config = {"amendment": 70, "seeds": seeds,
              "primary": "READ 1 D1 minus D0 ten order mean paired over seeds",
              "arms": ["D0 snap 2.5 exact ms", "D1 snap inf exact ms",
                       "D0L landed", "D1L landed"],
              "n_rows_each": 2000, "gate_tol": GATE_TOL,
              "refusal": "missing npz, row count mismatch, or gate miss"}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    content = {
        "run_id": f"w4_snapdecode_{ts.replace(':', '').replace('+00:00', 'Z')}_{config_hash[:8]}",
        "timestamp": ts, "workstream": "w4_snapdecode", "tier": 1,
        "confirms_run_id": None, "config": config, "config_hash": config_hash,
        "status": "ok",
        "metrics": {"read1_mean": m, "read1_sd": sd, "read1_se": se, "read1_t": t,
                    "read1_verdict": verdict, "read1_n_negative": n_neg,
                    "read1_official_mean": om, "read1_official_se": ose,
                    "read2_d1l_mean": read2["D1L"]["mean"],
                    "read2_d1l_se": read2["D1L"]["se"],
                    "read2_d0l_mean": read2["D0L"]["mean"],
                    "read2_d0l_se": read2["D0L"]["se"],
                    "read2_d1_mean": read2["D1"]["mean"],
                    "read2_d1_se": read2["D1"]["se"],
                    "read2_d0_mean": read2["D0"]["mean"],
                    "read2_d0_se": read2["D0"]["se"],
                    "read3_d1l_minus_d1_mean": read3["D1L"]["mean"],
                    "read3_d1l_minus_d1_se": read3["D1L"]["se"],
                    "read3_d0l_minus_d0_mean": read3["D0L"]["mean"],
                    "read3_d0l_minus_d0_se": read3["D0L"]["se"],
                    "read4_d0_mean": read4["D0"]["mean"],
                    "read4_d0_se": read4["D0"]["se"],
                    "read4_d1_mean": read4["D1"]["mean"],
                    "read4_d1_se": read4["D1"]["se"],
                    "read1_mean_s20_29": m1, "read1_mean_s112_123": m2},
        "safety": {}, "artifacts": [a.out],
        "notes": ("AMENDMENT 70. Whole pixel steps in the served decoder (snap inf), "
                  "read on the stored A69 mq2 token streams, four arms, eleven orders, "
                  "CPU. READ 1 is the paired decode difference; READ 2 the served "
                  "numbers on seeds 20 to 29."),
        "mock": False,
    }
    row = {k: None for k in schema_keys}
    row.update(content)
    print("\nledger row:")
    print(json.dumps(row))
    if a.write_ledger:
        with open(LEDGER, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"  appended {LEDGER}")
    else:
        print("  dry read, pass --write-ledger to append")


if __name__ == "__main__":
    main()
