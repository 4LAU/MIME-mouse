"""w4_q2serve_headline, AMENDMENT 69 in step0_prereg.md. Read that first.

The headline re-measured with the coupled first event head on the headline
seeds 20 to 29, from the research/w4_q2serve_s{seed}.json records.
  GATE    per seed, mq1 official contract equals research/w4_mserve_s{seed}
          mq1 within 0.0005 (bit exact expected); md5s identical across seeds.
  READ 1  mq2 official single order contract, mean and se over the ten seeds.
          The number that replaces 0.5795.
  READ 2  paired mq2 minus mq1 on ten order means, against A68's -0.0133 se
          0.0020 on seeds 112 to 123; consistent within two combined se.
Prints the ledger row as JSON; appends only with --write-ledger.
Exit 2 on a missing record, 3 on an md5 mismatch, 4 on a gate miss.
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

LEDGER = "research/autoloop/ledger.jsonl"
A68_MEAN, A68_SE = -0.0133, 0.0020
GATE_TOL = 0.0005


def stats(vals):
    v = np.asarray(vals, dtype=np.float64)
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(len(v))
    return mean, sd, se, mean / se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="20 21 22 23 24 25 26 27 28 29")
    ap.add_argument("--write-ledger", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split()]
    paths = [f"research/w4_q2serve_s{s}.json" for s in seeds]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"missing {missing}")
        sys.exit(2)
    docs = [json.load(open(p)) for p in paths]
    recs = [json.load(open(f"research/w4_mserve_s{s}.json")) for s in seeds]
    for s, d in zip(seeds, docs):
        if d["n_rows"] != 2000 or d["seed"] != s:
            print(f"seed {s}: n_rows {d['n_rows']} seed {d['seed']}, refusing")
            sys.exit(2)

    print("GATE: mq1 official against the w4_mserve record")
    print(f"  {'seed':>5}{'mq1 new':>10}{'record':>10}{'diff':>10}")
    miss = []
    for s, d, r in zip(seeds, docs, recs):
        new, old = d["arms"]["mq1"]["contract"], r["arms"]["mq1"]["contract"]
        print(f"  {s:>5}{new:>10.4f}{old:>10.4f}{new - old:>+10.5f}")
        if abs(new - old) > GATE_TOL:
            miss.append(s)
    md5_ref = docs[0]["md5"]
    mismatched = [s for s, d in zip(seeds, docs) if d.get("md5") != md5_ref]
    print(f"  md5 {json.dumps(md5_ref, sort_keys=True)}")
    if mismatched:
        print(f"  MD5 MISMATCH {mismatched}")
        sys.exit(3)
    if miss:
        print(f"  GATE MISS {miss}")
        sys.exit(4)
    print("  gate holds on every seed")

    mq2_off = [d["arms"]["mq2"]["contract"] for d in docs]
    mq1_off = [d["arms"]["mq1"]["contract"] for d in docs]
    mq1_ten = [d["arms"]["mq1"]["ten_mean"] for d in docs]
    mq2_ten = [d["arms"]["mq2"]["ten_mean"] for d in docs]

    print("\nREAD 1 (PRIMARY): mq2 official single order contract, the new headline")
    print(f"  {'seed':>5}{'mq1':>10}{'mq2':>10}")
    for s, b, m in zip(seeds, mq1_off, mq2_off):
        print(f"  {s:>5}{b:>10.4f}{m:>10.4f}")
    r1_mean, r1_sd, r1_se, _ = stats(mq2_off)
    o_mean, _, o_se, _ = stats(mq1_off)
    print(f"  mq2 official mean {r1_mean:.4f}  sd {r1_sd:.4f}  se {r1_se:.4f}")
    print(f"  mq1 official mean {o_mean:.4f}  se {o_se:.4f}  (the 0.5795 record, reproduced)")

    print("\nREAD 2 (CONSISTENCY): paired mq2 minus mq1 on ten order means")
    d = [m - b for b, m in zip(mq1_ten, mq2_ten)]
    print(f"  {'seed':>5}{'mq1 ten':>10}{'mq2 ten':>10}{'diff':>10}")
    for s, b, m, df in zip(seeds, mq1_ten, mq2_ten, d):
        print(f"  {s:>5}{b:>10.4f}{m:>10.4f}{df:>+10.4f}")
    r2_mean, r2_sd, r2_se, r2_t = stats(d)
    comb = math.sqrt(r2_se ** 2 + A68_SE ** 2)
    z = (r2_mean - A68_MEAN) / comb
    consistent = abs(z) <= 2.0
    print(f"  diffs  mean {r2_mean:+.4f}  sd {r2_sd:.4f}  se {r2_se:.4f}  t {r2_t:+.2f}")
    print(f"  against A68 {A68_MEAN:+.4f} se {A68_SE:.4f}: difference {r2_mean - A68_MEAN:+.4f}, "
          f"{z:+.2f} combined se, {'CONSISTENT' if consistent else 'SEED SET DEPENDENT'}")

    with open(LEDGER) as fh:
        tail = [json.loads(l) for l in fh.read().strip().splitlines()[-3:] if l.strip()]
    schema_keys = list(tail[-1].keys())
    config = {"amendment": 69, "seeds": seeds, "paths": paths,
              "primary": "READ 1 mq2 official contract mean, seeds 20 to 29",
              "n_rows_each": 2000, "gate_tol": GATE_TOL,
              "refusal": "missing json, n_rows != 2000, md5 mismatch, or gate miss"}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    content = {
        "run_id": f"w4_q2serve_headline_{ts.replace(':', '').replace('+00:00', 'Z')}_{config_hash[:8]}",
        "timestamp": ts, "workstream": "w4_q2serve", "tier": 1, "confirms_run_id": None,
        "config": config, "config_hash": config_hash, "status": "ok",
        "metrics": {"headline_mq2_official_mean": r1_mean, "headline_mq2_official_se": r1_se,
                    "mq1_official_mean": o_mean, "mq1_official_se": o_se,
                    "read2_paired_ten_mean": r2_mean, "read2_paired_ten_se": r2_se,
                    "read2_t": r2_t, "read2_z_vs_a68": z, "read2_consistent": consistent},
        "safety": {}, "artifacts": paths,
        "notes": ("AMENDMENT 69. The headline re-measured with the coupled first event head "
                  "training/w4_firsthead_q2.pt on the headline seeds 20 to 29, contract scorer, "
                  "2000 rows, one draw per request. READ 1 replaces the 0.5795 record; READ 2 "
                  "is the paired size against A68 on disjoint seeds. The serve decision was L's "
                  "after A68."),
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
