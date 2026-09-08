"""w4_q2serve_read, the AMENDMENT 68 reader for the w4_q2serve seed files.

Refuses a partial arm set: every seed's JSON must be present, n_rows 2000,
both arms present, and every seed's md5 dict must match the first seed's.
READ 1 (PRIMARY) paired read of mq2 minus mq1 on ten order means with the
registered t thresholds. READ 2 the same on the official single order
contract, descriptive. READ 3 e0 tick and pad shares, descriptive.
Prints the ledger row as JSON; appends only with --write-ledger.
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


def stats(vals):
    """Mean, sd (ddof=1), se = sd/sqrt(k), t = mean/se."""
    v = np.asarray(vals, dtype=np.float64)
    k = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(k)
    return mean, sd, se, mean / se


def paired_read(title, col, seeds, base, arm):
    """Per seed table of mq1 vs mq2 plus paired stats on the diffs."""
    d = [m - b for b, m in zip(base, arm)]
    print(f"\n{title}")
    print(f"  {'seed':>5}{'mq1 ' + col:>12}{'mq2 ' + col:>12}{'diff':>10}")
    for s, b, m_, df in zip(seeds, base, arm, d):
        print(f"  {s:>5}{b:>10.4f}{m_:>10.4f}{df:>+10.4f}")
    mean, sd, stats_se, t = stats(d)
    b_mean, _, b_se, _ = stats(base)
    a_mean, _, a_se, _ = stats(arm)
    print(f"  diffs  mean {mean:+.4f}  sd {sd:.4f}  se {stats_se:.4f}  t {t:+.2f}")
    print(f"  mq1 {col} mean {b_mean:+.4f} se {b_se:.4f}   "
          f"mq2 {col} mean {a_mean:+.4f} se {a_se:.4f}")
    return mean, sd, stats_se, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="112 113 114 115 116 117 118 119 120 121 122 123")
    ap.add_argument("--write-ledger", action="store_true")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split()]
    paths = [f"research/w4_q2serve_s{s}.json" for s in seeds]

    docs, bad = [], []
    for s, p in zip(seeds, paths):
        if not os.path.exists(p):
            bad.append(f"seed {s}: missing {p}")
            continue
        d = json.load(open(p))
        if d.get("n_rows") != 2000:
            bad.append(f"seed {s}: n_rows {d.get('n_rows')} != 2000")
        if "mq1" not in d.get("arms", {}) or "mq2" not in d.get("arms", {}):
            bad.append(f"seed {s}: arms missing mq1 or mq2")
        docs.append(d)
    if bad:
        print("AMENDMENT 68 read refused, partial arm set:")
        for b in bad:
            print(f"  {b}")
        sys.exit(2)

    mq1_ten = [d["arms"]["mq1"]["ten_mean"] for d in docs]
    mq2_ten = [d["arms"]["mq2"]["ten_mean"] for d in docs]
    read1_mean, read1_sd, read1_se, read1_t = paired_read(
        "READ 1 (PRIMARY): mq2 minus mq1, ten order mean, paired over seeds",
        "ten", seeds, mq1_ten, mq2_ten)
    if read1_t <= -2.2 and read1_mean < 0:
        read1_verdict = "REAL TOWARD HUMAN"
    elif read1_t >= 2.2:
        read1_verdict = "REAL AGAINST"
    else:
        read1_verdict = "NULL"
    print(f"  verdict {read1_verdict} (t <= -2.2 with a negative mean REAL TOWARD "
          f"HUMAN, t >= 2.2 REAL AGAINST, else NULL, as registered)")

    mq1_con = [d["arms"]["mq1"]["contract"] for d in docs]
    mq2_con = [d["arms"]["mq2"]["contract"] for d in docs]
    read2_mean, read2_sd, read2_se, read2_t = paired_read(
        "READ 2 (DESCRIPTIVE): official single order contract, no verdict",
        "contract", seeds, mq1_con, mq2_con)
    sign = ("matches" if np.sign(read2_mean) == np.sign(read1_mean)
            else "DIFFERS from")
    print(f"  READ 2 sign {sign} READ 1")

    print("\nREAD 3 (DESCRIPTIVE): e0 tick and pad shares, mq2 against mq1")
    print(f"  {'seed':>5}{'mq1 tick':>10}{'mq2 tick':>10}"
          f"{'mq1 pad':>10}{'mq2 pad':>10}")
    for s, d in zip(seeds, docs):
        print(f"  {s:>5}{d['arms']['mq1']['e0_tick_frac']:>10.3f}"
              f"{d['arms']['mq2']['e0_tick_frac']:>10.3f}"
              f"{d['arms']['mq1']['e0_pad_frac']:>10.3f}"
              f"{d['arms']['mq2']['e0_pad_frac']:>10.3f}")

    md5_ref = docs[0]["md5"]
    print(f"\n  md5 {json.dumps(md5_ref, sort_keys=True)}")
    mismatched = [s for s, d in zip(seeds, docs) if d.get("md5") != md5_ref]
    if mismatched:
        print(f"  MISMATCH {mismatched}")
        sys.exit(3)

    with open(LEDGER) as fh:
        tail = [json.loads(l) for l in fh.read().strip().splitlines()[-3:]
                if l.strip()]
    schema_keys = list(tail[-1].keys())
    config = {"amendment": 68, "seeds": seeds, "paths": paths,
              "primary": "READ 1 paired ten_mean, mq2 minus mq1",
              "n_rows_each": 2000,
              "refusal": "missing json, n_rows != 2000, or missing arm"}
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    content = {
        "run_id": f"w4_q2serve_{ts.replace(':', '').replace('+00:00', 'Z')}_{config_hash[:8]}",
        "timestamp": ts,
        "workstream": "w4_q2serve",
        "tier": 1,
        "confirms_run_id": None,
        "config": config,
        "config_hash": config_hash,
        "status": "ok",
        "metrics": {
            "read1_mean": read1_mean, "read1_sd": read1_sd, "read1_se": read1_se,
            "read1_t": read1_t, "read1_verdict": read1_verdict,
            "read2_mean": read2_mean, "read2_se": read2_se, "read2_t": read2_t,
            "mq1_ten_mean": float(np.mean(mq1_ten)),
            "mq2_ten_mean": float(np.mean(mq2_ten)),
        },
        "safety": {},
        "artifacts": paths,
        "notes": ("AMENDMENT 68. The coupled first event head "
                  "training/w4_firsthead_q2.pt vs the served q first event, "
                  "under the contract scorer. READ 1 primary on ten order "
                  "means, READ 2 official order descriptive, READ 3 e0 tick "
                  "and pad shares. Contract numbers, never comparable to the "
                  "e1nn diagnostic AUC rows. Not a serve decision."),
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
