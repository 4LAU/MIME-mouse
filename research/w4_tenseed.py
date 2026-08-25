"""w4_tenseed. Ten seed contract reading of the closed optimum, CPU only.

Two baselines for the SAME serving configuration disagree: the record's paired
mean 0.5792 (w4_ar_eval, batch 32, seeds 0 and 1) and w4_margfix's round zero
0.5963 (batch 250, seeds 0 and 1). The gap is 0.017, two point three se, and
each side is two seeds. The ten texcover stream files are ten seeds of the same
optimum at batch 32, so scoring them is a ten seed reading with a quarter of
the se, and it says which baseline to trust and whether batch size matters.

Nothing is generated. Paths are rebuilt from the stored token streams exactly
as w4_ar_eval decodes them, using make_specs for the start points. Scored both
as stored and shuffled, because w4_ar_eval did not shuffle and w4_margfix did.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

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

import experiments.event_stream_polar as esp                      # noqa: E402
import scoring                                                    # noqa: E402
from features import extract_feature_matrix                       # noqa: E402
from models.event_ar import class_to_dt_ms                        # noqa: E402
from phase_a_baseline import make_specs                           # noqa: E402


def paths_from(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    S, TH, DT, C, seed = z["s"], z["th"], z["dt"], z["cond"], int(z["seed"])
    meta = []
    for sx, sy, ex, ey in make_specs(len(S), seed):
        d = math.hypot(ex - sx, ey - sy)
        if d < 1e-6:
            continue
        meta.append((sx, sy, math.atan2(ey - sy, ex - sx), math.log(d)))
    if len(meta) != len(S):
        raise SystemExit(f"{npz_path}: {len(meta)} specs vs {len(S)} rows")
    # The stored cond carries log distance in column 0. If it does not match
    # the rebuilt spec the row order is not what this assumes.
    ld = np.array([m[3] for m in meta], dtype=np.float32)
    if np.abs(ld - C[:, 0]).max() > 1e-4:
        raise SystemExit(f"{npz_path}: spec order mismatch, max "
                         f"|dlogd| {np.abs(ld - C[:, 0]).max():.4f}")
    dt_ms = class_to_dt_ms(torch.from_numpy(DT.astype(np.int64)))
    dt_z = ((torch.log(dt_ms.clamp(min=0.05)) - esp._DT_MEAN)
            / esp._DT_STD).numpy()
    paths = []
    for j in range(len(S)):
        sx, sy, ang, _ = meta[j]
        p = esp._decode(dt_z[j], S[j], TH[j], sx, sy, ang)
        if p is not None:
            paths.append(np.asarray(p, dtype=np.float64))
    return paths, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", required=True)
    ap.add_argument("--out", default="research/w4_tenseed_results.json")
    a = ap.parse_args()
    print("w4_tenseed. ten seed contract reading of the closed optimum. CPU, "
          "nothing generated.", flush=True)
    rows = []
    print(f"  {'seed':>5}{'n':>7}{'as stored':>11}{'shuffled':>10}"
          f"{'collapse':>10}", flush=True)
    for f in a.streams:
        paths, seed = paths_from(f)
        F = extract_feature_matrix(paths)
        F = F[np.all(np.isfinite(F), 1)]
        r0 = scoring.score_features(F)
        Fs = F[np.random.default_rng(seed).permutation(len(F))]
        r1 = scoring.score_features(Fs)
        rows.append(dict(seed=seed, n=int(len(F)),
                         as_stored=float(r0["auc_rf_oob"]),
                         shuffled=float(r1["auc_rf_oob"]),
                         collapse=bool(r1["collapse_flag"]),
                         collapse_features=list(r1["collapse_features"])))
        print(f"  {seed:>5}{len(F):>7}{rows[-1]['as_stored']:>11.4f}"
              f"{rows[-1]['shuffled']:>10.4f}{str(rows[-1]['collapse']):>10}",
              flush=True)
    st = np.array([r["as_stored"] for r in rows])
    sh = np.array([r["shuffled"] for r in rows])
    k = len(rows)
    print(f"\n  as stored   mean {st.mean():.4f}  sd {st.std(ddof=1):.4f}  "
          f"se {st.std(ddof=1) / math.sqrt(k):.4f}")
    print(f"  shuffled    mean {sh.mean():.4f}  sd {sh.std(ddof=1):.4f}  "
          f"se {sh.std(ddof=1) / math.sqrt(k):.4f}")
    print(f"  record paired mean 0.5792 (seeds 0,1 batch 32)   "
          f"margfix round 0 0.5963 (seeds 0,1 batch 250)")
    seeds = [r["seed"] for r in rows]
    if 0 in seeds and 1 in seeds:
        s01 = sh[seeds.index(0)], sh[seeds.index(1)]
        print(f"  this run's seeds 0,1 shuffled: {s01[0]:.4f} {s01[1]:.4f}  "
              f"mean {np.mean(s01):.4f}")
    json.dump(dict(rows=rows, mean_as_stored=float(st.mean()),
                   mean_shuffled=float(sh.mean()),
                   sd_shuffled=float(sh.std(ddof=1)),
                   se_shuffled=float(sh.std(ddof=1) / math.sqrt(k))),
              open(a.out, "w"), indent=1)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
