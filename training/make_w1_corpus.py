"""W1 corpus: winner TOKENS from token-bearing pools + trust33 picks.

The July 6 distillation corpus (make_distill_corpus.py) held per-item SIR
winner tokens. W1's real delta is SET-LEVEL winners: pools regenerated with
EVENT_POOL_TOKENS=1 store every candidate's raw sampled token rows aligned
with the pool's X/owner_idx/trajs, and trust33.py's picks index those same
rows -- so the set-selected winner tokens come straight off disk, no lossy
re-encoding of decoded pixels.

Writes one shard per pool in the exact distill-corpus format
(dt_z, s_cls, th_cls, cond, length), consumable by
train_events_polar_distill.py's load_corpus unchanged.

Run:
    python training/make_w1_corpus.py --pools "pool_w1_s*_k16.npz" \
        --picks-suffix _picks_trust33_f20d85_r30_rf.npy
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.make_distill_corpus import sanitize  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="pool_w1_s*_k16.npz")
    ap.add_argument("--picks-suffix", default="_picks_trust33_f20d85_r30_rf.npy")
    ap.add_argument("--out-prefix", default="w1_corpus")
    args = ap.parse_args()

    pool_paths = sorted(glob.glob(args.pools))
    assert pool_paths, f"no pools match {args.pools}"
    total = 0
    for pp in pool_paths:
        # allow_pickle: pool npz files are produced by this repo's own poolgen
        # (experiments/event_stream_polar.py, object-dtype trajs array), never
        # third-party input.
        pool = np.load(pp, allow_pickle=True)
        assert "dt_z" in pool.files, f"{pp} has no token arrays (EVENT_POOL_TOKENS off?)"
        picks_path = Path(pp).with_suffix("").name + args.picks_suffix
        picks = np.load(picks_path)
        picks = picks[picks >= 0]
        sel_dt, sel_s, sel_th, sel_cond, sel_len = [], [], [], [], []
        for row in picks:
            dt, s, th, L = sanitize(pool["dt_z"][row], pool["s_cls"][row],
                                    pool["th_cls"][row])
            if L < 2:
                continue
            sel_dt.append(dt); sel_s.append(s); sel_th.append(th)
            sel_cond.append(pool["cond"][row]); sel_len.append(L)
        tag = Path(pp).stem.replace("pool_", "")
        shard = OUT_DIR / f"{args.out_prefix}_{tag}.npz"
        np.savez(shard,
                 dt_z=np.stack(sel_dt),
                 s_cls=np.stack(sel_s).astype(np.int16),
                 th_cls=np.stack(sel_th).astype(np.int16),
                 cond=np.stack(sel_cond).astype(np.float32),
                 length=np.asarray(sel_len, dtype=np.int32))
        total += len(sel_len)
        print(f"{pp}: {len(sel_len)} winners -> {shard.name}", flush=True)
    print(f"TOTAL winner tokens: {total}", flush=True)


if __name__ == "__main__":
    main()
