"""w4_e1rep_hum. AMENDMENT 46, registered in step0_prereg.md before this
file existed.

Build the L0_RAW human reference at the replication seeds. L0_RAW is
exact corpus geometry, no model and no tokens, so this is the cheap
part of w4_e1floor's build lifted out on its own. Row selection is the
same rule w4_e1floor and w4_qladder both use, rng(1000 + seed) over
held rows longer than KMAX, so the rows match the generated arms seed
for seed.

Correctness is not asserted, it is checked: the script rebuilds seed 40
first and requires the result to equal the committed
w4_e1floor_F_L0_RAW_s40.npy exactly before it writes anything new.

Diagnostic only, never a training signal, no selection.
"""
import os
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

from features import FEATURE_NAMES, extract_features, resample_trajectory  # noqa: E402

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
MAX_T = 256
KMAX = 4
N = 2000
NEW_SEEDS = [46, 47, 48, 49, 50, 51]
CHECK_SEED = 40


def time_axis(dt_ms):
    return np.concatenate([[0.0], np.cumsum(np.clip(dt_ms, 0.1, 1000.0) / 1000.0)])


def build_l0(seed, lengths, dt_all, dx_all, dy_all, elig):
    pick = np.sort(np.random.default_rng(1000 + seed).choice(elig, N, replace=False))
    dt_ms = np.asarray(dt_all[pick]).astype(np.float64)
    dx = np.asarray(dx_all[pick]).astype(np.float64)
    dy = np.asarray(dy_all[pick]).astype(np.float64)
    L = np.minimum(lengths[pick], MAX_T).astype(np.int64)
    out = np.full((N, len(FEATURE_NAMES)), np.nan)
    for i in range(N):
        n = int(L[i])
        x = np.concatenate([[0.0], np.cumsum(dx[i, :n])])
        y = np.concatenate([[0.0], np.cumsum(dy[i, :n])])
        t = time_axis(dt_ms[i, :n])
        if len(x) < 5:
            continue
        fv = extract_features(resample_trajectory(
            list(zip(x.tolist(), y.tolist(), t.tolist()))))
        if fv is not None and np.all(np.isfinite(fv)):
            out[i] = fv
    return out


def main():
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]
    arrs = [np.load(f"training/events_{k}.npy", mmap_mode="r")
            for k in ("dt", "dx", "dy")]

    ref = f"research/w4_e1floor_F_L0_RAW_s{CHECK_SEED}.npy"
    got = build_l0(CHECK_SEED, lengths, *arrs, elig)
    want = np.load(ref)
    same = np.array_equal(np.nan_to_num(got, nan=-1.0),
                          np.nan_to_num(want, nan=-1.0))
    print(f"  reproduction check against {ref}: {'EXACT' if same else 'MISMATCH'}",
          flush=True)
    assert same, "L0_RAW reproduction does not match the committed reference"

    for s in NEW_SEEDS:
        path = f"research/w4_e1floor_F_L0_RAW_s{s}.npy"
        assert not os.path.exists(path), f"{path} already exists"
        m = build_l0(s, lengths, *arrs, elig)
        np.save(path, m)
        print(f"  seed {s}: valid {int(np.isfinite(m).all(1).sum())}/{N}"
              f" -> {path}", flush=True)


if __name__ == "__main__":
    main()
