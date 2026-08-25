"""Why the corpus offset came out 0.511 and not the recorded 0.535.

w4_floor's reproduction arms did not reproduce. Its corpus draws are uniform over
all 4.03M rows; the 2026-08-10 diagnostic's were not, and that entry says the
corpus file is ordered by session. This measures the offset as a FUNCTION of how
wide the draw is, which either produces the recorded number from a narrow draw or
refutes the explanation.

Do not read this as an accusation against the old measurement until the narrow
arm actually lands on 0.535. It is a hypothesis with a control attached.

CPU only, scoring.score_features unmodified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "autoloop"))

import scoring  # noqa: E402

DATA = Path.home() / "mts_data"
SCRATCH = Path("/tmp/claude-1000/-home-aaronadmin/"
               "059c9656-a421-4ab6-9053-614d1dc15765/scratchpad")
N_USE = 2000
N_REP = 6


def score(synth, human, tag):
    p = SCRATCH / f"corpusdraw_w{tag}.npy"
    np.save(p, human)
    return scoring.score_features(synth, human_features_path=p)["auc_rf_oob"]


def main() -> None:
    grpo = np.load(REPO / "data" / "human_val_features_grpo.npy")
    ok = np.flatnonzero(np.load(DATA / "events_feat18_ok.npy"))
    corpus = np.load(DATA / "events_feat18.npy", mmap_mode="r")
    n = len(ok)

    # Draw widths, in rows of the session ordered file the 2000 come from.
    # "contig" is a single run of 2000 consecutive rows, the narrowest band of
    # people the file can give. The rest draw 2000 uniformly from a window of
    # the stated width, starting at a random offset.
    widths = [("contig", 2000), ("10k", 10_000), ("100k", 100_000),
              ("1M", 1_000_000), ("all", n)]
    out = {}
    print(f"corpus rows ok {n}\n")
    print(f"{'draw width':<12}{'mean':>9}{'sd':>9}{'se':>9}   values")
    for tag, w in widths:
        vals = []
        for r in range(N_REP):
            rng = np.random.default_rng(2000 + r)
            start = 0 if w >= n else int(rng.integers(0, n - w))
            win = ok[start:start + w]
            pick = np.sort(rng.choice(win, N_USE, replace=False)) \
                if w > N_USE else np.sort(win[:N_USE])
            block = np.asarray(corpus[pick], dtype=np.float64)
            vals.append(score(block, grpo, tag))
        v = np.array(vals)
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        out[tag] = {"window_rows": w, "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)), "se": se, "values": v.tolist()}
        print(f"{tag:<12}{v.mean():>9.4f}{v.std(ddof=1):>9.4f}{se:>9.4f}   "
              + " ".join(f"{x:.4f}" for x in v))

    p = REPO / "research" / "w4_floorwidth.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
