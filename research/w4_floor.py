"""Is the corpus to reference offset subtractable from the model's score?

Registered in /home/aaronadmin/w4_arms/floor_prereg.md before any number.

The record concluded "the served model reads about 0.63 against a 0.53 floor, so
there is still roughly 0.10 of real modelling gap". That subtraction assumes the
corpus offset and the model's defect ADD. Nobody has scored the model against the
corpus it was trained on, which is the arm that tells them apart.

CPU only. Uses scoring.score_features unmodified, the tier 1 contract recipe.
data/human_eval_features.npy is never touched; the module raises on that path.
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
N_REP = 10


def score(synth: np.ndarray, human: np.ndarray, tag: str) -> float:
    """score_features wants a PATH for the human side, so the draw is written
    out. Named corpusdraw_* deliberately: any path containing human_eval is
    refused by the scorer and that guard must keep working."""
    p = SCRATCH / f"corpusdraw_{tag}.npy"
    np.save(p, human)
    return scoring.score_features(synth, human_features_path=p)["auc_rf_oob"]


def main() -> None:
    model = np.load(REPO / "research" / "w4_fmfeats_event_ar_hm_mlp.npz")["F"]
    grpo = np.load(REPO / "data" / "human_val_features_grpo.npy")
    sir = np.load(REPO / "data" / "human_ref_features_sir.npy")

    # The corpus file is ordered by session. A sorted draw reads a narrow band of
    # people and the record says that mistake was made twice on 2026-08-10, so
    # every draw here is shuffled.
    ok = np.flatnonzero(np.load(DATA / "events_feat18_ok.npy"))
    corpus = np.load(DATA / "events_feat18.npy", mmap_mode="r")
    print(f"model {model.shape}  grpo {grpo.shape}  sir {sir.shape}  "
          f"corpus rows ok {len(ok)}", flush=True)

    arms = ["c_vs_c", "c_vs_grpo", "c_vs_sir", "grpo_vs_sir",
            "m_vs_grpo", "m_vs_sir", "m_vs_corpus"]
    rows = {a: [] for a in arms}

    for r in range(N_REP):
        rng = np.random.default_rng(1000 + r)
        m_sub = model[rng.choice(len(model), N_USE, replace=False)]
        s_sub = sir[rng.choice(len(sir), N_USE, replace=False)]
        pick = rng.choice(len(ok), 2 * N_USE, replace=False)
        # fancy-index a memmap in sorted order, then shuffle the result, so the
        # read is sequential but the ROWS handed to the forest are not.
        idx = np.sort(ok[pick])
        block = np.asarray(corpus[idx], dtype=np.float64)
        block = block[rng.permutation(len(block))]
        c1, c2 = block[:N_USE], block[N_USE:]

        got = {
            "c_vs_c": score(c1, c2, "a"),
            "c_vs_grpo": score(c1, grpo, "b"),
            "c_vs_sir": score(c1, s_sub, "c"),
            "grpo_vs_sir": score(grpo, s_sub, "d"),
            "m_vs_grpo": score(m_sub, grpo, "e"),
            "m_vs_sir": score(m_sub, s_sub, "f"),
            "m_vs_corpus": score(m_sub, c1, "g"),
        }
        for a in arms:
            rows[a].append(got[a])
        print(f"  rep {r}  " + "  ".join(f"{a} {got[a]:.4f}" for a in arms),
              flush=True)

    print(f"\n{'arm':<14}{'mean':>9}{'sd':>9}{'se':>9}")
    out = {}
    for a in arms:
        v = np.array(rows[a])
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        out[a] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                  "se": se, "values": v.tolist()}
        print(f"{a:<14}{v.mean():>9.4f}{v.std(ddof=1):>9.4f}{se:>9.4f}")

    # The registered contrast, paired within replicate because every arm in a
    # replicate shares one model subset and one corpus draw.
    d = np.array(rows["m_vs_corpus"]) - np.array(rows["m_vs_grpo"])
    out["paired_m_corpus_minus_m_grpo"] = {
        "mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    d2 = np.array(rows["m_vs_sir"]) - np.array(rows["m_vs_grpo"])
    out["paired_m_sir_minus_m_grpo"] = {
        "mean": float(d2.mean()), "se": float(d2.std(ddof=1) / np.sqrt(len(d2)))}
    print(f"\n  P2  m_vs_corpus minus m_vs_grpo  {d.mean():+.4f} "
          f"se {d.std(ddof=1) / np.sqrt(len(d)):.4f}")
    print(f"  P3  m_vs_sir    minus m_vs_grpo  {d2.mean():+.4f} "
          f"se {d2.std(ddof=1) / np.sqrt(len(d2)):.4f}")

    p = REPO / "research" / "w4_floor.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
