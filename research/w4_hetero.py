"""Does the corpus carry between person heterogeneity the model does not?

Follows w4_floorwidth, which found that 2000 CONSECUTIVE corpus rows read 0.755
against the scoring reference while 2000 rows drawn uniformly read 0.515. The
file is ordered by session, so consecutive rows are few people. A single person
is therefore trivially separable from the population, which means the reference
is a MIXTURE over people and any generator is being asked to reproduce the
mixture, not the average.

The model has cond_dim 4 and no person identity. It can only learn the
population average conditional. If a large share of the corpus's feature variance
is BETWEEN session rather than within, the model is structurally unable to
produce it and would look under dispersed in exactly those directions.

This measures the share and the direction. It builds nothing and authorises
nothing. CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "autoloop"))

from features import FEATURE_NAMES  # noqa: E402

DATA = Path.home() / "mts_data"
BLOCK = 200      # rows per pseudo session block
N_BLOCK = 400    # blocks sampled


def main() -> None:
    ok = np.flatnonzero(np.load(DATA / "events_feat18_ok.npy"))
    corpus = np.load(DATA / "events_feat18.npy", mmap_mode="r")
    model = np.load(REPO / "research" / "w4_fmfeats_event_ar_hm_mlp.npz")["F"]
    grpo = np.load(REPO / "data" / "human_val_features_grpo.npy")
    rng = np.random.default_rng(7)

    # Contiguous blocks of the session ordered file stand in for sessions. The
    # blocks are disjoint and their starts are spread over the whole file.
    starts = np.sort(rng.choice(len(ok) - BLOCK, N_BLOCK, replace=False))
    blocks = []
    for s in starts:
        idx = np.sort(ok[s:s + BLOCK])
        blocks.append(np.asarray(corpus[idx], dtype=np.float64))
    B = np.stack(blocks)                      # (N_BLOCK, BLOCK, 18)
    flat = B.reshape(-1, B.shape[-1])

    # Variance decomposition per feature. Robust scale, because several of these
    # features have heavy tails and a plain variance would be read by one row.
    print(f"{'feature':<26}{'between%':>10}{'model/corpus iqr':>18}"
          f"{'model/grpo iqr':>16}")
    out = {}
    for i, name in enumerate(FEATURE_NAMES):
        bm = B[:, :, i].mean(axis=1)          # per block mean
        v_between = float(bm.var(ddof=1))
        v_within = float(B[:, :, i].var(axis=1, ddof=1).mean())
        share = v_between / (v_between + v_within / BLOCK + 1e-30)
        # share above compares the block mean spread against what pure sampling
        # noise in a block mean would give, so it answers "do blocks differ",
        # not "how big is the effect". Report both.
        share_tot = v_between / (v_between + v_within + 1e-30)
        # IQR, not sd. Several of these features are jerk like and their sd is
        # set by a handful of extreme rows, which made the first pass of this
        # script read a 6x scale difference between two corpora that separate at
        # 0.51. The forest splits on order, so a robust scale is the honest one.
        def iqr(v):
            q = np.percentile(v, [25, 75])
            return float(q[1] - q[0])
        r_corp = iqr(model[:, i]) / (iqr(flat[:, i]) + 1e-30)
        r_grpo = iqr(model[:, i]) / (iqr(grpo[:, i]) + 1e-30)
        out[name] = {"between_over_total": share_tot,
                     "between_vs_blockmean_noise": share,
                     "iqr_model_over_corpus": r_corp,
                     "iqr_model_over_grpo": r_grpo}
        print(f"{name:<26}{100 * share_tot:>9.1f}%{r_corp:>18.3f}{r_grpo:>16.3f}")

    med_corp = float(np.median([v["iqr_model_over_corpus"] for v in out.values()]))
    med_grpo = float(np.median([v["iqr_model_over_grpo"] for v in out.values()]))
    med_share = float(np.median([v["between_over_total"] for v in out.values()]))
    print(f"\n  median between session share of variance   {100 * med_share:.1f}%")
    print(f"  median model iqr over corpus iqr             {med_corp:.3f}")
    print(f"  median model iqr over grpo iqr                 {med_grpo:.3f}")

    out["_summary"] = {"median_between_share": med_share,
                       "median_iqr_model_over_corpus": med_corp,
                       "median_iqr_model_over_grpo": med_grpo,
                       "block_rows": BLOCK, "n_blocks": N_BLOCK}
    p = REPO / "research" / "w4_hetero.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
