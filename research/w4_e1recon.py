"""w4_e1recon. AMENDMENT 37, registered in step0_prereg.md before this
file existed.

Reconcile the two values the record holds for one quantity: the training
corpus scored against the grpo anchor, 0.5353 on 2026-08-10 and 0.5111
on 2026-08-14. CPU only, human only, no model. Diagnostic, never a
training signal, never touches the protected eval file.
"""
import json
import sys

from pathlib import Path

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402

SEEDS = [40, 41, 42, 43, 44, 45]
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
KMAX = 4
N = 2000
REPS = 5
DRAW_SEED = 3205
SCRATCH = Path("/tmp/claude-1000/-home-aaronadmin/059c9656-a421-4ab6-9053-614d1dc15765/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)


def auc(mat, shuf):
    m = mat[np.isfinite(mat).all(1)]
    m = m[np.random.default_rng(shuf).permutation(len(m))]
    return float(scoring.score_features(m)["auc_rf_oob"])


def main():
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    f18 = np.load("training/events_feat18.npy", mmap_mode="r")
    ok = np.asarray(np.load("training/events_feat18_ok.npy"))
    print(f"  ok mask: {ok.sum()} of {len(ok)} rows true", flush=True)

    picks = {s: np.sort(np.random.default_rng(1000 + s).choice(elig, N, replace=False))
             for s in SEEDS}
    all_pick = np.concatenate([picks[s] for s in SEEDS])

    recon = np.concatenate([np.load(f"research/w4_e1floor_F_L0_RAW_s{s}.npy")
                            for s in SEEDS])
    f18_same = np.asarray(f18[all_pick]).astype(np.float64)
    okm = ok[all_pick]

    res = {}
    print("\n  READ 1, identical rows, my reconstruction against the corpus"
          " featurization:", flush=True)
    res["R_RECON"] = auc(recon, DRAW_SEED)
    res["R_FEAT18"] = auc(f18_same, DRAW_SEED)
    res["R_F18_OK"] = auc(f18_same[okm], DRAW_SEED)
    print(f"  R_RECON  {res['R_RECON']:.4f}")
    print(f"  R_FEAT18 {res['R_FEAT18']:.4f}   (same rows)")
    print(f"  R_F18_OK {res['R_F18_OK']:.4f}   ({int(okm.sum())} of {len(okm)} rows)")
    d1 = res["R_RECON"] - res["R_FEAT18"]
    print(f"  READ 1: recon minus feat18 {d1:+.4f}  "
          f"{'FEATURE PATHS DISAGREE' if abs(d1) >= 0.010 else 'feature paths agree'}")
    res["read1_delta"] = d1

    print("\n  READ 2, row population, %d replicates:" % REPS, flush=True)
    okidx = np.flatnonzero(ok)
    held_ok = np.intersect1d(okidx, held)
    elig_ok = np.intersect1d(okidx, elig)
    unif, heldo, eligo, calib = [], [], [], []
    for r in range(REPS):
        g = np.random.default_rng(DRAW_SEED + 100 * r)
        unif.append(auc(np.asarray(f18[np.sort(g.choice(okidx, N, replace=False))])
                        .astype(np.float64), DRAW_SEED + r))
        heldo.append(auc(np.asarray(f18[np.sort(g.choice(held_ok, N, replace=False))])
                         .astype(np.float64), DRAW_SEED + r))
        eligo.append(auc(np.asarray(f18[np.sort(g.choice(elig_ok, N, replace=False))])
                         .astype(np.float64), DRAW_SEED + r))
        two = g.choice(okidx, 2 * N, replace=False)
        a = np.asarray(f18[np.sort(two[:N])]).astype(np.float64)
        b = np.asarray(f18[np.sort(two[N:])]).astype(np.float64)
        av = a[np.isfinite(a).all(1)]
        bv = b[np.isfinite(b).all(1)]
        # score_features takes the reference side as a path; the name is
        # deliberately not human_eval so the scorer's guard keeps working
        ref = SCRATCH / f"corpusdraw_calib_{r}.npy"
        np.save(ref, bv)
        calib.append(float(scoring.score_features(
            av, human_features_path=ref)["auc_rf_oob"]))
        print(f"  rep {r}: uniform {unif[-1]:.4f}  held out {heldo[-1]:.4f}"
              f"  eligible {eligo[-1]:.4f}  calib {calib[-1]:.4f}", flush=True)

    for k, v in (("R_UNIFORM", unif), ("R_HELDOUT", heldo),
                 ("R_ELIGIBLE", eligo), ("C_CALIB", calib)):
        a = np.array(v, dtype=float)
        res[k] = dict(mean=float(np.nanmean(a)),
                      se=float(np.nanstd(a, ddof=1) / np.sqrt(len(a))))
        print(f"  {k:>11}: {res[k]['mean']:.4f} se {res[k]['se']:.4f}")

    with open("research/w4_e1recon.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1recon.json")
    print("  diagnostic only, never a training signal, no selection, no headline")

    rid = ledger.append_row(
        "w4_e1recon",
        {"seeds": SEEDS, "n": N, "reps": REPS, "draw_seed": DRAW_SEED},
        "ok",
        metrics={"R_RECON": res["R_RECON"], "R_FEAT18": res["R_FEAT18"],
                 "read1_delta": d1, "R_UNIFORM": res["R_UNIFORM"]["mean"],
                 "C_CALIB": res["C_CALIB"]["mean"]},
        artifacts=["research/w4_e1recon.json"],
        notes=f"AMENDMENT 37 reconciliation of 0.5349 against the record's"
              f" 0.5111. recon {res['R_RECON']:.4f} feat18 same rows"
              f" {res['R_FEAT18']:.4f} uniform {res['R_UNIFORM']['mean']:.4f}."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
