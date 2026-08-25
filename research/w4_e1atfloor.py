"""w4_e1atfloor. AMENDMENT 38, registered in step0_prereg.md before this
file existed.

Is the best arm at the human floor? The h02 and h0p1 arms and the L0 raw
human reconstruction were built from the same picks, so this compares
them paired on identical rows instead of as unpaired single draws.
CPU only, no generation, diagnostic, never a training signal.
"""
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402

SEEDS = [40, 41, 42, 43, 44, 45]
SHUF_SEED = 3206
ARMS = {"HUM": "research/w4_e1floor_F_L0_RAW_s{s}.npy",
        "H02": "research/w4_e1feat_F_h02_s{s}.npy",
        "H0P1": "research/w4_e1feat_F_h0p1_s{s}.npy"}


def auc(mat, shuf):
    m = mat[np.isfinite(mat).all(1)]
    m = m[np.random.default_rng(shuf).permutation(len(m))]
    return float(scoring.score_features(m)["auc_rf_oob"])


def verdict(mean, t):
    if abs(mean) < 0.010 and abs(t) < 2.0:
        return "AT THE FLOOR"
    if mean >= 0.010 and t >= 3.0:
        return "ABOVE THE FLOOR"
    return "BETWEEN"


def main():
    per = {a: {} for a in ARMS}
    for s in SEEDS:
        for a, tmpl in ARMS.items():
            per[a][s] = auc(np.load(tmpl.format(s=s)), SHUF_SEED + s)
        print(f"  seed {s}: " + "  ".join(f"{a} {per[a][s]:.4f}" for a in ARMS),
              flush=True)

    res = {"per_seed": {a: {str(s): per[a][s] for s in SEEDS} for a in ARMS}}
    for name, arm in (("read1", "H02"), ("read2", "H0P1")):
        d = np.array([per[arm][s] - per["HUM"][s] for s in SEEDS])
        m = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        t = m / se if se > 0 else float("inf")
        v = verdict(m, t)
        res[name] = dict(arm=arm, mean=m, se=se, t=t, verdict=v,
                         per_seed=[float(x) for x in d])
        label = "READ 1 (PRIMARY)" if name == "read1" else "READ 2"
        print(f"\n  {label}: {arm} minus HUM  mean {m:+.4f}  se {se:.4f}"
              f"  t {t:+.2f}  per seed " + " ".join(f"{x:+.4f}" for x in d))
        print(f"  VERDICT: {v}")

    hum = np.array([per["HUM"][s] for s in SEEDS])
    print(f"\n  READ 3, HUM per seed: " + " ".join(f"{x:.4f}" for x in hum)
          + f"   mean {hum.mean():.4f}  sd {hum.std(ddof=1):.4f}")
    res["read3"] = dict(mean=float(hum.mean()), sd=float(hum.std(ddof=1)))

    with open("research/w4_e1atfloor.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1atfloor.json")
    print("  diagnostic only, never a training signal, no selection, no headline")

    rid = ledger.append_row(
        "w4_e1atfloor", {"seeds": SEEDS, "arms": list(ARMS), "shuf_seed": SHUF_SEED},
        "ok",
        metrics={"read1_mean": res["read1"]["mean"], "read1_t": res["read1"]["t"],
                 "read2_mean": res["read2"]["mean"], "read2_t": res["read2"]["t"],
                 "hum_mean": res["read3"]["mean"]},
        artifacts=["research/w4_e1atfloor.json"],
        notes=f"AMENDMENT 38 paired arm against human floor on identical rows."
              f" H02 minus HUM {res['read1']['mean']:+.4f} t {res['read1']['t']:+.2f}"
              f" {res['read1']['verdict']}. H0P1 minus HUM"
              f" {res['read2']['mean']:+.4f}. Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
