"""w4_e1atfloor2. AMENDMENT 40, registered in step0_prereg.md before this
file existed.

AMENDMENT 38 READ 1 redone with permutation averaged scoring, which
AMENDMENT 39 showed is required: a single contract AUC carries about
0.006 of pure row ordering noise. Same rows, same arms, same bars.
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
K = 20
PERM_SEED = 3208
ARMS = {"HUM": "research/w4_e1floor_F_L0_RAW_s{s}.npy",
        "H02": "research/w4_e1feat_F_h02_s{s}.npy",
        "H0P1": "research/w4_e1feat_F_h0p1_s{s}.npy"}


def auc_mean(path):
    m = np.load(path)
    m = m[np.isfinite(m).all(1)]
    v = np.array([scoring.score_features(
        m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])["auc_rf_oob"]
        for k in range(K)])
    return float(v.mean()), float(v.std(ddof=1))


def verdict(mean, t):
    if abs(mean) < 0.010 and abs(t) < 2.0:
        return "AT THE FLOOR"
    if mean >= 0.010 and t >= 3.0:
        return "ABOVE THE FLOOR"
    return "BETWEEN"


def main():
    print(f"  every arm value is the mean of K={K} permutations\n", flush=True)
    per, sds = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for s in SEEDS:
        for a, tmpl in ARMS.items():
            per[a][s], sds[a][s] = auc_mean(tmpl.format(s=s))
        print("  seed %d: " % s + "  ".join(
            f"{a} {per[a][s]:.4f} (sd {sds[a][s]:.4f})" for a in ARMS), flush=True)

    res = {"k": K,
           "per_seed": {a: {str(s): per[a][s] for s in SEEDS} for a in ARMS},
           "per_seed_sd": {a: {str(s): sds[a][s] for s in SEEDS} for a in ARMS}}
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

    allsd = np.array([sds[a][s] for a in ARMS for s in SEEDS])
    print(f"\n  READ 3: within arm permutation sd, mean {allsd.mean():.4f},"
          f" so each averaged value carries about {allsd.mean() / np.sqrt(K):.4f}")
    res["read3"] = dict(mean_perm_sd=float(allsd.mean()),
                        se_of_mean=float(allsd.mean() / np.sqrt(K)))
    print(f"  A38 reported se 0.0047 on READ 1; here it is {res['read1']['se']:.4f}"
          f" ({'lower' if res['read1']['se'] < 0.0047 else 'not lower'})")

    with open("research/w4_e1atfloor2.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1atfloor2.json")
    print("  diagnostic only, never a training signal, no selection, no headline")

    rid = ledger.append_row(
        "w4_e1atfloor2", {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED}, "ok",
        metrics={"read1_mean": res["read1"]["mean"], "read1_se": res["read1"]["se"],
                 "read1_t": res["read1"]["t"], "read2_mean": res["read2"]["mean"],
                 "read2_t": res["read2"]["t"]},
        artifacts=["research/w4_e1atfloor2.json"],
        notes=f"AMENDMENT 40, A38 READ 1 with permutation averaged scoring."
              f" H02 minus HUM {res['read1']['mean']:+.4f} se {res['read1']['se']:.4f}"
              f" t {res['read1']['t']:+.2f} {res['read1']['verdict']}."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
