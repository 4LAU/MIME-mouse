"""w4_e1chan3. AMENDMENT 60, registered in step0_prereg.md before this
file existed, and disclosed there as motivated by A59's open question.

k0 sits below the zero inside every length band and neither A58 nor A59
can say which channel does it. A59 cleared movement_duration on size.
This asks all eighteen features at once, using the same 1 Wasserstein
instrument standardized by the anchor's own spread. It can name a
channel or it can exhaust itself, and the second outcome is registered
as a result rather than a failure.

CPU only, no torch, no generation, the same twelve seeds A56, A58 and
A59 used, so this is a control and NOT a replication. Nothing scored
here is a modified matrix: the distances are computed beside the
matrices and every AUC comes from an unmodified eighteen column matrix.
Diagnostic only, never a training signal, never a serve candidate, no
selection.
"""
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402
from features import FEATURE_NAMES   # noqa: E402

SEEDS = list(range(52, 64))
K = 20
PERM_SEED = 3208
N = 2000
MAX_T = 256
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
ANCHOR = "data/human_val_features_grpo.npy"
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
ZERO = "L3_FULL"
PRIMARY_BAND = ("17 to 32", 17, 32)
OTHER_BANDS = [("33 to 64", 33, 64), ("65 to 256", 65, 256)]
FULL = ("full", 0, MAX_T)
GATE_JSON = "research/w4_e1len2.json"
GATE_TOL = 1e-4
MEAN_BAR = 0.02      # anchor standard deviations
T_BAR = 3.0


def auc_mean(m):
    return float(np.mean([scoring.score_features(
        m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])["auc_rf_oob"]
        for k in range(K)]))


def paired(v):
    d = np.asarray(v, float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf"))


def bar(m, t):
    if abs(m) >= MEAN_BAR and abs(t) >= T_BAR:
        return "REAL"
    if abs(t) < 2.0:
        return "NULL"
    return "BETWEEN"


def w1_std(arm_rows, anchor_rows):
    """per feature 1 Wasserstein distance in anchor standard deviations."""
    sd = anchor_rows.std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    return np.mean(np.abs(np.sort(arm_rows, 0) - np.sort(anchor_rows, 0)),
                   axis=0) / sd


def main():
    print(f"  every AUC is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds {SEEDS[0]} to {SEEDS[-1]}, the SAME twelve"
          f" A56, A58 and A59 used, so this is a control and NOT a"
          f" replication\n", flush=True)

    anchor = np.load(ANCHOR)
    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    mats, per_ok, per_len = {}, {}, {}
    for s in SEEDS:
        m = {ZERO: np.load(FLOOR.format(a=ZERO, s=s))}
        for a in ("k0", "h016"):
            m[a] = np.load(ARM.format(a=a, s=s))
        ok = np.ones(len(m[ZERO]), dtype=bool)
        for v in m.values():
            ok &= np.isfinite(v).all(1)
        pick = np.sort(np.random.default_rng(1000 + s).choice(elig, N,
                                                              replace=False))
        mats[s], per_ok[s] = m, ok
        per_len[s] = np.minimum(lengths[pick], MAX_T).astype(np.int64)

    def rows(s, lo, hi):
        return np.flatnonzero(per_ok[s] & (per_len[s] >= lo)
                              & (per_len[s] <= hi))

    # GATE, see AMENDMENT 60 GATE CORRECTION
    ref = json.load(open(GATE_JSON))["levels"]["17 to 32"]["zero"]
    got = float(np.mean([auc_mean(mats[s][ZERO][rows(s, *PRIMARY_BAND[1:])])
                         for s in SEEDS]))
    gate = abs(got - ref) <= GATE_TOL
    print(f"  GATE (twelve seed mean of the zero on the 17 to 32 band):")
    print(f"    {got:.6f} against A59's {ref:.6f}"
          f"   {'ok, GATE PASSED' if gate else 'MISS, every read is VOID (registered)'}",
          flush=True)
    if not gate:
        return

    def terms(band, arm):
        lo, hi = band[1], band[2]
        out = []
        for s in SEEDS:
            ix = rows(s, lo, hi)
            n_use = min(len(anchor), len(ix))
            ha = anchor[:n_use]
            out.append(w1_std(mats[s][ZERO][ix][:n_use], ha)
                       - w1_std(mats[s][arm][ix][:n_use], ha))
        return np.asarray(out)

    def report(band, arm, title):
        d = terms(band, arm)
        rows_out = []
        for j, name in enumerate(FEATURE_NAMES):
            m, se, t = paired(d[:, j].tolist())
            rows_out.append(dict(feature=name, mean=m, se=se, t=t,
                                 bar=bar(m, t)))
        rows_out.sort(key=lambda r: -r["mean"])
        print(f"\n  {title}")
        print(f"    {'feature':<26}{'mean':>9}{'se':>8}{'t':>8}  bar")
        for r in rows_out:
            print(f"    {r['feature']:<26}{r['mean']:+9.4f}{r['se']:8.4f}"
                  f"{r['t']:+8.2f}  {r['bar']}", flush=True)
        return rows_out

    r1 = report(PRIMARY_BAND, "k0",
                "READ 1 (PRIMARY), band 17 to 32, standardized distance to the"
                " anchor, zero minus k0. POSITIVE means k0 is closer.")
    real_pos = [r["feature"] for r in r1
                if r["bar"] == "REAL" and r["mean"] > 0]
    if real_pos:
        print(f"    {len(real_pos)} feature(s) REAL and positive:"
              f" {', '.join(real_pos)}")
        print(f"    These NAME a candidate channel. A marginal difference is"
              f" NOT proof the classifier used it (registered limit).")
    else:
        print(f"    NO feature is REAL and positive. As registered, no"
              f" marginal channel accounts for the ordering, the ordering is"
              f" a JOINT effect, and this instrument is exhausted. The"
              f" ordering itself stays real: A56, A58 and A59 measured it at"
              f" t past 7.")

    r2 = {}
    for b in OTHER_BANDS:
        r2[b[0]] = report(b, "k0",
                          f"READ 2, band {b[0]}, same terms.")
    lead = {PRIMARY_BAND[0]: r1[0]["feature"]}
    for k, v in r2.items():
        lead[k] = v[0]["feature"]
    same = len(set(lead.values())) == 1
    print(f"\n    leading feature by band: "
          + "  ".join(f"{k} {v}" for k, v in lead.items()))
    print(f"    the same in all three: {'yes' if same else 'no'}")

    r3 = report(FULL, "k0",
                "READ 3, THE NEGATIVE CONTROL, full row set where k0 reads"
                " ABOVE the zero. The band leaders must be negative or absent"
                " here or READ 1 means nothing.")
    full_by_name = {r["feature"]: r for r in r3}
    print(f"\n    the band leaders on the full row set:")
    cleared = True
    for name in (real_pos or [r1[0]["feature"]]):
        f = full_by_name[name]
        ok = not (f["bar"] == "REAL" and f["mean"] > 0)
        cleared = cleared and ok
        print(f"    {name:<26}{f['mean']:+9.4f} t {f['t']:+7.2f}  {f['bar']:<8}"
              f"  {'clears the control' if ok else 'FAILS the control'}")
    print(f"    {'the instrument tracks the ordering' if cleared else 'THE INSTRUMENT DOES NOT TRACK THE ORDERING, READ 1 means nothing'}")

    r4 = report(PRIMARY_BAND, "h016",
                "READ 4, DESCRIPTIVE, no verdict. Same terms for h016.")

    res = dict(k=K, seeds=SEEDS, gate=gate, gate_value=got, gate_ref=ref,
               read1=r1, read2=r2, read3=r3, read4=r4,
               real_positive=real_pos, leaders=lead,
               same_leader=bool(same), control_cleared=bool(cleared))
    with open("research/w4_e1chan3.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1chan3.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1chan3",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "zero": ZERO,
         "primary_band": PRIMARY_BAND[0],
         "reference": "A59 open question, which channel orders the arms"},
        "ok" if gate else "failed",
        metrics={"gate": int(gate), "n_real_positive": len(real_pos),
                 "same_leader": int(same), "control_cleared": int(cleared),
                 "top_mean": r1[0]["mean"], "top_t": r1[0]["t"]},
        artifacts=["research/w4_e1chan3.json"],
        notes=f"AMENDMENT 60 marginal channel search inside the 17 to 32"
              f" band. {len(real_pos)} features REAL and positive"
              f" ({', '.join(real_pos) if real_pos else 'none, the ordering is joint'})."
              f" Leader {r1[0]['feature']} at {r1[0]['mean']:+.4f}"
              f" t {r1[0]['t']:+.2f}. Control"
              f" {'cleared' if cleared else 'FAILED'}. Marginal instrument,"
              f" registered limit. Diagnostic only.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
