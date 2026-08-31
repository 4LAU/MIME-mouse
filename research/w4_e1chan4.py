"""w4_e1chan4. AMENDMENT 61, registered in step0_prereg.md before this
file existed and before any number from it existed.

A60's marginal per feature instrument failed its own negative control,
so the channel that orders the arms inside a length band is still
unnamed. This asks a joint question instead: which features does the
FOREST USE differently when it is separating the arm from the anchor
than when it is separating the zero from the anchor. Importances come
out of the fitted forest, so interactions are already inside them.

Two controls are registered and either can void the primary read. READ 2
is the sign reversal control, applied PER FEATURE this time, because
A60's set level wording had to be resolved after the numbers arrived.
READ 3 is the fit strength control: a forest that separates less well
spreads importance more evenly, which would look exactly like a channel.

The twelve seeds are the SAME ones A56, A58, A59 and A60 used, so this
generates leads and confirms nothing. CPU only, no torch, no generation.
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
FLOOR = "research/w4_e1floor_F_{a}_s{s}.npy"
ARM = "research/w4_e1feat_F_{a}_s{s}.npy"
ZERO = "L3_FULL"
ARMS = ["k0", "h016"]
PRIMARY = ("17 to 32", 17, 32)
FULL = ("full", 0, MAX_T)
OTHER = [("5 to 15", 5, 15), ("33 to 64", 33, 64), ("65 to 256", 65, 256)]
GATE_JSON = "research/w4_e1len2.json"
GATE_TOL = 1e-4
EVEN = 1.0 / len(FEATURE_NAMES)
MEAN_BAR = 0.010      # importance share, about a fifth of an even share
T_BAR = 3.0
FLAT_VOID = -0.7      # READ 3, at or below this the shift is flattening
FLAT_WEAK = -0.4


def scored(m):
    """mean AUC and mean importance vector over K row permutations."""
    aucs, imps = [], []
    for k in range(K):
        r = scoring.score_features(
            m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])
        aucs.append(r["auc_rf_oob"])
        imps.append([r["importances"][n] for n in FEATURE_NAMES])
    return float(np.mean(aucs)), np.asarray(imps).mean(0)


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


def table(rowset, shifts, note=""):
    """shifts is seeds by features. prints sorted by mean, descending."""
    out = {}
    order = []
    for j, name in enumerate(FEATURE_NAMES):
        m, se, t = paired(shifts[:, j])
        out[name] = dict(mean=m, se=se, t=t, bar=bar(m, t))
        order.append((m, name))
    print(f"    {'feature':<28}{'mean':>8}{'se':>8}{'t':>8}  bar{note}")
    for _, name in sorted(order, reverse=True):
        d = out[name]
        print(f"    {name:<28}{d['mean']:+.4f}  {d['se']:.4f}"
              f"  {d['t']:+6.2f}  {d['bar']}", flush=True)
    return out


def main():
    print(f"  every value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds {SEEDS[0]} to {SEEDS[-1]}, the SAME twelve"
          f" A56, A58, A59 and A60 used.")
    print(f"  THIS GENERATES LEADS AND CONFIRMS NOTHING (registered)."
          f"  Importance shares sum to 1, an even share is {EVEN:.4f}.\n",
          flush=True)

    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]

    mats, per_ok, per_len = {}, {}, {}
    for s in SEEDS:
        m = {ZERO: np.load(FLOOR.format(a=ZERO, s=s))}
        for a in ARMS:
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

    cache = {}

    def get(band, which):
        """(aucs, imps) over seeds for one matrix on one row set."""
        key = (band[0], which)
        if key not in cache:
            a, im = [], []
            for s in SEEDS:
                x, y = scored(mats[s][which][rows(s, band[1], band[2])])
                a.append(x)
                im.append(y)
            cache[key] = (np.asarray(a), np.asarray(im))
        return cache[key]

    # GATE. Both reference values are read out of A59's json at run time,
    # never typed from memory (registered after making that mistake twice).
    ref = json.load(open(GATE_JSON))["levels"]
    print("  GATE (twelve seed mean of the zero's AUC, both row sets):",
          flush=True)
    gate = True
    for band in (PRIMARY, FULL):
        want = float(ref[band[0]]["zero"])
        got = float(get(band, ZERO)[0].mean())
        ok = abs(got - want) <= GATE_TOL
        gate = gate and ok
        print(f"    {band[0]:<10} loaded {want:.6f}  computed {got:.6f}"
              f"   {'ok' if ok else 'MISS'}", flush=True)
    print(f"  {'GATE PASSED' if gate else 'GATE FAILED, every read below is VOID (registered)'}",
          flush=True)
    if not gate:
        return

    def shifts(band, arm):
        return get(band, arm)[1] - get(band, ZERO)[1]

    print(f"\n  READ 1 (PRIMARY), band {PRIMARY[0]}, importance share,"
          f" k0 minus the zero. POSITIVE means the forest leans on it MORE"
          f" when separating k0.")
    r1 = table(PRIMARY, shifts(PRIMARY, "k0"))
    real1 = [n for n in FEATURE_NAMES if r1[n]["bar"] == "REAL"]
    print(f"    {len(real1)} feature(s) REAL: {', '.join(real1) if real1 else 'none'}")

    print(f"\n  READ 2, THE NEGATIVE CONTROL, PER FEATURE (registered)."
          f" On the full row set k0 is MORE detectable, so a feature that"
          f" explains the ordering must shift the OTHER WAY here.")
    r2 = table(FULL, shifts(FULL, "k0"))
    verdict = {}
    print(f"\n    each READ 1 feature judged ALONE:")
    for n in real1:
        a, b = r1[n], r2[n]
        if b["bar"] == "NULL":
            v = "UNDECIDED, full row set shift is NULL"
        elif np.sign(b["mean"]) != np.sign(a["mean"]):
            v = "SURVIVES, sign reverses with the ordering"
        else:
            v = "DISQUALIFIED, same sign in both regimes"
        verdict[n] = v
        print(f"    {n:<28}band {a['mean']:+.4f}  full {b['mean']:+.4f}"
              f" t {b['t']:+6.2f}  {v}", flush=True)
    surv = [n for n in real1 if verdict[n].startswith("SURVIVES")]
    if not real1:
        print("    READ 1 named nothing, so there is nothing to control.")
    elif not surv:
        print("    NOTHING SURVIVES. The joint instrument fails the same way"
              " the marginal one did (registered outcome).")
    else:
        print(f"    SURVIVING: {', '.join(surv)}."
              f"  Same twelve seeds, so this is a lead, not evidence.")

    print(f"\n  READ 3, THE FIT STRENGTH CONTROL. Per seed correlation across"
          f" the eighteen features between the shift and the zero's own"
          f" share minus {EVEN:.4f}. Strongly negative means the arm's"
          f" profile is simply flatter and READ 1 names nothing.")
    zi = get(PRIMARY, ZERO)[1]
    sh = shifts(PRIMARY, "k0")
    rs = [float(np.corrcoef(zi[i] - EVEN, sh[i])[0, 1]) for i in range(len(SEEDS))]
    rmean = float(np.mean(rs))
    rse = float(np.std(rs, ddof=1) / np.sqrt(len(rs)))
    flat = ("FLATTENING DOMINATES, READ 1 NAMES NOTHING" if rmean <= FLAT_VOID
            else "CONTAMINATED, any survivor is a weak lead only"
            if rmean <= FLAT_WEAK else "flattening is not driving it")
    print(f"    mean r {rmean:+.3f}  se {rse:.3f}"
          f"   range {min(rs):+.3f} to {max(rs):+.3f}   {flat}", flush=True)

    print(f"\n  READ 4, DESCRIPTIVE, NO VERDICT (bars printed but not"
          f" citable as evidence). h016 in band {PRIMARY[0]}:")
    r4 = table(PRIMARY, shifts(PRIMARY, "h016"))

    print(f"\n  READ 4 continued, k0's shift in the other bands, READ 1's"
          f" features only:")
    other = {}
    for band in OTHER:
        o = {}
        sh_b = shifts(band, "k0")
        for n in (real1 or FEATURE_NAMES[:0]):
            j = FEATURE_NAMES.index(n)
            m, se, t = paired(sh_b[:, j])
            o[n] = dict(mean=m, se=se, t=t, bar=bar(m, t))
            print(f"    {band[0]:<11}{n:<28}{m:+.4f}  se {se:.4f}"
                  f"  t {t:+6.2f}  {o[n]['bar']}", flush=True)
        other[band[0]] = o
    if not real1:
        print("    nothing to carry across bands")

    res = dict(k=K, seeds=SEEDS, gate=bool(gate), even_share=EVEN,
               read1=r1, read1_real=real1, read2=r2, read2_verdict=verdict,
               surviving=surv,
               read3=dict(mean_r=rmean, se=rse, per_seed=rs, call=flat),
               read4_h016=r4, read4_bands=other,
               auc={b[0]: {w: get(b, w)[0].tolist() for w in [ZERO] + ARMS}
                    for b in [PRIMARY, FULL] + OTHER})
    with open("research/w4_e1chan4.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1chan4.json")
    print("  one trajectory per row, no selection, diagnostic only,"
          " never a training signal, no serve decision")

    rid = ledger.append_row(
        "w4_e1chan4",
        {"seeds": SEEDS, "k": K, "perm_seed": PERM_SEED, "zero": ZERO,
         "arms": ARMS, "primary_band": PRIMARY[0],
         "reference": "A60 open question, the joint channel search"},
        "ok" if gate else "failed",
        metrics={"gate": int(gate), "n_real": len(real1),
                 "n_surviving": len(surv), "read3_r": rmean},
        artifacts=["research/w4_e1chan4.json"],
        notes=f"AMENDMENT 61 joint importance channel search in the"
              f" {PRIMARY[0]} band. {len(real1)} REAL, {len(surv)} surviving"
              f" the per feature sign reversal control, fit strength"
              f" correlation {rmean:+.3f}. Used seeds, leads only."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
