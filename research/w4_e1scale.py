"""w4_e1scale. AMENDMENT 51, registered in step0_prereg.md before this file
existed.

The third and last route to the spread confound. A49 tried to change the
spread by editing the numbers in a feature matrix and died in both
directions: real feature vectors lie on a curved surface and a rescaled one
does not. A50 tried to change it by choosing which real rows are in the set
and died too: selection dominates the thing being varied. Both failures are
in the prereg.

This changes the TRAJECTORIES and featurises again. Each row's x and y are
multiplied by one factor k with log k normal, mean zero, so the median is 1
and the centre of every feature stays put while the spread of the distance
based features widens. Every scored row is then by construction the feature
vector of an actual movement, which is what the other two routes could not
promise.

CPU only, no model generation. Diagnostic only, never a training signal,
never a serve candidate, no selection of trajectories.
"""
import json
import sys

import numpy as np

for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring   # noqa: E402
import ledger    # noqa: E402
from features import (FEATURE_NAMES, extract_features,   # noqa: E402
                      resample_trajectory)

SEEDS = list(range(40, 52))
K = 20
PERM_SEED = 3208
SCALE_SEED = 8800
N = 2000
MAX_T = 256
KMAX = 4
TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000
TARGET_GAP = 0.033      # the HUM to H02 IQR gap from the A50 addendum
ANCHOR = "data/human_val_features_grpo.npy"
REF = "research/w4_e1floor_F_L0_RAW_s{s}.npy"


def time_axis(dt_ms):
    return np.concatenate([[0.0],
                           np.cumsum(np.clip(dt_ms, 0.1, 1000.0) / 1000.0)])


def featurise(dx, dy, dt, L, k):
    """The committed corpus path, with each row's geometry scaled by k."""
    out = np.full((len(L), len(FEATURE_NAMES)), np.nan)
    for i in range(len(L)):
        n = int(L[i])
        x = np.concatenate([[0.0], np.cumsum(dx[i, :n])]) * k[i]
        y = np.concatenate([[0.0], np.cumsum(dy[i, :n])]) * k[i]
        t = time_axis(dt[i, :n])
        if len(x) >= 5:
            fv = extract_features(resample_trajectory(
                list(zip(x.tolist(), y.tolist(), t.tolist()))))
            if fv is not None and np.all(np.isfinite(fv)):
                out[i] = fv
    return out


def iqr(m):
    return np.percentile(m, 75, 0) - np.percentile(m, 25, 0)


def auc_mean(m):
    v = np.array([scoring.score_features(
        m[np.random.default_rng(PERM_SEED + k).permutation(len(m))])["auc_rf_oob"]
        for k in range(K)])
    return float(v.mean()), float(v.std(ddof=1))


def paired(a, b):
    d = np.array([a[s] - b[s] for s in SEEDS])
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, se, (m / se if se > 0 else float("inf"))


def main():
    A = np.load(ANCHOR)
    A = A[np.isfinite(A).all(1)]
    aiqr = iqr(A)

    lengths = np.load("training/events_len.npy")
    NT = len(lengths)
    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(NT, min(NT, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(NT), trained)
    elig = held[lengths[held] > KMAX]
    dt_a = np.load("training/events_dt.npy", mmap_mode="r")
    dx_a = np.load("training/events_dx.npy", mmap_mode="r")
    dy_a = np.load("training/events_dy.npy", mmap_mode="r")

    rows = {}
    for s in SEEDS:
        pick = np.sort(np.random.default_rng(1000 + s)
                       .choice(elig, N, replace=False))
        rows[s] = (np.asarray(dx_a[pick]).astype(np.float64),
                   np.asarray(dy_a[pick]).astype(np.float64),
                   np.asarray(dt_a[pick]).astype(np.float64),
                   np.minimum(lengths[pick], MAX_T).astype(np.int64))
    # one set of standard normal draws per seed, shared by every sigma, so
    # the ladder is paired and cannot cross by resampling noise
    z = {s: np.random.default_rng(SCALE_SEED + s).standard_normal(N)
         for s in SEEDS}

    print(f"  every value is the mean of K={K} permutations,"
          f" {len(SEEDS)} seeds\n", flush=True)

    # GATE 1, identity at sigma 0
    dx, dy, dt, L = rows[SEEDS[0]]
    base = featurise(dx, dy, dt, L, np.exp(0.0 * z[SEEDS[0]]))
    ref = np.load(REF.format(s=SEEDS[0]))
    g1 = np.array_equal(np.nan_to_num(base, nan=-9e18),
                        np.nan_to_num(ref, nan=-9e18))
    print(f"  GATE 1 (sigma 0 reproduces the committed matrix bit for bit):"
          f" {'ok' if g1 else 'FAILED'}", flush=True)
    if not g1:
        print("  GATE 1 FAILED: the scaling or featurisation path does not"
              " match the committed one. Everything below is VOID.")
        return

    # which columns the scaling actually moves, measured not assumed
    probe = featurise(dx, dy, dt, L, np.exp(0.30 * z[SEEDS[0]]))
    moved = np.where(np.abs(iqr(probe) / iqr(base) - 1.0) > 0.02)[0]
    print(f"  the scaling moves {len(moved)} of {len(FEATURE_NAMES)}"
          f" features: {[FEATURE_NAMES[j] for j in moved]}", flush=True)

    def gap(sig):
        """IQR ratio gain on the moved columns, averaged, at this sigma."""
        m = featurise(dx, dy, dt, L, np.exp(sig * z[SEEDS[0]]))
        return float(np.mean((iqr(m) - iqr(base))[moved] / aiqr[moved])), m

    # calibrate sigma by bisection to the registered target gap
    lo, hi = 0.0, 0.60
    ghi, _ = gap(hi)
    print(f"  calibrating sigma to an IQR gain of {TARGET_GAP:+.3f} on those"
          f" columns (sigma 0.60 gives {ghi:+.3f})", flush=True)
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        gm, _ = gap(mid)
        if gm < TARGET_GAP:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    gs, _ = gap(sigma)
    print(f"  calibrated sigma {sigma:.4f}, achieved IQR gain {gs:+.4f}",
          flush=True)

    LADDER = [("s0", 0.0), ("s10", sigma / 10.0), ("s33", sigma / 3.0),
              ("s100", sigma), ("s200", 2.0 * sigma)]
    print(f"  ladder sigmas: " + "  ".join(f"{n} {v:.4f}"
                                           for n, v in LADDER), flush=True)

    lo_a, hi_a = A.min(0), A.max(0)
    per, rat, oor, ratio_span = {n: {} for n, _ in LADDER}, {}, {}, {}
    for s in SEEDS:
        dx, dy, dt, L = rows[s]
        mats = {n: featurise(dx, dy, dt, L, np.exp(v * z[s]))
                for n, v in LADDER}
        ok = np.ones(N, dtype=bool)
        for m in mats.values():
            ok &= np.isfinite(m).all(1)
        b = mats["s0"][ok]
        for n, _ in LADDER:
            m = mats[n][ok]
            per[n][s], _ = auc_mean(m)
            rat.setdefault(n, {})[s] = float(
                np.mean((iqr(m) - iqr(b))[moved] / aiqr[moved]))
            oor.setdefault(n, {})[s] = int(
                ((m < lo_a) | (m > hi_a)).any(1).sum())
            r = m[:, 1] / m[:, 0]
            ratio_span.setdefault(n, {})[s] = (float(np.percentile(r, 1)),
                                               float(np.percentile(r, 99)))
        print(f"  seed {s} rows {int(ok.sum())}   " + "  ".join(
            f"{n} {per[n][s]:.4f}" for n, _ in LADDER), flush=True)

    # GATE 2, still on the surface
    print(f"\n  GATE 2 (the A49 killer must not recur):")
    o0 = float(np.mean([oor['s0'][s] for s in SEEDS]))
    o1 = float(np.mean([oor['s100'][s] for s in SEEDS]))
    sp = np.mean([ratio_span['s100'][s] for s in SEEDS], 0)
    sp0 = np.mean([ratio_span['s0'][s] for s in SEEDS], 0)
    g2 = o1 <= o0 + 25
    print(f"    rows outside the anchor's range: plain {o0:.1f},"
          f" widened {o1:.1f} of 2000   {'ok' if g2 else 'FAILED'}")
    print(f"    velocity spread over mean velocity, 1st to 99th pct:"
          f" plain {sp0[0]:.2f} to {sp0[1]:.2f},"
          f" widened {sp[0]:.2f} to {sp[1]:.2f}")

    # GATE 3, the calibration control
    m10, se10, t10 = paired(per["s10"], per["s0"])
    g3 = abs(m10) <= 0.005
    print(f"\n  GATE 3 (a scaling ten times too small must do nothing):"
          f" {m10:+.4f} se {se10:.4f} t {t10:+.2f}"
          f"   {'ok' if g3 else 'FAILED'}")
    if not g3:
        print("    GATE 3 FAILED: the pipeline moves the AUC even at a"
              " scaling that changes almost nothing, so it is adding signal"
              " of its own. Every read below is VOID (registered).")

    print(f"\n  THE LADDER, against sigma 0, paired over twelve seeds:")
    lad = {}
    for n, v in LADDER[1:]:
        m, se, t = paired(per[n], per["s0"])
        gmean = float(np.mean([rat[n][s] for s in SEEDS]))
        lad[n] = dict(sigma=v, mean=m, se=se, t=t, iqr_gain=gmean)
        print(f"    {n:<5} sigma {v:.4f}  IQR gain {gmean:+.4f}"
              f"   dAUC {m:+.4f}  se {se:.4f}  t {t:+.2f}")
    seq = [lad[n]["mean"] for n, _ in LADDER[1:]]
    mono = all(b >= a for a, b in zip(seq, seq[1:])) or \
        all(b <= a for a, b in zip(seq, seq[1:]))
    print(f"  monotone in sigma: {'yes' if mono else 'NO, quote no slope'}")

    m, se, t = paired(per["s100"], per["s0"])
    if abs(m) >= 0.010:
        v = "SPREAD CARRIES IT"
    elif abs(m) < 0.005:
        v = "SPREAD IS SMALL"
    else:
        v = "BETWEEN"
    if not (g2 and g3):
        v = "VOID, gate failed"
    print(f"\n  PRIMARY READ: widening HUM to the arms' width changes the"
          f" AUC by {m:+.4f} se {se:.4f} t {t:+.2f}")
    print(f"  against the observed HUM to H02 residual of +0.0195")
    print(f"  {v}")
    if v == "SPREAD IS SMALL":
        print("    a width difference the size of the one separating the"
              " corpus rows from the arms cannot carry the residual. The"
              " residuals against HUM stand as model error and the confound"
              " is closed after four amendments.")
    elif v == "SPREAD CARRIES IT" and m > 0:
        print("    widening makes rows MORE detectable, so an arm's extra"
              " width inflates its residual and every arm minus HUM number"
              " OVERSTATES model error. This is the opposite of what A48"
              " assumed.")
    elif v == "SPREAD CARRIES IT" and m < 0:
        print("    widening makes rows LESS detectable, which is what A48"
              " assumed, so arm minus HUM UNDERSTATES model error.")

    res = {"k": K, "seeds": SEEDS, "sigma": sigma, "target_gap": TARGET_GAP,
           "moved_features": [FEATURE_NAMES[j] for j in moved],
           "gate1": bool(g1), "gate2": bool(g2), "gate3": bool(g3),
           "gate3_mean": m10, "monotone": bool(mono), "verdict": v,
           "primary": {"mean": m, "se": se, "t": t},
           "ladder": lad,
           "auc": {n: {str(s): per[n][s] for s in SEEDS} for n, _ in LADDER},
           "iqr_gain": {n: {str(s): rat[n][s] for s in SEEDS}
                        for n, _ in LADDER},
           "out_of_range": {n: {str(s): oor[n][s] for s in SEEDS}
                            for n, _ in LADDER}}
    with open("research/w4_e1scale.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n  wrote research/w4_e1scale.json")
    print("  real trajectories featurised through the committed pipeline,"
          " diagnostic only, never a training signal, no selection")

    rid = ledger.append_row(
        "w4_e1scale",
        {"seeds": SEEDS, "n": N, "k": K, "perm_seed": PERM_SEED,
         "scale_seed": SCALE_SEED, "target_iqr_gap": TARGET_GAP,
         "construction": "per row spatial scaling of real trajectories",
         "reference": "w4_e1floor L0_RAW, corpus human rows"},
        "ok",
        metrics={"sigma": sigma, "gate1": int(g1), "gate2": int(g2),
                 "gate3": int(g3), "gate3_mean": m10,
                 "primary_mean": m, "primary_t": t, "monotone": int(mono)},
        artifacts=["research/w4_e1scale.json"],
        notes=f"AMENDMENT 51 spread confound, third route: real trajectories"
              f" scaled and featurised again. {v}: widening the corpus rows"
              f" to the arms' width moves the AUC {m:+.4f} t {t:+.2f}"
              f" against an observed residual of +0.0195. A49 (editing"
              f" feature matrices) and A50 (selecting rows) both failed."
              f" Diagnostic only, registered in advance.",
        tier=1)
    ledger.regenerate_leaderboard()
    print(f"  ledger {rid}")


if __name__ == "__main__":
    main()
