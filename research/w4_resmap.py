"""The conditional residual map, fitted once on corpus rows and frozen.

Why this exists. Four rollout arms have stopped between 0.595 and 0.60 against
the contract, and the reason is not tuning or credit assignment. The energy
distance those arms minimise cannot represent the part of the gap that is left.
A controlled experiment settles it: scramble a feature's agreement with the rest
of its own trajectory, leaving its mean unchanged, its sd changed by five parts
in ten thousand and every pairwise correlation by at most 0.001. A forest moves
from 0.505 to 0.715. The energy distance at 768 rows moves by less than its own
batch noise. The loss is blind to roughly half the gap and near its floor on the
half it can see.

What this fixes. Learn each feature's value given the others on corpus rows and
keep the residual. On generated rows those residuals are wider than the corpus's
with the median at zero, so the model obeys a tight rule and it is the wrong
rule. For this map, the trained twelve one, the ratio is 1.98 on
mean_acceleration, 1.78 on mean_jerk and 1.51 on curvature_mean. A map allowed
all eighteen features reaches 2.6 to 2.9 on the same rows, which is the cost of
the holdout again. A width difference is first order, which is what
the energy distance is good at. It is invisible in the raw space only because the
residual directions carry under two percent of the variance there. A linear
version of this map is just whitening, which was measured and does not work, so
the nonlinearity is the whole of it.

Two constraints are load bearing.

  the held out six never enter    Every conditional is a trained feature given
                                  the other eleven trained features. A map built
                                  from all eighteen carries the held out
                                  features into the loss through the back door
                                  and destroys the only clean read the arm has.
                                  It costs most of the map's power, from 3.55 to
                                  1.08 of separation at 768 rows, because the
                                  close relatives of the defect features are
                                  exactly the ones held out. Pay it anyway.

  the scorer's reference is never touched   The map is fitted on corpus rows and
                                  on nothing else.

Usage:
    python research/w4_resmap.py fit  research/w4_resmap.joblib
    python research/w4_resmap.py check research/w4_resmap.joblib
"""
import sys

import joblib      # forests do not serialise any other way. The file this reads
import numpy as np  # is written by the fit path below and never leaves the box.

sys.path.insert(0, ".")
sys.path.insert(0, "research")

from features import FEATURE_NAMES  # noqa: E402

HELD_OUT = ["max_acceleration", "velocity_skewness", "curvature_std",
            "num_direction_changes", "time_to_peak_velocity",
            "angular_velocity_std"]
TRAINED = [f for f in FEATURE_NAMES if f not in HELD_OUT]
TK = [FEATURE_NAMES.index(f) for f in TRAINED]
CLIP = 8.0


class ResMap:
    """Maps a raw eighteen column feature matrix to the twelve standardised
    conditional residuals of the trained features. Everything is fitted, so
    applying it to rows from a distribution it was not fitted on is the point:
    that is what makes the residual wide."""

    def __init__(self, regs, rsd):
        self.regs, self.rsd = regs, rsd

    def apply(self, X):
        T = np.asarray(X, dtype=np.float64)[:, TK]
        out = np.empty((len(T), len(TK)), dtype=np.float64)
        for j, (other, reg) in enumerate(self.regs):
            out[:, j] = (T[:, j] - reg.predict(T[:, other])) / self.rsd[j]
        return np.clip(out, -CLIP, CLIP)


def load(path):
    """The file holds a dict of plain sklearn regressors rather than a pickled
    ResMap. Pickling the wrapper records whatever module defined it, which is
    __main__ when the fit runs as a script, and every other process then fails to
    find the class."""
    d = joblib.load(path)
    return ResMap(d["regs"], d["rsd"])


def corpus_rows(n, seed=11):
    ok = np.flatnonzero(np.load("training/events_feat18_ok.npy"))
    Hall = np.load("training/events_feat18.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    C = np.asarray(Hall[np.sort(rng.choice(ok, n, replace=False))],
                   dtype=np.float64)
    C = C[np.isfinite(C).all(1)]
    return C[rng.permutation(len(C))]


def fit(path, n_fit=40000):
    from sklearn.ensemble import RandomForestRegressor
    C = corpus_rows(n_fit + 8000)
    T = C[:n_fit][:, TK]
    H = C[n_fit:][:, TK]
    regs, rsd = [], np.empty(len(TK))
    for j in range(len(TK)):
        other = [k for k in range(len(TK)) if k != j]
        # forty shallow trees were tried first, to keep the map cheap enough to
        # run on every batch beside a GPU job. They cost too much: the check
        # below fell from 2.16 to 1.40 of separation on the component the arm
        # exists to move. The map is the instrument, so it gets the depth.
        reg = RandomForestRegressor(n_estimators=60, min_samples_leaf=5,
                                    n_jobs=8, random_state=42)
        reg.fit(T[:, other], T[:, j])
        regs.append((other, reg))
        rsd[j] = max((H[:, j] - reg.predict(H[:, other])).std(), 1e-9)
        print(f"  {TRAINED[j]:<24} residual sd {rsd[j]:.4f}", flush=True)
    joblib.dump({"regs": regs, "rsd": rsd}, path, compress=3)
    print(f"  wrote {path}")


def check(path):
    """Reproduce the separation the arm was designed against, using the frozen
    map rather than the one the probe fitted. Reads the generated sample both as
    it comes and with its marginals and rank correlations already matched, which
    is the part no arm has moved."""
    from scipy.stats import norm

    def normal_scores(X):
        z = np.empty_like(X)
        for k in range(X.shape[1]):
            r = np.argsort(np.argsort(X[:, k], kind="stable"), kind="stable")
            z[:, k] = norm.ppf((r + 0.5) / len(X))
        return z

    def sym_power(C, p):
        w, V = np.linalg.eigh(C)
        return (V * np.maximum(w, 1e-10) ** p) @ V.T

    def energy(A, B):
        def m(P, Q):
            d = np.sqrt(np.maximum((P ** 2).sum(1)[:, None]
                                   + (Q ** 2).sum(1)[None, :] - 2 * P @ Q.T, 0.0))
            return float(d.mean())
        return 2 * m(A, B) - m(A, A) - m(B, B)

    rm = load(path)
    G = np.load("research/w4_ar_features.npy").astype(np.float64)
    C = corpus_rows(80000)
    sub, pool, ref = C[:2500], C[2500:8500], C[8500:]

    zs = normal_scores(G)
    A = sym_power(np.corrcoef(normal_scores(ref), rowvar=False), 0.5) \
        @ sym_power(np.corrcoef(zs, rowvar=False), -0.5)
    u = np.clip(norm.cdf(zs @ A.T), 1e-6, 1 - 1e-6)
    gB = np.empty_like(u)
    for k in range(18):
        gB[:, k] = np.quantile(ref[:, k], u[:, k])

    mu, sd = ref.mean(0), ref.std(0)

    def aug(X, w):
        return np.hstack([((X - mu) / sd)[:, TK], w * rm.apply(X)])

    print(f"  {'weight':<9}{'B 768':>10}{'B 1536':>10}{'raw 768':>10}"
          f"{'raw 1536':>11}{'corpus':>10}")
    for w in (0.0, 4.0):
        Z = {k: aug(X, w) for k, X in (("B", gB), ("raw", G), ("cor", sub))}
        Zp = aug(pool, w)
        out = []
        for k, b in (("B", 768), ("B", 1536), ("raw", 768), ("raw", 1536),
                     ("cor", 1536)):
            dg, dc = [], []
            r = np.random.default_rng(7)
            for _ in range(24):
                hp = Zp[r.choice(len(Zp), 2000, replace=False)]
                dg.append(energy(Z[k][r.choice(len(Z[k]), b, replace=False)], hp))
                dc.append(energy(Z["cor"][r.choice(len(Z["cor"]), b,
                                                   replace=False)], hp))
            dg, dc = np.array(dg), np.array(dc)
            s = np.sqrt((dg.var(ddof=1) + dc.var(ddof=1)) / 2) or 1e-12
            out.append((dg.mean() - dc.mean()) / s)
        print(f"  {w:<9.1f}" + "".join(f"{v:>10.2f}" for v in out[:4])
              + f"{out[4]:>10.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("fit", "check"):
        print(__doc__)
        raise SystemExit(2)
    (fit if sys.argv[1] == "fit" else check)(sys.argv[2])
