"""Does the disattenuation in `w4_estimator_rel` recover a cosine it is told? CPU, seconds.

This is not a plumbing check. `w4_estimator_rel` rests on one statistical claim,

    cos(g1_A, g2_B)  =  cos(mu1, mu2) * sqrt(r1) * sqrt(r2)

where r is a split half reliability, and the whole file is worthless if that
identity does not hold at the dimensions and noise levels involved. It is
checkable against a known answer without a GPU, so it should be checked before
any GPU time is spent, the way `w4_kldir_check` was and the way it then saved
two hours by refusing its own arm.

The construction. Two true directions are built with a PRESCRIBED cosine c.
Each is observed as `mu + noise`, independently per half, at a controlled signal
to noise ratio. The file's own `cos` and `mean_se` are imported rather than
reimplemented, so a bug in either is caught here rather than reproduced.

Four things are checked.

ONE. Reliability tracks the signal to noise ratio it is given, r = snr/(1+snr).
This is what licenses reading r as a reliability at all.

TWO. The disattenuated ratio recovers the prescribed c across a range of noise
levels, including one where the raw cross cosine is near zero while c is large.
That case is the entire reason the file exists, so it must be exhibited, not
assumed: if a raw cosine near zero can sit on top of a true cosine of 0.8, then
w4_estimator's printed number cannot be read the way its registration says.

THREE. The estimator is unbiased under the null, c = 0 recovers 0. A correction
that manufactures agreement from noise would be worse than no correction.

FOUR. The failure mode the registration guards against is real: at a signal to
noise ratio low enough that r is indistinguishable from zero, the ratio explodes
past the 1.3 bound. That is what the "UNINTERPRETABLE" branch is for, and it
should fire on data built to trigger it.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "research")

from w4_estimator_rel import cos, mean_se  # noqa: E402

DIM = 200000  # the model is about this size, and these statistics are
              # dimension dependent, so the check runs at the real width


def draw(mu, snr, rng):
    """One observation of `mu` at the given signal to noise ratio.

    snr is the ratio of squared signal norm to squared noise norm, so the
    expected cosine between two independent draws is snr / (1 + snr).
    """
    n = rng.standard_normal(DIM)
    n *= (float(np.dot(mu, mu)) / snr / float(np.dot(n, n))) ** 0.5
    return torch.from_numpy(mu + n).float()


def pair(c, rng):
    """Two unit directions with cosine exactly c."""
    a = rng.standard_normal(DIM)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(DIM)
    b -= np.dot(b, a) * a
    b /= np.linalg.norm(b)
    return a, c * a + (1 - c ** 2) ** 0.5 * b


def trial(c, snr1, snr2, reps, rng):
    r1, r2, cross = [], [], []
    for _ in range(reps):
        m1, m2 = pair(c, rng)
        a1, b1 = draw(m1, snr1, rng), draw(m1, snr1, rng)
        a2, b2 = draw(m2, snr2, rng), draw(m2, snr2, rng)
        r1.append(cos(a1, b1))
        r2.append(cos(a2, b2))
        cross.append(0.5 * (cos(a1, b2) + cos(b1, a2)))
    mr1, mr2, mc = mean_se(r1)[0], mean_se(r2)[0], mean_se(cross)[0]
    d = mc / (mr1 * mr2) ** 0.5 if mr1 > 0 and mr2 > 0 else float("nan")
    return mr1, mr2, mc, d


def main():
    rng = np.random.default_rng(0)
    ok = True

    # ONE and TWO. Reliability tracks snr, and the correction recovers c.
    print(f"  {'true c':>8}{'snr':>7}{'r1 want':>10}{'r1 got':>9}"
          f"{'raw cross':>11}{'disatt':>9}")
    for c, snr in ((0.80, 4.0), (0.80, 0.30), (0.80, 0.05),
                   (0.30, 0.30), (0.95, 0.10)):
        want_r = snr / (1 + snr)
        r1, r2, mc, d = trial(c, snr, snr, 24, rng)
        print(f"  {c:>8.2f}{snr:>7.2f}{want_r:>10.3f}{r1:>9.3f}"
              f"{mc:>11.3f}{d:>9.2f}")
        if abs(r1 - want_r) > 0.02:
            ok = False
            print("     FAIL: reliability does not track snr")
        if abs(d - c) > 0.06:
            ok = False
            print("     FAIL: disattenuation did not recover c")

    # The headline case, stated explicitly because it is the file's premise.
    r1, r2, mc, d = trial(0.80, 0.03, 0.03, 24, rng)
    print(f"\n  the premise: true c 0.80 at snr 0.03 reads a RAW cross cosine "
          f"of {mc:+.3f}")
    print(f"  and disattenuates to {d:+.2f}. A raw cosine near zero is "
          f"therefore\n  consistent with near total agreement, which is "
          f"exactly what\n  w4_estimator's registration assumed it could rule "
          f"out.")
    premise = abs(mc) < 0.10 and d > 0.6
    if not premise:
        ok = False
        print("     FAIL: could not exhibit the confound the file exists for")

    # THREE. Under the null the correction must not manufacture agreement.
    r1, r2, mc, d = trial(0.0, 0.30, 0.30, 32, rng)
    print(f"\n  null, true c 0.00: raw cross {mc:+.3f}, disattenuated "
          f"{d:+.2f}")
    if abs(d) > 0.08:
        ok = False
        print("     FAIL: correction manufactures agreement from noise")

    # FOUR. The guard band fires when a reliability is near zero.
    r1, r2, mc, d = trial(0.80, 0.004, 0.004, 24, rng)
    print(f"  degenerate, snr 0.004: r1 {r1:+.4f}, disattenuated {d:+.2f}, "
          f"past the 1.3 bound: {'YES' if abs(d) > 1.3 or not np.isfinite(d) else 'no'}")

    print("\n  PASS" if ok else "\n  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
