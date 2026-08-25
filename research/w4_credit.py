"""Split a rollout's scalar return into per token credit.

Registered in HANDOFF.md under task 12 before any of it was built, and gated on
research/w4_credit_check.py, which passed on 2026-08-10. This file is the shared
implementation: w4_credit_check.py verifies it against finite differences and
w4_rollout.py trains with it.

THE DEFECT IT ADDRESSES

w4_rollout's surrogate is (w_i * mean_t logp_it).mean(). One scalar per
trajectory multiplies the average log probability over roughly 250 tokens, so
every token receives identical credit. That can move what the model does on
average and cannot move which token in which context, which is the higher order
dependence the gap ladder says is left.

THE DECOMPOSITION

The score function coefficient w_i is whatever the objective says it is, and is
not touched here. What is added is a split of it,

    w_i = sum_t v_it + r_i

where v_it is the part attributable to the tokens and r_i is the remainder. For
a return that splits over time the estimator admits the standard reward to go
reduction, sum_t (grad log p_it) * (sum_{t' >= t} v_it' + r_i), which is
unbiased and lower variance. Setting every v to zero returns the old estimator
exactly, and w4_rollout's smoke path checks that it does.

v comes from a first order split of w through the features. With g_ik the
derivative of w_i with respect to standardised feature k, held constant,

    v_it = sum_k g_ik c_ikt / sd_k

over the features whose value really is a sum over the resampled path, with
c_ikt the contribution of resampled interval t. r_i is then w_i minus the total,
which absorbs the features that do not split, the standardisation offset and the
first order error, and is applied to every token exactly as the old weight was.
Nothing is approximated away: the split is exact by construction because the
remainder is measured rather than assumed to be zero.

WHICH FEATURES SPLIT, AND THE CAVEAT FOUND WHILE REGISTERING

Nine of the twelve trained features are a mean, a standard deviation or a sum
over per interval quantities of the 125 Hz resampled path, and those nine are
split. The standard deviations split through

    c_t = z / K + ((q_t - qbar)^2 - z^2) / (2 z K)

which sums to z exactly, so no linearisation enters there either.

Three trained features stay in the remainder. max_velocity is a max and does not
split at all. path_efficiency and max_deviation read absolute geometry, and
positions are cumulative, so changing a token rigidly shifts everything after it
and the credit would be wrong rather than merely coarse. Locality survives the
rigid shift only for the differential quantities, which are speed, acceleration,
jerk, curvature and angular velocity, because a shift leaves local differences
alone. Timing tokens move the resampling grid itself and are handled only
approximately, which is registered and not hidden.

MEASURED, in w4_credit_check on 244 corpus trajectories

    mirror disagreed with esp._decode                   0
    mean_velocity, attributed against actual       0.9806 correlation
    slope through the origin                       1.1955
    path_efficiency, predicted 0, actually moves   6.694e-04 median

The 20 percent overstatement in the slope is a common scale error and the
advantage is normalised to unit standard deviation before use, so it does not
survive into the update.
"""
from __future__ import annotations

import numpy as np
import torch

import experiments.event_stream_polar as esp
from features import resample_trajectory
from models.event_stream_polar import (
    S_PAD_CLASS, TICK_CLASS, class_to_dtheta, class_to_speed,
)

HZ = 125.0

# feature name, the per interval quantity it is built from, the offset of that
# quantity's first entry within the resampled intervals, and how it aggregates.
# The offsets are the diffs: acceleration, curvature and angular velocity are
# differences of interval quantities and are credited to the LATER of the two
# intervals they span, jerk to the latest of three. Crediting forward keeps the
# reward at t a function of tokens up to t, which is what the reward to go
# reduction needs.
DECOMP = [
    ("mean_velocity", "speed", 0, "mean"),
    ("std_velocity", "speed", 0, "std"),
    ("mean_acceleration", "acc", 1, "mean"),
    ("std_acceleration", "acc", 1, "std"),
    ("mean_jerk", "jerk", 2, "mean"),
    ("std_jerk", "jerk", 2, "std"),
    ("curvature_mean", "curv", 1, "mean"),
    ("angular_velocity_mean", "aomega", 1, "mean"),
    ("movement_duration", "dt", 0, "sum"),
]


def decode_indexed(dt_z, s_cls, th_cls, sx, sy, angle):
    """esp._decode line for line, plus the token index behind each path point.

    Returns (trajectory, tok) where tok[j] is the token that produced path
    point j + 1. Point 0 is the start and belongs to no token. A hand rolled
    token walk reads about 0.13 high in this repo, so this must stay a mirror
    and never become a reimplementation. w4_credit_check compares it against
    esp._decode on every trajectory it touches.
    """
    pad = s_cls >= S_PAD_CLASS
    n = int(np.argmax(pad)) if pad.any() else len(s_cls)
    if n < 2:
        return None, None

    s = class_to_speed(torch.from_numpy(s_cls[:n].astype(np.int64))).numpy()
    dth = class_to_dtheta(torch.from_numpy(th_cls[:n].astype(np.int64))).numpy()

    motion = s_cls[:n] > TICK_CLASS
    heading = angle + np.cumsum(np.where(motion, dth, 0.0))
    dx = np.where(motion, s * np.cos(heading), 0.0)
    dy = np.where(motion, s * np.sin(heading), 0.0)
    if esp._SNAP > 0:
        slow = motion & (s > 0) & (s < esp._SNAP)
        dx = np.where(slow, np.round(dx), dx)
        dy = np.where(slow, np.round(dy), dy)

    dt_ms = np.exp(dt_z[:n] * esp._DT_STD + esp._DT_MEAN)
    dt_s = np.clip(dt_ms, 0.1, 1000.0) / 1000.0

    tok = np.arange(n)
    if esp._TICKMERGE and n >= 3:
        spd = np.hypot(dx, dy)
        mid = ((~motion[1:-1]) & (spd[:-2] >= esp._TICKMERGE_MIN)
               & (spd[2:] >= esp._TICKMERGE_MIN))
        drop = np.zeros(n, dtype=bool)
        drop[1:-1] = mid
        if drop.any():
            dt_s = dt_s.copy()
            for i in np.flatnonzero(drop):
                dt_s[i + 1] += dt_s[i]
            keep = ~drop
            dx, dy, dt_s, tok = dx[keep], dy[keep], dt_s[keep], tok[keep]
            if len(dx) < 2:
                return None, None

    x = np.concatenate([[sx], sx + np.cumsum(dx)])
    y = np.concatenate([[sy], sy + np.cumsum(dy)])
    if esp._ROUND:
        x = np.round(x)
        y = np.round(y)
    t = np.concatenate([[0.0], np.cumsum(dt_s)])
    return list(zip(x.tolist(), y.tolist(), t.tolist())), tok


def token_of_grid_point(path, grid, tok):
    """For each 125 Hz resampled interval, the token whose segment covers it.

    Interval j runs between grid point j and j + 1 and is the thing every
    differential feature is built from, so credit is assigned per interval and
    not per point.
    """
    pt = np.asarray([p[2] for p in path], dtype=np.float64)
    gt = np.asarray([p[2] for p in grid], dtype=np.float64)
    mid = 0.5 * (gt[:-1] + gt[1:])
    seg = np.clip(np.searchsorted(pt, mid) - 1, 0, len(tok) - 1)
    return tok[seg]


def interval_parts(grid):
    """The per interval quantities extract_features is built from, computed the
    same way it computes them so the split is a split of the real feature and
    not of a lookalike."""
    pts = np.asarray(grid, dtype=np.float64)
    x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
    dx, dy = np.diff(x), np.diff(y)
    dt_raw = np.diff(t)
    dt = np.maximum(dt_raw, 1e-6)
    ds = np.hypot(dx, dy)

    vx, vy = dx / dt, dy / dt
    speed = ds / dt
    dt2 = np.maximum(dt[:-1], 1e-6)
    acc = np.diff(speed) / dt2 if len(speed) > 1 else np.zeros(0)
    jerk = (np.diff(acc) / np.maximum(dt2[:-1], 1e-6) if len(acc) > 1
            else np.zeros(0))

    if len(acc):
        ax = np.diff(vx) / dt2
        ay = np.diff(vy) / dt2
        speed_mid = np.maximum(speed[:-1], 1e-6)
        curv = np.clip(np.abs(vx[:-1] * ay - vy[:-1] * ax) / speed_mid ** 3,
                       0, 1e6)
    else:
        curv = np.zeros(0)

    ang = np.arctan2(dy, dx)
    ad = np.diff(ang)
    ad = (ad + np.pi) % (2 * np.pi) - np.pi
    aomega = (np.abs(np.clip(ad / dt[:-1], -1e6, 1e6)) if len(ad)
              else np.zeros(0))

    # movement_duration is t[-1] - t[0], which is the sum of the RAW diffs. The
    # 1e-6 floor the other quantities use would change the total on the last
    # resampled interval, which resample_trajectory appends short.
    return {"speed": speed, "acc": acc, "jerk": jerk, "curv": curv,
            "aomega": aomega, "dt": dt_raw}


def contributions(q, kind):
    """Per interval pieces that sum to the feature exactly."""
    k = len(q)
    if k == 0:
        return np.zeros(0)
    if kind == "sum":
        return q
    if kind == "mean":
        return q / k
    z = float(q.std())
    c = np.full(k, z / k)
    if z > 1e-9:
        c = c + ((q - q.mean()) ** 2 - z * z) / (2.0 * z * k)
    return c


def credit_terms(path, grid, tok, n_tok):
    """(C, gtok, n_tok) for one trajectory.

    C is len(DECOMP) by number of resampled intervals, holding each feature's
    per interval contribution. gtok says which token each interval belongs to.
    """
    gtok = token_of_grid_point(path, grid, tok)
    q = interval_parts(grid)
    C = np.zeros((len(DECOMP), len(gtok)), dtype=np.float64)
    for k, (_, key, off, kind) in enumerate(DECOMP):
        c = contributions(q[key], kind)
        if len(c):
            C[k, off:off + len(c)] = c
    return C, gtok, n_tok


# ------------------------------------------------------- objective gradients

def energy_grad(zt, ht, zpool=None):
    """d w_i / d z_i for the energy distance weight, matching the w computed in
    w4_rollout term for term:

        w_i = (2/m) sum_j ||z_i - h_j|| - (2/n) sum_k ||z_i - z_k||

    The n by m by d difference tensor is never formed. The derivative of a
    Euclidean norm sum collapses to z_i times a sum of reciprocals minus a
    reciprocal weighted average of the other side.

    zpool is the sample standing in for a second independent generated draw.
    None means the batch itself, which is what w4_rollout does unless it is
    given more than one step of buffer.
    """
    if zpool is None:
        zpool = zt
    n, m = zpool.shape[0], len(ht)
    inv_h = 1.0 / torch.cdist(zt, ht).clamp(min=1e-8)
    g = (2.0 / m) * (zt * inv_h.sum(1, keepdim=True) - inv_h @ ht)
    d_zz = torch.cdist(zt, zpool)
    inv_z = torch.where(d_zz > 1e-8, 1.0 / d_zz.clamp(min=1e-8),
                        torch.zeros_like(d_zz))
    return g - (2.0 / n) * (zt * inv_z.sum(1, keepdim=True) - inv_z @ zpool)


def moment_grad(zt):
    """d w_i / d z_i for the mean and log spread weight, with the batch mean and
    spread held constant, matching w4_rollout's w term for term."""
    m = zt.mean(0)
    sdev = zt.std(0).clamp(min=1e-4)
    return 2 * m + 2 * (torch.log(sdev) / sdev ** 2) * (zt - m)


# ------------------------------------------------------------- the advantage

def token_advantage(w, g_dec, sd_dec, cred, n_pos):
    """Reward to go per token, plus the unattributed remainder on every token.

    w         (n,)        the score function coefficient, unchanged
    g_dec     (n, 9)      d w_i / d z_ik for the nine features that split
    sd_dec    (9,)        their standardisation scales, raw units
    cred      list of n   (C, gtok, n_tok) from credit_terms
    n_pos     int         width of the token buffer

    Returns (A, live) as numpy, both n by n_pos. A[i, t] multiplies the log
    probability of token t of row i. Where every v is zero this is w_i on every
    live position, which is the old estimator.
    """
    n = len(w)
    A = np.zeros((n, n_pos), dtype=np.float64)
    live = np.zeros((n, n_pos), dtype=bool)
    coef = g_dec / sd_dec[None, :]
    for i in range(n):
        C, gtok, n_tok = cred[i]
        v = coef[i] @ C
        vtok = np.zeros(n_tok, dtype=np.float64)
        np.add.at(vtok, gtok, v)
        rtg = np.cumsum(vtok[::-1])[::-1]
        A[i, :n_tok] = rtg
        # the PAD decision at index n_tok is live and carries no path, so its
        # reward to go is empty and it sees the remainder only
        stop = min(n_tok + 1, n_pos)
        A[i, :stop] += w[i] - float(vtok.sum())
        live[i, :stop] = True
    return A, live
