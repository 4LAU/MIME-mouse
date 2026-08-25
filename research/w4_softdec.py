"""A differentiable path from token logits to contract features.

WHAT THIS IS FOR

w4_gradsnr2 showed the REINFORCE gradient carries no batch to batch signal at
batch 96, while ordinary supervised training on the same model does. REINFORCE's
variance comes from the score function term, log p times a scalar reward, which
this replaces with a pathwise gradient: differentiate the features with respect
to the token probabilities directly.

THE JOIN, AND WHY IT NEEDS NO POINT CORRESPONDENCE

Forward values come from the served decoder, unchanged, through decode_batch.
The relaxed path here supplies only the Jacobian:

    X = X_hard.detach() + X_soft - X_soft.detach()

so every number the objective sees is exactly what the contract scorer would
see, and the gradient is dX_soft/dtheta. Because the two are joined at the
eighteen features and not at the path points, the relaxed decode does not have
to reproduce the served decoder's snapping, rounding or tick merging, and does
not have to keep the same number of points. That matters: decode_batch's
docstring records that a hand rolled walk reads about 0.13 high, and the way to
respect that is to not build a second walk that has to be exact. This one is
explicitly an approximation whose only job is to point in a useful direction,
and validate() measures how well it does that.

THE RELAXATION

Each position's sampled token is replaced by a straight through one hot: the
forward value is the token that was actually sampled, the backward path goes
through the softmax over that head's logits. The decoded quantities are then
expectations under that relaxed distribution,

    speed    = y_s  @ class_to_speed table
    dtheta   = y_th @ class_to_dtheta table
    dt in ms = y_dt @ class_to_dt_ms table

which means a gradient can move probability mass between classes, including
between moving and not moving, since the speed table sends tick and pad to zero.

Geometry then follows prefix_state, which is the model's own arithmetic and not
a new derivation: heading integrates the turns, position integrates speed times
the heading's cosine and sine, time integrates the delays.

The autoregressive dependence is deliberately dropped. Logits are taken from one
teacher forced pass over the tokens that were sampled, so the gradient is
"holding this token sequence fixed, how should the probabilities that produced it
move", not "how would a different sequence have been sampled". That is the usual
straight through construction and it is what makes one backward pass enough.

WHAT IS APPROXIMATE, STATED PLAINLY

  the snap to whole pixels for sub-pixel steps      dropped in the backward
  the rounding of the final path to integer pixels  dropped in the backward
  the tick merge that removes still points          dropped in the backward
  the maxima                                        softmax weighted average
  everything downstream of a class the sampler did not pick   reached only
                                                    through the softmax

validate() prints, per feature, the correlation between the relaxed value and
the served one across a batch. A feature whose correlation is low has a Jacobian
that should not be trusted, and the trained twelve are the ones that have to
pass.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import models.event_stream_polar as esp
from models.event_ar import DT_MAX_MS, N_DT_CLASSES, class_to_dt_ms
from models.event_stream_polar import (
    N_S_CLASSES, N_TH_CLASSES, S_PAD_CLASS, TICK_CLASS, class_to_dtheta,
    class_to_speed,
)
from features import FEATURE_NAMES

HZ = 125.0
STEP = 1.0 / HZ
GRID_MAX = 512          # 4.1 s at 125 Hz, above any corpus duration
SOFT_MAX_BETA = 12.0    # in units of the row's own range, see soft_max
STILL_PX = 1e-2         # displacement below which an interval has no direction


def tables(dev):
    """The three class to value lookups as dense vectors, so a relaxed one hot
    can be contracted against them."""
    s = class_to_speed(torch.arange(N_S_CLASSES, device=dev))
    th = class_to_dtheta(torch.arange(N_TH_CLASSES, device=dev))
    dt = class_to_dt_ms(torch.arange(N_DT_CLASSES, device=dev).float())
    return s.float(), th.float(), dt.float()


def st_onehot(logits, cls, tau=1.0):
    """Straight through relaxed one hot at the token that was sampled.

    Forward is the exact one hot, so any quantity built from it equals the one
    built from the discrete token. Backward flows through the softmax.
    """
    soft = torch.softmax(logits / tau, dim=-1)
    hard = torch.zeros_like(soft).scatter_(-1, cls.unsqueeze(-1), 1.0)
    return hard + soft - soft.detach()


def soft_max(v, m, beta=SOFT_MAX_BETA):
    """Softmax weighted average, standing in for a maximum.

    The weight temperature is set from each row's own range so the same beta
    behaves the same way on a fast trajectory and a slow one.

    Masking is done with a finite offset rather than the dtype's minimum. The
    minimum overflows to negative infinity once it is scaled, and a softmax with
    an infinite input returns a clean zero forward and a NaN backward, which is
    exactly the failure this file hit on its first run.
    """
    vd = v.detach()
    big = torch.where(m, vd, torch.full_like(vd, -float("inf")))
    hi = big.max(1, keepdim=True).values
    small = torch.where(m, vd, torch.full_like(vd, float("inf")))
    lo = small.min(1, keepdim=True).values
    rng = (hi - lo).clamp(min=1e-6)
    lg = beta * (vd - hi) / rng
    lg = torch.where(m, lg, torch.full_like(lg, -1e4))
    w = torch.softmax(lg, dim=1)
    return (w * torch.where(m, v, torch.zeros_like(v))).sum(1)


def masked_mean(v, m):
    n = m.sum(1).clamp(min=1)
    return (torch.where(m, v, torch.zeros_like(v))).sum(1) / n


def masked_std(v, m):
    """Population standard deviation over the live entries.

    The floor inside the square root is not a guard against a value that cannot
    occur: a row whose entries are all identical has variance exactly zero, and
    sqrt has an infinite derivative there, so without it the gradient is NaN
    whenever a trajectory holds any quantity constant.
    """
    mu = masked_mean(v, m)
    var = masked_mean((v - mu.unsqueeze(1)) ** 2, m)
    return var.clamp(min=1e-12).sqrt()


def decode_soft(y_s, y_th, y_dt, tab, live, angle):
    """Relaxed token distributions to a path, following prefix_state's geometry.

    Returns per row point sequences (x, y, t) of length T + 1, and the point
    mask. Point 0 is the start and belongs to no token, exactly as
    decode_indexed has it.
    """
    ts, tth, tdt = tab
    B, T, _ = y_s.shape
    dev = y_s.device

    spd = y_s @ ts                                    # tick and pad send this to 0
    dth = y_th @ tth
    # the share of probability on a moving class, which is what prefix_state
    # masks the turn by
    mot = y_s[:, :, TICK_CLASS + 1:S_PAD_CLASS].sum(-1)

    spd = spd * live
    heading = angle.unsqueeze(1) + torch.cumsum(dth * mot * live, dim=1)
    dx = spd * torch.cos(heading)
    dy = spd * torch.sin(heading)

    dt_ms = (y_dt @ tdt).clamp(0.1, 1000.0)
    dt_s = (dt_ms / 1000.0) * live

    z = torch.zeros(B, 1, device=dev)
    px = torch.cat([z, torch.cumsum(dx, 1)], 1)
    py = torch.cat([z, torch.cumsum(dy, 1)], 1)
    pt = torch.cat([z, torch.cumsum(dt_s, 1)], 1)
    pm = torch.cat([torch.ones(B, 1, device=dev, dtype=torch.bool),
                    live.bool()], 1)
    return px, py, pt, pm


def resample(px, py, pt, pm):
    """Linear interpolation onto the 125 Hz grid, differentiable in the knot
    values and in the knot times.

    resample_trajectory lays the grid from t[0] to t[-1] in 1/125 steps and then
    appends t[-1] itself. Both are reproduced, because movement_duration and
    everything divided by it depend on that last interval.
    """
    B = px.shape[0]
    dev = px.device
    npts = pm.sum(1)                                    # points per row
    last = (npts - 1).clamp(min=1)
    dur = pt.gather(1, last.unsqueeze(1)).squeeze(1)     # t[0] is 0 by construction

    k = torch.arange(GRID_MAX, device=dev).float().unsqueeze(0)
    g = k * STEP
    nfull = torch.ceil(dur / STEP).long().clamp(min=1, max=GRID_MAX - 1)
    gm = k.long() <= nfull.unsqueeze(1)                  # keep one slot for the end
    # the final grid point is the trajectory's own end time, not a multiple of
    # the step, which is what resample_trajectory does when the duration does
    # not divide evenly
    g = torch.where(k.long() == nfull.unsqueeze(1), dur.unsqueeze(1),
                    g.expand(B, -1))

    # bracketing interval for each grid time. searchsorted needs the knot times
    # monotone, which cumsum of positive delays guarantees, and the index itself
    # carries no gradient, which is correct: it is piecewise constant.
    tk = pt.masked_fill(~pm, float("inf")).detach()
    j = (torch.searchsorted(tk.contiguous(), g.contiguous().clamp(min=0),
                            right=True) - 1).clamp(min=0)
    j = torch.minimum(j, (last - 1).clamp(min=0).unsqueeze(1))

    t0 = pt.gather(1, j)
    t1 = pt.gather(1, j + 1)
    w = ((g - t0) / (t1 - t0).clamp(min=1e-9)).clamp(0.0, 1.0)
    gx = px.gather(1, j) + w * (px.gather(1, j + 1) - px.gather(1, j))
    gy = py.gather(1, j) + w * (py.gather(1, j + 1) - py.gather(1, j))
    return gx, gy, g, gm


def soft_features(gx, gy, gt, gm):
    """The eighteen contract features on the resampled grid, differentiably.

    extract_features is followed term for term for the twelve that the rollout
    objective trains on. The six held out ones need an argmax, a skewness or a
    count of sign changes; they are returned so the column layout matches, but
    they are computed with relaxations that are not claimed to be faithful and
    the objective must not use them.
    """
    B = gx.shape[0]
    im = gm[:, 1:] & gm[:, :-1]                          # interval mask
    dx = gx[:, 1:] - gx[:, :-1]
    dy = gy[:, 1:] - gy[:, :-1]
    dt = (gt[:, 1:] - gt[:, :-1]).clamp(min=1e-6)
    ds = torch.sqrt(dx * dx + dy * dy + 1e-12)
    vx, vy = dx / dt, dy / dt
    speed = ds / dt

    im2 = im[:, 1:] & im[:, :-1]
    dt2 = dt[:, :-1].clamp(min=1e-6)
    acc = (speed[:, 1:] - speed[:, :-1]) / dt2
    im3 = im2[:, 1:] & im2[:, :-1]
    jerk = (acc[:, 1:] - acc[:, :-1]) / dt2[:, :-1].clamp(min=1e-6)

    npt = gm.sum(1)
    lastp = (npt - 1).clamp(min=1)
    xe = gx.gather(1, lastp.unsqueeze(1)).squeeze(1)
    ye = gy.gather(1, lastp.unsqueeze(1)).squeeze(1)
    x0, y0 = gx[:, 0], gy[:, 0]
    dstr = torch.sqrt((xe - x0) ** 2 + (ye - y0) ** 2 + 1e-12)
    dtrav = (ds * im).sum(1)
    peff = dstr / dtrav.clamp(min=1e-6)

    lx = (xe - x0).unsqueeze(1)
    ly = (ye - y0).unsqueeze(1)
    perp = (ly * (gx - x0.unsqueeze(1)) - lx * (gy - y0.unsqueeze(1))).abs() \
        / dstr.clamp(min=1e-6).unsqueeze(1)
    mdev = soft_max(perp, gm)

    axc = (vx[:, 1:] - vx[:, :-1]) / dt2
    ayc = (vy[:, 1:] - vy[:, :-1]) / dt2
    # a millipixel per second floor rather than 1e-6, because the cube of the
    # smaller one overflows float32 before the clamp below can catch it
    smid = speed[:, :-1].clamp(min=1e-3)
    cross = (vx[:, :-1] * ayc - vy[:, :-1] * axc).abs()
    curv = (cross / smid ** 3).clamp(0, 1e6)

    # The derivative of atan2 scales as one over the displacement, so a nearly
    # still interval contributes an arbitrarily large gradient from a direction
    # that carries no information. The floor is physical rather than numerical:
    # the served decoder rounds every path point to a whole pixel, so a
    # displacement of a hundredth of a pixel has no direction the contract
    # scorer can see. Below it the angle is defined as a constant, which is what
    # torch.where routes to, and no gradient flows.
    still = (dx * dx + dy * dy) < STILL_PX ** 2
    ang = torch.atan2(torch.where(still, torch.zeros_like(dy), dy),
                      torch.where(still, torch.ones_like(dx), dx))
    ad = ang[:, 1:] - ang[:, :-1]
    ad = torch.remainder(ad + math.pi, 2 * math.pi) - math.pi
    omega = (ad / dt[:, :-1].clamp(min=1e-6)).clamp(-1e6, 1e6)

    dur = gt.gather(1, lastp.unsqueeze(1)).squeeze(1) - gt[:, 0]

    # the six held out columns, relaxed only far enough to keep the layout. the
    # objective must not read these; HELD_OUT in w4_rollout is the same list.
    zero = torch.zeros(B, device=gx.device)
    out = [
        masked_mean(speed, im),
        masked_std(speed, im),
        soft_max(speed, im),
        zero,                                   # velocity_skewness, held out
        masked_mean(acc, im2),
        masked_std(acc, im2),
        soft_max(acc.abs(), im2),               # max_acceleration, held out
        masked_mean(jerk, im3),
        masked_std(jerk, im3),
        peff,
        mdev,
        masked_mean(curv, im2),
        masked_std(curv, im2),                  # curvature_std, held out
        zero,                                   # num_direction_changes, held out
        dur,
        zero,                                   # time_to_peak_velocity, held out
        masked_mean(omega.abs(), im2),
        masked_std(omega, im2),                 # angular_velocity_std, held out
    ]
    return torch.stack(out, dim=1)


def soft_forward(model, s, th, dt, cond, angle, amp=False, tau=1.0):
    """One teacher forced pass over the sampled tokens, then the relaxed decode.

    tau is the softmax temperature of the relaxation, and it is the one free
    parameter of this construction. A trained model's token distributions are
    sharp, so at tau 1 almost all the backward signal comes from the few
    positions where two classes are close; raising tau spreads it, trading bias
    for variance. Forward values do not depend on it at all, because the
    straight through one hot is exact either way.

    Returns the (B, 18) differentiable feature matrix and the live mask.
    """
    from models.event_ar import prefix_state
    with torch.amp.autocast("cuda", enabled=amp):
        s_lg, th_lg, dt_lg = model(*model.shift_inputs(s, th, dt),
                                   prefix_state(s, th, dt, cond), cond,
                                   s, th, dt)
    s_lg, th_lg, dt_lg = s_lg.float(), th_lg.float(), dt_lg.float()

    pad = s >= S_PAD_CLASS
    first = torch.where(pad.any(1), pad.float().argmax(1),
                        torch.full_like(s[:, 0], s.shape[1]))
    pos = torch.arange(s.shape[1], device=s.device).unsqueeze(0)
    live = (pos < first.unsqueeze(1)).float()

    tab = tables(s.device)
    y_s = st_onehot(s_lg, s.clamp(max=N_S_CLASSES - 1), tau)
    y_th = st_onehot(th_lg, th.clamp(max=N_TH_CLASSES - 1), tau)
    y_dt = st_onehot(dt_lg, dt.clamp(max=N_DT_CLASSES - 1), tau)

    px, py, pt, pm = decode_soft(y_s, y_th, y_dt, tab, live, angle)
    gx, gy, gt, gm = resample(px, py, pt, pm)
    return soft_features(gx, gy, gt, gm), live


def straight_through(X_hard, X_soft):
    """Exact values forward, relaxed Jacobian backward."""
    return X_hard.detach() + X_soft - X_soft.detach()
