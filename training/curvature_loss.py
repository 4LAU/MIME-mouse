"""Batch-level curvature-moment auxiliary loss for CANDI (opt-in, pilot).

Diagnosis this targets: synthetic paths realize only ~0.55-0.60 of the human
ACROSS-PATH spread of per-path curvature mean/std (see DIFFUSION_PILOT.md).
That is a whole-path statistic a per-token objective cannot see, but a
diffusion model can, because it predicts a clean sample x0_hat at every
training step.

Given x0_hat for a training batch (still in the polar, per-feature-scaled
representation the model is trained on), this module:
  1. reconstructs a cartesian path (heading = cumsum(delta_heading), same
     inverse transform the samplers use),
  2. computes a differentiable per-path curvature series by finite-differencing
     the *velocity direction* angle (bounded via atan2, mirroring features.py's
     atan2(dy, dx) + wrapped angle_diff, rather than replaying its non-smooth
     cross-product formula), respecting the pad mask,
  3. compares the four across-path moments (mean-of-mean, std-of-mean,
     mean-of-std, std-of-std) of the synthetic batch to the same four moments
     computed on the matched human batch from the same training step.

Everything here is inert unless a caller opts in; train_candi.py wires this
in behind --curv-loss-weight, which defaults to 0.0.
"""
from __future__ import annotations

import torch

_EPS = 1e-6


def reconstruct_x0(pred, x_t, t_or_tcont, model, pred_type):
    """Recover the model's predicted clean sample x0_hat, gradients intact.

    Mirrors the x0-reconstruction conventions already in this codebase
    (train_candi.py's _corr_loss/_path_loss, experiments/candi.py's guided
    samplers) rather than re-deriving them from scratch:

      - "flow": the forward process is x_t = (1 - t) * x0 + t * noise with
        velocity = noise - x0, so x_t = x0 + t * velocity  =>
        x0_hat = x_t - t * v_pred. `t_or_tcont` here is the continuous
        [0, 1] fraction (t_cont), NOT the diffusion-step-scaled value that
        gets passed into the model's forward() as `t`.
      - "x0": the model output already is x0_hat.
      - "eps": x0_hat = (x_t - sqrt(1 - abar) * eps_pred) / sqrt(abar).
      - "v": x0_hat = sqrt(abar) * x_t - sqrt(1 - abar) * v_pred.
    """
    if pred_type == "flow":
        t = t_or_tcont.view(-1, 1, 1)
        return x_t - t * pred
    if pred_type == "x0":
        return pred
    if pred_type == "eps":
        s_ab = model.sqrt_ab[t_or_tcont].view(-1, 1, 1)
        s_1mab = model.sqrt_1mab[t_or_tcont].view(-1, 1, 1)
        return (x_t - s_1mab * pred) / s_ab.clamp(min=_EPS)
    # "v"
    s_ab = model.sqrt_ab[t_or_tcont].view(-1, 1, 1)
    s_1mab = model.sqrt_1mab[t_or_tcont].view(-1, 1, 1)
    return s_ab * x_t - s_1mab * pred


def _path_curvature_moments(x0_polar, pad_mask, spd_scale, dh_scale, hz=125.0, eps=_EPS):
    """Per-path curvature_mean / curvature_std from a polar x0 tensor.

    x0_polar: (B, T, 2), normalized as (speed * spd_scale, dh * dh_scale) --
    the same convention CANDIDataset feeds the model, so both the synthetic
    x0_hat and the clean human batch can be passed through unchanged.

    Returns (path_mean, path_std, path_valid), each (B,). path_valid marks
    paths with >= 3 real points (>= 2 curvature steps); shorter paths are
    excluded from the moments rather than allowed to produce NaN/0 std.
    """
    mask = pad_mask.float()
    speed = torch.clamp(x0_polar[..., 0] / spd_scale, min=0.0) * mask
    dh = x0_polar[..., 1] / dh_scale * mask

    heading = torch.cumsum(dh, dim=1)
    vx = speed * torch.cos(heading)
    vy = speed * torch.sin(heading)

    # Bounded per-step turn angle: atan2 of the velocity direction wraps into
    # [-pi, pi] automatically, and the diff-of-atan2s trick wraps the turn
    # itself -- this is the "atan2 on deltas + wraparound" features.py mirrors,
    # applied to the reconstructed cartesian path rather than the model's raw
    # (potentially unbounded, since it is a regression target) dh channel.
    step_angle = torch.atan2(vy, vx)
    dtheta = step_angle[:, 1:] - step_angle[:, :-1]
    dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))

    step_valid = mask[:, 1:] * mask[:, :-1]
    dt = 1.0 / hz
    arc_len = speed[:, 1:] * dt
    curvature = (dtheta.abs() / (arc_len + eps)) * step_valid

    n_valid = step_valid.sum(dim=1)
    path_sum = curvature.sum(dim=1)
    path_mean = path_sum / n_valid.clamp(min=1.0)

    path_sqsum = (curvature ** 2).sum(dim=1)
    path_var = path_sqsum / n_valid.clamp(min=1.0) - path_mean ** 2
    path_std = torch.sqrt(path_var.clamp(min=eps))

    path_valid = n_valid >= 2
    return path_mean, path_std, path_valid


def _batch_moments(path_mean, path_std, path_valid, eps=_EPS):
    if path_valid.sum() < 2:
        z = path_mean.sum() * 0.0
        return z, z, z, z
    pm = path_mean[path_valid]
    ps = path_std[path_valid]
    return (
        pm.mean(),
        pm.std(unbiased=False).clamp(min=eps),
        ps.mean(),
        ps.std(unbiased=False).clamp(min=eps),
    )


def curvature_moment_loss(x0_hat_polar, human_polar, pad_mask, spd_scale, dh_scale,
                           hz=125.0, eps=_EPS):
    """Batch-level curvature-moment auxiliary loss.

    Compares the four across-path curvature moments (mean-of-mean,
    std-of-mean, mean-of-std, std-of-std) of the synthetic x0_hat batch
    against the same four moments computed on the matched human batch from
    the same training step (no extra data loading: it's the batch's own
    clean targets). The across-path std terms are the ones that encode the
    diagnosed variety deficit.

    Returns (loss, stats): loss is a scalar tensor with gradients flowing
    back through x0_hat_polar (human side has no grad -- it's the target
    and callers should pass the clean batch tensor, no .detach() needed but
    harmless either way since it isn't the tensor requiring grad). stats is
    a plain dict of python floats for logging.
    """
    s_mean, s_std_of_mean, s_mean_of_std, s_std_of_std = _batch_moments(
        *_path_curvature_moments(x0_hat_polar, pad_mask, spd_scale, dh_scale, hz, eps),
        eps=eps,
    )
    h_mean, h_std_of_mean, h_mean_of_std, h_std_of_std = _batch_moments(
        *_path_curvature_moments(human_polar, pad_mask, spd_scale, dh_scale, hz, eps),
        eps=eps,
    )

    # Each term is normalized by the human value squared, so the loss is a
    # dimensionless sum of four squared relative errors. At convergence each
    # term is O(1) or below, which makes weights in the 0.1-1.0 range mean
    # "aux comparable to base loss".
    def _rel_sq(a, b):
        return ((a - b) / (b.abs() + eps)) ** 2

    loss = (
        _rel_sq(s_mean, h_mean)
        + _rel_sq(s_std_of_mean, h_std_of_mean)
        + _rel_sq(s_mean_of_std, h_mean_of_std)
        + _rel_sq(s_std_of_std, h_std_of_std)
    )

    stats = {
        "mean_of_mean_syn": float(s_mean.detach()),
        "mean_of_mean_hum": float(h_mean.detach()),
        "std_of_mean_syn": float(s_std_of_mean.detach()),
        "std_of_mean_hum": float(h_std_of_mean.detach()),
        "mean_of_std_syn": float(s_mean_of_std.detach()),
        "mean_of_std_hum": float(h_mean_of_std.detach()),
        "std_of_std_syn": float(s_std_of_std.detach()),
        "std_of_std_hum": float(h_std_of_std.detach()),
    }
    # The pilot's pass gate reads directly off these two: the across-path
    # variety of per-path curvature std (primary, gate is >= 0.8 of human)
    # and of per-path curvature mean.
    stats["variety_ratio_std"] = stats["std_of_std_syn"] / max(stats["std_of_std_hum"], eps)
    stats["variety_ratio_mean"] = stats["std_of_mean_syn"] / max(stats["std_of_mean_hum"], eps)
    return loss, stats
