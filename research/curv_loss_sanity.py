"""CPU sanity check for the opt-in curvature-moment auxiliary loss.

Builds a small CANDI model (real architecture, reduced dims), loads a small
real batch of human paths from the zimt_* mmap arrays (used by
training/train_candi.py), then runs a handful of flow-matching training
steps twice from an identical starting point: once with
--curv-loss-weight-equivalent 0.0 (current behavior) and once with it > 0.

Demonstrates, with real printed numbers:
  1. curv_loss is finite and nonzero,
  2. gradients flow through it (total grad norm differs between the two
     runs even though every other input -- batch, noise, model weights -- is
     identical),
  3. no NaNs anywhere along the way.

CPU only. Does not touch data/human_eval_features.npy.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from models.candi import CANDIModel
from training.curvature_loss import curvature_moment_loss, reconstruct_x0

torch.manual_seed(0)
np.random.seed(0)

DATA_DIR = "training"
BATCH_SIZE = 32
MAX_LEN = 48
N_STEPS_TRAIN = 5
DISC_WEIGHT = 1.0
CURV_WEIGHT = 0.5


def load_tiny_batch(data_scale=None):
    lengths = np.load(f"{DATA_DIR}/zimt_lengths.npy")
    spd = np.load(f"{DATA_DIR}/zimt_polar_spd.npy", mmap_mode="r")
    dh = np.load(f"{DATA_DIR}/zimt_polar_dh.npy", mmap_mode="r")
    stall = np.load(f"{DATA_DIR}/zimt_stall.npy", mmap_mode="r")
    cond = np.load(f"{DATA_DIR}/zimt_conditions.npy")

    # A few dozen real trajectories, long enough to give the curvature loss
    # something to chew on (>= 3 points), short enough to stay fast on CPU.
    rng = np.random.default_rng(0)
    eligible = np.nonzero(lengths >= 10)[0]
    idx = rng.choice(eligible, size=BATCH_SIZE, replace=False)

    out = np.zeros((BATCH_SIZE, MAX_LEN, 2), dtype=np.float32)
    stall_out = np.zeros((BATCH_SIZE, MAX_LEN), dtype=np.float32)
    mask_out = np.zeros((BATCH_SIZE, MAX_LEN), dtype=np.float32)

    if data_scale is None:
        # Data scale computed on this tiny subset only (a real training run
        # computes it over up to 50k samples; here it just needs to be a
        # reasonable normalization for the sanity check).
        all_spd, all_dh = [], []
        for i, j in enumerate(idx):
            L = min(int(lengths[j]), MAX_LEN)
            all_spd.append(np.asarray(spd[j, :L]))
            all_dh.append(np.asarray(dh[j, :L]))
        spd_std = float(np.std(np.concatenate(all_spd)))
        dh_std = float(np.std(np.concatenate(all_dh)))
        data_scale = np.array([1.0 / spd_std, 1.0 / dh_std], dtype=np.float32)
    else:
        data_scale = np.asarray(data_scale, dtype=np.float32)

    for i, j in enumerate(idx):
        L = min(int(lengths[j]), MAX_LEN)
        out[i, :L, 0] = np.asarray(spd[j, :L]) * data_scale[0]
        out[i, :L, 1] = np.asarray(dh[j, :L]) * data_scale[1]
        stall_out[i, :L] = np.asarray(stall[j, :L]).astype(np.float32)
        mask_out[i, :L] = 1.0

    cond_out = cond[idx].astype(np.float32)

    return (
        torch.from_numpy(out),
        torch.from_numpy(stall_out),
        torch.from_numpy(mask_out).bool(),
        torch.from_numpy(cond_out),
        torch.tensor(data_scale, dtype=torch.float32),
    )


def build_tiny_model():
    return CANDIModel(
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_seq_len=MAX_LEN,
        cond_dim=4,
        n_diffusion_steps=100,
        cond_dropout=0.1,
        dropout=0.0,
    )


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def run(model, optimizer, batch, curv_weight, curv_scale_t, seeds, tag):
    dxdy_b, stall_b, pad_b, cond_b, _ = batch
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    print(f"\n--- run: {tag} (curv_weight={curv_weight}) ---")
    for step in range(len(seeds)):
        torch.manual_seed(seeds[step])
        B = dxdy_b.shape[0]
        t_cont = torch.rand(B)
        n_steps_model = model.n_steps
        dxdy_noisy, noise, velocity = model.q_flow(dxdy_b, t_cont)
        t_int = (t_cont * (n_steps_model - 1)).long()
        stall_masked, disc_mask = model.q_discrete(stall_b, t_int)
        t_for_model = t_cont * (n_steps_model - 1)

        dxdy_pred, stall_logit = model(
            dxdy_noisy, stall_masked, disc_mask.float(), t_for_model, cond_b, pad_b,
        )

        pad_f = pad_b.float().unsqueeze(-1)
        cont_loss = ((dxdy_pred - velocity) ** 2 * pad_f).sum() / pad_f.sum().clamp(min=1)

        disc_loss_raw = bce(stall_logit, stall_b)
        disc_w = disc_mask.float() * pad_b.float()
        disc_loss = (disc_loss_raw * disc_w).sum() / disc_w.sum().clamp(min=1)

        base_loss = cont_loss + DISC_WEIGHT * disc_loss

        x0_hat = reconstruct_x0(dxdy_pred, dxdy_noisy, t_cont, model, "flow")
        curv_loss, stats = curvature_moment_loss(
            x0_hat, dxdy_b, pad_b, curv_scale_t[0], curv_scale_t[1],
        )

        curv_finite = torch.isfinite(curv_loss).item()
        total_loss = base_loss + (curv_weight * curv_loss if curv_finite else 0.0 * curv_loss)

        # Isolate the gradient contribution of curv_loss alone (retain_graph
        # so the combined backward below can still run from the same graph).
        optimizer.zero_grad()
        (curv_weight * curv_loss).backward(retain_graph=True)
        curv_only_gnorm = grad_norm(model)

        optimizer.zero_grad()
        total_loss.backward()
        total_gnorm = grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        any_nan = (not torch.isfinite(base_loss).item()) or (not curv_finite)
        print(
            f"  step {step}: base_loss={base_loss.item():.5f} "
            f"curv_loss={curv_loss.item():.5f} (finite={curv_finite}) "
            f"total_loss={total_loss.item():.5f} "
            f"grad_norm(total)={total_gnorm:.5f} "
            f"grad_norm(curv_only)={curv_only_gnorm:.5f} "
            f"nan_detected={any_nan}"
        )
        print(
            f"           moments: mean-of-mean syn={stats['mean_of_mean_syn']:.5f} "
            f"hum={stats['mean_of_mean_hum']:.5f} | "
            f"std-of-mean syn={stats['std_of_mean_syn']:.5f} hum={stats['std_of_mean_hum']:.5f} | "
            f"mean-of-std syn={stats['mean_of_std_syn']:.5f} hum={stats['mean_of_std_hum']:.5f} | "
            f"std-of-std syn={stats['std_of_std_syn']:.5f} hum={stats['std_of_std_hum']:.5f}"
        )
        print(
            f"           variety_ratio_std={stats['variety_ratio_std']:.4f} "
            f"variety_ratio_mean={stats['variety_ratio_mean']:.4f} (gate >= 0.8)"
        )


def run_real_checkpoint():
    """One CPU training step from candi_polar_flow_best.pt.

    The tiny random-init model above is 5-19x off human in relative terms, so
    its curv_loss and grad norms are dominated by that garbage regime. The
    pilot's attempt 1 fine-tunes from this checkpoint, where syn moments sit
    near human -- this section shows the base-vs-aux gradient scales in the
    regime weight tuning actually operates in.
    """
    ckpt_path = f"{DATA_DIR}/candi_polar_flow_best.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(f"\n--- real checkpoint run skipped ({ckpt_path} not found) ---")
        return
    model = CANDIModel(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()
    data_scale = np.asarray(ckpt["data_scale"], dtype=np.float32)
    batch = load_tiny_batch(data_scale=data_scale)
    curv_scale_t = batch[4]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    run(model, optimizer, batch, curv_weight=CURV_WEIGHT, curv_scale_t=curv_scale_t,
        seeds=[2000], tag=f"real ckpt (epoch {ckpt.get('epoch')}), weight={CURV_WEIGHT}, 1 step")


def main():
    batch = load_tiny_batch()
    curv_scale_t = batch[4]

    model = build_tiny_model()
    init_state = copy.deepcopy(model.state_dict())

    seeds = [1000 + i for i in range(N_STEPS_TRAIN)]

    optimizer_a = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    run(model, optimizer_a, batch, curv_weight=0.0, curv_scale_t=curv_scale_t,
        seeds=seeds, tag="weight=0.0 (inert, current behavior)")

    model.load_state_dict(init_state)
    optimizer_b = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    run(model, optimizer_b, batch, curv_weight=CURV_WEIGHT, curv_scale_t=curv_scale_t,
        seeds=seeds, tag=f"weight={CURV_WEIGHT} (aux loss active)")

    run_real_checkpoint()

    print("\nSanity check complete.")


if __name__ == "__main__":
    main()
