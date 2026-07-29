"""Train the P3 pinned-endpoint inpainting flow model.

Data: the 3.7M-path 192-slot coordinate grid built by
prepare_training_data.py (train_positions.npy etc). Paths whose grid was
subsampled (n_real == 192, about 4 percent, durations above 1.53 s) are
excluded so every kept path has a uniform 8 ms step and the timestamps
can be reconstructed at serving time without a timing channel.

Per sample the inpainting mask marks slot 0, slot n_real - 1, and the
padding region as known; the model learns to fill the interior. The
stall bit for slot i is exact coordinate equality with slot i - 1,
corrupted with a CANDI-style absorbing mask and supervised with BCE on
the corrupted interior slots.

Usage:
  python -m training.train_inpaint_flow --max-steps 25000
  python -m training.train_inpaint_flow --timing   (200-step burst, exits)
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.inpaint_flow import InpaintFlowModel

_DIR = Path(__file__).resolve().parent
# Grid arrays can live on a faster disk than the repo (WSL: /mnt/c mmap
# random reads are slow); point MIME_GRID_DIR at a local copy.
_DATA_DIR = Path(os.environ.get("MIME_GRID_DIR", _DIR))
N_SLOTS = 192


class GridDataset(Dataset):
    # The memmaps are opened lazily, per process. Holding them as attributes
    # makes the dataset unpicklable under the spawn start method (Windows),
    # where DataLoader workers receive the dataset by pickle.
    def __init__(self, split: str = "train"):
        self.split = split
        self._positions = None
        self._conditions = None
        n_real = np.load(_DATA_DIR / f"{split}_n_real.npy")
        self.n_real = n_real
        self.valid = np.flatnonzero((n_real >= 5) & (n_real < N_SLOTS))

    @property
    def positions(self):
        if self._positions is None:
            self._positions = np.load(
                _DATA_DIR / f"{self.split}_positions.npy", mmap_mode="r")
        return self._positions

    @property
    def conditions(self):
        if self._conditions is None:
            self._conditions = np.load(
                _DATA_DIR / f"{self.split}_conditions.npy", mmap_mode="r")
        return self._conditions

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, i):
        idx = self.valid[i]
        pos = np.array(self.positions[idx], dtype=np.float32)
        cond = np.array(self.conditions[idx], dtype=np.float32)
        n = int(self.n_real[idx])

        known = np.zeros(N_SLOTS, dtype=np.float32)
        known[0] = 1.0
        known[n - 1:] = 1.0

        d = np.diff(pos, axis=0)
        stall = np.zeros(N_SLOTS, dtype=np.float32)
        stall[1:] = ((d[:, 0] == 0) & (d[:, 1] == 0)).astype(np.float32)

        interior = np.zeros(N_SLOTS, dtype=np.float32)
        interior[1:n - 1] = 1.0

        pad = np.zeros(N_SLOTS, dtype=bool)
        pad[n:] = True
        return (torch.from_numpy(pos), torch.from_numpy(cond),
                torch.from_numpy(known), torch.from_numpy(stall),
                torch.from_numpy(interior), torch.from_numpy(pad))


def lr_at(step, total, base, warmup=500):
    if step < warmup:
        return base * step / warmup
    import math
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * p))


def run_batch(model, batch, device, amp=False):
    pos, cond, known, stall, interior, pad = [b.to(device) for b in batch]
    B = pos.shape[0]
    t = torch.rand(B, device=device)
    x_t, _, v_target = InpaintFlowModel.q_flow(pos, t)

    mask_prob = t.view(-1, 1)
    corrupt = torch.rand_like(stall) < mask_prob
    stall_state = stall.clone()
    stall_state[corrupt] = InpaintFlowModel.STALL_MASK

    # Forward in bf16 when enabled; losses always in fp32. bf16 has the same
    # exponent range as fp32, so no gradient scaler is needed.
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        v_pred, sl = model(x_t, pos * known.unsqueeze(-1), known, stall_state,
                           t, cond, pad_mask=pad)
    v_pred = v_pred.float()
    sl = sl.float()

    w = interior.unsqueeze(-1)
    loss_v = ((v_pred - v_target) ** 2 * w).sum() / w.sum().clamp(min=1.0)
    bce_mask = interior * corrupt.float()
    loss_s = (F.binary_cross_entropy_with_logits(sl, stall, reduction="none")
              * bce_mask).sum() / bce_mask.sum().clamp(min=1.0)
    return loss_v, loss_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stall-weight", type=float, default=1.0)
    ap.add_argument("--save-name", default="inpaint_flow_v1.pt")
    ap.add_argument("--timing", action="store_true",
                    help="run 200 steps, report speed, exit without saving")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true",
                    help="disable TF32 + bf16 autocast (fp32 reference run)")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    if amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = InpaintFlowModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[p3] params {n_params/1e6:.2f}M device {device} "
          f"amp {'bf16+tf32' if amp else 'off'}", flush=True)

    train_ds = GridDataset("train")
    val_ds = GridDataset("val")
    print(f"[p3] train {len(train_ds):,} val {len(val_ds):,} "
          f"(excluded n_real==192 rows)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = 200 if args.timing else args.max_steps
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)

    best_val = float("inf")
    step = 0
    t0 = time.time()
    while step < total:
        for batch in loader:
            step += 1
            if step > total:
                break
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, total, args.lr)
            loss_v, loss_s = run_batch(model, batch, device, amp)
            loss = loss_v + args.stall_weight * loss_s
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                el = time.time() - t0
                print(f"[p3] step {step}/{total} loss_v {loss_v.item():.4f} "
                      f"loss_s {loss_s.item():.4f} {step/el:.1f} it/s", flush=True)

            if not args.timing and step % args.val_every == 0:
                vl = validate(model, val_ds, device, args.batch, amp)
                print(f"[p3] step {step} val_loss {vl:.4f}", flush=True)
                if vl < best_val:
                    best_val = vl
                    torch.save({"model_state_dict": model.state_dict(),
                                "step": step, "val_loss": vl},
                               _DIR / args.save_name)
                    print(f"[p3] new best -> {args.save_name}", flush=True)

    if args.timing:
        el = time.time() - t0
        print(f"[p3] TIMING: 200 steps in {el:.1f}s = {200/el:.2f} it/s; "
              f"25k steps ~= {25000/(200/el)/60:.0f} min", flush=True)
    else:
        vl = validate(model, val_ds, device, args.batch, amp)
        if vl < best_val:
            torch.save({"model_state_dict": model.state_dict(),
                        "step": step, "val_loss": vl}, _DIR / args.save_name)
        print(f"[p3] done. best_val {min(best_val, vl):.4f}", flush=True)


def validate(model, val_ds, device, batch, amp=False, n_batches=8):
    model.eval()
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(val_ds, batch_size=batch, shuffle=True, generator=g)
    tot, n = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= n_batches:
                break
            lv, ls = run_batch(model, b, device, amp)
            tot += (lv + ls).item()
            n += 1
    model.train()
    return tot / max(n, 1)


if __name__ == "__main__":
    main()
