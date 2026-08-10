"""Fine tune the event AR trunk with the eighteen contract features added to
its conditioning.

The idea in one paragraph. The trunk is told four numbers: how far to go, how
long to take, and in which direction. Everything else about the movement, how
curved it is, how jerky, how many direction changes, is something it has to
invent by accumulating 256 correct per event conditionals. Cross entropy gives
it no help with that: a tiny per step error is invisible to the loss and
enormous in a whole path aggregate, which is precisely what the eighteen
contract features are. So the trunk is asked to reproduce a global property
through a purely local objective. This changes that. The whole path character
becomes an input rather than something to be accumulated.

Why this is not a repeat of the closed feature conditioning work. The older FLOW
family was feature conditioned and HANDOFF's 2026-07-27 section prices its
obedience at a mean commanded to realized correlation of 0.41, with curvature
carrying no variety at all to steer. That same section closes by saying the only
emitter left which could pass is a learned autoregressive model over integer
steps, which is what this trunk is, and `research/w4_disp.py` shows this trunk
has ABUNDANT curvature variety, about 1.6 times the human median and 4 times at
the 95th percentile. The capability failure that killed the flow version is not
present here. Whether the steering failure is present here too is the question.

The ceiling. Measured 2026-08-07 on a random 4000 row sample: decoding real
human tokens through the serving decoder and scoring them against the human
validation features gives 0.5163, reproducing the 0.5118 in HANDOFF. So a
perfectly obedient model commanded from this distribution lands near 0.516
against the 0.6449 served today. That is the entire remaining gap.

The label. `training/prepare_event_features.py` writes the eighteen features of
the path the SERVING DECODER renders from each human's own tokens, not of the
original polled path. Commanding anything the decoder cannot render would be
commanding disobedience.

The representation. The eighteen features span five orders of magnitude and
several are brutally heavy tailed, curvature_mean has a 5th percentile of 0.003
against a mean of 994. Feeding raw numbers to a linear layer would let two
features own the whole conditioning vector. Each is therefore mapped through its
own empirical quantile function to a standard normal, and the knots are stored
in the checkpoint so serving applies exactly the same map.

The flag. Column 22 is 1 when the eighteen are present and 0 when they are
dropped, and the eighteen are zeroed whenever it is 0. Without the flag a
dropped command is indistinguishable from a genuine command of all medians.
Dropout keeps an unconditioned mode alive and leaves classifier free guidance
available; HANDOFF says guidance hurt the flow model, so it is not part of the
plan, only kept reachable.

Safety. Trains a NEW checkpoint. Never writes training/candi_polar_flow_best.pt
or event_ar_v2_s40000.pt. Reads no evaluation data. Local GPU.

Run:
    env NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW \
        AVX512DQ AVX512VL" PYTHONPATH=. ~/venvs/mime/bin/python -u \
        training/train_event_ar_featcond.py --steps 10000 \
        --init training/event_ar_v2_s40000.pt --save-name event_ar_fc.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.event_ar import (  # noqa: E402
    DT_MAX_MS, EventARModel, N_DT_CLASSES, STATE_DIM, prefix_state,
)
from models.event_stream_polar import (  # noqa: E402
    N_S_CLASSES, N_TH_CLASSES, S_PAD_CLASS, TH_NULL_CLASS,
    dth_lattice_to_class, s2_to_class,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
from w4_latent import gpu_c, throttle  # noqa: E402

N_FEAT = 18
COND_DIM = 4 + N_FEAT + 1          # geometry, features, present flag
N_KNOTS = 4096
GAUSS_CLIP = 3.5


def fit_knots(Fm, rng):
    """Per feature quantile knots. `np.interp` against these and the matching
    normal quantiles is the whole transform, in both directions."""
    n = min(len(Fm), 400_000)
    sub = Fm[rng.choice(len(Fm), n, replace=False)] if len(Fm) > n else Fm
    p = np.linspace(0.0, 1.0, N_KNOTS)
    return np.quantile(sub, p, axis=0).astype(np.float32)      # (N_KNOTS, 18)


def to_gauss(Fm, knots):
    """Empirical rank of each value, mapped to a standard normal quantile.

    Ties at the extremes are handled by the clip: a value at or beyond the
    largest knot maps to GAUSS_CLIP rather than to infinity.
    """
    p = np.linspace(0.0, 1.0, N_KNOTS)
    z = np.empty_like(Fm, dtype=np.float32)
    from scipy.special import ndtri
    lo, hi = 0.5 / N_KNOTS, 1.0 - 0.5 / N_KNOTS
    for j in range(Fm.shape[1]):
        k = knots[:, j]
        # knots are nondecreasing but can repeat; np.interp needs strict
        # monotonicity only for the inverse, which this direction is not
        u = np.interp(Fm[:, j], k, p, left=0.0, right=1.0)
        z[:, j] = ndtri(np.clip(u, lo, hi))
    return np.clip(z, -GAUSS_CLIP, GAUSS_CLIP).astype(np.float32)


class FCDataset(Dataset):
    """`ARDataset` from train_event_ar.py with the feature block appended to the
    condition vector and a dropout flag.

    MEMORY, and the reason this class holds paths rather than arrays.

    Python 3.14 changed the default multiprocessing start method on Linux from
    fork to forkserver. A dataloader worker therefore no longer inherits the
    parent's pages copy on write. It receives a PICKLED COPY of the dataset. Any
    array reachable from `self` is serialised in full, once per worker, and
    numpy pickles a memory map by reading the whole file into bytes, so mapping
    a file instead of loading it buys nothing at all across that boundary.

    The corpus is 8.2 GB. `train_event_ar.py` hands the dataset `s2[idx]` and
    two more like it, about 2.3 GB of real arrays, and asks for six workers.
    That is what killed this distro on 2026-08-06 and 2026-08-07, and it killed
    a 20,000 row smoke test at 8.2 GB resident on 2026-08-09, which is how it
    was finally caught.

    So nothing large is stored on the instance. The maps are opened lazily on
    first use, inside whichever process is doing the asking, and `_arr` pickles
    as None. Only the row index and the feature block cross to the workers.
    """

    def __init__(self, data_dir, rows, zfeat, max_len, drop_p, seed):
        self.data_dir = Path(data_dir)
        self.rows, self.zfeat = rows, zfeat
        self.max_len, self.drop_p = max_len, drop_p
        self.rng = np.random.default_rng(seed)
        self._arr = None

    @property
    def arrays(self):
        if self._arr is None:
            self._arr = {k: np.load(self.data_dir / f"events_{k}.npy",
                                    mmap_mode="r")
                         for k in ("s2", "dth", "dt", "len", "cond")}
        return self._arr

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        arr = self.arrays
        idx = int(self.rows[j])
        T = self.max_len
        L = min(int(arr["len"][idx]), T)

        s2 = torch.from_numpy(arr["s2"][idx, :L].astype(np.int64))
        dth = torch.from_numpy(arr["dth"][idx, :L].astype(np.int64))

        s_cls = torch.full((T,), S_PAD_CLASS, dtype=torch.long)
        s_cls[:L] = s2_to_class(s2)
        th_cls = torch.full((T,), TH_NULL_CLASS, dtype=torch.long)
        th_cls[:L] = torch.where(s2 > 0, dth_lattice_to_class(dth),
                                 torch.full_like(dth, TH_NULL_CLASS))
        dt_cls = torch.zeros(T, dtype=torch.long)
        dt_ms = torch.from_numpy(arr["dt"][idx, :L].astype(np.float32))
        dt_cls[:L] = torch.round(dt_ms).long().clamp(0, DT_MAX_MS)

        keep = float(self.rng.random() >= self.drop_p)
        cond = np.zeros(COND_DIM, dtype=np.float32)
        cond[:4] = arr["cond"][idx]
        cond[4:4 + N_FEAT] = self.zfeat[j] * keep
        cond[-1] = keep

        n_sup = min(L + 1, T)
        return (s_cls, th_cls, dt_cls, torch.tensor(n_sup),
                torch.from_numpy(cond))


def batch_losses(model, batch, device, amp):
    s_cls, th_cls, dt_cls, n_sup, cond = (x.to(device, non_blocking=True)
                                          for x in batch)
    B, T = s_cls.shape
    s_prev, th_prev, dt_prev = model.shift_inputs(s_cls, th_cls, dt_cls)
    with torch.no_grad():
        state = prefix_state(s_cls, th_cls, dt_cls, cond)

    with torch.amp.autocast("cuda", enabled=amp):
        s_logits, th_logits, dt_logits = model(
            s_prev, th_prev, dt_prev, state, cond, s_cls, th_cls, dt_cls)
        ar = torch.arange(T, device=device).unsqueeze(0)
        sup = (ar < n_sup.unsqueeze(1)).float()
        motion = (s_cls > 0) & (s_cls < S_PAD_CLASS)
        sup_th = sup * motion.float()
        ce_s = F.cross_entropy(s_logits.reshape(-1, N_S_CLASSES),
                               s_cls.reshape(-1), reduction="none").view(B, T)
        ce_th = F.cross_entropy(th_logits.reshape(-1, N_TH_CLASSES),
                                th_cls.reshape(-1), reduction="none").view(B, T)
        ce_dt = F.cross_entropy(dt_logits.reshape(-1, N_DT_CLASSES),
                                dt_cls.reshape(-1), reduction="none").view(B, T)
        return ((ce_s * sup).sum() / sup.sum().clamp(1),
                (ce_th * sup_th).sum() / sup_th.sum().clamp(1),
                (ce_dt * sup).sum() / sup.sum().clamp(1))


@torch.no_grad()
def validate(model, val_dl, device, amp):
    model.eval()
    tot, nb = np.zeros(3), 0
    for nb, b in enumerate(val_dl, 1):
        tot += [float(x) for x in batch_losses(model, b, device, amp)]
    model.train()
    return tot / max(nb, 1)


def widen_cond(model, ck_state):
    """Load the four dimensional checkpoint into the wider model.

    The first four columns of the condition projection keep their trained
    weights and every new column starts at exactly zero, so step zero of this
    fine tune reproduces the parent model's behaviour bit for bit on a batch
    whose flag is zero. Everything the model learns about the eighteen is
    learned here and nothing is disturbed on the way in.
    """
    own = model.state_dict()
    key = "cond_embed.0.weight"
    w = ck_state[key]
    new = own[key].clone()
    new.zero_()
    new[:, :w.shape[1]] = w
    ck_state = dict(ck_state)
    ck_state[key] = new
    missing, unexpected = model.load_state_dict(ck_state, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    torch.manual_seed(args.seed)

    print("Loading polar event data and the eighteen feature labels...",
          flush=True)
    # Only the feature labels and the validity mask are read here. Every token
    # array is opened per process inside FCDataset, because anything held on
    # this side would be pickled once per dataloader worker.
    feat = np.load(data_dir / "events_feat18.npy", mmap_mode="r")
    ok = np.load(data_dir / "events_feat18_ok.npy")
    N = len(ok)
    usable = np.flatnonzero(ok)
    print(f"  {len(usable):,} of {N:,} rows carry a feature label", flush=True)

    rng = np.random.default_rng(123)
    idx = np.sort(rng.choice(usable, min(len(usable), args.n_train),
                             replace=False))
    ftrain = np.asarray(feat[idx])
    knots = fit_knots(ftrain, np.random.default_rng(5))
    zfeat = to_gauss(ftrain, knots)
    del ftrain
    print(f"  quantile knots fitted on {len(idx):,} rows", flush=True)

    ds = FCDataset(data_dir, idx, zfeat, args.max_seq_len, args.feat_dropout,
                   args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True,
                    drop_last=True, persistent_workers=args.num_workers > 0)
    print(f"  {len(ds):,} trajectories", flush=True)

    val_dl = None
    if args.val_every:
        held = np.setdiff1d(usable, idx)
        if len(held) < args.val_n:
            raise SystemExit(f"only {len(held):,} unseen labelled rows")
        vi = np.sort(np.random.default_rng(7).choice(held, args.val_n,
                                                     replace=False))
        val_dl = DataLoader(
            FCDataset(data_dir, vi, to_gauss(np.asarray(feat[vi]), knots),
                      args.max_seq_len, args.feat_dropout, args.seed + 1),
            batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=True)
        print(f"  {args.val_n:,} held out for validation", flush=True)

    parent = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = dict(parent["config"])
    assert cfg["cond_dim"] == 4, cfg
    cfg["cond_dim"] = COND_DIM
    cfg["cond_dropout"] = 0.0     # the feature block has its own, in the dataset
    model = EventARModel(**cfg).to(device)
    widen_cond(model, parent["model_state_dict"])
    print(f"  initialised from {args.init} at step {parent.get('step')}, "
          f"cond_dim 4 -> {COND_DIM}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05,
        anneal_strategy="cos", div_factor=10.0, final_div_factor=20.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    save_path = data_dir / args.save_name
    model.train()
    step_i, t0, ema, val_hist = 0, time.time(), None, []
    data_iter = iter(dl)
    while step_i < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        s_loss, th_loss, dt_loss = batch_losses(model, batch, device, args.amp)
        loss = s_loss + th_loss + dt_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        step_i += 1
        # Sustained training runs hotter than the sampling loops the rest of
        # this repo does, and this machine crashed on a GPU workload at 79C on
        # 2026-08-06. Duty cycle on temperature rather than trusting the card.
        if args.thermal_every and step_i % args.thermal_every == 0:
            throttle()
        if step_i % args.log_every == 0 or step_i == 1:
            print(f"  step {step_i:6d}/{args.steps} | loss {ema:.4f} "
                  f"(s {s_loss.item():.3f} th {th_loss.item():.3f} "
                  f"dt {dt_loss.item():.3f}) | lr {sched.get_last_lr()[0]:.2e} "
                  f"| {gpu_c()}C | {time.time() - t0:.0f}s", flush=True)
        if val_dl is not None and (step_i % args.val_every == 0
                                   or step_i == args.steps):
            v = validate(model, val_dl, device, args.amp)
            val_hist.append(dict(step=step_i, train_ema=ema, s=float(v[0]),
                                 th=float(v[1]), dt=float(v[2]),
                                 total=float(v.sum())))
            print(f"  step {step_i:6d} | HELD OUT {v.sum():.4f} "
                  f"(s {v[0]:.3f} th {v[1]:.3f} dt {v[2]:.3f})", flush=True)
        if step_i % args.save_every == 0 or step_i == args.steps:
            torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                        "step": step_i, "loss_ema": ema, "val_hist": val_hist,
                        "feat_knots": knots, "n_feat": N_FEAT,
                        "gauss_clip": GAUSS_CLIP, "parent": args.init},
                       save_path)
        if args.snapshot_every and step_i % args.snapshot_every == 0:
            torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                        "step": step_i, "loss_ema": ema, "val_hist": val_hist,
                        "feat_knots": knots, "n_feat": N_FEAT,
                        "gauss_clip": GAUSS_CLIP, "parent": args.init},
                       save_path.with_stem(f"{save_path.stem}_s{step_i}"))
    print(f"Done. Final loss (ema): {ema:.4f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="training")
    p.add_argument("--init", default="training/event_ar_v2_s40000.pt")
    p.add_argument("--save-name", default="event_ar_fc.pt")
    p.add_argument("--n-train", type=int, default=1_500_000)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--feat-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-n", type=int, default=20_000)
    p.add_argument("--snapshot-every", type=int, default=2500)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--thermal-every", type=int, default=25,
                   help="duty cycle on GPU temperature every N steps; 0 disables")
    train(p.parse_args())
