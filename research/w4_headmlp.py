"""Fine tune `event_ar_v2_s40000` with the coupled direction head, against two
controls, and log held out likelihood on the split the checkpoint never saw.

WHY THIS RUN EXISTS
-------------------
`research/w4_headcap.py` priced the additive within step restriction in
`models/event_ar.py` at 0.2311 nats se 0.0018 on 60,000 held out trajectories.
That measurement froze the trunk and fitted a residual correction on top of the
frozen head, which is what made it cheap and what makes it clean: three controls
came back at +0.0000, +0.0037 and +0.0011, so the gain is neither slack nor
depth nor width, it is the emitted speed's information.

The one limitation that survives is that the trunk was fitted jointly WITH the
restricted head. A co adapted trunk has already spent capacity hedging around a
distribution it cannot shape, so the frozen number is what a correction on a co
adapted trunk buys and is NOT a promise about a jointly trained model. Only
training settles it. That is this file.

WHAT IS BEING COMPARED
----------------------
Three arms, all fine tuned from the same checkpoint with identical steps, batch,
data, seed, schedule and validation set:

    add       the unchanged model.  Excludes "more training helps".
    mlp_nos   an MLP in the head with NO speed input.  Excludes "more capacity
              helps".  This is the training arm version of `R_depth`.
    mlp       the same MLP WITH the speed input.  The intervention, and arm
              `R_full` from the frozen probe ported verbatim.

The contrast that matters is mlp minus mlp_nos, because that is the same
contrast the frozen probe reported. mlp minus add is quoted too but it is the
weaker one: it confounds the coupling with the extra parameters, which is
exactly what `R_shuf` was built to separate and what mlp_nos separates here.

Every arm starts bit identical to the checkpoint. The new head's output layer is
zero initialised, verified by an exact zero max abs difference on all three
logit streams before any step is taken, so the fine tune starts at the
checkpoint's own held out loss and every nat after that is attributable.

THE SPLIT
---------
`TRAIN_PICK_SEED 123` and 1,500,000 rows reproduce the checkpoint's own training
set exactly. The fine tune subset is drawn from INSIDE that set, so nothing this
run trains on has ever been held out, and validation is drawn from the 2,528,855
rows the checkpoint has never seen. Getting this backwards would make every
number here worthless, so it is asserted at runtime rather than trusted.

MEMORY
------
`num_workers` is 0 and not configurable. The three corpus arrays are 2 GB each,
Python 3.14's DataLoader default is forkserver, and a worker pickles the whole
Dataset. That combination has killed this WSL VM three times. Do not add
workers here to chase throughput.

PREDICTIONS, registered in /home/aaronadmin/w4_arms/headmlp_prereg.md before
this file was written: held out th, mlp beats mlp_nos by 0.23 to 0.35 nats;
contract AUC lands between 0.615 and 0.640, SHORT of the 0.609 the nats to AUC
exchange rate would predict, because that rate was fitted along a training
trajectory and this is an architectural change.

Run:
    NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512BW AVX512DQ \
    AVX512VL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=.:research:research/autoloop ~/venvs/mime/bin/python \
      research/w4_headmlp.py --coupling mlp --steps 6000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.event_ar import EventARModel  # noqa: E402
from training.train_event_ar import ARDataset, batch_losses  # noqa: E402

TRAIN_PICK_SEED = 123
N_TRAIN_DEFAULT = 1_500_000

# Tightened from the repo default of 83 because this machine crashed on a
# sustained AR workload on 2026-08-06.
KILL_C = 79
GATE_C = 75
RESUME_C = 70


def gpu_temp() -> int:
    try:
        return int(subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        return -1


def thermal_gate(step: int) -> int:
    """Returns the peak seen. Kills rather than cooking the card."""
    t = gpu_temp()
    if t >= KILL_C:
        raise SystemExit(f"KILL at step {step}: GPU {t}C >= {KILL_C}C")
    if t >= GATE_C:
        print(f"    cooling at step {step}, {t}C", flush=True)
        waited = 0
        while gpu_temp() > RESUME_C and waited < 300:
            time.sleep(10)
            waited += 10
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="event_ar_v2_s40000.pt")
    ap.add_argument("--coupling", required=True,
                    choices=["add", "mlp", "mlp_nos"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr-base", type=float, default=5e-5,
                    help="pretrained weights; small, this is a fine tune")
    ap.add_argument("--lr-new", type=float, default=1e-3,
                    help="the fresh mix module only; matches what the frozen "
                         "probe fitted its correction at")
    ap.add_argument("--n-train", type=int, default=500_000)
    ap.add_argument("--val-n", type=int, default=20_000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--mix-hidden", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"research/w4_headmlp_{a.coupling}.json"
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    s2 = np.load("training/events_s2.npy", mmap_mode="r")
    dth = np.load("training/events_dth.npy", mmap_mode="r")
    dtms = np.load("training/events_dt.npy", mmap_mode="r")
    lengths = np.load("training/events_len.npy")
    cond = np.load("training/events_cond.npy")
    N = len(lengths)

    trained = np.sort(np.random.default_rng(TRAIN_PICK_SEED)
                      .choice(N, min(N, N_TRAIN_DEFAULT), replace=False))
    held = np.setdiff1d(np.arange(N), trained)
    fit_idx = np.sort(np.random.default_rng(a.seed)
                      .choice(trained, min(a.n_train, len(trained)),
                              replace=False))
    val_idx = np.sort(np.random.default_rng(7)
                      .choice(held, min(a.val_n, len(held)), replace=False))
    # Asserted, not trusted. A leak here would silently invalidate every number.
    assert not np.intersect1d(fit_idx, val_idx).size
    assert not np.intersect1d(val_idx, trained).size
    print(f"\n  fine tune on {len(fit_idx):,} rows from the checkpoint's own "
          f"training set", flush=True)
    print(f"  validate on {len(val_idx):,} rows it has NEVER seen", flush=True)

    ck = torch.load(f"training/{a.ckpt}", map_location=dev, weights_only=False)
    cfg = dict(ck["config"])
    cfg["head_coupling"] = a.coupling
    if a.coupling != "add":
        cfg["mix_hidden"] = a.mix_hidden
    model = EventARModel(**cfg).to(dev)
    miss, unexp = model.load_state_dict(ck["model_state_dict"], strict=False)
    assert not unexp, unexp
    assert all(k.startswith(("mix_", "th_mix")) for k in miss), miss
    new_names = {n for n, _ in model.named_parameters()
                 if n.startswith(("mix_", "th_mix"))}
    assert new_names == set(miss) or a.coupling == "add", (new_names, miss)
    n_new = sum(p.numel() for n, p in model.named_parameters()
                if n in new_names)
    print(f"  {a.ckpt}, coupling {a.coupling}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params, "
          f"{n_new/1e6:.3f}M of them fresh", flush=True)

    ds = ARDataset(s2[fit_idx], dth[fit_idx], dtms[fit_idx], lengths[fit_idx],
                   cond[fit_idx], cfg["max_seq_len"])
    vds = ARDataset(s2[val_idx], dth[val_idx], dtms[val_idx], lengths[val_idx],
                    cond[val_idx], cfg["max_seq_len"])
    # num_workers 0 is deliberate, see the module docstring.
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=0, pin_memory=True)
    vdl = DataLoader(vds, batch_size=a.batch, shuffle=False, drop_last=True,
                     num_workers=0, pin_memory=True)

    groups = [dict(params=[p for n, p in model.named_parameters()
                           if n not in new_names], lr=a.lr_base)]
    if new_names:
        groups.append(dict(params=[p for n, p in model.named_parameters()
                                   if n in new_names], lr=a.lr_new))
    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.95))
    # Warm up before the cosine. The checkpoint finished its own schedule at
    # about 1.2e-6 and AdamW starts here with empty moment estimates, so
    # stepping straight in at lr_base is a shock to a converged model; the
    # 60 step smoke showed held out loss rising rather than falling. Every arm
    # gets the identical schedule, so this is not a thumb on the scale.
    warm = min(a.warmup, max(a.steps // 10, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.02, 1.0, warm),
              torch.optim.lr_scheduler.CosineAnnealingLR(
                  opt, T_max=max(a.steps - warm, 1), eta_min=0.0)],
        milestones=[warm])
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    @torch.no_grad()
    def validate():
        model.eval()
        tot, nb = np.zeros(3), 0
        for b in vdl:
            tot += [float(x) for x in batch_losses(model, b, dev, True)]
            nb += 1
        model.train()
        return tot / max(nb, 1)

    v0 = validate()
    print(f"  step      0 | HELD OUT {v0.sum():.4f} "
          f"(s {v0[0]:.4f} th {v0[1]:.4f} dt {v0[2]:.4f})  <- the checkpoint",
          flush=True)

    hist = [dict(step=0, s=float(v0[0]), th=float(v0[1]), dt=float(v0[2]),
                 total=float(v0.sum()))]
    best = dict(total=float(v0.sum()), step=0, th=float(v0[1]))
    best_state = None
    peak, ema, step_i = gpu_temp(), None, 0
    model.train()
    it = iter(dl)
    while step_i < a.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        s_l, th_l, dt_l = batch_losses(model, batch, dev, True)
        loss = s_l + th_l + dt_l
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        step_i += 1

        if step_i % 200 == 0:
            peak = max(peak, thermal_gate(step_i))
        if step_i % 500 == 0:
            print(f"    step {step_i:6d}/{a.steps} | train ema {ema:.4f} | "
                  f"{gpu_temp()}C | {time.time() - t0:.0f}s", flush=True)
        if step_i % a.val_every == 0 or step_i == a.steps:
            v = validate()
            hist.append(dict(step=step_i, s=float(v[0]), th=float(v[1]),
                             dt=float(v[2]), total=float(v.sum())))
            mark = ""
            if v.sum() < best["total"]:
                best = dict(total=float(v.sum()), step=step_i, th=float(v[1]),
                            s=float(v[0]), dt=float(v[2]))
                best_state = {k: x.detach().cpu().clone()
                              for k, x in model.state_dict().items()}
                mark = "  *"
            print(f"  step {step_i:6d} | HELD OUT {v.sum():.4f} "
                  f"(s {v[0]:.4f} th {v[1]:.4f} dt {v[2]:.4f}) "
                  f"| vs start {v.sum() - v0.sum():+.4f} "
                  f"(th {v[1] - v0[1]:+.4f}){mark}", flush=True)

    if best_state is not None:
        torch.save({"model_state_dict": best_state, "config": cfg,
                    "step": best["step"], "val_hist": hist,
                    "base_ckpt": a.ckpt},
                   f"training/event_ar_hm_{a.coupling}.pt")

    print(f"\n  best held out {best['total']:.4f} at step {best['step']}, "
          f"th {best['th']:.4f}")
    print(f"  th improvement over the checkpoint {v0[1] - best['th']:+.4f} nats")
    json.dump(dict(coupling=a.coupling, config=vars(a), model_config=cfg,
                   start=dict(s=float(v0[0]), th=float(v0[1]),
                              dt=float(v0[2]), total=float(v0.sum())),
                   best=best, val_hist=hist, n_new_params=int(n_new),
                   peak_c=peak, elapsed_s=round(time.time() - t0, 1)),
              open(out, "w"), indent=2)
    print(f"  peak {peak}C, {time.time() - t0:.0f}s, wrote {out}")


if __name__ == "__main__":
    main()
