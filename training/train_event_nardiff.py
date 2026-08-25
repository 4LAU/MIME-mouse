"""Train the non autoregressive event path model. Registered in nardiff_prereg.md.

num_workers is 0 and not configurable. A memmap held on a Dataset handed to a
multi worker DataLoader serialises in full per worker under this Python and has
killed this WSL VM three times.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.event_stream_polar import (s2_to_class, dth_lattice_to_class,
                                       S_PAD_CLASS, TH_NULL_CLASS)
from models.event_ar import DT_MAX_MS
from models.event_nardiff import EventNARDiff, MASKS, VOCABS

DATA = Path("/home/aaronadmin/mts_data")


class Corpus:
    """Draws batches straight from the memmaps in the MAIN process."""

    def __init__(self, max_len, n_train):
        self.s2 = np.load(DATA / "events_s2.npy", mmap_mode="r")
        self.dth = np.load(DATA / "events_dth.npy", mmap_mode="r")
        self.dt = np.load(DATA / "events_dt.npy", mmap_mode="r")
        self.len = np.load(DATA / "events_len.npy")
        self.cond = np.load(DATA / "events_cond.npy")
        self.T = max_len
        ok = np.flatnonzero(self.len >= 2)
        self.train_ids = ok[:n_train]
        self.val_ids = ok[-20000:]

    def batch(self, ids, device):
        ids = np.sort(ids)
        T = self.T
        L = np.minimum(self.len[ids], T)
        s2 = torch.from_numpy(np.asarray(self.s2[ids, :T], dtype=np.int64))
        dth = torch.from_numpy(np.asarray(self.dth[ids, :T], dtype=np.int64))
        dtm = torch.from_numpy(np.asarray(self.dt[ids, :T], dtype=np.float32))
        live = torch.from_numpy(np.arange(T)[None] < L[:, None])

        s = torch.where(live, s2_to_class(s2),
                        torch.full_like(s2, S_PAD_CLASS))
        th = torch.where(live & (s2 > 0), dth_lattice_to_class(dth),
                         torch.full_like(dth, TH_NULL_CLASS))
        dt = torch.where(live, torch.round(dtm).long().clamp(0, DT_MAX_MS),
                         torch.zeros_like(s2))
        cond = torch.from_numpy(np.asarray(self.cond[ids], dtype=np.float32))
        return (s.to(device), th.to(device), dt.to(device), live.to(device),
                torch.from_numpy(L.astype(np.int64)).to(device),
                cond.to(device))


def diffusion_loss(model, batch, amp):
    """Absorbing chain CE on masked cells, 1/t weighted, plus the length head.

    The 1/t weighting makes the token term an UPPER BOUND on the negative log
    likelihood. It is NOT the AR model's exact NLL and must never be quoted
    against 4.4024 as if it were.
    """
    s, th, dt, live, L, cond = batch
    B, T = s.shape
    dev = s.device
    tgt = (s, th, dt)

    t = torch.rand(B, device=dev).clamp(min=1e-3)
    # each (channel, position) cell is masked independently, live cells only
    corrupt, mask = [], []
    for c in range(3):
        m = (torch.rand(B, T, device=dev) < t[:, None]) & live
        corrupt.append(torch.where(m, torch.full_like(tgt[c], MASKS[c]),
                                   tgt[c]))
        mask.append(m)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(corrupt[0], corrupt[1], corrupt[2], t, cond)
        len_logits = model.length_logits(cond)

    tok = torch.zeros((), device=dev)
    denom = live.sum().clamp(min=1).float()
    for c in range(3):
        lg = logits[c].float()
        ce = F.cross_entropy(lg.reshape(-1, VOCABS[c]), tgt[c].reshape(-1),
                             reduction="none").reshape(B, T)
        w = (mask[c].float() / t[:, None])
        tok = tok + (ce * w).sum() / denom
    len_loss = F.cross_entropy(len_logits.float(), (L - 1).clamp(min=0))
    return tok, len_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-name", default="event_nardiff_v1.pt")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=10)
    ap.add_argument("--d-ff", type=int, default=1888)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--snap-every", type=int, default=2000)
    # A hard machine crash on 2026-08-18 cost 5800 steps because the snapshot
    # carried weights only. It now carries optimiser, schedule, sampling rng
    # and step, so --resume continues the SAME run rather than starting a
    # second one with fresh Adam moments and a restarted schedule.
    ap.add_argument("--resume", default="")
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = Corpus(a.max_seq_len, a.n_train)
    model = EventNARDiff(d_model=a.d_model, n_heads=a.n_heads,
                         n_layers=a.n_layers, d_ff=a.d_ff,
                         max_seq_len=a.max_seq_len).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  params {n_par/1e6:.2f}M  train rows {len(corpus.train_ids)}  "
          f"device {dev}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.03)
    rng = np.random.default_rng(a.seed)
    ema, t0, hist, start = None, time.time(), [], 0

    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        if "opt" not in ck:
            raise SystemExit(f"{a.resume} predates resume support and carries "
                             f"weights only. Resuming from it would restart "
                             f"Adam and the schedule, which is a different run. "
                             f"Train from scratch instead.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["cuda_rng"]])
        start, ema, hist = ck["step"], ck["ema"], ck["hist"]
        init_tok = ck["init_elbo"]
        print(f"  resumed {a.resume} at step {start}, ELBO at init "
              f"{init_tok:.4f}", flush=True)
    else:
        # G3 needs the value at initialisation, so measure it before any step
        model.eval()
        with torch.no_grad():
            vb = corpus.batch(corpus.val_ids[:a.batch_size], dev)
            init_tok, _ = diffusion_loss(model, vb, a.amp)
        init_tok = float(init_tok)
        print(f"  ELBO at init {init_tok:.4f}", flush=True)
    model.train()

    for step in range(start + 1, a.steps + 1):
        ids = rng.choice(corpus.train_ids, a.batch_size, replace=False)
        tok, ln = diffusion_loss(model, corpus.batch(ids, dev), a.amp)
        loss = tok + ln
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema = float(tok) if ema is None else 0.98 * ema + 0.02 * float(tok)

        if step % a.log_every == 0:
            print(f"  step {step:6d}  elbo_ema {ema:.4f}  len {float(ln):.4f}  "
                  f"{(time.time()-t0)/(step-start):.3f}s/step", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                vt = [float(diffusion_loss(
                    model, corpus.batch(
                        corpus.val_ids[i * a.batch_size:(i + 1) * a.batch_size],
                        dev), a.amp)[0]) for i in range(8)]
            model.train()
            hist.append({"step": step, "val_elbo": float(np.mean(vt))})
            print(f"  == val step {step}  elbo {np.mean(vt):.4f}", flush=True)
        if step % a.snap_every == 0 or step == a.steps:
            # Written to a sibling then renamed. A crash during the write
            # then costs the snapshot, not the previous good one.
            dst = Path("training") / a.save_name
            tmp = dst.with_suffix(dst.suffix + ".part")
            torch.save({"model": model.state_dict(), "step": step,
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": (torch.cuda.get_rng_state_all()
                                     if torch.cuda.is_available() else None),
                        "ema": ema,
                        "init_elbo": init_tok, "hist": hist,
                        "config": {"d_model": a.d_model, "n_heads": a.n_heads,
                                   "n_layers": a.n_layers, "d_ff": a.d_ff,
                                   "max_seq_len": a.max_seq_len}}, tmp)
            os.replace(tmp, dst)
            print(f"  saved training/{a.save_name} at step {step}", flush=True)

    json.dump({"init_elbo": init_tok, "hist": hist, "params": n_par},
              open(f"research/w4_nardiff_train_{a.save_name}.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
