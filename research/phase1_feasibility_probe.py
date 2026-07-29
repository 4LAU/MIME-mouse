"""Phase 1 feasibility probe for DIFFUSION_PILOT_V2.md's differentiable-tail
fix. NO TRAINING happens here: optimizer.step() is never called, no
checkpoint is ever written. training/candi_polar_flow_best.pt is loaded
read-only and its MD5 is checked before and after (expected
91326a29750789f3167055324ef377c5).

Question this answers: how large can K (the number of last sampler steps
that carry gradient back through the curvature loss) practically be on this
machine (RTX 4070 laptop, 8 GB VRAM), including K=200 (the full chain), with
gradient checkpointing off and on. This is a measurement script, not a
training script -- it reports wall time and peak VRAM per (K, checkpoint,
batch) cell and confirms gradient actually reaches model parameters. It does
NOT reproduce phase0's exact detector-space numbers and does not need to:
correctness of the sampler's numerical output is phase0's job, this script's
job is compute-cost feasibility of backpropagating through it.

Deliberately does NOT import experiments/candi.py (protected-file import at
module level). Sampler body is adapted from research/phase0_collapse_curve.py
/ research/phase_a_baseline.py's verbatim port of
experiments/candi.py's _sample_guided_flow.

Hard constraint: never loads data/human_eval_features.npy, and in fact never
loads ANY human data file -- the curvature-loss "human" target is a
hardcoded dummy tensor (see make_dummy_human below), since this probe only
measures whether gradient flows and how expensive it is, not the loss's
actual numeric value against real humans.

Known differentiability blockers this script fixes relative to the
production sampler (research/phase_a_baseline.py / experiments/candi.py):

  (a) The guide block's `cx[-1].item()` / `cy[-1].item()` conversions
      detach raw_mag/raw_ang from the graph before they're used to build
      `rot`, which silently zeroes the gradient path through the guidance
      term. Fixed here by keeping raw_mag/raw_ang as tensors end to end and
      replacing the python `if raw_mag > 1e-6:` guard with
      `torch.where(raw_mag > 1e-6, rot, zeros)`.

  (b) The stall-reveal block's `torch.where(reveal, ...)` on `stall_s` /
      `mflag` is built from `>` comparisons, which never carry gradient in
      autograd regardless -- but to satisfy the letter of the hard
      constraint (freeze the discrete state, no gradient, only the
      continuous polar path carries signal) this script does not merely
      rely on that fact: inside the differentiable K-step tail, stall_s and
      mflag are frozen at whatever value they held on window entry and are
      never updated again for the remainder of generation. Only `dp` (the
      continuous x0_hat estimate) keeps evolving with gradient intact.
"""
from __future__ import annotations

import argparse
import csv
import functools
import gc
import hashlib
import math
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from experiments._common import get_device
from training.curvature_loss import curvature_moment_loss

TRAIN_DIR = Path("training")
CKPT_NAME = "candi_polar_flow_best.pt"
EXPECTED_MD5 = "91326a29750789f3167055324ef377c5"

SEQ_LEN = 192          # fixed representative sequence length for this probe
N_SAMPLE_STEPS = 200
GUIDE = 0.15
SEED = 42

K_GRID = [8, 16, 50, 100, 200]
CKPT_GRID = [False, True]
BATCH_START = 16
BATCH_FLOOR = 4

# Windows WDDM silently spills CUDA allocations into shared system memory
# instead of raising OOM, which destroys both timing and VRAM measurements
# (observed: "peak_vram 12431MB" on an 8GB card at 30s/step). Cap the
# allocator at this fraction of physical VRAM so an over-budget cell raises
# a real torch.OutOfMemoryError and the batch-halving loop handles it.
VRAM_FRACTION = 0.80


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(device):
    ckpt_path = TRAIN_DIR / CKPT_NAME
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    from models.candi import CANDIModel
    model = CANDIModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # inference-mode dropout behavior, matches sampling; params still require grad
    for p in model.parameters():
        p.requires_grad_(True)
    data_scale = ckpt["data_scale"]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[phase1] loaded {ckpt_path} device={device} params={n_params:,}", flush=True)
    return model, data_scale


def make_cond(batch, seq_len, device):
    """Fixed synthetic condition batch. Values only need to be in a
    plausible range (matches train_conditions.npy's rough scale); this probe
    never scores detector features, so exact realism doesn't matter."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    log_dist = torch.empty(batch).uniform_(4.0, 6.5, generator=g)
    log_dur = torch.empty(batch).uniform_(-1.5, 0.5, generator=g)
    angle = torch.empty(batch).uniform_(0.0, 2 * math.pi, generator=g)
    cond = torch.stack(
        [log_dist, log_dur, torch.cos(angle), torch.sin(angle)], dim=1
    ).to(device)
    tgt_angle = angle.to(device)
    return cond, tgt_angle


def make_dummy_human(batch, seq_len, spd_s, dh_s, device):
    """HARDCODED DUMMY variety target -- NOT real human data (hard
    constraint: this probe never loads any human data file, including but
    not limited to data/human_eval_features.npy). Only used to exercise
    curvature_moment_loss's shape/control-flow and produce a real,
    backward-able scalar loss for the feasibility measurement; its absolute
    value is meaningless and is not a result of this probe."""
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    speed = torch.rand(batch, seq_len, generator=g) * spd_s * 0.5
    dh = (torch.rand(batch, seq_len, generator=g) - 0.5) * dh_s * 0.3
    return torch.stack([speed, dh], dim=-1).to(device)


def prefix_step(model, xt, stall_s, mflag, cond, tgt_angle, i, n_steps, spd_s, dh_s, guide):
    """No-grad sampler step (steps outside the differentiable tail). Guide
    block is vectorized across the batch (no python per-item loop / .item()
    calls) purely as a speed simplification -- this branch never carries
    gradient anyway (wrapped in torch.no_grad() by the caller)."""
    dt = 1.0 / n_steps
    t_cont = 1.0 - i * dt
    B = xt.shape[0]
    dev = xt.device
    t_scaled = torch.full((B,), t_cont * (model.n_steps - 1), device=dev)
    v_pred, sl = model(xt, stall_s, mflag, t_scaled, cond)
    dp = xt - t_cont * v_pred

    frac = 1.0 - t_cont
    if frac > 0.3:
        conf = torch.abs(sl)
        thresh = max(0.5, 3.0 * (1.0 - frac))
        reveal = (conf > thresh) & (mflag > 0.5)
        stall_s = torch.where(reveal, (torch.sigmoid(sl) > 0.5).float(), stall_s)
        mflag = torch.where(reveal, torch.zeros_like(mflag), mflag)

    if frac > 0.3 and guide > 0:
        spd = torch.clamp(dp[..., 0] / spd_s, min=0)
        dh = dp[..., 1] / dh_s
        active_stall = (stall_s > 0.5) & (mflag < 0.5)
        keep = (~active_stall).float()
        heading = torch.cumsum(dh * keep, dim=1)
        cx = torch.cumsum(spd * keep * torch.cos(heading), dim=1)
        cy = torch.cumsum(spd * keep * torch.sin(heading), dim=1)
        raw_mag = torch.hypot(cx[:, -1], cy[:, -1])
        raw_ang = torch.atan2(cy[:, -1], cx[:, -1])
        rot = (tgt_angle - raw_ang) * guide * frac
        rot = torch.where(raw_mag > 1e-6, rot, torch.zeros_like(rot))
        dp = dp.clone()
        dp[:, 0, 1] = dp[:, 0, 1] + rot * dh_s

    if t_cont > 1e-6:
        v_guided = (xt - dp) / t_cont
    else:
        v_guided = v_pred
    xt = xt - dt * v_guided
    return xt, stall_s, mflag


def tail_step(xt, *, model, stall_frozen, mflag_frozen, cond, tgt_angle,
              i, n_steps, spd_s, dh_s, guide):
    """Differentiable sampler step for the last-K window. stall_frozen /
    mflag_frozen are constants (blocker (b): discrete state frozen, no
    reveal update happens inside the tail at all). Blocker (a) fixed: raw_mag
    / raw_ang stay tensors, the near-zero-magnitude guard uses torch.where
    instead of a python `if` on a converted float."""
    dt = 1.0 / n_steps
    t_cont = 1.0 - i * dt
    B = xt.shape[0]
    dev = xt.device
    t_scaled = torch.full((B,), t_cont * (model.n_steps - 1), device=dev)
    v_pred, sl = model(xt, stall_frozen, mflag_frozen, t_scaled, cond)
    dp = xt - t_cont * v_pred

    frac = 1.0 - t_cont
    if frac > 0.3 and guide > 0:
        spd = torch.clamp(dp[..., 0] / spd_s, min=0)
        dh = dp[..., 1] / dh_s
        active_stall = (stall_frozen > 0.5) & (mflag_frozen < 0.5)
        keep = (~active_stall).float()
        heading = torch.cumsum(dh * keep, dim=1)
        cx = torch.cumsum(spd * keep * torch.cos(heading), dim=1)
        cy = torch.cumsum(spd * keep * torch.sin(heading), dim=1)
        raw_mag = torch.hypot(cx[:, -1], cy[:, -1])
        raw_ang = torch.atan2(cy[:, -1], cx[:, -1])
        rot = (tgt_angle - raw_ang) * guide * frac
        rot = torch.where(raw_mag > 1e-6, rot, torch.zeros_like(rot))
        dp = dp.clone()
        dp[:, 0, 1] = dp[:, 0, 1] + rot * dh_s

    if t_cont > 1e-6:
        v_guided = (xt - dp) / t_cont
    else:
        v_guided = v_pred
    xt = xt - dt * v_guided
    return xt, dp


def simulate_training_step(model, data_scale, device, batch, seq_len, k,
                            use_ckpt, n_steps=N_SAMPLE_STEPS, guide=GUIDE):
    """One full simulated training step: [no_grad prefix] -> [grad-enabled
    last-k tail] -> [curvature loss on the tail's final x0_hat] ->
    [backward()] -> [grad-norm check] -> [zero grad, no optimizer step].
    Returns a dict of measurements. Raises the underlying exception on OOM
    (caller decides retry/backoff)."""
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])
    model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    cond, tgt_angle = make_cond(batch, seq_len, device)
    xt = torch.randn(batch, seq_len, 2, device=device)
    stall_s = torch.full((batch, seq_len), model.STALL_MASK, device=device)
    mflag = torch.ones(batch, seq_len, device=device)

    n_prefix = n_steps - k
    with torch.no_grad():
        for i in range(n_prefix):
            xt, stall_s, mflag = prefix_step(
                model, xt, stall_s, mflag, cond, tgt_angle, i, n_steps,
                spd_s, dh_s, guide,
            )

    # Entering the differentiable tail: freeze discrete state (hard
    # constraint (b)); xt needs requires_grad for checkpoint's recompute
    # bookkeeping (harmless -- its own .grad is unused, only model
    # parameters' gradients are the measurement target).
    stall_frozen = stall_s.detach()
    mflag_frozen = mflag.detach()
    xt = xt.detach().requires_grad_(True)

    dp_final = None
    for i in range(n_prefix, n_steps):
        step_fn = functools.partial(
            tail_step, model=model, stall_frozen=stall_frozen,
            mflag_frozen=mflag_frozen, cond=cond, tgt_angle=tgt_angle,
            i=i, n_steps=n_steps, spd_s=spd_s, dh_s=dh_s, guide=guide,
        )
        if use_ckpt:
            xt, dp_final = checkpoint(step_fn, xt, use_reentrant=False)
        else:
            xt, dp_final = step_fn(xt)

    pad_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
    dummy_human = make_dummy_human(batch, seq_len, spd_s, dh_s, device)

    loss, stats = curvature_moment_loss(dp_final, dummy_human, pad_mask, spd_s, dh_s)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss.item()}")

    loss.backward()

    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    total_grad_norm = math.sqrt(sum(g * g for g in grad_norms)) if grad_norms else 0.0
    n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )

    model.zero_grad(set_to_none=True)  # no optimizer.step() ever called

    return {
        "elapsed_s": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "loss": float(loss.detach().item()),
        "grad_norm": total_grad_norm,
        "n_params_with_grad": n_params_with_grad,
    }


def baseline_no_grad_generation(model, data_scale, device, batch, seq_len,
                                 n_steps=N_SAMPLE_STEPS, guide=GUIDE):
    """Plain no_grad full-chain generation timing, same shapes, no loss/backward."""
    spd_s, dh_s = float(data_scale[0]), float(data_scale[1])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    cond, tgt_angle = make_cond(batch, seq_len, device)
    xt = torch.randn(batch, seq_len, 2, device=device)
    stall_s = torch.full((batch, seq_len), model.STALL_MASK, device=device)
    mflag = torch.ones(batch, seq_len, device=device)
    with torch.no_grad():
        for i in range(n_steps):
            xt, stall_s, mflag = prefix_step(
                model, xt, stall_s, mflag, cond, tgt_angle, i, n_steps,
                spd_s, dh_s, guide,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda" else 0.0
    )
    return elapsed, peak_vram_mb


def is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or isinstance(
        exc, getattr(torch.cuda, "OutOfMemoryError", tuple())
    )


CSV_FIELDS = [
    "mode", "k", "ckpt", "batch_requested", "batch_achieved", "status",
    "peak_vram_mb", "sec_per_step", "loss", "grad_norm", "n_params_with_grad", "note",
]


def append_row(out_csv: Path, row: dict, write_header: bool):
    with open(out_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_cell(model, data_scale, device, k, use_ckpt, out_csv, header_written,
             batch_start=BATCH_START, batch_floor=BATCH_FLOOR, seq_len=SEQ_LEN):
    batch = batch_start
    while True:
        note = ""
        try:
            print(f"[phase1] trying K={k} ckpt={use_ckpt} batch={batch} ...", flush=True)
            res = simulate_training_step(model, data_scale, device, batch, seq_len, k, use_ckpt)
            row = {
                "mode": "grad_tail", "k": k, "ckpt": use_ckpt,
                "batch_requested": batch_start, "batch_achieved": batch,
                "status": "ok", "peak_vram_mb": f"{res['peak_vram_mb']:.1f}",
                "sec_per_step": f"{res['elapsed_s']:.4f}",
                "loss": f"{res['loss']:.6f}", "grad_norm": f"{res['grad_norm']:.6e}",
                "n_params_with_grad": res["n_params_with_grad"], "note": note,
            }
            append_row(out_csv, row, not header_written[0])
            header_written[0] = True
            print(f"[phase1] OK K={k} ckpt={use_ckpt} batch={batch} "
                  f"sec/step={res['elapsed_s']:.3f} peak_vram={res['peak_vram_mb']:.0f}MB "
                  f"grad_norm={res['grad_norm']:.4e}", flush=True)
            return row
        except Exception as exc:  # noqa: BLE001
            oom = is_oom(exc)
            print(f"[phase1] {'OOM' if oom else 'ERROR'} K={k} ckpt={use_ckpt} "
                  f"batch={batch}: {exc}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            if not oom:
                row = {
                    "mode": "grad_tail", "k": k, "ckpt": use_ckpt,
                    "batch_requested": batch_start, "batch_achieved": batch,
                    "status": "error", "peak_vram_mb": "", "sec_per_step": "",
                    "loss": "", "grad_norm": "", "n_params_with_grad": "",
                    "note": str(exc)[:200],
                }
                append_row(out_csv, row, not header_written[0])
                header_written[0] = True
                return row
            if batch <= batch_floor:
                row = {
                    "mode": "grad_tail", "k": k, "ckpt": use_ckpt,
                    "batch_requested": batch_start, "batch_achieved": batch,
                    "status": "oom", "peak_vram_mb": "", "sec_per_step": "",
                    "loss": "", "grad_norm": "", "n_params_with_grad": "",
                    "note": "OOM even at batch floor",
                }
                append_row(out_csv, row, not header_written[0])
                header_written[0] = True
                return row
            batch = max(batch_floor, batch // 2)


def smoke_test():
    """CPU smoke test: K=2, batch=2. Verifies non-zero finite grad norm
    before any GPU work is attempted."""
    device = torch.device("cpu")
    model, data_scale = load_model(device)
    res = simulate_training_step(model, data_scale, device, batch=2, seq_len=SEQ_LEN, k=2, use_ckpt=False)
    print(f"[phase1] SMOKE TEST result: {res}", flush=True)
    assert math.isfinite(res["loss"]), "loss not finite"
    assert res["grad_norm"] > 0.0, "grad norm is zero -- gradient did not reach parameters"
    assert res["n_params_with_grad"] > 0
    print("[phase1] SMOKE TEST PASSED: finite loss, non-zero grad norm, "
          f"{res['n_params_with_grad']} parameter tensors received gradient.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "grid"], default="grid")
    ap.add_argument("--k-list", type=str, default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--ckpt-list", type=str, default="off,on")
    ap.add_argument("--batch-start", type=int, default=BATCH_START)
    ap.add_argument("--batch-floor", type=int, default=BATCH_FLOOR)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--out-csv", type=str, default="research/phase1_results.csv")
    ap.add_argument("--pid-file", type=str, default="research/phase1.pid.txt",
                     help="Write this process's own PID here at startup so an "
                          "external watchdog can be pointed at it.")
    ap.add_argument("--resume", action="store_true",
                     help="Skip (k, ckpt) cells already present in --out-csv")
    args = ap.parse_args()

    import os
    Path(args.pid_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.pid_file, "w") as fh:
        fh.write(str(os.getpid()))
    print(f"[phase1] PID={os.getpid()} written to {args.pid_file}", flush=True)

    ckpt_path = TRAIN_DIR / CKPT_NAME
    md5_before = md5_file(ckpt_path)
    print(f"[phase1] MD5 before: {md5_before} (expected {EXPECTED_MD5})", flush=True)

    if args.mode == "smoke":
        smoke_test()
        md5_after = md5_file(ckpt_path)
        print(f"[phase1] MD5 after: {md5_after}", flush=True)
        assert md5_before == md5_after, "checkpoint file changed -- should never happen"
        return

    device = get_device()
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, device=0)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[phase1] VRAM cap: {VRAM_FRACTION:.2f} x {total_mb:.0f}MB = "
              f"{VRAM_FRACTION * total_mb:.0f}MB (spill-to-shared-memory guard)", flush=True)
    model, data_scale = load_model(device)

    k_list = [int(x) for x in args.k_list.split(",")]
    ckpt_list = [x.strip().lower() in ("on", "true", "1") for x in args.ckpt_list.split(",")]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header_written = [out_csv.exists() and out_csv.stat().st_size > 0]

    done = set()
    if args.resume and out_csv.exists():
        with open(out_csv) as fh:
            for row in csv.DictReader(fh):
                if row["mode"] == "grad_tail":
                    done.add((int(row["k"]), row["ckpt"] == "True", int(row["batch_requested"])))

    results = []
    for k in k_list:
        for use_ckpt in ckpt_list:
            key = (k, use_ckpt, args.batch_start)
            if key in done:
                print(f"[phase1] skip (resume) K={k} ckpt={use_ckpt}", flush=True)
                continue
            row = run_cell(model, data_scale, device, k, use_ckpt, out_csv,
                            header_written, args.batch_start, args.batch_floor,
                            args.seq_len)
            results.append(row)

    # Baseline no-grad generation timing, at batch_start and at the largest
    # achieved batch for a feasible K=200 cell (direct throughput comparison).
    baseline_batches = {args.batch_start}
    for row in results:
        if row["k"] == 200 and row["status"] == "ok":
            baseline_batches.add(int(row["batch_achieved"]))
    for b in sorted(baseline_batches):
        try:
            elapsed, peak = baseline_no_grad_generation(model, data_scale, device, b, args.seq_len)
            row = {
                "mode": "baseline_no_grad", "k": 200, "ckpt": "n/a",
                "batch_requested": b, "batch_achieved": b, "status": "ok",
                "peak_vram_mb": f"{peak:.1f}", "sec_per_step": f"{elapsed:.4f}",
                "loss": "", "grad_norm": "", "n_params_with_grad": "",
                "note": "plain no_grad 200-step generation, no backward",
            }
            append_row(out_csv, row, not header_written[0])
            header_written[0] = True
            print(f"[phase1] baseline no_grad batch={b} elapsed={elapsed:.3f}s "
                  f"peak_vram={peak:.0f}MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[phase1] baseline no_grad batch={b} FAILED: {exc}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    md5_after = md5_file(ckpt_path)
    print(f"[phase1] MD5 after: {md5_after}", flush=True)
    if md5_before != md5_after:
        print("[phase1] *** WARNING: checkpoint MD5 CHANGED -- this should "
              "never happen (no writes were performed) ***", flush=True)
    else:
        print("[phase1] MD5 unchanged, confirmed read-only.", flush=True)

    print(f"[phase1] DONE. results in {out_csv}", flush=True)


if __name__ == "__main__":
    main()
