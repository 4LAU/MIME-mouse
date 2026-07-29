"""W0 Task A: serve-latency benchmark for candi_polar_flow_best.pt.

Measures wall-clock to produce ONE final served trajectory through the
standard serving recipe (steps=200, guide=0.15, perp=0.85, correct=rotate --
research/phase_a_baseline.py's generate_paths, the exact 0.752/0.757
generation path). Does NOT modify the checkpoint or any training script;
only imports research/phase_a_baseline.py's functions (load_model,
generate_paths, make_specs) and times them.

Batch sizes 1/8/16/32: "same spec repeated, one batched sampler call" --
K copies of the SAME (start,end) spec passed to generate_paths() in a single
call. Duration is still sampled per item (DurationModel has no fixed seed),
so items can land in different seq_len groups inside generate_paths, exactly
as they would for K independently-sampled candidates of one real request.

Reports, separately: model load time (paid once by a client) and warm
per-call generation time for batch sizes [1, 8, 16, 32] (median of N_REPEAT
timed calls, after one untimed warm-up call).

Usage:
    .venv/Scripts/python.exe research/w0_latency.py --device-tag gpu
    CUDA_VISIBLE_DEVICES="" .venv/Scripts/python.exe research/w0_latency.py --device-tag cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from experiments._common import DurationModel  # noqa: E402
from phase_a_baseline import load_model, generate_paths, make_specs, DUR_STD  # noqa: E402

TRAIN_DIR = REPO_ROOT / "training"
RESEARCH_DIR = REPO_ROOT / "research"
BATCH_SIZES = [1, 8, 16, 32]
N_REPEAT = 5


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-tag", required=True,
                     help="Label for the output file/JSON, e.g. gpu or cpu "
                          "(does not force device; that is controlled by "
                          "CUDA_VISIBLE_DEVICES in the environment)")
    ap.add_argument("--n-repeat", type=int, default=N_REPEAT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[w0_latency] cuda_available={torch.cuda.is_available()} "
          f"device_tag={args.device_tag}", flush=True)

    t0 = time.perf_counter()
    model, data_scale, device, max_seq_len_cfg = load_model()
    model.max_seq_len_cfg = max_seq_len_cfg
    sync(device)
    load_elapsed = time.perf_counter() - t0
    print(f"[w0_latency] model load time: {load_elapsed:.3f}s (device={device})",
          flush=True)

    duration_model = DurationModel(TRAIN_DIR, std_mult=DUR_STD)

    # One fixed spec (mid-range distance, seed 42), repeated per batch size.
    base_spec = make_specs(1, args.seed)[0]

    # Untimed warm-up: JIT/cuDNN autotune, first-call allocator overhead.
    _ = generate_paths(model, data_scale, device, duration_model, [base_spec] * 4)
    sync(device)

    results = {}
    for K in BATCH_SIZES:
        specs = [base_spec] * K
        times = []
        for r in range(args.n_repeat):
            sync(device)
            t0 = time.perf_counter()
            trajs = generate_paths(model, data_scale, device, duration_model, specs)
            sync(device)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            n_valid = sum(1 for t in trajs if t is not None and len(t) >= 2)
            print(f"[w0_latency] K={K:2d} rep={r}: {elapsed:.4f}s "
                  f"(valid {n_valid}/{K}, {elapsed / K * 1000:.1f} ms/spec)",
                  flush=True)
        times.sort()
        median = times[len(times) // 2]
        results[str(K)] = {
            "batch_size": K,
            "all_times_sec": times,
            "median_sec": median,
            "mean_sec": sum(times) / len(times),
            "min_sec": min(times),
            "max_sec": max(times),
            "median_ms_per_spec": median / K * 1000,
        }
        print(f"[w0_latency] K={K:2d} median={median:.4f}s "
              f"({median / K * 1000:.1f} ms/spec)", flush=True)

    out = {
        "device_tag": args.device_tag,
        "torch_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "model_load_sec": load_elapsed,
        "n_repeat": args.n_repeat,
        "seed": args.seed,
        "generation_config": "steps=200 guide=0.15 perp=0.85 correct=rotate cfg=0.0 "
                              "(research/phase_a_baseline.py, the published "
                              "0.752/0.757 recipe)",
        "checkpoint": "candi_polar_flow_best.pt (untouched, MD5 "
                       "91326a29750789f3167055324ef377c5)",
        "batches": results,
    }
    out_path = RESEARCH_DIR / f"w0_latency_{args.device_tag}.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[w0_latency] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
