"""Do the two feature pipelines agree on the SAME trajectory?

The model is commanded in one space and judged in another. Training builds the
character command with train_events_polar_dm.detector_features, a GPU
frame-based reimplementation of the 18 statistics. The contract scores with
features.py, which resamples to 125Hz and then extracts. detector_features
documents itself as "differentiable analogs" and argues the approximation bias
cancels because gen and real both go through it. That argument holds for the
MMD term it was written for. It does NOT hold for conditioning: there the
command comes from one pipeline and the grade comes from the other, so any
systematic disagreement is a standing error between what the model is told to
do and what it is measured on.

w4_command_ceiling's control found the checkpoint's bank and features.py human
values disagree hard, worst on time_to_peak_velocity. But that compared
different path sets, so distribution shift was not excluded.

This excludes it. One set of real human paths, both pipelines, column by column:
correlation, and the linear fit realized = a * commanded + b. A column with
r near 1 and a near 1 is a faithful bridge. A column with low r is a command
the model cannot obey no matter how well it trains, because the thing it is
told to produce is not the thing that gets measured.

CPU only, no GPU, no generation.

Usage:
  env PYTHONPATH=. NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX \
    AVX512BW AVX512DQ AVX512VL" \
    ~/venvs/mime/bin/python research/w4_pipeline_agreement.py --n 1500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))

OUT = R / "research" / "w4_pipeline_agreement_results.json"


def frames_from_paths(paths, hz=125.0, max_len=None):
    """Resample exactly as features.py does, then lay the result out as the
    padded (x, y, mask) frame tensors detector_features expects. Same numbers,
    two containers, so nothing but the extractor differs."""
    from features import resample_trajectory

    rs = [np.asarray(resample_trajectory(p, hz), dtype=np.float64) for p in paths]
    keep = [r for r in rs if len(r) >= 4]
    L = max_len or max(len(r) for r in keep)
    n = len(keep)
    x = np.zeros((n, L), dtype=np.float32)
    y = np.zeros((n, L), dtype=np.float32)
    m = np.zeros((n, L), dtype=np.float32)
    for i, r in enumerate(keep):
        k = min(len(r), L)
        x[i, :k] = r[:k, 0]
        y[i, :k] = r[:k, 1]
        m[i, :k] = 1.0
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    t0 = time.time()
    from features import FEATURE_NAMES, extract_features  # noqa: E402
    from phase0_critic import reconstruct_human_val_paths  # noqa: E402
    from train_events_polar_dm import detector_features  # noqa: E402
    from cond_realization_probe import to_detector_space  # noqa: E402

    paths, _ = reconstruct_human_val_paths(args.n, args.seed, verbose=False)
    paths = paths[: args.n]
    print(f"[agree] {len(paths)} real human paths", flush=True)

    x, y, m, keep = frames_from_paths(paths)
    print(f"[agree] frames {tuple(x.shape)}", flush=True)

    with torch.no_grad():
        D = detector_features(x, y, m).cpu().numpy().astype(np.float64)

    # features.py on the SAME resampled paths, then bridged into detector space
    raw = []
    ok = []
    for i, r in enumerate(keep):
        f = extract_features([tuple(v) for v in r])
        if f is not None and np.all(np.isfinite(f)):
            raw.append(f)
            ok.append(i)
    raw = np.asarray(raw, dtype=np.float64)
    F = to_detector_space(raw)
    D = D[ok]
    good = np.all(np.isfinite(D), axis=1) & np.all(np.isfinite(F), axis=1)
    D, F = D[good], F[good]
    print(f"[agree] {len(D)} usable pairs\n", flush=True)

    out = {"n": int(len(D)), "seed": args.seed, "cols": {}}
    print(f"{'feature':<24}{'r':>7}{'slope':>8}{'det mean':>10}"
          f"{'fpy mean':>10}{'det sd':>9}{'fpy sd':>9}")
    for i, f in enumerate(FEATURE_NAMES):
        a, b = D[:, i], F[:, i]
        sa, sb = a.std(), b.std()
        r = float(np.corrcoef(a, b)[0, 1]) if sa > 1e-12 and sb > 1e-12 else float("nan")
        slope = float(np.polyfit(a, b, 1)[0]) if sa > 1e-12 else float("nan")
        out["cols"][f] = {"r": r, "slope": slope, "det_mean": float(a.mean()),
                          "fpy_mean": float(b.mean()), "det_sd": float(sa),
                          "fpy_sd": float(sb)}
        print(f"{f:<24}{r:>7.3f}{slope:>8.3f}{a.mean():>10.3f}"
              f"{b.mean():>10.3f}{sa:>9.3f}{sb:>9.3f}")

    rs = [v["r"] for v in out["cols"].values()]
    out["mean_r"] = float(np.nanmean(rs))
    bad = sorted((v["r"], k) for k, v in out["cols"].items())[:5]
    print(f"\nmean agreement r across 18 columns: {out['mean_r']:.3f}")
    print("worst bridges: " + ", ".join(f"{k} {v:.2f}" for v, k in bad))

    out["wall_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[agree] wrote {args.out} ({out['wall_sec']:.0f}s)")


if __name__ == "__main__":
    main()
