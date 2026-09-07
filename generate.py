"""Generate a human-like mouse trajectory between two screen points.

This is the front door to the served MIME model. Give it a start point and
an end point in pixels; it returns a trajectory that begins exactly at the
start, ends exactly at the end, and moves like a person in between.

Each requested trajectory is ONE draw from the autoregressive event recipe of
research/w4_mserve.py arm mq1: a commanded duration sampled from the pool of
held out human rows nearest the requested distance, a first event drawn by
the dedicated first event head, then the autoregressive model free running
from that seed at its served temperatures. There is no oversampling and no
selection: what the model draws is what you get. The events are decoded into
a path, which is then rotated and scaled around the start point so the last
point lands exactly on the target; typically a few percent, which leaves the
path's character intact. Pass land=False (or --no-land) to see the raw
decoded path.

Usage:
    python generate.py 200 600 1500 300
    python generate.py 200 600 1500 300 --n 5 --seed 7 --format csv
    python generate.py 200 600 1500 300 --plot out.png

Output (default JSON, one object per point, the shape a replay client wants):
    [{"x": 200, "y": 600, "delay": 0}, {"x": 235, "y": 602, "delay": 12.2}, ...]

`delay` is the wait in milliseconds before moving to that point. Feed the
points to any input-automation layer as-is.

From Python:
    from generate import generate
    traj = generate(200, 600, 1500, 300)          # one (m, 3) array of x, y, t_seconds
    trajs = generate(200, 600, 1500, 300, n=5)    # list of five

Seeds: with seed=g the commanded durations come from numpy's default_rng(g)
and both sampling draws, first event then the rest, come from torch's
generator seeded once with g. With no seed both are drawn from system
entropy. The same seed and the same request reproduce the same trajectory.

Needs `training/event_ar_hm_mlp.pt`, `training/firsthead_q.pt` and
`training/duration_pool.npy`; `python setup_data.py` downloads all three.
The first call loads the models; after that a CPU run takes well under a
second per trajectory.

A row whose event stream decodes to fewer than two events is drawn again,
up to three attempts. That is a resample of a degenerate row, not a quality
selection: no draw is ever compared against another.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models.event_ar import EventARModel
    from models.firsthead import FirstHead

_TRAIN_DIR = Path(os.environ.get("TRAIN_DIR", "./training"))

# Sequence length of both served checkpoints and width of the force tensors.
MAX_T = 256

# Served sampling temperatures of the autoregressive model: speed, turn, dt.
# research/w4_mserve.py arm mq1 and every w4 arm since w4_occupancy sampled
# the contract at these.
AR_TEMPS = (0.95, 0.90, 1.00)


@dataclass
class Serve:
    """Everything generate() needs, built once and kept.

    model is the autoregressive event model in eval mode, q the first event
    head in eval mode, pool the (N, 2) array of held out human rows sorted by
    column 0 (log distance, log duration), device where the models live.
    """

    model: "EventARModel"
    q: "FirstHead"
    pool: np.ndarray
    device: str


_SERVE: Serve | None = None


def load_serve(device: str | None = None) -> Serve:
    """Load the served models once and return the same Serve thereafter.

    The first call fixes the device ("cuda" if available else "cpu", or the
    one given) and every later call returns that cached instance.
    """
    global _SERVE
    if _SERVE is None:
        import torch

        from models.event_ar import EventARModel
        from models.firsthead import FirstHead

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(_TRAIN_DIR / "event_ar_hm_mlp.pt", map_location=dev,
                        weights_only=True)
        model = EventARModel(**ck["config"]).to(dev).eval()
        model.load_state_dict(ck["model_state_dict"])
        qk = torch.load(_TRAIN_DIR / "firsthead_q.pt", map_location=dev,
                        weights_only=True)
        q = FirstHead(**qk["config"]).to(dev).eval()
        q.load_state_dict(qk["model_state_dict"])
        _SERVE = Serve(model=model, q=q, pool=np.load(_TRAIN_DIR / "duration_pool.npy"),
                       device=dev)
    return _SERVE


def draw_log_durations(log_dist: np.ndarray, rng: np.random.Generator,
                       k: int = 64) -> np.ndarray:
    """One commanded log duration per requested log distance, in order.

    The durmatch matcher of research/w4_mserve.py, verbatim: empirical
    p(log dur | log dist) from held out human rows in the duration pool. For
    each request, sample one of the k nearest pool rows by |log distance
    difference| and take its log duration verbatim. One draw per request,
    no selection.
    """
    serve = load_serve()
    pd, pdur = serve.pool[:, 0], serve.pool[:, 1]
    out = np.empty(len(log_dist), dtype=np.float64)
    for i, ld in enumerate(log_dist):
        j = np.searchsorted(pd, ld)
        lo, hi = max(0, j - k), min(len(pd), j + k)
        cand = np.arange(lo, hi)
        near = cand[np.argsort(np.abs(pd[cand] - ld), kind="stable")[:k]]
        out[i] = pdur[near[rng.integers(len(near))]]
    return out


def first_event_force(q: "FirstHead", cond) -> tuple:
    """Draw the first event from q and wrap it as a force tuple for
    EventARModel.sample: PAD/NULL/zero streams with column 0 replaced by the
    draw, and a mask making exactly that column forced."""
    import torch

    from models.event_ar import DT_MAX_MS
    from models.event_stream_polar import S_PAD_CLASS, TH_NULL_CLASS

    qs, qth, qdt = q.sample(cond, 1.0, 1.0, 1.0)
    b = cond.shape[0]
    fs = torch.full((b, MAX_T), S_PAD_CLASS, device=cond.device, dtype=torch.long)
    fth = torch.full((b, MAX_T), TH_NULL_CLASS, device=cond.device, dtype=torch.long)
    fdt = torch.zeros((b, MAX_T), device=cond.device, dtype=torch.long)
    fs[:, 0], fth[:, 0], fdt[:, 0] = qs, qth, qdt.clamp(max=DT_MAX_MS)
    mask = torch.zeros((b, MAX_T), device=cond.device, dtype=torch.bool)
    mask[:, 0] = True
    return fs, fth, fdt, mask


def _check_assets() -> None:
    missing = [
        p.name
        for p in (_TRAIN_DIR / "event_ar_hm_mlp.pt", _TRAIN_DIR / "firsthead_q.pt",
                  _TRAIN_DIR / "duration_pool.npy")
        if not p.exists()
    ]
    if missing:
        sys.exit(
            f"Missing {', '.join(missing)} in {_TRAIN_DIR}/.\n"
            "Run `python setup_data.py` first to download the model and its data."
        )


def _land_on_target(traj: np.ndarray, ex: float, ey: float) -> np.ndarray:
    """Rotate and scale a path around its start so its last point is (ex, ey).

    A similarity transform: multiply each point's offset from the start by the
    complex ratio (desired end vector) / (raw end vector). Preserves every
    turn angle and every relative timing; only the overall heading and length
    change, both by whatever small amount the raw path missed by.
    """
    start = traj[0, :2]
    raw_vec = complex(*(traj[-1, :2] - start))
    if abs(raw_vec) < 1e-9:
        return traj
    ratio = complex(ex - start[0], ey - start[1]) / raw_vec
    offsets = (traj[:, 0] - start[0]) + 1j * (traj[:, 1] - start[1])
    moved = offsets * ratio
    out = traj.copy()
    out[:, 0] = start[0] + moved.real
    out[:, 1] = start[1] + moved.imag
    out[-1, :2] = [ex, ey]  # kill floating-point residue on the endpoint
    return out


def generate(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    *,
    n: int = 1,
    seed: int | None = None,
    land: bool = True,
):
    """Generate trajectories from (start_x, start_y) to (end_x, end_y).

    Returns one (m, 3) float array of columns [x, y, t_seconds] when n == 1,
    or a list of n such arrays. With land=True (default) each path ends
    exactly on the target; with land=False you get the raw decoded model
    output. One draw per requested trajectory, no candidate selection. A row
    that decodes to fewer than 2 events is resampled, up to 3 attempts: that
    is a redo of a degenerate draw, not a choice between candidates.
    """
    _check_assets()
    dist = math.hypot(end_x - start_x, end_y - start_y)
    if dist < 1e-6:
        raise ValueError("start and end must differ by at least 1e-6 pixels")
    ang = math.atan2(end_y - start_y, end_x - start_x)
    log_dist = math.log(dist)

    # torch, and the model modules that pull it in, are imported inside the
    # functions that need them so --help, --help-adjacent failures and the
    # asset check above stay cheap.
    import torch

    from models.event_ar import class_to_dt_ms
    from models.event_stream_polar import decode_events

    serve = load_serve()
    if seed is not None:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
    else:
        rng = np.random.default_rng()

    paths: list[np.ndarray | None] = [None] * n

    def draw(rows: list[int]) -> None:
        """Sample and decode one event stream per index in rows, in place.

        All of a request's rows share the one spec, so every pass is a single
        batched call: durations, condition, first event, autoregressive body.
        """
        cond = np.stack([
            np.full(len(rows), log_dist),
            draw_log_durations(np.full(len(rows), log_dist), rng),
            np.full(len(rows), math.cos(ang)),
            np.full(len(rows), math.sin(ang)),
        ], 1).astype(np.float32)
        cond_t = torch.from_numpy(cond).to(serve.device)
        force = first_event_force(serve.q, cond_t)
        with torch.no_grad():
            s_cls, th_cls, dt_cls = serve.model.sample(
                cond_t, temperature=AR_TEMPS[0], th_temperature=AR_TEMPS[1],
                dt_temperature=AR_TEMPS[2], force=force, kv_cache=True)
        s_np, th_np, dt_np = (s_cls.cpu().numpy(), th_cls.cpu().numpy(),
                              dt_cls.cpu().numpy())
        for row, i in enumerate(rows):
            dt_ms = class_to_dt_ms(torch.from_numpy(dt_np[row])).numpy()
            paths[i] = decode_events(s_np[row], th_np[row], dt_ms,
                                     start_x, start_y, ang)

    pending = list(range(n))
    for _ in range(4):  # the initial draw plus up to 3 resamples of bad rows
        if not pending:
            break
        draw(pending)
        pending = [i for i, p in enumerate(paths) if p is None]
    if pending:
        raise RuntimeError(
            f"{len(pending)} of {n} row(s) decoded to fewer than 2 events on "
            "4 draws each; try a different seed")

    out = [_land_on_target(p, end_x, end_y) if land else p for p in paths]
    return out[0] if n == 1 else out


def _as_points(traj: np.ndarray) -> list[dict]:
    """Convert an (m, 3) [x, y, t_seconds] array to per-point delay records."""
    t_ms = traj[:, 2] * 1000.0
    delays = np.diff(t_ms, prepend=t_ms[0])
    return [
        {"x": round(float(x), 2), "y": round(float(y), 2), "delay": round(float(d), 2)}
        for x, y, d in zip(traj[:, 0], traj[:, 1], delays)
    ]


def _print_csv(trajs: list[np.ndarray]) -> None:
    print("traj,x,y,delay_ms")
    for i, traj in enumerate(trajs):
        for p in _as_points(traj):
            print(f"{i},{p['x']},{p['y']},{p['delay']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a human-like mouse trajectory between two points.",
    )
    parser.add_argument("start_x", type=float)
    parser.add_argument("start_y", type=float)
    parser.add_argument("end_x", type=float)
    parser.add_argument("end_y", type=float)
    parser.add_argument("--n", type=int, default=1, help="How many to generate (default 1)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument(
        "--format", choices=["json", "csv"], default="json", help="Output format (default json)"
    )
    parser.add_argument(
        "--no-land",
        action="store_true",
        help="Return raw model output instead of landing exactly on the target",
    )
    parser.add_argument("--plot", metavar="PATH", help="Also save a PNG picture of the paths")
    args = parser.parse_args(argv)

    result = generate(
        args.start_x, args.start_y, args.end_x, args.end_y,
        n=args.n, seed=args.seed, land=not args.no_land,
    )
    trajs = result if isinstance(result, list) else [result]

    if args.format == "csv":
        _print_csv(trajs)
    else:
        payload = [_as_points(t) for t in trajs]
        print(json.dumps(payload[0] if args.n == 1 else payload))

    if args.plot:
        _save_plot(trajs, args.start_x, args.start_y, args.end_x, args.end_y, args.plot)
        print(f"Saved {args.plot}", file=sys.stderr)


def _save_plot(trajs, sx, sy, ex, ey, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for traj in trajs:
        ax.plot(traj[:, 0], traj[:, 1], "-", color="#4040E0", lw=1.0, alpha=0.5)
        ax.plot(traj[:, 0], traj[:, 1], ".", color="#4040E0", ms=3)
    ax.plot([sx], [sy], "o", color="#2CB25C", ms=10, label="start")
    ax.plot([ex], [ey], "X", color="#E04040", ms=11, label="end")
    ax.invert_yaxis()  # screen coordinates: y grows downward
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.set_title(f"({sx:.0f}, {sy:.0f}) to ({ex:.0f}, {ey:.0f})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
