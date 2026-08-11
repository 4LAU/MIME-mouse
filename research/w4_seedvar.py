"""How much does the contract score move between two sampler draws from the
SAME frozen model? Registered before it ran.

WHY THIS IS THE IMPORTANT ONE

Two noise figures are already recorded. Neither covers this.

  same rows, reshuffled, forest randomness only        sd 0.0041
  disjoint halves, adds finite sample                  sd 0.0072

Both hold the generated trajectories fixed. Neither prices what happens when the
sampler runs again. w4_rollout's evaluate() does NOT reset the torch seed, so
every evaluation inside a training run draws a fresh set of trajectories. That
means sampler draw variance is inside every eval to eval difference this
workstream has ever quoted, including the energy arm's 0.0863 fall, which was
reported as roughly twelve standard deviations on the 0.0072 figure. If sampler
draw variance is large, that claim is wrong and has to be restated.

The budget probe made this unavoidable. It changed seq_len and found the first
token differed in 2493 of 2500 rows, so changing the budget also changed the
draw. That probe therefore confounds two effects and cannot settle either. This
one separates them.

THE ARMS, all from the frozen base checkpoint event_ar_v2_s40000, the SAME cond
rows, the SAME scoring shuffle, varying ONLY the torch seed

  seq_len 160, seeds 17 23 31 37 41     the setting every rollout arm uses
  seq_len 256, seeds 17 23 31           the setting the ladder was measured on

The spread within each block is sampler draw variance, which is the error bar
this workstream actually needs. The difference between block means is the budget
effect, now measured against that error bar instead of against one draw.

PREDICTION, fixed before the run: sampler draw variance is larger than 0.0072
but well under the 0.0863 the energy arm moved, so the energy result survives
with a smaller effect size. The budget difference shrinks toward nothing once
compared against the right error bar.

FALSIFIER: sampler draw sd at or above about 0.025 at seq_len 160. That would
put the energy arm's fall under four standard deviations and would mean every
single eval number in this workstream needs repeat draws before it can be read.

One trajectory per cond row, no selection, no best of. Tokens are cached per
arm so a thermal stop costs at most one arm. The protected eval sample is never
read and no model file is written.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("EVENT_SNAP", "2.5")
for p in (".", "research", "research/autoloop"):
    if p not in sys.path:
        sys.path.insert(0, p)

import scoring  # noqa: E402
from models.event_ar import EventARModel  # noqa: E402
from w4_rollout import feature_matrix, gpu_temp, to_paths  # noqa: E402

D = "training"
CKPT = f"{D}/event_ar_v2_s40000.pt"
SCR = os.environ.get("W4_CACHE", "/tmp/w4_cache")
os.makedirs(SCR, exist_ok=True)
OUT = "research/w4_seedvar.json"
SEED = 17          # cond selection and scoring shuffle, held fixed throughout
HUMAN_N = 4000
EVAL_N = 2500
BATCH = 96
ARMS = [(160, s) for s in (17, 23, 31, 37, 41)] + [(256, s) for s in (17, 23, 31)]

KILL_C = 79
COOL_C = 74
RESUME_C = 70
COOL_MAX_S = 300

dev = "cuda" if torch.cuda.is_available() else "cpu"

rng = np.random.default_rng(SEED)
ok = np.flatnonzero(np.load(f"{D}/events_feat18_ok.npy"))
perm = ok[rng.permutation(len(ok))]
eval_rows = perm[HUMAN_N:HUMAN_N + 400000][:EVAL_N]
cond_all = np.load(f"{D}/events_cond.npy", mmap_mode="r")
eval_cond = torch.tensor(np.asarray(cond_all[np.sort(eval_rows)], dtype=np.float32))
eval_ang = np.arctan2(eval_cond[:, 3].numpy().astype(np.float64),
                      eval_cond[:, 2].numpy().astype(np.float64))

ck = torch.load(CKPT, map_location=dev, weights_only=False)
model = EventARModel(**ck["config"]).to(dev)
model.load_state_dict(ck["model_state_dict"])
model.eval()

cooled_s = 0.0
peak = 0


def gate():
    global cooled_s, peak
    t = gpu_temp()
    peak = max(peak, t)
    if t >= COOL_C:
        c0 = time.time()
        while gpu_temp() > RESUME_C and time.time() - c0 < COOL_MAX_S:
            time.sleep(10)
        cooled_s += time.time() - c0
        t = gpu_temp()
        peak = max(peak, t)
    if t >= KILL_C:
        raise SystemExit(f"GPU {t}C, at or above the {KILL_C}C kill. Stopping.")


def draw(seq_len, tseed):
    path = f"{SCR}/seedvar_tok_{seq_len}_{tseed}.npz"
    if os.path.exists(path):
        z = np.load(path)
        print(f"  reusing cached seq_len {seq_len} seed {tseed}", flush=True)
        return z["s"], z["th"], z["dt"]
    S, TH, DT = [], [], []
    torch.manual_seed(tseed)
    t0 = time.time()
    for c0 in range(0, len(eval_cond), BATCH):
        gate()
        c = eval_cond[c0:c0 + BATCH].to(dev)
        s, th, dt = model.sample(c, seq_len=seq_len)
        S.append(s.cpu().numpy()); TH.append(th.cpu().numpy())
        DT.append(dt.cpu().numpy())
    S = np.concatenate(S); TH = np.concatenate(TH); DT = np.concatenate(DT)
    np.savez_compressed(path, s=S, th=TH, dt=DT)
    print(f"  sampled seq_len {seq_len} seed {tseed} in {time.time() - t0:.0f}s",
          flush=True)
    return S, TH, DT


rows = []
for seq_len, tseed in ARMS:
    S, TH, DT = draw(seq_len, tseed)
    paths, keep = to_paths(list(S), list(TH), list(DT), eval_ang)
    X, xok = feature_matrix(paths)
    X = X[xok]
    np.random.default_rng(SEED).shuffle(X)
    r = scoring.score_features(X)
    rows.append({"seq_len": seq_len, "torch_seed": tseed,
                 "auc": float(r["auc_rf_oob"]), "n": int(len(X))})
    print(f"  seq_len {seq_len}  seed {tseed:>3}  auc {rows[-1]['auc']:.4f}  "
          f"n {rows[-1]['n']}", flush=True)

print()
blocks = {}
for L in (160, 256):
    v = np.array([r["auc"] for r in rows if r["seq_len"] == L])
    blocks[L] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                 "n": int(len(v)), "min": float(v.min()), "max": float(v.max())}
    print(f"  seq_len {L}   mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
          f"range {v.max() - v.min():.4f}  over {len(v)} draws")

sd160 = blocks[160]["sd"]
diff = blocks[160]["mean"] - blocks[256]["mean"]
print(f"\n  sampler draw sd at the setting the arms use   {sd160:.4f}")
print(f"  previously quoted error bar                   0.0072")
print(f"  budget difference                             {diff:+.4f}, "
      f"{abs(diff) / max(sd160, 1e-9):.1f} sd")
print(f"\n  the energy arm moved 0.0863, which is "
      f"{0.0863 / max(sd160, 1e-9):.1f} sd on this figure")
if sd160 >= 0.025:
    print("  FALSIFIER TRIPPED. every eval number needs repeat draws.")
print(f"\n  peak {peak}C, {cooled_s / 60:.1f} min cooling")

with open(OUT, "w") as f:
    json.dump({"rows": rows, "blocks": {str(k): v for k, v in blocks.items()},
               "peak_temp_c": peak, "cooldown_min": round(cooled_s / 60, 1)},
              f, indent=2)
print(f"  wrote {OUT}")
