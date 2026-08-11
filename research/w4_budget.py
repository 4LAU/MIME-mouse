"""Does the sampling budget advantage survive training, and where does the
energy checkpoint actually sit against the corpus floor? Registered before it ran.

WHY

Two facts now established on the FROZEN base checkpoint, each over repeat draws:

  sampler draw sd, either budget                    0.0060
  seq_len 160 mean over 5 draws                     0.6746
  seq_len 256 mean over 3 draws                     0.6301
  the budget is worth                               0.0444, 7.4 sd

Every rollout arm samples at seq_len equal to its cap, 160, which is the worse
of the two. The w4_gapsplit ladder was measured at 256. So the energy arm's
endpoint of 0.5949 and the ladder's corpus floor of 0.5455 are NOT on the same
scale and must not be compared, which is the comparison the write up wanted to
make.

If the 0.0444 offset carried over unchanged, the energy checkpoint would sit
near 0.551 at the ladder's setting, which is essentially the corpus floor. That
would be a large claim resting on an offset measured only on an untrained model,
so it gets measured directly instead of assumed.

THE ARMS, all from research/w4_rollout_pilot_energy.pt, the SAME cond rows as
the training run used, two sampler draws each

  seq_len 160, seeds 17 23     reproduces the arm's own reported endpoint
  seq_len 256, seeds 17 23     the ladder's setting

PREDICTION, fixed before the run: the 160 block reproduces 0.5949 inside about
0.006. The 256 block lands below it. Whether it reaches the 0.5455 floor is the
open question and the prediction is deliberately not committed on it.

FALSIFIER: the 256 block scores at or ABOVE the 160 block. That would mean the
budget advantage is a property of the untrained model that training destroys,
and it would make the energy checkpoint worse at the serving budget than at the
one it trained on, which matters more than the ladder comparison does.

One trajectory per cond row, no selection, no best of. Thermally gated, tokens
cached per arm. The protected eval sample is never read and no model file is
written.
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
CKPT = "research/w4_rollout_pilot_energy.pt"
SCR = os.environ.get("W4_CACHE", "/tmp/w4_cache")
os.makedirs(SCR, exist_ok=True)
OUT = "research/w4_budget.json"
SEED = 17
HUMAN_N = 4000
EVAL_N = 2500
BATCH = 96
ARMS = [(160, 17), (160, 23), (256, 17), (256, 23)]

KILL_C = 79
COOL_C = 74
RESUME_C = 70
COOL_MAX_S = 300

# measured on the frozen base checkpoint, for reference in the printout
BASE_160, BASE_256, SD = 0.6746, 0.6301, 0.0060
FLOOR_256 = 0.5455          # w4_gapsplit rung C, the corpus itself
RUNG_A, RUNG_B = 0.6061, 0.5826

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
print(f"loaded {CKPT}", flush=True)

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
    path = f"{SCR}/trained_tok_{seq_len}_{tseed}.npz"
    if os.path.exists(path):
        z = np.load(path)
        print(f"  reusing cached {seq_len}/{tseed}", flush=True)
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
    print(f"  sampled {seq_len}/{tseed} in {time.time() - t0:.0f}s", flush=True)
    return S, TH, DT


rows = []
for seq_len, tseed in ARMS:
    S, TH, DT = draw(seq_len, tseed)
    paths, keep = to_paths(list(S), list(TH), list(DT), eval_ang)
    X, xok = feature_matrix(paths)
    X = X[xok]
    np.random.default_rng(SEED).shuffle(X)
    r = scoring.score_features(X)
    lens = np.array([len(p) for p in paths], dtype=np.float64)
    rows.append({"seq_len": seq_len, "torch_seed": tseed,
                 "auc": float(r["auc_rf_oob"]), "n": int(len(X)),
                 "mean_len": float(lens.mean())})
    print(f"  seq_len {seq_len}  seed {tseed:>3}  auc {rows[-1]['auc']:.4f}  "
          f"n {rows[-1]['n']}  mean len {rows[-1]['mean_len']:.1f}", flush=True)

m = {}
for L in (160, 256):
    v = np.array([r["auc"] for r in rows if r["seq_len"] == L])
    m[L] = float(v.mean())
    print(f"\n  trained, seq_len {L}   mean {v.mean():.4f}  over {len(v)} draws")

print(f"\n  the arm reported 0.5949 at seq_len 160, this reads {m[160]:.4f}")
print(f"  budget effect on the BASE model      {BASE_160 - BASE_256:+.4f}")
print(f"  budget effect on the TRAINED model   {m[160] - m[256]:+.4f}")
print("\n  on the ladder's own scale, seq_len 256")
print(f"    energy checkpoint                  {m[256]:.4f}")
print(f"    rung A, marginals matched          {RUNG_A:.4f}")
print(f"    rung B, plus correlations          {RUNG_B:.4f}")
print(f"    rung C, the corpus floor           {FLOOR_256:.4f}")
print(f"    distance still to the floor        {m[256] - FLOOR_256:+.4f}, "
      f"{(m[256] - FLOOR_256) / SD:.1f} sd")
if m[256] >= m[160]:
    print("\n  FALSIFIER TRIPPED. training destroyed the budget advantage.")

print(f"\n  peak {peak}C, {cooled_s / 60:.1f} min cooling")
with open(OUT, "w") as f:
    json.dump({"rows": rows, "mean_160": m[160], "mean_256": m[256],
               "peak_temp_c": peak, "cooldown_min": round(cooled_s / 60, 1)},
              f, indent=2)
print(f"  wrote {OUT}")
