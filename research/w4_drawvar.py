"""How much does the contract score move between sampler draws from one frozen
ROLLOUT checkpoint at seq_len 256? Registered before it ran.

WHY THIS IS THE FIRST GPU JOB

Every verdict in HANDOFF.md about which rollout objective is better rests on a
number nobody has measured. Two figures are in the repository and they disagree
by a factor of two.

  w4_seedvar.json, frozen base checkpoint, seq_len 256, 3 torch seeds   sd 0.0059
  four draws of w4_rollout_pilot_zbuf_step100.pt, 0.5894 0.6050 0.6181
  0.6206                                                                sd 0.0143

A standard deviation from three or four samples is almost uninformative, so
these are two weak estimates and not an established disagreement. The comment in
w4_rollout.py's --eval-draws help quotes sd 0.0141 and credits it to w4_seedvar,
which does not contain that figure. This run replaces all of it with one
estimate from eight draws.

It matters because the five best arm endpoints span 0.5949 to 0.5997, a range of
0.0048. At either candidate sd no arm is distinguishable from any other, so the
plateau verdict does not depend on the answer. What does depend on it is how
fine a threshold any future registration may use, and whether the 0.0221 gap
between two evaluations of one checkpoint was ordinary or worth chasing.

THE ARM. One checkpoint, w4_rollout_pilot_zbuf_step100.pt, seq_len 256, eight
torch seeds, the same cond rows and the same scoring shuffle throughout. The
only thing that varies is the sampler draw.

THE ALIGNMENT CONTROL, free and read first. w4_rollout builds its eval rows as
perm[4000:6500] from default_rng(17) over the same ok mask this script uses, so
a fresh seed 17 draw here should reproduce the critic arm's first base draw of
0.6181 to four decimals. If it does, the two scripts agree and the four existing
draws pool with these eight for twelve. If it does not, they measure slightly
different things and nothing is pooled. Either outcome is reported.

PREDICTION, fixed before the run: sd lands between 0.006 and 0.014, nearer
0.010. On that figure the 0.0221 same checkpoint gap is about two draw standard
deviations and is ordinary.

FALSIFIER: sd at or below 0.0075 would mean the four draw 0.0143 was an unlucky
spread and the base checkpoint figure transfers to fine tuned checkpoints. sd at
or above 0.018 would mean every eval to eval difference this workstream has
quoted is smaller than its own error bar, including several already written down
as findings, and those would have to be restated.

WHAT THIS CANNOT SETTLE. It measures one checkpoint. The tempting reading, that
fine tuned checkpoints are noisier draw to draw than the frozen base, needs the
base measured with the same number of seeds and is not tested here.

One trajectory per cond row, no selection, no best of. Tokens are cached per
draw so a thermal stop costs at most one draw. The protected eval sample is
never read and no model file is written.
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
# w4_seedvar.py imports feature_matrix and to_paths from here. Neither name
# exists in w4_rollout any more, so w4_seedvar cannot run as written; the
# current entry point is decode_batch, which returns the contract features of
# the surviving rows directly.
from w4_rollout import decode_batch, gpu_temp  # noqa: E402

D = "training"
CKPT = "research/w4_rollout_pilot_zbuf_step100.pt"
SCR = os.environ.get("W4_CACHE", "/tmp/w4_cache")
os.makedirs(SCR, exist_ok=True)
OUT = "research/w4_drawvar.json"
TAG = "zbuf100"    # part of the token cache key, so a different checkpoint
                   # can never silently reuse another one's cached draws
SEED = 17          # cond selection and scoring shuffle, held fixed throughout
HUMAN_N = 4000
EVAL_N = 2500
SEQ_LEN = 256
BATCH = 96
SEEDS = (17, 23, 31, 37, 41, 43, 47, 53)

# the critic arm's first base draw from this same checkpoint, which a matching
# seed 17 draw here should reproduce if the two scripts are aligned
ALIGN_REF = 0.6181
PRIOR_DRAWS = [0.5894, 0.6050, 0.6181, 0.6206]

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


def draw(tseed):
    path = f"{SCR}/drawvar_tok_{TAG}_{SEQ_LEN}_{tseed}.npz"
    if os.path.exists(path):
        z = np.load(path)
        print(f"  reusing cached seed {tseed}", flush=True)
        return z["s"], z["th"], z["dt"]
    S, TH, DT = [], [], []
    torch.manual_seed(tseed)
    t0 = time.time()
    for c0 in range(0, len(eval_cond), BATCH):
        gate()
        c = eval_cond[c0:c0 + BATCH].to(dev)
        s, th, dt = model.sample(c, seq_len=SEQ_LEN)
        S.append(s.cpu().numpy()); TH.append(th.cpu().numpy())
        DT.append(dt.cpu().numpy())
    S = np.concatenate(S); TH = np.concatenate(TH); DT = np.concatenate(DT)
    np.savez_compressed(path, s=S, th=TH, dt=DT)
    print(f"  sampled seed {tseed} in {time.time() - t0:.0f}s", flush=True)
    return S, TH, DT


print(f"\n  checkpoint {CKPT}")
print(f"  seq_len {SEQ_LEN}, {len(SEEDS)} torch seeds, {EVAL_N} cond rows, "
      f"scoring shuffle fixed at {SEED}\n", flush=True)

rows = []
for tseed in SEEDS:
    S, TH, DT = draw(tseed)
    X, keep, _ = decode_batch(list(S), list(TH), list(DT), eval_ang)
    np.random.default_rng(SEED).shuffle(X)
    r = scoring.score_features(X)
    rows.append({"torch_seed": tseed, "auc": float(r["auc_rf_oob"]),
                 "n": int(len(X))})
    print(f"  seed {tseed:>3}  auc {rows[-1]['auc']:.4f}  n {rows[-1]['n']}  "
          f"{gpu_temp()}C", flush=True)
    # written after every draw, not once at the end, so that stopping this at
    # the edge of an authorised window still leaves every completed draw on
    # disk. The tokens are cached too, so a later run adds seeds without
    # resampling the ones already done.
    with open(OUT, "w") as f:
        json.dump({"checkpoint": CKPT, "seq_len": SEQ_LEN, "rows": rows,
                   "complete": len(rows) == len(SEEDS)}, f, indent=2)

v = np.array([r["auc"] for r in rows])
sd = float(v.std(ddof=1))
se_of_sd = sd / (2 * (len(v) - 1)) ** 0.5

print(f"\n  mean {v.mean():.4f}  sd {sd:.4f}  se of that sd {se_of_sd:.4f}  "
      f"range {v.max() - v.min():.4f}  over {len(v)} draws")

a0 = v[0]
aligned = abs(a0 - ALIGN_REF) < 5e-4
print(f"\n  alignment control, seed 17 here {a0:.4f} against the critic arm's "
      f"first base draw {ALIGN_REF:.4f}")
if aligned:
    pooled = np.concatenate([v, np.array(PRIOR_DRAWS)])
    print(f"  aligned, so the four earlier draws pool: {len(pooled)} draws, "
          f"mean {pooled.mean():.4f} sd {pooled.std(ddof=1):.4f}")
else:
    pooled = None
    print("  NOT aligned, so nothing is pooled and only the eight draws above "
          "are used")

print(f"\n  the two figures this replaces      0.0059 (n=3)   0.0143 (n=4)")
print(f"  the five best arm endpoints span   0.0048")
print(f"  standard error of a two draw mean  {sd / 2 ** 0.5:.4f}")
print(f"  of the difference of two such      {sd:.4f}")
if sd <= 0.0075:
    print("  FALSIFIER SIDE A. the base checkpoint figure transfers and the "
          "0.0143 was an unlucky spread.")
if sd >= 0.018:
    print("  FALSIFIER SIDE B. every eval to eval difference quoted in this "
          "workstream is inside its own error bar and must be restated.")
print(f"\n  peak {peak}C, {cooled_s / 60:.1f} min cooling")

with open(OUT, "w") as f:
    json.dump({"checkpoint": CKPT, "seq_len": SEQ_LEN, "rows": rows,
               "mean": float(v.mean()), "sd": sd, "se_of_sd": float(se_of_sd),
               "n_draws": int(len(v)), "aligned": bool(aligned),
               "align_seed17": float(a0), "align_ref": ALIGN_REF,
               "pooled_sd": (float(pooled.std(ddof=1)) if pooled is not None
                             else None),
               "peak_temp_c": peak, "cooldown_min": round(cooled_s / 60, 1)},
              f, indent=2)
print(f"  wrote {OUT}")
