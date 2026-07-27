# Handoff: W1 completeness negative, ready to train

STATUS 2026-07-20: W1 is CLOSED as a clean negative. The fresh-init
winner-only model reads 0.8331 one-shot, far above the 0.70 gate, and is
worse than the 0.6544 baseline it started from. Ledger rows
W1_scratch_2026-07-21T024741+0000_9407d9fd (poolgen),
...T033224+0000_b4996735 (training) and ...T033224+0000_694e74a5 (gate).
Step 4 below was corrected before use; read that note before rerunning
anything here. Next move is the W3 design proposal, still undrafted.

You are resuming a paused research task on this repo (MIME-mouse, WSL2 Ubuntu).
Read this whole file before touching anything. The prior session paused at a
clean point: all data artifacts are built, training has not started.

## What W1 is

PLAN.md (lines 459-598) defines the program. W2 closed with a clean FAIL
(steering generation toward target feature values hit those values without
reducing the two-sample AUC; see
research/autoloop/ledger.jsonl and the memory files). L then ordered the
optional W1 detour: train a FRESH-INIT generator by ordinary supervised
learning on ONLY set-level (trust33) winner trajectories, and check whether a
single draft from it scores materially below 0.70 on the project metric.

Metric (standing, non-negotiable): RF-OOB AUC at N=2000 per class vs
data/human_val_features_grpo.npy, RandomForest n_estimators=100,
oob_score=True, random_state=42. Target 0.50. Headline safety net 0.504/0.513
(do not disturb). Pre-registered W1 gate: one-shot N=2000 RF-OOB materially
below 0.70 is a go signal; at or above it, W1 is a negative and the program
moves to W3.

## Hard rules (verbatim from the standing mandate)

- NEVER modify training/candi_polar_flow_best.pt
  (MD5 91326a29750789f3167055324ef377c5). Verify with md5sum after every run.
- NEVER touch data/human_eval_features.npy.
- NEVER modify scoring code.
- git add files individually, never `git add .`.
- Repo is public: docs must read human-written, no em or en dashes.
- L is non-technical: report what and why in plain terms, brief summary
  first, no praise, max 25 words between tool calls.
- Thermal: launch gate 75C, watchdog kill 83C, cooldown only if peak >= 79C.
- Log every run as a row in research/autoloop/ledger.jsonl via
  research/autoloop/ledger.py append_row (status must be ok/failed/killed).

## Environment

- Repo: /mnt/c/Users/aaron/Code/mouse-trajectory-synthesis
- Python: ~/venvs/mime/bin/python, always with env PYTHONPATH=.
- Check for stray GPU processes BEFORE launching anything, both sides:
  Linux: nvidia-smi. Windows:
  /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe Get-Process
  (WSL cannot see Windows processes any other way).
- Long runs: nohup + research/gpu_watchdog.py --pid <PID> --threshold 83
  --interval 60 --max-minutes 90, log under research/.

## State on disk (all verified except the last item)

- 5 token-bearing candidate pools, seeds 61-65, 2000 specs x K=16 each:
  pool_w1_s{61..65}_k16.npz. Each holds specs/X/owner_idx/trajs plus the raw
  token rows dt_z/s_cls/th_cls/cond aligned row-for-row with X. Generated
  with the locked recipe (EVENT_CKPT=event_polar_4m_fc_v2.pt gumbel snap=2.5
  dur_std=1.0 choice_temp=10 dur_empirical sir=16 sir_temp=0.7
  sir_dur_diverse) plus EVENT_POOL_TOKENS=1. ~39 min each on this GPU.
- trust33 winner picks per pool: pool_w1_s{61..65}_k16_picks_trust33_f20d85_r30_rf.npy
  (2000 rows each; s61 proxy33 hit 0.4856). The extra _f05_ picks files are
  a side product, ignore them.
- Corpus shards in training/: w1_corpus_w1_s{61..64}_k16.npz confirmed
  (2000 winners each, keys dt_z/s_cls/th_cls/cond/length, median length 55).
  The s65 shard was still being written by a background process at pause
  time (about 10 min per shard, slow npz load off the Windows mount).
  FIRST TASK: confirm training/w1_corpus_w1_s65_k16.npz exists and matches
  the others (2000 rows, same keys). If missing, rebuild just it:
    env PYTHONPATH=. ~/venvs/mime/bin/python training/make_w1_corpus.py \
      --pools "pool_w1_s65_k16.npz"

## Code already written for this detour

- experiments/event_stream_polar.py: EVENT_POOL_TOKENS=1 makes poolgen save
  candidate tokens. Leave this env var UNSET for everything below.
- training/make_w1_corpus.py: pools + picks -> corpus shards.
- training/train_events_polar_distill.py: new --fresh-init flag. It builds
  the model from the checkpoint's config but skips loading weights (random
  init), keeps dt_mean/dt_std/feat_mu/feat_sd/feat_bank from the checkpoint,
  and does NOT freeze the dt head.

## Next steps, in order

1. Verify environment: md5sum training/candi_polar_flow_best.pt, GPU idle
   both sides, temperature under 75C.
2. Confirm the s65 corpus shard (above). Total corpus should be 10,000
   winner trajectories across 5 shards.
3. Launch from-scratch training (nohup, watchdog attached):
     env PYTHONPATH=. ~/venvs/mime/bin/python \
       training/train_events_polar_distill.py \
       --load-from event_polar_4m_fc_v2.pt \
       --save-name event_polar_4m_w1_scratch_v1.pt \
       --corpus "w1_corpus_*.npz" \
       --fresh-init --lr 3e-4 --steps 6000 --batch-size 128 \
       --snapshot-every 500 --auto-resume \
       > train_w1_scratch.log 2>&1 &
   Notes: --corpus is relative to --data-dir (training/), and the default
   glob is the old distill corpus, so passing --corpus is mandatory. 10k
   trajectories at batch 128 is ~78 steps per epoch, so 6000 steps is ~77
   epochs; watch the loss for overfitting and rely on snapshots. lr 3e-4 is
   deliberate (fresh init); the 1e-5 default is for fine-tuning only.
   --auto-resume restarts from the _latest checkpoint if a burst is killed.
4. One-shot eval of the final checkpoint AND the best-loss snapshots.
   CORRECTED 2026-07-20, see the two notes below. Use:
     env PYTHONPATH=. ~/venvs/mime/bin/python research/w1_oneshot_score.py \
       --ckpt event_polar_4m_w1_scratch_v1.pt --n 2000 --seed 42
   Sanity anchor: the same command with
   --ckpt event_polar_4m_fc_v2.pt reads 0.6544, which matches the ~0.65
   one-shot figure in README.md and confirms the harness is behaving.

   Do NOT use evaluate.py for this. Two reasons, both verified.
   (a) evaluate.py loads data/human_eval_features.npy (evaluate.py line
   173), the final untouched eval sample. research/autoloop/scoring.py is
   the metric contract and forbids that file in anything feeding a
   search-space decision; it raises on any path containing "human_eval".
   The W1 gate is exactly such a decision. The metric stated at the top of
   this file is against data/human_val_features_grpo.npy, which is a
   completely different array (all 2000 rows differ). w1_oneshot_score.py
   keeps every other convention identical to evaluate.py (same spec loop,
   same feature extraction, same one-shot settings) and swaps only the
   scorer for scoring.score_features.
   (b) The old command here omitted EVENT_CHOICE_TEMP=10 from the locked
   recipe, which inflates one-shot AUC badly: event_polar_4m_fc_v2 reads
   0.9387 without it against 0.6544 with it. w1_oneshot_score.py defaults
   choice_temp to 10. Note the 0.596 to 0.60 numbers in EXPERIMENTS.md are
   WITH selection (EVENT_SIR=8), not one-shot.
5. Judge the gate: materially below 0.70 = go (winner-only supervised
   training captures the set-selection signal); otherwise W1 is a clean
   negative. Either way: ledger rows for poolgen (one row covering seeds
   61-65 is fine), training, and eval; verify best.pt MD5 again; report the
   verdict to L in plain terms (what it means for the program, not code).
6. After reporting: the agreed next move is drafting the W3 design proposal
   (dispersion-correct conditional density; see memory mime-w3-direction).
   Cloud spend needs L sign-off; local GPU work does not.

## Session hygiene

L authorized a 10-hour work window starting 2026-07-20 morning; most of it
remains. Check that the 20 percent compaction rule is active for the session
and subagents (CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=20); if absent, pause and ask
L to compact instead of running long.
