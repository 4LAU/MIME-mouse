# Handoff: Windows -> Ubuntu migration (July 20, 2026)

Paste the PROMPT block at the bottom into the first Claude Code session on Ubuntu.
Everything above it is the checklist for getting the files and environment over.

## 1. Environment facts (verified from the Windows side, July 20)

Ubuntu is WSL2 on this same machine. NOTHING needs to move:

- Repo is directly accessible at `/mnt/c/Users/aaron/Code/mouse-trajectory-synthesis`
  (verified, PLAN.md and all gitignored data/checkpoints included).
- GPU passthrough works: `nvidia-smi` inside WSL sees the RTX 4070 Laptop GPU.
- Claude Code is installed (`~/.local/bin/claude`), and the `clauded` alias
  (`claude --dangerously-skip-permissions`) is already in `~/.bashrc`.
- WSL python3 is 3.14.4. Build a FRESH Linux venv (the Windows `.venv` is unusable):
  put it on the Linux filesystem for speed (e.g. `~/venvs/mime`), then
  `pip install torch --index-url https://download.pytorch.org/whl/cu128` (Windows side
  ran torch 2.11.0+cu128) plus numpy and scikit-learn per `pyproject.toml`.
  Verify `python -c "import torch; print(torch.cuda.is_available())"` prints True.
- Verify model integrity: `md5sum training/candi_polar_flow_best.pt` ==
  `91326a29750789f3167055324ef377c5`.
- Two scripts have Windows assumptions; port before use:
  - `research/w2_main_supervisor.sh` (hardcoded `C:/` path, `.venv/Scripts/python.exe`,
    `powershell` process checks) - rewrite paths/python and use `kill`/`ps`.
  - `research/gpu_watchdog.py` - check its kill mechanism uses something Linux-safe
    (if it calls `taskkill`, swap for `os.kill`/`pkill`).

## 3. State at pause (all processes stopped cleanly, nothing corrupted)

- W2 stat-guided probe MAIN RUN (tag=main, N=2000, batch=64, m_steps=10, lr=0.05,
  capped-loss, with control arm) was checkpointed at batch 19/32 (~60%).
  Resume state: `research/w2_progress_checkpoint.pkl` + `research/w2_specs_checkpoint.pkl`.
- best.pt MD5 verified unchanged after shutdown. Max GPU temp this round: 73C.
- Full narrative state: `research/w2_agent_state.md` (read it top to bottom).
- Experiment ledger: `research/autoloop/` (ledger.jsonl has all W2 rows backfilled).

---

## PROMPT (paste into the new Ubuntu Claude Code session)

You are resuming a paused ML research program (mouse trajectory synthesis, repo
MIME-mouse). Work in /mnt/c/Users/aaron/Code/mouse-trajectory-synthesis (the Windows
repo, fully accessible from this WSL2 Ubuntu; GPU passthrough verified). Read
HANDOFF_UBUNTU.md section 1 for environment setup (fresh Linux venv needed; two
scripts need Windows-to-Linux porting). The Windows session paused everything cleanly.
Auto-memory from the Windows machine did not transfer, so these standing facts are
restated here; treat them as binding.

CONTEXT AND GOAL
- Single metric: RF-OOB AUC at N=2000 per class vs data/human_val_features_grpo.npy
  (RandomForestClassifier n_estimators=100, oob_score=True, random_state=42). Target 0.50.
  N=100-scale AUC reads ~0.33 low; never trust small-N absolute values.
- Headline safety net (do not disturb): 0.504 tuning / 0.513 out-of-sample via set-level
  selection on event_polar_4m_fc_v2.pt; always disclosed with GBM 0.523 and raw-NN 0.508.
- Standing mandate from L (July 19): weeks-long net-new generative model program.
  Serving constraint: one trajectory per (A,B) request in <=2 seconds; pre-generated
  pools are REJECTED. Five converging negatives closed the fine-tune-the-generator
  family; per-token architectures are exhausted. Read PLAN.md sections "NET-NEW MODEL
  PROGRAM" and "W2 PROBE IS NEXT" for the full program and pre-registered gates.
- L directives in force: Karpathy-style autoresearch loops (harness in research/autoloop/,
  every experiment logged to its ledger; two-tier metrics, tier2 = fresh seed + RF/GBM/
  raw-NN panel before anything is quotable); anti-overfit discipline (collapse-flag
  sanity battery every run); thermal policy (back-to-back GPU runs default, launch gate
  75C, watchdog hard-kill 83C, evidence-based cooldown only if peak >=79C).
- HARD RULES: never modify training/candi_polar_flow_best.pt (MD5
  91326a29750789f3167055324ef377c5, verify after every run); never touch
  data/human_eval_features.npy; never modify scoring code; git add files individually,
  never `git add .`; repo is public - docs must read human-written, no em/en dashes.
- L is non-technical: report what/why in plain terms, brief summary first, no praise.
  Orchestrate via subagents where sensible; keep subagent contexts lean (no streaming
  training logs into context; state files on disk; ledger as the system of record).

IMMEDIATE TASK
1. Read research/w2_agent_state.md fully, then PLAN.md "W2 PROBE IS NEXT".
2. Verify environment: CUDA available, best.pt MD5 matches, gpu_watchdog.py kill path
   works on Linux (port if needed), port research/w2_main_supervisor.sh to Linux paths.
3. Resume the paused W2 main run from its checkpoint (batch 19/32):
   python research/w2_stat_guided_probe.py --main-run --n 2000 --batch-size 64 \
     --m-steps 10 --lr 0.05 --max-minutes 90 --tag main --resume \
     --pid-file research/w2_main.pid.txt
   with a gpu_watchdog attached (--threshold 83 --interval 60 --max-minutes 100), or
   via the ported supervisor which handles multi-round resume automatically.
4. When research/w2_probe_results.json shows status=COMPLETE tag=main: verify MD5,
   log the run into research/autoloop/ledger, and judge the PRE-REGISTERED GATE:
   PASS = guided AUC <= 0.70 with control ~0.757 and no single-feature collapse
   (dispersion ratio <0.2 or >5.0 = suspect). Otherwise FAIL with this structural
   caveat, already established: only shape/dispersion features (curvature, angular
   velocity, path efficiency) were steerable; the velocity/acceleration family is
   structurally unreachable by noise-space optimization because each spec's duration
   is fixed before steering begins and the stat targets were sampled from marginals
   that mix over durations. Design implication for W2 proper regardless of gate
   outcome: stage 1 must sample duration JOINTLY with the stat targets (log_duration
   is already in the generator conditioning vector, so this is feasible).
5. Report the verdict to L in plain terms, then propose the next bounded autoresearch
   loop consistent with the PLAN.md program order.
