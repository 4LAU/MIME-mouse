#!/bin/bash
# W1 pool supervisor: wait for the seed-61 evaluate.py (arg 1) to release the
# GPU, then generate token-bearing pools for the remaining seeds sequentially,
# each with a gpu_watchdog attached. Exits when all pools exist.
set -u
cd /mnt/c/Users/aaron/Code/mouse-trajectory-synthesis
PY="$HOME/venvs/mime/bin/python"
WAIT_PID="${1:-0}"
SEEDS="${2:-62 63 64 65}"

if [ "$WAIT_PID" -gt 0 ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

for SEED in $SEEDS; do
  POOL="pool_w1_s${SEED}_k16.npz"
  if [ -f "$POOL" ]; then
    echo "[supervisor] $POOL already exists, skipping" >> research/w1_pool_supervisor.log
    continue
  fi
  echo "[supervisor] $(date -Is) launching seed $SEED" >> research/w1_pool_supervisor.log
  env EVENT_CKPT=event_polar_4m_fc_v2.pt EVENT_ORDER=gumbel EVENT_SNAP=2.5 \
      EVENT_DUR_STD=1.0 EVENT_CHOICE_TEMP=10 DUR_EMPIRICAL=1 EVENT_SIR=16 \
      EVENT_SIR_TEMP=0.7 EVENT_SIR_DUR_DIVERSE=1 EVENT_POOL_TOKENS=1 \
      EVENT_POOL_SAVE="$POOL" PYTHONPATH=. \
      "$PY" evaluate.py --experiment experiments.event_stream_polar --seed "$SEED" \
      > "eval_poolgen_w1_s${SEED}.log" 2>&1 &
  GEN_PID=$!
  "$PY" research/gpu_watchdog.py --pid "$GEN_PID" \
      --log "research/w1_poolgen_s${SEED}_temp.log" \
      --threshold 83 --interval 60 --max-minutes 90 &
  WD_PID=$!
  wait "$GEN_PID"
  RC=$?
  kill "$WD_PID" 2>/dev/null
  if [ ! -f "$POOL" ]; then
    echo "[supervisor] $(date -Is) seed $SEED FAILED rc=$RC, stopping" >> research/w1_pool_supervisor.log
    exit 1
  fi
  echo "[supervisor] $(date -Is) seed $SEED done rc=$RC" >> research/w1_pool_supervisor.log
done
echo "[supervisor] $(date -Is) ALL POOLS DONE" >> research/w1_pool_supervisor.log
