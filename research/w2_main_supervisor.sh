#!/usr/bin/env bash
# Supervises the W2 N=2000 main run (research/w2_stat_guided_probe.py --main-run)
# across repeated 90-min --max-minutes chunks via --resume, since one chunk
# cannot finish the full N=2000 in the wall-clock cap. Safety: waits for GPU
# cooldown if the previous chunk's watchdog log ever saw >=79C, verifies
# best.pt MD5 every round, attaches a fresh gpu_watchdog.py per round.
set -uo pipefail
cd "/mnt/c/Users/aaron/Code/mouse-trajectory-synthesis"
PY="$HOME/venvs/mime/bin/python"
BASE_ARGS="--main-run --n 2000 --batch-size 64 --m-steps 10 --lr 0.05 --max-minutes 90 --joint-duration --tag units_jd_main"
PIDFILE="research/w2_main.pid.txt"
SUPLOG="research/w2_supervisor.log"
EXPECTED_MD5="91326a29750789f3167055324ef377c5"

CUR_PID="$1"
CUR_WD_LOG="$2"
ROUND="${3:-1}"
MAX_ROUNDS=8

slog() { echo "[supervisor] $(date -u +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$SUPLOG"; }

slog "adopting round $ROUND, pid=$CUR_PID, watchdog_log=$CUR_WD_LOG"

while true; do
  # --- wait for this round's process to exit ---
  while kill -0 "$CUR_PID" 2>/dev/null; do
    sleep 30
  done
  slog "round $ROUND: pid $CUR_PID exited"

  # --- GPU cooldown gate ---
  MAXTEMP=$(awk -F',' 'NR>1 && $2 ~ /^[0-9]+$/ {print $2}' "$CUR_WD_LOG" 2>/dev/null | sort -n | tail -1)
  slog "round $ROUND max temp seen: ${MAXTEMP:-unknown}"
  if [ -n "${MAXTEMP:-}" ] && [ "$MAXTEMP" -ge 79 ] 2>/dev/null; then
    slog "peak >=79C, waiting for cooldown to <=65C before relaunch"
    while true; do
      T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
      slog "cooldown check: ${T:-unknown}C"
      if [ -n "$T" ] && [ "$T" -le 65 ] 2>/dev/null; then break; fi
      sleep 30
    done
  fi

  # --- MD5 check ---
  MD5=$("$PY" -c "import sys; sys.path.insert(0,'.'); from training.train_candi_chain import md5_file; print(md5_file('training/candi_polar_flow_best.pt'))" 2>/dev/null)
  slog "round $ROUND best.pt MD5: $MD5"
  if [ "$MD5" != "$EXPECTED_MD5" ]; then
    slog "*** MD5 MISMATCH (expected $EXPECTED_MD5) -- STOPPING SUPERVISOR, DO NOT RELAUNCH ***"
    exit 1
  fi

  # --- completion check ---
  STATUS_TAG=$("$PY" -c "import json; d=json.load(open('research/w2_probe_results.json')); print(d.get('status'), d.get('tag'))" 2>/dev/null)
  slog "round $ROUND results.json status/tag: $STATUS_TAG"
  if [ "$STATUS_TAG" = "COMPLETE units_jd_main" ]; then
    slog "MAIN RUN COMPLETE. Supervisor exiting cleanly."
    exit 0
  fi

  if [ "$ROUND" -ge "$MAX_ROUNDS" ]; then
    slog "*** MAX_ROUNDS ($MAX_ROUNDS) reached without COMPLETE status -- STOPPING SUPERVISOR for manual review ***"
    exit 2
  fi

  # --- relaunch with --resume ---
  ROUND=$((ROUND + 1))
  WDLOG="research/w2_gpu_temp_round${ROUND}.log"
  slog "relaunching round $ROUND with --resume, watchdog log=$WDLOG"
  nohup "$PY" research/w2_stat_guided_probe.py $BASE_ARGS --resume --pid-file "$PIDFILE" \
    > "research/w2_main_round${ROUND}_stdout.log" 2>&1 &
  sleep 8
  NEWPID=$(cat "$PIDFILE" 2>/dev/null | tr -d ' \n\r')
  slog "round $ROUND relaunched, pid=$NEWPID"
  if [ -z "$NEWPID" ]; then
    slog "*** could not read new PID from $PIDFILE -- STOPPING SUPERVISOR ***"
    exit 3
  fi
  nohup "$PY" research/gpu_watchdog.py --pid "$NEWPID" --log "$WDLOG" --threshold 83 --interval 60 --max-minutes 100 \
    > "research/w2_watchdog_round${ROUND}_stdout.log" 2>&1 &
  sleep 3
  CUR_PID="$NEWPID"
  CUR_WD_LOG="$WDLOG"
done
