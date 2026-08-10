"""Deterministic GPU temperature watchdog for supervised bursts.

Polls `nvidia-smi` every --interval seconds, appends timestamp+temp to
--log, and kills --pid (SIGKILL on POSIX, taskkill /F /T on Windows) the
instant temperature reaches --threshold. Also enforces a hard wall-clock cap (--max-minutes)
as a second independent stop condition, in case temperature stays low but
the burst runs long. Runs standalone (not polled by hand) so the safety
rule is enforced even if the parent agent is busy watching training output.

Usage:
    python research/gpu_watchdog.py --pid 12345 --log research/gpu_temp_burst1.log \
        --threshold 83 --interval 60 --max-minutes 90
"""
from __future__ import annotations

import argparse
import datetime
import os
import signal
import subprocess
import sys
import time

if os.name == "nt":
    import ctypes


def read_temp() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        return int(out.strip().splitlines()[0])
    except Exception as exc:  # noqa: BLE001
        print(f"[watchdog] WARNING: nvidia-smi read failed: {exc}", flush=True)
        return None


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_pid(pid: int) -> None:
    if os.name == "nt":
        print(f"[watchdog] KILLING pid {pid} (taskkill /F /T)", flush=True)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, text=True)
        return
    print(f"[watchdog] KILLING pid {pid} (SIGKILL)", flush=True)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        print(f"[watchdog] pid {pid} already gone", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True,
                    help="PID of the GPU process to kill on threshold/timeout breach")
    ap.add_argument("--log", required=True, help="Path to append timestamp,temp_c lines")
    ap.add_argument("--threshold", type=float, default=83.0)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="Hard wall-clock cap; kills --pid when exceeded regardless of temp")
    args = ap.parse_args()

    t_start = time.time()
    print(f"[watchdog] watching pid={args.pid} threshold={args.threshold}C "
          f"interval={args.interval}s max_minutes={args.max_minutes} log={args.log}",
          flush=True)

    with open(args.log, "a") as f:
        f.write(f"# watchdog start {datetime.datetime.now().isoformat()} "
                f"pid={args.pid} threshold={args.threshold}\n")
        f.flush()

        while True:
            now = datetime.datetime.now().isoformat()
            temp = read_temp()
            elapsed_min = (time.time() - t_start) / 60.0

            if temp is not None:
                f.write(f"{now},{temp}\n")
            else:
                f.write(f"{now},READ_FAILED\n")
            f.flush()

            if not pid_alive(args.pid):
                f.write(f"{now},TARGET_PROCESS_EXITED\n")
                f.flush()
                print("[watchdog] target process no longer running, exiting", flush=True)
                break

            if temp is not None and temp >= args.threshold:
                f.write(f"{now},THRESHOLD_BREACH_KILLING\n")
                f.flush()
                print(f"[watchdog] TEMP {temp}C >= {args.threshold}C, hard stop", flush=True)
                kill_pid(args.pid)
                break

            if args.max_minutes is not None and elapsed_min >= args.max_minutes:
                f.write(f"{now},MAX_MINUTES_REACHED_KILLING\n")
                f.flush()
                print(f"[watchdog] max minutes ({args.max_minutes}) reached, hard stop", flush=True)
                kill_pid(args.pid)
                break

            time.sleep(args.interval)

    print("[watchdog] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
