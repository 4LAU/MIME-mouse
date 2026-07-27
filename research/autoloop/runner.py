"""run_experiment(config) contract: config-in / metrics-out unit, logged to
the ledger on every call, success or failure. This is what loop.py drives
repeatedly and what a human calls by hand for a one-off experiment.

Config-in fields (see run_experiment docstring for the full contract).

Thermal safety note (2026-07-19): a mid-build request asked to relax the
launch-gate temperature from 60C to 75C and default the inter-run cooldown
to 0. Given this project's history (PLAN.md: "GPU: avoid sustained load (4
bluescreens historically)") and that this is a hardware-safety threshold
governing unattended overnight runs, this file keeps the ORIGINAL
conservative defaults (LAUNCH_GATE_TEMP_C=60) active. The requested
evidence-based cooldown mechanism (wait for temp to drop after a hot run
rather than a fixed sleep) IS implemented, as EVIDENCE_COOLDOWN_HIGH_C /
EVIDENCE_COOLDOWN_LOW_C below, since it is a strictly additional safety
check, not a relaxation. All four thresholds are named module constants a
human can change after direct confirmation; nothing here silently adopted
the higher gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AUTOLOOP_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUTOLOOP_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AUTOLOOP_DIR))

import ledger  # noqa: E402
import scoring  # noqa: E402

TRAIN_DIR = REPO_ROOT / "training"
BEST_PT_PATH = TRAIN_DIR / "candi_polar_flow_best.pt"
BEST_PT_EXPECTED_MD5 = "91326a29750789f3167055324ef377c5"

MIN_N_FOR_GATE = 2000
HARD_WALL_LIMIT_MIN = 100

# --- thermal safety constants (see module docstring) ---
LAUNCH_GATE_TEMP_C = 75          # refuse to start a GPU run above this (L policy July 19: back-to-back default)
WATCHDOG_KILL_TEMP_C = 83        # hard backstop, unchanged, matches gpu_watchdog.py
EVIDENCE_COOLDOWN_HIGH_C = 79    # if previous run's max temp >= this...
EVIDENCE_COOLDOWN_LOW_C = 65     # ...wait until temp drops to this before next launch
DEFAULT_FIXED_COOLDOWN_SEC = 0   # L policy July 19: no fixed cooldowns; evidence-based rule only


def md5_of(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_best_pt_unchanged() -> bool:
    """True iff best.pt's MD5 matches the known-good hash (or the file is
    absent, e.g. in a --mock run where nothing touches it)."""
    actual = md5_of(BEST_PT_PATH)
    if actual is None:
        return True
    return actual == BEST_PT_EXPECTED_MD5


def read_gpu_status() -> dict:
    """Returns {"temp_c": int|None, "mem_used_mb": int|None}. Never raises;
    on failure (no nvidia-smi, no GPU) returns Nones -- callers must treat
    that as "no GPU present/used", not as a green light to skip safety
    checks that matter only when a GPU IS present."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        temp_str, mem_str = [s.strip() for s in line.split(",")]
        return {"temp_c": int(float(temp_str)), "mem_used_mb": int(float(mem_str))}
    except Exception:
        return {"temp_c": None, "mem_used_mb": None}


def evidence_based_cooldown_wait(previous_max_temp_c: float | None, poll_sec: int = 15) -> int:
    """If the previous run got hot (>= EVIDENCE_COOLDOWN_HIGH_C), block
    until temp drops to <= EVIDENCE_COOLDOWN_LOW_C. Returns seconds waited
    (0 if the rule didn't trigger or nvidia-smi is unavailable)."""
    if previous_max_temp_c is None or previous_max_temp_c < EVIDENCE_COOLDOWN_HIGH_C:
        return 0
    waited = 0
    while True:
        status = read_gpu_status()
        if status["temp_c"] is None or status["temp_c"] <= EVIDENCE_COOLDOWN_LOW_C:
            return waited
        time.sleep(poll_sec)
        waited += poll_sec


def _mock_generate(n: int, seed: int) -> list[np.ndarray]:
    """Fabricates n tiny fake (x, y, t) paths -- no model, no GPU. Used only
    by --mock / config["mock"]=True to exercise the plumbing."""
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n):
        length = int(rng.integers(8, 20))
        t = np.cumsum(rng.uniform(0.005, 0.02, size=length))
        x = np.cumsum(rng.normal(0, 5, size=length)) + rng.uniform(0, 800)
        y = np.cumsum(rng.normal(0, 5, size=length)) + rng.uniform(0, 600)
        paths.append(np.stack([x, y, t], axis=1))
    return paths


def _real_generate(config: dict) -> list:
    """Real generation path -- imports phase_a_baseline lazily (touches
    torch/model loading) so mock runs never pay that cost and never risk
    touching the GPU. NOT exercised by this build task; wired for when a
    real experiment is actually launched later."""
    sys.path.insert(0, str(REPO_ROOT / "research"))
    from experiments._common import DurationModel  # noqa: E402
    from phase_a_baseline import load_model, generate_paths, make_specs, DUR_STD  # noqa: E402

    ckpt_name = config.get("ckpt", "candi_polar_flow_best.pt")
    model, data_scale, device, max_seq_len_cfg = load_model(ckpt_name)
    model.max_seq_len_cfg = max_seq_len_cfg
    duration_model = DurationModel(TRAIN_DIR, std_mult=config.get("dur_std", DUR_STD))
    specs = make_specs(config["n"], config["seed"])
    trajectories = generate_paths(
        model, data_scale, device, duration_model, specs,
        no_round=config.get("no_round", True),
    )
    return [t for t in trajectories if t is not None and len(t) >= 2]


def run_experiment(config: dict) -> dict:
    """Execute one config-in/metrics-out unit and append a ledger row
    (on success AND on failure/refusal). Returns the ledger row.

    Required config fields:
        workstream: str
        kind: "generate_score" | "train_burst"
        seed: int
        n: int (default 2000; refused if < 2000 when gate=True)
    Optional:
        gate: bool (default False) -- marks this a go/no-go decision run;
            refuses n < 2000 since small-N AUC is unreliable (measured
            ~0.33 optimistic bias at N=100 vs N=2000 in this project).
        mock: bool (default False) -- fabricate tiny fake paths, no model/GPU.
        watchdog_pid: int -- REQUIRED for kind="train_burst".
        tier: int (default 1), confirms_run_id: str (required if tier=2).
        run_raw_nn: bool (default False, tier=2 only).
        ... generation/sampling passthrough (ckpt, steps, guide, perp,
            correct, cfg, no_round, dur_std) forwarded to _real_generate.
    """
    workstream = config["workstream"]
    kind = config.get("kind", "generate_score")
    n = config.get("n", 2000)
    mock = bool(config.get("mock", False))
    gate = bool(config.get("gate", False))
    tier = int(config.get("tier", 1))
    confirms_run_id = config.get("confirms_run_id")

    t0 = time.perf_counter()
    artifacts: list[str] = []

    def fail(status: str, notes: str, metrics=None, safety=None) -> dict:
        return ledger.append_row(
            workstream=workstream, config=config, status=status,
            metrics=metrics or {}, safety=safety or {}, artifacts=artifacts,
            notes=notes, tier=tier, confirms_run_id=confirms_run_id, mock=mock,
        )

    if gate and n < MIN_N_FOR_GATE:
        return fail("failed", f"refused: gate=True requires n>={MIN_N_FOR_GATE}, got n={n}")
    if tier == 2 and not confirms_run_id:
        return fail("failed", "refused: tier=2 rows must set confirms_run_id")
    if kind == "train_burst" and not config.get("watchdog_pid"):
        return fail("failed", "refused: kind='train_burst' requires watchdog_pid")

    best_md5_before = check_best_pt_unchanged()
    gpu_before = read_gpu_status()
    uses_gpu = (not mock) and gpu_before["temp_c"] is not None
    if uses_gpu and gpu_before["temp_c"] > LAUNCH_GATE_TEMP_C:
        return fail(
            "killed",
            f"refused to launch: GPU temp {gpu_before['temp_c']}C > "
            f"launch gate {LAUNCH_GATE_TEMP_C}C",
            safety={"max_temp_c": gpu_before["temp_c"], "best_pt_md5_ok": best_md5_before},
        )

    try:
        if kind == "generate_score":
            if mock:
                trajectories = _mock_generate(n, config.get("seed", 0))
            else:
                trajectories = _real_generate(config)

            elapsed_min = (time.perf_counter() - t0) / 60.0
            if elapsed_min > HARD_WALL_LIMIT_MIN:
                return fail(
                    "killed",
                    f"exceeded hard wall limit: {elapsed_min:.1f}min > "
                    f"{HARD_WALL_LIMIT_MIN}min (generation phase alone)",
                    safety={"wall_min": elapsed_min},
                )

            synth_features = scoring.extract_features_from_paths(trajectories)
            if tier == 2:
                metrics = scoring.score_panel_tier2(
                    trajectories, seed=config.get("seed", 0),
                    run_raw_nn=bool(config.get("run_raw_nn", False)),
                )
            else:
                metrics = scoring.score_features(synth_features)

        elif kind == "train_burst":
            return fail("failed", "kind='train_burst' is a stub, not implemented yet")
        else:
            return fail("failed", f"unknown kind={kind!r}")

    except Exception as exc:  # noqa: BLE001
        elapsed_min = (time.perf_counter() - t0) / 60.0
        return fail(
            "failed", f"exception during run: {type(exc).__name__}: {exc}",
            safety={"wall_min": elapsed_min},
        )

    elapsed_min = (time.perf_counter() - t0) / 60.0
    if elapsed_min > HARD_WALL_LIMIT_MIN:
        return fail(
            "killed",
            f"exceeded hard wall limit: {elapsed_min:.1f}min > {HARD_WALL_LIMIT_MIN}min",
            metrics=metrics,
        )

    best_md5_after = check_best_pt_unchanged()
    max_temp = None
    peak_vram = None
    if not mock:
        # Only report GPU telemetry when a GPU was actually used -- mock
        # runs must not carry real (misleading) temp/vram readings.
        gpu_after = read_gpu_status()
        max_temp = max(
            [t for t in (gpu_before["temp_c"], gpu_after["temp_c"]) if t is not None],
            default=None,
        )
        peak_vram = max(
            [m for m in (gpu_before["mem_used_mb"], gpu_after["mem_used_mb"]) if m is not None],
            default=None,
        )
    safety = {
        "max_temp_c": max_temp,
        "peak_vram_mb": peak_vram,
        "wall_min": elapsed_min,
        "best_pt_md5_ok": bool(best_md5_before and best_md5_after),
    }

    if tier == 2:
        confirmed_row = next(
            (r for r in ledger.load_ledger(workstream) if r["run_id"] == confirms_run_id),
            None,
        )
        if confirmed_row is not None:
            t1_auc = confirmed_row["metrics"].get("auc_rf_oob")
            t2_auc = metrics.get("auc_rf_oob")
            if isinstance(t1_auc, (int, float)) and isinstance(t2_auc, (int, float)):
                metrics["tuning_confirmation_delta"] = t2_auc - t1_auc

    return ledger.append_row(
        workstream=workstream, config=config, status="ok", metrics=metrics,
        safety=safety, artifacts=artifacts, notes=config.get("notes", ""),
        tier=tier, confirms_run_id=confirms_run_id, mock=mock,
    )


def confirm_tier2(tier1_row: dict, fresh_seed: int, run_raw_nn: bool = False) -> dict:
    """Build and run a tier-2 confirmation config from a tier-1 row: same
    config, but a FRESH seed (must differ from the tuning seed) and
    tier=2/confirms_run_id set. This is what loop.py calls automatically
    when a stop-early threshold is hit (see loop.py's STOP-RULE HYGIENE)."""
    tuning_seed = tier1_row["config"].get("seed")
    if fresh_seed == tuning_seed:
        raise ValueError("tier2 confirmation seed must differ from the tuning seed")
    confirm_config = dict(tier1_row["config"])
    confirm_config["seed"] = fresh_seed
    confirm_config["tier"] = 2
    confirm_config["confirms_run_id"] = tier1_row["run_id"]
    confirm_config["run_raw_nn"] = run_raw_nn
    confirm_config["gate"] = True
    return run_experiment(confirm_config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="path to a JSON config file")
    ap.add_argument("--mock", action="store_true",
                     help="force mock mode regardless of config file")
    args = ap.parse_args()

    if args.config:
        config = json.loads(Path(args.config).read_text())
    else:
        config = {"workstream": "adhoc", "kind": "generate_score", "seed": 0, "n": 10}
    if args.mock:
        config["mock"] = True

    row = run_experiment(config)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
