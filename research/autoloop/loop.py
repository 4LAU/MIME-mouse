"""Loop driver skeleton: runs a BOUNDED search space of configs
sequentially through runner.py, resumable via the ledger, stops on a
pre-registered rule, writes a summary. Never generates configs on the fly
-- the search space is an explicit, finite, pre-registered list or grid
loaded from a JSON file, so an overnight run cannot wander into open-ended
territory.

STOP-RULE HYGIENE (per L): stop_early_if_metric_leq is a tier1 threshold.
Hitting it does NOT declare success -- it triggers exactly one automatic
tier2 confirmation run (fresh seed, full panel, via runner.confirm_tier2)
and then the loop stops, so the summary always reports a confirmation
outcome instead of a raw tier1 number.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

AUTOLOOP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUTOLOOP_DIR))

import ledger  # noqa: E402
import runner  # noqa: E402

MAX_GRID_EXPANSION = 500  # refuse to expand a grid larger than this -- bounded, not open-ended


def _expand_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    combos: list[dict] = [{}]
    for k in keys:
        values = grid[k]
        combos = [dict(c, **{k: v}) for c in combos for v in values]
        if len(combos) > MAX_GRID_EXPANSION:
            raise ValueError(
                f"grid expansion exceeds MAX_GRID_EXPANSION={MAX_GRID_EXPANSION} "
                f"(got {len(combos)}+) -- shrink the search space, the loop "
                "must stay pre-registered and bounded"
            )
    return combos


def load_search_space(path: Path) -> list[dict]:
    """Loads either {"configs": [...]} (explicit list) or {"grid": {...}}
    (cartesian product, capped at MAX_GRID_EXPANSION). Never both."""
    spec = json.loads(Path(path).read_text())
    if "configs" in spec and "grid" in spec:
        raise ValueError("search space file must have 'configs' OR 'grid', not both")
    if "configs" in spec:
        configs = spec["configs"]
    elif "grid" in spec:
        configs = _expand_grid(spec["grid"])
    else:
        raise ValueError("search space file must have a 'configs' or 'grid' key")
    if not configs:
        raise ValueError("search space is empty")
    return configs


def run_loop(
    workstream: str,
    search_space_path: Path,
    stop_rules: dict,
    cooldown_sec: int = runner.DEFAULT_FIXED_COOLDOWN_SEC,
    ledger_path=None,
) -> dict:
    """stop_rules keys (all optional):
        max_runs: int
        max_gpu_minutes: float
        stop_early_if_metric_leq: {"metric": "auc_rf_oob", "threshold": 0.55}
    Resumable: any config whose stable hash already appears in the ledger
    (any status) is skipped, so re-running the same search space file is
    safe and continues where it left off.
    """
    configs = load_search_space(search_space_path)
    for cfg in configs:
        cfg.setdefault("workstream", workstream)

    already_done = ledger.known_config_hashes(workstream, ledger_path)
    max_runs = stop_rules.get("max_runs")
    max_gpu_minutes = stop_rules.get("max_gpu_minutes")
    stop_early = stop_rules.get("stop_early_if_metric_leq")

    n_run = 0
    n_skipped = 0
    gpu_minutes_used = 0.0
    stopped_reason = None
    confirmation_row = None
    previous_max_temp = None
    peak_temp_session = None
    cooldowns_triggered = 0
    rows_this_loop = []

    for cfg in configs:
        cfg_hash = ledger.stable_hash(cfg)
        if cfg_hash in already_done:
            n_skipped += 1
            continue
        if max_runs is not None and n_run >= max_runs:
            stopped_reason = f"max_runs={max_runs} reached"
            break
        if max_gpu_minutes is not None and gpu_minutes_used >= max_gpu_minutes:
            stopped_reason = f"max_gpu_minutes={max_gpu_minutes} reached"
            break

        if not cfg.get("mock"):
            waited = runner.evidence_based_cooldown_wait(previous_max_temp)
            if waited > 0:
                cooldowns_triggered += 1
            elif cooldown_sec > 0 and n_run > 0:
                time.sleep(cooldown_sec)

        row = runner.run_experiment(cfg)
        rows_this_loop.append(row["run_id"])
        already_done.add(cfg_hash)
        n_run += 1

        wall_min = row.get("safety", {}).get("wall_min") or 0.0
        gpu_minutes_used += wall_min
        temp = row.get("safety", {}).get("max_temp_c")
        if temp is not None:
            previous_max_temp = temp
            peak_temp_session = temp if peak_temp_session is None else max(peak_temp_session, temp)

        if stop_early and row["status"] == "ok":
            metric_val = row.get("metrics", {}).get(stop_early["metric"])
            if metric_val is not None and metric_val <= stop_early["threshold"]:
                stopped_reason = (
                    f"stop_early: {stop_early['metric']}={metric_val} <= "
                    f"{stop_early['threshold']} -- running tier2 confirmation, "
                    "NOT declaring success on tier1 alone"
                )
                confirmation_row = runner.confirm_tier2(
                    row, fresh_seed=cfg.get("seed", 0) + 100_000,
                )
                break

    summary = {
        "workstream": workstream,
        "search_space_file": str(search_space_path),
        "n_configs_total": len(configs),
        "n_run": n_run,
        "n_skipped_already_in_ledger": n_skipped,
        "gpu_minutes_used": gpu_minutes_used,
        "peak_temp_c_session": peak_temp_session,
        "cooldowns_triggered": cooldowns_triggered,
        "stopped_reason": stopped_reason,
        "run_ids": rows_this_loop,
        "tier2_confirmation_run_id": confirmation_row["run_id"] if confirmation_row else None,
    }
    out_path = AUTOLOOP_DIR / f"{workstream}_loop_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    summary["summary_file"] = str(out_path)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workstream")
    ap.add_argument("search_space", type=str)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--max-gpu-minutes", type=float, default=None)
    ap.add_argument("--stop-early-metric", type=str, default=None)
    ap.add_argument("--stop-early-threshold", type=float, default=None)
    ap.add_argument("--cooldown-sec", type=int, default=runner.DEFAULT_FIXED_COOLDOWN_SEC)
    args = ap.parse_args()

    stop_rules = {"max_runs": args.max_runs, "max_gpu_minutes": args.max_gpu_minutes}
    if args.stop_early_metric and args.stop_early_threshold is not None:
        stop_rules["stop_early_if_metric_leq"] = {
            "metric": args.stop_early_metric, "threshold": args.stop_early_threshold,
        }

    summary = run_loop(args.workstream, Path(args.search_space), stop_rules,
                        cooldown_sec=args.cooldown_sec)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
