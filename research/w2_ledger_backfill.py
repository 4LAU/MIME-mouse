"""One-off ledger entries for the W2 stat-guided probe's tuning trail
(research/w2_stat_guided_probe.py). No compute -- reads only the numbers
already measured and recorded in research/w2_agent_state.md /
research/w2_probe_run.log. Run once:
    .venv/Scripts/python.exe research/w2_ledger_backfill.py
"""
from __future__ import annotations

import sys
from pathlib import Path

AUTOLOOP_DIR = Path(__file__).resolve().parent / "autoloop"
sys.path.insert(0, str(AUTOLOOP_DIR))

import ledger  # noqa: E402

WORKSTREAM = "W2_stat_guided_probe"
BASELINE_REF = 0.7573


def row_lrtest_prod():
    config = {
        "workstream": WORKSTREAM, "tag": "lrtest_prod", "kind": "tuning_smoke",
        "n": 64, "batch_size": 64, "m_steps": 15, "lr": 0.05, "loss_norm": "capped",
        "k": 200, "n_steps": 200, "guide": 0.15, "perp_scale": 0.85,
        "skip_control": True, "skip_validation": True,
    }
    return ledger.append_row(
        workstream=WORKSTREAM, config=config, status="ok",
        metrics={
            "auc_rf_oob": 0.8134, "control_auc_reference": BASELINE_REF,
            "target_hit_frac_1.0std_overall": 0.31,
            "mean_velocity_frac_hit_1.0std": 0.0,
            "mean_velocity_mean_abs_err_std": 9.5,
            "loss_init": 422.0, "loss_final": 365.0,
        },
        safety={"step_time_s_per_optimizer_step_batch64": 27.55, "best_pt_md5_ok": True},
        artifacts=["research/w2_probe_run.log"],
        notes=(
            "Tuning iteration (skip-control/skip-validation), lr=0.05 m_steps=15 "
            "batch=64. Guided AUC worse than baseline (0.8134 vs 0.7573 reference) "
            "and targets not hit (mean_velocity 0% hit, 9.5std mean error). Gate "
            "not applicable (control/validation skipped by design for fast "
            "iteration). Superseded by lrtest_b256."
        ),
    )


def row_lrtest_b256():
    config = {
        "workstream": WORKSTREAM, "tag": "lrtest_b256", "kind": "tuning_smoke",
        "n": 256, "batch_size": 256, "m_steps": 3, "lr": 0.05, "loss_norm": "capped",
        "k": 200, "n_steps": 200, "guide": 0.15, "perp_scale": 0.85,
        "skip_control": True, "skip_validation": True,
    }
    return ledger.append_row(
        workstream=WORKSTREAM, config=config, status="ok",
        metrics={
            "auc_rf_oob": 0.725311279296875, "control_auc_reference": BASELINE_REF,
            "target_hit_frac_1.0std_overall": 0.2823153409090909,
            "mean_velocity_frac_hit_1.0std": 0.0,
            "mean_velocity_mean_abs_err_std": 9.078455681945709,
            "loss_init": 386.38482666015625, "loss_final": 357.6329650878906,
        },
        safety={
            "step_time_s_per_optimizer_step_batch256": 136.50684316665865,
            "best_pt_md5_ok": True, "gpu_after_c": 65, "gpu_after_vram_mib": 7015,
        },
        artifacts=["research/w2_probe_run.log", "research/w2_agent_state.md"],
        notes=(
            "Tuning iteration testing batch=256 throughput. AUC improved vs "
            "lrtest_prod (0.7253 vs 0.8134) but still above the 0.70 gate; "
            "target-hit rate WORSENED (0.28 vs 0.31). Per-sample step time got "
            "WORSE at batch=256 (136.5s/step for 4x specs vs 27.55s/step at "
            "batch=64) -- GPU was near the WDDM VRAM cap (7015 MiB); batch=64 "
            "confirmed as the compute-efficient choice for the main run."
        ),
    )


def row_tune1_stdnorm_killed():
    config = {
        "workstream": WORKSTREAM, "tag": "tune1_stdnorm", "kind": "tuning_smoke",
        "n": 64, "batch_size": 64, "m_steps": 20, "lr": 0.1, "loss_norm": "stdnorm",
        "k": 200, "n_steps": 200, "guide": 0.15, "perp_scale": 0.85,
        "skip_control": True, "skip_validation": True,
    }
    return ledger.append_row(
        workstream=WORKSTREAM, config=config, status="killed",
        metrics={},
        safety={"best_pt_md5_ok": True},
        artifacts=[],
        notes=(
            "Killed before any batch completed: launch collided with a "
            "pre-existing untracked process (smoke_final) sharing the same "
            "checkpoint file paths, causing GPU contention. No result produced. "
            "Intended to test whether uncapped 1/target_std^2 loss weighting "
            "(vs the target model's median-capped weight) could fix "
            "mean_velocity's near-total miss rate. Superseded by a structural "
            "finding (not re-attempted): mean_velocity/std_velocity are "
            "approx path_length/duration, and duration is fixed per spec at "
            "build time (never touched by the optimized noise z), while the "
            "target model's bucket marginals mix over ALL durations in the "
            "training pool -- the targets may be structurally unreachable by "
            "z-optimization regardless of loss weighting, not a scaling bug. "
            "W2-proper fix: sample duration jointly with the stat targets "
            "(log_duration is already in the generator's conditioning vector)."
        ),
    )


def row_smoke_final():
    config = {
        "workstream": WORKSTREAM, "tag": "smoke_final", "kind": "smoke_full",
        "n": 64, "batch_size": 64, "m_steps": 10, "lr": 0.05, "loss_norm": "capped",
        "k": 200, "n_steps": 200, "guide": 0.15, "perp_scale": 0.85,
        "skip_control": False, "skip_validation": False,
    }
    return ledger.append_row(
        workstream=WORKSTREAM, config=config, status="ok",
        metrics={
            "auc_rf_oob": 0.7637, "control_auc_rf_oob": 0.4745,
            "control_auc_reference": BASELINE_REF,
            "target_hit_frac_1.0std_overall": 0.3040,
        },
        safety={"best_pt_md5_ok": True},
        artifacts=["research/w2_probe_run.log", "research/w2_smokefinal_watchdog.log"],
        notes=(
            "First full smoke run with control+validation enabled (default "
            "capped-loss config, lr=0.05 m_steps=10 batch=64). control_auc=0.4745 "
            "at N=64 is NOT a real control-arm failure -- known project-wide bias, "
            "N=64/100 AUC reads roughly 0.33 below the N=2000 reference (0.7573), "
            "consistent with 0.4745 here. guided_auc=0.7637 still above the 0.70 "
            "gate; target-hit 0.30 overall, below the ad hoc 0.6 tuning bar. "
            "Per-feature: curvature_mean/std dispersion ratio reached ~1.0-1.06 "
            "(near-human, up from historical 0.55-0.75), std_velocity/"
            "std_acceleration/std_jerk near 0.9-1.15, but mean_acceleration/"
            "mean_jerk/path_efficiency/angular_velocity remain under-dispersed "
            "and mean_velocity/std_velocity target-hit ~0%. Guided RF top "
            "importances shifted to residual curvature/angular precision and "
            "UNCOVERED features (max_deviation, velocity_skewness, "
            "num_direction_changes) with no gradient -- the multiplicity-of-tells "
            "pattern. Decision: proceed to N=2000 main run despite the ad hoc "
            "bar not being met (bar was this session's own invention, not the "
            "pre-registered gate; N=64 numbers are directionally informative "
            "only, per project memory on small-N unreliability)."
        ),
    )


if __name__ == "__main__":
    for fn in (row_lrtest_prod, row_lrtest_b256, row_tune1_stdnorm_killed, row_smoke_final):
        row = fn()
        print(f"[w2_ledger_backfill] {row['run_id']} status={row['status']} "
              f"auc={row['metrics'].get('auc_rf_oob')}")
    print(f"[w2_ledger_backfill] done. Ledger: {ledger.LEDGER_PATH}")
