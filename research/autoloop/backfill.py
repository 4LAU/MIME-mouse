"""One-off data entry: append ledger rows for results already measured
earlier today (before this harness existed), so the leaderboard starts
honest instead of empty. No compute, no GPU, no model loads -- reads only
the existing JSON result files and writes ledger rows.

Run once:
    .venv/Scripts/python.exe research/autoloop/backfill.py
Re-running is safe: it is idempotent by config_hash only in the sense
that loop.py's resumability check would skip these hashes later, but this
script itself does not check-before-append, so do not run it twice without
first checking research/autoloop/ledger.jsonl for duplicates.
"""
from __future__ import annotations

import sys
from pathlib import Path

AUTOLOOP_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUTOLOOP_DIR.parent.parent
sys.path.insert(0, str(AUTOLOOP_DIR))

import ledger  # noqa: E402


def backfill_phase1_finetune():
    config = {
        "workstream": "phase0b_critic_finetune",
        "kind": "generate_score",
        "ckpt": "candi_polar_flow_phase1.pt",
        "steps": 200, "guide": 0.15, "perp": 0.85, "correct": "rotate",
        "cfg": 0.0, "no_round": True, "seed": 42, "n": 2000, "gate": True,
    }
    row = ledger.append_row(
        workstream="phase0b_critic_finetune",
        config=config,
        status="ok",
        metrics={
            "auc_rf_oob": 0.7661579999999999,
            "n_per_class": 2000,
            "critic_auc": 0.8011204999999999,
            "critic_phase0b_oof_reference": 0.632,
        },
        safety={
            "wall_min": 464.139877300011 / 60.0,
            "best_pt_md5_ok": True,
        },
        artifacts=[
            "research/phase1_score_phase1_results.json",
            "research/phase1_score_phase1_features.npy",
            "research/phase1_score_phase1_trajectories.pkl",
        ],
        notes=(
            "BACKFILL, no fresh compute. Phase 1 fine-tune of "
            "candi_polar_flow_phase1.pt (fool-critic training on frozen "
            "phase0b critic) scored via research/phase1_score.py's RF "
            "block. Result WORSENED vs the 0.7573 baseline reference (moved "
            "away from chance, not toward it) despite fooling the frozen "
            "critic (0.632 in-loop -> 0.801 out-of-sample is worse "
            "fooling, not better) -- consistent with the 5th converging "
            "negative for the critic/adversarial-fooling approach. "
            "Provenance: research/phase1_score_phase1_results.json."
        ),
    )
    print(f"[backfill] phase1 fine-tune row: {row['run_id']} "
          f"auc={row['metrics']['auc_rf_oob']}")


def backfill_baseline_reference():
    config = {
        "workstream": "phase0b_critic_finetune",
        "kind": "generate_score",
        "ckpt": "candi_polar_flow_best.pt",
        "steps": 200, "guide": 0.15, "perp": 0.85, "correct": "rotate",
        "cfg": 0.0, "no_round": True, "seed": 42, "n": 2000, "gate": True,
        "note_tag": "baseline_reference",
    }
    row = ledger.append_row(
        workstream="phase0b_critic_finetune",
        config=config,
        status="ok",
        metrics={"auc_rf_oob": 0.7573, "n_per_class": 2000},
        safety={"best_pt_md5_ok": True},
        artifacts=["research/phase1_score_phase1_results.json"],
        notes=(
            "BACKFILL, no fresh compute. Published baseline reference for "
            "candi_polar_flow_best.pt (research/phase_a_gpu_run3_noround.log "
            "recipe: steps=200 guide=0.15 perp=0.85 correct=rotate no_round=True). "
            "Recorded as-is from research/phase1_score.py's "
            "rf_oob_reference_baseline_best_pt_noround field; NOT "
            "independently rescored by this backfill."
        ),
    )
    print(f"[backfill] baseline reference row: {row['run_id']} "
          f"auc={row['metrics']['auc_rf_oob']}")


def backfill_w0_k_filter_floors():
    results = {
        8: 0.580923875,
        16: 0.564605875,
        32: 0.5391421249999999,
    }
    for k, auc in results.items():
        config = {
            "workstream": "W0",
            "kind": "generate_score",
            "model": "event_polar_4m_fc_v2.pt",
            "mechanism": "per_item_sir_judge",
            "K": k, "sir_temp": 0.7, "sir_seed": 0, "n": 2000, "gate": True,
        }
        row = ledger.append_row(
            workstream="W0",
            config=config,
            status="ok",
            metrics={"auc_rf_oob": auc, "n_per_class": 2000},
            safety={},
            artifacts=["research/w0_sir_floor_results.json", "research/w0_sir_floor.py"],
            notes=(
                f"BACKFILL, no fresh compute. Per-item SIR judge floor at K={k} "
                "(GradientBoostingClassifier(n_estimators=200, max_depth=3, "
                "subsample=0.8, random_state=0), tempered Gumbel-max lottery "
                "temp=0.7), reused pool_s42_k32.npz (July 6, no fresh GPU "
                "generation). This is the per-request floor the net-new "
                "model (W1/W2) must beat inside the <=2s serving budget. "
                "Provenance: research/w0_sir_floor_results.json."
            ),
        )
        print(f"[backfill] W0 K={k} row: {row['run_id']} auc={auc}")


if __name__ == "__main__":
    backfill_phase1_finetune()
    backfill_baseline_reference()
    backfill_w0_k_filter_floors()
    print(f"[backfill] done. Ledger: {ledger.LEDGER_PATH}")
    print(f"[backfill] Leaderboard regenerated: {ledger.LEADERBOARD_PATH}")
