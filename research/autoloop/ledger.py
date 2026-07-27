"""Append-only JSONL ledger for the net-new model program's autoresearch
loop (PLAN.md "NET-NEW MODEL PROGRAM"). Every experiment run by runner.py
appends one row here, win or fail, so the orchestrator can review a
leaderboard instead of babysitting individual runs.

Row schema (see append_row):
    run_id, timestamp, workstream, tier (1=loop metric, 2=confirmation),
    confirms_run_id (tier2 only), config, config_hash, status
    (ok/failed/killed), metrics (dict; "auc_rf_oob" is the canonical tier1
    key), safety (max_temp_c, peak_vram_mb, wall_min, best_pt_md5_ok),
    artifacts (list of paths), notes, mock (bool).

Anti-Goodhart discipline (added per L, after 5 prior Goodharted training
signals): tier1 is what loops iterate against and is inherently
selection-biased ("winner's curse" -- the best of N noisy runs looks
better than its true quality). Only a tier2 confirmation row (fresh seed,
full detector panel, see scoring.py) may be called "confirmed". The
leaderboard never prints a tier1 best without printing its tier2 status
next to it, and always prints n_trials for the workstream so the reader
sees how many chances the tier1 best had to look good by chance.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUTOLOOP_DIR = Path(__file__).resolve().parent
LEDGER_PATH = AUTOLOOP_DIR / "ledger.jsonl"
LEADERBOARD_PATH = AUTOLOOP_DIR / "LEADERBOARD.md"

CANONICAL_METRIC = "auc_rf_oob"
CHANCE_AUC = 0.5
WINNERS_CURSE_NOTE = (
    "Best-of-N tier1 scores are selection-biased; only tier2 confirmations "
    "are quotable."
)


def stable_hash(config: dict) -> str:
    """Deterministic short hash of a config dict, order-independent."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(workstream: str, config_hash: str) -> str:
    return f"{workstream}_{_now_iso().replace(':', '').replace('+00:00', 'Z')}_{config_hash[:8]}"


def append_row(
    workstream: str,
    config: dict,
    status: str,
    metrics: dict | None = None,
    safety: dict | None = None,
    artifacts: list[str] | None = None,
    notes: str = "",
    tier: int = 1,
    confirms_run_id: str | None = None,
    mock: bool = False,
    ledger_path: Path | None = None,
) -> dict:
    """Append one row to the ledger and regenerate LEADERBOARD.md.

    status must be one of "ok", "failed", "killed". Returns the row as
    written (including generated run_id/timestamp/config_hash).
    """
    if status not in ("ok", "failed", "killed"):
        raise ValueError(f"bad status {status!r}, must be ok/failed/killed")
    if tier not in (1, 2):
        raise ValueError(f"tier must be 1 or 2, got {tier!r}")
    if tier == 2 and not confirms_run_id:
        raise ValueError("tier=2 rows must set confirms_run_id")

    config_hash = stable_hash(config)
    row = {
        "run_id": make_run_id(workstream, config_hash),
        "timestamp": _now_iso(),
        "workstream": workstream,
        "tier": tier,
        "confirms_run_id": confirms_run_id,
        "config": config,
        "config_hash": config_hash,
        "status": status,
        "metrics": metrics or {},
        "safety": safety or {},
        "artifacts": artifacts or [],
        "notes": notes,
        "mock": bool(mock),
    }

    path = ledger_path or LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")

    regenerate_leaderboard(ledger_path=path)
    return row


def remove_last_row(ledger_path: Path | None = None) -> dict | None:
    """Pop the most recently appended row. Used only to strike mock/dry-run
    rows after a validation test so no fake data persists in the ledger."""
    path = ledger_path or LEDGER_PATH
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    last = json.loads(lines[-1])
    remaining = lines[:-1]
    path.write_text(
        "\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8"
    )
    regenerate_leaderboard(ledger_path=path)
    return last


def load_ledger(workstream: str | None = None, ledger_path: Path | None = None) -> list[dict]:
    path = ledger_path or LEDGER_PATH
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if workstream is None or row.get("workstream") == workstream:
            rows.append(row)
    return rows


def known_config_hashes(workstream: str | None = None, ledger_path: Path | None = None) -> set[str]:
    """All config_hash values already present in the ledger -- loop.py uses
    this for resumability (skip configs already run, any status)."""
    return {row["config_hash"] for row in load_ledger(workstream, ledger_path)}


def leaderboard(
    workstream: str,
    metric: str = CANONICAL_METRIC,
    ascending: bool = True,
    ledger_path: Path | None = None,
) -> list[dict]:
    """Sorted table of tier1 status=="ok" non-mock rows for a workstream,
    by the given metric. Generic sort utility -- for the actual
    LEADERBOARD.md this project sorts by distance-to-chance instead, since
    lower AUC is not automatically "better" past 0.5 (see auc_gap)."""
    rows = [
        r for r in load_ledger(workstream, ledger_path)
        if r.get("status") == "ok" and not r.get("mock") and r.get("tier") == 1
        and r.get("metrics", {}).get(metric) is not None
    ]
    rows.sort(key=lambda r: r["metrics"][metric], reverse=not ascending)
    return rows


def auc_gap(auc: float) -> float:
    """Distance from chance (0.50) -- the actual program objective. Used to
    rank the leaderboard; a 0.40 AUC is not "better" than 0.55 even though
    it is numerically lower."""
    return abs(auc - CHANCE_AUC)


def _tier2_status(workstream_rows: list[dict], tier1_run_id: str) -> str:
    for r in workstream_rows:
        if r.get("tier") == 2 and r.get("confirms_run_id") == tier1_run_id:
            return "CONFIRMED" if r.get("status") == "ok" else "CONFIRM-FAILED"
    return "UNCONFIRMED"


def _fmt_auc(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    return f"{v:.4f}"


def format_leaderboard_md(ledger_path: Path | None = None) -> str:
    """Human-readable leaderboard across all workstreams. Regenerated on
    every append_row call. Never shows a tier1 best without its tier2
    confirmation status next to it."""
    all_rows = load_ledger(ledger_path=ledger_path)
    workstreams = sorted({r["workstream"] for r in all_rows})

    lines = [
        "# Autoloop Leaderboard",
        "",
        f"_Regenerated {_now_iso()}_",
        "",
        f"**{WINNERS_CURSE_NOTE}**",
        "",
        "Program metric: RF-OOB AUC vs data/human_val_features_grpo.npy "
        "(n_estimators=100, oob_score, random_state=42). Target is chance "
        "(0.50), not minimum -- ranked by distance from 0.50, not by raw "
        "value. `human_eval_features.npy` is never used inside this loop.",
        "",
    ]

    if not workstreams:
        lines.append("_(ledger empty)_")
        return "\n".join(lines) + "\n"

    for ws in workstreams:
        ws_rows = [r for r in all_rows if r["workstream"] == ws]
        tier1_ok = [
            r for r in ws_rows
            if r.get("tier") == 1 and r.get("status") == "ok" and not r.get("mock")
            and r.get("metrics", {}).get(CANONICAL_METRIC) is not None
        ]
        n_trials = len(tier1_ok)
        lines.append(f"## {ws}")
        lines.append("")
        lines.append(f"n_trials (tier1, status=ok): {n_trials}")
        lines.append("")

        if not tier1_ok:
            lines.append("_(no completed tier1 runs)_")
            lines.append("")
            continue

        ranked = sorted(tier1_ok, key=lambda r: auc_gap(r["metrics"][CANONICAL_METRIC]))
        lines.append(
            "| rank | run_id | tier1 AUC | gap-to-0.5 | tier2 status | "
            "tier2 AUC | collapse | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(ranked[:15], 1):
            m = r["metrics"]
            t1_auc = m.get(CANONICAL_METRIC)
            t2_status = _tier2_status(ws_rows, r["run_id"])
            t2_row = next(
                (x for x in ws_rows if x.get("tier") == 2
                 and x.get("confirms_run_id") == r["run_id"]),
                None,
            )
            t2_auc = t2_row["metrics"].get(CANONICAL_METRIC) if t2_row else None
            collapse = "SUSPECT" if m.get("collapse_flag") else "ok"
            note = (r.get("notes") or "")[:60]
            lines.append(
                f"| {i} | {r['run_id']} | {_fmt_auc(t1_auc)} | "
                f"{_fmt_auc(auc_gap(t1_auc))} | {t2_status} | "
                f"{_fmt_auc(t2_auc)} | {collapse} | {note} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def regenerate_leaderboard(ledger_path: Path | None = None) -> None:
    LEADERBOARD_PATH.write_text(format_leaderboard_md(ledger_path), encoding="utf-8")
