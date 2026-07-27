"""THIS FILE IS THE METRIC CONTRACT. Loops (loop.py, runner.py) must never
edit the scoring logic in here -- if the recipe changes, every prior
ledger row becomes non-comparable. Only add new read-only panel functions;
never alter score_features' RF recipe.

Tier 1 (loop metric): score_features() -- RF-OOB AUC vs
data/human_val_features_grpo.npy, the exact block factored out of
research/phase1_score.py's rf_oob_auc(). This is what runner.py logs on
every generate_score run and what loop.py iterates against.

Tier 2 (confirmation): score_panel_tier2() -- adds a GBM 5-fold CV AUC
(mirrors external_validation/validate_adserp.py's gbm_cv_auc) and,
optionally, the raw-trajectory CNN AUC (detector_raw.raw_nn_auc, 3-fold).
A config may only be called "confirmed" via this panel, on a FRESH seed,
never the tuning seed. See runner.py's confirm_tier2().

data/human_eval_features.npy is FORBIDDEN in this file and everywhere in
this harness -- it is the final, untouched eval sample and must never be
seen by anything that feeds a loop or a search-space decision. Any path
containing "human_eval" raises.

Identity/acceptance test (run this file directly, CPU-only):
    .venv/Scripts/python.exe research/autoloop/scoring.py
must print auc_rf_oob == 0.7661579999999999 (to 1e-9) when rescoring
research/phase1_score_phase1_features.npy, matching
research/phase1_score_phase1_results.json's recorded number exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from features import FEATURE_NAMES, extract_feature_matrix  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
DEFAULT_HUMAN_FEATURES_PATH = DATA_DIR / "human_val_features_grpo.npy"
FORBIDDEN_SUBSTRING = "human_eval"

RF_SEED = 42
RF_N_ESTIMATORS = 100
GBM_N_ESTIMATORS = 100
GBM_MAX_DEPTH = 4
GBM_N_SPLITS = 5

# Reference number this module's identity test must reproduce exactly.
REFERENCE_AUC_PHASE1 = 0.7661579999999999
REFERENCE_FEATURES_FILE = REPO_ROOT / "research" / "phase1_score_phase1_features.npy"

# Sanity-battery thresholds (per L, anti-Goodhart hardening): a per-feature
# path-to-path dispersion ratio outside this band means the model has
# collapsed (or exploded) a feature relative to humans -- a future
# detector's easy tell, regardless of how good the AUC looks.
COLLAPSE_RATIO_LOW = 0.2
COLLAPSE_RATIO_HIGH = 5.0


def _guard_human_path(path) -> Path:
    p = Path(path)
    if FORBIDDEN_SUBSTRING in p.name.lower() or FORBIDDEN_SUBSTRING in str(p).lower():
        raise ValueError(
            f"refusing to load {p}: paths containing "
            f"'{FORBIDDEN_SUBSTRING}' are forbidden anywhere in the "
            "autoloop harness (final untouched eval sample; never feeds "
            "a loop or search-space decision)."
        )
    return p


def extract_features_from_paths(trajectories, hz: float = 125.0) -> np.ndarray:
    """(x, y, t) trajectories -> 18-col feature matrix, via features.py's
    extract_feature_matrix exactly (same function research/phase1_score.py
    and every other scoring script in this repo uses)."""
    return extract_feature_matrix(trajectories, hz=hz)


def dispersion_ratios(synth_features: np.ndarray, human_features: np.ndarray) -> dict:
    """Per-feature path-to-path std(synth)/std(human) ratio, plus a
    collapse flag if any feature falls outside [COLLAPSE_RATIO_LOW,
    COLLAPSE_RATIO_HIGH]. Cheap, CPU-only, run on every generate_score row.

    Lesson this encodes: burst 3 had a great-looking loss while two
    features collapsed to near-constants -- an AUC near 0.5 with a
    collapsed feature is not a win, it is a tell waiting to be found.
    """
    ratios = {}
    collapsed = []
    for i, name in enumerate(FEATURE_NAMES):
        h_std = float(np.std(human_features[:, i]))
        s_std = float(np.std(synth_features[:, i]))
        if h_std < 1e-12:
            ratio = float("inf") if s_std > 1e-12 else 1.0
        else:
            ratio = s_std / h_std
        ratios[name] = ratio
        if ratio < COLLAPSE_RATIO_LOW or ratio > COLLAPSE_RATIO_HIGH:
            collapsed.append(name)
    return {
        "dispersion_ratios": ratios,
        "collapse_flag": len(collapsed) > 0,
        "collapse_features": collapsed,
    }


def score_features(
    synth_features_18col: np.ndarray,
    human_features_path=DEFAULT_HUMAN_FEATURES_PATH,
) -> dict:
    """THE metric contract. Identical recipe to research/phase1_score.py's
    rf_oob_auc(): balance to min(n_human, n_synth), vstack human(0)/synth(1),
    RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1,
    random_state=42), OOB decision function AUC.

    Returns {"auc_rf_oob", "n_per_class", "importances"} plus the sanity
    battery ("dispersion_ratios", "collapse_flag", "collapse_features").
    """
    human_features_path = _guard_human_path(human_features_path)
    human = np.load(human_features_path)
    synth = np.asarray(synth_features_18col, dtype=np.float64)

    n_use = min(len(human), len(synth))
    human_bal = human[:n_use]
    synth_bal = synth[:n_use]

    X = np.vstack([human_bal, synth_bal])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, oob_score=True, n_jobs=-1, random_state=RF_SEED
    )
    clf.fit(X, y)
    oob_proba = clf.oob_decision_function_[:, 1]
    auc = float(roc_auc_score(y, oob_proba))
    importances = dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist()))

    result = {"auc_rf_oob": auc, "n_per_class": n_use, "importances": importances}
    result.update(dispersion_ratios(synth_bal, human_bal))
    return result


def gbm_cv_auc(
    synth_features_18col: np.ndarray,
    human_features_path=DEFAULT_HUMAN_FEATURES_PATH,
    seed: int = RF_SEED,
) -> dict:
    """Tier-2 panel member: GBM 5-fold CV AUC. Mirrors
    external_validation/validate_adserp.py's gbm_cv_auc exactly
    (GradientBoostingClassifier(n_estimators=100, max_depth=4,
    random_state=seed), StratifiedKFold(5, shuffle=True), cross_val_predict)."""
    human_features_path = _guard_human_path(human_features_path)
    human = np.load(human_features_path)
    synth = np.asarray(synth_features_18col, dtype=np.float64)
    n_use = min(len(human), len(synth))
    X = np.vstack([human[:n_use], synth[:n_use]])
    y = np.concatenate([np.zeros(n_use), np.ones(n_use)])

    cv = StratifiedKFold(n_splits=GBM_N_SPLITS, shuffle=True, random_state=seed)
    gbm = GradientBoostingClassifier(
        n_estimators=GBM_N_ESTIMATORS, max_depth=GBM_MAX_DEPTH, random_state=seed
    )
    proba = cross_val_predict(gbm, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    return {"auc_gbm_cv": auc, "n_per_class": n_use}


def raw_nn_auc_panel(trajectories, seed: int = RF_SEED) -> dict:
    """Tier-2 panel member: raw-trajectory CNN 3-fold CV AUC
    (detector_raw.raw_nn_auc, imported not reimplemented). This trains a
    small CNN and will use CUDA if available -- callers running this for
    real (not this harness-build task) must not do so while another GPU
    job is active. Marked "pending" by callers that choose to skip it."""
    from detector_raw import raw_nn_auc  # imported lazily; torch + CUDA-capable

    auc = raw_nn_auc(trajectories, train_dir=str(REPO_ROOT / "training"), seed=seed)
    return {"auc_raw_nn": auc}


def score_panel_tier2(
    trajectories,
    human_features_path=DEFAULT_HUMAN_FEATURES_PATH,
    seed: int = RF_SEED,
    run_raw_nn: bool = False,
) -> dict:
    """Full confirmation panel for a tier-2 row: RF-OOB (tier1 recipe,
    reused) + GBM 5-fold CV + raw-NN (if run_raw_nn, else "pending" -- never
    silently skipped)."""
    human_features_path = _guard_human_path(human_features_path)
    synth_features = extract_features_from_paths(trajectories)

    out = score_features(synth_features, human_features_path)
    out.update(gbm_cv_auc(synth_features, human_features_path, seed=seed))
    if run_raw_nn:
        out.update(raw_nn_auc_panel(trajectories, seed=seed))
    else:
        out["auc_raw_nn"] = "pending"
    return out


def _run_identity_test() -> None:
    if not REFERENCE_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"identity test fixture missing: {REFERENCE_FEATURES_FILE}"
        )
    synth = np.load(REFERENCE_FEATURES_FILE)
    result = score_features(synth)
    auc = result["auc_rf_oob"]
    delta = abs(auc - REFERENCE_AUC_PHASE1)
    print(f"[scoring] reproduced auc_rf_oob={auc!r} "
          f"(reference={REFERENCE_AUC_PHASE1!r}, delta={delta:.2e})")
    assert delta < 1e-9, f"IDENTITY TEST FAILED: delta={delta}"
    print("[scoring] IDENTITY TEST PASSED (delta < 1e-9)")
    print(f"[scoring] collapse_flag={result['collapse_flag']} "
          f"collapse_features={result['collapse_features']}")


if __name__ == "__main__":
    _run_identity_test()
