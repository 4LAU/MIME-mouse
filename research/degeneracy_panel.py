"""Read-only control panel: how much of a score is motion, how much arithmetic?

Real recordings are whole pixels at irregular times. features.resample_trajectory
interpolates them onto the 125 Hz grid, which leaves long runs of points that are
bitwise exactly collinear, so consecutive step directions are identical and their
angle difference is exactly 0.0. np.sign(0.0) is 0, so those runs contribute
nothing to num_direction_changes. Generated paths carry no such exact structure:
every angle difference is a tiny nonzero number of essentially random sign, and
the count roughly doubles. research/p3_ceiling_probe.py measured this: nudging
real paths by one billionth of a pixel, far below anything that changes how the
motion looks, moves the contract AUC from 0.6456 to 0.8328 on its own.

So a single AUC blends two things that are not the same: how human the motion is,
and whether the coordinates happen to carry that exact numerical structure. Any
operation that moves points and re-rounds them (arrival correction, rotation,
scaling) erases the structure, and shows up as a realism penalty it may not be.

This panel does not change the metric. scoring.score_features stays exactly as
it is, so every prior ledger row stays comparable. It adds a second reading of
the same arms in which BOTH sides have been nudged by JITTER_PX, which destroys
the exact structure symmetrically and leaves only motion. Read the two together:

  delta survives the control   the difference was about movement, trust it
  delta collapses              the difference was arithmetic, the conclusion
                               it supported needs re-deriving

Nudging the human side needs human PATHS, not the stored 18-column reference
matrix, so the panel rebuilds a reference from the raw segmented recordings in
data/demo_pool.npz. It reads 0.4922 against data/human_val_features_grpo.npy,
chance, so it is a fair stand-in and the rebuilt and control columns can be read
on the same 0.50 scale as the contract column.

Getting that right mattered. An earlier version built the reference from the
{split}_positions.npy grid instead, which prepare_training_data.py made by
resampling and distance-normalizing. Those paths are fractional and their
duration is quantized to 175 distinct values against the contract reference's
903, so the rebuild read 0.6382 against the contract reference, and every arm
scored against it inherited that bias. Any reference built for this panel must
clear the acceptance test in _self_test: whole pixels, and near 0.50.

Usage:
  env PYTHONPATH=. ~/venvs/mime/bin/python research/degeneracy_panel.py \
    --self-test
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research"))
sys.path.insert(0, str(R / "research" / "autoloop"))

import scoring  # noqa: E402  (metric contract, imported never edited)
from features import FEATURE_NAMES, extract_features, resample_trajectory  # noqa: E402

HZ = 125.0
N_FEAT = len(FEATURE_NAMES)
# Raw segmented Balabit movements: whole pixels at their own irregular
# timestamps, cut by setup_data._segment_movements, the same segmentation every
# other part of this program treats as "a human movement". The {split}_positions
# arrays cannot be used here: prepare_training_data.py builds them by calling
# resample_trajectory and then distance-normalizing, so they are fractional and
# their stored duration is quantized, which the scorer reads as a difference
# between humans and humans (0.6382 with a rebuild made that way).
POOL_NPZ = R / "data" / "demo_pool.npz"

# One billionth of a pixel. p3_ceiling_probe showed the effect saturates
# immediately: 1e-9, 1e-6 and 1e-3 px all read the same AUC, so the amount is
# not a tuning knob, only the presence or absence of exact structure matters.
# 1e-9 is chosen as the smallest of those, far below any physical meaning.
JITTER_PX = 1e-9

# Where the panel's rebuilt reference paths are placed on screen. Arbitrary and
# shared by every arm, so it cancels out of arm-to-arm comparisons.
REF_START = (500.0, 500.0)


def features_with_jitter(trajs, jitter_px=0.0, seed=0, hz=HZ):
    """features.extract_feature_matrix, with a nudge inserted after the resample.

    Returns one row per input trajectory, NaN where the extractor rejected it,
    so callers can hold the same subset fixed across arms. With jitter_px=0 and
    any seed, each row equals extract_feature_matrix's row for that trajectory.

    The nudge goes after resample_trajectory rather than before because the
    interpolation between two nearby whole-pixel samples can itself be exact,
    which would leave some collinear runs intact.
    """
    rng = np.random.default_rng(seed)
    rows = np.full((len(trajs), N_FEAT), np.nan)
    for i, tr in enumerate(trajs):
        p = np.asarray(resample_trajectory(tr, hz=hz), dtype=np.float64)
        if p.ndim != 2 or len(p) < 5:
            continue
        if jitter_px:
            p = p.copy()
            p[:, :2] += rng.normal(0.0, jitter_px, size=(len(p), 2))
        f = extract_features(p)
        if f is not None and np.all(np.isfinite(f)):
            rows[i] = f
    return rows


def real_paths(n_paths=2000, seed=42, half="ref"):
    """Real val-split recordings in pixel space, the same source the contract
    reference was extracted from, in a form that can be nudged.

    Whole pixels at their own irregular timestamps, untouched by any resample,
    which is what the extractor expects to be handed and what creates the exact
    collinearity this panel exists to control for.

    Drawn in one shot and split in two disjoint halves: "ref" builds the panel's
    human reference, "holdout" is a second sample of real paths that every table
    scores as an arm. The holdout arm is what a perfect model would emit, so it
    is each column's floor, measured under that column's own treatment rather
    than assumed to be 0.50.
    """
    # allow_pickle is NOT set: demo_pool.npz holds plain numeric arrays
    d = np.load(POOL_NPZ)
    flat, off, t = d["flat"], d["offsets"], d["t"]
    pick = np.random.default_rng(seed).choice(
        len(off) - 1, size=min(2 * n_paths, len(off) - 1), replace=False)
    pick = pick[:n_paths] if half == "ref" else pick[n_paths:2 * n_paths]

    out = []
    for i in pick:
        a, b = int(off[i]), int(off[i + 1])
        xy = np.asarray(flat[a:b], np.float64)
        # translate to a common start by a whole-pixel offset, so the paths stay
        # on the lattice; every feature is translation invariant anyway
        xy = xy - xy[0] + np.array(REF_START)
        out.append(np.c_[xy, np.asarray(t[a:b], np.float64)])
    return out


def build_reference(jitter_px=0.0, n_paths=2000, seed=42, cache={}):
    """18-column reference matrix from real paths, nudged by jitter_px.

    Also reports how far the rebuild sits from the contract reference: the
    largest per-feature mean shift in units of the contract's own spread, and
    the AUC the rebuild reads against it.
    """
    key = (jitter_px, n_paths, seed)
    if key in cache:
        return cache[key]
    X = features_with_jitter(real_paths(n_paths, seed, "ref"), jitter_px, seed)
    X = X[np.all(np.isfinite(X), axis=1)]
    contract = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    z = np.abs(X.mean(0) - contract.mean(0)) / np.maximum(contract.std(0), 1e-12)
    info = {"X": X,
            "n": len(X),
            "max_abs_z": float(z.max()),
            "worst_feature": FEATURE_NAMES[int(z.argmax())],
            "vs_contract_auc": float(scoring.score_features(X)["auc_rf_oob"])}
    cache[key] = info
    return info


def _score_against(X_synth, X_ref, tmp_dir=None):
    """scoring.score_features against an arbitrary reference matrix.

    Written to disk and passed by path so the RF recipe used here is literally
    score_features' own, never a copy of it that could drift.
    """
    d = Path(tmp_dir or "/tmp/claude-1000/-home-aaronadmin/degeneracy_panel")
    d.mkdir(parents=True, exist_ok=True)
    f = d / "panel_reference.npy"
    np.save(f, X_ref)
    return scoring.score_features(X_synth, human_features_path=f)


def panel(arms, n_paths=2000, seed=42, jitter_px=JITTER_PX):
    """arms: {name: [trajectory, ...]}. Every arm must be the same length and
    in the same order, so the panel can hold one valid subset fixed across all
    of them and across both jitter settings.

    Returns {name: {"contract", "rebuilt", "control"}} where

      contract  scoring.score_features exactly as it stands today, the number
                that belongs in the ledger
      rebuilt   same arm, scored against the rebuilt human reference with no
                nudge, which isolates the cost of rebuilding the reference
      control   same arm nudged, against the reference nudged the same way

    A "real (holdout)" arm is always added: a second, disjoint sample of real
    paths, which is what a perfect model would emit. It is each column's floor.
    Compare arms within a column. Do not compare across columns.
    """
    names = list(arms)
    lengths = {len(arms[k]) for k in names}
    if len(lengths) != 1:
        raise ValueError(f"arms differ in length: "
                         f"{ {k: len(arms[k]) for k in names} }")

    plain = {k: features_with_jitter(arms[k], 0.0, seed) for k in names}
    nudged = {k: features_with_jitter(arms[k], jitter_px, seed) for k in names}

    ok = np.ones(len(arms[names[0]]), bool)
    for k in names:
        ok &= np.all(np.isfinite(plain[k]), axis=1)
        ok &= np.all(np.isfinite(nudged[k]), axis=1)
    rows = np.flatnonzero(ok)

    # the floor arm is scored on its own valid subset; it is a reference point,
    # not a paired comparison, so it does not have to share the arms' rows
    hold = real_paths(n_paths, seed, "holdout")
    hold_plain = features_with_jitter(hold, 0.0, seed)
    hold_nudged = features_with_jitter(hold, jitter_px, seed)
    hok = (np.all(np.isfinite(hold_plain), axis=1)
           & np.all(np.isfinite(hold_nudged), axis=1))
    hold_plain, hold_nudged = hold_plain[hok], hold_nudged[hok]

    ref_plain = build_reference(0.0, n_paths, seed)
    ref_nudged = build_reference(jitter_px, n_paths, seed)

    out = {"_reference": {
        "n": ref_plain["n"],
        "max_abs_z_vs_contract": ref_plain["max_abs_z"],
        "worst_feature": ref_plain["worst_feature"],
        "rebuild_vs_contract_auc": ref_plain["vs_contract_auc"],
        "jitter_px": jitter_px,
        "n_holdout": int(hok.sum()),
        "n_paths_shared": int(len(rows))}}
    for k in names:
        out[k] = {
            "contract": float(scoring.score_features(
                plain[k][rows])["auc_rf_oob"]),
            "rebuilt": float(_score_against(
                plain[k][rows], ref_plain["X"])["auc_rf_oob"]),
            "control": float(_score_against(
                nudged[k][rows], ref_nudged["X"])["auc_rf_oob"]),
        }
    out["real (holdout)"] = {
        "contract": float(scoring.score_features(hold_plain)["auc_rf_oob"]),
        "rebuilt": float(_score_against(hold_plain,
                                        ref_plain["X"])["auc_rf_oob"]),
        "control": float(_score_against(hold_nudged,
                                        ref_nudged["X"])["auc_rf_oob"]),
    }
    return out


def print_panel(res, title=""):
    if title:
        print(f"\n{title}")
    r = res["_reference"]
    print(f"[panel] rebuilt reference: {r['n']} real paths, worst feature mean "
          f"shift {r['max_abs_z_vs_contract']:.2f} sd ({r['worst_feature']}), "
          f"reads {r['rebuild_vs_contract_auc']:.4f} against the contract one")
    print(f"[panel] {r['n_paths_shared']} paths shared across arms, "
          f"{r['n_holdout']} held-out real paths, "
          f"jitter {r['jitter_px']:.0e} px")
    print(f"\n{'arm':<14}{'contract':>10}{'rebuilt':>10}{'control':>10}")
    for k, v in res.items():
        if k == "_reference":
            continue
        print(f"{k:<14}{v['contract']:>10.4f}{v['rebuilt']:>10.4f}"
              f"{v['control']:>10.4f}")


def _self_test():
    """features_with_jitter at jitter 0 must reproduce extract_feature_matrix,
    and a nudged real reference must move num_direction_changes the way
    p3_ceiling_probe found. Neither touches the GPU."""
    from features import extract_feature_matrix

    trajs = real_paths(200, 42, "ref")
    a = features_with_jitter(trajs, 0.0, 0)
    b = extract_feature_matrix(trajs)
    a = a[np.all(np.isfinite(a), axis=1)]
    assert a.shape == b.shape, f"row counts differ: {a.shape} vs {b.shape}"
    delta = float(np.abs(a - b).max())
    print(f"[selftest] jitter 0 matches extract_feature_matrix to {delta:.3e} "
          f"over {a.shape[0]} paths")
    assert delta == 0.0, "jitter 0 must be bit-identical to the extractor"

    # real recordings are whole pixels; if the rebuild is not, the scale or the
    # source is wrong and the whole reference is suspect
    off = max(float(np.abs(np.asarray(p)[:, :2] - np.round(p)[:, :2]).max())
              for p in trajs)
    print(f"[selftest] rebuilt paths are whole pixels to {off:.3e}")
    assert off < 1e-6, "rebuilt reference paths are not on the pixel lattice"

    dur = FEATURE_NAMES.index("movement_duration")
    ndc = FEATURE_NAMES.index("num_direction_changes")
    n0 = build_reference(0.0, 2000, 42)
    n1 = build_reference(JITTER_PX, 2000, 42)
    contract = np.load(scoring.DEFAULT_HUMAN_FEATURES_PATH)
    print(f"[selftest] num_direction_changes: contract "
          f"{contract[:, ndc].mean():.2f}, rebuilt {n0['X'][:, ndc].mean():.2f}, "
          f"rebuilt+nudge {n1['X'][:, ndc].mean():.2f}")
    print(f"[selftest] rebuilt reference vs contract reference: "
          f"{n0['vs_contract_auc']:.4f} plain, {n1['vs_contract_auc']:.4f} "
          f"nudged")
    print(f"[selftest] worst feature mean shift {n0['max_abs_z']:.2f} sd "
          f"({n0['worst_feature']})")
    print(f"[selftest] movement_duration distinct values: contract "
          f"{len(np.unique(np.round(contract[:, dur], 6)))}, rebuilt "
          f"{len(np.unique(np.round(n0['X'][:, dur], 6)))}")
    assert n1["X"][:, ndc].mean() > n0["X"][:, ndc].mean(), \
        "nudging real paths must raise the direction-change count"
    # acceptance test: real is real. A reference the contract scorer can pick
    # apart from its own reference biases every arm scored against it.
    assert n0["vs_contract_auc"] < 0.55, (
        f"rebuilt reference reads {n0['vs_contract_auc']:.4f} against the "
        "contract reference; it is not a fair stand-in")

    # the two halves must be disjoint, or the floor arm is scoring itself
    def sigs(ps):
        return {(len(p), float(p[-1, 0]), float(p[-1, 1])) for p in ps}
    ref, hold = real_paths(2000, 42, "ref"), real_paths(2000, 42, "holdout")
    shared = len(sigs(ref) & sigs(hold))
    print(f"[selftest] ref/holdout halves: {len(ref)} and {len(hold)} paths, "
          f"{shared} shared signatures of {len(sigs(ref))} distinct")
    assert shared * 20 < len(sigs(ref)), "halves are not disjoint"
    print("[selftest] ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(__doc__)
