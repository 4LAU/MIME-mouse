# Research archive

These are the one-off scripts behind the experiments journaled in
EXPERIMENTS.md: the autoresearch waves, the diagnostic probes, and the
recipe sweeps. They are kept for the record, not maintained as a library.

The `w4_*.py` scripts are the single-trajectory workstream, one file per
measured arm, each writing its result to the `.json` file of the same name.
They are journaled in HANDOFF.md rather than EXPERIMENTS.md, and they all
score through `autoloop/scoring.py`. Each one carries its hypothesis, its
control and its prediction in the module docstring, written before the run,
so a file that reads as a null is a recorded null and not an abandoned
script.

They import the core modules at the repo root, so run them from there:

    PYTHONPATH=. python research/sweeps/sweep_round2.py

The shell runners in scripts/ work the same way; run them from the repo
root, for example `bash scripts/run_oos.sh`.
