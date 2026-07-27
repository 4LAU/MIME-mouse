# Archive

Planning documents that have been overtaken by results. Nothing here is a
current instruction. They are kept because each one records what was believed
at the time it was written, and several of the program's negatives are only
legible against the plan that predicted otherwise.

Read HANDOFF.md in the repo root for current state.

- **PLAN.md** (5 July 2026). Kept out of the public repo by .gitignore, a
  decision made when the repo was published and left alone here. It exists on
  the working machine only. The five-day hardening push after the repo went
  public. Its "W3 = SCALE" direction is superseded: the deficit was measured to
  be conditional under-dispersion, not capacity, so scaling would have spent
  money against the wrong diagnosis.
- **HANDOFF_W1.md** (20 July 2026). Session handoff from the W1 workstream.
  W1 closed as a clean negative: fresh-init supervised training on set-level
  selection winners reads 0.8331 one-shot against a 0.70 gate, worse than the
  0.6544 baseline it started from. Winner imitation does not internalise the
  selection signal.
- **HANDOFF_UBUNTU.md** (20 July 2026). Checklist for moving the work from
  Windows to WSL2 Ubuntu on the same machine. The move is done.
- **W3_PROPOSAL.md** (21 July 2026). Proposed teaching the model to arrive at
  its target (P1) and to vary its movement character (P2). Both closed as
  failures, P1 after six fine-tunes and P2 after three. The P3 addendum at the
  bottom is the only part with a live successor, and its content is carried in
  HANDOFF.md.
- **DIFFUSION_PILOT.md** and **DIFFUSION_PILOT_V2.md** (July 2026). A
  whole-trajectory diffusion model trained with a whole-path curvature loss.
  Version 1 trained cleanly and changed nothing about the generated paths,
  because the loss measured curvature on a one-step shortcut the model already
  satisfied. Version 2 was scoped to fix that and was overtaken by the learned
  critic route, which failed for a related reason at Phase 1
  (see ADVERSARIAL_CRITIC.md).
