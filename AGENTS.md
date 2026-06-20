# Agent rules (always loaded)

Behavioral rules for any agent working in this repo. Keep this file small — it is
loaded on every turn. The *why* (frame, roadmap, scope) lives in
`RESEARCH_PROGRAM.md`; what to read for a given task lives in `INDEX.md`. Read
those on demand, not by default.

## Orientation

Do not bulk-read the large docs. Start from `INDEX.md` and open only the
specific files your task names. For an experiment, that is its
`experiments/NN-*.md` writeup plus the protocol/formalism sections it points to —
not the whole experiment log or benchmark.

## Working norms

- Keep everything runnable on modest hardware with exact or cheap ground truth as
  long as possible. Quarantine heavy dependencies. Make analysis stages consume
  caches, not models. Torch code should use accelerators (mps) when available.
- When a result surprises you, suspect the harness before the science — then, if
  the harness holds, write the surprise into the verdict logic so the code can
  diagnose it next time without you. Prefer adding a typed verdict over adding a
  number.
- When you fix a flaw, document the flaw and its lesson where the fix lives; the
  failures are the curriculum.
- Terminology discipline: standard terms are fine; repo-local terms must preserve
  their scope indices; experiment-local labels stay local unless deliberately
  promoted; avoid broad labels when the measured construct is narrow.

## Library home, not frozen imports

Shared infrastructure — `build_candidates`, the observable ρ helper
(`rho_obs`/`_mnorm`/`rho_band`), registered constants, and any verdict-partition
helper — belongs in the living library (`battery.py` / `expcommon.py`).
Concluded/frozen scripts import *from* the library, never the reverse. A frozen
record must not be load-bearing infrastructure.

`battery.py` exposes the promoted observable ρ helper, candidate builder,
directional tolerance partition helper, and the verdict-machinery trio
(`majority_vote` seed aggregation, `first_precedence` routing, `status_rollup`
for the FORMALISM 6.1 recalibrate ≠ fail discipline) for new live scripts. The
trio is computation sharing only — branch labels stay caller-owned and
experiment-local; no shared JSON record format or global verdict ontology.
Promote forward from the library; do not retro-edit frozen scripts unless a
review finds a reproducibility bug.

## Experiment reviews

Use `EXPERIMENT_REVIEW_PROTOCOL.md`. These reviews are not generic code review:
they evaluate whether the experiment is the right one for the research plan,
whether the code implements the registered construct, and whether verdicts and
conclusions stay within what was measured. Review pauses are part of the method.

Pre-registration is a two-part artifact — the writeup (goals, assumptions, scope,
predictions, adjudication rules, reviewable failure modes) and the runnable
script that already implements those rules, guards, self-checks, and output
tables. Both must exist before it counts as pre-registered. Pause after
committing a pre-registration, before the first run, for review of design and
code. Pause again after running, before writing conclusions, for review of
results, verdict logic, LLM-work creep, and maintainability.
