# Handoff: Where To Resume

This document is outdated. 

Last updated: 2026-06-19.

This project is in a good place to pause. The oracle-withdrawal reference arc
is concluded, the first structured-completion bridge experiment is concluded,
and the next valuable work is conceptual/design work before another heavy run.

## Fast Re-Entry Order

When you come back, read in this order:

1. `AGENTS.md` — the standing method commitments and scope honesty.
2. `EXPERIMENTS.md`— the recent arc in one screen.
6. `experiments/31-read-transport-atlas.md` — the pre-I2 read-transport
   diagnostic that re-typed the exp-30 negative as position-specific reads.
7. `ORIGINAL_SIN.md` — future-directions memo, now the best place for the
   conceptual path back toward the original idea.
8. `INTERVENTION_CLASS_BENCHMARK.md` — Phase 3 design map for intervention
   classes; not a pre-registration.
9. `AGENT_WORKFLOW.md` — optional worker/reviewer loop automation.
10. `EXPERIMENT_REVIEW_PROTOCOL.md` before any new preregistration or
   conclusion review.

Then run:

```bash
git status --short
```

At the time this handoff was written, recent work included updates to
`ORIGINAL_SIN.md`, exp 29's writeup/script/output, and `EXPERIMENTS.md`.
Treat the detailed experiment writeups as canonical; index rows are only
pointers.

## Current Scientific State

TBD

## Label Hygiene

This is now important enough to be a project norm.

Experiment-local labels are allowed, but keep them local. A label such as
`ECHO(...)`, `ALIGNED(...)`, or a predicate identifier like
`phi1_next_closes` is meaningful only inside the experiment that defines its
thresholds, domain, patch class, and aggregation rule.

Project-level labels need stricter treatment. They should be promoted only
when:

- the definition is stable across experiments;
- the scope indices are explicit;
- the failure branch is reviewable;
- the label is added to `FORMALISM.md`, `BATTERY.md`, or another canonical
  project document deliberately.

When in doubt, write the conclusion in ordinary language first, and keep the
script's decision string as a local audit trail.

## What Not To Do Next

Do not immediately run another `pstack` mining experiment just because exp 29
found angles near 90 degrees. Those angles were descriptive because the
registered intervention failed.

Do not expand all axes at once. Predicate language, toy process, interpreter
class, patch class, trajectory semantics, and sampled uncertainty can each
become a full design space. Pick one load-bearing question per experiment.

Do not let cheaper preflight vehicles become headline evidence about
transformer residual streams. Cheaper vehicles are useful for known-answer
predicate-layer tests, process-design screening, and verdict validation; they
do not transport architecture-specific claims until rerun in the transformer
vehicle.

Do not promote exp-local verdict names into `EXPERIMENTS.md` or conceptual
docs unless you are intentionally defining a project-level concept.

## Best Next Work

TBD