---
name: computational-experiment-review
description: Use when reviewing a pre-registration, experiment script, run result, conclusion, or LLM-authored amendment for a claim-producing computational experiment, especially when the question is whether the right experiment is being done, whether the implementation matches the registered construct, or whether conclusions overclaim the measured result.
---

# Computational Experiment Review

Use this skill for this repository's experiment reviews. This is not
generic code review; it is review of an experiment as a claim-producing
machine.

## First Step

Read `EXPERIMENT_REVIEW_PROTOCOL.md` from the repository root. Treat it
as canonical. Then read the task-local files needed for the review:

- `AGENTS.md` for standing research commitments;
- current phase/block plan when relevant, for example `PHASE2.md`;
- `BATTERY.md` or `FORMALISM.md` when the experiment invokes registered
  battery members, failure types, or ledger assumptions;
- `EXPERIMENTS.md` for experiment status and pointers;
- the relevant experiment writeup, script, and run artifact.

## Review Posture

Lead with findings. Evaluate conceptual alignment before style:

- Is this the right experiment for the stated research plan?
- Does the implementation measure the construct named in the writeup?
- Are predictions, verdict predicates, and adjudication rules registered
  before the run?
- Are privileged ground-truth signals evaluation-only?
- Do the conclusions preserve tolerance, horizon, distribution, reference
  patch, coordinate regime, and probe/interpreter class?

Always include explicit checks for LLM-work creep and maintainability
regressions. Common problems are broad labels replacing narrow
constructs, skipped predicates counted as successes, existential claims
from finite probes, and stale claims copied from prior experiments.

## Output Shape

Use this order unless the user asks otherwise:

1. Findings with file/line references, ordered by severity.
2. Open questions or assumptions.
3. LLM-work creep and maintainability notes.
4. Verification performed or not performed.

If no blocking findings exist, say so directly and name any residual
risk.
