# Handoff: Where To Resume

Last updated: 2026-06-15.

This project is in a good place to pause. The oracle-withdrawal reference arc
is concluded, the first structured-completion bridge experiment is concluded,
and the next valuable work is conceptual/design work before another heavy run.

## Fast Re-Entry Order

When you come back, read in this order:

1. `AGENTS.md` — the standing method commitments and scope honesty.
2. `EXPERIMENTS.md` rows 23-29 — the recent arc in one screen.
3. `experiments/28-consolidation.md` — the oracle-withdrawal conclusion.
4. `experiments/29-predicate-targeting.md` — the first predicate pilot and its
   result.
5. `ORIGINAL_SIN.md` — future-directions memo, now the best place for the
   conceptual path back toward the original idea.
6. `EXPERIMENT_REVIEW_PROTOCOL.md` before any new preregistration or
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

### Oracle Withdrawal

The oracle-withdrawal reference arc closes with a mixed result, not a clean
victory:

- Observable diagnostics did not uniquely earn a compact reference on
  `pstack`.
- The declared-by-convention `cegar` anchor nevertheless supports a usable
  six-member battery transfer.
- `pstack` is near variance-mimicry: useful as a workflow stress test, but
  probably not rich enough to keep mining for conceptual payoff.

The honest claim is:

```text
The hidden-oracle workflow transferred under a declared anchor,
while oracle-free unique reference selection returned a typed negative.
```

Do not round that up to "oracle withdrawal works." It worked for scoring and
transfer under a declared anchor; unique reference selection remains open.

### Original Sin / Structured Completion

The original idea is now best phrased as a relation between two abstraction
languages:

- residual abstractions over model states;
- completion abstractions over futures.

The current battery mostly scores full finite-horizon completion
distributions. That was necessary for exact audit, but it stripped away most
of the original semantic content: named predicates, composition, temporal
structure, and typed counterexamples on the completion side.

The bridge back is:

```text
residual abstraction alpha
tested against
completion abstraction phi
```

where `phi` is a measured property of futures. The key distinction is not
"concrete vs abstract"; it is what `phi` is a function of:

- within-horizon m-gram predicates: cheap, exact, no new semantic substrate;
- trajectory-level predicates: richer, but require new evaluator machinery.

Start with within-horizon predicates only long enough to validate the layer.
Do not mistake that cheap bridge for the final target.

### Experiment 29

Exp 29 was a measurement pilot, not the predicate phase.

Its local decision string was:

```text
ECHO(phi1_next_closes,phi2_net_return)
```

Keep that string local to exp 29. The project-level lesson is ordinary:

```text
Some predicate probabilities were linearly readable and exact-calibrated,
but the registered rank-1 writes did not causally control them.
```

One registered predicate was flat on the distribution; one was not recovered
by tested interpreters. No geometry-routing result fired. Do not use exp 29
to claim that predicates in general align with PCA/core, diverge from
PCA/core, fail on `pstack`, or make predicate CEGAR unnecessary.

The transferable lesson is a design constraint:

```text
Predicate-targeting experiments must separately check:
1. predicate variation,
2. interpreter recovery,
3. observable/exact endpoint calibration,
4. causal control under the registered patch class,
before making geometry or routing claims.
```

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

### First: Clean Re-Entry And Commit Boundary

Before designing the next experiment, make the repository state boring:

```bash
git status --short
uv run python battery.py
uv run python predicates.py
uv run python scripts/predicates/predicate_targeting.py --selftest
```

The post-oracle-withdrawal library-home debt has been paid forward for new
live code: `battery.py` exposes `rho_obs`, `rho_band`,
`directional_tolerance_partition`, `CandidateConfig`, and `build_candidates`.
The old oracle-withdrawal scripts remain frozen historical records; do not
retro-edit them unless a review finds a reproducibility bug.

If the exp 29 files and this handoff are uncommitted, commit them as a
checkpoint. This is not just housekeeping; the next phase will be easier if
the predicate pilot is frozen before new design work starts.

### Then: Decide The Next Scientific Move

There are two good choices.

Choice A: design a predicate-backward toy.

This is the most valuable path toward `ORIGINAL_SIN.md`. The toy should be
chosen because its latent factors imply known relationships among predicates.
Candidate families:

- colored Dyck;
- `Z1R x Dyck` or `Mess3 x Dyck`;
- weakly coupled mode x stack;
- switching grammar.

The goal is not "make a richer toy." The goal is to create a known-answer
environment where predicate-specific abstractions, composition, and causal
control can be tested.

Choice B: revise the intervention/read-write class before another predicate
geometry experiment.

Exp 29 says affine readout plus Euclidean rank-1 same-read/write patch is not
enough on the registered predicates. A follow-up could ask whether a better
patch class turns decoded predicate information into causal control. This is
useful only if the patch-class question is the point; otherwise it risks
becoming more `pstack` mining.

Optional lower-value work: finish oracle-withdrawal units 4/5 from
`ORACLE_WITHDRAWAL.md` only if sampled-completion uncertainty or proposal
family competition becomes the specific goal. They are not needed before the
structured-completion direction.

## Prerequisites Before The Next Claim-Producing Experiment

Nail these down before writing a preregistration:

- Is `phi` within-horizon over the existing m-gram, or trajectory-level?
- What is the exact evaluator?
- What is the observable estimator?
- What is the no-information baseline for each predicate?
- What is the full/reference patch baseline for each predicate?
- What counts as predicate sufficiency: event-probability error, KL,
  thresholded classification, or something else?
- What is a typed counterexample?
- Are you refining residual abstractions, predicate partitions, or both?
- How will predicate cherry-picking be prevented?
- What composition relation is actually being tested?
- What failure would change the next experiment?

If those answers are not stable, write a design note instead of an
experiment.

## Candidate First Registration After The Break

A reasonable first serious registration:

```text
On a colored or weakly coupled stack process, define a small fixed predicate
suite: one stack predicate, one mode/color predicate, one interaction
predicate, one redundant control, and one rare/vacuity control.

Ask whether residual abstractions that preserve the full m-gram also preserve
these predicate marginals, whether predicate-specific abstractions differ
from full-distribution abstractions, and whether the registered composition
relation holds.
```

Keep it exact. Keep the predicate suite small. Register the known-answer
relationships before training or running the model.

Possible outcomes:

- full-distribution preservation implies all predicate preservation: coherent
  but not yet adding pressure;
- predicate preservation diverges from full-distribution preservation: the
  original conceptual payload has resurfaced;
- composition fails: real abstract-interpretation content;
- observable predicate estimates fail exact audit: measurement repair before
  semantic expansion;
- causal control fails again: revise patch/intervention class before
  interpreting geometry.

## Useful Commands

Run local predicate checks:

```bash
uv run python predicates.py
uv run python scripts/predicates/predicate_targeting.py --selftest
```

Regenerate exp 29 output:

```bash
uv run python -u scripts/predicates/predicate_targeting.py --outdir out/pstack-L4 | tee out/exp29_pstack-L4.txt
```

Inspect recent record:

```bash
sed -n '1,120p' EXPERIMENTS.md
sed -n '1,220p' experiments/28-consolidation.md
sed -n '297,379p' experiments/29-predicate-targeting.md
sed -n '1,220p' ORIGINAL_SIN.md
```

## The Re-Entry Mantra

```text
richer target,
same discipline.
```

The next phase should make the completion target more structured without
giving up the property that made the toy phase valuable: every claim is
checkable, every failure is typed, and broad labels are earned only after
their scope is nailed down.
