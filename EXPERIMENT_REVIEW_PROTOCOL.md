# Computational Experiment Review Protocol

This protocol is for reviewing claim-producing computational experiments
in this repository. It is not generic code review. The reviewer evaluates
whether the experiment, implementation, verdict logic, and conclusion all
refer to the same registered construct.

Use it at two pauses:

1. **Pre-registration review**: after the writeup and runnable script are
   committed, before the first run.
2. **Result review**: after the run artifact exists, before conclusions are
   accepted as the canonical record.

The default review stance is conservative. A clean run is not enough if
the experiment measures the wrong object, uses a privileged signal in the
wrong place, or lets a narrow probe support a broad conclusion.

A phase-closing **consolidation** (e.g. exps 22, 28) is reviewed under this
protocol too, but its once-per-phase write obligations — promote to
`BATTERY.md`, settle the ledger, archive the phase doc — live in
`END_OF_PHASE.md`.

## Pre-Registration Review

Ask first whether this is the right experiment.

- **Phase fit**: the question should advance the current research plan
  rather than chase an attractive side thread. If it is a follow-up, it
  should name the prior failure type or scope debt it resolves.
- **Construct validity**: each arm should measure the concept named in the
  writeup. Names must match implemented quantities. Do not allow
  "depth", "state", "rank-1 existence", "transport", or similar terms if
  the script measures only a proxy with narrower semantics.
- **Verdict-name discipline**: cap new verdict names. A new branch must be
  justified as a genuinely new failure type, not a relabeled local outcome;
  local branches stay local until a conclusion deliberately promotes them. Every
  local verdict name carries a one-line plain-language gloss in the writeup, so a
  reader never has to infer a custom label's meaning from the code.
- **Outcome table**: the writeup should register the matrix of possible results
  with the route each implies, and the script should fill it — not a narrative
  prediction interpreted after the run. Prefer a small outcome space (roughly
  4–6) with a real chance of stopping a branch over a many-celled table whose
  cells cannot each change the next step.
- **Interpretability of outcomes**: every registered prediction should
  have adjudication rules before the run. The rules should distinguish
  transfer, recalibration, known failure type, new failure type, and
  investigation halt.
- **Scope indexing**: tolerance policy, horizon, distribution, reference
  patch, probe/interpreter class, and coordinate regime should be explicit
  where they affect the claim.
- **Ground-truth discipline**: privileged labels, beliefs, hidden states,
  or exact oracles may be used for evaluation only unless the experiment
  explicitly registers a supervised-on-ground-truth control. Proposal and
  probe fitting should use observable completion behavior.
- **Anti-vacuity**: the no-information baseline, full/reference patch, and
  relevant controls should be present when the conclusion depends on
  improvement or separation.
- **Code implements the registration**: the script should already contain
  guards, self-checks, verdict predicates, output tables, and halt
  conditions. Do not accept "we will interpret this after seeing it" as a
  preregistered verdict.
- **Maintenance cost**: new helpers should either be genuinely
  experiment-specific or belong in shared scaffolding. Avoid adding a new
  dependency or abstraction when a local pattern already exists.

The most important pre-run question is:

> If every registered predicate returns a clean number, will we actually
> learn the thing this experiment claims to test?

If the answer is no, block the run until the writeup, script, or question
is narrowed.

## Result Review

Start from the committed output and compare it to the registered
predictions. Findings should lead the response, ordered by severity.

- **Verdict fidelity**: each predicate should be evaluated exactly as
  registered. Conditional, skipped, or untestable predicates must not be
  counted as passes.
- **Conclusion discipline**: conclusions must not outrun the measured
  construct. A one-write probe failure is not proof that no direction
  exists; a two-condition result is not a general robustness theorem; a
  proxy label is not the latent variable unless that equivalence was
  validated.
- **Failed predictions**: failed predictions are findings, not material to
  edit away. Decide whether they are known failure types, recalibrations,
  or new failure types requiring a follow-up.
- **Calibration and baselines**: check whether observable and exact scores
  still agree where the conclusion relies on observables. Check that the
  baseline is not already sufficient under the registered tolerance.
- **Scope wording**: summaries must preserve tolerance, horizon,
  distribution, probe class, and reference patch. Prefer narrower wording
  over graceful prose.
- **Record consistency**: experiment writeup, script docstring,
  `EXPERIMENTS.md`, and run artifact should agree. The detailed writeup is
  canonical; index rows and docstrings should be **short pointers, not
  summaries** (exp-15 policy) — a back-annotation belongs in the prior
  writeup, not inlined into its index row.
- **Verdict partition (FORMALISM §6.1)**: when a member has a
  recalibrate/tolerance branch, check it against §6.1 rule 9 — opposite
  failure directions are separate branches and recalibrate ≠ fail. This
  exact collapse recurred three times (exps 25–27); do not rely on review
  to catch it a fourth.
- **On conclusion — propagate the resolution (mandatory grep step).** When
  a conclusion resolves or revises a quantity an earlier experiment
  asserted, `grep` *every* doc for the resolved quantity (`EXPERIMENTS.md`,
  `ASSUMPTIONS.md` ledger, `BATTERY.md`, `SYNTHESIS.md`, the prior writeups)
  and update or back-annotate each mention — the ledger's "update every row
  that mentions it" rule applied across docs, not just the ledger. Stale canonical records
  were caught three times (exp 24←25, 26←27, ledger/27←28) for lack of this
  step; it is the last action of writing a conclusion.

## Conceptual-Error Forcing Functions

Construct/code correctness review catches bugs; it does not catch a verdict that
turns on a number the design cannot support — the error class that survives a
clean run and an approving review. These two checks are required for any verdict-
or routing-bearing experiment (a verdict resting on a threshold comparison is the
central case), at both review pauses. The same checks belong author-side: the
pre-registration writeup should already contain the confound table and the
baselines, and review verifies them. Each is an artifact to produce in writing,
not a box to tick.

- **Confound enumeration on the load-bearing quantity.** Identify the single
  quantity the headline verdict most depends on. List at least three distinct
  mechanisms that could produce the same value, and for each name the design
  element that excludes it, or write "not excluded." A headline resting on a
  quantity with mostly-unexcluded confounds is a blocking finding, independent of
  code correctness. At result review, re-score the same list against the realized
  numbers: state which confounds the data excluded and which remain live.
  Conclusions may rest only on what the data excluded.
- **Measured-but-unadjudicated, and the missing baseline.** Two failures hide in
  "the numbers came out clean." First, list every quantity the script computes
  but the verdict function never reads: a measurement printed but unused carries
  interpretive weight a reader assigns it that the verdict does not — either fold
  it into the verdict or state it is descriptive only. Second, for every
  threshold ask what value the "obviously true" case actually reaches: a cutoff
  is only interpretable against the value a genuine positive achieves (its
  ceiling) and the value pure noise achieves (its floor). If neither is measured,
  the threshold comparison cannot be read, and the verdict that turns on it is
  provisional — require the baseline before the threshold is load-bearing.

## LLM-Work Creep

When an LLM implemented or amended the experiment, explicitly check for:

- broad labels replacing narrow constructs;
- skipped or failed predicates smuggled into a positive summary;
- existential claims from finite probes;
- post-hoc interpretation added without preregistered adjudication;
- stale claims copied from prior phases or experiments;
- verbose narrative that obscures the actual verdict;
- unreviewed helper functions, dependencies, or generated files.

LLM-work creep is usually a record problem before it is a numerical
problem. Fixing the wording can be enough if the code measured a useful
narrow construct.

## Maintainability Review

Review maintainability only as it affects future experiments and the
trustworthiness of the record.

- Prefer existing helpers and local patterns.
- Keep concluded scripts frozen unless a review explicitly calls for a
  fix to their record or reproducibility.
- Remove dead code, unused imports, and stale comments in living scripts.
- Do not introduce heavy dependencies for analysis that should consume
  existing caches or tracked artifacts.
- Shared abstractions are justified only when they reduce repeated live
  machinery or support a registered back-check.

## Review Output

Use this shape unless the user asks for something else:

1. Findings, ordered by severity, with file and line references. For any
   verdict- or routing-bearing experiment, Findings must include the two
   conceptual-error artifacts — the confound table for the load-bearing
   quantity, and the baseline (ceiling/floor) check for each threshold — or an
   explicit statement of why they are not applicable. A review that omits them is
   incomplete, not approving.
2. Open questions or assumptions, if any.
3. LLM-work creep and maintainability notes.
4. Verification performed or not performed.

If there are no blocking findings, say that clearly. Do not hide a
non-blocking conceptual mismatch; label it as residual risk or wording
polish.
