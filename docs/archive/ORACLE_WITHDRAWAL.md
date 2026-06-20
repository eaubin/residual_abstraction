# Oracle-withdrawal rehearsal — next experimental program

This is the next program after Dyck transfer, named by its goal rather
than by a phase number.

## Goal

Test whether the method can select, validate, and compare residual
abstractions without privileged ground-truth structure, while keeping an
oracle hidden for audit.

The exact oracle is still available, but it is not allowed to construct
proposal families, choose the trusted reference, tune abstraction/verdict
thresholds, or write verdicts. It is revealed only at registered audit
points. This is the rehearsal for settings where the oracle will not
exist.

There is one controlled exception: the measurement-substrate experiment
may use exact audit to calibrate estimator budgets, confidence intervals,
and sampled-vs-exact error bars. That calibration must be completed
before abstraction results are interpreted. It may not tune proposal
selection, reference selection, acceptance thresholds, or verdict
branches after observable abstraction results are known.

The central question:

> Can observable diagnostics choose and validate residual abstractions
> well enough that the hidden oracle later confirms the protocol was
> honest, or do we discover typed reasons that oracle-free work is not
> ready?

## Position

I agree with the broad reviewer recommendation: do not jump to
TinyStories or real LLMs yet. The next target should be one new process
class with richer ambiguity than Dyck and an exact or high-quality audit
oracle.

Preferred target: a small PCFG or probabilistic stack grammar with
inside/outside-style exact completion measures. If that target fails
pre-registered feasibility gates, use a Dyck/PCFG hybrid that preserves
cheap exact evaluation while stressing reference selection and sampled
scoring.

The first preregistration must state concrete go/no-go gates before
choosing between those targets: runtime budget, model competence
threshold, exact/sampled audit tolerance, maximum horizon, PairSet size,
and what counts as brittle. Target fallback is allowed only through
those gates, not through convenience after seeing abstraction results.

The point is not to prove the battery again. Dyck already did a transfer
check. The point is to make the workflow live under future constraints:
references are earned, scoring may be sampled, and exact truth is
review-only.

## Non-Negotiable Discipline

- Proposal families may see only tokens, residuals, model outputs, and
  sampled completions.
- Reference selection must be done without oracle labels, hidden states,
  or exact belief features.
- Observable verdicts are written before exact audit is revealed.
- Exact closure is evaluation-only and appears only at registered audit
  points.
- Measurement-calibration thresholds (estimator budgets/error bars) are
  separate from abstraction/verdict thresholds. Exact audit may calibrate
  the former in the substrate block only; it may not tune the latter.
- Every claim remains indexed by tolerance, horizon, distribution,
  reference patch, proposal/interpreter class, and sampling budget.
- Failures are findings. They update the battery or the protocol; they
  are not edited away.

## Planned Experiments

Experiment numbers are administrative. The units below are named by what
they decide.

### 1. Measurement Substrate And Target Choice

Build the process, train a small model, and establish whether exact and
sampled completion measurement are trustworthy enough for abstraction
experiments.

Questions:

- Can the model reach a non-vacuous competence gate?
- Can exact or high-confidence completion measures be computed within
  the budget?
- Do sampled completion estimates agree with exact scores within usable
  error bars?
- What horizons and PairSet sizes are affordable?
- Does the full patch establish a non-vacuous ceiling, and do compact
  candidate references have enough observable room to be meaningfully
  compared?

The preregistration for this experiment must give numbers, not just
intent. At minimum:

- wall-clock and memory budget for exact evaluation and sampled
  evaluation;
- model competence threshold and validity-gate estimator;
- exact-vs-sampled agreement tolerance and confidence level;
- maximum horizon to attempt and stopping rule if cost grows too fast;
- PairSet size and minimum stratum counts;
- concrete brittleness criteria, such as unstable training across seeds,
  exact inference exceeding budget, sampled confidence intervals larger
  than the target diagnostic margin, or self-check failures.

Deliverables:

- process implementation;
- model checkpoint and validity gate;
- exact-vs-sampled calibration table;
- runtime budget;
- initial PairSet/self-check harness.

Stop before abstraction work if the measurement substrate is unstable.
This is a substrate gate, not a science result to finesse.

### 2. Hidden-Oracle Reference Selection

Select a compact trusted reference candidate without oracle access, then
reveal exact closure only for audit.

Candidate abstraction references should include the CEGAR core,
PCA/PLS/embedding controls, random/destructive controls, and cheap
observable-supervised candidates such as diff-in-means or probes trained
only on observable completions. SAE or dictionary features are optional
only if they are cheap and quarantined.

The full patch is not a selectable successful abstraction reference. It
is a ceiling, non-vacuity baseline, and calibration/control object. If
the protocol can only trust the full residual, the registered outcome is
a fallback/failure branch: no compact earned reference was found.

Registered outcomes:

- reference selected correctly: observable diagnostics choose a candidate
  that exact audit confirms is strong and well separated;
- full-patch fallback: compact candidates fail, while the full patch
  remains the only trustworthy ceiling/control; this is not a successful
  earned-reference result;
- selected reference too weak: the chosen candidate is internally
  coherent but exact closure is too low to anchor rho;
- reference ambiguity: multiple candidates pass observable diagnostics
  but differ exactly or disagree on downstream rho;
- no trustworthy reference: no candidate clears the non-vacuity,
  separation, and held-out checks;
- oracle-reveal inversion: observable ranking disagrees with exact
  ranking enough to invalidate the selection policy.

This is the most load-bearing experiment. Do not proceed to a battery
transfer experiment until reference selection has either worked or
failed in a typed, protocol-updating way.

### 3. Battery Transfer With Earned Reference

Run the six battery members using the selected reference from the prior
experiment, not an oracle-blessed reference.

Claims:

- observable closure calibration;
- rho separation around the earned reference;
- held-out-position or held-out-slice gain;
- shift-retention with competence and clean-gain guards;
- accepted-cell exact audit at registered reveal points;
- CEGAR staircases over registered tolerance and horizon grids.

Expected new failure types:

- reference overtrust: rho appears meaningful relative to a weak
  reference;
- candidate-reference overfitting: the selected reference passes
  discovery but fails held-out slices;
- observable/exact inversion: observable closure accepts cells that
  exact audit rejects;
- contrast-distribution miss: PairSets miss rare but
  completion-relevant structure.

The result does not need to be all-positive. It needs to say whether the
battery still knows how to diagnose its own failure when the reference
is earned rather than privileged.

### 4. Sampled-Completion Uncertainty Policy

Degrade the oracle deliberately. Run the same verdicts with finite
sampled completions at several budgets, with exact audit revealed only
after observable/sample-based verdicts are fixed.

Questions:

- How many sampled completions are needed for stable accept/reject?
- Which diagnostics remain robust under sampling noise?
- Which verdicts become uncertain rather than pass/fail?
- When should the protocol halt instead of forcing a typed result?

Deliverable: an uncertainty policy for future no-oracle work. The policy
should define confidence intervals, indifference bands, minimum effective
sample sizes, and an explicit `UNCERTAIN` verdict when uncertainty
dominates the claimed gap.

Expected new failure types:

- sampled verdict instability: accept/reject flips across seeds or
  completion samples;
- uncertainty domination: confidence intervals are larger than the
  diagnostic margin;
- sampled/exact drift: sampled scoring is stable but biased relative to
  exact audit.

### 5. Constrained Proposal-Family Competition

Only after reference selection and sampled-scoring policy are usable,
compare proposal families under one shared process, PairSet construction,
reference policy, horizon grid, tolerance policy, and scoring budget.

Candidate families:

- current CEGAR;
- PCA, PLS, embedding, and random controls;
- diff-in-means directions;
- linear probes trained on observables;
- cheap sparse dictionary or SAE-style latents if implementation cost
  stays low and dependencies are quarantined.

Question:

> Given the same hidden-oracle protocol and scoring budget, which
> proposal families preserve completion behavior, which fail, and what
> failure type do they instantiate?

This is not a broad literature comparison. `RESIDUAL_METHODS_NOTE.md`
becomes relevant here only insofar as it supplies candidate proposal
families that can be evaluated under the existing battery.

### 6. Consolidation

Update `BATTERY.md`, `FORMALISM.md`, and the experiment index only after
the hidden-oracle program earns a clear record.

Possible conclusions:

- hidden-oracle reference selection works under declared indices;
- reference selection fails for a named reason and becomes a new battery
  member or guard;
- sampled completion scoring works only above a budget threshold;
- sampled scoring is not stable enough, so the next required work is
  substrate improvement rather than larger models;
- one or more diagnostics are demoted, narrowed, or given an
  `UNCERTAIN` branch.

## Surprise Triage

Surprises are not all equal. The decision to spend extra time pinning
one down depends on whether later experiments would inherit it.

### Halt Immediately

Stop the current experiment, fix or register a substrate follow-up, and
do not interpret numbers if:

- process or PairSet self-checks fail;
- the model fails the competence/validity gate;
- full patch or no-information baselines make the target vacuous;
- exact-vs-sampled calibration is unstable in the measurement substrate;
- oracle information leaked into proposal, reference selection, or
  verdict construction;
- a code path lacks an adversarial or synthetic self-check for a verdict
  branch that can affect the conclusion.

### Follow Up Before The Next Experiment

Register a focused follow-up before moving on if the surprise affects a
quantity that later experiments will use as infrastructure:

- the selected reference is ambiguous, weak, or overfit;
- observable ranking inverts under exact audit;
- sampled-score uncertainty changes accept/reject decisions;
- PairSet construction misses a rare but completion-relevant contrast;
- a threshold or tolerance policy becomes load-bearing;
- a new failure type changes the interpretation of a battery member.

This is the practical rule: spend extra time when the failed prediction
would be an input to the next experiment's design, reference, threshold,
or verdict logic.

### Record Inline And Continue

Record the failure as an expected typed branch and continue when:

- the failure was pre-registered and does not feed later blocks;
- it affects one proposal family but not the shared measurement or
  reference protocol;
- exact audit narrows a claim but leaves the decision rule intact;
- a diagnostic becomes descriptive rather than adjudicating for this
  experiment.

### Defer To Consolidation

Defer only when the surprise changes interpretation but not subsequent
experimental machinery. Examples: a proposal family underperforms, a
working reference is uglier than expected, or two non-selected controls
swap order without affecting the selected reference or verdict bands.

## Closing Bar

Do not require all tests to pass. Require that the protocol knows what
to do with its failures.

The program closes when it can honestly say one of:

- hidden-oracle reference selection works under these indices;
- hidden-oracle reference selection fails for typed reason X, and the
  battery/protocol has been updated;
- sampled completion scoring is not stable enough, and the next required
  work is an uncertainty/substrate repair rather than a larger model;
- proposal-family comparison is meaningful under the shared hidden-oracle
  protocol, or it fails for a typed reason.

The next step after this document is not to run everything. It is to
pre-register the measurement-substrate experiment with code and pause for
review.
