# Intervention Class Benchmark — Phase 3 Design

**Status: live (Phase 3).** Archive to `docs/archive/` at the Phase 3
consolidation, per `END_OF_PHASE.md`.

This document scopes the next phase after the oracle-withdrawal arc and the
predicate-targeting pilot. It is **not** a pre-registration. It is a design
map for a rigorous, falsifiable sequence of experiments about intervention
classes in small transformers.

The immediate motivation is experiment 29: two within-horizon predicate
probabilities on `pstack` were linearly readable and exact-calibrated, but a
rank-1 same-read/same-write patch along the readout direction did not move the
predicate probability. That result does not show that predicate-relevant
structure is absent. It shows that the tested intervention class was not a
causal handle.

The phase question is therefore:

```text
Given a candidate model-internal variable, which intervention classes can
write, erase, swap, or preserve it in a way that predictably changes model
behavior?
```

The point is not to add another toy for its own sake. The point is to learn
which intervention primitives deserve to be carried into later toy families,
composition experiments, and eventually larger models.

## Terminology Hygiene

Use terms at the narrowest level where they are defined.

### Widely Used Terms

These terms are broadly legible in mechanistic-interpretability or causal
modeling contexts:

| term | intended meaning here |
|---|---|
| toy transformer | a small transformer trained on a controlled synthetic process |
| residual stream | the transformer activation stream at a named layer/position |
| activation patching | replacing or modifying activations and measuring behavioral effects |
| interchange intervention | swapping an internal candidate variable between examples |
| linear probe / readout | a fitted linear map from activations to a target quantity |
| path patching | patching a component output or path, such as attention-head or MLP output |
| causal control | changing an internal quantity changes the measured behavior in the predicted direction |

### Repository Terms

These terms have project-specific definitions. Use them only with their scope
indices, or translate them in outward-facing summaries.

| repo term | local meaning |
|---|---|
| observable closure | model-vs-model behavioral improvement under a patch |
| exact closure | the same kind of score audited against the toy process's ground truth |
| `rho` / equivalence ratio | behavioral distance from a trusted reference patch, normalized by unpatched distance |
| battery | the six calibrated diagnostics in `BATTERY.md` |
| typed failure | a named failure mode with a concrete diagnostic and branch |
| trusted reference / declared anchor | the reference patch used to normalize behavioral comparisons |
| predicate marginal | `E_q[phi]`, a scalar property of a completion distribution |

### Phase-Local Terms

These terms are allowed in this phase but should not be promoted to
project-level labels without a later consolidation.

| phase-local term | meaning in this document |
|---|---|
| intervention class | a family of activation modifications, such as same-read/write, oblique read/write, learned write, or path patch |
| read covector | a linear functional used to measure the value to be copied or set |
| write vector | a direction in activation space used to change the residual stream |
| same-read/same-write patch | the exp-29-style patch where the read direction and write direction are the same vector |
| oblique read/write patch | a patch that reads with one linear functional and writes along another direction, constrained so the write changes the read value by the intended amount |
| fixed-read write search | hold a registered readout fixed and search only for the write side of the intervention |
| predicate control | reduction in error between patched predicate probability and source predicate probability |

Experiment-local verdict names may be introduced later, but they must remain
local until a consolidation deliberately promotes them.

## Scope

This phase is about **intervention classes**, not about proving a new toy
process is semantically rich. The first experiments should therefore reuse the
existing `pstack-L4` checkpoint and exp-29 predicates where possible.

In scope:

- small transformer checkpoints with exact or cheap ground truth;
- layer-local interventions on residual stream or component outputs;
- within-horizon predicate marginals already supported by `predicates.py`;
- observable scoring with exact endpoint audit where available;
- explicit controls for predicate room, specificity, and held-out transfer.

Out of scope for the first block:

- claims about real LLMs;
- broad claims about all predicates;
- trajectory-level predicate semantics;
- SAE/circuit claims unless introduced as proposal families in a later block;
- new process training unless the existing `pstack` test is non-diagnostic.

## What We Know Before This Phase

From experiment 29:

- `phi1_next_closes` and `phi2_net_return` vary on `pstack`, decode affinely,
  and pass exact endpoint audit.
- Same-read/same-write rank-1 patches along their affine readouts do not move
  predicate probability.
- Angles from those readouts to PCA/core are descriptive only, because causal
  control failed.
- Therefore, predicate readout success is not enough to infer predicate
  intervention geometry.

From experiments 10-16:

- Separating read and write can matter enormously: the same write can be weak
  under the wrong read and useful under a clean diagnostic read.
- Fixed read menus and simple spectral read families were insufficient in the
  adversarial Mess3 setting.
- Learned reads can work under behavioral scoring, and observable/exact
  agreement survived strong optimization pressure.
- Learned reads can also be position-entangled or optimizer-landscape-specific.
- Therefore, oblique read/write patches are promising, but they must earn
  causal control, specificity, and transfer; they are not explanations by
  themselves.

## Falsifiability Standard For This Phase

Every experiment in this phase should be able to return at least one result
that would change the next experiment. A useful experiment distinguishes among
these possibilities:

1. the variable is readable but not writable by the tested class;
2. the variable is writable, but not through the registered readout;
3. the intervention moves the target behavior but also moves unrelated
   behavior too much;
4. the intervention works only on discovery positions or seeds;
5. the full/reference patch has no room to move the predicate, making the
   test non-diagnostic;
6. the observable score and exact audit disagree in a dangerous direction;
7. the result is positive and transportable enough to justify testing a richer
   toy process.

A positive result is only meaningful if it beats no-information, random-write,
and same-read/same-write baselines, and if a full or reference patch shows
there is predicate-level room to close.

## Standard Measurement Table

Each experiment should produce a table with one row per intervention family
and target. Detailed thresholds belong in the later pre-registration, but the
columns should be stable:

| column | purpose |
|---|---|
| target | predicate or latent variable being controlled |
| read source | how the value to be set is measured |
| write source | how the activation update direction is chosen |
| patch point | layer, position scope, and component if not residual-stream-wide |
| predicate room | unpatched-vs-source room and full/reference-patch room |
| target control | how much the intervention moves the target behavior toward source |
| specificity | how much non-target behavior changes |
| exact endpoint audit | whether observable endpoints match toy ground truth |
| held-out transfer | position/seed/process split behavior |
| failure branch | experiment-local typed outcome |

The output should report curves where natural: write-strength dose response,
rank/dimension staircases, and train/held-out trajectories. Single points are
acceptable only for preflight implementation checks.

## Non-Experimental Implementation Steps

These should happen before the first claim-producing pre-registration.

1. **Patch API.** Add a small intervention module or battery helper for:
   same-read/same-write patches, rank-1 oblique patches, rank-k oblique
   composition, read-fixed/write-search patches, and strength-scaled patches.

2. **Predicate-control scorer.** Factor the exp-29 predicate-closure logic into
   a reusable helper: compute predicate room, target-control score,
   random-write floor, full/reference ceiling, and exact endpoint audit.

3. **Write-source constructors.** Provide reusable write candidate builders:
   affine-read direction, CEGAR/core basis directions, source-target residual
   deltas, random directions, and learned write directions.

4. **Self-tests with known answers.** Include synthetic linear tests where an
   oblique patch must set a read functional exactly, where singular
   `C^T W` is caught, where strength zero is no-op, and where a full patch has
   no predicate room.

5. **Report schema.** Add a common printed table for intervention-family
   comparisons. Avoid experiment-local prose doing the aggregation manually.

These steps are implementation scaffolding, not scientific evidence. They can
be committed before the pre-registration pause.

## Experiment Sequence

The numbers below are design labels, not experiment numbers. Final experiment
numbers and detailed thresholds belong in pre-registration files. I0-I4 are the
intervention-class sequence; choosing or training a richer toy is an exit gate,
not an experiment in this phase. I2-I4 are conditional branches: a result from
I1 may make some of them unnecessary or change their order.

This is a forward outline, not a stateful router. The detailed arms and entry
conditions for steps past the immediate next one are **illustrative** — they will
be firmed (and often reshaped) at each step's own pre-registration. When a result
diverges from a sketch below, that divergence is recorded in the experiment
writeup, which is canonical; this section is at most strike-and-pointed to it, not
re-derived in place.

**Next-experiment bar.** Before registering the next experiment, name the
carry-forward decision its outcomes would change — whether the program carries
residual rank-1 oblique, manifold/interchange, or path/component interventions
forward (or stops residual-level work for the target). If no registered outcome
would move that decision, do not run it: the project's value is typed surprise,
not volume, and re-measuring `pstack` only earns its cost when a result changes
which class advances. A second read atlas, for instance, does not clear this bar;
the I2 read/write discriminator does.

### I0 — Implementation And Parity Preflight

**Purpose.** Build the reusable intervention and predicate-control helpers, and
show that they reproduce exp-29's same-read/same-write result on the existing
artifact.

**Vehicle.** Existing `pstack-L4`, L1, `m=3`, positions and predicates from
experiment 29.

**Non-claim outputs.**

- same-read/same-write helper reproduces `c_w ~= 0` for the two decoded
  predicates;
- full-patch predicate room is reported for all registered predicates;
- synthetic oblique-patch self-tests pass;
- table format is stable.

**What would block I1.** If the helper cannot reproduce exp 29, or if the
full/reference patch has no predicate room for the decoded predicates, do not
run an oblique experiment. Fix the measurement stack or choose a different
predicate target.

### I1 — Fixed-Read Oblique Write Search On `pstack`

**Purpose.** Decide whether exp 29 was a write-parameterization failure.

**Question.** Given the affine predicate readout from exp 29 as a fixed read,
can a separate write direction causally control the same predicate probability?

**Targets.** Start with `phi1_next_closes` and `phi2_net_return`, because they
were readable and exact-calibrated in exp 29. The all-neutral and first-match
predicates remain controls for vacuity and interpreter limits unless a later
registration changes their role.

**Intervention arms.**

1. same-read/same-write baseline from exp 29;
2. write constrained to the declared `cegar` core;
3. write from source-target residual deltas stratified by predicate difference;
4. learned write direction with the read fixed;
5. random write controls matched in norm and strength.

**Potential outcomes and weights.**

| outcome | interpretation | generalization weight | informativeness |
|---|---|---:|---:|
| core write works | predicate read can be written through existing behavioral state | medium | high |
| delta/learned write works but core write fails | predicate control exists outside the declared core or through a different chart | medium | high |
| learned write works only on discovery positions | oblique writes are overfitting or position-entangled | low-medium | high |
| no fixed-read write works, but full patch has room | affine predicate readout is not a causal read or L1 is wrong | medium | high |
| no full/reference patch room | target is non-diagnostic for intervention testing | low | medium |
| observable/exact endpoint drift appears | predicate scoring cannot support geometry claims yet | high for method, low for intervention claim | high |

**Decision use.** A positive, held-out-stable fixed-read result justifies a
richer intervention comparison. A clean negative with full-patch room routes to
interchange/path interventions before designing new toys.

### I2 — Read/Write Pair Discriminator

**Entry condition.** Run this only if the read side is genuinely ambiguous after
I1 — fixed-read writes fail despite predicate room, or work weakly/nonspecifically
enough that the exp-29 readout may be the wrong causal read. A strong, specific,
held-out-stable fixed-read write narrows I2 to confirmation.

**Pre-I2 read status (resolved downstream — see exp 30/31/32 writeups; not
re-derived here).** I1 (exp 30) stalled before the write question: the exp-29
single global read did not transport. Exps 31–32 then typed the read as genuinely
**position-specific** (`POSITION_SPECIFIC_CONFIRMED`, not shared-with-drift). Two
consequences bind the next preregistration: (a) fix a **position-conditioned**
(transport-valid) read, never the exp-29 single global readout, which would
re-fight the transport wall rather than test write freedom; (b) there is **no
inherited "I1 best fixed-read intervention"** — I1 never adjudicated a write — so
the fixed-read baseline must be **run explicitly** as a position-conditioned
fixed-read write search (the I1 question, finally answered). Note this makes the
genuinely-next step a fixed-read write search with a repaired read (an I1 re-run),
*not* the read-varying comparison below; the read-varying arm earns its cost only
if that baseline is weak or nonspecific. That baseline was run as
**exp 33** (`experiments/33-poscond-read-write-search.md`) and concluded
`NO_POSCOND_READ_WRITE_WORKS(phi1_next_closes,phi2_net_return)`: the repaired
position-conditioned reads decode with room, but the registered fixed-read
rank-1 oblique write menu does not stably control either predicate. Canonical
detail and caveats live in the exp 33 writeup; read-status caveats (e.g. a
milder shared subspace is not excluded) live in the exp 31/32 writeups.

**Cost-control amendment before full I2.** Because a negative learned read/write
pair search could be optimizer-limited rather than scientifically decisive,
**exp 34** (`experiments/34-matched-delta-gate.md`) is registered as an I3-lite
matched activation-delta feasibility gate before spending on full I2. If matched
near-manifold deltas work at interpolation strength, full I2 has a concrete
signal to approximate; if they fail despite own-delta room, further rank-1
search should be downscoped or redirected to patch point/path diagnostics.
Exp 34 concluded `NONSPECIFIC_DELTA`: the matched deltas move each predicate
(over floors, with transfer) but co-move the other predicates, failing per-target
specificity. Because the two targets are coupled stack-state readouts, this may be
correct identification of one variable rather than non-specific leakage. Before
full I2, **exp 35** (`experiments/35-joint-state-separability.md`, design label
2b) re-adjudicates the exp-34 moves as joint bundle control (`phi1`+`phi2`) vs
out-of-bundle specificity (`phi4` + m-gram). Its branches route the carry-forward:
`JOINT_STACK_VARIABLE` -> cheap rank-1 joint-write confirmation (target
decomposition was wrong, not the class); `BROAD_STATE_REPLACEMENT` -> I4
patch-point or consolidation; `SEPARABLE_PREDICATES` -> entanglement rejected,
full I2 or I4. Canonical detail lives in the exp 35 writeup.

**Purpose.** Distinguish "wrong write for a good read" from "wrong readout for
the predicate."

**Question.** Holding the target predicate fixed, does allowing the read side
to vary improve causal control, specificity, or transfer over a
position-conditioned fixed-read baseline?

**Minimal arms.** Keep this small. The preregistration should choose at most two
new read families in addition to the fixed-read baseline.

- a position-conditioned fixed-read write search (the fixed-read baseline, run
  explicitly — see Pre-I2 read status; there is no inherited I1 result);
- one jointly learned rank-1 `(read, write)` pair with `read(write)=1`;
- one constrained read family, such as delta-predictive reads or core-coordinate
  reads;
- matched random read/write controls.

Do not add rank-k composition in this experiment unless two rank-1 pairs have
already passed target control, specificity, and held-out transfer. Composition
is a follow-up, not a way to rescue a weak rank-1 result.

**Required controls.**

- target predicate control;
- non-target predicate movement;
- full m-gram movement, so a predicate win is not just broad distribution
  replacement;
- held-out positions and fresh seeds;
- exact endpoint audit;
- parameter-count or search-budget accounting, so a learned pair is not
  compared naively against a much smaller fixed-read baseline.

**Potential outcomes.**

- fixed-read and learned/constrained-read arms both work: the exp-29 readout was
  adequate; write choice was the main limitation.
- learned/constrained read works and fixed-read fails: the affine predicate
  readout was readable but not a usable causal read.
- learned/constrained read works only in-split or nonspecifically: extra read
  freedom is overfitting or broad behavior replacement.
- no read/write pair works while full patch has room: rank-1 residual-level
  oblique intervention is not the right primitive for this target.

**Decision use.** I2 should return one routing decision: carry fixed-read
oblique patches forward, carry learned/constrained read/write pairs forward, or
stop residual-level rank-1 oblique work for this target.

### I3 — Interchange-Matched Activation Deltas

**Purpose.** Test a more causal-scrubbing-like intervention that uses observed
activation differences rather than arbitrary linear writes.

**Question.** If examples are matched on nuisance variables and differ in the
target predicate probability, does swapping or adding their activation delta
move the predicate more reliably than learned linear writes?

**Why this matters.** If I1/I2 fail, the problem may be that linear write
vectors are off-manifold. Matched-example deltas stay closer to observed
activation geometry.

**Design requirements.**

- matching criteria must be registered before the run;
- predicate-difference bins must be fixed before the exact audit;
- random matched deltas and mismatched deltas are controls;
- dose response should be reported.

**Decision use.** If matched deltas work where linear oblique writes fail, the
next phase should emphasize manifold-aware or interchange interventions. If
matched deltas also fail despite full-patch room, the target may live outside
L1 residual writes or require component/path-level intervention.

### I4 — Patch-Point Localization Follow-Up

**Entry condition.** Run this only after residual-level tests have produced a
specific localization question. Two examples justify I4: (a) I1-I3 fail cleanly
despite full-patch room, suggesting the residual patch point may be wrong; or
(b) a residual-level intervention works but is too broad, suggesting the target
control may be localized to a component path.

**Purpose.** Decide whether the failure or nonspecificity comes from patching
the wrong place, not from the target or score.

**Question.** For the same target and the best residual-level intervention from
I1-I3, does moving the patch point to a component output or neighboring layer
improve target control, specificity, or transfer?

**Minimal arms.** This is not a full circuit search. The preregistration should
name a small, justified set of patch points.

- best residual-stream arm from I1/I2/I3;
- one earlier or later residual layer, chosen from a registered depth/profile
  reason;
- attention-output patch at the same layer/positions;
- MLP-output patch at the same layer/positions;
- no-op, random-component, and full-component controls.

Avoid head-by-head or path-by-path search unless a prior diagnostic identifies
a small candidate set. Otherwise I4 becomes a new circuit-discovery phase.

**Potential outcomes.**

- component patch works and residual patch fails: the target is better treated
  as path/computation-local than as a residual-state variable.
- residual works and components do not: residual-state intervention is adequate
  for this target.
- component patch improves specificity over a broad residual patch: carry the
  component patch point forward for this target class.
- all component patches are nonspecific or weak: do not keep expanding patch
  points without a new diagnostic; consolidate the residual-level failure.

**Decision use.** I4 should either nominate a patch point for later experiments
or close the patch-point branch. It should not open an unbounded component
search.

## Exit Gate: Whether To Start A New Toy Phase

Choosing or training a richer toy is **not** part of this intervention-class
benchmark. It is the decision this phase should make more informed.

Only after I0-I4 should the program decide whether the next phase should use
colored Dyck, a weakly coupled mode x stack process, a switching grammar, or no
new toy yet. That choice should be backward-designed from known predicate
relationships and registered as its own phase or experiment block.

Use the intervention benchmark to route the handoff:

- If an intervention class can control `pstack` predicates with specificity and
  held-out transfer, carry that class to a richer toy for binding or
  composition questions.
- If no intervention class controls `pstack` predicates despite full-patch
  room, do not train a richer toy yet. First resolve patch point, path, or
  manifold issues.
- If `pstack` is non-diagnostic because there is no predicate room, the next
  phase may need a toy designed specifically to create predicate room and known
  causal directions.

## What Else Is Needed For A Well-Scoped Phase

### Stable Target Definition

Every experiment needs a target whose status is known before intervention
claims:

- non-vacuous variation on the evaluation distribution;
- observable estimator defined without exact ground truth;
- exact endpoint audit available for the toy;
- full/reference patch room;
- no-information baseline.

Without these, an intervention failure is ambiguous.

### Intervention-Specific Baselines

Each intervention family needs controls matched to its degrees of freedom:

- same-read/same-write baseline;
- random write/read controls with matched norm or rank;
- no-op and full-patch controls;
- source-target swap direction controls;
- mismatched or shuffled-pair controls for interchange-style interventions;
- strength curves, not only one patch strength.

### Specificity Metric

Target movement alone is not enough. Each experiment needs a pre-registered
specificity measurement, for example:

- movement of non-target predicate marginals;
- full m-gram KL movement relative to target-predicate movement;
- degradation on unrelated held-out slices;
- whether the patch simply imitates full residual replacement.

A useful intervention should move the target more than it indiscriminately
moves everything.

### Transfer Axes

At least one transfer axis should be registered per claim-producing experiment:

- held-out positions;
- fresh pair/basis seeds;
- fresh model seed if cheap enough;
- mild process/distribution shift;
- target predicates not used in fitting.

Do not require every axis in every experiment. Pick the axis that would most
change the next step.

### Exact Audit Boundary

Exact ground truth may audit endpoints and calibration, but proposal fitting
and intervention selection should remain supervised on observables unless a
control explicitly registers privileged training.

For predicate interventions, be clear about what exact audit means. It audits
source/target predicate endpoints. It does not automatically define the exact
truth of an off-manifold patched activation.

### Outcome Weighting Discipline

Generalization weights should be assigned before the run in coarse terms:
low, medium, high. These weights are not statistical probabilities. They say
how much the outcome should influence later experiments.

Default weighting for this phase:

- high weight for failures that persist despite full/reference patch room and
  exact endpoint calibration;
- medium weight for positives on `pstack` that transfer across positions/seeds;
- low weight for positives that appear only on discovery splits or only under a
  single heavily optimized read/write pair;
- low weight for any result where the target has little predicate room;
- high methodological weight for any observable/exact disagreement.

### Stop Rules

The phase should stop or redirect when one of these occurs:

- fixed-read and joint-read oblique interventions both fail cleanly while
  matched activation deltas succeed: move to interchange/manifold-aware work;
- residual-level interventions fail cleanly while component/path patches
  succeed: move to path localization;
- all tested interventions fail cleanly despite full-patch room: write a
  consolidation before adding new toys;
- a stable, specific intervention class succeeds on `pstack`: move to a
  richer toy designed for binding or composition;
- target room is absent: redesign the target or toy before any further
  intervention comparison.

## Relation To Later Experiments

This phase should make later work more informative by deciding what primitive
is allowed to stand in for "we can manipulate the internal variable." If that
primitive is still same-read/same-write projection, later predicate and
composition experiments inherit exp-29's known weakness. If the primitive is a
validated oblique, interchange, or path intervention, later experiments can ask
more meaningful questions about factorization, binding, and composition.

The intended deliverable is not a new final battery. It is a ranked set of
intervention classes with typed limitations:

```text
For this toy, target, layer, and distribution:
- which intervention classes control the target;
- which are specific;
- which transfer;
- which fail for readable-not-writable, no-room, off-manifold, nonspecific,
  position-entangled, or patch-point reasons.
```

Only after that consolidation should the program train or select a richer toy
for the next semantic question.
