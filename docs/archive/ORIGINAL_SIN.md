# Original Sin: How To Grow Back Toward The Original Idea

## What The Original Idea Was

The original target was not "find a low-dimensional subspace." It was not
about any particular behavior class either. It was about a relation between
two abstraction languages:

- an abstraction language over residual states;
- an abstraction language over possible futures.

A residual abstraction might group states by any feature we choose to name:
a syntactic obligation, a discourse role, an entity being tracked, a latent
task variable, a hidden process state, a refusal condition, or a safety-
relevant feature.

A completion abstraction might group futures by any behavioral predicate:
whether a bracket is closed, whether a referenced entity is mentioned,
whether a question is answered, whether a plan is completed, whether a tone
changes, or whether some temporal pattern occurs.

These examples are deliberately arbitrary. The point is the machinery: given
abstraction families on both sides, can we measure which residual
abstractions are sufficient for which completion abstractions, how precision
is lost, and how abstractions compose?

The AI-relevant content was the abstract-interpretation machinery itself:
how abstractions refine, compose, lose precision, and concretize. The
residual-to-completion link was Galois-connection-shaped: not a literal
classical Galois connection, because the concrete semantics is a
probability kernel over futures, but still a relation between two
structured spaces of abstractions.

That was the payload.

## What The Program Became

The current project deliberately retreated to the smallest objects that
could be audited exactly.

- Residual side: a linear subspace of the residual stream. Refinement is
  mostly "add another direction." Composition is mostly direct sum.
- Completion side: the finite-horizon next-token distribution at a fixed
  position. Evaluation is mostly one scalar: KL or a derived closure score.

This lost most of the original semantic content. There are no named
completion predicates, no temporal structure over generated segments, and
little of the completion-side lattice the original idea needed. CEGAR kept
its name, but in its current form it is mostly a behavioral
direction-adding loop, not refinement against named spurious
counterexamples.

That was not a mistake. It was the price of exact calibration. Tiny HMMs,
Dyck, and `pstack` made it possible to validate the measurement harness:
observable scoring, intervention tests, reference selection, rho,
obs/exact agreement, and typed failures. The current machinery is a
calibration instrument. The danger is letting the instrument become the
whole project.

## The Bridge

The way back is not to discard the current program. The way back is to
recognize the semantic tower the current program sits inside:

```text
trajectory distribution
  -> finite-horizon m-gram q
    -> predicate marginal E_q[phi]
```

None of these objects is intrinsically "concrete" or "abstract." Those are
roles inside a chosen relation. The finite-horizon m-gram is an abstraction
of the full trajectory distribution, but it has served as the concrete
target for residual abstractions in the current experiments. A predicate
marginal is an abstraction of the m-gram, but it can serve as the concrete
target when asking whether a residual abstraction preserves that predicate.

The next conceptual object should be:

```text
residual abstraction alpha
tested against
completion abstraction phi
```

where `phi` is a measurable property of futures:

- the full finite-horizon token distribution, which is the current case;
- an event probability, such as "returns to depth 0 within k";
- a temporal predicate, such as "opens before a terminal, then closes";
- a binding predicate, such as "the close matches the top-stack color";
- eventually, a human-facing behavioral class, such as "asks a question."

This keeps the honest part of the current method: every claim is still
about a measured future behavior. But it reintroduces the missing
structure: named predicates, logical composition, temporal sequencing, and
typed counterexamples.

The decision-relevant fork is not whether `phi` is concrete or abstract.
The fork is what `phi` is a function of:

- **Within-horizon predicate.** `phi` is defined on the existing
  finite-horizon m-gram `q`. This is the cheap bridge: no new ground-truth
  object is needed, because `E_q[phi]` is a marginal of the exact
  distribution already computed.
- **Trajectory-level predicate.** `phi` is defined on a richer future object
  than the current `q`, such as a cross-segment property or an
  eventually/then/until property whose horizon is not the current fixed
  m-gram. This climbs one level up the tower. It is still exactly auditable
  on toys, but it needs new evaluation machinery.

Start with within-horizon predicates to recover a completion-side lattice
cheaply. Move to trajectory-level predicates only when the within-horizon
case has shown that the predicate layer and verdict logic are sound.

## Which Reduction To Attack First

There are three reductions in the current setup:

1. The interpreter class is linear or patch-based.
2. The residual abstraction is a subspace.
3. The completion abstraction is just a local finite-horizon distribution.

The first attack should be #3. The completion side is where the original
"temporal predicate over generated behavior" idea lived. The residual side
at least still has a refinement order. The completion side currently has no
named lattice at all.

So the post-oracle-withdrawal move should not be "find fancier residual
probes" first. It should be:

```text
keep exact toy audit,
but make the completion target structured.
```

## When To Do This

Do not start this while oracle withdrawal is still unsettled.

The right time is after oracle withdrawal has produced one of two clean
outcomes:

- `BATTERY_TRANSFERS` under the earned `pstack` reference, followed by
  consolidation; or
- a typed failure whose lesson is understood well enough that it does not
  contaminate the next phase.

The reason is practical. Structured predicates will add new degrees of
freedom: predicate choice, predicate algebra, event probability
estimation, composition claims, and richer counterexamples. If observable
scoring, earned references, obs/exact audit, or threshold policy are still
in flux, the new phase will multiply ambiguity instead of answering the
original question.

## Prerequisites

Before the first structured-completion experiment, the following should be
settled:

- The earned-reference story is closed: either usable under a declared
  anchor or failed for a named reason.
- Observable-vs-exact agreement is calibrated on the latest toy substrate,
  including intermediate-strength probes.
- The sampled-completion uncertainty policy is at least sketched, even if
  the first structured-predicate experiments use exact event probabilities.
- `BATTERY.md` and `FORMALISM.md` reflect the oracle-withdrawal outcome, so
  the new phase does not inherit live-edge folklore.
- There is reusable code for exact event probabilities at the intended
  semantic level. For the first bridge, that can be
  `E_q[phi]` for predicates over the existing finite-horizon m-gram.
  Trajectory-level predicates require separate evaluator machinery.

That last item is the key scaffold. Without it, predicate experiments will
be one-off scripts instead of a new semantic layer.

## The Required Scaffolding

Build a small completion-predicate layer before running a claim-producing
experiment.

Minimum useful interface:

- `Predicate` or `Event` objects with an explicit domain: finite m-gram
  continuation, bounded trajectory, or automaton-recognized trajectory
  property.
- Exact evaluator for within-horizon predicates:
  `E_q[phi]`, where `q` is the exact m-gram distribution.
- Later exact evaluator for trajectory-level predicates:
  `P(phi | belief)`, implemented by dynamic programming or a product with a
  predicate automaton.
- Observable estimator: `P(event | model continuation)` or sampled
  model-based estimate.
- Simple algebra: `not`, `and`, `or`, and one temporal operator such as
  `then` or `until`.
- Baselines: no-information event rate, full-patch event score,
  reference-patch event score, and raw m-gram score.
- Self-tests with hand-computable predicates and adversarial predicates.

Do not start with a large predicate language. Start with a few predicates
whose exact meanings are obvious and whose expected relationships are
registered.

## The Toy Processes Must Be Designed Backward

The current toys are good calibration instruments, but probably not rich
enough for the original conceptual payload. `pstack` was intended to be
richer than Dyck, but it came out close to variance mimicry: one stable
`k ~= 4` behavioral reference, `cegar` close to `pca`, and geometric
differences mostly behaviorally inert.

The toys themselves are arbitrary. Dyck, `pstack`, product processes,
colored stacks, and switching grammars are not the subject. They are
calibration environments: small systems whose latent factors, future
predicates, and interaction structure are known well enough that the method
can be audited. A toy is good only insofar as it instantiates the semantic
relationships the experiment claims to test.

The next toy should be designed backward from the predicate suite.

The criterion:

```text
Each predicate should correspond to a real, separable,
completion-relevant latent factor or interaction in the generator.
The known relationships among predicates should imply known
relationships among the residual abstractions, at least qualitatively.
```

A richer state space is not enough. A big HMM with many states can still
collapse onto one sufficient statistic. What is needed is semantic
structure whose composition matters.

The toy should include:

- a stack/depth factor, for temporal dependency;
- a mode factor, for hidden ambiguity not visible from depth alone;
- a color or binding factor, for attribute tracking across time;
- a high-variance junk factor, tempting but completion-irrelevant;
- a coupling rule, so some predicates require interactions such as
  mode x stack or color x stack.

## Candidate Process Ladder

A sensible progression:

1. Independent product, such as `Z1R x Dyck` or `Mess3 x Dyck`.
   This tests whether predicate-specific abstractions factor and whether
   the joint abstraction looks like a direct sum. It is a sanity check, not
   the destination.

2. Colored Dyck.
   Stack items carry type plus color. This supports binding predicates such
   as "the next close matches the top-stack color" and temporal predicates
   such as "open red, then close red before terminal."

3. Weakly coupled product.
   A hidden mode modulates Dyck opening probabilities, color probabilities,
   or terminal emissions. This creates factor structure plus interaction
   directions. It asks whether composed predicates require more than the
   sum of single-factor abstractions.

4. Switching grammar.
   A latent mode switches among small grammars. This is closer to real LLM
   ambiguity: the same surface prefix can imply different future regimes.

The first serious target should probably be a colored or weakly coupled
stack process, not another one-factor toy.

## The Experimental Vehicle Is Also Not Sacred

Transformers matter because the long-run target is a language model
residual stream. But a transformer is not required for every calibration
experiment. The required ingredients are:

- an internal continuous state that can be abstracted;
- an autoregressive or predictive semantics over futures;
- meaningful interventions on the internal state;
- exact or auditable completion-side targets;
- enough learned representation pressure that the result is not just the
  generator's hand-coded state exposed directly.

This should not become another open design axis. The default claim vehicle
remains the small transformer. Cheaper systems are allowed only as preflight
harnesses for narrow questions about the semantic layer or verdict logic.
They should not produce headline conclusions about residual streams.

Use a cheaper vehicle when the experiment would still be meaningful if the
internal state were just a learned continuous vector with interventions.
Use a transformer when "residual stream," layer locality, attention-built
state, or compatibility with the existing battery is load-bearing.

Good uses for cheaper vehicles:

- Predicate-layer self-tests: exact event probabilities, `and`, `or`,
  `then`, `until`, baselines, and vacuity checks.
- Known-answer CEGAR tests: construct states that must split under predicate
  `A` but not under predicate `B`.
- Composition sanity checks: verify that `A`, `B`, and `A and B` produce the
  expected verdict branches before involving transformer geometry.
- Failure-branch tests: make sure `PREDICATE_TOO_VACUOUS`,
  `COMPOSITION_FAIL`, `OBS_EXACT_DRIFT`, and related typed verdicts fire.
- Fast process-design checks: does the toy actually make the intended
  predicates separable, redundant, interacting, or rare?

Bad uses for cheaper vehicles:

- claims about transformer residual streams;
- claims about layer-local patching;
- claims that the existing battery transfers;
- claims of readiness for LLMs;
- architecture-specific representation claims.

Possible cheap vehicles:

- A recurrent model can test whether predicate-specific state abstractions
  and composition work at all, with much cheaper training and cleaner state
  interpretation.
- A learned next-state or predictive model can separate measurement
  questions from transformer-specific architecture questions. This should be
  treated as implementation scaffolding, not as a new research object.
- A hand-controlled nonlinear state model can provide adversarial
  known-answer cases for the predicate layer.

The transformer should re-enter when the question depends on transformer
facts: residual-stream geometry, layer-local patching, attention-mediated
state construction, or compatibility with the existing battery. It should
not be treated as mandatory overhead for questions about completion
predicate semantics, composition, or CEGAR against named counterexamples.

The transfer warning is serious. Cheaper preflight harnesses can introduce
bad assumptions: clean state factorization, easy interventions, low
estimator noise, or composition laws that hold only because the vehicle is
too transparent. Therefore any cheaper-vehicle result must be labeled as one
of:

- implementation validation;
- known-answer verdict validation;
- process-design screening;
- hypothesis generation.

It is not evidence that the result holds in a transformer until the same
registered construct is rerun in the transformer vehicle.

## Should The Battery Transfer?

The battery should not be expected to transfer unchanged in the strong
sense. It should transfer as a discipline:

- compute baselines;
- earn or declare the reference;
- separate observable from exact audit;
- report curves rather than points;
- use typed failures;
- halt when uncertainty or vacuity dominates.

Individual numeric thresholds and even some members may fail when the
semantic target changes from full m-gram distributions to structured
predicates. That is not automatically bad. In fact, predicate abstractions
should stress the battery. If every member transfers trivially, the new
target may be too close to the old scalar distribution setting.

What should transfer:

- the insistence on no-information and full/reference baselines;
- observable-vs-exact calibration where exact audit exists;
- held-out positions or held-out predicate slices;
- shift or contrast-retention checks;
- accept-count staircases instead of single fixed points;
- typed failure branches rather than post-hoc interpretation.

What may not transfer:

- rho's `0.25/0.5` bands;
- the meaning of "equivalent reference" for a predicate family;
- obs/exact agreement bands calibrated on full m-grams;
- the CEGAR proposal miner;
- the assumption that preserving the full m-gram behavior and preserving
  named predicates have the same acceptance geometry.

So the right expectation is not "the battery should pass." The right
expectation is:

```text
the battery should know how to fail.
```

If structured predicates break a member, the result is useful if the
failure is typed: lenient rho, predicate-vacuous target, composition
failure, observable/exact drift, contrast miss, reference overtrust, or
proposal collapse. The only bad outcome is an untyped surprise that forces
post-hoc interpretation.

## Predicate Suite Requirements

A useful predicate suite should contain at least:

- Single-factor predicates:
  - `A_depth`: return to depth 0 within horizon `m`;
  - `A_mode`: emit a mode-indicative terminal before return;
  - `A_color`: next close has the top-stack color.
- Interaction predicates:
  - `A_depth_and_color`;
  - `A_mode_until_close`;
  - "return after emitting a terminal whose color matches the top stack."
- Controls:
  - a high-variance but completion-irrelevant observable feature;
  - a redundant predicate whose event probability is already determined by
    another predicate;
  - a rare predicate to expose distributional invisibility.

The controls matter. Without them, the experiment can only find positive
structure; it cannot show whether the method distinguishes load-bearing
predicates from tempting but irrelevant ones.

## Questions To Nail Down Before The First Experiment

The most important formal question is not whether a predicate is "concrete"
or "abstract." It is:

```text
At what semantic level is the predicate defined?
```

The answer determines the experimental commitment:

- If `phi` is a function of the existing m-gram `q`, then the experiment is
  a cheap completion-side abstraction experiment. It adds named predicates
  and composition without changing the exact-oracle substrate.
- If `phi` is a function of trajectories richer than `q`, then the
  experiment expands the semantic domain. That can still be exact on toys,
  but it needs new evaluator machinery and should be registered as a larger
  move.

The same predicate marginal can play two roles. It is abstract relative to
the full future distribution, but it is the concrete target when judging
whether a residual abstraction preserves that predicate. The role depends on
the connection being studied.

Other questions that need registered answers:

- Is this first experiment restricted to within-horizon predicates over
  `q`, or does it include trajectory-level predicates?
- What counts as sufficiency for a predicate marginal: event-probability
  error, calibrated binary distribution KL, classification accuracy under a
  threshold, or something else?
- Is the residual abstraction judged against one predicate, a fixed
  pre-registered family, or a generated predicate lattice?
- What is the no-information baseline for each predicate?
- What is a typed counterexample?
  A natural version: two prefixes are close under the residual abstraction
  but separated by `E[phi]`.
- Does CEGAR refine the residual abstraction, the predicate partition, or
  both? If both are allowed, what prevents moving the target after seeing
  counterexamples?
- How is predicate cherry-picking prevented?
  Pre-register a small suite and the reason each predicate exists.
- What composition claim is being tested?
  For example: does the abstraction for `A and B` relate to those for `A`
  and `B` by direct sum, containment, or an interaction direction?
- Are full-distribution preservation and predicate preservation expected to
  have the same geometry, or is divergence between them the point of the
  experiment?
- What failure would actually matter?
  Examples: the predicate is vacuous; predicate scoring disagrees with full
  m-gram scoring; the composition relation fails; CEGAR collapses to PCA
  again; observable event estimates fail exact audit; a trajectory-level
  evaluator is too expensive or unstable.

## The First Bridge Experiment

After oracle-withdrawal consolidation, do one small bridge experiment before
declaring a new phase.

Example:

```text
On a colored or weakly coupled stack process, define three exact
continuation predicates: one stack predicate, one mode/color predicate,
and one interaction predicate. Ask whether residual abstractions that
preserve the full completion distribution also preserve these predicate
abstractions, and whether the predicate-specific abstractions compose in
the registered way.
```

This experiment should be small enough to fail informatively:

- If full-distribution preservation implies all predicate preservation,
  the predicate layer is coherent but not yet adding pressure.
- If predicate abstractions diverge from full-distribution abstractions,
  the original conceptual payload has resurfaced.
- If composition fails, that is a real abstract-interpretation finding.
- If observable predicate estimates fail exact audit, the next work is
  measurement repair, not semantic expansion.

## The Caution

Do not let "richer abstractions" become an excuse to drop oracle
discipline. The path back to the original idea is not to jump straight to
un-auditable human ontology in a real LLM. The path is:

```text
richer target,
same discipline.
```

Structured predicates should first be exact and boring enough to audit.
Only after that should they become human-facing and open-ended.
