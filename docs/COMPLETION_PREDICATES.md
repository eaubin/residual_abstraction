# Completion-predicate bridge — phase design map

**Status: ACTIVE (opening).** Provisional name; rename freely.

> **TENTATIVE — read this first.** This is a design map, **not** a plan and **not**
> a pre-registration. It scopes the question, the deliverable, the load-bearing
> decisions, and the reusable harness. It does **not** prescribe an experiment
> order. The items under "Goals" and "Candidate first moves" are *things to do and
> targets to hit*, not a sequence to execute — the next experiment is chosen by what
> the last one returns. The previous phase's L0–L4 sequence was reordered three
> times (rungs pulled ahead, deferred, and one trigger left unsatisfiable as
> written); that is the expected fate of any order written in advance here. Each
> claim still gets its own `experiments/NN-*.md` pre-registration and the
> `EXPERIMENT_REVIEW_PROTOCOL.md` review pauses; this map binds none of those
> numbers.

Checked against `docs/PHASE_PLANNING.md`: it serves the aim (how completion-relevant
quantities are stored/maintained/**combined**/protected as relations in the parts),
touches primarily the **composition** cell (binding of what-to-where; the
read/write asymmetry; one-variable-vs-several) and re-enters **dynamics** and
**packing** as the loop demands, and it is leashed to the intervention/completion
thread (commitment 5). Durable commitments are not restated here.

## Vocabulary (load-bearing terms; full definitions in `FORMALISM.md`)

- **m-gram `q`** — the model's exact next-`m`-token continuation distribution at a
  position (the probability it assigns to each length-`m` continuation). This is the
  "completion" everything is scored against.
- **`E_q[φ]`** — the expected value of predicate `φ` under `q`: the average predicate
  score over the continuations the model would produce.
- **Residual abstraction** — a grouping/coarsening of residual-stream states (e.g. a
  projection onto a subspace). "Sufficient for `φ`" means it preserves `E_q[φ]`.
- **Interchange** — a causal test: copy one unit's activation from a *source* run
  into a *clean* run and measure how far the output moves toward the source.
  On-manifold, unlike zero-ablation.
- **Observable supervision** — discovery and probes may use only model outputs and
  completions; exact process state / ground truth is evaluation-only, never a
  training signal.
- **V-information** — how much a *bounded probe class* can extract about a target;
  the resource-indexed measure behind "what is findable" and the stopping rule.
- **CEGAR** — the propose-abstraction → test → refine loop; its value is the typed
  failures it produces, not the fixed point.

## Why this phase, why now

This is the bridge `docs/archive/ORIGINAL_SIN.md` deferred — and its gating
conditions are now **met**: oracle-withdrawal closed (exp 28), state-localization
closed (exp 44), the battery and ledger reflect both. The bridge is therefore not
blocked; it is **overdue**.

The original payload was a relation between two abstraction languages — one over
residual states, one over **named properties of futures** — measured for which
residual abstractions are sufficient for which completion predicates, how precision
is lost, and how they compose. The program retreated to a single completion-side
scalar (full-horizon KL) and probed the residual side, and every recent phase ran
into the same wall: intervention questions are secretly structural, and we were
answering them with a degenerate completion side. The completion-side lattice — named
predicates, an algebra, typed counterexamples — is the missing object. It is the
shared prerequisite for the two long-range goals this phase exists to serve.

## The two goals, and why they are one loop

- **Inference** — infer completion-relevant structure *from observations*, rather
  than detecting behaviors we named in advance.
- **Inverse intervention** — map sets of rollouts with a named property back to
  residual structure that can be modified at runtime.

These are not separable. Correlational inference from the residual side provably
collapses to variance — unguided residual mining keeps rediscovering the
highest-variance directions, which need not be the completion-relevant ones
(exps 6/8/9); the only non-collapsing inference is **causal**, and causal inference
*is* intervention. And intervention needs a candidate structure to act on — which is
the inference. The loop has a name in this project: **CEGAR**. It degenerated last
time into a behavioral direction-adding loop for one reason — the **test side lost
its teeth** (counterexamples mined by variance, not by intervention). This phase
puts causal interventions back on the test side. Inference and intervention are the
two faces of one refinement loop; the phase is organized around the loop, not around
either half.

## Phase question

```text
Relating residual abstractions to NAMED completion predicates (not the full-horizon
scalar): which residual abstractions are sufficient for which predicates, how do
they compose, and — in an entangled regime — what is the reachable, precision-graded
structure of the map from a desired completion-property to a runtime residual edit?
Measured behaviorally, supervised on observables, with no privileged decomposition.
```

## The deliverable, and the regime it assumes

**Entanglement is the operating condition, not an obstacle to design away.** LLMs do
not keep completion-relevant facets separable; a *learned* model entangled a
generator built to be separable (Dyck-2, exps 40/42/44). Backward-designing a toy
*for* separability builds something unrepresentative. So the earlier finding that
you cannot tell whether an intervention is *specific* without a control variable
that is both movable and known to lie outside the target bundle (exp 36) stops
being a limitation to escape and becomes the subject.

The consequence is that the honest deliverable is **not** "I isolated predicate P."
It is the **reachability and precision structure of the map**: given an intervention
class, which regions of the completion-predicate lattice are reachable, what
co-moves, and at what cost — a surface with error bars, indexed by probe class,
resources, distribution, and tolerance. This is the program's endgame —
quantifying how much completion-relevant structure escapes a given probe, and
stating what does not — turned toward control. A relationship across a family of
toys is worth more than another isolated result.

Obstacles here are **measured boundaries, not walls**: entanglement beyond a null,
higher-order structure invisible to a linear probe, behavior unidentifiable within a
resource class — each is a boundary to *measure and index* (V-information measures
how much a given probe class can extract), not a wall to remove.

## Registered design decisions (load-bearing; settle in the first pre-registration)

These are the calls that get encoded in the scaffold, so they are fixed here and
inherited, not re-litigated per experiment.

1. **Sufficiency criterion = mean-predicate-error.** A predicate is a graded
   function `φ: continuation → [0,1]` (a span score, not a boolean event). An
   abstraction is sufficient for φ to the extent it preserves `E_q[φ]` — the
   expected score over the completion distribution — measured as `|E_q[φ] −
   E_q′[φ]|`. For within-horizon φ this is an exact marginal of the m-gram `q` we
   already compute — the cheapest possible criterion.
   - **Not KL.** KL is the *full-distribution* object — the thing this phase moves
     off of. Do not carry it to predicates.
   - **`accuracy@threshold` is parked** for the labeled/real-LLM regime, where φ is a
     *label you predict* (a classifier with label noise — a V-information question),
     not a defined function. On a toy φ *is* ground truth; there is nothing to be
     accurate against. Same predicate, two regimes, two criteria.
2. **Correlational sufficiency before the causal loop.** Establish the predicate
   layer and its verdicts with correlational sufficiency first; add causal
   interventions (the loop's teeth) only once the layer is validated. The scary
   measurement problem (next item) does not exist until then.
3. **The null model of entanglement is a prerequisite for the causal stage, not a
   blocker for the early ones.** Once interventions are on, "P moved and Q tagged
   along" means nothing without a baseline for how much Q moves for boring reasons
   (finite-step curvature, shared normalization, non-orthogonal basis, sampling
   noise — the geometric account from exp 42: pushing along any direction through a
   smooth readout picks up cross-terms). A finding is structure only if the
   coupling exceeds/differs-from that null. Without it the loop is unfalsifiable
   ("it's entangled"). Design the null **before** the first causal rung.
4. **Inference runs in two lanes** (see Scout procedure): a non-claim
   hypothesis-generation lane from the start, and a claim lane gated behind a
   **pre-registered inference *procedure*** (not pre-registered predicates) with
   held-out evaluation and a causal test. Registering the procedure, not the
   predicates, is what keeps inferred suites from drifting to whatever flatters the
   current abstraction — and is exactly where CEGAR failed before, so the procedure
   must have causal teeth or it collapses again.
5. **Predicate suite: pre-registered and small to start** (ORIGINAL_SIN's
   anti-cherry-pick), with the reason each predicate exists. Inference of the suite
   itself is the claim lane above, earned later, not assumed now.
6. **Horizon is set by the predicate suite, not inherited.** The standing `m=3`
   horizon is **too short** for this phase — temporal and binding predicates over a
   three-token window are largely vacuous. Choose `m` per experiment so the
   registered predicates are non-vacuous (expect longer horizons than prior phases).
   If a property needs more than a bounded window, **defer it** — do not reach for
   trajectory-level/unbounded predicates now. Exact `E_q[φ]` stays the criterion;
   watch its cost as `m` grows.
7. **Predicate representation = a small fixed menu of bounded templates, NOT an
   open-ended language.** Keep it deliberately tiny. Each predicate is a templated
   check over the next `m` tokens, seeded by the prefix's exact state, with **bounded
   quantifiers only** (every "exists/within" ranges over the ≤`m`-token window) and
   **saturating counters** (count up to a small constant, then stop — finite by
   construction, so the Turing-complete failure mode cannot arise). Fixed template
   shapes, each with a couple of parameters:
   - **threshold count** — "#(symbol `s`) ≥ `c` in the next `m`";
   - **bounded reachability** — "depth 0 reached within `k`";
   - **next-match binding** — "the next close matches the prefix top" (no quantifier);
   - **one bounded-order template** — "an open-`x` precedes the next close within `k`".

   Combine **only by `∧` (composition needs it) and `¬`** — no recursive nesting, no
   general grammar. **Pushdown and trajectory-level/unbounded predicates are out of
   scope**, not an escape hatch; binding stays prefix-seeded and bounded. Everything
   is finite by construction, so `E_q[φ]` is exact and cheap.

   **This menu is the ceiling of what is permitted, not what to build first.** Start
   with one or two trivial predicates computed directly from `q` (no template engine,
   no automata compiler); build a template or a counter only when a registered
   predicate actually needs it. If the simplest predicates show no pressure over the
   plain scalar, the phase abandons cheaply — that is the leash, not a failure.

## Vehicle policy (the toy is OPEN)

The vehicle is **not chosen** in this map; it is chosen per experiment, by the
phenomenon to force. Selection criteria:

- **Exactly auditable** completion targets (exact `E_q[φ]`), cheap checkpoint.
- Hosts **named, graded predicates** with at least one **binding/interaction**
  predicate (the composition cell needs more than one factor).
- Rich enough to teach something about the *layer*, not re-derive a known result.

**Dyck-2 is in scope only as a machinery testbed** (layer self-tests with
hand-computable answers — its poverty is a feature there); it is **not** a
vehicle for a claim (its depth/type entanglement is already characterized and
exhausted — a claim there would be a dot already plotted).

**Open candidate for the first claim toy (undecided):**

| candidate | adds | cost |
|---|---|---|
| colored Dyck-2 | a color/binding factor on top of depth/type → binding + temporal predicates ("next close matches top color"; "open red then close red before terminal") | small extension of existing Dyck machinery; one new checkpoint |
| weakly-coupled mode×stack | a hidden mode modulating the stack → latent ambiguity (same prefix, different regime) + genuine interaction predicates | larger step; new generator + checkpoint |

Both stay exactly auditable. Colored Dyck is the smaller step and maps onto the
span/binding intuition; mode×stack is closer to real-LLM ambiguity. **Decision
deferred** to the first claim pre-registration.

## Process-agnostic by construction (reuse = the generalization lever)

Reuse is a first-class requirement, not a convenience — and it is **how this phase
pays the standing generalization debt.** Every result so far is indexed to one
checkpoint; nothing has been shown to survive a vehicle change. If the predicate
layer, generator interface, exact evaluator, intervention harness, and scorers are
written **process-generic**, then adding colored-Dyck-3, mode×stack, a parity
language, etc. is *config + a generator + an exact evaluator*, not a rewrite — and
**running the same registered construct across a toy ladder becomes the transfer
test itself.** A finding that holds across the ladder is a line; one that breaks is a
typed transfer failure (both are useful results).

Build only the seams known to vary (process, predicate, intervention class,
granularity); do not pre-build for needs not yet seen (add a seam only when an
experiment needs it). Shared code lives in the library (`battery.py` / `expcommon.py`, a new
predicate module if it earns one); frozen scripts import from it, never the reverse.

## Scout procedure (non-claim lane — procedures need clarity)

A **scout** is an exploratory probe that may peek, use one seed, or run unguarded.
Its rules are exact:

1. A scout **must never** enter a verdict, an `ASSUMPTIONS.md` row, a `BATTERY.md`
   row, or a phase conclusion. It informs *routing and hypotheses* only.
2. A scout is recorded as **clearly non-claim**, with its date, seed(s), and the
   script, in the scout log (`docs/SCOUTS.md`, created on first scout). It is
   "burned": peeked or calibration data cannot later be re-used as evidence
   (generalizing the burned-seed rule).
3. **Promotion to a claim** requires a fresh, pre-registered confirmatory run on
   unburned seeds, under the full review protocol. For *inferred* structure,
   promotion additionally requires the registered inference **procedure** + held-out
   evaluation + a causal test (decision 4 above).
4. If a scout settles a question cheaply, the response is to **raise the next
   question or change the vehicle — not to pay for a confirmatory rung that would be
   theater** (exp 43/44 lesson). Confirm only what will be *cited*.

## Goals (what the phase must return — not an order)

- A **completion-predicate layer**: graded `φ: continuation → [0,1]`, exact `E_q[φ]`,
  an observable estimator, a small algebra (¬/∧/∨ + one temporal operator), the
  baseline set (no-information, full-patch, reference-patch, raw m-gram), and
  self-tests with hand-computable and adversarial predicates that fire the verdict
  branches (`PREDICATE_VACUOUS`, `COMPOSITION_FAIL`, `OBS_EXACT_DRIFT`).
- A **sufficiency result**: do residual abstractions sufficient for the full m-gram
  (the `k*` sufficient subspace the existing discovery machinery already produces)
  remain sufficient for named predicates, and do predicate abstractions **compose**?
  Composition is computed by running the predicate automata for `A` and `B` against
  `q` in parallel to get `E_q[φ_{A∧B}]` (this needs only a computable expectation,
  not language-class closure), then asking whether the residual abstraction
  sufficient for `φ_{A∧B}` relates to those for `φ_A`, `φ_B` (direct sum /
  containment / interaction direction). *The exact comparison and its tolerance are
  the first claim's to fix.*
- The **causal loop** with teeth: interchange-tested refinement, against the
  entanglement null.
- The **reachability/precision structure** of the property→edit map in the entangled
  regime, indexed by probe class and resources (with a declared, V-information-style
  stopping rule).
- **Transfer evidence**: the same construct run across a toy ladder (the
  generalization lever above).

## Candidate first moves (revisable — NOT a sequence)

Read as "plausible next steps," reordered freely by results:

- Build + self-test the predicate layer on Dyck-2 (machinery only; non-claim).
  Open the scout lane here.
- A first correlational sufficiency + composition claim on a richer toy (colored
  Dyck or mode×stack — undecided).
- Turn the loop causal: interchange on the test side, the entanglement null, and
  (optionally) promote inference from the scout lane to the registered-procedure
  claim.

The hard problem (the entanglement null) is deliberately **downstream** of the cheap
layer-validation, so the early moves cannot be blocked by it.

## Falsifiability and routing

Each experiment must return a result that changes the next.

| outcome | reading | routes to |
|---|---|---|
| full-distribution sufficiency **implies** all predicate sufficiency | predicate layer coherent but **adding no pressure** — target too close to the old scalar | enrich the predicate suite or the toy; the layer is validated but not yet earning its keep |
| predicate abstractions **diverge** from full-distribution abstractions | the original payload has resurfaced — predicates see structure the scalar hides | characterize the divergence; this is the phase's central positive |
| **composition fails** (A∧B unrelated to A, B as registered) | a real abstract-interpretation finding | study the interaction direction; higher-order structure is live |
| coupling present but **within the entanglement null** | co-movement is geometric, not structural (per exp 42) | report as null-consistent; do not call it binding |
| coupling **exceeds the null**, reachable map is **low-dimensional** | a genuine, runtime-relevant control handle exists | characterize the property→edit map; feed the inverse-intervention goal |
| observable predicate estimates **fail exact audit** | measurement repair, not semantic expansion | fix the estimator before any predicate claim |
| inference **collapses to variance** again | the causal teeth are insufficient | repair the test side; do not promote any inferred suite |

## Practical norms

- **Accelerators**: torch code uses MPS/CUDA when available (`load_model` moves to
  the accelerator); never run heavy forward passes on CPU; confirm the device before
  a long run. (`AGENTS.md` working norms.)
- **Cheap by construction**: frozen-checkpoint probes, analysis stages consume
  caches not models, the cheapest toy that exhibits the phenomenon. Recent phases
  became expensive; reuse (above) is the antidote.
- **Pre-registration + two review pauses** per claim (after registration/before
  first run; after run/before conclusions). A missed pause quarantines the
  experiment (exp 41).

## Scope and non-goals

- No claim beyond each experiment's registered checkpoint, predicates, positions,
  intervention class, and tolerance. Toy facts until the ladder shows transfer.
- No privileged decomposition; units are architecture-given; directions, where
  reached, are defined by behavioral effect.
- No real-LLM or labeled-data claim. The annotated-span / safety direction is a
  *different project*, downstream of this phase earning the diagnostics it would
  reuse; it is explicitly out of scope here.
- No isolation claim in an entangled regime; the deliverable is reachability/
  precision structure, not "I moved P alone."

## To settle at consolidation (forward-bind, light)

- Whether the predicate layer / exact-`E_q[φ]` evaluator is promoted to `BATTERY.md`
  infrastructure or stays library code.
- What the sufficiency/composition results say for the coupled-stack-state ledger
  row (`ASSUMPTIONS.md`) and whether the schema finally goes process-agnostic /
  multi-axis (the debt carried from exp 40/44).
- Whether `docs/SCOUTS.md` and the scout rule graduate into the review protocol /
  `END_OF_PHASE.md` as standing fixtures.
</content>
</invoke>
