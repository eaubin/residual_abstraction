# State localization — phase design map

**Status: live (draft).** A design map for the next phase, not a
pre-registration: it scopes the question, vehicle, target, method, and the
reusable harness, and sketches a forward experiment sequence. Each claim-producing
experiment gets its own `experiments/NN-*.md` pre-registration and the
`EXPERIMENT_REVIEW_PROTOCOL.md` review pauses. Archive to `docs/archive/` at the
phase consolidation, per `END_OF_PHASE.md`.

This phase is checked against `docs/PHASE_PLANNING.md`: it is the
architecture-first opening move that document's commitment 2 mandates, it touches
the **packing** and **dynamics** cells (admitted together, per commitment 4), and
it is leashed to the intervention/completion thread (commitment 5). The durable
commitments are not restated here; this document holds only phase-specific content.

## Why this phase, why now

The intervention-class benchmark (Phase 3) ended on a substrate verdict: `pstack`
could not decide whether an intervention was *specific*, because its two readable
targets were facets of one coupled stack-state bundle and the toy had **no
separable, high-room, out-of-bundle control**. Its exit gate asked for a richer
toy that supplies exactly that.

It also left a deeper diagnosis (`PHASE_PLANNING.md`): intervention questions are
structural questions we were answering blind, because the toys were never
understood as mechanisms. Before asking "can I move this variable specifically,"
ask the prior question: **where does the variable live across the model's parts,
and is it one thing or several?**

This phase answers that prior question on a toy that can finally separate the
targets.

## Phase question

```text
On a stack toy with genuinely distinct state facets, where does each
completion-relevant facet live across the architecture-given parts, is it
localized or spread, and do distinct facets occupy the same parts or different
ones — measured behaviorally, with no privileged decomposition?
```

A "different parts for different facets" result is the mechanistic form of the
separability `pstack` could not adjudicate; "same parts" is the coupling, given a
mechanism instead of an observation.

## Vehicle: Dyck-2

Dyck-2 (two bracket types) is the vehicle, and the reason is informativeness, not
richness for its own sake. `pstack` has one bracket type, so its state is a single
**where** (depth/parity) with no **what** — which is *why* it had no separable
control. Dyck-2's state has a genuine **what × where**:

- **depth** — how deep the stack is (a "where");
- **top-of-stack type** — which bracket type must close next (a "what").

These are *candidate* separable facets and constitute a **binding** (a type bound
to a stack position). Dyck-2 is therefore a candidate for the separable control
`phi4` failed to be — a candidate facet inventory the L0 gate must confirm
(room + dissociability), not a guarantee. It is already trained and
battery-validated (exps 19–22), so the checkpoint is cheap and its completions are
exactly auditable.

`pstack` is **not** in scope as a second vehicle; it adds a near-mimicry data point
that does not earn phase budget. A continuity cross-check there is a near-free
later add, not a pillar.

**Substrate gate (required before any claim).** "Separable, high-room" is a claim
to *check*, not assume — the exp-23/I0 precedent. The first rung establishes that
both facets are non-vacuous, observably estimable, exact-auditable, and that a
full/reference patch has room to move each. No room ⇒ redesign the target before
localizing.

## Target: {depth, top-of-stack type}

Chosen because their separability is the open question and Dyck-2 provides them
naturally. Each facet is defined as an **observable** on the completion
distribution (depth-sensitive continuations; mass on the valid closer for the top
type), with the exact Dyck oracle used for endpoint audit only. The headline
measurement is whether depth and type localize to the **same or different parts**.

## Method: localization by interchange patching

Units of analysis are **architecture-given** (commitment 1/2): block outputs
(attention vs MLP per layer × position), refining toward heads and then directions
only where a coarser map demands it.

Localization is by **interchange patching**, not zero-ablation: take a clean run
and a source run that differ in the target facet, patch one unit's activation from
source into clean, and measure how far the target observable moves toward source.
Interchange keeps the patched activation near the observed manifold (the Phase-3
lesson that on-manifold moves are the ones that act); zero-ablation would go
off-distribution and confound the robustness cell with an artifact. Attention-head
and position-resolved units are included, so cross-position propagation — the
exp-4/5 "propagated state vs per-position summary" question — is measured by the
same machinery (commitment 4: dynamics admitted, not bolted on).

**Scoring is fixed by what the experiment measures, not chosen** (all reuse
existing observable scorers):

- **target effect** — movement of the facet observable toward source under the
  patch (per unit);
- **total effect** — movement of the full next-token / m-gram distribution, so a
  unit specific to the facet is distinguished from one that matters for everything
  (the specificity axis);
- **standard controls** — random/irrelevant-unit floor, full-patch ceiling, and a
  mismatched-source check (a pair that does *not* differ in the facet must score
  ≈ 0). Seed majority and held-out positions per the battery.

Honesty is unchanged: patch selection and scoring are supervised on observables;
the exact oracle audits endpoints only.

## Reusable harness and modularity

The phase's first deliverable is infrastructure, because the four design axes
(toy, unit granularity, target, scorer) are exactly the seams that vary across
experiments — so building one configurable harness, rather than per-experiment
scripts, is what serves reuse, rigor, and informativeness together. Stages, each
with known-answer self-tests, in their library home (extend `battery.py` /
`interventions.py`; a new `localize.py` if it earns one; experiment scripts in
`scripts/localization/`):

```text
load checkpoint  →  build target-conditioned PairSet (clean vs source differ in facet)
  →  enumerate units at granularity g  →  interchange-patch each unit
  →  score (target effect, total effect, controls)  →  aggregate → importance map + verdict
```

- **Unit enumerator(granularity)** is the novel reusable core, designed so one
  knob *can* span block → head → direction. **L0/L1 build only the block (and
  head) levels** and self-test them; the direction level is deferred until L1
  demonstrates a reason to refine — do not build it ahead of that evidence. Coarse
  and fine then being the same code is what makes the later granularity sweep
  cheap, but the seam is added when earned, not up front.
- **Self-tests (rigor, built once, inherited by every experiment):** a synthetic
  where a known unit carries the facet must be recovered; a no-difference pair must
  score ≈ 0; a full patch must reach the ceiling; a mismatched source must not move
  the facet.
- Claim experiments become **thin configs** over validated stages — "bigger"
  experiments are more configurations, not more code.

Guardrail: build only these known-varying seams; do not pre-build stages for needs
not yet seen.

## The granularity sweep and the refinement boundary

Refining *below* an architectural unit is where the hard phenomena live
(`PHASE_PLANNING` commitment 2). Sweeping the enumerator's granularity knob and
recording where finer units stop being behaviorally adjudicable (importance no
longer separates from the floor; effects no longer compose) produces a deliverable
**line — informativeness vs granularity** — and that boundary is precisely where
superposition would announce itself. Meeting that difficulty is a goal, not a
failure (`PHASE_PLANNING` open tensions).

## Experiment sequence (forward sketch, design labels)

Numbers and thresholds belong in each pre-registration; entry conditions past the
next step are illustrative and will be reshaped by results.

- **L0 — harness + substrate gate (non-claim).** Build and self-test the harness;
  gate Dyck-2 for non-vacuous, exact-auditable, room-bearing {depth, type}.
  Committable before the pre-registration pause (the I0 precedent). Blocks claims
  if no room.
- **L1 — coarse localization.** Block granularity, layer × position, attention
  included. Importance map for depth and for type; the same-vs-different-parts
  verdict. First claim.
- **L2 — refinement.** Push granularity below blocks where L1 shows importance;
  locate the refinement boundary; the informativeness-vs-granularity line.
- **L3 — dynamics (conditional).** If L1 shows state carried across positions,
  localize the cross-position propagation (which heads move the facet forward) —
  the exp-4/5 question, re-asked mechanistically.

## Falsifiability and routing

Each experiment must return a result that changes the next. The phase's load-
bearing outcomes and what they route:

| outcome | reading | routes to |
|---|---|---|
| facets localize to **different** parts | separability has a mechanism; the `pstack` coupling was substrate, not law | an intervention phase at the located parts — the well-posed version of ICB |
| facets localize to the **same** parts | genuine coupling/binding into shared parts | characterize the binding; the coupling is a finding with a mechanism, not an artifact |
| state is **distributed/superposed** (refinement boundary is low) | the expected difficulty, now located | a how-to-handle-superposition phase; do not force a clean handle |
| no full/reference-patch room | target non-diagnostic on Dyck-2 | redesign the facet/target before localizing |
| target effect with no specificity (total effect dominates) | parts are not facet-specific; broad replacement again | refine granularity, or treat the facet as not separately localized |

## Graduation

The phase returns: a localization map per facet, a same-vs-different-parts verdict,
the granularity/refinement boundary, and (conditionally) the propagation map. It
graduates by routing the next phase — toward intervention at located parts, toward
characterizing a binding, or toward handling distribution/superposition — and by
feeding the intervention/completion leash with a structural picture the previous
phase lacked. This is a routing decision, not a generalization claim
(`PHASE_PLANNING` open tensions: experiment-verification here, generalization still
the standing open problem).

## Scope and non-goals

- No claim beyond the registered Dyck-2 checkpoint, layers, positions, target
  facets, granularities, and patch family.
- No privileged decomposition: parts are architecture-given; directions, where
  reached, are defined by the behavioral effect, never asserted as the model's
  intrinsic features.
- No mechanism-circuit completeness claim; localization is importance + specificity
  under interchange patching, with its typed confounds (redundancy/backup, off-
  manifold) named and controlled, not eliminated.
- No richer-toy or real-LLM claim; the vehicle is fixed to Dyck-2 for this phase.

## Open questions and risks (carried)

- **Room is not guaranteed** until the L0 gate passes; the separability premise is
  checked, not assumed.
- **Redundancy confounds localization:** if computation is duplicated across parts,
  single-unit patches under-read importance. Named as the robustness cell; a
  registered multi-unit/path control is the response if L1 shows it.
- **Superposition is expected** at fine granularity; a low refinement boundary is a
  result. The phase should not design around it.
- **Naming:** "state localization" is provisional; rename freely.
