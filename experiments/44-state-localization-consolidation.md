# Experiment 44 — State-localization phase consolidation (closes the phase)

**Status: CONCLUDED — phase closed.** A phase-boundary record, not a
claim-producing run (precedent: exp 22 closed Phase 2, exp 28 closed the
oracle-withdrawal arc). No script, no new measurement. Reviewed under
`EXPERIMENT_REVIEW_PROTOCOL.md` (fresh-session review, all phase writeups read —
faithful, no blocking findings; verdict softened to exp-42 scope). The once-per-phase
writes (`END_OF_PHASE.md` checklist) are executed and recorded at the foot. Phase
doc: `docs/archive/STATE_LOCALIZATION.md` (archived at this close).

## Phase verdict (fixed)

**On the registered Dyck-2 checkpoint, the two candidate completion-relevant
facets — `depth` ("where") and `top_type` ("what") — do not separate. They are
read through a shared bracket-readout mechanism, their coupling is geometric to
first order (curvature-dominated; no representational binding established — though
not excluded, exp 42), and graded depth is recomputed from the prefix bracket-
embeddings each step rather than stored internally. No specific intervention handle
for either facet that leaves the other untouched is *established* by the registered
results (40's depth direction drags type; 42's curvature; 43's locus with
specificity untested).**

This is the **"same parts" branch** of the phase routing table, delivered with a
characterized binding: the what×where binding Dyck-2 was *chosen* to supply (the
separable, high-room control `pstack` lacked) turns out, mechanistically, **not to
be separable**. That is a real answer — a coupling *with a mechanism*, not an
observation — but it is the deflationary one, and it routes the program off this
vehicle, not deeper into it.

Scope: this verdict is indexed to one Dyck-2 checkpoint (exp-19 config), the m=3
horizon, the registered positions/horizons, and the interchange/steering patch
families. It is a routing decision on a toy, **not** a generalization claim.

## The phase question and how each rung answered it

> Where does each completion-relevant facet live across the architecture-given
> parts, is it localized or spread, and do distinct facets occupy the same parts or
> different ones — measured behaviorally, with no privileged decomposition?

| rung | experiment | what it settled |
|---|---|---|
| L0 | 37 — substrate gate | `top_type → GO` (clean on every axis); `depth → FLOOR_FAIL`/HELD (close-readiness purity uncertified at m=1). Harness + enumerator built and self-tested. Dissociable single-facet pairs abundant (the real unknown, resolved favorably). |
| L1 | 38 — propagation gate | Graded depth (m≥2 forced-close conditional) is **carried/transportable** but **`DISTRIBUTED`**, not point-localized (recency-weighted; no small prefix window saturates). Subsumed the unsatisfiable old m=1 L3 trigger. |
| L-int | 40 — directional specificity | At full spatial support, the depth- and `top_type`-carrying **directions are not cleanly separable**: asymmetric `CROSS_DRAG` — `top_type` direction is `SPECIFIC`, but the depth direction **drags `top_type`** (larger at the deeper k=2 contrast). Both facets have a rank-1-per-position additive handle. (40's *registered headline* was `SEED_UNSTABLE`, driven by a borderline k=1; the substance is the stable k=2 `CROSS_DRAG`.) |
| — | 42 — readout mechanism | The depth→`top_type` drag is **geometric / curvature-dominated**, separable to first order — **not** a representational depth-conditional readout (the representational reading is withdrawn; the test that would establish it is absent by design). |
| L4 (pulled ahead) | 43 — counting mechanism | Graded depth is **recomputed from the bracket-embeddings** (premise gate PASS 4/4 fresh seeds; not internally stored), and the recomputation **`LOCALIZED_COUNTER`** to a small set (≈3 readout-window units) dominated by `(3,attn,3)@t+k`. Specificity at the locus left untested (F1). |

- **39** is an intentional gap (the original L2 coarse-localization, deferred behind
  L-int and never revived — no `DISSOCIATED`/`MIXED` outcome ever re-motivated it).
- **41** (product-counter substrate) is a **procedural failure / quarantined pilot**
  and contributes **no** evidence to this verdict.
- **L2 / L3** (coarse localization; refinement boundary + informativeness-vs-
  granularity line) were **not delivered.** L2 was conditional on a separable
  intervention target that never appeared (it was gated on a `DISSOCIATED`/`MIXED`
  L-int outcome; 40 returned `CROSS_DRAG`), so it is a correctly-unmet conditional,
  not a drop. L3's refinement-boundary / informativeness-vs-granularity line is the
  one stated phase deliverable that is **carried to the next phase, not retired** —
  it is low-value on this exhausted toy but is exactly the deliverable the next
  vehicle is meant to force (superposition appearing at a measurable boundary).

## Same-vs-different-parts — the verdict in detail

Two independent threads converge on **same parts / entangled**:

1. **Residual-direction level (40 → 42).** The depth and type directions are not
   cleanly separable: steering depth drags type, and the drag is geometric
   (readout curvature), not a representational binding.
2. **Architecture level (43, registered).** Graded depth localizes to a small
   readout-window set dominated by `(3,attn,3)@t+k`, and 43 establishes that
   specificity at that locus is *untested* — so on the registered record the head is
   not a *demonstrated* depth-specific handle. The same-parts verdict rests on this
   (43's registered locus) plus the registered coupling of thread 1; the burned
   scout below only adds confirmatory color (the head is positively `NONSPECIFIC`,
   type point-localized to it) and is **not** load-bearing for the verdict.

So `(3,attn,3)@t+k` is best understood not as a "depth counter" but as **the head
that fetches the next-relevant bracket from the prefix embeddings** — carrying that
bracket's *identity (type)* and its *stack-position (depth)* together, because they
are two faces of one fetched embedding. Depth and type do not occupy different
parts; they share the readout, and there is no separable handle on either alone.

This is the mechanistic resolution of the separability `pstack` could not
adjudicate (Phase-3 exit): the coupling is real and has a mechanism (geometric, at
a shared readout), not an artifact — but it is a coupling, so the well-posed
specific intervention the intervention thread (ICB) was building toward is **not
available on this vehicle**.

## Exploratory probes (non-claim, burned — never cited as evidence)

Recorded for the record, not for the verdict. A burned probe may inform routing; it
**must not** enter a verdict, ledger status, or battery row without a fresh
confirmatory run (the seed-700-burned rule, generalized).

- **Locus specificity scout (2026-06-23; 1 seed; `scratchpad/scout_locus_specificity.py`).**
  Type-contrast pairs (depth-matched, differing only in the readout `top_type`),
  splicing `(3,attn,3)@t+k`: the locus transports **type ≈1.0 vs depth ≈0.2** (4/4
  cells, type/depth 3.7–5.3×; depth-purity ≈0.01). Indicates the locus is
  **`NONSPECIFIC`** — the bracket-readout reading above. This is *why* a clean depth
  intervention at the locus is not well-posed, and it settled the question cheaply
  enough that a pre-registered specificity rung would have been confirmatory theater
  — the in-phase illustration of "if calibration settles it, raise the question or
  change the vehicle; don't pay the protocol." Burned (peeked).

## Program disposition

- **Closed.** The state-localization phase. Its question is answered (same parts /
  entangled / geometric / recompute-from-tokens), and the `END_OF_PHASE.md` trigger
  applies verbatim — the scout was the last cheap diagnostic and it confirmed rather
  than opened.
- **Optional / not pursued.** L3 refinement boundary on Dyck-2; a clean
  representational-binding test (same-depth matched-displacement / gating control);
  a confirmatory specificity rung. All low marginal value on an exhausted toy.
- **Next phase.** A **richer vehicle that forces what Dyck-2 refused to show** — a
  genuinely separable facet, or genuine propagated state that *cannot* be recomputed
  from the token bag (a task with real memory: a hidden register, a long-range
  dependency, modular counting), and/or enough scale that superposition appears and
  the refinement boundary becomes a measurable line. The vehicle is chosen by the
  **phenomenon to force**, not by convenience. This also finally puts the untouched
  **generalization debt** on the table: nothing in this phase has been shown to
  survive a vehicle change.

This is a routing decision (`PHASE_PLANNING` open tensions: experiment-verification
here, generalization still the standing open problem), not a generalization claim.

## Open debts carried forward

- **Generalization is untested.** Every phase result is indexed to one tiny
  checkpoint. The next phase must include a transfer check — does any of this
  motif survive a wider/different model — or it keeps accumulating toy facts.
- **Specificity instruments are the methodological weak point.** Position-resolved
  readout splices cannot probe a facet defined at a position they cannot causally
  reach (the 43-F1 wall). A specificity-capable instrument (matched baseline at the
  intervention's own operating point) is a prerequisite for the next phase's
  intervention questions, and should be designed before, not during, a rung.
- **Ledger-schema decision (registered 2026-06-23, exp 40).** The coupled-stack-
  state bet (`ASSUMPTIONS.md` row) now carries evidence from two processes (pstack,
  Dyck-2). Decide at this consolidation whether to promote it to a process-agnostic
  proposition with pstack + Dyck-2 as listed instances, and whether ledger status
  goes explicitly multi-axis. This is the one substantive consolidation call.
- **Recording home for non-claim probes.** Scouts currently have no durable,
  clearly-non-claim home. This consolidation's "Exploratory probes" subsection is
  the interim home; if scouts recur next phase, promote to `docs/SCOUTS.md` and add
  the burned-probe rule to `END_OF_PHASE.md` / the review protocol.

## Once-per-phase writes (END_OF_PHASE checklist — executed at close, 2026-06-23)

1. **Promote to `BATTERY.md`: nothing promoted (done).** Confirmed at review: none
   of the 6 frozen battery members is a Dyck-2 depth/type mechanism fact, and the
   durable artifact — the component-write interchange enumerator — is infrastructure
   in `localize.py`, not a battery member. The phase findings are toy-indexed ledger
   items, settled in `ASSUMPTIONS.md`, not battery rows.
2. **Settle `ASSUMPTIONS.md` (done).** The depth-recompute item is settled (final
   phase status); the coupled-stack-state row carries the phase-final Dyck-2 note
   (geometric, exp 42 + the 43 architecture finding). **Schema left as-is** by
   decision — no process-agnostic promotion, no multi-axis status this cycle.
3. **Cross-doc propagate-grep (done).** `EXPERIMENTS.md`, `ASSUMPTIONS.md`,
   `BATTERY.md` consistent. The exp-38 back-annotation and the `ASSUMPTIONS.md`
   depth-recompute item were already written by exp 43's conclusion. `SYNTHESIS.md`
   carries **no** depth-carried/propagated-state claim (grep clean) → no refresh
   needed (human-owned; the standing synthesis did not move).
4. **Archived `docs/STATE_LOCALIZATION.md` → `docs/archive/` (done);** `INDEX.md`
   row moved to the Archived section; the archived doc carries a phase-closed banner.
5. **`RESEARCH_PROGRAM.md`: no edit (done).** The phase order did not change —
   state-localization was the architecture-first opening move and it routes into the
   roadmap's existing "coherence under generation / richer processes" steps. Per the
   no-status-in-frame-docs norm, phase status lives here (exp 44) + `EXPERIMENTS.md`
   + `ASSUMPTIONS.md`, not in the roadmap.

## Scope honesty

This phase saw what is load-bearing for completions on the Dyck-2 evaluation
distribution, under interchange/steering patches, at m=3. "Same parts / entangled /
geometric / recompute-from-tokens" certifies a mechanism *on this toy under these
probes* — never identity with a human ontology, never off-distribution, and (until
the next phase tests it) never transfer. The strongest honest summary is: on a toy
built to host a separable what×where binding, the binding is not separable, and the
program's intervention thread needs a different vehicle to find a well-posed target.
