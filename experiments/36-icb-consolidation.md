# Experiment 36 — Intervention-Class Benchmark (Phase 3) consolidation

**Script:** none. This is the Phase-3 consolidation record and documentation
update, not a new model run (precedent: exp 22 closed Phase 2, exp 28 closed the
oracle-withdrawal arc).

**Status: concluded.** Consolidates the intervention-class benchmark
(exps 29–35), records the typed result and the carried-forward intervention-class
ordering, settles the `ASSUMPTIONS.md` ledger, promotes the durable typed
findings to `BATTERY.md`, archives `INTERVENTION_CLASS_BENCHMARK.md`, and states
program disposition.

## Question

Phase 3 (`INTERVENTION_CLASS_BENCHMARK.md`) asked, of a candidate model-internal
variable on `pstack`:

> Which intervention classes can write, erase, swap, or preserve it in a way that
> predictably changes model behavior — with specificity and held-out transfer?

The motivating wound was exp 29: two within-horizon predicates (`phi1_next_closes`,
`phi2_net_return`) that *decode and pass exact endpoint audit* but do **not** move
under a rank-1 same-read/same-write patch. The intended deliverable was a ranked
set of intervention classes with typed limitations, used to route the handoff to
a richer toy (the exit gate). The consolidation question:

> What did exps 29–35 settle about intervention classes on `pstack`, what carries
> forward, and is the phase's exit gate decidable from the evidence?

## The result — a substrate-adequacy verdict, not a class ranking (the headline)

**No intervention class earned a specific, transferable positive on `pstack`
predicates.** But the phase did not fail to measure; it resolved into a different,
sharper claim than the one it set out to make:

> `pstack`'s predicate inventory cannot support the specificity discrimination
> that intervention-class ranking requires. Its two readable targets behave as
> facets of **one coupled stack-state bundle**, and the process offers **no
> high-room, known-out-of-bundle control predicate**. So "moves the target
> specifically" is not decidable on this toy, independent of the intervention
> class tested.

This is a typed **substrate** result, and it is the most useful thing the phase
produced: it converts exp 29's possibly-trivial gripe ("one rank-1 patch did not
work") into a falsifiable design constraint for the next toy.

Two things did get adjudicated cleanly along the way, and both carry forward.

### Carry-forward 1 — an intervention-class ordering (the partial deliverable)

Across the phase, exactly one class moved the targets at all:

| intervention class | read | result on `phi1`/`phi2` | exp |
|---|---|---|---|
| same-read/same-write rank-1 | readout direction | no movement | 29 |
| fixed-read oblique rank-1 | single global read | **non-adjudicable** — read did not transport off discovery positions | 30 |
| fixed-read oblique rank-1 | position-conditioned in-place read | clean negative — no stable control despite read room | 33 |
| matched near-manifold activation deltas | (interchange-style; no covector) | **moves each target** over no-information floors, with held-out transfer | 34, 35 |

So for these targets, **manifold/interchange-style interventions outrank rank-1
linear oblique writes** (under any read tested). The next phase should carry
matched-delta / interchange interventions forward as the validated-as-promising
primitive, and should **not** treat rank-1 residual oblique writes as a clean
predicate-control handle. The matched-delta win is a movement result only —
specificity was never adjudicable (see headline) — so it is "promising primitive,"
not "works."

### Carry-forward 2 — the read is position-specific (a method finding)

Exp 30's negative was first read as a write failure; exps 31–32 retyped it as a
**read-transport** failure. Both predicates *are* linearly readable in place at
held-out positions at `R²` comparable to discovery, but the read direction is
**genuinely position-specific** (exp 32 froze the discovered direction and
refit only gain/bias: held `R²` ≤ ~0.36 against an in-place ceiling ~0.55–0.75,
`recovers=False`), not one global direction needing recalibration. The
cosine-sharing instrument was itself unreliable (ceiling below 0.50), so a milder
shared subspace is **not excluded** — the claim is "no single-position read
carries," not "the reads are orthogonal." Consequence for any successor: fix a
**position-conditioned** read; the exp-29 "single global readout" was the wrong
object, and re-using it re-fights the transport wall rather than testing write
freedom.

### The `phi1`/`phi2` coupling — a finding to investigate, valence open

The load-bearing new measurement (exp 35) is that the two targets are **causally
coupled**: a directed near-manifold delta built to move one co-moves the other
toward source, beyond no-information floors (4/4 seeds, both arms; coupling
margins 0.27–0.53 over the mismatched/shuffled floors). Read geometry corroborates
a bundle structure the write does not respect (`cos(phi1,phi2)` 0.54–0.65 vs
`cos(phi*,phi4)` mostly 0.10–0.27).

**The valence of this is genuinely open**, and the phase should hand it forward as
a question, not a settled claim:

- *Benign reading* — correct identification of one variable. The model carries
  stack state as a single object that both predicates read off; "non-specificity"
  (exp 34) was bundle-internal coupling, not broad replacement. Under this reading
  the intervention class is fine and the **target decomposition** was wrong.
- *Adverse reading* — broad state replacement. The move drags `phi4` above the
  sparing ceiling (`|c(phi4)|` 0.36–0.62 > 0.35), so exp 35's registered verdict
  was `BROAD_STATE_REPLACEMENT`.

Exp 35's own conclusion is that the data **cannot separate these two**: the
discriminating `phi4` axis is underpowered (room 0.025–0.037, ~6–8× smaller than
the bundle, in the marginal band 4/4 seeds) **and** possibly biased (`phi4` is a
binding predicate plausibly itself partly stack-coupled, so a clean stack write
*should* move it). Net robust claim: the bundle is coupled and the move is **not
demonstrably bundle-specific** — "broad" is the registered branch, not an
established mechanism. The coupling itself is the interesting object; understanding
whether `pstack` represents `phi1`/`phi2` as one stack-state variable (and what
that says about how the model factors stack state) is a **forward research
question**, deliberately not closed here.

## Per-experiment ledger (29–35)

| # | what it decided |
|---|---|
| 29 | Two predicates decode and pass exact endpoint audit, but rank-1 same-read/same-write patches do not move them; one predicate flat, one not recovered by tested interpreters. Readout ≠ control. |
| 30 | I1: with the exp-29 single global read fixed, oblique writes cannot be adjudicated — the read fails held-out-position `R²`. `FIXED_READ_LIMIT`. Stalled before the write question. |
| 31 | Diagnostic: both predicates *are* readable in place at held-out positions, but the read direction is position-specific (near-orthogonal covectors). Exp 30's negative is transport-of-read, not representational absence. |
| 32 | Gate: confirmed `POSITION_SPECIFIC` (freeze direction, refit gain/bias → `recovers=False`, held `R²` ≤ ~0.36 vs ceiling ~0.55–0.75); cosine-sharing instrument `UNRELIABLE`; a milder shared subspace not excluded. |
| 33 | I1′: with the read repaired to position-conditioned in-place fits, the registered fixed-read rank-1 oblique write menu does not stably control either predicate despite read room. `NO_POSCOND_READ_WRITE_WORKS` — clean negative, the I1 write question finally answered. |
| 34 | I3-lite: matched same-position, same-sign, target-matched activation deltas *move* each predicate (control 0.63–0.78, over mismatched/shuffled floors, with transfer; `own_delta` ceiling closes to 1.00) but co-move the others. `NONSPECIFIC_DELTA`. Near-manifold move exists but is entangled. |
| 35 | 2b: re-adjudicate exp-34 moves as joint-bundle vs out-of-bundle. Bundle is directionally coupled (beyond floors, 4/4), move is partial (m-gram ~0.50), but drags `phi4` above the sparing ceiling → `BROAD_STATE_REPLACEMENT`. Robust claim narrower: `phi4` is underpowered and possibly biased, so the move is only *not demonstrably bundle-specific*; `pstack` cannot tell broad from joint. |

The shape of the arc is a **regress of prerequisites**: each step peeled back a
condition the previous step had assumed settled (read transports → read is
position-specific → fixed-read write still fails → matched deltas move but are not
specific → specificity itself is un-adjudicable). That is the falsifiability
standard working as designed — every step returned a result that reshaped the
next — but it also means the original exp-29 write question was never given a
clean yes/no, because the binding constraint was upstream of it at every step.

## What did not run, and why it should not (on `pstack`)

The nominal ladder reached I3 (matched deltas) but **not** I2 (full read/write
discriminator) or I4 (patch-point localization). These were not skipped for
budget; both are **blocked upstream by the same substrate defect**:

- I2's routing turns on specificity (target movement vs non-target movement);
  with no high-room, known-out-of-bundle control, specificity is un-adjudicable on
  `pstack`, so I2 would inherit exp 35's un-decidability.
- I4 is "defensible only if it can be scored against a better control than `phi4`"
  (exp 35) — which `pstack` does not have.

So running I2/I4 here would re-spend on a substrate that cannot answer the
question. This trips two of the phase's own stop rules: *target non-diagnostic for
the specificity question* and *tempted to keep mining the same substrate without a
new diagnostic → consolidate and move*.

## Promotions to `BATTERY.md`

Phase 3 produced **no new battery member** (it studied interventions, not
diagnostics). What is promoted is a small set of durable typed findings into the
failure-modes map, each a pointer to the canonical writeup:

- **position-specific predicate read (exps 31–32)** — a readable target need not
  have a transportable read; freeze-and-refit (32) is the instrument that
  separates a genuinely different direction from gain/bias drift; cosine-sharing
  is unreliable below its `1/√d`-relative ceiling.
- **rank-1 residual oblique write is not a clean predicate-control primitive
  (exps 29/30/33)** — same-read and position-conditioned fixed-read rank-1 oblique
  writes fail to control coupled stack-state predicates despite read room; record
  as a scoped intervention-class negative, not a battery-member failure (compare
  the exp-20 single-write probe note).
- **specificity needs a separable high-room control (exps 34–35)** — on `pstack`
  the only out-of-bundle predicate is underpowered and possibly semi-coupled, so
  matched-delta movement cannot be typed as specific vs broad. A specificity
  metric requires an out-of-bundle control with real room, scored in absolute
  marginal terms, not the room-normalised closure fraction.

## `ASSUMPTIONS.md` ledger settling

- **"linear rank-1..k patches are an adequate intervention class"** moves from a
  generic **scoped** to **scoped — falsified for `pstack` predicate targets**:
  rank-1 residual oblique writes (same-read and position-conditioned fixed-read)
  do not control `phi1`/`phi2` despite read room (exps 29/30/33); matched
  near-manifold deltas move them but specificity is un-adjudicable on `pstack`
  (exps 34/35). Manifold/interchange interventions outrank rank-1 linear writes
  for these targets.
- **"position-locality of learned reads"** gains a `pstack` annotation: the
  predicate *read* is genuinely position-specific on `pstack` (exps 31–32), an
  in-place readability + non-transport pattern, distinct from the κ-graded Mess3
  learned-read entanglement.
- **New open row — `phi1`/`phi2` coupling.** Add a standing row: *on `pstack`,
  `phi1_next_closes` and `phi2_net_return` are causally coupled (a directed delta
  co-moves both beyond floors); whether they are facets of one stack-state
  variable or a broad-replacement artifact is **open — valence uncertain**, and is
  a backward-design input for the next toy.* exps 34–35.

## Decision — program disposition (the exit gate)

Phase 3 closes. The intervention-class benchmark did **not** yield a positive,
specific, transferable intervention class on `pstack`; its deliverable is the
typed substrate verdict plus carry-forwards 1–2 above. Per the phase's own exit
gate and stop rules, the decision is:

> **Go to the exit gate — a backward-designed richer toy — rather than another
> residual-level probe on `pstack`.** Do not run I2 or I4 on `pstack`; both
> inherit the un-decidable specificity. The next toy must be designed with a
> **separable, high-room, known-out-of-bundle control predicate**, scored in
> **absolute marginal terms** (not the room-normalised fraction that sank `phi4`),
> so that "specific" is measurable. Carry forward: (a) matched-delta /
> interchange interventions as the promising primitive over rank-1 linear writes;
> (b) position-conditioned reads; (c) the `phi1`/`phi2` coupling as a question to
> understand, with `pstack` itself as a reference case.

This keeps the roadmap order: the "interventional upgrade" step (`RESEARCH_PROGRAM.md`
roadmap item 1) was attempted on `pstack` and returned a substrate verdict that
defines the toy the next step needs; it does not re-order the program.

## Scope

This consolidation makes no new measurement. Every claim carries its experiment's
indices: `pstack-L4`, L1 patch point, `m=3` horizon, the registered
discovery→held-out positions, the `{phi1,phi2}` bundle and single `phi4`
out-of-bundle control, and the registered seeds. The negatives do not prove no
intervention controls these predicates, nor that the bundle is unwritable at
another patch point; they type the residual-level rank-1 result and the substrate
limit. No real-LLM claim. The `phi1`/`phi2` coupling is a flagged finding, not an
established mechanism.

---

## Result

Consolidated. Phase 3 (intervention-class benchmark) closes with a typed
**substrate-adequacy** result: no intervention class earned a specific,
transferable positive on `pstack` predicates, because the toy cannot decide
specificity for its coupled stack-state targets. Carried forward: matched-delta /
interchange interventions outrank rank-1 linear oblique writes (movement only);
predicate reads are position-specific; and the `phi1`/`phi2` coupling is an open
question routed, with the rest of the phase, to a backward-designed richer toy at
the exit gate.
