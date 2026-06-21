# Program goals and phase-planning constraints

Purpose: the longer-term aim and the standing constraints that any phase should
respect. It is deliberately **not** a phase plan and **not** organized around any
one phenomenon. Read it when proposing or judging a *phase* (not a single
experiment): a phase that does not serve the aim below, or violates a commitment
below, is out of scope regardless of how interesting it is in isolation.

This sits under `RESEARCH_PROGRAM.md` and refines its "where this is going" for
the work after the intervention-class benchmark (Phase 3). The standing
intellectual frame, the honesty constraints, and the measured-missability endgame
still live there; this document says what changed in our understanding of *what
has to be built next*, and how to plan around it.

## Why the turn

The toy phases produced many typed findings but few connected ones — points, not
lines. Two causes, both structural:

- The toys were never **understood as mechanisms**. We probed them and scored
  patches against completions, but never characterized how a given model
  represents, maintains, and combines the quantities that drive its completions.
  So each intervention result is an isolated fact with no skeleton to hang on.
- Intervention questions are **secretly structural questions**. "Can I move this
  internal quantity specifically" is a question about how the computation is
  organized, and we were answering it with behavioral search over a narrow class
  of edits. The intervention-class benchmark ended on exactly this wall: a
  coupling it could observe but not explain, on a substrate that could not decide
  the question.

The work ahead is to build the missing layer: a structural picture of the parts
and the relations among them, in toys where ground truth still exists, in service
of the same endgame — what is findable about completion-relevant structure with
no oracle, with error bars.

## The aim (and the endpoint we will not pin)

Understand how a model's internal parts carry the quantities that matter for its
completions: how those quantities are **stored, maintained across the sequence,
combined, and protected**, as relations that live in the parts rather than as a
list of isolated readouts.

The *final* form of that understanding — geometry, circuits, or some
relational-abstraction language we do not yet have — is real but **deliberately
left open**. We have no informed guess, and positing one now would repeat the
mistake of positing a privileged decomposition. The territory below is approached
empirically; the endpoint is discovered, not declared.

## Durable commitments (every phase must respect these)

These were settled deliberately and are timeline-invariant — they hold whatever
the next phase is.

1. **No privileged decomposition.** There is no "true" set of the model's
   features to recover. Units of analysis must be defined by an external referent
   (the architecture, the behavior, or the process), never asserted as the
   model's intrinsic basis.

2. **Architecture-given units are the scaffold, not the endpoint.** Because of
   (1), the only non-arbitrary place to start is what the architecture hands us —
   heads, MLP/attention block outputs, the residual stream at a (layer,
   position). These are a coarse scaffold; the finer relational structure is the
   open aim, not these units themselves. Refining *below* an architectural unit is
   where the hard phenomena — and the open endpoint — live.

3. **Honesty is not relaxed.** Discovery and proposal families remain supervised
   on observables only. Privileged ground truth — beliefs, exact process state,
   any mechanistic label — is evaluation-only, exactly as in every prior phase.

4. **Representation and computation are one object.** How a quantity is stored and
   how it is updated do not cleanly separate; do not plan a phase that treats the
   static representation as if it could be understood apart from the dynamics that
   maintain it.

5. **Leashed and abandonable.** This work is pursued because a structural picture
   is a prerequisite for the intervention and completion questions — not as an end
   in itself. A phase earns its place by what it returns to that thread; the work
   may be paused, resumed, or abandoned on what it uncovers. It does not replace
   the program.

## Aspirations (not commitments)

Nice to have, not binding. A phase is not out of scope for missing these, but
prefer them where they come for free:

- **Cheap by construction.** Recent phases became expensive (multi-seed
  intervention sweeps with dose curves). Frozen-checkpoint probes and the
  cheapest model that still exhibits the phenomenon are preferable; cost is mostly
  experiment design, not model size.
- **Lines over dots.** Consolidating existing typed findings into structure, or
  producing a relation/curve across a varied family of toys, is worth more than
  one more isolated point. Several threads already carry partial findings waiting
  to be unified (the read/write asymmetry, state-vs-summary, off-manifold
  behavior).

## The territory (a map, not a plan)

Of any completion-relevant quantity, four structural questions can be asked. They
are the territory the work explores; **no order is implied, and no single one is
the goal.** Naming them here constrains phase proposals to this map; it does not
schedule them.

- **Packing — how it is stored.** Distribution vs localization across parts; how
  linearly readable the storage is; whether many quantities share few directions.
- **Dynamics — how it is maintained.** Whether the stream carries propagated state
  or a per-position summary; how information moves across positions; whether
  abstractions stay coherent under generation.
- **Composition — how pieces are combined.** Binding of what-to-where; the
  read/write asymmetry already observed; whether jointly-moving quantities are one
  variable or several.
- **Robustness — what survives perturbation.** Redundant or backup computation;
  behavior on- vs off-manifold; why some edits are routed around.

Any single phenomenon — superposition, binding, propagation — is **one cell of
this map, not the program.** A phase organized around a single phenomenon as if it
were the whole aim is mis-scoped.

## What makes a good phase here

A phase proposal should be checkable against this document:

- it serves the aim and respects all five commitments;
- it names the cell(s) of the territory it touches, and why that cell now;
- it is leashed — its result changes something about the intervention/completion
  thread, or it should not run;
- it carries the open verification tension explicitly (below);
- preferably cheap, and preferably a line rather than another isolated point.

## Open tensions (carried, not resolved)

- **The fine endpoint is uninformed.** We cannot yet say what the relational
  structure *is*; phases discover it, and early phases may only sharpen the
  question.
- **Generalization-verification is unsolved.** The existing harness verifies an
  experiment's *result*; whether a structural finding *generalizes* is the
  program's standing open problem, not something to assume away. Keep the two
  scopes of "verified" distinct in every claim.
- **Distributed/superposed structure may resist clean handles.** That is expected;
  a typed negative there is a result, not a failure. Meet that difficulty early
  rather than design toys to avoid it.

## Maintenance

This is a frame document, not a status log. Update it only when the *aim or the
commitments* move — not per phase, and not to record what a phase found (that
belongs in the phase writeup, `EXPERIMENTS.md`, and the `ASSUMPTIONS.md` ledger).
Apply the same no-status-in-frame-docs discipline here as to `RESEARCH_PROGRAM.md`,
`SYNTHESIS.md`, and `BATTERY.md`.
