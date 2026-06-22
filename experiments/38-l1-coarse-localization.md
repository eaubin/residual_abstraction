# Experiment 38 — Propagating-state instrument + "does the model propagate graded depth?" gate — DESIGN DRAFT

**Status: design draft (supersedes the prior m=1 block-localization sketch).** Not
yet pre-registered: the question, instrument, and discriminator are fixed below;
thresholds, positions, and counts are deferred to the pre-registration (marked
`<TBD-prereg>`). State-localization phase, rung after L0 (exp 37). This rung is an
**instrument build + gate**, like L0 — it validates a tool and decides *whether
there is graded state to localize* before any localization claim.

## Why this, not block localization (what exp 37 changed)

L0 established that the residual-at-`t` interchange patch transports only the m=1
(next-token) prediction, so "depth" collapsed to 3-level **close-readiness**
(empty / interior / full) and the phase question quietly slid from *where does
graded state live* to *where does the next-token summary live* — the exp-4/5
summary-not-state wall, recurring (its **second** independent sighting).

The phase doc deferred graded state to a *conditional* dynamics rung ("if L1 shows
state carried across positions"). That trigger is **unsatisfiable**: an m=1
instrument is exactly the one that cannot see propagation, so the deferral can
never fire. The decision can't be delayed to "let L1 decide" — it is made here:
build an instrument that can see graded depth and gate the prior question directly.

## The question (a typed fork, well-posed in both directions)

```text
Does this Dyck-2 model PROPAGATE graded stack depth in its residual stream —
carried at a position and used downstream — or does it RECOMPUTE depth from the
token prefix at each step, holding only a per-position next-token summary?
```

- **A — PROPAGATED & localizable:** a localized residual patch carries source
  graded depth through a rollout → graded state exists to localize; proceed to the
  same-vs-different-parts localization with the validated instrument.
- **B — RECOMPUTED:** no localized patch moves graded depth → the summary-not-state
  wall is a **measured model fact**, not an instrument limit. A clean B is a real
  finding (the exp-4/5 wall pinned with the tool that could have refuted it) and
  reframes "state localization" toward summary features + the recompute finding.

## Instrument: rollout from the patched state

Fixed-continuation scoring re-anchors — feeding a clean continuation token at `t+1`
makes the model recompute from that observed token, which forces B by construction
and is why L0 saw only m=1. Instead:

1. Interchange-patch the residual (clean ← source) at the L0 patch layer.
2. **Roll out** the continuation autoregressively from the patched state (the
   model's generated tokens feed back), so the patched residual propagates forward
   through attention instead of being overwritten by observed clean tokens.
3. Read a **graded** statistic of the rollout (below), not the m=1 scalar.

## The discriminator (the centerpiece — the part that must be right)

On a Dyck prefix the tokens *determine* the stack state, so propagation vs
recomputation is undecidable on a clean prefix. Interchange creates the needed
disagreement — **residual says source-depth, tokens say clean-depth** — and which
one drives behavior is the verdict:

- **Single-position patch at `t`** (patch only `t`'s residual; leave `0..t-1`
  clean). Rollout graded depth follows **source** ⇒ `t` carries a propagated graded
  summary (**A**); follows **clean** (downstream re-reads clean `0..t-1`) ⇒ depth is
  recomputed from the distributed token history, not summarized at `t` (**B**). This
  is the exp-4/5 per-position-summary-vs-propagation question, operationalized.

Graded observable (depth-1 vs depth-2, invisible to m=1):

- **`P(close at step 2 | close at step 1)`** over the rollout. At depth 2, after one
  close the stack is at depth 1 and can close again; at depth 1, after one close it
  is empty and must open. The conditional separates the two depths. (A graded
  depth-trajectory statistic over the rollout is an alternative; pick one at prereg.)

Controls (mirror L0's calibration/floor/ceiling):

- **Ceiling — full-prefix patch:** running source from the patch layer up; the
  rollout must reach source graded depth, else the rollout procedure is broken
  (a harness fail, not a model fact).
- **Floor — same-depth source:** no residual/token disagreement ⇒ no move
  (≈ clean). Plus a random/mismatched-position floor.
- **Oracle target:** the Dyck oracle gives the true source graded-depth value — the
  endpoint audit and the A-direction target (endpoints only, never selection).

## Verdict shape (exhaustive; to finalize at prereg)

```text
HARNESS_FAIL          — ceiling (full-prefix) rollout fails to reach source graded depth, or a self-test fails; blocks all
OBS_DRIFT             — rollout graded observable vs oracle endpoint gap too large (uninterpretable)
RECOMPUTED            — single-position graded transport at/below the floor (graded depth not summarized at t) — verdict B
PROPAGATED            — single-position graded transport clears the floor toward source (graded depth carried at t) — verdict A, routes to localization
PARTIAL/SEED_UNSTABLE — between floor and ceiling without a stable majority; underpowered, add rollouts/seeds
```

## What carries from L0 (not re-derived)

- the Dyck parser labels, the facet split (selection label / observable estimator /
  oracle audit), the certified `top_type` substrate and the `depth`→close-readiness
  scope, the dissociable-pair abundance (cells 510–512), the checkpoint + validity
  gate + bit-exact model guards;
- the m=1 close-readiness observable becomes the **step-wise** scorer the rollout
  reads at each generated step;
- the random-unit floor obligation L0 deferred is built here as the propagation
  floor.

## Self-tests (known-answer, before any model claim)

- planted propagation: a constructed/teacher-forced trajectory where source depth
  *is* carried must be recovered at the ceiling;
- no-difference (same-depth source) rollout scores ≈ 0;
- a full-prefix patch reaches the source graded target;
- a rollout of the unpatched run reproduces clean graded depth (sanity).

## Code consolidation (this rung)

- `scripts/localization/l0_substrate_gate.py` is a **frozen** record — not touched.
- New **`localize.py`** earns its existence here: extract the durable facet/Dyck
  core from L0 (`stack_labels`, `facet_from_q1`/`facet_observable`, `facet_pairs`,
  `floor_pairs`, `make_Xc`/`q_at`/`exact_joint`), then add the **rollout patch +
  graded scorer + propagation controls** as the new reusable core, self-tested.
  L0 keeps its inline copy (the accepted exp-15 duplication). Verdict helpers
  reused from `battery.py`.
- The block/head **unit enumerator is still deferred** — built only if this rung
  returns `PROPAGATED` (no point enumerating units for a localization not shown
  possible). "Build the seam when earned."

## Open design points (fill at pre-registration)

- patch layer (L0's L1 residual vs a sweep), registered positions, rollout length,
  number of rollouts / sampling (greedy vs sampled), graded-observable choice and
  its floor/ceiling thresholds, seed set, source-pair construction (depth-2 source
  vs depth-1 clean, matched on `top_type`).

## Non-goals

- No same-vs-different-parts localization claim here (that is the next rung, gated
  on `PROPAGATED`). No granularity sweep, no head/direction enumerator. No real-LLM
  claim. The instrument is minimal: the smallest rollout patch that decides A vs B
  at the registered positions, not a general dynamics framework.
