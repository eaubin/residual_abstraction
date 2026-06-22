# Experiment 38 — Propagating-state instrument + "does the model propagate graded depth?" gate — DESIGN DRAFT

**Status: design draft (v2 — exact teacher-forced probe; supersedes both the m=1
block sketch and the v1 rollout draft).** Not yet pre-registered: the question,
instrument, and discriminator are fixed below; thresholds, positions, and counts
are deferred to the pre-registration (`<TBD-prereg>`). State-localization phase,
rung after L0 (exp 37). This rung is an **instrument build + gate**, like L0 — it
validates a tool and decides *whether there is graded state to localize* before any
localization claim.

## Why this, not block localization (what exp 37 changed)

L0 established that reading the **m=1 marginal** of the completion gives only 3-level
**close-readiness** (empty / interior / full), so "depth" collapsed and the phase
question slid from *where does graded state live* to *where does the next-token
summary live* — the exp-4/5 summary-not-state wall, **second** sighting. The phase
doc's conditional dynamics rung ("if L1 shows propagation") is **unsatisfiable** by
an m=1 instrument, so the decision is made here, not deferred.

## The question (a typed fork, well-posed in both directions)

```text
Does this Dyck-2 model carry graded stack depth as a LOCALIZED residual summary
that downstream computation uses — or is graded depth only RECOVERABLE from the
distributed token history (no single locus summarizes it)?
```

- **A — PROPAGATED & localizable:** a localized residual patch transports source
  graded depth → graded state is summarized at a place; proceed to same-vs-different
  parts localization with the validated instrument.
- **B — RECOMPUTED / distributed:** no localized patch moves graded depth → the
  summary-not-state wall is a **measured model fact** for localization purposes. A
  clean B reframes the phase toward summary features + the recompute finding.

## Instrument: exact teacher-forced multi-step conditionals (no rollout)

The graded signal m=1 cannot see is the **conditional after a forced close**.
Interchange-patch source-depth onto clean tokens, feed a *close* token at `t+1`,
read the prediction at `t+2`: the forced close advances the stack by one from
*whatever pre-close depth the model believes*, so the post-close close-readiness
**reveals** that believed depth (source vs clean). This needs **no free rollout** —
it is the exact m≥2 teacher-forced conditional, one forward pass, deterministic.

It reuses L0's existing machinery almost wholesale: `make_Xc`/`q_at`/`exact_joint`
already build the full `V^m` completion joint; L0 only read its m=1 marginal. Here
we read the **m=2 / m=3 conditionals** of the same joint. The only new patch
mechanics are single-position and windowed `prefix_state` slices (L0 had full-prefix
and no-op).

Graded observable — **close-readiness after `k` forced closes**, `k = 0,1,2` (within
the standing m=3): `k=0` is L0's signal; `k=1,2` are the graded extensions that
separate depth 1 / 2 / 3 (depth `d` can absorb `d` closes before the stack empties).

## The discriminator (the centerpiece — two curves, not one contrast)

On a Dyck prefix the tokens *determine* the stack, so propagation vs recomputation
is undecidable on a clean prefix. Interchange creates the disagreement — **residual
says source-depth, tokens say clean-depth** — and the *shape* of how transport
depends on what we patch is the verdict:

- **Sufficiency / locality curve.** Patch a growing prefix window ending at `t`
  (just `t`; `t-1..t`; …; full prefix); trace graded transport vs window size.
  **Propagated-and-summarized** ⇒ saturates *early* (small window suffices).
  **Recomputed-from-distributed-history** ⇒ ramps late, needs nearly the whole
  prefix. (The full-prefix patch is the curve's endpoint, not a standalone
  tautological ceiling.)
- **Necessity.** Patch full-prefix *minus* `t`. If `t` carries the summary, removing
  it blocks transport. Sufficiency + necessity = the mediation logic; far stronger
  than either alone.
- **Depth-over-horizon curve.** Transport at `k = 0,1,2` forced closes — *how many*
  graded levels propagate, a shape in its own right.

Normalization & references (same logic as L0's floor): transport is expressed as a
fraction of the **oracle-calibrated source−clean gap**. Floor = same-depth source
(no residual/token disagreement → no move) + random/mismatched-position. Ceiling =
full-prefix patch on the same probe (calibration vs oracle endpoints, `OBS_DRIFT`).

## What a negative excludes (claim bound — required honesty)

`RECOMPUTED` means **"graded depth is not localizably summarized at a position,"**
consistent with *both* true per-position recomputation *and* **redundant distributed
carrying** (the named redundancy confound). It does **not** license "the model does
not propagate." The sufficiency curve partly separates these — if some window
restores transport, the state is carriable; necessity says whether a locus is
required — but redundancy is not fully excludable here, so the claim is bounded to
localizability, not existence.

## Verdict shape (exhaustive; to finalize at prereg)

```text
HARNESS_FAIL  — full-prefix (ceiling) conditional fails to reach source graded depth, or a self-test fails; blocks all
OBS_DRIFT     — conditional-vs-oracle endpoint gap too large (uninterpretable)
PROPAGATED    — locality curve saturates early AND t is necessary: graded depth localizably summarized (verdict A) -> localization rung
DISTRIBUTED   — transport only at near-full window, no single locus necessary: carriable but not localized (a typed middle)
RECOMPUTED    — graded transport at/below floor even at full window minus drift: not localizably summarized (verdict B, with the redundancy bound above)
SEED_UNSTABLE — no stable majority across seeds; underpowered
```

## What carries from L0 (not re-derived)

Dyck parser labels; the facet split and the certified `top_type` substrate; the
`depth`→close-readiness scope; dissociable-pair abundance (cells 510–512, so
matched depth-1/depth-2 source/clean pairs exist); checkpoint + validity gate +
bit-exact model guards; the m=1 close-readiness scorer becomes the `k=0` rung of the
horizon curve; the random-unit floor obligation L0 deferred is built here.

## Self-tests (known-answer, before any model claim)

- planted summary: a synthetic where depth *is* carried at a known position must be
  recovered (sufficiency saturates at that position, necessity flags it);
- no-difference (same-depth source) → ≈ 0 at every window;
- full-prefix patch reaches the source graded target (ceiling sanity);
- unpatched run reproduces clean graded depth;
- single-position / windowed patches are bit-exact reconstructions of the intended
  residual splice (no off-by-one in the prefix slice).

## Code consolidation (this rung)

- `scripts/localization/l0_substrate_gate.py` is a **frozen** record — not touched.
- New **`localize.py`** earns its existence: extract the durable facet/Dyck core
  from L0 (`stack_labels`, `facet_from_q1`/`facet_observable`, `facet_pairs`,
  `floor_pairs`, `make_Xc`/`q_at`/`exact_joint`), then add the **windowed/
  single-position patch + the m≥2 conditional scorer + the locality/horizon curve
  reducers**, self-tested. L0 keeps its inline copy (the accepted exp-15
  duplication). Verdict helpers reused from `battery.py`. The block/head unit
  enumerator stays **deferred** until this rung returns `PROPAGATED`/`DISTRIBUTED`.

## Open design points (fill at pre-registration)

- patch layer (L0's layer vs a small sweep), registered positions and window
  ladder, which forced-close horizons (`k` up to 2), source-pair construction
  (depth-2 source vs depth-1 clean matched on `top_type`; and depth-3 if pairs
  abundant), the saturation/necessity thresholds defining the curve verdicts,
  seed set, and the oracle-calibration band.

## Non-goals

- No same-vs-different-parts localization claim here (next rung, gated on
  A/`DISTRIBUTED`). No granularity sweep, no head/direction enumerator, no real-LLM
  claim. The instrument is minimal: exact teacher-forced conditionals over patch
  windows — not a rollout/dynamics framework.
