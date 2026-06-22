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

## Feasibility precondition (smoke before pre-registration — gates the rung)

The instrument's premise — that the m≥2 conditional at `t+1`/`t+2` is sensitive to
the patched prefix — is **inherited, not measured on this model**. The patch enters
at `LAYER=1` (input to `blocks[1]`); continuation positions reach it through
`blocks[1:]` (3 of 4 blocks), but their `block-0` representation already encodes the
**clean** prefix depth, so the patched source-depth must overcome that through
attention alone. Exp 37 certified only **m=1** transport; "multi-step leaks the clean
prefix" is carried from exp-4/5, never tested here. So the **ceiling** — full-prefix
patch reaching source graded depth on the `k=1` conditional — is an empirical
unknown, not a near-tautology. **Smoke it first:** if the full-prefix ceiling cannot
move the `k≥1` conditional above the random floor, the rung is `HARNESS_FAIL` and no
curve is interpretable. The whole graded claim lives in `k≥1`, which is exactly the
contaminated readout, so this precondition gates the pre-registration.

**Smoke result (`scripts/localization/exp38_ceiling_smoke.py`, seed 700, GREEN —
ceiling viable).** On depth-1-clean / depth-2-source pairs matched on `top_type`,
the `k=1` forced-close conditional cleanly separates the depths (clean `≈0.02`,
source-oracle `≈0.60`). The full-prefix patch transports **0.88–0.98 of the oracle
gap** across positions `{8,12,16,20}`, while the **same-depth (depth-1) patch floor
sits at `≈0`** — so the move is a function of the *patched graded depth*, not a
generic full-prefix disturbance. The inherited "multi-step leaks the clean prefix"
worry does **not** hold for the full-prefix patch here. (`k=0` reproduces L0's exact
m=1 transport, `f≈1.00` — wiring sanity.) **Scope:** this de-risks only the curve's
full-prefix *endpoint*; localization — whether a *small window* suffices — is the
rung's discriminator and is **not** run pre-registration. The floor must be
**same-depth**: a same-source-pool floor (e.g. a permuted *depth-2* source) would
not vary the patched depth, so it would move `cr1` as much as the ceiling and mask
the result — which is why the registered floor is same-depth, not same-source-pool.

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
  tautological ceiling.) **Matched control (required):** at each window size also
  patch an equal-count **randomly-placed** set of source positions. Window size is
  correlated with injected source-signal mass, so the locus reading is the
  contiguous-ending-at-`t` curve *relative to* the random-placement curve — never
  the raw curve, which a recomputing model can ramp purely from added signal mass.
- **Necessity.** Patch full-prefix *minus* `t`. If `t` carries the summary, removing
  it blocks transport. **Matched control:** full-prefix *minus a random position*;
  necessity of `t` is its drop *relative to* the random-drop drop (dropping any
  position removes some mass). Sufficiency + necessity = the mediation logic; far
  stronger than either alone.
- **Depth-over-horizon curve.** Transport at `k = 0,1,2` forced closes — *how many*
  graded levels propagate, a shape in its own right.

Normalization & references (same logic as L0's floor): transport is expressed as a
fraction of the **oracle-calibrated source−clean gap**. Floor = same-depth source
(no residual/token disagreement → no move) + random/mismatched-position. Ceiling =
full-prefix patch on the same probe (calibration vs oracle endpoints, `OBS_DRIFT`).
The curve verdicts ("saturates early", "ramps late") are read only against two
**measured** reference curves, not absolute window indices: the **planted-locus**
synthetic (known single-position summary → the early-saturation ceiling shape) and
the **random-placement** curve above (the no-locus floor shape). The
saturation/necessity thresholds are *defined* by these references — the load-bearing
calibration of this rung, as the floor was at L0 — with values fixed at
pre-registration.

## Confound table — load-bearing quantity (graded transport vs patch window)

| mechanism producing the curve shape | excluded by? |
|---|---|
| genuine locus: depth summarized at/near `t`, so a small window transports (the intended signal) | this is what the locality curve is built to detect |
| injected-signal mass: a wider window simply attends in more source residual, so transport ramps with size with no locus — a recomputing model then looks "distributed" for a non-locus reason | the **equal-count random-placement** control; the locus reading is contiguous-vs-random at matched mass, never the raw curve |
| representational-inconsistency artifact: source residual over `[t-w..t]` atop clean `block-0` at `[0..t-w-1]` is off-distribution, so its readout may not be "believed depth" | partly — the oracle endpoint audit (`OBS_DRIFT`) bounds the endpoints; mid-curve inconsistency is **not fully excluded** and bounds the shape reading |
| necessity confounded by mass: dropping any single position removes some signal | the **random-drop** control; necessity of `t` is its drop relative to a random-position drop |
| instrument can't move the conditional at all (`block-0` clean contamination wins) | the full-prefix **ceiling** smoke precondition; a failed ceiling is `HARNESS_FAIL`, not `RECOMPUTED` |
| redundant distributed carrying mimics a null (the redundancy confound) | **not excludable** by interchange — bounds the negative to "not localizably summarized", see below |

## What a negative excludes (claim bound — required honesty)

`RECOMPUTED` means **"graded depth is not localizably summarized at a position,"**
consistent with *both* true per-position recomputation *and* **redundant distributed
carrying** (the named redundancy confound). It does **not** license "the model does
not propagate." The sufficiency curve partly separates these — if some window
restores transport, the state is carriable; necessity says whether a locus is
required — but redundancy is not fully excludable here, so the claim is bounded to
localizability, not existence. **The conclusion template pre-commits this wording:**
a `RECOMPUTED`/`DISTRIBUTED` result is written as "not localizably summarized" and
may not be reworded to "the model recomputes" or "carries no propagated state".

## Verdict shape (exhaustive; to finalize at prereg)

```text
HARNESS_FAIL  — full-prefix (ceiling) conditional fails to reach source graded depth, or a self-test fails; blocks all
OBS_DRIFT     — conditional-vs-oracle endpoint gap too large (uninterpretable)
PROPAGATED    — locality curve saturates early (vs random-placement) AND t necessary (vs random-drop): graded depth localizably summarized (verdict A) -> localization rung
DISTRIBUTED   — full-window transport clears the random floor by the registered margin, but no small window suffices and no single locus is necessary: carriable but not localized (a typed middle). Requires the ceiling to actually move it; if it does not, the verdict is RECOMPUTED/HARNESS_FAIL, not DISTRIBUTED
RECOMPUTED    — graded transport at/below the random floor even at full window minus drift: not localizably summarized (verdict B). Pre-committed bound: this NEVER licenses "the model does not carry/propagate graded depth" — redundant distributed carrying is not excludable by interchange
SEED_UNSTABLE — no stable majority across seeds; underpowered
```

## What carries from L0 (not re-derived)

Dyck parser labels; the facet split and the certified `top_type` substrate; the
`depth`→close-readiness scope; dissociable-pair abundance (cells 510–512, so
matched depth-1/depth-2 source/clean pairs exist); checkpoint + validity gate +
bit-exact model guards; the m=1 close-readiness scorer becomes the `k=0` rung of the
horizon curve. The **random-unit floor obligation L0 deferred is built here**: L0's
depth `FLOOR_FAIL` was conservative precisely because the random-unit baseline was
unmeasured, and that baseline is this rung's random/mismatched-position floor. Depth
transport is read against it; whether this lifts L0's "purity uncertified" flag is
stated in the conclusion, not assumed here.

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
  abundant), the saturation/necessity thresholds (anchored to the planted-locus and
  random-placement reference curves) defining the curve verdicts, the random-drop
  and random-placement control specs, seed set, and the oracle-calibration band.

## Non-goals

- No same-vs-different-parts localization claim here (next rung, gated on
  A/`DISTRIBUTED`). No granularity sweep, no head/direction enumerator, no real-LLM
  claim. The instrument is minimal: exact teacher-forced conditionals over patch
  windows — not a rollout/dynamics framework.
