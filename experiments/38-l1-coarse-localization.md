# Experiment 38 — L1 coarse localization — SKETCH

**Status: sketch (blocked on exp 37).** Not a design draft and not pre-registered:
a forward placeholder so L0's choices can be checked against what L1 needs.
Promote to a design draft only after exp 37 returns `GO`; thresholds and counts
below are deliberately unfilled until then.

## Entry conditions (from exp 37)

- exp 37 returns **`GO`** (both facets pass room + dissociability; harness
  self-tests pass).
- The validated harness, the target-conditioned PairSet builder, and the
  block-level unit enumerator exist and are self-tested.
- The facet constructs and the residual-full reference are fixed by exp 37.

## What L1 measures (if GO)

For each facet `f ∈ {depth, top_type}`, an **importance map** over the
architecture-given block units (attention vs MLP output, per layer × position):
the per-unit interchange-patch effect on `f`, with the specificity (total-effect)
reading alongside it, against the exp-37 floor and residual-full ceiling.

Headline claim: **do depth and top_type localize to the same block units or
different ones** — the separability question posed mechanistically. Secondary:
whether either facet is localized (a few units carry it) or spread.

## How each exp-37 NO-GO reroutes or blocks L1

| exp-37 verdict | effect on L1 |
|---|---|
| `GO` | L1 proceeds as above |
| `NO_ROOM(f)` | L1 blocked for `f`; redesign that facet's observable before any localization |
| `NOT_DISSOCIABLE(f)` | L1 blocked; adjust registered positions/lengths to get held-fixed pairs, re-run exp 37 |
| `OBS_EXACT_DRIFT(f)` | L1 blocked for `f`; redefine the estimator (the map would be uninterpretable) |
| `TARGET_VACUOUS(f)` / `SMALL_SOURCE_DELTA(f)` | L1 blocked for `f`; the facet has nothing to localize — change target |
| `HARNESS_FAIL` | L1 blocked entirely; fix the harness/method first |

## Definitions L1 inherits from exp 37 (not re-derived)

- the per-facet **label / estimator / oracle-audit** split and the depth
  horizon-limit and empty-stack `top_type` handling;
- the **residual-full reference** as the ceiling and block outputs as its
  sub-units;
- target-conditioned **pairing** (single-facet clean/source pairs);
- scoring: **target effect + total effect + floor/ceiling controls**, seed
  majority, held-out positions;
- the **observable-only** honesty boundary.

## Placeholders (fill after exp 37)

- importance threshold separating a unit from the floor: `<TBD from exp-37 floor>`
- same-vs-different-parts margin: `<TBD>`
- pairs per cell actually available at the registered positions: `<TBD from exp-37>`
- per-facet room and floor measured in exp 37: `<TBD>`
- unit set (which layers × positions, attn/MLP) entering the map: `<TBD>`

## Verdict shape (to be completed at promotion)

Expected branches: `SEPARABLE_PARTS` (different units carry the facets),
`SHARED_PARTS` (same units — the coupling, mechanistically), `DISTRIBUTED(f)`
(no small unit set carries `f`), `NONSPECIFIC(f)` (target effect without
specificity). Exhaustive partition + precedence to be registered with the design
draft.
