# Experiment 37 — Dyck-2 localization substrate gate + harness — DESIGN DRAFT (pre-code)

**Script:** `scripts/localization/l0_substrate_gate.py` — **not yet written.**
**Output:** `out/exp37_dyck2.txt`.

**Status: design draft (pre-code).** Not pre-registered: under `AGENTS.md` and
`EXPERIMENT_REVIEW_PROTOCOL.md`, pre-registration requires this writeup *plus* a
runnable script with guards, verdict logic, and output tables implemented. This
becomes pre-registered only when that script exists and passes the first review
pause. L0 of the state-localization phase (`docs/STATE_LOCALIZATION.md`):
harness build + a GO/NO-GO substrate gate. No localization claim here — that is L1.

## What greenlighting this approves (read this first)

Three load-bearing choices; the rest is mechanics.

1. **Vehicle and target are gated, not assumed.** Dyck-2 + {depth, top_type}
   proceed to L1 only if this gate shows room *and* dissociable pairs. A typed
   NO-GO reroutes a specific choice — a success of the gate, not a failure.
2. **Localization will use interchange patching, validated first.** The harness
   must recover known answers on synthetics before any real use; if it does not,
   L1 does not run.
3. **Each facet is three separated objects:** a prefix-computable *selection
   label* (builds pairs), an *observable estimator* (scores room/effect), and an
   *exact oracle audit* (endpoints only). The gate validates the estimator against
   the oracle; if they disagree, the target is redefined before localizing.

## Target construct (label / estimator / audit, per facet)

| facet | selection label (prefix-computable) | observable estimator (on completions) | empty-stack / horizon handling |
|---|---|---|---|
| `depth` | raw stack depth `d` at the scored position | a horizon-`m` statistic monotone in depth (e.g. expected net closes within `m`, or P(stack reaches a shallower level within `m`)) | only depth up to the horizon is behaviorally observable; pairs and bins use **horizon-relevant depth** `min(d, m)`; deeper differences are not claimed |
| `top_type` | type of the top-of-stack bracket (the valid next closer) | completion mass on each closer type at the scored position | `top_type = ∅` at empty stack; empty-stack positions are **excluded** from type cells and flagged |

Exact audit (both facets): the Dyck oracle gives the exact completion
distribution, hence the exact estimator value, used for the endpoint gap only —
never for selection or scoring.

## Two parts

**A — harness self-tests (non-claim).** Build the stage pipeline
(`load → target-conditioned PairSet → unit enumerator(g) → interchange patch →
score → aggregate`) and pass known-answer checks: planted-unit recovery; null pair
(no facet difference) scores `≤ NULL_TOL`; full-reference patch reaches ceiling
(`≥ CEIL_MIN`); mismatched source (differs in the *other* facet) does not move this
facet beyond `NULL_TOL`.

**B — substrate gate (GO/NO-GO).** For each facet `f ∈ {depth, top_type}`:
non-vacuous; estimable; estimator audited vs oracle; clean/source pairs separated;
full/reference patch has room; no-info baseline computed; **dissociable** —
single-facet pairs exist in adequate count at the registered positions.

## Full / reference patch (named, to fix the ceiling)

The room ceiling is **full residual-stream interchange at L1 over the scored
positions** (clean ← source) — the exp-19 identity-patch analog, a single
well-defined "everything at this layer/positions" reference. The block/component
outputs that L1 will localize are **sub-units** of this; the ceiling never
double-patches a residual-plus-component union. L0 patches components only inside
the part-A self-tests (to validate the hooks), not in the part-B room ceiling.

## Thresholds and expected baselines (registered here, not deferred to code)

| name | value | meaning |
|---|---|---|
| `VAR_MIN` | 0.05 | min std of a facet observable over the eval distribution |
| `OE_BAND` | 0.10 | max estimator-vs-oracle endpoint gap (Dyck obs/exact ran 0.064–0.073, exps 19–21) |
| `SRC_DELTA_MIN` | 0.05 | min clean-vs-source facet separation (below it, room is ill-defined) |
| `ROOM_MIN` | 0.50 | full residual patch must close ≥ 50% of the clean→source facet gap |
| `NULL_TOL` | 0.05 | self-test null / mismatched-source ceiling |
| `CEIL_MIN` | 0.90 | full-reference patch must close ≥ 90% (else the ceiling itself is broken) |
| `MIN_PAIRS_PER_CELL` | 256 | min single-facet pairs per held-fixed cell at the registered positions |

Expected baselines the gate confirms: no-info / random-unit patch closes ≈ 0
(`≤ NULL_TOL`); the residual-full reference closes ≈ 1 (`≥ CEIL_MIN`). Threshold
*values* are gate cutoffs, not claims; they are printed and audited symmetrically
(FORMALISM §6.1 rule 8).

## Confound table — load-bearing quantities (dissociable-pair count, room)

| confound that would fake a GO | excluded by |
|---|---|
| `depth` and `top_type` are correlated in Dyck, so "hold one fixed" pairs are scarce or skewed | count *actual* held-fixed pairs per cell; `NOT_DISSOCIABLE` fires below `MIN_PAIRS_PER_CELL` |
| depth "room" inflated by horizon-unobservable depth | estimator is horizon-`m` limited; pairs differ in `min(d, m)` |
| `top_type` "room" vacuous at empty stack | empty-stack positions excluded from type cells |
| self-test passes because the planted unit is trivially easy | null *and* mismatched-source must also score `≤ NULL_TOL` |
| source delta so large the patched run is off-distribution | interchange from real source runs; `SRC_DELTA` kept within the observed range |

## Predictions

- **P1 (self-tests pass; expected).** A failure is a method/implementation bug,
  not a substrate fact.
- **P2 (room; likely).** Both facets non-vacuous and room-bearing (exps 19–22
  showed rich stack structure).
- **P3 (dissociable pairs; the real unknown).** Dyck-2 should admit depth-only and
  type-only pairs by construction, but their *count at the registered positions*
  is what this gate de-risks. Thin counts route to a position/length adjustment.

## Verdict (exhaustive, non-overlapping; precedence high → low)

```text
HARNESS_FAIL           — any part-A self-test misses its known answer (checked first; blocks all)
OBS_EXACT_DRIFT(f)     — estimator-vs-oracle gap > OE_BAND (room uninterpretable under drift)
TARGET_VACUOUS(f)      — std(f) < VAR_MIN, or the no-info baseline already predicts f
SMALL_SOURCE_DELTA(f)  — clean/source facet separation < SRC_DELTA_MIN
NO_ROOM(f)             — residual-full patch closes < ROOM_MIN of the clean→source gap
NOT_DISSOCIABLE(f)     — single-facet pairs < MIN_PAIRS_PER_CELL at registered positions
GO                     — none of the above fires for either facet
```

Simultaneous failures: every fired branch is reported; routing follows the
highest-precedence one. Each NO-GO reroutes a specific choice, not the whole phase.

## Honesty, halt, non-goals

- **Honesty:** labels, estimators, and pairing are observable; the Dyck oracle
  audits endpoints only.
- **Halt:** checkpoint config mismatch, or any self-test failing its known answer.
- **Non-goals:** no localization, separability, or intervention claim (L1+); no
  granularity sweep and no head/direction enumerator yet (L0 builds block-level
  hooks only); no real-LLM claim. The phase doc is loose and may adapt as L1 lands.
