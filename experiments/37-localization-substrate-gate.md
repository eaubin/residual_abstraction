# Experiment 37 — Dyck-2 localization substrate gate + harness — PRE-REGISTRATION (awaiting review)

**Script:** `scripts/localization/l0_substrate_gate.py` — written; `--selftest`
passes and the model-level guards hold on the checkpoint (integration-validated,
not the claim run).
**Output:** `out/exp37_dyck2.txt` (the registered GO/NO-GO run is post-review).

**Status: pre-registration, awaiting the review pause.** The writeup and a runnable
script with guards, verdict logic, self-tests, and output exist; the claim-bearing
4-seed run has not been made. L0 of the state-localization phase
(`docs/STATE_LOCALIZATION.md`): harness build + a GO/NO-GO substrate gate. No
localization claim here — that is L1.

**Construct changes forced while coding (flagged for review):** (a) all estimators
are **m=1** — the L1 position-`t` patch transports only the next-token prediction
(see below), so multi-step observables are not usable; (b) the depth facet is the
m=1 **close-readiness** proxy, a coarse depth signal; (c) the floor is pinned to
the facet-matched-source full patch with an explicit normalization (below);
(d) component/block self-tests are **deferred to L1** (no component enumerator is
built or needed at L0). A single-seed integration smoke returned `OK/OK` for both
facets with the depth floor borderline (~0.048 vs `NULL_TOL=0.05`); the registered
run decides.

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

**All estimators are m=1 (next-token), and that is forced.** The L1 position-`t`
interchange patch transports only the model's *next-token* prediction: positions
`> t` recompute from the clean prefix (block-0 attention), so multi-step
completion observables leak the clean state and are **not** transportable by this
patch (a live, small instance of the exp-4/5 "summary, not propagated state"
finding — verified by the model guard below). So both facets are m=1 functions of
the next-token distribution, which cleanly factor the close behavior as
**close-readiness × type-fraction**:

| facet | selection label (prefix-computable) | observable estimator — m=1 scalar | empty-stack / horizon handling |
|---|---|---|---|
| `depth` (close-readiness proxy) | raw stack depth `d` at the scored position | **`close-readiness = q(2) + q(3)`** (next token is a closer), scalar in `[0, 1]` | a **coarse, horizon-1** depth signal: separates depth-0 (cannot close → 0) / interior / full-depth (must close → 1); it does not resolve interior depth 1 vs 2. Honest name is close-readiness; "depth" is the label it proxies |
| `top_type` | type of the top-of-stack bracket (the valid next closer) | **`type-fraction = q(2) / (q(2) + q(3))`** (type-0 share of close mass), **depth-invariant**; defined only where close mass `≥ CLOSE_MASS_MIN` (denominator guard, else row excluded). Scalar in `[0, 1]` | `top_type = ∅` at empty stack; empty-stack positions **excluded** from type cells |

**Selection labels are computed from the observed token prefix by the Dyck
parser** — never from model internals or from oracle completion labels — so
pairing stays observable. The estimators are read from the model's m=1 completion
marginal `q`.

**Scoring scalar and closure (both facets).** For a single-facet pair, the
clean→source gap is `g = obs_src − obs_un`; a patch's facet closure is
`c = (|obs_un − obs_src| − |obs_patch − obs_src|) / |obs_un − obs_src|` (1 =
reaches source, 0 = no move). `SRC_DELTA`, `ROOM_MIN`, and the floors below are all
defined on this one scalar per facet.

**Exact audit (both facets):** the Dyck oracle gives the exact completion
distribution, hence the exact `obs` value, used for the endpoint gap only — never
for selection or scoring.

## Scope indices

| index | value |
|---|---|
| checkpoint | exp-19 Dyck-2 config: `seq_len 32`, `burn_in 4`, `m=3`, `V=4`, the registered `dyck_baseline.py` training command; no retraining (regenerate from the command) |
| patch point | residual stream **L1** (battery patch point), position-`t` prefix interchange. No component/block hooks at L0 (the enumerator is L1's) |
| horizon | standing **`m=3`** (no `mm` sweep in L0; the staircase was Block 3) |
| positions | L0 audits substrate availability at interior positions `{8, 12, 16, 20}` (after `burn_in=4`, within `seq_len=32`), `{12, 20}` included — L0 makes no transfer claim, so there is no held-out reservation here. L1's discovery/held split (with `{12, 20}` as held-out) draws **fresh pairs under fresh seeds**, so L0 sampling does not contaminate it |
| seeds | `700..703` (4 fresh, relative to exps 19–22) |
| pairs | target ≈ 512 single-facet pairs per held-fixed cell per seed; gate floor `MIN_PAIRS_PER_CELL = 256` |
| oracle use | endpoint/estimator audit only; never selection or scoring |

## Two parts

**A — harness self-tests + model guards (non-claim).** Pipeline:
`load → facet-conditioned pairs → m=1 interchange patch → score → aggregate`.
Pure-function `--selftest` (no checkpoint): parser labels, the two m=1
observables, closure, the facet-pairing invariants, verdict precedence, and
majority — known answers. Model-level guards (run in the main path, the AGENTS
bit-for-bit discipline): a no-op (own-residual) patch reproduces unpatched
bit-exact, and a full source-residual patch reproduces source's **m=1** prediction
(only m=1 transports — this is what fixes the estimator horizon). A guard failure
is `HARNESS_FAIL`. *Planted-unit / component-patch self-tests are deferred to L1*,
where the unit enumerator is first built.

**The floor (pinned).** The L0 floor is the **facet-matched-source full residual
patch**: clean and source match on the target facet, so the patch carries no
facet information and should not move it. `floor_score(f) = mean |obs_patch −
obs_un| / G_f`, where `G_f` is the median real-pair gap `|obs_un − obs_src|`.
`FLOOR_FAIL` if `floor_score > NULL_TOL`. (The random-unit floor proper arrives at
L1 with the enumerator.)

**B — substrate gate (GO/NO-GO).** For each facet `f ∈ {depth, top_type}`:
non-vacuous; estimable; estimator audited vs oracle; clean/source pairs separated;
full/reference patch has room; no-info baseline computed; **dissociable** —
single-facet pairs exist in adequate count at the registered positions.

## Full / reference patch (named, to fix the ceiling)

The room ceiling is **full residual-stream interchange at L1 over the scored
prefix `:t+1`** (clean ← source) — the exp-19 identity-patch analog, a single
well-defined "everything at this layer/position" reference, here read at the m=1
prediction it transports. The block/component outputs that L1 will localize are
**sub-units** of this. **L0 does not patch components at all** (the enumerator is
L1's); the room ceiling is the residual-full reference, nothing else.

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
| `CLOSE_MASS_MIN` | 0.05 | denominator guard for `type_obs`: rows with total close mass below this are excluded from type cells |

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
| the harness "works" trivially | the model guards (no-op bit-exact; full-source patch reproduces source's m=1) plus the facet-matched floor `≤ NULL_TOL` must all hold |

Source-delta *magnitude* (large clean→source gaps) is not gated here: every real
pair is in-range by construction, so a `SRC_DELTA_MAX` would exclude nothing at L0.
Matched-delta binning of the effect estimate is an **L1** concern (effect
estimation), deferred to that design draft.

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
TARGET_VACUOUS(f)      — std(f_obs) < VAR_MIN  (variance vacuity; the no-info random-unit floor is a separate control, not this branch)
SMALL_SOURCE_DELTA(f)  — clean/source facet separation < SRC_DELTA_MIN
FLOOR_FAIL(f)          — no-info / random-unit patch closes > NULL_TOL (floor not clean; importance would be uninterpretable)
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
