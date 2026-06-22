# Experiment 37 — Dyck-2 localization substrate gate + harness — CONCLUDED

**Script:** `scripts/localization/l0_substrate_gate.py`. **Output:**
`out/exp37_dyck2.txt`. **Result (4 seeds 700–703):** `top_type → GO`
(certified L1 substrate), `depth` close-readiness `→ FLOOR_FAIL` (HELD, purity
uncertified); dissociability abundant for both (**P3 resolved**). See **Results**
at the foot of this file; the body below is the pre-registration as reviewed.

**Status: concluded.** L0 of the state-localization phase
(`docs/STATE_LOCALIZATION.md`): harness build + a GO/NO-GO substrate gate. No
localization claim here — that is L1.

**Construct changes forced while coding (flagged for review):** (a) all estimators
are **m=1** — the L1 position-`t` patch transports only the next-token prediction
(see below), so multi-step observables are not usable; (b) the depth facet is the
m=1 **close-readiness** proxy, a coarse depth signal; (c) the floor is pinned to
the facet-matched-source full patch with an explicit normalization (below);
(d) component/block self-tests are **deferred to L1**; (e) the
**room/`NO_ROOM`/`CEIL_MIN` machinery is removed** — under exact m=1 transport
closure ≡ 1, so a room gate is tautological (movability is the model guard);
(f) dissociability is gated **per (position × held-value) cell**; (g) the **floor
is label→observable determinism** (within-class spread ÷ gap), computed directly —
not a patch test — so it no longer assumes transport away from the guard;
(h) `TARGET_VACUOUS` now uses an **unconditioned** eval std, independent of the
gap; (i) a **conservative floor cut** (`FLOOR_MARGIN_LO=0.04`) is pre-committed
because the floor baseline is uncharacterized at L0; (j) `SEED_UNSTABLE` is
registered in the verdict block. The model guard now asserts transport at **every
registered position**. Single-seed smoke: **`top_type` GO-able (floor ≈ 0),
`depth` → `FLOOR_FAIL` (floor ≈ 0.0499, marginal)**; depth gap boundary-carried
(~2700 vs ~100). The registered 4-seed run decides.

## What greenlighting this approves (read this first)

Three load-bearing choices; the rest is mechanics.

1. **Vehicle and target are gated, not assumed.** Dyck-2 + {depth, top_type}
   proceed to L1 only if this gate shows a real clean-source gap, a clean floor,
   *and* dissociable pairs. A typed
   NO-GO reroutes a specific choice — a success of the gate, not a failure.
2. **Localization will use interchange patching, validated first.** The harness
   must recover known answers on synthetics before any real use; if it does not,
   L1 does not run.
3. **Each facet is three separated objects:** a prefix-computable *selection
   label* (builds pairs), an *observable estimator* (the m=1 facet scalar), and an
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
marginal `q`. The run reports which **depth contrasts** carry the surviving
clean-source gap (boundary depth-0/full vs interior-only), since interior-vs-interior
contrasts are mostly filtered by `SRC_DELTA` — so a depth GO is read as
*close-readiness*, not graded depth (the smoke: ~2700 boundary vs ~100 interior).

**Live quantities (per facet), all on the one m=1 scalar.** The clean→source
**gap** `|obs_un − obs_src|` (must exceed `SRC_DELTA_MIN`); its **std** across
examples (`VAR_MIN`); the **observable-vs-oracle** gap (`OE_BAND`); and the
**floor** — movement of the observable under a facet-matched-source patch,
normalized by the median gap (`NULL_TOL`). There is **no room/closure gate**: the
full prefix patch transports the m=1 marginal exactly (model guard), so closure ≡ 1
would be tautological. Movability is established by that guard, not scored here.

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
| pairs | a **cell = (registered position × held-fixed value)** (held = `top_type` for the depth facet, `depth` for the type facet); up to ≈ 512 deduped pairs generated per cell. Dissociability requires **≥ `MIN_CELLS` (2)** cells each with **≥ `MIN_PAIRS_PER_CELL` (256)** pairs, and metrics pool only qualifying cells — so a facet cannot pass on one easy cell. (Sampled with replacement, then deduped.) |
| oracle use | endpoint/estimator audit only; never selection or scoring |

## Two parts

**A — harness self-tests + model guards (non-claim).** Pipeline:
`load → facet-conditioned pairs → m=1 interchange patch → score → aggregate`.
Pure-function `--selftest` (no checkpoint): parser labels, the two m=1
observables, the facet-pairing invariants, verdict precedence, and
majority — known answers. Model-level guards (run in the main path, the AGENTS
bit-for-bit discipline): a no-op (own-residual) patch reproduces unpatched
bit-exact, and a full source-residual patch reproduces source's **m=1** prediction
(only m=1 transports — this is what fixes the estimator horizon). A guard failure
is `HARNESS_FAIL`. *Planted-unit / component-patch self-tests are deferred to L1*,
where the unit enumerator is first built.

**The floor (pinned) — what it actually measures.** Under exact m=1 transport, a
facet-matched-source patch's m=1 marginal equals the source's, so its observable
*is* `obs(fb)` exactly. The floor is therefore **not a patch test**; it is
**label→observable determinism**: over facet-matched (same-facet, same-position)
pairs `(fa, fb)`, `floor_score(f) = mean |obs(fb) − obs(fa)| / G_f`, the
**within-class observable spread ÷ between-class gap** `G_f` (the mean real-pair
gap) — a purity/SNR ratio. The numerator (spread) and the denominator `G_f` (gap)
are pooled over the **same population of positions** — floor pairs are collected
only at positions that produced a qualifying dissociable cell — so a position
cannot feed one side of the ratio and not the other. It is computed **directly**
(no patch), which also removes any reliance on transport holding away from the
guard. A high floor means
the observable depends on nuisance beyond the facet label. `FLOOR_FAIL` if
`floor_score > FLOOR_MARGIN_LO` (conservative, see below). The random-unit floor
proper — the real baseline — arrives at L1 with the enumerator; **read an L0
FLOOR_FAIL as "purity uncertified," not "proven impure."**

**Conservative floor routing (pre-committed).** The floor's baseline is
uncharacterized at L0 (we have no measured pure≈0 or impure-ceiling reference,
only near-threshold values), so a floor in `(FLOOR_MARGIN_LO=0.04, NULL_TOL=0.05]`
routes `FLOOR_FAIL` (flagged *marginal*), not `OK`. The single-seed smoke puts the
**depth** (close-readiness) floor at ≈0.0499 → `FLOOR_FAIL`; `top_type` ≈0 → clean.
So the expected L0 outcome is `top_type` GO-able, depth held for purity.

**Honest reading of the cut (the load-bearing number has no measured impurity
reference).** The floor is a ratio: `0` is label-determined and the no-information
case is `≫ 1` (between-class gap → 0). The only empirical references from this run
are the **clean** end (`top_type` floor ≈ 0) and that no-info `≫ 1` ceiling. Against
that scale, depth's ≈0.05 sits **near the clean end** — within-class spread ≈ 5%
of the between-class gap — **far from the no-information point** (`≫ 1`, where
within-class spread ≈ the gap); it fails only because `FLOOR_MARGIN_LO=0.04` is set
essentially just below the observed value. `NULL_TOL=0.05` is a deliberately strict
*registered* purity cut, not a measured impurity point: neither bound of the cut is
*measured* for depth (clean ≈ 0 and no-info `≫ 1` bracket it, but the impurity point
is registered by guess, not calibrated). So the honest reading of a depth `FLOOR_FAIL` is "fairly
pure but **uncertified** pending L1's random-unit baseline" — not neutral, and
certainly not impure (pairs with the model-approximation row above).

**B — substrate gate (GO/NO-GO).** For each facet `f ∈ {depth, top_type}`:
non-vacuous; estimable; estimator audited vs oracle; clean/source pairs separated;
**observably pure** (a facet-matched-source patch does not move the facet);
**dissociable** — single-facet pairs in adequate count per cell. Movability by the
full prefix patch is exact m=1 transport (model guard), not a gate.

## Movability is the transport guard, not a gate

The full residual-stream interchange at L1 over the scored prefix `:t+1`
(clean ← source) transports the model's **m=1** next-token prediction **exactly** —
`model_guards` asserts `marginal(q_full, m=1) == marginal(q_src, m=1)` bit-for-bit.
Because both observables are functions of that marginal, a "room"/closure gate would
read 1 for every pair — tautological — so **there is none**. Movability is
*established* by the guard; the gate routes on calibration, variation, gap, purity,
and dissociability. Component patching is L1's, not L0's.

## Thresholds and expected baselines (registered here, not deferred to code)

| name | value | meaning |
|---|---|---|
| `VAR_MIN` | 0.05 | min std of the facet observable over an **unconditioned** eval sample at the registered positions (not the contrastive pairs) |
| `OE_BAND` | 0.10 | max estimator-vs-oracle endpoint gap (Dyck obs/exact ran 0.064–0.073, exps 19–21) |
| `SRC_DELTA_MIN` | 0.05 | min mean clean-vs-source facet separation (below it a cell is non-diagnostic) |
| `NULL_TOL` | 0.05 | clear-impurity floor ceiling (floor = within-class spread ÷ gap) |
| `FLOOR_MARGIN_LO` | 0.04 | conservative cut: floor > this → `FLOOR_FAIL` (the band up to `NULL_TOL` is *marginal* but still fails), pending an L1 random-unit baseline |
| `MIN_PAIRS_PER_CELL` | 256 | min pairs per (position × held-value) cell |
| `MIN_CELLS` | 2 | min qualifying cells, else `NOT_DISSOCIABLE` |
| `CLOSE_MASS_MIN` | 0.05 | denominator guard for `type_obs`: rows with total close mass below this are excluded from type cells |

Threshold *values* are gate cutoffs, not claims; they are printed and audited
symmetrically (FORMALISM §6.1 rule 8). The full transport is bit-exact (model
guard at every position). The floor baseline (pure≈0 vs an impure ceiling) is
**not** measured at L0 — only near-threshold values — which is exactly why the
`FLOOR_MARGIN_LO` conservative cut is registered; the real baseline is L1's.

## Confound table — load-bearing quantity (the floor, which decides depth at ≈0.0499)

| mechanism that could produce a near-threshold depth floor | excluded by? |
|---|---|
| genuine nuisance-dependence of close-readiness on stack history (real impurity) | the intended signal — this is what `FLOOR_FAIL` should catch |
| the trained model's imperfect stack tracking (an approximate transformer, not an exact stack): at a fixed depth label, close-readiness still varies — within-class spread that is model quality, not substrate impurity | **not excluded at L0**; calibrated by L1's random-unit floor, whose baseline carries the *same* model-error floor — which is precisely why that baseline, not 0, is the real reference. This is also why an L0 `FLOOR_FAIL` ≠ impure |
| within-class spread from depth capping `min(d, m)` (depth-3 vs depth-5 examples differ) | **moot here**: Dyck-2 is depth-bounded at 3 = m, so `min(d, m) = d` always — no capping occurs |
| horizon: close-readiness varies with length-to-go at fixed depth | **controlled**: floor pairs are *within a single position* `t`, so length-to-go (`seq_len − t`) is constant across the pair; pooling is over per-position spreads |
| small-sample noise in `mean|obs(fb) − obs(fa)|` per cell | partly — `MIN_PAIRS_PER_CELL=256`; the per-cell sizes are printed (`cell[min,med]`) so a thin pass is visible |
| pairs scarce/skewed because `depth`,`top_type` correlate in Dyck | `NOT_DISSOCIABLE` below `MIN_PAIRS_PER_CELL × MIN_CELLS` per (position × held-value) cell |
| depth "GO" misread as graded depth movable | the depth-contrast report (boundary vs interior); interior-vs-interior gaps are mostly filtered by `SRC_DELTA` |

The depth-cap and horizon confounds — the two the reviewer flagged as live — are
moot and controlled respectively for *this* checkpoint, so a near-threshold depth
floor most plausibly reflects either genuine impurity or small-sample noise; both
route conservatively to `FLOOR_FAIL` until L1's baseline. (The "harness works
trivially" mechanism is covered by `HARNESS_FAIL` — the model guards must hold.)

The gate's `delta` is the **pooled mean over all kept pairs** in qualifying cells,
including small-gap interior-depth pairs (the per-pair `≥ SRC_DELTA_MIN` filter
feeds only the descriptive depth-contrast counter, not the metric). For depth,
interior-vs-interior pairs are *not* in-range — so the pooled `delta` is pulled
down by them, which is **conservative** (it can only make `SMALL_SOURCE_DELTA`
easier to fire, never fake a GO). No `SRC_DELTA_MAX` is gated; matched-delta
binning of the effect estimate is an **L1** concern, deferred.

## Predictions

- **P1 (self-tests pass; expected).** A failure is a method/implementation bug,
  not a substrate fact.
- **P2 (top_type GO-able; depth FLOOR_FAIL; expected from the smoke).** Both
  facets are non-vacuous (unconditioned std), calibrated, and clean-source
  separated. `top_type` floor ≈ 0 (clean) → GO-able; **depth** (close-readiness)
  floor ≈ 0.0499 → `FLOOR_FAIL` under the conservative cut. So L0 most likely
  certifies `top_type` as a localization substrate and holds depth for purity.
- **P3 (dissociable pairs; the real unknown).** Dyck-2 should admit depth-only and
  type-only pairs by construction, but their *count at the registered positions*
  is what this gate de-risks. Thin counts route to a position/length adjustment.

## Verdict (exhaustive, non-overlapping; precedence high → low)

```text
HARNESS_FAIL           — a model guard fails (no-op not bit-exact, or full patch != source m=1, at any position); blocks all
OBS_EXACT_DRIFT(f)     — estimator-vs-oracle gap > OE_BAND (gap uninterpretable under drift)
TARGET_VACUOUS(f)      — std over an UNCONDITIONED eval sample of the facet observable < VAR_MIN (the facet barely varies on-distribution). Determinable without pairs, so it is evaluated even when cells are thin and outranks NOT_DISSOCIABLE there (OBS_EXACT_DRIFT, which needs the oracle gap from cells, is the only higher predicate and is undeterminable on that path)
SMALL_SOURCE_DELTA(f)  — mean clean/source facet separation < SRC_DELTA_MIN
FLOOR_FAIL(f)          — floor_score (label->observable determinism: within-class spread / gap) > FLOOR_MARGIN_LO; the band (FLOOR_MARGIN_LO, NULL_TOL] is flagged *marginal* but still fails (conservative, pending an L1 baseline)
NOT_DISSOCIABLE(f)     — fewer than MIN_CELLS cells with >= MIN_PAIRS_PER_CELL pairs
SEED_UNSTABLE(f)       — no branch is a unique >=3/4 majority across the 4 seeds (a split). NOT a construct NO-GO: the gate is underpowered for f -> add seeds / tighten sampling, then re-run. Sits at the bottom of precedence.
GO                     — none of the above fires for either facet
```

Simultaneous failures: every fired branch is reported; routing follows the
highest-precedence one. Each NO-GO reroutes a specific choice, not the whole phase.
The run emits an explicit **per-facet routing** line (each facet `GO`/`HELD → label`)
as the primary output; the single `DECISION` headline is only the highest-precedence
reroute across facets, so e.g. expected `top_type` GO + `depth` FLOOR_FAIL prints a
GO for `top_type` even though the headline shows `FLOOR_FAIL(depth)`.

## Honesty, halt, non-goals

- **Honesty:** labels, estimators, and pairing are observable; the Dyck oracle
  audits endpoints only.
- **Halt:** checkpoint config mismatch, or any self-test failing its known answer.
- **Non-goals:** no localization, separability, or intervention claim (L1+); no
  granularity sweep and no head/direction enumerator yet (L0 builds block-level
  hooks only); no real-LLM claim. The phase doc is loose and may adapt as L1 lands.

## Results

Run artifact: `out/exp37_dyck2.txt` (`device=mps`). Validity gate PASS
(gap-to-optimal −0.0121 nats); model guards OK (no-op patch bit-exact, full patch
m=1 = source m=1 at every registered position); `--selftest` OK (P1). 4 seeds
700–703.

```text
per-facet routing:
  depth    : HELD -> FLOOR_FAIL
  top_type : GO (certified for L1)

DECISION (highest-precedence reroute): FLOOR_FAIL(depth)
```

Per-facet verdict was stable 4/4 seeds (`{depth: FLOOR_FAIL, top_type: OK}`).
Held quantities (ranges across the four seeds):

| quantity | `top_type` | `depth` (close-readiness) |
|---|---|---|
| verdict (4/4) | **OK → GO** | **FLOOR_FAIL → HELD** |
| qualifying cells | 12 | 8 |
| pairs per cell `[min, med]` | `[510, 512]` | `[512, 512]` |
| clean/source gap `delta` | 1.000 | 0.272–0.278 |
| floor (within-class spread ÷ gap) | 0.000 | **0.051–0.054** |
| estimator-vs-oracle `oe` | 0.000 | 0.008 |
| unconditioned `std` | 0.479–0.483 | 0.295–0.298 |

### What the run establishes

**P3 (the real unknown) resolved strongly favorably.** Every (position × held-value)
cell cleared `MIN_PAIRS_PER_CELL=256` with room to spare (510–512 pairs), for both
facets at all four positions — `top_type` 12 cells, `depth` 8 cells. Dissociable
single-facet pairs are abundant in Dyck-2 at the registered positions; no
`NOT_DISSOCIABLE`, no position/length adjustment needed. This is the substrate
fact the gate was built to de-risk, and it is the cleanest positive: Dyck-2 *does*
supply the separable, dissociable inventory `pstack` lacked.

**`top_type` is certified clean on every axis** — perfect clean/source separation
(`delta` 1.000), zero floor (the type-fraction is label-determined: same top_type
⇒ same observable), zero estimator-vs-oracle gap, non-vacuous (`std` ≈ 0.48). It
goes to L1 as a localization substrate without caveat.

**`depth` close-readiness gap is boundary-carried, as scoped.** The surviving
clean/source gap is carried by boundary (empty/full) contrasts — boundary
*n* ≈ 2721–2778 vs interior *n* ≈ 78–111 across seeds — confirming the m=1 signal
separates empty/interior/full close-readiness, not graded interior depth. A depth
result is read as close-readiness, per the registered scope.

### The depth floor (the load-bearing number) — honest reading

The depth floor came in at **0.051–0.054**, i.e. just **above** `NULL_TOL=0.05`,
not in the `(FLOOR_MARGIN_LO, NULL_TOL]` *marginal* band the single-seed smoke
(0.0499) had predicted — so it is a **non-marginal** `FLOOR_FAIL` (the run shows no
`*MARGINAL` flag), and prediction **P2's verdict holds but its "marginal"
sub-classification does not**. The verdict (`depth → HELD`) is unchanged.

The reading is the one registered, and it survives the slightly-higher value:
on the ratio's natural scale (0 = label-determined, ≫1 = no information) a floor of
≈0.052 means within-class spread is ≈5% of the between-class gap — **near the clean
end, far from the no-information point** — failing only the deliberately strict
registered cut. Per the confound table, that residual within-class spread has at
least two unseparated sources (genuine impurity vs the model's imperfect stack
tracking), and L0 has **no measured impurity reference**. So depth is held as
**"observable purity uncertified, not proven impure"**; L1's random-unit floor —
which carries the same model-approximation floor — is the real baseline that
adjudicates it.

### Routing

`top_type` proceeds to L1 (exp 38) as a **certified** localization substrate.
`depth` (close-readiness) proceeds to L1 **carried but flagged**: dissociable and
well-separated, purity uncertified pending the random-unit baseline. No facet is
dropped and no redesign is triggered — L0 did its job (de-risk dissociability,
flag the one open purity question for the rung that can measure it). The
summary-not-state scope (depth = close-readiness, not graded depth) stands and is
L3's burden, not L1's.
