# Experiment 32 — Pre-I2 read-transport discriminator on pstack-L4 — PRE-REGISTRATION

**Script:** `scripts/interventions/read_transport_discriminator.py`.

**Status: pre-registered, not yet run.** This writeup and the runnable script are
committed for pre-registration review *before the first claim-producing run*, per
`EXPERIMENT_REVIEW_PROTOCOL.md`.

Diagnostic-only. This experiment produces **no** writability, controllability, or
intervention claim. It produces a typed *routing* decision about a
read/representational-geometry question, and is the gate
`INTERVENTION_CLASS_BENCHMARK.md`'s **Pre-I2 gate** now requires before any
read-freedom work.

## Phase fit and the prior failure this resolves

Exp 31 (read-transport atlas) concluded
`ATLAS(phi1_next_closes=POSITION_SPECIFIC_READ, phi2_net_return=POSITION_SPECIFIC_READ)`:
both predicates are linearly readable **in place** at the held-out positions
`{26,34}` at `R2 ≈ 0.6–0.75`, but the disc/held read covectors are
near-orthogonal (`cos ≈ −0.12…0.24`) and the disc→held transfer fails. Its
Result-Review Addendum flagged that the **shared-vs-position-specific** half of
that verdict is under-determined:

1. the call turns on one scalar (`cos < COS_SHARED = 0.70`) sitting at the `d=64`
   noise floor (`1/√64 ≈ 0.125`) **with no reliability baseline**; and
2. `cos ≈ 0` between two *strong* in-place reads is equally consistent with
   (a) a genuinely position-specific read direction, **or** (b) a single shared
   direction that two underdetermined collinear ridge fits failed to agree on.

(a) and (b) route differently, so exp 31's routing to I2 with position-conditioned
reads is **provisional**. This experiment runs the cheap discriminator the
addendum named, plus the missing cosine-reliability baseline, to resolve it. It
does not re-open the readable-late result (firmly established in exp 31); it only
adjudicates shared-vs-specific.

## Question

```text
Is the held-out-position read failure a genuinely position-specific read
direction, or a shared direction that only needs per-position gain/bias
recalibration?
```

This is a read/representational-geometry question. **No interventions, no
patching, no learned writes, no CEGAR.** If any patch were added the experiment
would have to be re-scoped as an intervention experiment with the corresponding
baselines.

## Registered Command

After preregistration review only:

```bash
uv run python scripts/interventions/read_transport_discriminator.py \
  --outdir out/pstack-L4 | tee out/exp32_pstack-L4.txt
```

Review-only checks (allowed before approval):

```bash
uv run python scripts/interventions/read_transport_discriminator.py --selftest
uv run python -m py_compile interventions.py battery.py \
  scripts/interventions/read_transport_discriminator.py
```

## Scope Indices

| index | value |
|---|---|
| process/checkpoint | `pstack-L4`, registered config from exp 29/30/31 |
| measurement point | residual stream L1 target-run residual at the scored prefix position |
| horizon | `m=3` within-horizon completion distribution |
| target predicates | `phi1_next_closes`, `phi2_net_return` |
| control predicates | `phi3_all_neutral`, `phi4_first_matched` (read controls only) |
| read class | affine ridge readout from per-position-centered L1 residuals to observable model `p_phi` (same as exp 31) |
| grouped discovery bin | positions `{10,18}`, 512 pairs/seed (exp-30/31 construction) |
| grouped held-out bin | positions `{26,34}`, 1024 pairs/seed (exp-30/31 construction) |
| single-position bins | `{6,10,14,18,22,26,30,34}`, 512 pairs/seed each (exp-31 construction) |
| seeds | `400..403` (same as exp 31, for continuity) |
| exact oracle use | none; observable-only (no patch endpoints to audit) |

The bin construction, per-position centering, full-affine fit, and full-map `R2`
scoring **mirror exp 31 exactly**, so the in-place `R2` and the cross cosine
reproduce exp 31. The helpers are re-declared locally (not imported) because exp
31 is a concluded/frozen script (`AGENTS.md` library-home rule: frozen scripts
are never imported from); they are thin wrappers over the living-library
primitives `interventions.pairset_residual_frame` / `abstraction.center_by_position`
/ `interventions.affine_readout` / `interventions.r2_score` / `predicates.obs_pphi`.
Seed aggregation uses the shared `battery.majority_vote`.

## Registered Measurements (observable-only)

Per `(target, seed)`:

**Premise reproduction (gate).** In-place disc `R2` (`r2_inplace_disc`), in-place
held `R2` (`r2_inplace_held`), and the label-shuffle floor at held
(`r2_shuffle_held`) — exactly as exp 31. The discriminator is only meaningful if
exp 31's *readable-late* result reproduces.

**Measurement 1 — gain/bias-refit discriminator.** Freeze the discovery-fit
direction `wd`; project the held bin's centered residuals onto it to a scalar
`s = Rc_held @ wd`; refit **only** a scalar gain+bias `y ≈ g·s + b` on the held
train half; score `r2_refit_held` on the held test half. This is the direct
shared-vs-specific test and uses no cosine: if a SHARED direction underlies both
bins, recalibrating its per-position scale/bias must recover held in-place `R2`.
A **pooled-positions cross-check** (`r2_pooled_min`, `r2_pooled_mean`) fits one
shared affine read pooled over all eight single positions and scores per-position
in-place `R2`.

**Measurement 2 — cosine reliability ceiling.** Within each single-position
frame, split the train rows into two disjoint halves, fit a full affine read on
each, and take the signed cosine of the two unit covectors; average across
positions → `cos_ceiling`. This is the cosine a *genuinely shared* read achieves
under fit noise at fixed position/predicate — the missing baseline that makes
`COS_SHARED = 0.70` interpretable. Reported against the noise floor `1/√d`. The
exp-31 cross cosine `cos_cross = cos(unit wd, unit wh)` is recomputed for
continuity.

## Thresholds

```text
R2_MIN        = 0.50   # "readable"/"recovers" threshold (exp 30/31)
VAR_MIN       = 0.05   # p_phi std vacuity floor (exp 30/31)
FLOOR_MARGIN  = 0.10   # an R2 must beat its own shuffle floor by this (exp 31)
COS_SHARED    = 0.70   # read covectors counted as a shared direction (exp 31)
CEILING_MIN   = 0.50   # cosine-reliability floor: same-position/same-predicate
                       # fits must agree at >= this for the cosine to be usable;
                       # set well above noise floor 1/sqrt(64)≈0.125 and below
                       # COS_SHARED
LAM           = 1e-2   # ridge penalty (exp 30/31)
SEED_MAJORITY = 3      # >=3/4 seeds for an aggregate, else SEED_UNSTABLE
```

`CEILING_MIN` is the one genuinely new threshold; it is the reliability bar this
experiment exists to establish. The cosine ceiling splits three ways: `< 0.50`
(**unreliable** — same-position fits do not even agree above noise); `0.50…0.70`
(**reliable but below the sharing bar** — a genuinely shared read agrees with
itself yet cannot reach `COS_SHARED`, so exp-31's cosine test was rigged toward
"specific"); `≥ 0.70` (**sharp** — a shared read reaches `COS_SHARED`, so a
near-zero cross cosine is real specificity).

## Registered Outcome Table

Exactly one branch per `(target, seed)`, by the precedence below; the script's
`classify_target` fills it. Each row names the carry-forward decision it changes.
`recovers` ≔ `r2_refit_held ≥ R2_MIN` **and** `r2_refit_held − r2_shuffle_held ≥
FLOOR_MARGIN`.

| branch | plain-language gloss | condition | routes to (changes I2) |
|---|---|---|---|
| `PREMISE_NOT_REPRODUCED` | exp-31 readable-late result does not reproduce here | `std_disc<VAR_MIN` or `std_held<VAR_MIN` or `r2_inplace_disc<R2_MIN` or `r2_inplace_held<R2_MIN` or `r2_inplace_held−r2_shuffle_held<FLOOR_MARGIN` | fix substrate/measurement before interpreting; **I2 blocked** |
| `COSINE_UNRELIABLE` | the cosine baseline cannot decide direction sharing here | `cos_ceiling<CEILING_MIN`, **or** the refit and ceiling do not jointly adjudicate (residual of the two rows below) | cosine cannot adjudicate; pick a different sharing test **before** I2 |
| `SHARED_WITH_DRIFT` | one shared read, just rescaled per position — the exp-31 "specific" call was a fit-underdetermination artifact | `recovers` **and** `cos_ceiling<COS_SHARED` | **DROP** I2 position-conditioned reads; re-ask I1 with a recalibrated transport-valid read (I2 narrows to recalibration confirmation) |
| `POSITION_SPECIFIC_CONFIRMED` | the held read is a genuinely different direction; cos~0 is real | `not recovers` **and** `cos_ceiling≥COS_SHARED` | **PROCEED** to I2 with position-conditioned reads |

**Precedence** (mutually exclusive, exhaustive): (1) premise gate; (2) cosine
instrument check (`cos_ceiling<CEILING_MIN` → `COSINE_UNRELIABLE`); (3) joint
adjudication — `SHARED_WITH_DRIFT` if `recovers ∧ cos_ceiling<COS_SHARED`,
`POSITION_SPECIFIC_CONFIRMED` if `¬recovers ∧ cos_ceiling≥COS_SHARED`, else
`COSINE_UNRELIABLE`. The residual `COSINE_UNRELIABLE` in step (3) catches the two
combinations where the direct refit and the cosine ceiling cannot be reconciled
into one clean shared/specific reading (`recovers ∧ cos_ceiling≥COS_SHARED`: the
refit says shared but a sharp cosine should then have shown a high cross cosine —
contradiction; `¬recovers ∧ CEILING_MIN≤cos_ceiling<COS_SHARED`: the refit says
specific but the cosine cannot corroborate at `COS_SHARED`). In both the honest
route is a different sharing test, not a committed I2 form.

The pooled-positions read (`r2_pooled_*`) is **descriptive / cross-check only**:
the registered discriminator is the gain/bias refit (Measurement 1), and the
pooled read is reported as an independent robustness check that should agree with
it. A refit-vs-pooled disagreement is residual risk to report at result review,
not a verdict input.

## Multi-Seed Aggregation

A target aggregate is the branch appearing in `≥ 3/4` seeds (via
`battery.majority_vote`); otherwise `SEED_UNSTABLE`. The top-level decision is a
routing string over both load-bearing targets, with no precedence implying one
target dominates:

```text
GATE(phi1_next_closes=<verdict>, phi2_net_return=<verdict>)
```

## Predictions

- **P1 (guards; enforced).** Config guard passes, PairSet known-answer
  self-checks pass on all ten bins for all four seeds, and `--selftest` passes.
- **P2 (premise reproduction; likely).** For both targets the grouped in-place
  held `R2` reproduces exp 31 (`≥ R2_MIN`, beating the shuffle floor by
  `FLOOR_MARGIN`); otherwise `PREMISE_NOT_REPRODUCED` blocks interpretation.
- **P3 (refit discriminator; headline).** `r2_refit_held` — does the frozen
  discovery direction, rescaled, recover held in-place `R2`? High → shared with
  drift; low → genuinely specific.
- **P4 (cosine ceiling; headline baseline).** `cos_ceiling` against the
  `1/√d ≈ 0.125` floor and the `0.50` / `0.70` bars. This decides whether the
  exp-31 cosine could ever have adjudicated sharing.
- **P5 (pooled cross-check; report).** `r2_pooled_min/mean` should track the
  refit verdict; a disagreement is reported, not silently resolved.
- **P6 (controls; expected).** `phi3_all_neutral` stays vacuity-limited and
  `phi4_first_matched` stays non-decodable as a read on both grouped bins.
  Controls are reported, never promoted to targets.

## Confound table — load-bearing quantities (author-side, per protocol)

Two quantities carry the headline. For each, three mechanisms that could produce
the same value and the design element that excludes each (or "not excluded").

**Q1 — recovered held `R2` after gain/bias recalibration (`r2_refit_held`).**

| confound producing high `r2_refit_held` | excluded by |
|---|---|
| genuine shared direction (the intended reading) | this is the construct, not a confound |
| spurious fit on few rows / low variance | label-shuffle floor (`FLOOR_MARGIN`) + `VAR_MIN` gate; `recovers` must beat the shuffle floor on the **held test half** |
| `wd` accidentally aligned with a high-variance position-confound axis, not the predicate | per-position centering removes positional means before the fit (`center_by_position`); the disc fit `wd` is a predicate read, scored on held p_phi, not on position |
| the scalar refit overfits 2 parameters | only gain+bias (2 params) fit on the train half, scored out-of-sample on the test half — cannot overfit a 1-D feature |

| confound producing low `r2_refit_held` (→ "specific") | excluded by |
|---|---|
| genuinely position-specific direction (the intended reading) | this is the construct |
| ridge over-shrinkage flattening `wd` | same `LAM` as exp 31's decodable in-place reads, which are not flat; the disc in-place read decodes (premise gate) |
| held `p_phi` too flat to fit anything | `VAR_MIN` vacuity gate on `std_held`; in-place held `R2` (held-fit direction) is high by the premise gate, so the target *is* fittable there |

**Q2 — disc/held read cosine, and its reliability ceiling (`cos_cross`,
`cos_ceiling`).**

| confound producing low `cos_cross` (exp-31 "specific") | excluded / not excluded |
|---|---|
| genuinely different direction (intended) | construct |
| two underdetermined collinear fits disagreeing on a shared direction | **the whole point of `cos_ceiling`**: if same-position/same-predicate fits also fail to agree (`cos_ceiling` low), low `cos_cross` is uninformative — *this confound is what exp 31 left unexcluded and exp 32 measures* |
| sign ambiguity of ridge covectors | both fits target the same-signed `p_phi`, so the covector sign is consistent; signed cosine is used (as in exp 31) |

## Reliability baselines for the thresholds (author-side, per protocol)

Every load-bearing threshold here is paired with the value a genuine positive and
pure noise reach:

- **`COS_SHARED = 0.70` (sharing bar).** Ceiling = `cos_ceiling` (two genuinely
  shared fits); floor = `1/√d ≈ 0.125` (independent random covectors). Without
  the ceiling, `cos_cross < 0.70` is the foregone branch regardless of truth —
  the exact gap exp 31 left open. `cos_ceiling` is now measured and folded into
  the verdict (it is not a print-only number).
- **`R2_MIN = 0.50` (recovers bar) for `r2_refit_held`.** Ceiling = in-place held
  `R2` with the held-fit direction (the best a position-specific read does);
  floor = `r2_shuffle_held` (label-shuffle). `recovers` is read against both.
- **`CEILING_MIN = 0.50` (cosine usability bar).** Floor = `1/√d ≈ 0.125`;
  the bar sits well above it and below `COS_SHARED`.

## Measured-but-unadjudicated audit (author-side, per protocol)

Quantities the script computes that `classify_target` does **not** read, declared
descriptive-only here so a reader does not over-weight them:

- `r2_pooled_min` / `r2_pooled_mean` / `pooled_per_pos` — pooled-read cross-check
  on the refit (descriptive; the gain/bias refit is the registered
  discriminator). Disagreement with the refit is reported as residual risk.
- `cos_cross` — exp-31 continuity number; the *verdict* uses the ceiling, not the
  cross cosine directly (exp 31 already established `cos_cross ≈ 0`).
- `noise_floor` — reported as the reference for `cos_ceiling`; not a verdict
  input.

## Carry-forward decision this gate changes

This experiment **gates whether I2 runs and in what form**, per
`INTERVENTION_CLASS_BENCHMARK.md` § I2 Pre-I2 gate:

- `SHARED_WITH_DRIFT` → **drop** the I2 position-conditioned-read mandate; I2
  narrows to a recalibration confirmation, and I1 is re-asked with a recalibrated
  transport-valid read.
- `POSITION_SPECIFIC_CONFIRMED` → **proceed** to I2 with position-conditioned
  reads, as exp 31 provisionally routed.
- `COSINE_UNRELIABLE` → do **not** start I2 yet; choose a different sharing test
  first.
- `PREMISE_NOT_REPRODUCED` → fix the substrate/measurement before any I2
  interpretation.

## Halt Conditions

The run halts if the checkpoint config differs from the registered `pstack-L4`
config, or any PairSet known-answer self-check fails. There is no I0 routing gate:
I0 routes the *intervention* benchmark, and this diagnostic applies no patch.

## Reviewable Failure Modes

- broad labels replacing the narrow construct: every verdict is an L1 / `pstack`
  / `m=3` / position-split **readability** claim, never a writability claim;
- a headline resting on a threshold with no baseline: addressed by the cosine
  ceiling (`cos_ceiling`) and the refit floor/ceiling, both folded into the
  verdict;
- spurious high `R2` from few pairs or low variance: separated by the
  label-shuffle floor and the `VAR_MIN` gate;
- conflating shared-direction scale drift with a different direction: this is the
  experiment's object, separated by the gain/bias refit;
- mistaking the discriminator for an intervention result: the script applies no
  patch and emits no control/specificity number.

## Non-goals / Scope Guard

- No claim about whether the predicate is writable or controllable.
- No new process training; existing `pstack-L4` only.
- No interventions; if any patch is added the experiment must be re-scoped.
- MPS acceleration deferred: the registered run uses the living CPU evaluator
  `midstream.chain_probs`, as in exp 31.

## Results

*(to be filled after the preregistered run; do not run before review)*
