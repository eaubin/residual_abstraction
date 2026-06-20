# Experiment 32 — Pre-I2 read-transport discriminator on pstack-L4 — PRE-REGISTRATION

**Script:** `scripts/interventions/read_transport_discriminator.py`.

**Status: pre-registered (amended), not yet run.** This writeup and the runnable
script are committed for pre-registration review *before the first claim-producing
run*, per `EXPERIMENT_REVIEW_PROTOCOL.md`.

> **Amendment (pre-run revision, not a result change).** The first registration
> co-gated the gain/bias refit on the cosine-ceiling band, so both decisive
> outcomes required a noisy ~n/4-row statistic to land in a narrow band and the
> most likely "shared" world could still return a non-decision (`COSINE_UNRELIABLE`)
> that fails to clear the gate this experiment exists to settle. This revision
> makes the **refit primary** (the direct, well-controlled test) and **demotes the
> cosine ceiling to a reported overlay** (`cosine_instrument`), never a verdict
> input. The verdict is now a 3-branch partition decided by the refit alone;
> `COSINE_UNRELIABLE` is dropped as a verdict branch. (Also: the registered run
> now uses an accelerator — MPS on Apple silicon, else CUDA — with CPU fallback;
> see Non-goals.)

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

**Measurement 1 — gain/bias-refit discriminator (the verdict).** Freeze the
discovery-fit direction `wd`; project the held bin's centered residuals onto it to
a scalar `s = Rc_held @ wd`; refit **only** a scalar gain+bias `y ≈ g·s + b` on
the held train half; score `r2_refit_held` on the held test half. This is the
direct shared-vs-specific test and uses no cosine: if a SHARED direction underlies
both bins, recalibrating its per-position scale/bias must recover held in-place
`R2`. A **pooled-positions cross-check** (`r2_pooled_min`, `r2_pooled_mean`) fits
one shared affine read pooled over all eight single positions and scores
per-position in-place `R2`.

**Measurement 2 — cosine reliability ceiling (reported overlay, not the
verdict).** Within each single-position frame, split the train rows into two
disjoint halves, fit a full affine read on each, and take the signed cosine of the
two unit covectors; average across positions → `cos_ceiling`. This is the cosine a
*genuinely shared* read achieves under fit noise at fixed position/predicate — the
missing baseline that makes `COS_SHARED = 0.70` interpretable, used here to drive
the reported `cosine_instrument` overlay (it answers whether exp-31's cosine test
was even a valid sharing instrument). Reported against the noise floor `1/√d`. The
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

`CEILING_MIN` and `COS_SHARED` calibrate the **reported overlay**, not the
verdict (see the overlay subsection). They split the cosine ceiling three ways:
`< 0.50` (**UNRELIABLE** — same-position fits do not even agree above noise);
`0.50…0.70` (**BELOW_SHARING_BAR** — a genuinely shared read agrees with itself
yet cannot reach `COS_SHARED`, so exp-31's cosine test was rigged toward
"specific"); `≥ 0.70` (**SHARP** — a shared read reaches `COS_SHARED`, so a
near-zero cross cosine would be real specificity). Only `R2_MIN` (with
`FLOOR_MARGIN` against the shuffle floor) is load-bearing for the verdict.

## Registered Outcome Table

Exactly one branch per `(target, seed)`, by the precedence below; the script's
`classify_target` fills it. The verdict is decided by the **gain/bias refit
alone** — the cosine ceiling is not a verdict input. Each row names the
carry-forward decision it changes. `recovers` ≔ `r2_refit_held ≥ R2_MIN` **and**
`r2_refit_held − r2_shuffle_held ≥ FLOOR_MARGIN`.

| branch | plain-language gloss | condition | routes to (changes I2) |
|---|---|---|---|
| `PREMISE_NOT_REPRODUCED` | exp-31 readable-late result does not reproduce here | `std_disc<VAR_MIN` or `std_held<VAR_MIN` or `r2_inplace_disc<R2_MIN` or `r2_inplace_held<R2_MIN` or `r2_inplace_held−r2_shuffle_held<FLOOR_MARGIN` | fix substrate/measurement before interpreting; **I2 blocked** |
| `SHARED_WITH_DRIFT` | the frozen disc direction, just rescaled per position, decodes held — one shared read, so exp-31's "specific" call was a fit-underdetermination artifact | `recovers` | **DROP** I2 position-conditioned reads; re-ask I1 with a recalibrated transport-valid read (I2 narrows to recalibration confirmation) |
| `POSITION_SPECIFIC_CONFIRMED` | rescaling the frozen disc direction does not decode held — the held read is a genuinely different direction | `not recovers` | **PROCEED** to I2 with position-conditioned reads |

**Precedence** (mutually exclusive, exhaustive): (1) premise gate; (2) `recovers`
→ `SHARED_WITH_DRIFT`; (3) `not recovers` → `POSITION_SPECIFIC_CONFIRMED`. The
refit is a 2-parameter (gain+bias) out-of-sample fit on a 1-D feature, scored on
the held test half against the same shuffle floor as the in-place read, so the
verdict no longer depends on any narrow-band cosine statistic.

The pooled-positions read (`r2_pooled_*`) is **descriptive / cross-check only**:
the registered discriminator is the gain/bias refit (Measurement 1), and the
pooled read is reported as an independent robustness check that should agree with
it. A refit-vs-pooled disagreement is residual risk to report at result review,
not a verdict input.

### Cosine-instrument overlay (reported; never a verdict input)

Modeled on exp 31's `positions_exchangeable` overlay, the script computes and
prints a 3-state flag beneath each verdict — it **annotates** the verdict, it
does not change it. It answers the side-question "was exp-31's cosine test even a
valid sharing instrument here?", using the within-position cosine ceiling
(`cos_ceiling`) reported alongside `cos_cross` and the `1/√d` noise floor:

| `cosine_instrument` | condition | how to read it as a caveat on the verdict |
|---|---|---|
| `UNRELIABLE` | `cos_ceiling < CEILING_MIN` | even same-position/same-predicate fits do not agree above noise → exp-31's cosine carried no sharing signal; the cosine half of exp-31's call was uninformative regardless of this experiment's verdict |
| `BELOW_SHARING_BAR` | `CEILING_MIN ≤ cos_ceiling < COS_SHARED` | genuinely shared fits agree with themselves but cannot reach `COS_SHARED` → exp-31's cosine test was rigged toward "specific"; a `SHARED_WITH_DRIFT` verdict is *expected* to co-occur with this state |
| `SHARP` | `cos_ceiling ≥ COS_SHARED` | a shared read does reach `COS_SHARED`, so exp-31's near-zero cross cosine *was* a meaningful "specific" signal |

The one combination worth a **second look at result review** (reported, not
re-routed) is `SHARED_WITH_DRIFT` under a `SHARP` instrument: the refit says one
shared read decodes held, yet a sharp cosine instrument says exp-31's near-zero
cross cosine should have been a real specificity signal — a tension between the
two reads of the same geometry. The verdict still follows the refit; the overlay
flags the case for narrative scrutiny.

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
- **P3 (refit discriminator; the headline decider).** `r2_refit_held` — does the
  frozen discovery direction, rescaled, recover held in-place `R2`? High
  (`recovers`) → `SHARED_WITH_DRIFT`; low → `POSITION_SPECIFIC_CONFIRMED`. This is
  the sole load-bearing verdict quantity.
- **P4 (cosine ceiling; reported instrument diagnostic, NOT a verdict input).**
  `cos_ceiling` against the `1/√d ≈ 0.125` floor and the `0.50` / `0.70` bars,
  emitted as the `cosine_instrument` overlay. It tells us whether exp-31's cosine
  could ever have adjudicated sharing; it does not change this experiment's
  verdict.
- **P5 (pooled cross-check; report).** `r2_pooled_min/mean` should track the
  refit verdict; a disagreement is reported, not silently resolved.
- **P6 (controls; expected).** `phi3_all_neutral` stays vacuity-limited and
  `phi4_first_matched` stays non-decodable as a read on both grouped bins.
  Controls are reported, never promoted to targets.

## Confound table — load-bearing quantity (author-side, per protocol)

After the amendment a **single** quantity carries the verdict —
`r2_refit_held` — so the confound table for it (Q1) is the load-bearing one. The
cosine ceiling is now diagnostic-only; its confounds are listed afterward as
caveats on the *side-finding*, no longer verdict risks.

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

**Diagnostic-only note — the cosine ceiling (`cos_ceiling`, `cos_cross`).** No
longer a verdict input, so these are caveats on the reported `cosine_instrument`
overlay, not verdict risks. The ceiling is the very statistic the reviewer
flagged as biased low, and the overlay should be read with its own confounds in
mind:

| caveat on the ceiling side-finding | effect |
|---|---|
| per-fit data volume is only ~`n/4` rows | each ceiling fit uses half of a single-position bin's train half; at small row counts in `d=64` the two covectors are noisier and their cosine is **biased low**, so a low/`BELOW_SHARING_BAR` reading may understate true sharing |
| ridge shrinkage at half data | `LAM` is held fixed while the per-fit sample shrinks, so the half-data fits are relatively more shrunk than the full in-place reads — again pushing the ceiling down |
| sign ambiguity of ridge covectors | both fits target the same-signed `p_phi`, so the covector sign is consistent; signed cosine is used (as in exp 31) |

Because the verdict no longer reads the cosine, these biases can no longer flip a
shared/specific call; they only color how strongly the overlay annotates it.

## Reliability baselines for the thresholds (author-side, per protocol)

The one **load-bearing** threshold is paired with the value a genuine positive
and pure noise reach; the cosine bars now calibrate the reported overlay only:

- **`R2_MIN = 0.50` (recovers bar) for `r2_refit_held` — load-bearing.**
  Ceiling = in-place held `R2` with the held-fit direction (the best a
  position-specific read does, established by the premise gate); floor =
  `r2_shuffle_held` (label-shuffle). `recovers` is read against both, so the
  verdict's only threshold sits between a measured ceiling and a measured floor.
- **`COS_SHARED = 0.70` and `CEILING_MIN = 0.50` — overlay calibration, not the
  verdict.** Ceiling = `cos_ceiling` (two genuinely shared fits); floor =
  `1/√d ≈ 0.125` (independent random covectors). These bars now bucket the
  reported `cosine_instrument` overlay; they no longer gate any verdict branch,
  which is the whole point of the amendment (a noisy `n/4`-row statistic must not
  decide the gate).

## Measured-but-unadjudicated audit (author-side, per protocol)

After the amendment `classify_target` reads **only** the refit + premise
quantities (`r2_refit_held`, `r2_shuffle_held`, `r2_inplace_disc`,
`r2_inplace_held`, `std_disc`, `std_held`). Everything else the script computes is
declared descriptive/overlay here so a reader does not over-weight it:

- `cos_ceiling` — drives the reported `cosine_instrument` overlay; not a verdict
  input.
- `cos_cross` — exp-31 continuity number, printed in the overlay line; the verdict
  reads neither it nor the ceiling. *This also moots the prior registration's
  "partition assumes `cos_cross ≈ 0` but does not gate on it" risk — there are no
  longer any cosine-driven contradiction cells.*
- `noise_floor` — reported as the reference for `cos_ceiling`; not a verdict input.
- `r2_pooled_min` / `r2_pooled_mean` / `pooled_per_pos` — pooled-read cross-check
  on the refit (descriptive; the gain/bias refit is the registered
  discriminator). Disagreement with the refit is reported as residual risk.

## Carry-forward decision this gate changes

This experiment **gates whether I2 runs and in what form**, per
`INTERVENTION_CLASS_BENCHMARK.md` § I2 Pre-I2 gate:

- `SHARED_WITH_DRIFT` → **drop** the I2 position-conditioned-read mandate; I2
  narrows to a recalibration confirmation, and I1 is re-asked with a recalibrated
  transport-valid read.
- `POSITION_SPECIFIC_CONFIRMED` → **proceed** to I2 with position-conditioned
  reads, as exp 31 provisionally routed.
- `PREMISE_NOT_REPRODUCED` → fix the substrate/measurement before any I2
  interpretation.

The reported `cosine_instrument` overlay does not change which branch fires; it
annotates the verdict (e.g. a `SHARED_WITH_DRIFT` under a `SHARP` instrument is the
combination flagged for a second look at result review).

## Halt Conditions

The run halts if the checkpoint config differs from the registered `pstack-L4`
config, or any PairSet known-answer self-check fails. There is no I0 routing gate:
I0 routes the *intervention* benchmark, and this diagnostic applies no patch.

## Reviewable Failure Modes

- broad labels replacing the narrow construct: every verdict is an L1 / `pstack`
  / `m=3` / position-split **readability** claim, never a writability claim;
- a headline resting on a threshold with no baseline: the sole verdict threshold
  (`R2_MIN` on `r2_refit_held`) is read against a measured ceiling (in-place held
  `R2`) and a measured floor (`r2_shuffle_held`); the noisy cosine ceiling is
  demoted to a reported overlay so it cannot decide the gate;
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
- Accelerator: the registered run moves the model to `model.pick_device()` — MPS
  on Apple silicon, else CUDA, else CPU. The living eval primitives
  (`midstream.stream_to` / `chain_run`) now follow the model's device and return
  CPU tensors, so this is a no-op for the CPU-loaded frozen scripts (exp 30/31
  stay bit-for-bit on CPU); only this script moves the model. Cross-device float
  results differ slightly, so this experiment's own canonical output should be
  reproduced on the same device it was first run on.

## Results

*(to be filled after the preregistered run; do not run before review)*
