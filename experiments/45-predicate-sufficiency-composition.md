# Experiment 45 — Facet-factor decodability and composition (type × color) on colored Dyck-2

**Status: DRAFT (pre-registration), revised after the first pre-registration
review.** The runnable script (`scripts/predicate_sufficiency/exp45_*.py`) is the
next artifact and must be committed beside this writeup before the first claim
run, per `EXPERIMENT_REVIEW_PROTOCOL.md`. No claim seed has been run. This
revision addresses the review's two blocking findings — the multiplicative
closing-gate confound (now excluded by **gate-normalization** + a
closes-orthogonal geometry, not asserted) and the missing **empirical floors**
for the composition thresholds (now emitted by `--calibrate`) — and the scope,
budget, and decision-6 deviation notes.

## Phase fit and the scope debt this resolves

First **claim** of the completion-predicate phase (`docs/COMPLETION_PREDICATES.md`):
relate residual abstractions to **named** completion predicates (not the
full-horizon scalar), and measure **composition**. This is the original payload
the program deferred (`docs/archive/ORIGINAL_SIN.md`) — a relation between two
abstraction languages, one over residual states, one over named properties of
futures. It runs on **colored Dyck-2** because composition needs **two** binding
factors (type AND color); dyck2 has only one, and its depth/type entanglement is
already exhausted (exps 40/42/44). The machinery (graded predicate layer,
ctx-reader, colored_dyck2) was built and self-tested as non-claim infrastructure
(commits 7479a8d…a8d2180); this rung is the first to stake a claim on it.

## The predicate suite, the gate, and gate-normalization (load-bearing)

The intuitive "the next close **matches** the prefix top" is **degenerate for
composition**: colored-Dyck matching is grammar-forced, so a closing top matches
on **both** facets at once. Calibration (seed 777) confirmed `matches_type` =
`matches_color` = `matches_both` numerically (all exact-mean 0.811, identical).
So the suite uses **facet-VALUE** predicates — the close that pops the seeded top
has a *specific* facet value — whose values key on **different** facets:

- `phi_type0(c; ctx)` — the close popping the seeded top has **type 0**.
- `phi_color0(c; ctx)` — …has **color 0**.
- `phi_both00 = phi_type0 ∧ phi_color0` — …has **type 0 AND color 0** (the joint).
- `phi_closes(c; ctx)` — …**pops the seeded top in-window** (any facet). This is
  the **gate**: `E_q[phi_closes] = p_close(x)` exactly.

**The multiplicative-gate problem (review finding 1), and its exclusion.** Each
facet score is a PRODUCT of the gate and a facet bit:

```text
E_q[phi_type0](x)  = p_close(x) · P(type=0 | closes, x)
E_q[phi_both00](x) = p_close(x) · P(type=0, color=0 | closes, x)
```

`p_close(x)` varies per prefix with depth. Left in, it confounds **all three**
load-bearing quantities: a product is non-linear in `r` even if `r` encodes the
facet cleanly (spurious `NOT_LINEARLY_DECODED`); `w_type0` and `w_color0` both
load on the shared `p_close` direction (deflated angle → spurious
`ENTANGLED_FACTORS`); and `phi_both00` is a triple product (interaction forced by
the gate, not by the factors → spurious `INTERACTION_PRESENT`). Naming it is not
excluding it.

**Registered exclusion — gate-normalization.** The decode target is the
**gate-normalized conditional**

```text
psi_facet(x) = E_q[phi_facet](x) / E_q[phi_closes](x) = P_model(facet | closes, x)
```

restricted to prefixes with `E_q[phi_closes] ≥ CLOSE_FLOOR` (ratio stability;
calibration: determined tops close with mean ≈0.81, so most qualify). Division
removes `p_close` **by construction** — `psi` is the model's *conditional* facet
probability given the top closes, observable (both terms from the model's `q`).
The remaining structure `psi_both ≈ psi_type · psi_color` (type ⊥ color in the
generator) is now the **genuine composition subject**, not the nuisance gate: the
ΔR² test asks whether the residual encodes the (type,color) **joint** linearly
(conjunction available) or only the marginals (conjunction higher-order) — the
real abstract-interpretation question. A residual-side **closes-orthogonal**
geometry (below) is the additional safeguard for the angle.

`matches_*` is **retained as a registered DEGENERACY control** (the three must
coincide; separation → `HARNESS_FAIL`).

## Scope of the construct (review finding 3 — what is and isn't tested)

Forced matching makes *facet-of-close ≡ facet-of-top*. So the suite decodes the
seeded **top's** (type, color) and their conjunction — a prefix feature read out
through the closing dynamics — **not the open↔close binding relation** (the two
are identified by the grammar and cannot be distinguished here). "Composition of
factors" is genuinely tested (the joint vs marginals question is real); **binding
itself is assumed, not tested**. Wording throughout is "facet factors," not
"binding."

## The question

```text
On colored Dyck-2, at the registered probe layer/positions: from which residual
directions are the gate-normalized facet conditionals psi_facet = P_model(facet |
top closes, x) (observable, from the model's own completions) decodable by a
bounded LINEAR probe — and do the two facet FACTORS (type, color) compose?
Specifically, in the closes-orthogonal complement: (a) are the type and color
readout directions SEPARABLE (angle near the independent-facet reference) or
ENTANGLED (angle below it), and (b) is the conjunction psi_both decodable from the
span of the two marginal directions (a linear DIRECT SUM) or does it require an
INTERACTION direction (the joint (type,color) is higher-order in the residual)?
```

This is **correlational** sufficiency by **decodability** (a V-information
question in a fixed probe class), explicitly distinguished from causal
patch-sufficiency: per phase decision 2, correlational sufficiency runs **before**
the causal loop, and the on-manifold interchange test + the entanglement null
(decision 3) are the **next** rung, a registered non-goal here.

## Why decodability, and the compression reference (not a tautology)

Full-m-gram sufficiency **trivially** implies predicate sufficiency: if an
abstraction preserves the whole completion `q`, it preserves `E_q[phi] = mask·q`
for every `phi`. So "does the m-gram-sufficient abstraction stay sufficient for
predicates" is vacuously yes and is **not** the claim. The non-trivial questions
are **compression** (a scalar predicate should need far fewer residual dimensions
than the full `q`) and **composition geometry** (how the factor directions
relate). The full-m-gram `k*` (the sufficient-subspace dimension from the exp-6 /
`discover.cegar_loop` machinery, pinned to these checkpoints) is reported as the
**compression ceiling** the predicate readouts are measured against.

## Instrument

- **Vehicle:** fresh-seed `colored_dyck2` checkpoints (depth 3, V=8), the
  `train.py` config registered below. Validity gate (gap-to-optimal ≤ 0.005)
  must pass — the noise floor is only interpretable on a converged model.
- **Residual:** the stream after block `ℓ` (registered `PROBE_LAYER`), at
  position `t`, paired with the exact belief (for the ctx-reader / ground-truth
  controls only).
- **Observable target:** `E_q[phi]` per prefix from the model's completion `q`
  via `predicates.eq_obs_from_model` (device-aware `chain_probs`, m=3). This is
  the **observable** the probe predicts; ground truth (`eq_exact`) enters only as
  the calibration audit (`OBS_DRIFT`) and the registered ground-truth control.
- **Prefixes:** DETERMINED-ctx prefixes only (the ctx-reader is exact there;
  `predicates.ctx_along`), at `t ∈ {8,12,16}`, pooled. Inherited-top /
  belief-integrated prefixes are out of scope here (their observable semantics is
  the latent-toy question — `docs/COMPLETION_PREDICATES.md` design discipline).

## Probe class and the sufficiency criterion (calibration-grounded)

- **Probe class = affine ridge** (the registered bounded class; V-information is
  indexed to it). The target is the **gate-normalized** `psi_facet` (above), on
  prefixes with `E_q[phi_closes] ≥ CLOSE_FLOOR`. Fit `psi_facet ≈ w_facet · r + b`
  on the TRAIN split, score on a disjoint HELD-OUT split (decision 4). The fitted
  **`w_facet`** is the rank-1 residual abstraction for that facet.
- **Closes-orthogonal fits (review finding 1, geometry safeguard).** Fit the gate
  direction `w_closes` (ridge on `E_q[phi_closes]`), and **partial it out of `r`**
  (`r_⊥ = r − (r·ŵ_closes)ŵ_closes`) before the facet fits used for the angle/ΔR².
  Facet geometry is measured in `r_⊥`, so any residual `p_close` loading is gone
  in addition to the ratio normalization. The full-`r` geometry is reported as a
  robustness companion, the `r_⊥` geometry is the verdict.
- **kNN R²** (registered `k`) on the same `psi` target is the
  **present-but-not-linear** reference (exp-1/exp-29 interpreter gap): high kNN R²
  with low linear R² = present but not affine → richer-probe route, not "absent."
- **Decodable iff** held-out linear `R²_facet ≥ R2_MIN` **and** pooled-mean
  `|psi_facet − decode|` ≤ `TAU`. `TAU` is set above the estimator noise floor
  (calibration corr 0.98–0.999, plumbing validated; the pooled-mean drift is
  **re-confirmed on the gate-normalized facet suite** by `--calibrate`, since the
  777 floor was measured on the now-degenerate matches suite). **Pooled-mean only
  — never per-prefix** (calibration: per-prefix drift up to ~0.3; the verdict
  aggregates over prefixes, the OUTCOME_STRUCTURE replicate axis).

## Composition: two measured axes (in `r_⊥`, against calibrated floors)

Held-out split, gate-normalized targets, closes-orthogonal residual `r_⊥`:

1. **Marginal separability** — `angle(w_type0, w_color0)` in `r_⊥`. Judged
   against two `--calibrate`-measured references (review finding 2): the
   **separable ceiling** = the angle of the ground-truth `top.type`/`top.color`
   decode directions (a known-independent pair in this residual), and the
   **entangled floor** = the angle two independent facets produce *before* the
   gate fixes (the artifact magnitude). `SEPARABLE` iff angle ≥ `SEP_ANGLE`,
   which is set between those measured references with margin — not a round value.
2. **Conjunction availability** — fit `psi_both` (a) from the **2-D span**
   `[w_type0, w_color0]`, (b) from full `r_⊥`. `ΔR² = R²_full − R²_span`. Because
   type ⊥ color, `psi_both ≈ psi_type·psi_color` is a product, so `ΔR²` measures
   whether the residual carries the (type,color) **joint** linearly (conjunction
   available → `DIRECT_SUM`) or only the marginals (conjunction higher-order →
   `INTERACTION_PRESENT`). Judged against a `--calibrate`-measured **direct-sum
   floor**: the `ΔR²` of a control conjunction that *is* linearly available at the
   **same positive count** (the low-positive-count artifact, confound row 3);
   `COMP_GAP` is set above that floor. The interaction direction (residualized
   `w_both ⟂ span`) is recorded.

## Baselines and controls (anti-vacuity)

- **No-information floor:** `psi_facet` predicted by its train mean (`R²=0`); the
  decode must beat it.
- **Full-residual ceiling:** linear `R²` from the whole `r_⊥` — the best the
  probe class can do; `w_facet` is read here.
- **Degeneracy control:** `matches_type/color/both` must coincide (forced
  matching); if they *separate*, the instrument is suspect → `HARNESS_FAIL`.
- **Ground-truth facet directions (registered, eval-only — the separability
  ceiling):** belief-derived linear directions decoding the true `top.type` /
  `top.color` from `r` (Shai-style probe on ground-truth labels). Their pairwise
  angle is the **separable reference** for axis 1 (a known-independent pair in
  this residual); each is also compared to its `w_facet`. Marked ground-truth
  control, never in the observable verdict — but the angle ceiling it supplies
  *calibrates* `SEP_ANGLE` (finding 2).
- **Gate predicate `phi_closes`:** measured for every prefix (the normalizer and
  the `w_closes` partial-out direction); the `CLOSE_FLOOR` selection is reported.
- **Prefix-free anchor:** `net_return` — a non-facet predicate decodable from
  depth structure, anchoring "the method finds decode directions at all"
  (it is *not* the gate proxy; `phi_closes` is the exact gate).

## Verdict (per `(seed, layer-if-swept)`; prefixes/positions pooled; then ≥3/4 seeds)

```text
HARNESS_FAIL          — a guard fails: predicate-layer self-tests, the validity
                        gate, the OBS_DRIFT audit (|eq_obs − eq_exact| pooled >
                        OE_BAND), or the degeneracy control (matches_* separate).
                        Blocks all.
BASELINE_VACUOUS      — a facet psi has std < VAR_MIN across prefixes (no signal
                        to decode): PREDICATE_VACUOUS for that facet; report, do
                        not interpret its direction.
NOT_LINEARLY_DECODED  — a facet psi has linear R² < R2_MIN but kNN R² ≥ R2_MIN
                        (on the GATE-NORMALIZED target, so not a p_close artifact):
                        present but not affine — route to a richer probe class
                        (the V-information ladder), do not claim a direction.
SEPARABLE_DIRECTSUM   — type & color decoded (R² ≥ R2_MIN), angle(w_type,w_color)
                        in r_⊥ ≥ SEP_ANGLE (above the calibrated entangled floor),
                        AND psi_both direct-sum (ΔR² ≤ COMP_GAP, above the
                        calibrated direct-sum floor): factors separable and
                        linearly composing — the clean-composition cell, measured.
ENTANGLED_FACTORS     — type & color decoded but r_⊥ angle < SEP_ANGLE: the
                        marginal directions overlap beyond the gate artifact —
                        entangled factors (exp-40/42 redux on a new pair).
INTERACTION_PRESENT   — type & color decoded but psi_both needs an interaction
                        direction (ΔR² > COMP_GAP): the (type,color) joint is
                        higher-order in the residual; factors do not linearly
                        compose — the central positive for the composition cell.
SEED_UNSTABLE         — no ≥3/4 cross-seed majority on the headline cell.
```

### Routing (each outcome changes the next step)

| outcome | reading | routes to |
|---|---|---|
| `SEPARABLE_DIRECTSUM` | factors separable, compose linearly | the causal rung: do interchange edits on `w_color` move color without type? (the inverse-intervention goal, with the entanglement null) |
| `INTERACTION_PRESENT` | the conjunction is higher-order | characterize the interaction direction; higher-order structure is live (abstract-interpretation finding) |
| `ENTANGLED_FACTORS` | type/color overlap in the residual | the entanglement is the subject (per the phase's entangled-regime deliverable); a clean color-only edit is not well-posed |
| `NOT_LINEARLY_DECODED` | present, not affine | climb the V-information probe ladder before any direction claim |
| `BASELINE_VACUOUS` | predicate adds no decodable signal over its mean | enrich the suite/toy (the leash); the layer is validated but not earning its keep |

## Registered prediction (walled off from adjudication; credences never enter a predicate)

To be frozen at the pre-run commit, informed by the **burned calibration seed**
(its results read in full → excluded from the claim seeds). The **test** is
whether the headline cell replicates on the fresh out-of-design claim seeds at
≥3/4.

| configuration | credence | what it would teach |
|---|---|---|
| `INTERACTION_PRESENT` | ~0.40 | the conjunction needs a joint (type,color) direction — factors don't linearly compose; the composition cell's central positive. **Post gate-normalization**, so this is the genuine joint-encoding question, not the p_close artifact the review flagged (which would have inflated it) |
| `SEPARABLE_DIRECTSUM` | ~0.30 | the residual carries (type,color) as a clean separable, linearly-composing pair — routes straight to the causal inverse-intervention rung |
| `ENTANGLED_FACTORS` | ~0.20 | type/color overlap (exp-40/42 generalizes to a fresh factor pair) — entanglement is the subject |
| `NOT_LINEARLY_DECODED` / `BASELINE_VACUOUS` | ~0.10 | a facet psi isn't affinely decodable, or is vacuous — climb the probe ladder or enrich |

**Worth-running judgment:** yes — this is the first measurement relating residual
abstractions to *named* predicates with *composition*, on a 2-factor toy built
for it. Every substantive outcome routes differently (causal rung vs interaction
characterization vs entanglement subject vs probe-ladder), and the compression
reference (`k*` vs rank-1 readouts) plus the composition geometry are the
phase's central positive object regardless of which cell fires.

## Confound table — load-bearing quantities (decodability R²; angle(w_type,w_color); ΔR²)

| mechanism producing the reading | excluded by? |
|---|---|
| **multiplicative closing gate** `p_close(x)` — confounds R² (product non-linear), angle (both load on `p_close`), and ΔR² (triple product) | **EXCLUDED by gate-normalization** (`psi = E_q[phi_facet]/E_q[phi_closes]` divides `p_close` out by construction) **AND** the closes-orthogonal geometry (`w_closes` partialled out of `r` before the angle/ΔR² fits). Both are registered computations, not prose. Residual robustness: full-`r` geometry reported beside `r_⊥` |
| **probe overfit** — `w_facet` fits noise, R² spurious | held-out split (fit TRAIN, score HELD-OUT, split at the SEQUENCE level — finding 4); ridge `λ` registered; no-information floor must be beaten on held-out |
| **interaction-as-artifact** — `ΔR²` large because `psi_both` is just noisier (fewer positives), not higher-order | the `--calibrate` **direct-sum floor**: `ΔR²` of a known-linear conjunction at the **same positive count**; `COMP_GAP` sits above it. `ΔR²` is full-vs-span at matched samples; conjunction no-info floor + ceiling read in the same split |
| **entanglement-as-artifact** — small angle from a degenerate or undertrained model, not real factor coupling | validity gate (converged model) + the degeneracy control (matches_* coincide) + ground-truth control (do the true facet directions ALSO overlap? if the labels are separable but the predicate readouts aren't, that's a probe issue, not entanglement) |
| **off-manifold / wrong layer** — the chosen layer doesn't represent the facets | per-layer profile (descriptive) + decodability gate (if no layer decodes a facet, that is itself reported, not forced) |
| **estimator drift** — `E_q[phi]` mis-measured | `OBS_DRIFT` audit vs `eq_exact` (calibration: corr 0.98–0.999, plumbing validated); pooled-mean tolerance above the measured floor |

## Self-tests / controls (known-answer, before any model claim)

- Predicate-layer self-tests (`predicates._selftest`): graded masks, exact vs
  observable estimators, the seeded ctx-reader + color-scramble discrimination,
  verdict branches — all passing.
- **Degeneracy control** computed first: `matches_*` coincide on the model
  (forced matching); separation → `HARNESS_FAIL` (suite/instrument bug).
- **No-information floor & full-residual ceiling** measured for every predicate;
  the decodable threshold is read against both (the protocol's ceiling/floor
  requirement for the load-bearing R²).
- **OBS_DRIFT** audit (`eq_obs` vs `eq_exact`) on the claim sample, pooled-mean ≤
  `OE_BAND`; `--calibrate` **re-confirms the noise floor on the gate-normalized
  facet suite** (the 777 floor was on the degenerate matches suite) and the script
  **asserts seed 800 ∉ claim seeds** (burned).
- **`--calibrate` emits the two composition floors (review finding 2), as measured
  numbers, and sets the thresholds above them — not round values:** the
  **entangled-angle floor** (independent-facet angle before the gate fixes) and
  the **direct-sum ΔR² floor** (a known-linear conjunction at the matched positive
  count). `SEP_ANGLE`, `COMP_GAP` below are PROVISIONAL until this runs.
- Ridge / kNN / angle / ΔR² reducers unit-tested on synthetic planted directions
  (a known separable pair → `SEPARABLE_DIRECTSUM`; a planted product → `INTERACTION_PRESENT`);
  the gate-normalization unit-tested to recover a planted facet bit from a
  `p_close·bit` target (the finding-1 exclusion has a known-answer test).

## Registered constants (to finalize at the freeze; calibration-derived where marked)

| knob | value | note |
|---|---|---|
| checkpoint | fresh `out/colored_dyck2-sNNN` per claim seed | `train.py --process colored_dyck2`; config below; **seed 777 (calib) is burned**, not a claim seed |
| train config | L4 d64 seq_len32 m3, steps 8000 | matches the calibration checkpoint; validity gate ≤ 0.005 |
| claim seeds | `{801, 802, 803, 804}` | 4 fresh out-of-design; ≥3/4 majority; calibration seed `800` burned |
| `PROBE_LAYER` | calibration-selected (default block 2 of 4) | per-layer profile descriptive; headline at the registered layer |
| read points `t` | `{8, 12, 16}` | determined-ctx prefixes pooled (calibration: ~180–200 determined per t per 250 seqs) |
| horizon `m` | `3` | **deviation from decision 6** (m=3 "too short"); justified by calibration — tops close fast, matches-mean 0.811 → non-vacuous. `--calibrate` must report per-`psi` std at m=3 vs `VAR_MIN`; 0.811 is also the finding-1 gate magnitude |
| predicates | `phi_type0`, `phi_color0`, `phi_both00`, **gate `phi_closes`**; control `net_return`; degeneracy `matches_*` | facet-value suite, gate-normalized (NOT matches — degenerate) |
| `CLOSE_FLOOR` | `0.30` | keep prefixes with `E_q[phi_closes] ≥` this (ratio stability); calibration: mean ≈0.81, so most qualify |
| `TAU` (decode error) | `0.03` | pooled-mean on `psi`; above the ~0.01 floor; re-confirmed on the gate-normalized suite by `--calibrate` |
| `R2_MIN` | `0.50` | linear-decodable cut (exp-29 precedent); kNN `R²` ≥ this = "present" |
| `SEP_ANGLE` | **PROVISIONAL `45°`** | set by `--calibrate` between the measured entangled-floor and ground-truth-pair ceiling, with margin (finding 2) |
| `COMP_GAP` (ΔR²) | **PROVISIONAL `0.10`** | set by `--calibrate` above the measured direct-sum floor (known-linear conjunction at matched positives) |
| `VAR_MIN` | `0.05` | predicate non-vacuity (std of `psi` across prefixes) |
| `OE_BAND` | `0.02` | `OBS_DRIFT` audit, pooled-mean |
| ridge `λ` / kNN `k` | `1e-2` / `10` | the bounded probe class |
| eval sequences / split | `≥ 3000` seqs, split **by sequence** 50/50 | yields ≥ ~3500 determined-ctx prefixes per split over `{8,12,16}` (reconciles finding 4: the ~600 figure was per-250-seqs); disjoint at the sequence level (no prefix leakage); `d=64` so each split ≫ 20·d |
| `SEED_MAJORITY` | `3` of 4 | cross-seed headline |

(Thresholds marked "finalize from calibration" are set by the script's
`--calibrate` on burned seed 800, like exp-43; seed 800 is then excluded.)

## Reuse vs single-use

- **Import:** `predicates.py` (graded layer, ctx-reader, `eq_obs_from_model`,
  `ctx_along`, `eq_exact_seeded`), `processes.colored_dyck2`, `train.py`,
  `expcommon.load_model`/`validity_gate`, `midstream.chain_probs`,
  `discover` (`cegar_loop`/`mined_direction` for the `k*` reference;
  `principal_angles_deg`), `abstraction.PCAAbstraction`/`center_by_position`. The
  predicate-targeted readout follows exp-29's `predicate_targeting.py` approach.
- **Rung-specific** (`scripts/predicate_sufficiency/exp45_*.py`): the facet-value
  template `tmpl_next_close_is(facet, value)` and the gate `phi_closes`
  (seeded locate; `value=None` = "pops") — promote to `predicates.py` only if a
  second rung needs them; **gate-normalization** (`psi`) + `CLOSE_FLOOR`
  selection + the `w_closes` partial-out; the ridge/kNN decode + sequence-level
  held-out scoring; the angle/ΔR² composition reducers and their calibrated
  floors; `cell_verdict`; `--calibrate`.

## Non-goals

- **No causal claim.** Decodability is correlational (the info is THERE in the
  probe class). Whether an interchange edit on `w_color` *moves* color alone —
  with the entanglement null (decision 3) — is the **next** rung, explicitly
  deferred. `SEPARABLE_DIRECTSUM`/`INTERACTION_PRESENT` are decode-geometry
  verdicts, not "I moved color alone."
- **Probe-class indexed.** All verdicts are relative to affine-ridge (+ kNN as
  the present reference). `NOT_LINEARLY_DECODED` is bounded to "not affine," never
  "not represented"; a higher probe class is the V-information ladder, not a
  refutation.
- **Toy-indexed.** Colored Dyck-2, the registered config/layer/positions/horizon.
  No transfer claim until the toy ladder runs it (the generalization lever).
- **No privileged decomposition / no inherited-top prefixes** (latent-observability
  is the latent-toy question). Ground-truth facet directions are an eval-only
  control, never in the observable verdict.
```
