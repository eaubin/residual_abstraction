# Experiment 45 — Predicate-sufficient residual directions and the composition of binding factors on colored Dyck-2

**Status: DRAFT (pre-registration, pre-review).** Not yet committed for the
pre-registration review pause; the runnable script (`scripts/predicate_sufficiency/exp45_*.py`)
is the next artifact and must be committed beside this writeup before the first
claim run, per `EXPERIMENT_REVIEW_PROTOCOL.md`. No claim seed has been run.

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

## The corrected predicate suite (a calibration finding, load-bearing)

The intuitive binding predicate "the next close **matches** the prefix top" is
**degenerate for composition**: colored-Dyck matching is grammar-forced, so
whenever the top closes in-window it matches on **both** type and color at once.
The calibration (seed 777, below) confirmed `matches_type` = `matches_color` =
`matches_both` numerically (all exact-mean 0.811, identical across prefixes).
Identical predicates cannot separate factors. So the registered suite is
**facet-VALUE** predicates — the close that pops the seeded top has a *specific*
facet value — whose values key on **different** facets:

- `phi_type0(c; ctx)` — the close popping the seeded top has **type 0**. E_q
  depends on the top's **type** (≈ P(close in-window) if top.type=0, else ≈0).
- `phi_color0(c; ctx)` — …has **color 0**. Depends on the top's **color**.
- `phi_both00 = phi_type0 ∧ phi_color0` — …has **type 0 AND color 0**. Depends on
  the **joint** (type, color). This is the composition target.

Type and color are independent in the generator (`type_split`, `color_split`
independent), so `phi_type0` and `phi_color0` are genuinely different functions
(correlated only through the shared "closes-in-window" factor). `matches_*` is
**retained as a registered DEGENERACY control** — the three facets must coincide,
confirming the forced-matching reading and that the instrument is not inventing
separation.

## The question

```text
On colored Dyck-2, at the registered probe layer/positions: from which residual
directions are the named binding predicates' expected scores E_q[phi]
(observable, from the model's own completions) decodable by a bounded LINEAR
probe — and do the binding FACTORS compose? Specifically (a) are the type and
color readout directions SEPARABLE (orthogonal) or ENTANGLED (aligned), and
(b) is the conjunction phi_both decodable from the span of the two marginal
directions (a linear DIRECT SUM) or does it require an INTERACTION direction
(the conjunction is higher-order, the factors do not linearly compose)?
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
  indexed to it). Fit `E_q[phi] ≈ w_phi · r + b` on TRAIN prefixes, score on a
  disjoint HELD-OUT split (decision 4: held-out evaluation). The fitted **readout
  direction `w_phi`** is the rank-1 residual abstraction for `phi`.
- **kNN R²** (registered `k`) is reported as the **present-but-not-linear**
  reference (exp-1/exp-29 interpreter gap): high kNN R² with low linear R² means
  the predicate is *there* but not affinely decodable, routing to a richer probe
  — not "absent."
- **Decodable iff** held-out linear `R²_phi ≥ R2_MIN` **and** pooled-mean
  `|E_q[phi] − decode|` ≤ `TAU`. `TAU` is set above the estimator noise floor
  (calibration: pooled-mean obs↔exact drift ~0.005–0.009 on the matches suite;
  re-confirmed on the facet-value suite by the script's `--calibrate` on the
  burned seed). **Pooled-mean only — never per-prefix** (calibration: per-prefix
  drift up to ~0.3; the verdict aggregates over prefixes, the OUTCOME_STRUCTURE
  replicate axis).

## Composition: two measured axes

Computed on the held-out split, both from the fitted directions:

1. **Marginal separability** — the principal angle `angle(w_type0, w_color0)`.
   Near 90° = SEPARABLE factors; small = ENTANGLED (the exp-40/42 phenomenon on a
   fresh factor pair, now in the composition cell).
2. **Conjunction availability** — fit `phi_both00` (a) from the **2-D span**
   `[w_type0, w_color0]` only, and (b) from the **full** residual. `ΔR² =
   R²_full − R²_span`. Small `ΔR²` (≤ `COMP_GAP`) = the conjunction lives in the
   marginal span → **DIRECT SUM**. Large `ΔR²` = an **INTERACTION direction**
   outside the marginals is needed (the conjunction `[type=0 ∧ color=0]` is a
   product — not linear in separate factor directions — unless the residual
   encodes the joint (type,color) linearly). The **interaction direction** itself
   (residualized `w_both` ⟂ span) is recorded.

## Baselines and controls (anti-vacuity)

- **No-information floor:** `E_q[phi]` predicted by its train mean (`R²=0`); the
  decode must beat it.
- **Full-residual ceiling:** linear `R²` from the whole `r` — the best the probe
  class can do; `w_phi` is read here.
- **Degeneracy control:** `matches_type/color/both` must coincide (forced
  matching); if they *separate*, the instrument is suspect → `HARNESS_FAIL`.
- **Ground-truth control (registered, eval-only):** the belief-derived linear
  directions decoding the true `top.type` / `top.color` from `r` (Shai-style
  probe on ground-truth labels). Compared (principal angle) to `w_type0` /
  `w_color0` — does the predicate-fit direction align with the
  ground-truth-fit facet direction? Marked **ground-truth control**, never in the
  observable verdict.
- **Prefix-free anchor:** `net_return` (non-binding) — a predicate that should be
  decodable from depth structure, anchoring "the method finds decode directions
  at all."

## Verdict (per `(seed, layer-if-swept)`; prefixes/positions pooled; then ≥3/4 seeds)

```text
HARNESS_FAIL          — a guard fails: predicate-layer self-tests, the validity
                        gate, the OBS_DRIFT audit (|eq_obs − eq_exact| pooled >
                        OE_BAND), or the degeneracy control (matches_* separate).
                        Blocks all.
BASELINE_VACUOUS      — a binding predicate's E_q[phi] has std < VAR_MIN across
                        prefixes (no signal to decode): PREDICATE_VACUOUS for that
                        phi; report, do not interpret its direction.
NOT_LINEARLY_DECODED  — a binding predicate has linear R² < R2_MIN but kNN R² ≥
                        R2_MIN: present but not affine — route to a richer probe
                        class (the V-information ladder), do not claim a direction.
SEPARABLE_DIRECTSUM   — type & color decoded (R² ≥ R2_MIN), angle(w_type,w_color)
                        ≥ SEP_ANGLE, AND phi_both direct-sum (ΔR² ≤ COMP_GAP):
                        the factors are separable and compose linearly — the
                        clean-composition cell, measured.
ENTANGLED_FACTORS     — type & color decoded but angle < SEP_ANGLE: the marginal
                        directions overlap — entangled factors (exp-40/42 redux on
                        a new pair).
INTERACTION_PRESENT   — type & color decoded but phi_both needs an interaction
                        direction (ΔR² > COMP_GAP): the conjunction is higher-
                        order; factors do not linearly compose — the central
                        positive for the composition cell.
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
| `INTERACTION_PRESENT` | ~0.45 | the conjunction needs a joint (type,color) direction — factors don't linearly compose; the composition cell's central positive |
| `SEPARABLE_DIRECTSUM` | ~0.25 | the residual carries (type,color) as a clean separable, linearly-composing pair — routes straight to the causal inverse-intervention rung |
| `ENTANGLED_FACTORS` | ~0.20 | type/color overlap (exp-40/42 generalizes to a fresh factor pair) — entanglement is the subject |
| `NOT_LINEARLY_DECODED` / `BASELINE_VACUOUS` | ~0.10 | a binding predicate isn't affinely decodable, or is vacuous — climb the probe ladder or enrich |

**Worth-running judgment:** yes — this is the first measurement relating residual
abstractions to *named* predicates with *composition*, on a 2-factor toy built
for it. Every substantive outcome routes differently (causal rung vs interaction
characterization vs entanglement subject vs probe-ladder), and the compression
reference (`k*` vs rank-1 readouts) plus the composition geometry are the
phase's central positive object regardless of which cell fires.

## Confound table — load-bearing quantities (decodability R²; angle(w_type,w_color); ΔR²)

| mechanism producing the reading | excluded by? |
|---|---|
| **shared "closes-in-window" factor** inflates angle/decode agreement (both predicates ride depth) | `net_return` anchor + the ground-truth `top.type`/`top.color` control: separability is judged on the FACET directions after the shared depth component, and the angle is between the facet-keyed readouts, not the raw scores |
| **probe overfit** — `w_phi` fits noise, R² spurious | held-out split (fit TRAIN, score HELD-OUT); ridge `λ` registered; no-information floor must be beaten on held-out |
| **interaction-as-artifact** — `ΔR²` large because `phi_both` is just noisier (fewer positives) | the conjunction's no-information floor and ceiling are read in the same split; `ΔR²` is full-vs-span at matched samples; degeneracy control bounds the suite's self-consistency |
| **entanglement-as-artifact** — small angle from a degenerate or undertrained model, not real factor coupling | validity gate (converged model) + the degeneracy control (matches_* coincide) + ground-truth control (do the true facet directions ALSO overlap? if the labels are separable but the predicate readouts aren't, that's a probe issue, not entanglement) |
| **off-manifold / wrong layer** — the chosen layer doesn't represent binding | per-layer profile (descriptive) + decodability gate (if no layer decodes a binding predicate, that is itself reported, not forced) |
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
  `OE_BAND` — measurement repair branch before any decode claim.
- Ridge / kNN / angle / ΔR² reducers unit-tested on synthetic planted directions
  (a known separable pair → `SEPARABLE_DIRECTSUM`; a planted product → `INTERACTION_PRESENT`).

## Registered constants (to finalize at the freeze; calibration-derived where marked)

| knob | value | note |
|---|---|---|
| checkpoint | fresh `out/colored_dyck2-sNNN` per claim seed | `train.py --process colored_dyck2`; config below; **seed 777 (calib) is burned**, not a claim seed |
| train config | L4 d64 seq_len32 m3, steps 8000 | matches the calibration checkpoint; validity gate ≤ 0.005 |
| claim seeds | `{801, 802, 803, 804}` | 4 fresh out-of-design; ≥3/4 majority; calibration seed `800` burned |
| `PROBE_LAYER` | calibration-selected (default block 2 of 4) | per-layer profile descriptive; headline at the registered layer |
| read points `t` | `{8, 12, 16}` | determined-ctx prefixes pooled (calibration: ~180–200 determined each) |
| horizon `m` | `3` | calibration: m=4 buys nothing for these predicates (tops close fast) |
| predicates | `phi_type0`, `phi_color0`, `phi_both00`; control `net_return`; degeneracy `matches_*` | facet-value suite (NOT matches — degenerate for composition) |
| `TAU` (decode error) | `0.03` | pooled-mean; above the ~0.01 floor; re-confirmed on the facet suite by `--calibrate` |
| `R2_MIN` | `0.50` | linear-decodable cut (exp-29 precedent); kNN `R²` ≥ this = "present" |
| `SEP_ANGLE` | `45°` | ≥ → separable; < → entangled (finalize from calibration) |
| `COMP_GAP` (ΔR²) | `0.10` | ≤ → direct sum; > → interaction direction (finalize from calibration) |
| `VAR_MIN` | `0.05` | predicate non-vacuity (std of `E_q[phi]`) |
| `OE_BAND` | `0.02` | `OBS_DRIFT` audit, pooled-mean |
| ridge `λ` / kNN `k` | `1e-2` / `10` | the bounded probe class |
| TRAIN/HELD-OUT prefixes | `≥ 1500 / ≥ 1500` per (t pooled) | disjoint; V-information held-out |
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
  template `tmpl_next_close_is(facet, value)` (seeded locate + fixed-value check —
  promote to `predicates.py` only if a second rung needs it), the ridge/kNN
  decode + held-out scoring, the angle/ΔR² composition reducers, `cell_verdict`,
  and `--calibrate`.

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
