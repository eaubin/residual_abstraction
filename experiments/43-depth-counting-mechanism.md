# Experiment 43 — Localizing the depth-counting mechanism: which heads recompute graded depth from the bracket-embeddings?

**Status: PRE-REGISTRATION DRAFT — not yet run.** Predictions to be frozen (committed)
before the first claim run, with the script, per `EXPERIMENT_REVIEW_PROTOCOL.md`.

**Reworked after a calibration finding (see "What calibration forced" below).** This rung was
first drafted as *carrier* localization — "which architecture-given components carry the
distributed graded depth exp 38 found." A calibration run of the component enumerator
**pre-answered that question and reframed it**: under the forced-close instrument, *no* internal
prefix-position write carries graded depth — the depth signal is **recomputed from the prefix
token embeddings** at the readout. So the live question is not *where depth is stored* (it
isn't, internally) but *where it is computed*: **which attention heads, at the readout
positions, aggregate the bracket-embeddings into the graded-depth conditional?** The carrier
draft is preserved in `git` history (`43-depth-carrier-localization.md`).

This is the state-localization phase's **dynamics rung** (the old `L4` — "which heads move/compute
the facet"), and the **first experiment in the phase to enumerate architecture-given components**
(37–42 worked on the full residual at a position or on additive steering directions). The
component-write enumerator (the phase doc's deferred "novel reusable core") is built and
self-tested in `localize.py`.

## What calibration forced (the premise this rung now rests on)

The component-write enumerator (`localize.chain_probs_components`; emb + per-layer attention
heads + MLP, position-resolved, exact interchange) was run on `top_type`-matched `depth_triples`
pairs at horizon `k=1` (seed 700, `t∈{12,20}`, 256 pairs; `scripts/localization/exp43_calibration.py`).
Transport `f = (P−C)/(S−C)` of the forced-close conditional:

| splice | `f` | reading |
|---|---|---|
| all components incl. embedding | 1.00 | sanity — reproduces the source (completeness identity) |
| **source embeddings only** (internal recomputes) | **1.00** | the prefix tokens alone drive full source depth |
| `emb + block 0` (= exp-38's `LAYER=1` residual) | 1.00 | reproduces exp-38's `f_full` ✓ |
| **all internal writes, embedding clean** | **0.13 / 0.24** | transplanted internal state recovers ~none of the depth |
| per layer, internal-only (emb clean) | 0.09 → 0.04 → 0.03 → 0.00 | declines to zero by the last layer |

**Reading (the premise).** Graded depth at the forced-close conditional is **recomputed from the
prefix token embeddings every step; it is not stored in any internal prefix-position write** in a
way the conditional uses. This **sharpens exp 38**: its `f_full`≈0.83–0.97 was patched at
`LAYER=1`, whose residual *includes* the token embeddings — and the embeddings are the entire
transporting term. So 38's "carried, distributed, recency-weighted" carrier **is the bag of
bracket-embeddings spread across prefix positions** — the exp-4/5 "summary recomputed from
tokens, not propagated as state" property, now pinned at component granularity. (Calibration is
seed 700 only; this rung **replicates it across seeds/positions as a registered substrate gate**,
and on conclusion back-annotates exp 38 / `ASSUMPTIONS.md`, per the propagate-the-resolution
protocol step.)

The recomputation is done by the model reading the prefix bracket-embeddings via attention **at
the readout positions** (the forced-close continuation positions re-attend to the prefix). That
is where the depth computation lives, and where this rung localizes.

## The question

```text
Given that graded depth is recomputed from the prefix bracket-embeddings (the premise, gated
below): WHICH attention heads, at WHICH readout positions, aggregate those embeddings into the
forced-close graded-depth conditional — is the counting CONCENTRATED in a few heads (a localized
counting mechanism) or DISTRIBUTED across many, and if a small set suffices is it a genuine
locus or REDUNDANT (each sufficient alone, none necessary)?
```

A localized counting mechanism is the sharp positive (intervene/refine at those heads). A
distributed one is the expected-difficulty (superposition) outcome at the mechanism level. The
**redundancy** axis is separable here via the necessity arm (38 could not separate it spatially).

## Premise gate (registered precondition — runs before the localization claim)

The localization is only well-posed if the premise holds on the registered seeds. Gate, per
`(seed, k)`:

- `f_emb` (source-embeddings-only splice, all positions ≤ t) **≥ `EMB_MIN`** (the tokens drive
  depth), AND
- `f_internal` (all internal writes, embedding clean) **≤ `INT_MAX`** (internal prefix writes do
  not carry it), AND
- necessity-of-internal: reverting all internal writes to clean while holding embeddings source
  drops transport by **< `INT_NEC_MAX`** (internal writes are not necessary).

`EMBEDDING_RECOMPUTED` (gate PASS) → proceed to localize the counting mechanism (below). If the
gate **fails** (some internal component *does* store depth), the rung **reverts to the carrier
question** (the `git`-preserved draft): a stored-depth result is itself a finding and reroutes to
carrier localization. The gate is thus a real fork, not a formality.

## Instrument: the deterministic forced-close conditional, read at the readout positions

38's `cr_cond` reads the forced-close conditional from the `V^m` grid marginal — but head outputs
at *grid* continuation positions are ill-defined for patching (they vary across the grid). So this
rung uses the **deterministic** equivalent: append the **oracle-legal forced closers** `w_{t+1..t+k}`
to the prefix (one continuation), read `P(close) = q2+q3` at position `t+k`. This equals `cr_cond`'s
quantity for the closer continuation, but fixes positions `t+1..t+k`, so head outputs at the
**readout window** `{t, t+1, …, t+k}` are well-defined and patchable. Pairs, transport, and the
oracle-endpoint audit are unchanged from 38.

- **Pairs:** `depth_triples` (clean/same-depth-floor at depth `lo`, source at depth `hi`,
  `top_type`-matched), horizons `k ∈ {1,2}` (depth contrasts (1,2) and (2,3)). Reused.
- **Transport:** `f = (P−C)/(S−C)`, pooled over pairs with oracle gap `≥ GAP_MIN`; `S` audited
  against the exact oracle (`OBS_DRIFT`).

## Units: readout-window attention-head outputs (+ MLP; prefix as the ≈0 control)

```text
unit u = (layer ℓ ∈ {0..3}, sublayer s ∈ {head h∈{0..3}, mlp}, readout position p ∈ {t..t+k})
```

Interchange-splice unit `u`'s **output** from the depth-`hi` run into the depth-`lo` run at its
position `p`, continue the forward pass, read the conditional at `t+k`. Heads are resolved from
the start (the cross-position aggregators are the point). **Prefix-position units (`p ≤ t`) are
included as the calibration-established ≈0 control** — the localization headline is the readout
window; a non-trivial prefix-position counter would itself be a finding. The position axis stays
resolved (per-position × per-head map). The **direction** level stays deferred (L3).

This reuses the `localize.py` enumerator (`record_component_writes` / `chain_probs_components`,
position-resolved splicing, self-tested) — the only new code is the deterministic forced-closer
sequence builder and reading `P(close)` at `t+k`.

## Construct: two attributions cross-checked, then a cumulative curve (repointed to readout heads)

Per unit `u` (both via the conditional transport `f`):

1. **Sufficiency** `suff(u)` — splice **only** `u` (hi→lo); transport toward source.
2. **Necessity** `nec(u)` — splice **all** readout-window units **except** `u`; `nec(u)` =
   (all-window transport) − this.

The **suff × nec disagreement is the redundancy measurement** (centerpiece, not a flagged
confound):

| | necessary (high `nec`) | not necessary (low `nec`) |
|---|---|---|
| **sufficient** (high `suff`) | **genuine counting locus** | **redundant counter** (backed up) |
| **not sufficient** (low `suff`) | **distributed-essential** (part of a needed set) | **irrelevant** |

3. **Cumulative sufficiency curve** — rank readout-window units by `suff`, splice top-`j`
   cumulatively, transport vs `j`. Concentration = how few reach the ceiling. **Matched control
   (required):** equal-count **random readout-window units** at each `j`; concentration is
   ranked-vs-random, never the raw curve (38's injected-mass lesson).
4. **Specificity (cross-facet):** each counter's `top_type` drag (ratio component of its
   `(q2,q3)` effect) vs its depth transport; a counter that also moves `top_type` reads
   `NONSPECIFIC`.
5. **Layer/position profile (descriptive → folded):** where in (layer × readout-position) the
   counting concentrates — reported, folded into the verdict only via the resolved map.

**Ceiling** for the cumulative curve = all readout-window units spliced (not the full source
residual — the embeddings are deliberately *not* a unit here; this rung localizes the
computation reading them, the premise gate having established the embeddings are the input). The
**all-window ceiling vs the embedding ceiling** gap is reported (how much of the recomputation
the readout-window heads alone reconstruct).

## Verdict (per `(seed, k)`; readout-units reduced, then ≥3/4 seeds; horizon axis kept)

```text
HARNESS_FAIL        — a guard fails: no-op not bit-exact; the completeness identity (all
                      components incl. emb = source) breaks; the planted-head reference not
                      recovered. Blocks all.
OBS_DRIFT           — conditional-vs-oracle endpoint gap > OE_BAND.
PREMISE_FAIL        — the premise gate fails (some internal write stores depth): NOT a counting
                      result — reroute to carrier localization (the git-preserved draft).
LOCALIZED_COUNTER   — ranked cumulative reaches the all-window ceiling within SAT_K readout units
                      (beating random by LOCUS_MARGIN), specific, and the top units are genuine
                      loci (suff high AND nec high, |suff−nec| < REDUND_GAP): depth counted by a
                      small set of heads.
REDUNDANT_COUNTER   — small-SAT_K saturation AND specific, BUT top units are redundant
                      (suff high, nec low; suff−nec ≥ REDUND_GAP): a small set suffices, backed up.
DISTRIBUTED_COUNTER — no small set reaches the ceiling; the ranked curve ramps gradually:
                      counting spread across heads (the expected difficulty, located).
NONSPECIFIC         — the counters also move top_type above SPEC_MARGIN: not depth-specific heads.
SEED_UNSTABLE       — no ≥3/4 cross-seed majority, or a horizon has no qualifying readout units.
```

### Routing (each outcome changes the next step)

| outcome | reading | routes to |
|---|---|---|
| `LOCALIZED_COUNTER` | a few heads do the counting | refine to head-directions (L3); intervene on the counting heads — the well-posed ICB |
| `REDUNDANT_COUNTER` | backup counting heads — the robustness cell, measured | a path / multi-head intervention; the redundancy thread |
| `DISTRIBUTED_COUNTER` | counting is superposed across heads, located | a distribution-handling phase; do not force a clean handle |
| `NONSPECIFIC` | counting heads also carry type | the counting is entangled with `top_type` — back to characterization |
| `PREMISE_FAIL` | depth IS stored internally after all | revert to carrier localization (the git-preserved draft) |

## Registered prediction (walled off from adjudication; credences never enter a predicate)

Frozen at `<commit-before-run>`. Informed by the **design seed (700)** calibration; the **test**
is whether `LOCALIZED_COUNTER` and the `(3,attn,3)`-dominated locus replicate on the unseen seeds
701–703 and hold across positions at ≥3/4.

| configuration | credence | what it would teach |
|---|---|---|
| `LOCALIZED_COUNTER` (both k) (**predicted**) | ~0.55 | a few late-layer heads (design seed: `(3,attn,3)` at the readout, genuine + specific) count depth — the sharp positive; routes to intervention/refinement (L3) |
| `LOCALIZED_COUNTER` (one k only) | ~0.15 | the localization holds at one depth contrast but the other is distributed/underpowered |
| `DISTRIBUTED_COUNTER` (≥1 k) | ~0.15 | counting spreads across many heads on the unseen seeds — superposition at the mechanism level |
| `REDUNDANT_COUNTER` (≥1 k) | ~0.05 | backup counting heads; the suff×nec gap lights up (design seed showed genuine loci, not redundant) |
| `PREMISE_FAIL` / `NONSPECIFIC` / `SEED_UNSTABLE` / guard | ~0.10 | the premise does not replicate, counters co-carry `top_type`, or the curves are unstable at ≥3/4 |

**Worth-running judgment:** yes — the premise gate corrects the exp-38 record either way (PASS
confirms recompute-from-embeddings across seeds; FAIL is itself a finding), and the substantive
counting outcomes route differently. The design-seed signal is strong (`LOCALIZED_COUNTER` at
t∈{8,12,20}, both horizons), so the dominant value is the **replication test** on 701–703 and the
identity/stability of the located heads — the first experiment to put depth on the architecture,
pointed where calibration showed the computation is (the readout, not the prefix store).

## Calibration validated the instrument (design seed 700; not a claim)

`--calibrate` (seed 700, `t∈{8,12,20}`, both horizons) confirmed the reframe is well-posed: the
**premise gate PASSes** everywhere (`f_emb`=1.00, `f_internal`=0.06–0.24, `int_nec`=0.00), and —
unlike prefix-position writes (≈0) — **readout-window head splices transport depth**: the ranked
cumulative reaches ~0.83–0.94 of the all-window ceiling within 3–4 units (random floor <0.2),
the top unit is consistently **`(3,attn,3)` at the readout position `t+k`** (`suff` 0.22–0.43), the
top units are **genuine loci** (`suff−nec` ≤ 0.12, not redundant), and `top_type` **drag ≡ 0.00**
(depth-specific). This shaped the thresholds and the prediction above; the claim run tests
replication on 701–703.

## Confound table — load-bearing quantities (premise transport split; cumulative saturation; suff−nec gap)

| mechanism producing the reading | excluded by? |
|---|---|
| **premise artifact** — `f_internal`≈0 because the transplanted internal-write residual is off-manifold, not because internal writes are uninformative | the **necessity** arm of the gate (reverting internal from the *source* side, an on-manifold perturbation, also shows no drop) + `f_emb`≈1 (an on-manifold source run). The two agree, so the premise is not a one-sided off-manifold artifact |
| **injected-signal mass** — more readout units = more source signal regardless of locus | the equal-count **random readout-unit** cumulative control; concentration is ranked-vs-random at matched `j` |
| **off-manifold splice** — a head output atop the otherwise-`lo` residual is off-distribution | partly — oracle endpoint audit bounds the endpoints; the planted-head reference and random control are read in the same regime, keeping the *relative* shape interpretable; absolute mid-curve values are not (38's bound) |
| **redundancy masquerading as a null** (single-unit `suff` under-reads a backed-up counter) | **measured** by the necessity arm + suff×nec 2×2 (`REDUNDANT_COUNTER` vs `DISTRIBUTED_COUNTER`); fully-symmetric redundancy needing all units reads `DISTRIBUTED_COUNTER`, bounded to "not concentrated," never "not computed" |
| **importance without specificity** — a head that moves the whole close distribution | cross-facet `top_type` drag (sum vs ratio); a non-specific counter reads `NONSPECIFIC` |
| **wiring error** in the per-head splice | the **completeness identity** self-test (all components incl. emb = source, exact); bit-exact single-unit reconstruction; no-op bit-exact |

## Self-tests / controls (known-answer, before any model claim)

- **Completeness identity:** splicing every component (emb + all heads + all MLPs) reproduces the
  source conditional exactly (built + passing in `localize._selftest`).
- **No-op** (empty splice) reproduces the per-head unpatched run bit-exact; single-unit splice is
  a bit-exact reconstruction; position-subset splicing matches (all positions = list form, no
  positions = no-op) — all passing in `localize._selftest`.
- **Random readout-unit cumulative curve** = the no-locus **floor** (the key measured
  reference, 38's random-placement role); the all-readout-window splice = the **ceiling**.
  Detectability of a small carrier set is shown **empirically** by the ranked curve clearing the
  random floor (calibration: ranked ≥0.83·ceiling at `j=3` over a <0.04 floor). *(A frozen
  planted-head anchor was dropped: forcing all-but-one component to clean recorded writes
  suppresses the downstream recomputation that amplifies a live single-unit effect — calibration
  showed frozen 0.07 vs live 0.24 for the same head — so the frozen construction underestimates
  and is the wrong reference. The live random-floor/ceiling pair is the right one.)*
- **No-difference** (same-depth source) → `suff ≈ 0` (gap-filtered out); the `top_type`-drag
  control = 0.00 for depth-specific units.
- The verdict-branch logic, the reducers, and the top-`k` closer-matching (the k≥2 control) are
  unit-tested.

## Registered constants (to finalize at the freeze; calibration-derived where marked)

| knob | value | note |
|---|---|---|
| checkpoint | `out/dyck2-L4` | exp-19 config; `require_expected_config` halts on mismatch |
| read points `t` / horizons `k` / seeds | `{8,12,16,20}` / `{1,2}` / `{700–703}` | 38's positions/horizons/seeds |
| units | `(layer 0–3) × (head 0–3, mlp) × (readout position t..t+k)`; prefix `p≤t` as the ≈0 control | position-resolved; heads from the start |
| `GAP_MIN` / `OE_BAND` | `0.10` / `0.10` | inherited from 38 |
| `EMB_MIN` / `INT_MAX` / `INT_NEC_MAX` | `0.80` / `0.30` / `0.20` | premise gate; calibration: `f_emb`=1.0, `f_internal`=0.06–0.24, `int_nec`=0.00 — cuts sit well inside |
| `SAT_FRAC` / `SAT_K` | `0.80` / `5` | localized iff ranked cumulative ≥ `SAT_FRAC`·ceiling within `SAT_K` units; calibration: ranked ≥0.83·ceiling by `j=3` |
| `LOCUS_MARGIN` | `0.15` | ranked must beat random by this; calibration margins ≥0.77 at `j=3` |
| `REDUND_GAP` | `0.25` | `suff−nec ≥` this → redundant; calibration genuine-locus gaps ≤0.12 |
| `SPEC_MARGIN` | `0.15` | `top_type` drag above this → nonspecific; calibration drag ≡0.00 (anchored to 40's scale) |
| pairs (`MIN_PAIRS` / `PAIR_CAP`) / `SEED_MAJORITY` | `≥256` / `1024` per (t×k) / `3` | top-`k`-matched (the k≥2 closer control); ≥3/4 seeds |

**Thresholds frozen from the calibration reference** (`--calibrate`, seed 700 = **design seed**,
`t∈{8,12,20}`, both horizons): all-readout-window ceiling `f`=1.00; random-component floor <0.2
for `j≤5`; ranked cumulative ≥0.83·ceiling by `j=3` (margin >0.77); top units genuine
(`suff−nec`≤0.12); `top_type` drag ≡0.00. The claim run adds seeds **701–703** as the
out-of-design test (the 38/42 design-seed precedent).

## Reuse vs single-use

- **Import** (`localize.py`): the **component-write enumerator** (`record_component_writes`,
  `chain_probs_components`, position-resolved interchange splicing, `component_ids`) — built and
  self-tested here for the phase; `cr_cond`, `depth_triples`, `make_Xc`, `q_at`,
  `require_expected_config`, the checkpoint/guard contract. 38 is frozen.
- **Rung-specific** (`scripts/localization/exp43_*.py`): the deterministic forced-closer
  instrument (append oracle-legal closers, read `P(close)` at `t+k`), the premise gate,
  `suff`/`nec` attribution at readout positions, the cumulative ranked + random curves, the
  suff×nec classifier, `cell_verdict`, reducers. `exp43_calibration.py` (the reference run) shares
  these. Promoted to `localize.py` only when a second rung needs them.

## Non-goals

- No direction/head-subspace claim (L3). No real-LLM claim; vehicle fixed to the registered
  Dyck-2 checkpoint, layers, positions, horizons, and patch family.
- No privileged decomposition: components are architecture-given; importance is interchange
  transport + specificity, confounds (redundancy measured, off-manifold bounded) named not
  eliminated. A `DISTRIBUTED_COUNTER` is bounded to "counting not concentrated in a small head
  set," never "not computed."
- The premise establishes depth is **not internally stored** under this instrument; it does **not**
  claim the model carries no propagated state of any kind (the m≥2 conditional is one readout).
- No same-vs-different-parts m=1 summary verdict (that is L2; `top_type` is its remaining live
  target — depth does not localize to internal parts, so the same/different question is moot for
  depth).
