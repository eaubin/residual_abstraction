# Experiment 42 — Guarded directional specificity: is the depth→`top_type` drag geometric or representational? — DRAFT

**Status: DRAFT pre-registration** (not yet registered, not implemented, not run). The
question, construct, verdict partition, and confounds below are proposed; the **two
decisions flagged in "Decisions needed"** must be settled before this is frozen and the
script written. Builds directly on exp 40 (`CROSS_DRAG`) and its geometry pre-check
(`scripts/localization/exp42_geometry_precheck.py`, `out/exp42_geometry_precheck.txt`).
State-localization phase — the **directional-binding characterization** rung, the
registered route out of exp 40's `CROSS_DRAG` ("characterize the binding").

## Why this, and what exp 40 left open

Exp 40 found an **asymmetric `CROSS_DRAG`**: the `top_type` difference-direction steers
cleanly, but the `depth` difference-direction drags `top_type` (stable at `k=2`, 4/4;
borderline at `k=1`), drag larger at the deeper contrast. It could **not** separate the
two explanations of that drag:

- **(geometric)** `v_depth` and the type readout's directions are merely **non-orthogonal**
  in the residual, so steering depth bleeds into the type readout — a removable basis
  artifact; or
- **(representational)** the depth contrast is genuinely **bound into** the type-carrying
  subspace — moving depth *requires* moving the type readout.

This is the causal-vs-geometric confound the `docs/OUTCOME_STRUCTURE.md` lens flags as
deserving a targeted instrument rather than another descriptive map — and it is the
sharpened form of the `pstack` one-variable-vs-broad-replacement question (ledger row 37).

## The question

```text
Does a depth-steering direction exist that transports graded depth to its full-
replacement ceiling while leaving the top_type READOUT invariant — i.e. is there a
type-readout-orthogonal depth handle?
```

- **SEPARABLE** → such a handle exists: the exp-40 drag was **geometric** (removable by a
  better basis); the facets *are* independently steerable.
- **BOUND** → no readout-orthogonal direction transports depth: the depth contrast lives
  **inside** the type-readout subspace — a genuine representational coupling.

## What the geometry pre-check already establishes (design motivation)

`out/exp42_geometry_precheck.txt` (seed 700), measuring the exp-40 directions:

- **The mean `v_type` direction is the wrong guard target.** `cos(v_depth, v_type)` at the
  head position is ≈0 (|cos| < 0.2 mostly). Orthogonalizing against the single `v_type`
  would remove almost nothing — a single-direction guard is ruled out.
- **`top_type` is a subspace, not rank-1.** Type-difference participation ratio ≈ 2.6 at
  every position (top component ~52% of variance) → the guard must be a **subspace
  erasure**.
- **The depth↔type overlap is strongly `k`-graded.** Fraction of `v_depth@t` inside the
  type subspace: **`k=2`: 0.77–0.94** vs **`k=1`: 0.09–0.36**. This explains exp 40's
  `k`-graded drag geometrically and yields a **registered prediction** (below).

Caveat, and why the experiment still runs: the pre-check used the type-**difference**
subspace, which contains nuisance; direction *energy* inside it is not readout *effect*.
The experiment guards the type-**readout** subspace and controls over-erasure.

## Construct: the readout-subspace guard

Per prefix position `p` over the steered support `[0..t]`:

1. **Type-readout subspace `G_p`** — fit a **linear decoder of `top_type`** from the clean
   residual at `p` over observable-matched type pairs; `G_p` = span of its top-`r` readout
   directions (the directions along which the type *observable* linearly moves), `r` the
   registered `GUARD_RANK` (sized from the measured PR ≈ 2.6). The decoder is fit on
   **observable labels** (the Dyck parser), never the oracle — the standing honesty rule.
2. **Guarded depth direction** — erase `G_p` from the exp-40 depth diff-in-means:
   `v_depth_guard(p) = (I − P_{G_p}) v_depth(p)`. Steer with `v_depth_guard` via the exp-40
   additive splice (`apply_additive_steer`), full support `[0..t]`.
3. Read both observables (`read_facet`): depth transport (graded `cr_cond`, the target) and
   `top_type` drag (m=1, the off-target), exactly as exp 40.

**Reference guards (registered baselines, not the headline):** the raw single-`v_type`
guard (to confirm it under-erases, per the pre-check) and a **point-Jacobian** guard
(`∇`type-readout through `chain_probs` at the operating point) as a nonlinear cross-check
on the linear-decoder subspace.

## Discriminator and verdict (per horizon `k`)

Reuse the exp-40 transport/drag machinery, comparing the **guarded** depth direction to
its full-replacement ceiling, the random-direction floor, and a **new random-subspace
floor**:

```text
SEPARABLE  — guarded depth transport ≥ REF_FRAC·ceiling AND type drag ≤ DRAG_BOUND over
             the random floor, at matched transport: a readout-orthogonal depth handle.
BOUND      — guarded depth transport does NOT clear the random-SUBSPACE-erasure floor: the
             depth contrast is inside the type-readout subspace (genuine coupling).
NUISANCE_KILL — depth transport also dies under a random matched-dim erasure (erasing ANY
             r-dim subspace kills depth): the guard is UNINFORMATIVE here, not evidence of
             coupling. (The over-erasure guard; routes to a smaller r or a different guard.)
HARNESS_FAIL  — a self-test/positive-control fails (e.g. erasing G does NOT reduce the
             unguarded direction's type drag → G mis-estimated; projection not idempotent).
OBS_DRIFT     — endpoint estimator-vs-oracle gap > OE_BAND, or off-target definedness
             < OFF_DEF_MIN under the guarded steer (the exp-40 attrition guard).
SEED_UNSTABLE — no ≥3/4 seed majority.
```

**Reporting (the `OUTCOME_STRUCTURE` lens, dogfooded here).** Reduce the **replicate** axes
(seeds by ≥3/4 majority, positions by majority) but do **not** collapse the **horizon**
axis to one severity-ranked scalar. The headline is the `(k1, k2)` **configuration** (an
antichain), and routing is keyed on the configuration:

| configuration | reading | routes to |
|---|---|---|
| `{k1: SEPARABLE, k2: SEPARABLE}` | drag was geometric throughout; facets independently steerable | the coupling was a basis artifact; localize each (deferred L2 now motivated) |
| `{k1: SEPARABLE, k2: BOUND}` (**predicted**) | depth–type binding is **depth-graded**: shallow contrast separable, deep contrast bound | characterize the deep binding; the separability has a depth horizon |
| `{k1: BOUND, k2: BOUND}` | genuine representational coupling at both contrasts | accept shared substrate; the facets are not low-rank-separably steerable |
| any `NUISANCE_KILL` | guard uninformative at that contrast | re-size `r` / switch to the Jacobian guard before reading SEPARABLE/BOUND |

**Registered prediction (from the pre-check):** `{k1: SEPARABLE, k2: BOUND}`. Registering it
makes the run a test, not a fishing expedition.

## Confound table — load-bearing quantity (guarded depth transport)

| mechanism producing low guarded transport (apparent BOUND) | excluded by? |
|---|---|
| depth contrast genuinely inside the type-readout subspace (the intended signal) | what the experiment detects |
| **over-erasure**: the guard removed a nuisance/depth-shared subspace, not type per se | the **random-subspace floor** — depth must survive a random matched-`r`-dim, matched-norm erasure; if it doesn't, `NUISANCE_KILL`, not `BOUND` |
| **`r` set too large** → erases more than the type subspace | `r` registered from the measured PR; report transport sensitivity to `r ∈ {1,2,3}` as descriptive, headline at registered `r` |
| guard valid only to first order / breaks at the α needed to move depth | drag/transport read **at matched transport** over the α-ladder; endpoint audited (`OBS_DRIFT`) |

| mechanism producing high guarded transport + low drag (apparent SEPARABLE) | excluded by? |
|---|---|
| a genuine readout-orthogonal depth handle (the intended signal) | what the experiment detects |
| **guard didn't actually erase type** (`G` mis-estimated, `r` too small) | positive control: erasing `G` from the **unguarded** `v_depth` must reduce its measured type drag, and `top_type` must be **non-decodable** from the guarded residual — else `HARNESS_FAIL` |
| readout invariant to first order but type moves off-manifold at steer magnitude | `OBS_DRIFT` / `OFF_DEF_MIN` audit across the ladder (bounded, not eliminated) |

## Self-tests (known-answer, before any model claim)

- projection idempotent and orthogonal (`P_G² = P_G`; `(I−P_G)v ⊥ G`);
- synthetic **depth ⊥ type** → guarded = unguarded → `SEPARABLE`;
- synthetic **depth = type** (fully bound) → guard removes all of `v_depth` → `BOUND`;
- random matched-`r` erasure leaves a planted **orthogonal** depth direction intact
  (`NUISANCE_KILL` does **not** fire on a separable planted case);
- positive control: erasing the type subspace drives a planted type-drag to ≈0;
- α=0 guarded steer reproduces clean bit-exact (inherited from `apply_additive_steer`).

## Decisions needed before registration (flagged)

1. **Guard operationalization — linear type-decoder subspace (headline) vs point-Jacobian.**
   Recommend the **decoder subspace** as headline (robust to the operating point, matches
   the measured ~2.6-dim type structure), Jacobian as a registered cross-check. Confirm.
2. **`GUARD_RANK` `r`.** The type PR ≈ 2.6 ⇒ `r ∈ {2,3}`. Larger `r` erases more (safer
   against under-erasure, riskier for over-erasure — but `NUISANCE_KILL` catches that).
   Propose **`r = 3`** headline with `r ∈ {1,2}` reported. Confirm.

## Registered constants (proposed; inherit exp 40 unless noted)

| knob | value | note |
|---|---|---|
| α-ladder / positions / seeds / horizons / support | `(0.5,1,2,4)` / `{8,12,16,20}` / `{700–703}` / `k∈{1,2}` / full `[0..t]` | as exp 40 |
| `REF_FRAC` / `HANDLE_MARGIN` / `DRAG_BOUND` / `OE_BAND` / `OFF_DEF_MIN` | `0.50` / `0.15` / `0.15` / `0.10` / `0.80` | inherited |
| `GAP_MIN` / `N_SEQS` / `MIN_PAIRS` / `EVAL_CAP` / `SEED_MAJORITY` | `0.10` / `6000` / `256` / `400` / `3` | inherited; decoder fit on the fit-half, guarded steer scored out-of-fit |
| **`GUARD_RANK`** | **`3`** (proposed — decision 2) | type-readout subspace dimension; `r∈{1,2}` reported |
| **`R_SUBSPACE`** | **`4`** | random matched-`r`-dim, matched-norm erasures for the over-erasure floor |

## Reuse vs single-use

- **Import the promoted steering core** (`localize.py`): `facet_diff_vector`,
  `apply_additive_steer`, `read_facet`, `drag_fraction`, `transport_fraction`,
  `random_matched_direction`, plus `depth_triples` / `facet_pairs` / `q_at` /
  `make_patched_prefix` and the guards. exp 40 is **frozen**; nothing is imported from it.
- **Promote to `localize.py` (this rung's reusable core):** `readout_decoder_subspace`
  (fit a linear facet decoder → orthonormal subspace), `subspace_erase` (per-position
  orthogonal projection removing a subspace from a direction), `random_matched_subspace`
  (the over-erasure control) — each self-tested, facet- and toy-agnostic.
- **Rung-specific (in the exp-42 script):** the guarded-vs-unguarded scorer, the
  configuration reducer, the verdict. **Single-use:** the Dyck-2 thresholds.

## Non-goals

- No claim of separability/coupling **by any intervention** — scoped to the rank-1-per-
  position additive direction class with a **linear readout-subspace** guard (Jacobian as
  cross-check). No claim the directions/subspaces are the model's intrinsic features.
- No spatial localization (still full support) and no real-LLM claim; Dyck-2 checkpoint only.
