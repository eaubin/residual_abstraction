# Experiment 42 — Guarded directional specificity: is the depth→`top_type` drag geometric or representational? — DRAFT

**Status: DRAFT pre-registration** (not yet implemented or run). The question, construct,
verdict partition, confounds, and the expected-information-gain case below are proposed;
the two guard decisions are now **settled** (see "Decisions (settled)"). Remaining before
freeze: build the promoted primitives + the script with its self-tests, then the
pre-registration **review pause**. Builds directly on exp 40 (`CROSS_DRAG`) and its
geometry pre-check
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

- **SEPARABLE** → the probed guarded direction `(I − P_G)·v_depth` *is* such a handle: the
  exp-40 drag was **geometric** (removable by a better basis); the facets are independently
  steerable by this construction. (SEPARABLE exhibits a witness, so it is a clean existential.)
- **BOUND** → the guarded exp-40 depth direction does **not** transport depth even rescaled:
  this diff-in-means depth contrast lives **inside** the type-readout subspace — a genuine
  representational coupling *for the probed direction*. This is a statement about the
  diff-in-means direction the construction points at, **not** a proof that no
  readout-orthogonal depth direction exists anywhere (see non-goals).

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
4. **Conditioning diagnostic `cap_G`** — `‖P_{G_p}·v_depth‖² / ‖v_depth‖²`, the fraction of
   `v_depth`'s energy captured by the guard subspace `G_p` (the share the erasure removes).
   Reported per `(position, k)` as a load-bearing diagnostic and registered because it gates trust
   in the *linear* verdict: when `cap_G` is high the guarded direction is a small residual of
   a near-total projection, and a linear `BOUND` cannot be told from the linear decoder
   over-erasing a direction the *nonlinear* type readout is actually invariant to. Above
   `CAP_G_MAX` the linear verdict at that contrast is **not trusted** and the Jacobian guard
   (below) **arbitrates** it.

**Guards: linear-decisive, Jacobian-arbitrated when ill-conditioned.** The linear
type-decoder subspace is the primary guard. The **point-Jacobian** guard (`∇`type-readout
through `chain_probs` at the operating point — the exact, nonlinear readout sensitivity) is
a cross-check *and* the **decisive arbiter on any contrast where `cap_G > CAP_G_MAX`** (it
erases only what the readout truly moves along, so it does not over-erase a
nonlinearly-type-invariant depth direction). The raw single-`v_type` guard is a third
baseline, registered only to confirm it under-erases (the pre-check showed `cos≈0`).

## Discriminator and verdict (per horizon `k`)

Reuse the exp-40 transport/drag machinery, comparing the **guarded** depth direction to
its full-replacement ceiling, the random-direction floor, and a **new random-subspace
floor**:

```text
SEPARABLE  — guarded depth transport ≥ REF_FRAC·ceiling AND type drag ≤ DRAG_BOUND over
             the random floor, at matched transport: a readout-orthogonal depth handle.
BOUND      — the probed guarded depth direction's transport does NOT clear the random-
             SUBSPACE-erasure floor (and cap_G ≤ CAP_G_MAX): this diff-in-means depth
             contrast is inside the type-readout subspace (coupling for the probed direction,
             not a universal non-existence claim). If cap_G > CAP_G_MAX the Jacobian guard
             arbitrates instead.
NUISANCE_KILL — depth transport also dies under a random `r`-dim erasure (averaged over
             R_SUBSPACE draws): erasing ANY `r`-dim subspace kills depth, so the guard is
             UNINFORMATIVE here, not evidence of coupling (routes to a smaller `r`).
HARNESS_FAIL  — a self-test/positive-control fails (e.g. erasing G does NOT reduce the
             unguarded direction's type drag → G mis-estimated; projection not idempotent).
OBS_DRIFT     — endpoint estimator-vs-oracle gap > OE_BAND, or off-target definedness
             < OFF_DEF_MIN under the guarded steer (the exp-40 attrition guard).
SEED_UNSTABLE — no ≥3/4 seed majority.
```

**Verdict precedence and operating point.** Each `(position, k)` cell is reduced by position
majority then cross-seed `≥3/4` (the 38/40 `majority_vote` / `first_precedence` pattern), with
registered precedence `HARNESS_FAIL > OBS_DRIFT > SEED_UNSTABLE > NUISANCE_KILL > BOUND >
SEPARABLE` (uninformative/failure verdicts dominate; `BOUND` over `SEPARABLE` matches exp 40's
"the coupling reading wins ties"). The `cap_G > CAP_G_MAX` gate is applied **before** a cell is
read as SEPARABLE/BOUND: when it fires the cell is handed to the Jacobian arbiter and reported as
`BOUND(Jac)`/`SEPARABLE(Jac)`. **Matched-transport operating point:** type drag is read at the
smallest α on the ladder where guarded transport first reaches `REF_FRAC·ceiling`; if no α reaches
it, transport has failed and the cell is `BOUND` (or `NUISANCE_KILL` / Jacobian-arbitrated per the
floor and `cap_G`) with drag moot — this is the `BOUND` operating point, where "matched transport"
is otherwise undefined.

**Reporting (the `OUTCOME_STRUCTURE` lens, dogfooded here).** Reduce the **replicate** axes
(seeds by ≥3/4 majority, positions by majority) but do **not** collapse the **horizon**
axis to one severity-ranked scalar. The headline is the `(k1, k2)` **configuration** (an
antichain), and routing is keyed on the configuration:

| configuration | reading | routes to |
|---|---|---|
| `{k1: SEPARABLE, k2: SEPARABLE}` | drag was geometric throughout; facets independently steerable | the coupling was a basis artifact; localize each (deferred L2 now motivated) |
| `{k1: SEPARABLE, k2: BOUND}` | depth–type binding is **depth-graded**: shallow contrast separable, deep contrast bound | characterize the deep binding; the separability has a depth horizon |
| `{k1: SEPARABLE, k2: BOUND(Jac)}` (**predicted**) | same depth-graded binding, but the deep contrast is read by the **Jacobian arbiter** because `cap_G > CAP_G_MAX` made the linear `BOUND` untrustworthy | as above; plus the linear subspace guard is recorded as too coarse at the deep contrast |
| `{k1: BOUND, k2: BOUND}` | genuine representational coupling at both contrasts | accept shared substrate; the facets are not low-rank-separably steerable |
| any `NUISANCE_KILL` | guard uninformative at that contrast (random `r`-dim erasure also kills depth) | re-size `r` before reading SEPARABLE/BOUND |

**Registered prediction (re-pointed — see below):** `{k1: SEPARABLE, k2: BOUND(Jac)}` — k=1
resolved by the linear guard, k=2 arbitrated by the Jacobian because the pre-check's high
capture predicts `cap_G > CAP_G_MAX` there. Registering it makes the run a test, not a
fishing expedition.

## Expected information gain (predictions + credences; walled off from adjudication)

This section states what we *expect* and what each outcome would *teach*, to justify
running the experiment and to calibrate surprise. It is **deliberately separate from the
verdict logic** — credences here never enter a predicate; the adjudication rules above
stand on their own (the anti-bias reason predictions are not normally registered). Read it
as the value-of-information case (`docs/OUTCOME_STRUCTURE.md`, layer 4), not a thumb on the
scale.

**Prior belief (before the run).** exp 40: the depth→`top_type` drag is real and `k`-graded;
`top_type`→depth is clean. The geometry pre-check raises credence that depth-2v3 sits inside
the type subspace (capture 0.77–0.94) and depth-1v2 mostly outside (0.09–0.36) — but that is
*difference-variance* energy (includes nuisance), measured on the **wrong subspace** (the
guard is the nuisance-stripped *readout-decoder* subspace `G`), so it informs, not decides.
The model is `d=64`, so a random `r=3` erasure drops ~5% of dimensions — **near-harmless**,
which makes true `NUISANCE_KILL` (random erasure killing depth) *unlikely*. The real crux is
that the *targeted* erasure may remove most of `v_depth` at `k=2` (high `cap_G`); that is not
`NUISANCE_KILL` — it routes to the **Jacobian arbiter** (the reviewer's correction: a clean
"linear guard too coarse here, the nonlinear readout decides" is a better outcome than a
confounded linear `BOUND`).

**Credences over the configuration space** (re-pointed for the `cap_G`/Jacobian gate; rough,
summing ~1):

| configuration | credence | what it would teach (belief update) |
|---|---|---|
| `{k1: SEP, k2: BOUND(Jac)}` | ~0.45 | **separability is depth-graded**, deep contrast read by the Jacobian arbiter (linear guard too coarse there); resolves exp 40's confound as "geometric when shallow, bound when deep"; sharpens ledger row 37 |
| `{k1: SEP, k2: SEP}` | ~0.25 | the deep drag was geometric / nonlinearly-type-invariant — the Jacobian recovers a clean depth handle the linear decoder over-erased; flips the phase toward localize/intervene |
| `{k1: SEP, k2: BOUND}` (linear-decisive) | ~0.10 | `cap_G` stayed low and the linear guard alone resolved a deep binding |
| `{k1: BOUND, k2: BOUND}` | ~0.10 | genuine coupling even at the shallow contrast → **abandon low-rank separability**; route to subspace/manifold methods |
| `NUISANCE_KILL` (random `r`-dim erasure kills depth) | ~0.10 | the floor itself is too aggressive → re-size `r`; methodological, not scientific, info |

**Where the information actually comes from, and the dominant risk.** Every non-`NUISANCE_KILL`
configuration changes the next phase move and they are mutually distinguishing, so expected
info-gain is high. The honest **dominant risk is no longer `NUISANCE_KILL`** (random `r`-dim
erasure at `d=64` is near-harmless) but the **`cap_G`/Jacobian hand-off at `k=2`**: most of the
credence routes the deep contrast through the Jacobian arbiter, so the conclusion there is
only as trustworthy as the Jacobian guard — whose first-order validity must hold at the α
that moves depth. That is where the rigor budget now goes: the over-erasure controls
(random-subspace floor, depth-decoder retention, the erase-reduces-unguarded-drag positive
control) **and** a first-order-validity audit of the Jacobian guard across the α-ladder.
**Worth-running judgment:** yes — ~0.9 of the credence mass lands on an outcome that moves
the phase (the `cap_G` gate converts the old "confounded `BOUND` at k=2" worry into a clean
Jacobian-arbitrated result), and the residual is caught by the controls and reroutes the
instrument rather than wasting the conclusion.

## Confound table — load-bearing quantity (guarded depth transport)

| mechanism producing low guarded transport (apparent BOUND) | excluded by? |
|---|---|
| depth contrast genuinely inside the type-readout subspace (the intended signal) | what the experiment detects |
| **over-erasure**: the guard removed a nuisance/depth-shared subspace, not type per se | the **random-subspace floor** — depth must survive a random matched-`r`-dim, matched-norm erasure; if it doesn't, `NUISANCE_KILL`, not `BOUND` |
| **`cap_G` over-projection**: `cap_G` high, so the guard removes nearly all of `v_depth` and the small orthogonal residual fails to transport regardless of any causal coupling (the random floor does **not** catch this — random subspaces have low capture, so the floor is ≈ceiling while the targeted residual fails by construction) | the **`cap_G > CAP_G_MAX` gate** — the cell is not read as linear `BOUND`; the Jacobian arbiter (erases only what the readout truly moves along) decides, distinguishing readout binding from geometric over-projection |
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

## Decisions (settled)

1. **Guard operationalization — linear type-decoder subspace (headline), point-Jacobian
   cross-check.** The decoder subspace is robust to the operating point and matches the
   measured ~2.6-dim type structure; the Jacobian (`∇`type-readout through `chain_probs`) is
   a registered cross-check that the linear subspace did not miss a nonlinear readout
   direction.
2. **`GUARD_RANK` `r = 3`** (type PR ≈ 2.6), with `r ∈ {1,2}` reported as sensitivity. Larger
   `r` risks over-erasure, but `NUISANCE_KILL` (the random-subspace floor) catches that
   rather than letting it masquerade as `BOUND`.

## Registered constants (proposed; inherit exp 40 unless noted)

| knob | value | note |
|---|---|---|
| α-ladder / positions / seeds / horizons / support | `(0.5,1,2,4)` / `{8,12,16,20}` / `{700–703}` / `k∈{1,2}` / full `[0..t]` | as exp 40 |
| `REF_FRAC` / `DRAG_BOUND` / `OE_BAND` / `OFF_DEF_MIN` | `0.50` / `0.15` / `0.10` / `0.80` | inherited (exp-40 `HANDLE_MARGIN` dropped — unused by any exp-42 predicate) |
| `GAP_MIN` / `N_SEQS` / `MIN_PAIRS` / `EVAL_CAP` / `SEED_MAJORITY` | `0.10` / `6000` / `256` / `400` / `3` | inherited; decoder fit on the fit-half, guarded steer scored out-of-fit |
| **`GUARD_RANK`** | **`3`** (proposed — decision 2) | type-readout subspace dimension; `r∈{1,2}` reported |
| **`R_SUBSPACE`** | **`4`** | random matched-`r`-dim, matched-norm erasures for the over-erasure floor |
| **`CAP_G_MAX`** | **`0.85`** | above `cap_G = ‖P_G·v_depth‖²/‖v_depth‖²` the linear verdict is untrusted → the cell is Jacobian-arbitrated. Sits above the pre-check's `k=1` difference-capture (0.09–0.36) and inside its `k=2` range (0.77–0.94), so it operationalizes the registered prediction (`k=2`→Jacobian); note the gate reads `cap_G` on the narrower readout subspace `G`, so the actual hand-off is decided at runtime |

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
- **`BOUND` is not an existential non-existence claim.** Only one direction is probed — the
  guarded exp-40 diff-in-means direction `(I − P_G)·v_depth`. `BOUND` says *that* direction
  fails to transport depth; it does **not** exclude some other readout-orthogonal depth
  direction the diff-in-means construction never points at. (`SEPARABLE` is a clean
  existential — it exhibits a working handle.)
- No spatial localization (still full support) and no real-LLM claim; Dyck-2 checkpoint only.
