# Experiment 42 — Readout mechanism: is the depth→`top_type` drag curvature or a depth-conditional readout?

**Status: PRE-REGISTRATION** (predictions frozen; claim run = 4 seeds). Supersedes the
guard-and-steer draft (`git` history: `42-guarded-specificity.md`): two pre-checks showed
that design could not work — the type readout is rank-1 and ~orthogonal to `v_depth`
(pre-check #1), and a single-point guard cannot separate the two mechanisms because it is fit
at one operating point while the steer moves off it (pre-check #2). This is the
**directional-binding characterization** rung out of exp 40's `CROSS_DRAG`.

## Why this, and what exp 40 left open

Exp 40 found an asymmetric `CROSS_DRAG`: the `depth` difference-direction drags `top_type`
(stable at `k=2`, larger at the deeper contrast), and could not separate:

- **(geometric)** `v_depth` and the type readout are merely non-orthogonal — steering depth
  bleeds into the type readout, a removable basis artifact; or
- **(representational)** the depth contrast is genuinely bound into the type-carrying
  subspace.

Unpacked at the readout, the drag has exactly two mechanisms, and they *are* this fork:

1. **Curvature (geometric).** The type readout is depth-**independent**; the finite-α drag is
   nonlinearity of the readout and **vanishes as α→0** — separable by a fixed basis.
2. **Depth-conditional readout (representational).** The type **readout direction rotates
   with depth**, so steering depth moves into a region that reads type differently —
   removable by **no** fixed basis.

Guard-and-steer (any guard, linear or Jacobian) blends these: the guard is fit at one
operating point and the steer leaves it. This experiment measures the two mechanisms
**directly**.

## What the two pre-checks establish (design motivation)

- **#1 geometry** (`out/exp42_geometry_precheck.txt`): the drag is `k`-graded; `v_depth` is
  ~orthogonal to the **mean** type direction (cos≈0); the type-**difference** subspace is
  rank≈2.6 but is variance, not readout. → a difference-subspace guard would be nuisance, and
  the mean-direction guard erases ~nothing.
- **#2 readout mechanism** (`out/exp42_readout_mechanism.txt`, the seed-700 **design** run of
  this script): drag is **curvature-dominated** at the clean interior positions (rising-prefix
  slope `p≈3–4`), with a **modest, reliable, `k`-graded readout rotation** (`cos(g_lo,g_hi)`
  ≈0.95 at `k=1` vs ≈0.86 at `k=2`; split-half reliability 0.88–0.999). The single-position
  type-readout gradient is tiny (`|g|≈2e-4`) but reliably directed, so `v_depth`'s overlap with
  it yields negligible first-order drag — the observed drag is ~100× the first-order prediction.

Seed 700 is the **design seed** (it set the thresholds below). The claim run adds **701–703**
as the genuine out-of-design test.

## The question

```text
Is exp-40's depth→top_type drag (geometric) curvature of a depth-INDEPENDENT type readout,
or (representational) a depth-CONDITIONAL type readout whose direction rotates with depth?
```

## Construct: two direct measurements per (position `p`, horizon `k = lo→hi`)

1. **Readout rotation** `cos(g_lo, g_hi)`. `g_d` = the type-readout gradient at position `p`
   evaluated at depth-`d` operating points — `∇ top_type` w.r.t. the residual at `p`, by
   **central finite differences** through `chain_probs` (the model's own m=1 `top_type`
   observable; **no autodiff** — that is deferred to its own review). Computed on two disjoint
   halves so **split-half reliability** `rel_d = cos(g_d^A, g_d^B)` gates trust: a rotation cos
   is read only where both depths' gradients are reliably directed (`rel ≥ REL_MIN`). `|g|` is
   small (type is a flat single-position readout) — `rel`, not `|g|`, validates the direction.
   `cos ≥ ROT_HI` → depth-**independent** (geometric); `cos ≤ ROT_LO` → depth-**conditional**
   (representational).
2. **Drag slope** `p`. `v_depth` (the exp-40 `lo→hi` diff-in-means, `top_type`-matched) steered
   over `[0..p]` on the α-ladder; type drag = mean|Δ`top_type`|. `p` = log-log slope of drag
   vs α over the **rising prefix** (robust to the readout saturating at the top of the ladder).
   `p ≈ 1` → a first-order (linear-limit) coupling; `p ≳ 2` → curvature (drag→0 as α→0).

Diagnostics reported, not adjudicated: `cos(v_depth, g_lo)` and `cos(v_depth, g_hi − g_lo)`
(does the steer align with the readout **rotation** rather than the readout itself?).

## Verdict (per `(position, k)`, reduced across positions then ≥3/4 seeds)

```text
GEOMETRIC             — p > P_LINEAR AND cos(g_lo,g_hi) >= ROT_HI: curvature drag, depth-
                        INDEPENDENT readout (separable by a fixed basis).
CURVATURE_W_ROTATION  — p > P_LINEAR AND ROT_LO < cos < ROT_HI: curvature drag, but a modest
                        depth-conditional readout rotation underneath.
DEPTH_CONDITIONAL     — cos <= ROT_LO: the readout rotates substantially with depth
                        (representational coupling, removable by no fixed basis).
FIRST_ORDER           — p <= P_LINEAR: the drag has a linear-limit component (genuine first-
                        order coupling), readout rotation modest.
NO_DRAG               — max drag < DRAG_FLOOR: no coupling to characterize at this cell.
NONMONOTONE           — drag present but the rising-prefix slope is undefined.
RELIABILITY_FAIL      — a depth gradient is not reliably directed (rel < REL_MIN) or pairs thin.
```

Reduction (the `OUTCOME_STRUCTURE` lens): reduce **replicate** axes (positions by substantive
majority, then seeds by ≥3/4), but **do not collapse the horizon axis** — the headline is the
`(k1, k2)` **configuration**. `GUARD` cells (`NO_DRAG`/`NONMONOTONE`/`RELIABILITY_FAIL`) route
out of the per-position vote; a `(seed, k)` with <2 substantive cells reports its dominant guard.

## Registered prediction (walled off from adjudication)

Informed by the seed-700 design run; credences never enter a predicate. The **test** is whether
the pattern holds on the unseen seeds 701–703 and whether the `k`-graded rotation gap replicates.

| configuration | credence | what it would teach |
|---|---|---|
| `{k1: GEOMETRIC, k2: CURVATURE_W_ROTATION}` (**predicted**) | ~0.45 | the drag is predominantly geometric (large-α curvature) with a modest depth-conditional readout that sharpens at the deep contrast; resolves exp 40's confound as "mostly removable basis, weak genuine coupling at depth" |
| `{k1: GEOMETRIC, k2: GEOMETRIC}` | ~0.25 | the `k=2` rotation was seed-700 noise — purely geometric throughout; exp-40 drag is curvature, facets linearly separable; flips toward localize/intervene |
| `{k1: GEOMETRIC, k2: DEPTH_CONDITIONAL}` | ~0.15 | the deep contrast is genuinely depth-conditional (cos clears `ROT_LO` on replication) — a real `k`-graded representational binding |
| `{k1: CURVATURE_W_ROTATION, k2: …}` | ~0.10 | rotation reaches the shallow contrast too — coupling less `k`-graded than the design seed suggested |
| any `SEED_UNSTABLE` / guard-dominated | ~0.05 | the rotation/slope are not stable enough to read at ≥3/4 — methodological, re-power or re-site |

**Worth-running judgment:** yes — the four substantive configurations move the next phase
differently and are mutually distinguishing; the dominant risk is that `k2` sits near a
threshold (`cos≈0.86` between `ROT_LO` and `ROT_HI`), which `CURVATURE_W_ROTATION` is designed
to name honestly rather than force to GEOMETRIC or DEPTH_CONDITIONAL.

## Confound table — load-bearing quantities (rotation cos, drag slope)

| mechanism producing the reading | excluded by? |
|---|---|
| **noise** masquerading as readout rotation (tiny `|g|`) | **split-half reliability** gate (`rel ≥ REL_MIN`): a noise gradient has `rel≈0`; only reliably-directed gradients are read |
| readout **saturating / undefined** at the top of the α-ladder inflating/deflating the slope | slope fit over the **rising prefix** `[0..argmax drag]`; `NONMONOTONE` when no rise |
| **no coupling** at a cell read as a mechanism | `NO_DRAG` floor — a drag below `DRAG_FLOOR` is not classified |
| rotation read at one position while the steer is multi-position (the gradient is `p`-local) | scoped in non-goals; the **slope** (curvature vs first-order) uses the real all-position steer and is robust to this; only the rotation *magnitude* is `p`-local |
| FD step `eps` too large (out of the linear regime) / too small (numerical noise) | `eps = EPS_FRAC·std`; the linear-readout self-test recovers the known gradient; reliability catches a noise-dominated `eps` |

## Self-tests / controls (known-answer, before any model claim)

- `loglog_slope` recovers the exponent of `c·α^p`; `drag_slope` uses the rising prefix
  (quadratic-then-dropoff → `p≈2`, flat → `NONMONOTONE`);
- `fd_gradient` on a synthetic **linear** readout `σ(w·x)` recovers a vector ∥ `w`;
- synthetic **depth-conditional** readout (`w_lo` rotated to `w_hi`) → `cos(g_lo, g_hi)` =
  `cos(w_lo, w_hi)` (the rotation measure is faithful);
- `cell_verdict` on the registered threshold logic (each branch);
- `reduce_positions` majority / guard-routing / precedence-on-ties; `majority_vote` ≥3/4.

## Registered constants

| knob | value | note |
|---|---|---|
| positions / depths / seeds / horizons | `{8,12,16,20}` / `{1,2,3}` / `{700–703}` / `k∈{1,2}` | 700 = design seed |
| α-ladder / `GRAD_N` / `N_SEQS` / `GAP_MIN` | `(0.25,0.5,1,2,4)` / `256` / `6000` / `0.10` | ladder overlaps exp 40 |
| `EPS_FRAC` | `0.10` | FD step = `EPS_FRAC · per-coord residual std at p` |
| `REL_MIN` | `0.80` | split-half reliability floor (both depths) |
| `ROT_LO` / `ROT_HI` | `0.70` / `0.90` | depth-conditional (>~45°) / depth-independent (<~26°) |
| `P_LINEAR` | `1.5` | drag slope ≤ this → a first-order component |
| `DRAG_FLOOR` | `0.02` | max drag below this → `NO_DRAG` |
| `SEED_MAJORITY` | `3` | ≥3/4 seeds for a stable per-`k` verdict |

## Reuse vs single-use

- **Import** (`localize.py`): `facet_diff_vector`, `apply_additive_steer`, `read_facet`,
  `transport_fraction`, `facet_observable`, `q_at`, `depth_triples`, `stack_labels`. exp 40 is
  frozen; nothing is imported from it.
- **Rung-specific** (in this script): `fd_gradient` (finite-difference readout gradient),
  `readout_gradient_at_depth` (split-half), `drag_slope` (rising-prefix), `cell_verdict`,
  `reduce_positions`. Not promoted to `localize.py` until a second rung needs them.

## Non-goals

- No interventional claim and no intrinsic-feature claim — the readout gradient and `v_depth`
  are the operationalized objects, not the model's "true" features.
- The rotation is measured at a **single position** `p` while the steer is multi-position; the
  curvature-vs-first-order conclusion is robust to this (it uses the real steer), but the
  rotation **magnitude** is `p`-local. No spatial-localization claim, no real-LLM claim.
