# Experiment 38 — Propagating-state instrument + "does the model propagate graded depth?" gate — CONCLUDED

**Script:** `scripts/localization/exp38_propagation_gate.py` (core in
`localize.py`). **Output:** `out/exp38_propagation_gate.txt` (`device=mps:0`).
**Result (4 seeds 700–703):** `DISTRIBUTED` at both horizons `k=1` and `k=2`,
**stable 4/4**. Graded depth **is carried and transportable** (full-/large-window
patch transports 0.83–0.97 of the oracle gap, cleanly above the ≈0 same-depth
floor) — so it is **not** recomputed-from-scratch — **but it is not localized to a
small window** (contiguous transport ramps gradually with window size, beating random
placement on average, but no small window saturates). The binding reason is the
**early-saturation failure**: `PROPAGATED` requires it and it fails at every
position-cell but one (the seed-701 `t=8 k=1` outlier — also the lone `PROPAGATED`
cell). Necessity of `t` is, by contrast, largely *satisfied* — but that is a
recency signature (dropping the most-recent position hurts most), not evidence of a
locus, so it does not by itself localize. The exp-4/5 "summary, not propagated state" picture is updated *for
graded depth under this instrument*: it is **carried state, but distributed
(recency-weighted), not point-localized**. See **Results** at the foot; the body
below is the pre-registration as reviewed.

> **Back-annotation (exp 43, 2026-06-23).** Exp 43's component-write enumerator localized the
> transporting term: it is the **token (bracket) embeddings**, not an internal carried summary.
> 38's `f_full`≈0.83–0.97 was patched at `LAYER=1`, whose residual **includes the embeddings**, and
> 38's premise gate shows the embeddings alone carry full source depth (`f_emb`=1.00) while *all*
> internal prefix-position writes carry ≈0 (`f_int`≤0.24, `int_nec`=0.00, 4/4 fresh seeds). So 38's
> **"not recomputed-from-scratch" stands** (depth is not re-derived from nothing) — but "carried
> state" is sharpened to **recomputed-from-the-bracket-embeddings each step**: the "carrier" is the
> bag of bracket-embeddings spread across prefix positions (the exp-4/5 "summary recomputed from
> tokens, not propagated state" property, now at component granularity), **not** an internal
> summary. 38's spatial `DISTRIBUTED` (no small *prefix* window saturates) and 43's
> `LOCALIZED_COUNTER` (a small *readout* head set recomputes it) are not in tension: they are about
> different position families — 38 patched prefix positions, 43 splices the readout-window heads
> that *read* the embeddings. The recomputation localizes to `(3,attn,3)` at the readout `t+k`
> (small set, genuine). Specificity at that locus is untested (43's F1).

**Status: concluded.** State-localization phase, **L1** (rung after L0 / exp 37) —
an instrument build + gate, like L0. The registered parameters/thresholds are in
**Registered parameters** below; the feasibility precondition (ceiling + planted
locus) was **GREEN**.

## Why this, not block localization (what exp 37 changed)

L0 established that reading the **m=1 marginal** of the completion gives only 3-level
**close-readiness** (empty / interior / full), so "depth" collapsed and the phase
question slid from *where does graded state live* to *where does the next-token
summary live* — the exp-4/5 summary-not-state wall, **second** sighting. The phase
doc's conditional dynamics rung ("if L1 shows propagation") is **unsatisfiable** by
an m=1 instrument, so the decision is made here, not deferred.

## The question (a typed fork, well-posed in both directions)

```text
Does this Dyck-2 model carry graded stack depth as a LOCALIZED residual summary
that downstream computation uses — or is graded depth only RECOVERABLE from the
distributed token history (no single locus summarizes it)?
```

- **A — PROPAGATED & localizable:** a localized residual patch transports source
  graded depth → graded state is summarized at a place; proceed to same-vs-different
  parts localization with the validated instrument.
- **B — RECOMPUTED / distributed:** no localized patch moves graded depth → the
  summary-not-state wall is a **measured model fact** for localization purposes. A
  clean B reframes the phase toward summary features + the recompute finding.

## Instrument: exact teacher-forced multi-step conditionals (no rollout)

The graded signal m=1 cannot see is the **conditional after a forced close**.
Interchange-patch source-depth onto clean tokens, feed a *close* token at `t+1`,
read the prediction at `t+2`: the forced close advances the stack by one from
*whatever pre-close depth the model believes*, so the post-close close-readiness
**reveals** that believed depth (source vs clean). This needs **no free rollout** —
it is the exact m≥2 teacher-forced conditional, one forward pass, deterministic.

It reuses L0's existing machinery almost wholesale: `make_Xc`/`q_at`/`exact_joint`
already build the full `V^m` completion joint; L0 only read its m=1 marginal. Here
we read the **m=2 / m=3 conditionals** of the same joint. The only new patch
mechanics are single-position and windowed `prefix_state` slices (L0 had full-prefix
and no-op).

Graded observable — **close-readiness after `k` forced closes**, `k = 0,1,2` (within
the standing m=3): `k=0` is L0's signal; `k=1,2` are the graded extensions that
separate depth 1 / 2 / 3 (depth `d` can absorb `d` closes before the stack empties).

## Feasibility precondition (smoke before pre-registration — gates the rung)

The instrument's premise — that the m≥2 conditional at `t+1`/`t+2` is sensitive to
the patched prefix — is **inherited, not measured on this model**. The patch enters
at `LAYER=1` (input to `blocks[1]`); continuation positions reach it through
`blocks[1:]` (3 of 4 blocks), but their `block-0` representation already encodes the
**clean** prefix depth, so the patched source-depth must overcome that through
attention alone. Exp 37 certified only **m=1** transport; "multi-step leaks the clean
prefix" is carried from exp-4/5, never tested here. So the **ceiling** — full-prefix
patch reaching source graded depth on the `k=1` conditional — is an empirical
unknown, not a near-tautology. **Smoke it first:** if the full-prefix ceiling cannot
move the `k≥1` conditional above the random floor, the rung is `HARNESS_FAIL` and no
curve is interpretable. The whole graded claim lives in `k≥1`, which is exactly the
contaminated readout, so this precondition gates the pre-registration.

**Smoke result (`scripts/localization/exp38_ceiling_smoke.py`, seed 700; artifact
`out/exp38_ceiling_smoke.txt`; GREEN — ceiling viable at both graded horizons).** On
depth-matched pairs (`top_type`-matched, the contrast is graded depth), the
forced-close conditional cleanly separates the depths (clean `≈0.01–0.06`,
source-oracle `≈0.60`). The full-prefix patch transports a large fraction of the
oracle gap, with the **same-depth patch floor at `≈0`** at every position — so the
move is a function of the *patched graded depth*, not a generic full-prefix
disturbance:

| horizon | depths separated | net transport above same-depth floor (4 positions) |
|---|---|---|
| `k=1` | depth 1 vs 2 | 0.89 – 0.97 |
| `k=2` | depth 2 vs 3 | 0.84 – 0.94 |

`k=0` reproduces L0's exact m=1 transport (`f≈1.00`, wiring sanity). Depth-2/depth-3
`top_type`-matched pairs are **abundant** (523–576 per position), so the `k=2`
horizon and the depth-3 contrast are not pair-starved. The inherited "multi-step
leaks the clean prefix" worry does **not** hold for the full-prefix patch here.
**Scope:** this de-risks only the curve's full-prefix *endpoint* (and the `k≤2`
horizon); localization — whether a *small window* suffices — is the rung's
discriminator and is **not** run pre-registration. The floor must be **same-depth**:
a same-source-pool floor (e.g. a permuted same-as-source-depth instance) does not
vary the patched depth, so it would move the conditional as much as the ceiling and
mask the result — which is why the registered floor is same-depth.

## The discriminator (the centerpiece — two curves, not one contrast)

On a Dyck prefix the tokens *determine* the stack, so propagation vs recomputation
is undecidable on a clean prefix. Interchange creates the disagreement — **residual
says source-depth, tokens say clean-depth** — and the *shape* of how transport
depends on what we patch is the verdict:

- **Sufficiency / locality curve.** Patch a growing prefix window ending at `t`
  (just `t`; `t-1..t`; …; full prefix); trace graded transport vs window size.
  **Propagated-and-summarized** ⇒ saturates *early* (small window suffices).
  **Recomputed-from-distributed-history** ⇒ ramps late, needs nearly the whole
  prefix. (The full-prefix patch is the curve's endpoint, not a standalone
  tautological ceiling.) **Matched control (required):** at each window size also
  patch an equal-count **randomly-placed** set of source positions. Window size is
  correlated with injected source-signal mass, so the locus reading is the
  contiguous-ending-at-`t` curve *relative to* the random-placement curve — never
  the raw curve, which a recomputing model can ramp purely from added signal mass.
- **Necessity.** Patch full-prefix *minus* `t`. If `t` carries the summary, removing
  it blocks transport. **Matched control:** full-prefix *minus a random position*;
  necessity of `t` is its drop *relative to* the random-drop drop (dropping any
  position removes some mass). Sufficiency + necessity = the mediation logic; far
  stronger than either alone.
- **Depth-over-horizon curve.** Transport at `k = 0,1,2` forced closes — *how many*
  graded levels propagate, a shape in its own right.

Normalization & references (same logic as L0's floor): transport is expressed as a
fraction of the **oracle-calibrated source−clean gap**. Floor = same-depth source
(no residual/token disagreement → no move) + random/mismatched-position. Ceiling =
full-prefix patch on the same probe (calibration vs oracle endpoints, `OBS_DRIFT`).
The curve verdicts ("saturates early", "ramps late") are read only against two
**measured** reference curves, not absolute window indices: the **planted-locus**
reference (known single-position summary → the early-saturation ceiling shape) and
the **random-placement** curve above (the no-locus floor shape). The
saturation/necessity thresholds are *defined* by these references — the load-bearing
calibration of this rung, as the floor was at L0 — with values fixed at
pre-registration.

### The planted-locus reference (the rung's central calibration — specified here)

This reference is doubly load-bearing: it sets the early-saturation shape *and* it
is the only thing that tells us whether a *true* single-position summary can survive
the small-window contamination regime at all — a small window leaves most of the
prefix's clean `block-0` unpatched, so "small window doesn't transport" is otherwise
ambiguous between *no locus here* and *clean-`block-0` from the unpatched majority
overwhelms a small patch*. It must therefore be built **in the real model's forward
pass, in the same contamination regime as the model curve** — a synthetic residual
field run through `blocks[1:]` would have *no* clean-`block-0` contamination at the
continuation positions and so its curve would not transfer.

**Construction (in-model, shared-prefix):** build clean/source pairs that are
**token-identical on `[0, t-1]`** and diverge in graded depth only from position `t`
onward (the depth-changing bracket sits at `t`). Then residuals on `[0, t-1]` are
identical across the pair, so any window extending into `[0, t-1]` adds a **no-op**
patch — the entire transportable difference is concentrated at/after `t` *by
construction of the tokens*, and the locality curve **must** saturate at the
smallest window for a model that summarizes at a position. This is a known-answer
early-saturation reference that carries the real continuation contamination, because
it is the real model run on real Dyck continuations.

**Transfer-validity gate (pre-committed).** Before the model curve is read, the
planted-locus self-test must show its small-window transport **clears the
random-placement floor** in *this* model. If even a constructed single-position
summary cannot overcome small-window contamination here, the locality axis is
uninterpretable: the run returns `HARNESS_FAIL` (not `RECOMPUTED`/`DISTRIBUTED`),
because "ramps late" could not be distinguished from "no small patch can win
against contamination." A planted-locus that lives in a different contamination
regime than the model pairs is not a valid reference and is a registration defect.

## Confound table — load-bearing quantity (graded transport vs patch window)

| mechanism producing the curve shape | excluded by? |
|---|---|
| genuine locus: depth summarized at/near `t`, so a small window transports (the intended signal) | this is what the locality curve is built to detect |
| injected-signal mass: a wider window simply attends in more source residual, so transport ramps with size with no locus — a recomputing model then looks "distributed" for a non-locus reason | the **equal-count random-placement** control; the locus reading is contiguous-vs-random at matched mass, never the raw curve |
| representational-inconsistency artifact: source residual over `[t-w..t]` atop clean `block-0` at `[0..t-w-1]` is off-distribution, so its readout may not be "believed depth" | partly — the oracle endpoint audit (`OBS_DRIFT`) bounds the **endpoints** only; every *interior* window is a hybrid, so this bounds the **entire locality reading**, not just the ends. The planted-locus reference (also a hybrid, same regime) and the random-placement control are read in the same off-manifold regime, which is what keeps the *relative* shape interpretable; the absolute mid-curve values are not. Stated as a bound on the verdict, not eliminated |
| necessity confounded by mass: dropping any single position removes some signal | the **random-drop** control; necessity of `t` is its drop relative to a random-position drop |
| instrument can't move the conditional at all (`block-0` clean contamination wins) | the full-prefix **ceiling** smoke precondition; a failed ceiling is `HARNESS_FAIL`, not `RECOMPUTED` |
| redundant distributed carrying mimics a null (the redundancy confound) | **not excludable** by interchange — bounds the negative to "not localizably summarized", see below |

## What a negative excludes (claim bound — required honesty)

`RECOMPUTED` means **"graded depth is not localizably summarized at a position,"**
consistent with *both* true per-position recomputation *and* **redundant distributed
carrying** (the named redundancy confound). It does **not** license "the model does
not propagate." The sufficiency curve partly separates these — if some window
restores transport, the state is carriable; necessity says whether a locus is
required — but redundancy is not fully excludable here, so the claim is bounded to
localizability, not existence. **The conclusion template pre-commits this wording:**
a `RECOMPUTED`/`DISTRIBUTED` result is written as "not localizably summarized" and
may not be reworded to "the model recomputes" or "carries no propagated state".

## Verdict (exhaustive; registered, implemented in `verdict_one` / `_reduce_positions`)

Per `(position, horizon k)`; reduced to a per-`k` verdict by position majority
(`PROPAGATED` if a position-majority propagates, `RECOMPUTED` if a majority
recompute, else the `DISTRIBUTED` middle; any `OBS_DRIFT` position → `OBS_DRIFT`; no
qualifying positions → `SEED_UNSTABLE`), then to a per-`k` cross-seed majority
(`≥3/4`, else `SEED_UNSTABLE`). `f` = transport fraction `(P−C)/(S−C)`, `C`/`S` the
clean/source-oracle conditional, pooled over pairs with oracle gap `≥ GAP_MIN`.
`f_full` = contiguous full-prefix patch; `samedepth` = full-prefix patch with a
**same-depth** source (the ≈0 floor).

**Cross-horizon reduction (registered).** Per-horizon routing is the **primary**
output — a horizon reroutes itself (e.g. k=1 `PROPAGATED` → that depth contrast
localizes), mirroring L0's per-facet routing. The single headline `DECISION` is the
**highest-severity** horizon, with severity order `HARNESS_FAIL > OBS_DRIFT >
SEED_UNSTABLE > RECOMPUTED > DISTRIBUTED > PROPAGATED`: so `PROPAGATED` is the
headline **only if every horizon propagates**, and an uninterpretable or
underpowered horizon (`HARNESS_FAIL`/`OBS_DRIFT`/`SEED_UNSTABLE`) is **surfaced, not
masked** by a stable substantive verdict on the other horizon.

```text
HARNESS_FAIL  — a model guard fails (no-op not bit-exact, or full patch m=1 != source m=1), OR the planted-locus transfer-validity gate fails (a known single-position summary cannot clear the random floor at window 1); blocks all
OBS_DRIFT     — conditional-vs-oracle endpoint gap (|S − oracle|) > OE_BAND (uninterpretable)
PROPAGATED    — carriable AND saturates early (best small-window f >= SAT_FRAC * f_full AND beats random-placement by LOCUS_MARGIN) AND t necessary (nec_t >= NEC_MARGIN and > random-drop): graded depth localizably summarized -> L2 localization
DISTRIBUTED   — carriable (f_full − samedepth >= FULL_MIN) but NOT early-saturating or t NOT necessary: carriable but not localized (a typed middle)
RECOMPUTED    — NOT carriable: f_full − samedepth < FULL_MIN, i.e. even the full-prefix patch does not clear the same-depth floor. Pre-committed bound: this NEVER licenses "the model does not carry/propagate graded depth" — redundant distributed carrying is not excludable by interchange
SEED_UNSTABLE — no >=3/4 cross-seed majority, or a horizon has no qualifying positions; underpowered (add seeds/seqs)
```

## What carries from L0 (not re-derived)

Dyck parser labels; the facet split and the certified `top_type` substrate; the
`depth`→close-readiness scope; dissociable-pair abundance (cells 510–512, so
matched depth-1/depth-2 source/clean pairs exist); checkpoint + validity gate +
bit-exact model guards; the m=1 close-readiness scorer becomes the `k=0` rung of the
horizon curve. The **random-unit floor obligation L0 deferred is built here**: L0's
depth `FLOOR_FAIL` was conservative precisely because the random-unit baseline was
unmeasured, and that baseline is this rung's random/mismatched-position floor. Depth
transport is read against it; whether this lifts L0's "purity uncertified" flag is
stated in the conclusion, not assumed here.

## Self-tests (known-answer, before any model claim)

- planted locus (the in-model shared-prefix construction above): sufficiency must
  saturate at the smallest window and necessity must flag `t`, **and** its
  small-window transport must clear the random-placement floor (the transfer-validity
  gate) — else `HARNESS_FAIL`, the locality axis is uninterpretable in this model;
- no-difference (same-depth source) → ≈ 0 at every window;
- full-prefix patch reaches the source graded target (ceiling sanity);
- unpatched run reproduces clean graded depth;
- single-position / windowed patches are bit-exact reconstructions of the intended
  residual splice (no off-by-one in the prefix slice).

## Registered parameters (implemented in the script)

| index | value |
|---|---|
| checkpoint | `out/dyck2-L4` (exp-19 Dyck-2 config; `require_expected_config` halts on mismatch). Validity gate must pass |
| patch point | residual stream `LAYER=1` (L0's; the ceiling smoke validated multi-step sensitivity at this layer). No layer sweep this rung |
| positions | `{8, 12, 16, 20}` (L0's interior positions) |
| horizons | `k ∈ {1, 2}`: depth `(1,2)` and `(2,3)` source/clean pairs, `top_type`-matched; `k=0` is L0's m=1 signal (wiring sanity) |
| window ladder | contiguous windows ending at `t` of sizes `1,2,4,8,…` plus the full prefix `[0..t]`; random-placement control at each size, averaged over `R_RAND=3` draws |
| pairs | `depth_triples`: clean + same-depth floor at depth `lo`, source at depth `hi`, `top_type`-matched; `≥256` pairs per (position×horizon), else the position is skipped (depth-3 abundance confirmed 520+/pos by the smoke) |
| seeds | `700..703` (4) |
| oracle | exact Dyck joint, endpoint calibration audit only (`OBS_DRIFT`); never selection/scoring |

**Registered thresholds** (gate cutoffs, printed/audited). These are **tolerance
choices read against measured references, not data-tuned cutoffs**: `SAT_FRAC` is a
fraction of *each curve's own* `f_full`, the locus/necessity margins are over *each
curve's own* measured random floor, and carriability is over the per-pair same-depth
floor. The ceiling/floor those tolerances must separate are **committed**, not a
model peek:
- **full-prefix endpoint** (`FULL_MIN`): `out/exp38_ceiling_smoke.txt` — full-prefix
  transports 0.84–0.98 of the oracle gap, same-depth floor ≈0, at k=1 and k=2.
- **locality axis** (`SAT_FRAC`, `LOCUS_MARGIN`, `NEC_MARGIN`, `PLANTED_MIN`):
  `out/exp38_locality_reference.txt` — the planted-locus (known single-position
  summary) contiguous curve saturates at window 1 at `f ≈ 0.70–0.83` with a
  random-placement floor of `0.00` at window 1, and planted `nec_t ≈ 0.70–0.83`
  vs `nec_rand ≈ 0`. The registered cuts sit well inside that ceiling/floor band.

| name | value | meaning |
|---|---|---|
| `GAP_MIN` | 0.10 | min oracle gap `|S−C|` for a pair to be scored |
| `W_SMALL` | 2 | "small window" = ≤ this many positions ending at `t` |
| `SAT_FRAC` | 0.50 | early-saturation: small-window `f` ≥ this × `f_full` |
| `LOCUS_MARGIN` | 0.15 | small-window `f` must beat its matched-mass random-placement `f` by this |
| `NEC_MARGIN` | 0.15 | necessity: dropping `t` drops `f` by ≥ this (and > random-drop) |
| `FULL_MIN` | 0.30 | carriability: `f_full − samedepth` ≥ this |
| `PLANTED_MIN` | 0.30 | transfer-validity: planted window-1 `f` beats random by this |
| `OE_BAND` | 0.10 | max conditional-vs-oracle endpoint gap |

**Residual risk (depth-gap).** The planted-locus reference is depth-1-vs-3 (gap 2),
forced by the single-divergence shared-prefix construction, whereas the model pairs
are gap 1. Because `SAT_FRAC` is a fraction of *each* curve's own `f_full`, the
*shape* (early saturation vs late ramp) transfers; the early-saturation *magnitude*
is not directly comparable across the depth gap. The reference calibrates the shape,
not an absolute transport level.

## Code consolidation (this rung) — done

- `scripts/localization/l0_substrate_gate.py` stays **frozen** — not touched.
- **`localize.py`** created: the durable L0 facet/Dyck core promoted verbatim
  (`stack_labels`, `facet_from_q1`/`facet_observable`, `make_Xc`/`q_at`/`exact_joint`,
  `facet_pairs`/`floor_pairs`, the checkpoint contract), plus this rung's new
  primitives (`cr_cond`, `make_patched_prefix`, `depth_triples`, `planted_locus_pairs`),
  self-tested. L0 keeps its inline copy (the accepted exp-15 duplication); verdict
  helpers reused from `battery.py`; patch/forward/joint mechanics from `midstream.py`.
- The curve reducers and verdict live in the rung script `exp38_propagation_gate.py`.
  The block/head unit enumerator stays **deferred** until this rung returns
  `PROPAGATED`/`DISTRIBUTED`.

## Non-goals

- No same-vs-different-parts localization claim here (next rung, gated on
  A/`DISTRIBUTED`). No granularity sweep, no head/direction enumerator, no real-LLM
  claim. The instrument is minimal: exact teacher-forced conditionals over patch
  windows — not a rollout/dynamics framework.

## Results

Run artifact: `out/exp38_propagation_gate.txt` (`device=mps:0`). Validity gate PASS
(gap −0.0121 nats); model guards OK (no-op bit-exact, full patch m=1 = source m=1);
**planted-locus transfer-validity gate OK** (window-1 transport 0.70–0.84 over a
0.00 random floor at all positions — a known single-position summary saturates at
window 1, so the locality axis is interpretable in this model). 4 seeds 700–703.

```text
per-horizon routing:  k=1: HELD -> DISTRIBUTED   k=2: HELD -> DISTRIBUTED
DECISION: DISTRIBUTED   (stable 4/4 seeds, both horizons)
```

Quantities (ranges across 4 seeds × 4 positions):

| quantity | `k=1` (depth 1 vs 2) | `k=2` (depth 2 vs 3) | reads as |
|---|---|---|---|
| verdict (4/4) | **DISTRIBUTED** | **DISTRIBUTED** | carriable, not localized |
| full-prefix transport `f_full` | 0.86–0.97 | 0.83–0.90 | graded depth **is** transportable |
| same-depth floor (full) | −0.07 … −0.00 | −0.09 … +0.01 | transport is depth-specific (≈0 floor) |
| `f_full − samedepth` ≥ `FULL_MIN`(0.30)? | yes (≫) | yes (≫) | **carriable** → not `RECOMPUTED` |
| small-window contiguous (w=1 / w=2) | 0.17–0.22 / 0.36–0.43 | 0.11–0.29 / 0.13–0.32 | below `SAT_FRAC·f_full` (≈0.43–0.49) → **no early saturation** |
| contiguous vs random (matched w, e.g. w=4) | 0.51–0.66 vs 0.13–0.63 | 0.31–0.55 vs 0.21–0.63 | contiguous beats random **on average / at most cells** (a few k=2 w=4 cells reverse) → recency structure, not pure mass |
| necessity `nec_t` / `nec_rand` | 0.15–0.21 / 0.00–0.35 | 0.23–0.34 / −0.02–0.21 | `nec_t` clears `NEC_MARGIN` and **mostly exceeds** random-drop (all 16 k=2 cells, ~14/16 k=1) — a recency signature, **not** a locus; the binding `DISTRIBUTED` reason is the saturation failure above, not a necessity failure |
| `oe` (conditional vs oracle) | 0.009–0.012 | 0.010–0.014 | ≪ `OE_BAND` → no drift |

### What the run establishes

**Graded depth is carried, not recomputed-from-scratch (the positive half).** The
m≥2 forced-close conditional follows the *patched source residual* rather than the
clean tokens: the full/large-window patch transports 0.83–0.97 of the oracle graded
gap, far above the ≈0 same-depth floor and above the random-placement floor. So
downstream computation *uses* carried graded state — the m≥2 instrument breaks the
m=1 wall L0 hit, **for graded depth, under this interchange instrument**. The
same-depth floor ≈0 shows the move is graded-depth-specific, not generic disturbance.

**But it is not point-localized (the negative half → the typed middle).** Transport
ramps roughly monotonically with the contiguous window (w=1 ≈0.2, w=2 ≈0.4, w=4
≈0.5–0.6, w=8 ≈0.7–0.85, full ≈0.9); **no small window reaches `SAT_FRAC·f_full`** —
this early-saturation failure is the binding reason for `DISTRIBUTED` (it is required
for `PROPAGATED` and fails at every position-cell but one, the seed-701 `t=8 k=1`
outlier disclosed below). Necessity, by contrast, is *not*
the failing arm: `nec_t` clears `NEC_MARGIN` and exceeds the random-drop at all 16
k=2 cells and ~14/16 k=1 cells — but this is precisely the recency signature
(dropping the most-recent position `t` removes the most-weighted mass), **not**
evidence of a point summary, which is why `PROPAGATED` is gated on saturation *and*
necessity together rather than necessity alone. The contiguous window beats
matched-mass random placement on average, so the carrying is **recency-weighted**
(positions near `t` matter more), not uniform — but it is **spread across the prefix,
not summarized at a position**. Verdict: `DISTRIBUTED`, stable 4/4 at both horizons.
(One position-cell, seed 701 `t=8 k=1`, was `PROPAGATED`; position-majority and the
other 15 cells are `DISTRIBUTED` — borderline noise, not a split.)

### Confound re-scoring (against the realized numbers)

- **Injected-signal mass** (the headline confound) — *excluded on average*: contiguous
  windows beat equal-count random placement at most matched sizes (a few k=2 w=4 cells
  reverse), so the ramp is not just "more source mass," there is genuine
  recency/locality structure. What the data shows is **distribution with a recency
  gradient**, not a point locus and not pure mass.
- **Representational-inconsistency (mid-curve hybrid)** — *bounded, not excluded*, as
  registered: the relative contiguous-vs-random shape is read in the same
  off-manifold regime; absolute mid-curve magnitudes are not interpreted. The
  verdict turns on shape (no small-window saturation), read relatively.
- **Redundancy** — *not excludable by interchange*, as pre-committed. `DISTRIBUTED`
  is consistent with both genuine spread carrying and redundant carrying; both are
  "carriable but not localizably summarized," which is exactly the claim made.

### Claim bound (pre-committed, honored)

`DISTRIBUTED` = **graded depth is carriable but not localizably summarized at a
position.** This is *not* "the model recomputes" (carriability refutes that here),
and *not* "graded depth is localized somewhere we just missed" (the planted-locus
gate shows a true single-position summary *would* have saturated at window 1, so the
null small-window result is informative, not an instrument failure). It is also not
a claim about *which* distributed mechanism (genuine spread vs redundancy) — that is
not separable by interchange.

### L0 carry-forward (depth purity)

L0 held `depth` as "purity uncertified" pending a random-unit baseline. This rung's
same-depth floor came in at ≈0 (the graded-conditional transport is depth-specific,
not nuisance-driven), which is reassuring — but it is a **different estimator** (m≥2
conditional transport) than L0's m=1 within-class close-readiness spread, so it
complements rather than directly retires L0's flag. The m=1 purity question stays as
L0 left it; this rung adds that the *graded* signal is clean and depth-specific.

### Routing

Graded-depth localization to a single position/block is **not supported** → the L2
coarse same-vs-different-parts rung should **not** expect a clean graded-depth locus;
it localizes the certified `top_type` substrate and the m=1 close-readiness summary,
reading any graded-depth importance as distributed. The recency-weighted ramp is a
**propagation** signature (graded state accumulated across the prefix) and is the
natural input to the dynamics rung (old L4): *where along the prefix* the distributed
carrying concentrates. No redesign; the gate did its job — it found graded state is
carried but distributed, which reroutes localization away from a single-locus
expectation.

> **Back-annotation (exp 40).** Spatial `DISTRIBUTED` is *not* "no low-rank handle": exp
> 40 finds a rank-1-per-position diff-in-means **direction** at full support that
> transports the graded-depth gap to its full-replacement ceiling. The handle is itself
> spread across *positions* (one direction per position); within each position it is
> low-rank. So this does not contradict the spatial distribution — it adds that a
> per-position facet direction carries depth (and, asymmetrically, drags `top_type`).
