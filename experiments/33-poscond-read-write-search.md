# Experiment 33 — I1′ position-conditioned fixed-read oblique write search on pstack-L4 — PRE-REGISTERED (awaiting review)

**Script:** `scripts/interventions/i1prime_poscond_read_write_search.py`
(harness promoted to the living `intervention_eval.py`).
**Output (after approval):** `out/exp33_pstack-L4.txt`.

**Status: pre-registered, awaiting pre-run review.** Per `AGENTS.md` /
`EXPERIMENT_REVIEW_PROTOCOL.md`, both the writeup and the runnable script
(guards, self-checks, verdict predicates, output tables, halt conditions) exist
before this counts as pre-registered. Pause here for design + code review before
the first claim-producing run. Everything below `## Results` is intentionally
absent until then.

**Decision form (filled by the run):**

```text
<branch>(phi1_next_closes[, phi2_net_return])   # see Decision precedence
```

This is the **fixed-read baseline** `INTERVENTION_CLASS_BENCHMARK.md` §I2's
"Pre-I2 read status" block requires be **run explicitly** (there is no inherited
I1 fixed-read result — exp 30 never adjudicated a write). It is the I1 write
question, finally asked with a transport-valid read.

## Phase fit and the prior failure this resolves

The intervention-class chain so far:

- **exp 29** — `phi1_next_closes`, `phi2_net_return` decode affinely on `pstack`
  and pass exact endpoint audit, but the same-read/same-write rank-1 patch does
  not move predicate probability. Typed a decode/control split for that patch
  class only.
- **exp 30 (I1)** — fixed-read oblique write search. Returned
  `FIXED_READ_LIMIT`: the single global affine read fit on discovery `{10,18}`
  did **not transport** to held `{26,34}` (held `R2` 0.06–0.26), so the
  read-transport gate fired and **the write was never adjudicated**.
- **exp 31 (atlas)** — the predicates *are* linearly readable **in place** at
  `{26,34}` (`R2` 0.55–0.75), but the read direction is **position-specific**.
- **exp 32 (discriminator)** — `POSITION_SPECIFIC_CONFIRMED`: a gain/bias refit
  of the discovery direction recovers held `R2` ≤ ~0.36 vs a ~0.55–0.75 in-place
  ceiling, so the held read is a genuinely different direction, not scale drift.

I1′ takes exactly the consequence exps 31/32 routed: **repair the read to a
position-conditioned in-place fit** (the transport-valid object), so exp 30's
read-transport wall is removed and the write question it was built to answer can
finally be adjudicated. The write search, candidate set, scores, thresholds,
specificity, and exact audit are exp 30's, unchanged. The *only* construct change
is the read: fit in place per bin instead of fit-on-discovery-then-transported.

**Carry-forward decision this changes** (the `INTERVENTION_CLASS_BENCHMARK.md`
next-experiment bar): whether the program carries **residual rank-1 oblique**
writes forward for these predicates. A specific, held-out-stable positive →
carry the fixed-read oblique class into I2 (read-varying earns its cost only as
an *improvement* over this baseline). A clean negative with full-patch room →
the in-place read is readable-but-not-writable at rank-1 oblique, routing to I2
read/write-pair or I3 interchange/path before any new toy. These route
differently, so the experiment clears the bar.

## Question

```text
On pstack-L4 at L1, m=3, with the predicate read repaired to a
position-conditioned IN-PLACE affine read (a separate fit per position bin, not
a single transported read), can a separate oblique WRITE direction causally
control phi1_next_closes / phi2_net_return, and does the discovery-selected
write transfer to held-out positions?
```

Two predicates, two position bins, one read class (position-conditioned
in-place), the exp-30 write arms. No read-varying / joint read-write pairs — that
is I2, and is out of scope here by design (this experiment is the baseline I2
must beat).

## Registered Command

After preregistration review only:

```bash
uv run python scripts/interventions/i1prime_poscond_read_write_search.py \
  --outdir out/pstack-L4 | tee out/exp33_pstack-L4.txt
```

Review-only checks (allowed before approval):

```bash
uv run python scripts/interventions/i1prime_poscond_read_write_search.py --selftest
uv run python -m py_compile interventions.py intervention_eval.py \
  scripts/interventions/i1prime_poscond_read_write_search.py
```

## Scope Indices

| index | value |
|---|---|
| process/checkpoint | `pstack-L4`, registered config from exp 29/30 (config guard) |
| patch point | residual stream L1, prefix-wide PairSet patch at the scored position |
| horizon | `m=3` within-horizon completion distribution |
| target predicates | `phi1_next_closes`, `phi2_net_return` |
| control predicates | `phi3_all_neutral`, `phi4_first_matched` (read controls only) |
| read class | **position-conditioned**: affine ridge readout fit IN PLACE per bin from per-position-centered L1 residuals to observable model `p_phi`; never transported across bins |
| intervention class | rank-1 oblique patch `P = c w^T / (c·w)`, fixed (position-conditioned) read `c`, searched write `w` |
| write arms | same-read/same-write; CEGAR-core directions + core-projected read; predicate-delta-stratified residual writes; one learned write (read fixed); norm-matched random writes |
| discovery bin | positions `{10,18}`, 512 pairs/seed (exp-30 construction) |
| held-out bin (transfer axis) | positions `{26,34}`, 1024 pairs/seed (exp-30 construction) |
| seeds | `400..403` (continuity with exp 30/31/32) |
| exact oracle use | endpoint audit only; no direction, threshold, write, or strength uses exact labels |

## Design

For each seed and each target predicate:

1. Build the discovery and held PairSets (exp-30 pair construction). Run
   `discover.self_checks` on each (known-answer guard).
2. **Position-conditioned reads.** Fit the affine read **in place** on the
   discovery bin (`c_disc`) and **independently in place** on the held bin
   (`c_held`), each via `intervention_eval.fit_inplace_read`: per-position-centered
   residuals → observable `p_phi`, ridge fit on a random train half, in-place
   `R2` scored on the same bin's test half. Neither read is transported.
3. Endpoints per bin (`p_un`, `p_src`, full-patch `p_full`), full-patch room, and
   the exact endpoint audit (`intervention_eval.endpoints`).
4. **Write candidates** built on discovery with `c_disc`
   (`build_write_candidates`): same-read/same-write, CEGAR-core directions and
   the core-projected read, predicate-delta writes, random writes; plus one
   learned write (`learn_write`, read fixed, `w = c + u_perp` so `c·w = 1`,
   observable predicate MSE to source on discovery pairs).
5. **Selection on discovery.** Each candidate's dose curve over
   `α ∈ {0,.25,.5,1,1.5,2}` is scored on the discovery bin with `c_disc`; per
   family the best-α control is kept; the best non-random family by discovery
   control is the headline write `w*`.
6. **Transfer to held.** Score `w*` on the held bin with **`c_held`** (the held
   in-place read) → `best_heldout`; `retention = best_heldout / best_disc`.
   Specificity is the max absolute non-target predicate control of `w*` on held
   (non-targets with held full-patch room ≥ `0.01`).
7. **Baselines.** Same-read/same-write held control (`same_heldout`), random-write
   held control (`random_heldout`) and discovery control (`random_disc`),
   full-patch room (both bins), endpoint audit (both bins).
8. **Descriptive cross-check (not a verdict input).** `best_held_inplace`: the
   best held-bin control achievable by *any* candidate write paired with
   `c_held`. See "Measured-but-unadjudicated."

The learned-write arm is typed `LEARNED_DIVERGED` and cannot carry a positive if
its norm is non-finite or exceeds `1e4`.

## Scores

Predicate control is the exp-29/I0 closure fraction
(`interventions.predicate_control`):

```text
c(P) = [MSE(p_un, p_src) - MSE(p_P, p_src)] / [MSE(p_un, p_src) - MSE(p_full, p_src)]
```

The denominator is full-patch room; `room ≤ ROOM_TOL` → control is NaN
(`NO_PATCH_ROOM`). No-op ≈ 0, full ≈ 1. Specificity is the max `|c(P)|` over
room-cleared non-target predicates (lower is better) — a predicate-level check,
not a full-distribution noninterference theorem. Exact endpoint audit is
`max(mean|p_un_model − p_tgt_exact|, mean|p_src_model − p_src_exact|)`; it audits
observable endpoint calibration only, never the off-manifold patched activation.

## Per-Seed Verdicts

Exactly one branch per `(target, seed)` by the precedence below (the script's
`classify_target`). Mutually exclusive and exhaustive over run outcomes.

| branch | plain-language gloss | condition | routes to |
|---|---|---|---|
| `TARGET_VACUOUS` | predicate too flat to read at a bin | `std_disc<VAR_MIN` or `std_held<VAR_MIN` | change target |
| `READ_NOT_DECODABLE` | the position-conditioned in-place read fails at a bin — exp 31/32 premise not reproduced | `r2_disc<R2_MIN` or `r2_held<R2_MIN` | fix substrate/measurement |
| `NO_PATCH_ROOM` | full/reference patch cannot move the predicate | `room_disc≤TOL` or `room_held≤TOL` | change target/patch point/process |
| `OBS_EXACT_DRIFT` | observable endpoints not calibrated to exact truth | `oe_disc>OE_BAND` or `oe_held>OE_BAND` | repair predicate scoring before geometry |
| `DISCOVERY_ONLY_WRITE` | a write controls on discovery but does not transfer to held (even with the held read) | `best_disc≥C_MIN` and (`best_held<C_MIN` or `retention<RETENTION_MIN`) | position-entangled write; I2/I3, not a primitive |
| `NO_POSCOND_READ_WRITE_WORKS` | no registered write controls through the in-place read despite room | `best_disc<C_MIN` | readable-not-writable at rank-1 oblique → I2 read-pair / I3 interchange |
| `NONSPECIFIC_CONTROL` | target control is too broad | `specificity>SPEC_MAX` | add specificity/full-distribution controls first |
| `RANDOM_MATCHED_CONTROL` | best write fails to beat matched random | `best_disc−random_disc<C_MARGIN` or `best_held−random_held<C_MARGIN` | not evidence beyond no-information writes |
| `SAME_READ_BASELINE_CONTROL` | oblique write fails to beat the exp-29 same-read baseline | `best_held−same_held<C_MARGIN` | oblique did not improve the exp-29 baseline |
| `POSCOND_READ_WRITE_CONTROL` | a position-conditioned fixed read admits an oblique write that controls the predicate with transfer, specificity, and margins | all gates above pass | carry fixed-read oblique into I2 |

**Thresholds** (all inherited from exp 30, unchanged): `VAR_MIN=0.05`,
`R2_MIN=0.50`, `C_MIN=0.50`, `C_MARGIN=0.20`, `RETENTION_MIN=0.50`,
`SPEC_MAX=0.35`, `SPEC_ROOM_MIN=0.01`, `OE_BAND=0.10`, `SEED_MAJORITY=3`,
`LEARN_NORM_MAX=1e4`, `LAM=1e-2`.

## Multi-Seed Aggregation and Decision precedence

A target aggregate is the branch in `≥3/4` seeds via `battery.majority_vote`
(else `SEED_UNSTABLE`). The top-level routing is `battery.first_precedence` over
both targets (no precedence implies one target dominates; both are load-bearing):

```text
OBS_EXACT_DRIFT > POSCOND_READ_WRITE_CONTROL > DISCOVERY_ONLY_WRITE
  > NONSPECIFIC_CONTROL > RANDOM_MATCHED_CONTROL > SAME_READ_BASELINE_CONTROL
  > NO_POSCOND_READ_WRITE_WORKS > NO_PATCH_ROOM > READ_NOT_DECODABLE
  > TARGET_VACUOUS
```

`DISCOVERY_ONLY_WRITE` is surfaced in the routing string as
`POSITION_ENTANGLED_WRITE`. The wrapped string is `<branch>(<targets>)`.

## Predictions

- **P1 (guards; enforced).** I0 routes `GO`, config guard passes, PairSet
  self-checks pass on both bins for all four seeds, `--selftest` passes.
- **P2 (read premise; likely).** Both in-place reads decode at both bins
  (`r2_disc`, `r2_held ≥ R2_MIN`), reproducing exp 31's in-place result, so
  `READ_NOT_DECODABLE` does **not** fire — the write question is reached. (This
  is the whole point of the read repair; if it fails, the substrate moved.)
- **P3 (room + calibration; likely).** Both targets have full-patch room and
  endpoint audit `≤ OE_BAND` on both bins, matching exp 29/30.
- **P4 (headline; uncertain).** Whether any registered write controls the
  predicate through the in-place read (`best_disc ≥ C_MIN`) and transfers
  (`best_held ≥ C_MIN`, `retention ≥ RETENTION_MIN`), specifically, beating
  random and same-read margins. A clean negative with room is **equally
  informative** and routes to I2/I3. No directional bet is registered.
- **P5 (descriptive held-inplace; report).** `best_held_inplace` is reported to
  separate a write-entanglement reading of `DISCOVERY_ONLY_WRITE` from a
  held-read-unwritability reading; it is never a verdict input.
- **P6 (controls; expected).** `phi3_all_neutral` stays vacuity-limited and
  `phi4_first_matched` stays non-decodable as an in-place read on the grouped
  bins; reported, never promoted to targets.

## Confound table — load-bearing quantity (author-side, per protocol)

The headline verdict turns on **`best_heldout`** (best discovery-selected write's
held-out control with the held read) for a positive, and on **`best_disc`** for
the `NO_POSCOND_READ_WRITE_WORKS` negative. Confound enumeration on both
directions:

**High `best_heldout` (→ positive).**

| confound producing high control | excluded by |
|---|---|
| broad distribution replacement, not predicate-specific | specificity gate (`SPEC_MAX`) on room-cleared non-targets + the same-read baseline margin |
| a no-information write scoring by chance | norm-matched random floor with `C_MARGIN` margin on both bins |
| same-read/same-write already sufficient (no oblique gain) | `SAME_READ_BASELINE_CONTROL` margin |
| the patch trivially sets the read but the predicate moves for unrelated reasons | control is normalized by measured full-patch room; endpoint audit gates calibration; no-op α=0 is exactly 0 |
| the held read leaks the answer (read fit on held) | the read is fit to decode observable `p_phi` on a train half and is **not** selected by control; control uses full-bin endpoints; the read object is the same one exp 31 validated as in-place decodable |

**Low `best_disc` (→ `NO_POSCOND_READ_WRITE_WORKS`).**

| confound producing low control | excluded by |
|---|---|
| genuinely not writable at rank-1 oblique (the construct) | this is the intended reading |
| under-powered write search (one-write failure ≠ no-write-exists) | five write families incl. a gradient-learned write; dose curve over six strengths; search budget reported. **Conclusion discipline:** a negative says "the registered rank-1 oblique write class fails," never "no write exists" |
| wrong strength | dose curve over `α ∈ {0,.25,.5,1,1.5,2}` reported, not a single α |
| in-place read itself bad | `READ_NOT_DECODABLE` gate has higher precedence; reached only when both reads decode |
| no room to move | `NO_PATCH_ROOM` gate has higher precedence |

A distinct confound for `DISCOVERY_ONLY_WRITE`: a disc-selected write may fail on
held not because the *write* is position-entangled but because the disc-tuned
write and the held read are geometrically mismatched. The descriptive
`best_held_inplace` (best held control by any candidate paired with `c_held`)
separates these — high `best_held_inplace` with low transferred control is genuine
write-entanglement; both low is held-read-unwritability. Reported, not routed (a
future split could promote it).

## Reliability baselines for the thresholds (author-side, per protocol)

Each load-bearing threshold is paired with the value a genuine positive (ceiling)
and pure noise (floor) reach:

- **`C_MIN=0.50` on `best_disc`/`best_heldout`.** Ceiling = full-patch control
  ≈ 1 (measured per bin as the closure denominator's own numerator); floor =
  no-op control = 0 (α=0, exact) and the norm-matched random-write control
  (measured). `C_MARGIN=0.20` is read against the measured random floor on both
  bins, so a positive sits above a measured floor and below the measured ceiling.
- **`R2_MIN=0.50` on the in-place reads.** Ceiling = exp 31's in-place held `R2`
  0.55–0.75 (the best a position-specific read does); floor = label-shuffle `R2`
  near 0 (exp 31 measured −0.07…−0.16). The premise gate sits between them.
- **`OE_BAND=0.10`.** Ceiling/floor are exp 29/30's measured endpoint drift
  (`phi1` ~0.009, `phi2` ~0.007), far inside the band; a breach means scoring
  moved.
- **`RETENTION_MIN=0.50`, `SPEC_MAX=0.35`.** Retention's denominator is the
  measured discovery control; specificity's reference is 0 (a perfectly specific
  write) with the random/same-read floors bounding "broad replacement."

## Measured-but-unadjudicated (author-side, per protocol)

`classify_target` reads only the gate quantities listed under Per-Seed Verdicts.
Everything else the script prints is descriptive and must not be over-weighted:

- `best_held_inplace` — separates write-entanglement from held-read-unwritability
  for a `DISCOVERY_ONLY_WRITE`; **not** a verdict input.
- `k_core`, `learned_norm`, the per-family intervention table, `spec_included` /
  `spec_skipped_low_room` — reported context; the verdict reads only the best /
  random / same-read family summaries and the gates.
- the dose curves — printed via the family table; only the best-α control enters
  the verdict.

## Halt Conditions

The run halts if: the checkpoint config differs from the registered `pstack-L4`
config; the I0 preflight artifact is missing or does not contain `I1 ROUTING:
GO`; or any PairSet known-answer self-check fails.

## Reviewable Failure Modes

- broad labels replacing the narrow construct: every verdict is a
  position-conditioned-read, L1, `pstack`, `m=3`, position-split **write-control**
  claim; a positive is "this rank-1 oblique write class controls the predicate
  here," never "the predicate is writable" in general;
- existential overreach from a finite write search: a negative is typed
  `NO_POSCOND_READ_WRITE_WORKS` ("the registered class fails"), never "no write
  exists";
- learned-write overfitting: separated by `DISCOVERY_ONLY_WRITE` (held transfer)
  and `LEARNED_DIVERGED` (norm guard);
- broad behavior replacement: separated by `NONSPECIFIC_CONTROL` + the held
  specificity score with the non-target room floor;
- no-information / baseline-already-sufficient: separated by
  `RANDOM_MATCHED_CONTROL` and `SAME_READ_BASELINE_CONTROL`;
- exact/observable mismatch: gated by `OBS_EXACT_DRIFT` before any geometry claim;
- read leakage: the in-place read is fit to decode observable `p_phi`, not to
  maximize control, and is the object exp 31 validated.

## Non-goals / Scope Guard

- No claim about whether the predicate is writable in general, at other layers,
  positions, or processes; no real-LLM claim.
- No read-varying / joint read-write pairs (that is I2; this experiment is the
  fixed-read baseline I2 must beat).
- No rank-k composition (a follow-up only after two rank-1 pairs pass).
- No new process training; existing `pstack-L4` only.
- No interventions beyond the registered rank-1 oblique class; the same-read/same-write
  and random arms are controls, not new primitives.
- Device: the registered run uses the living CPU evaluator path (as exp 30/31).
  An accelerator move would change float results and is out of scope for this
  baseline; if added later it is its own reviewed change, and the canonical
  output is reproduced on the device it was first run on.
