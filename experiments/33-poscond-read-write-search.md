# Experiment 33 — I1′ position-conditioned fixed-read oblique write search on pstack-L4 — CONCLUDED

**Script:** `scripts/interventions/i1prime_poscond_read_write_search.py`
(harness promoted to the living `intervention_eval.py`).
**Output:** `out/exp33_pstack-L4.txt`.

**Status: concluded.** Pre-registration review completed, the first completed
claim-producing run is recorded in `out/exp33_pstack-L4.txt`, and the result is
written below. The detailed conclusion here is canonical; index/docstring entries
are short pointers only.

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
   (`build_write_candidates`): same-read/same-write for the discovery bin,
   CEGAR-core directions and the core-projected read, predicate-delta writes,
   random writes; plus one learned write (`learn_write`, read fixed,
   `w = c + u_perp` so `c·w = 1`, observable predicate MSE to source on
   discovery pairs). The held same-read/same-write baseline is rebuilt with
   `c_held` in step 7; it is not a transfer of the discovery read.
5. **Selection on discovery.** Each candidate's dose curve over
   `α ∈ {0,.25,.5,1,1.5,2}` is scored on the discovery bin with `c_disc`; per
   family the best-α control is kept; the best non-random family by discovery
   control is the headline write `w*`.
6. **Transfer to held.** Score `w*` on the held bin with **`c_held`** (the held
   in-place read) → `best_heldout`; `retention = best_heldout / best_disc`.
   Specificity is the max absolute non-target predicate control of `w*` on held
   (non-targets with held full-patch room ≥ `0.01`).
7. **Baselines.** Same-read/same-write held control (`same_heldout =`
   `oblique(c_held,c_held)`), random-write held control (`random_heldout`) and
   discovery control (`random_disc`), full-patch room (both bins), endpoint audit
   (both bins). This makes `SAME_READ_BASELINE_CONTROL` the real held-side
   exp-29 baseline, not the cross-pair `oblique(c_held,c_disc)`.
8. **Held-read writability split.** `best_held_inplace`: the best held-bin
   control achievable by registered **non-random** candidate writes paired with
   `c_held` (including held same-read/same-write). It is a verdict input only
   when a discovery-selected write fails held transfer; random writes are excluded
   so no-information writes cannot make the held read look writable.

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
| `HELD_READ_NOT_WRITABLE` | a discovery write exists, but no registered non-random write controls through the held in-place read | `best_disc≥C_MIN`, (`best_heldout<C_MIN` or NaN, or `retention<RETENTION_MIN` or NaN), and `best_held_inplace<C_MIN` or NaN | held positions are readable-but-not-writable for this rank-1 oblique menu → I2 read/write-pair or I3 interchange |
| `DISCOVERY_ONLY_WRITE` | the discovery-selected write does not transfer, while some non-random write still controls through the held read | `best_disc≥C_MIN`, (`best_heldout<C_MIN` or NaN, or `retention<RETENTION_MIN` or NaN), and `best_held_inplace≥C_MIN` | measured position-entangled write under this menu; I2/I3, not a primitive |
| `NO_POSCOND_READ_WRITE_WORKS` | no registered write controls through the in-place read despite room | `best_disc<C_MIN` | readable-not-writable at rank-1 oblique → I2 read-pair / I3 interchange |
| `NONSPECIFIC_CONTROL` | target control is too broad | `specificity>SPEC_MAX` | add specificity/full-distribution controls first |
| `RANDOM_MATCHED_CONTROL` | best write fails to beat matched random | `best_disc−random_disc<C_MARGIN` or `best_heldout−random_heldout<C_MARGIN` | not evidence beyond no-information writes |
| `SAME_READ_BASELINE_CONTROL` | oblique write fails to beat the exp-29 same-read baseline | `best_heldout−same_heldout<C_MARGIN` | oblique did not improve the exp-29 baseline |
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
OBS_EXACT_DRIFT > POSCOND_READ_WRITE_CONTROL > HELD_READ_NOT_WRITABLE
  > DISCOVERY_ONLY_WRITE > NONSPECIFIC_CONTROL > RANDOM_MATCHED_CONTROL
  > SAME_READ_BASELINE_CONTROL
  > NO_POSCOND_READ_WRITE_WORKS > NO_PATCH_ROOM > READ_NOT_DECODABLE
  > TARGET_VACUOUS
```

The wrapped string is `<branch>(<targets>)`; branch labels are emitted as
registered, with no mechanism-strengthening relabel.

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
  (`best_heldout ≥ C_MIN`, `retention ≥ RETENTION_MIN`), specifically, beating
  random and same-read margins. A clean negative with room is **equally
  informative** and routes to I2/I3. No directional bet is registered.
- **P5 (held-inplace split; enforced on transfer failures).**
  `best_held_inplace` separates two transfer-failure mechanisms: high
  `best_held_inplace` routes to `DISCOVERY_ONLY_WRITE` (the held read is writable
  by some non-random candidate, but not by the transferred `w*`); low or NaN
  routes to `HELD_READ_NOT_WRITABLE` (no registered non-random held-side write
  controls through `c_held`).
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
| same-read/same-write already sufficient (no oblique gain) | `SAME_READ_BASELINE_CONTROL` margin against the held-side `oblique(c_held,c_held)` baseline |
| the patch trivially sets the read but the predicate moves for unrelated reasons | control is normalized by measured full-patch room; endpoint audit gates calibration; no-op α=0 is exactly 0 |
| the held read leaks the answer (read fit on held) | leakage would inflate the real held same-read/same-write and random arms too, raising the margin a positive must clear; the read is fit to decode observable `p_phi` on a train half and is **not** selected by control |

**Low `best_disc` (→ `NO_POSCOND_READ_WRITE_WORKS`).**

| confound producing low control | excluded by |
|---|---|
| genuinely not writable at rank-1 oblique (the construct) | this is the intended reading |
| under-powered write search (one-write failure ≠ no-write-exists) | five write families incl. a gradient-learned write; dose curve over six strengths; search budget reported. **Conclusion discipline:** a negative says "the registered rank-1 oblique write class fails," never "no write exists" |
| wrong strength | dose curve over `α ∈ {0,.25,.5,1,1.5,2}` reported, not a single α |
| in-place read itself bad | `READ_NOT_DECODABLE` gate has higher precedence; reached only when both reads decode |
| no room to move | `NO_PATCH_ROOM` gate has higher precedence |

A distinct confound for transfer failure is now routed explicitly: a
disc-selected write may fail on held not because the *write* is
position-entangled, but because the held read has no registered non-random
writable handle. `best_held_inplace` (best held control by non-random
candidates paired with `c_held`) separates these: high `best_held_inplace` with
low transferred control is measured position-entangled write; low or NaN is
`HELD_READ_NOT_WRITABLE`.

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
Everything else the script prints is descriptive and must not be over-weighted.
`best_held_inplace` is no longer descriptive-only; it is a branch input for the
transfer-failure split.

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
  maximize control; leakage cannot manufacture a positive because the real held
  same-read/same-write and random arms raise the margin a positive must clear.

## Pre-run amendment — accelerator execution

After review, before the first completed claim-producing run, the device policy
was amended to follow the repository working norm that Torch code should use
accelerators when available. The script now loads the model on `pick_device()`
(`mps` on Apple silicon, then `cuda`, else `cpu`) and keeps the promoted
`intervention_eval.py` completion scorer plus the experiment-local learned-write
optimizer on that device. `PairSet` residual caches remain CPU records, as in
the shared `midstream.stream_to` contract, and patch/read summaries are converted
back to NumPy for the registered verdict logic.

This is not a scientific condition: CPU vs MPS is not treated as a separate
construct, branch, or scope axis. The registered thresholds, baselines, candidate
menu, verdict partition, and exact endpoint audit are unchanged. The prior
CPU-only parity check was removed; review-only checks are the script selftest and
`py_compile`, and the claim-producing output records the selected device.

## Non-goals / Scope Guard

- No claim about whether the predicate is writable in general, at other layers,
  positions, or processes; no real-LLM claim.
- No read-varying / joint read-write pairs (that is I2; this experiment is the
  fixed-read baseline I2 must beat).
- No rank-k composition (a follow-up only after two rank-1 pairs pass).
- No new process training; existing `pstack-L4` only.
- No interventions beyond the registered rank-1 oblique class; the same-read/same-write
  and random arms are controls, not new primitives.
- Device: the registered run uses the live accelerator when available
  (`mps`, then `cuda`, else `cpu`). Device choice is an engineering/runtime
  detail, not a scientific scope index; conclusions must not depend on CPU vs
  MPS float noise. The output records the selected device.

## Results

Run artifact: `out/exp33_pstack-L4.txt` (completed on `device=mps`).

```text
DECISION: NO_POSCOND_READ_WRITE_WORKS(phi1_next_closes,phi2_net_return)
```

Multi-seed aggregation:

| target | per-seed verdicts | aggregate |
|---|---|---|
| `phi1_next_closes` | `HELD_READ_NOT_WRITABLE`, `NO_POSCOND_READ_WRITE_WORKS`, `NO_POSCOND_READ_WRITE_WORKS`, `NO_POSCOND_READ_WRITE_WORKS` | `NO_POSCOND_READ_WRITE_WORKS` |
| `phi2_net_return` | `NO_POSCOND_READ_WRITE_WORKS`, `NO_POSCOND_READ_WRITE_WORKS`, `NO_POSCOND_READ_WRITE_WORKS`, `NO_POSCOND_READ_WRITE_WORKS` | `NO_POSCOND_READ_WRITE_WORKS` |

### Registered gates

The run reached the write question. PairSet known-answer self-checks passed for
every target/bin, the target predicates were not vacuous, the repaired in-place
reads decoded on both bins, full-patch room was present, and observable endpoints
remained calibrated to exact predicate truth.

Target read and room ranges:

| target | in-place `R2` disc | in-place `R2` held | room disc | room held | endpoint audit disc/held |
|---|---:|---:|---:|---:|---:|
| `phi1_next_closes` | 0.53-0.64 | 0.56-0.64 | 0.0642-0.0696 | 0.0725-0.0807 | 0.009-0.010 / 0.009-0.010 |
| `phi2_net_return` | 0.65-0.72 | 0.68-0.75 | 0.0910-0.0970 | 0.1008-0.1105 | 0.008-0.009 / 0.007-0.008 |

Control predicates stayed controls: `phi3_all_neutral` remained vacuity-limited
(`std` about 0.014-0.016), and `phi4_first_matched` did not decode under the
registered in-place read gate (`R2` at most 0.34). They were not promoted to
targets.

### Write result

The fixed-read rank-1 oblique write menu did not produce stable discovery
control for either target. `phi2_net_return` had `best_disc < C_MIN` in all four
seeds (0.27-0.42). `phi1_next_closes` had one discovery-success seed
(seed 400, `best_disc=0.52`), but that write failed held transfer
(`best_heldout=0.19`, `retention=0.36`) and no registered non-random held-side
write controlled through `c_held` (`best_held_inplace=0.19`), so that seed routed
`HELD_READ_NOT_WRITABLE`. The other three `phi1` seeds had `best_disc < C_MIN`
(0.19-0.49). By the registered 3/4 majority rule, both targets aggregate to
`NO_POSCOND_READ_WRITE_WORKS`.

The controls did not mask a positive. Held same-read/same-write control was
near zero throughout (reported as 0.000-0.001), so the real exp-29 held baseline
was not already sufficient. Matched random held control was also low (worst
0.066 for `phi1`, 0.012 for `phi2`). Some non-target specificity scores were
large on weak candidate arms, but specificity was not load-bearing because no
candidate reached the registered positive gates.

### Result-side confound check

High-`best_heldout` positive confounds are moot: there was no stable positive to
explain. The negative reached the intended construct because the higher-priority
escape hatches did not fire: the repaired reads decoded in place, full-patch room
was nonzero, endpoint drift stayed far below `OE_BAND`, same-read and random
baselines were low, and the dose curve was evaluated over the registered six
strengths. The remaining live limitation is the registered finite-menu scope:
this result says the tested rank-1 oblique fixed-read write class failed; it does
not prove that no write, no joint read/write pair, no rank-k composition, or no
other patch point can control the predicates.

### Conclusion

Exp 33 resolves the exp-30 ambiguity in the negative direction for the fixed-read
baseline. The exp-30 write question was previously blocked by read transport; in
I1′ the read was repaired to the position-conditioned in-place object validated
by exps 31/32, and the target predicates were readable with full-patch room. Even
under that repaired read, the registered fixed-read rank-1 oblique write menu did
not stably control either predicate.

The carried-forward claim is narrow:

```text
NO_POSCOND_READ_WRITE_WORKS(phi1_next_closes,phi2_net_return)
```

On `pstack-L4`, at L1, `m=3`, positions `{10,18}->{26,34}`, with observable
position-conditioned in-place predicate reads and the exp-30 fixed-read oblique
write arms, the read is decodable but the registered write class fails. This is
not a general unwritability theorem. It is the fixed-read baseline I2 needed to
beat.

Routing: do **not** carry fixed-read rank-1 oblique writes forward as a
successful primitive for these predicates. The next intervention-class step may
spend complexity on I2 read/write-pair search or I3 interchange/path
interventions before introducing a new process or rank-k composition.
