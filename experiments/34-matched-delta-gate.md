# Experiment 34 — I3-lite matched activation-delta feasibility gate on pstack-L4 — PRE-REGISTERED

**Script:** `scripts/interventions/i3_matched_delta_gate.py`.
**Output:** `out/exp34_pstack-L4.txt`.

**Status: pre-registered; awaiting pre-run review; NOT YET RUN.** This file and
its runnable script are the preregistration. Pause here before the first
claim-producing run.

**Decision form (filled by the run):**

```text
<branch>(phi1_next_closes[, phi2_net_return])
```

## Phase fit and why this is the next experiment

Exp 33 resolved the fixed-read baseline in the negative direction:
`NO_POSCOND_READ_WRITE_WORKS(phi1_next_closes,phi2_net_return)`. The repaired
position-conditioned reads decode with room, but the registered fixed-read
rank-1 oblique write menu does not stably control either target.

The tempting next step is a full I2 learned read/write-pair search. That is
prematurely expensive and too optimizer-limited: if it fails, the result may say
only that the learned pair search failed. This experiment inserts a cheaper,
more diagnostic I3-lite gate before that search. It asks whether **observed
activation deltas** from nearby source-target examples can move the predicate.
If they can at interpolation strength, then residual-level near-manifold writes
exist and a compact learned read/write approximation is worth spending on. If
they require `alpha > 1`, the result is only a direction signal. If they cannot
move the target under the registered controls, then more rank-1 search is weakly
motivated; the next branch should be patch point/path or consolidation, not
another optimizer-heavy write menu.

## Question

```text
On pstack-L4 at L1, m=3, for the exp-33 target predicates and position split,
do same-position, same-sign, target-marginal-matched observed activation deltas
move predicate probability on held-out positions at interpolation strength
(`alpha <= 1`) with specificity and a finite full-m-gram replacement audit?
```

This is a feasibility gate for near-manifold residual deltas, not a learned
read/write-pair experiment and not a general interchange proof.

## Registered Command

After preregistration review only:

```bash
uv run python scripts/interventions/i3_matched_delta_gate.py \
  --outdir out/pstack-L4 | tee out/exp34_pstack-L4.txt
```

Review-only checks:

```bash
uv run python scripts/interventions/i3_matched_delta_gate.py --selftest
uv run python -m py_compile interventions.py intervention_eval.py \
  scripts/interventions/i3_matched_delta_gate.py
```

## Scope Indices

| index | value |
|---|---|
| process/checkpoint | `pstack-L4`, same registered config as exp 33 |
| patch point | residual stream L1, prefix-wide PairSet patch up to the scored position |
| horizon | `m=3` within-horizon completion distribution |
| target predicates | `phi1_next_closes`, `phi2_net_return` |
| control predicates | `phi3_all_neutral`, `phi4_first_matched` plus full m-gram movement |
| intervention class | observed source-target activation deltas applied to target prefixes; no linear read/write covector |
| discovery bin | positions `{10,18}`, 512 pairs/seed |
| held-out bin | positions `{26,34}`, 1024 pairs/seed |
| seeds | `500..503` (fresh relative to exps 30-33) |
| exact oracle use | endpoint audit only; no matching, donor choice, strength choice, or verdict uses exact labels |
| device | live accelerator when available (`mps`, then `cuda`, else `cpu`); runtime detail only |

## Design

For each seed and target:

1. Build discovery and held PairSets with the exp-33 positions and sizes. Run
   `discover.self_checks` on both bins.
2. Compute observable target endpoints (`p_un`, `p_src`, `p_full`) and exact
   endpoint audit. Exact truth is used only for this audit.
3. Select eligible rows separately in each bin: within each position group, keep
   the top `35%` by `|p_src - p_un|` for the target predicate. This makes the
   experiment a high-difference feasibility gate, not a claim about all random
   pairs.
4. For each eligible row `i`, define observed prefix deltas
   `Delta_j = pref_src[j] - pref_tgt[j]` from rows at the same position.
5. Registered donor arms:
   - `own_delta`: donor `j=i`; this is the source-target interpolation ceiling
     and checks that the selected rows still have a usable full-patch path.
   - `matched_delta`: same position, same sign of `p_src-p_un`, same target-side
     `p_un` bin where possible, nearest `|p_src-p_un|` magnitude. This is the
     load-bearing arm. A self donor is not allowed for this arm.
   - `mismatched_delta`: same position but opposite sign of `p_src-p_un`, same
     target-side `p_un` bin where possible. This is the sign-control floor.
   - `shuffled_delta`: random eligible same-position non-self donor. This is the
     no-information observed-delta floor.
6. Apply each arm as `pref_tgt[i] + alpha * Delta_j` for
   `alpha in {0,.25,.5,1,1.5,2}` and score the best observable predicate control
   on discovery and held separately. A positive `MATCHED_DELTA_CONTROL` requires
   the selected matched-delta strength to be interpolation-scale (`alpha <= 1`)
   on both bins; `alpha > 1` routes to `EXTRAPOLATED_DELTA_CONTROL`. No parameter
   is learned from exact labels.
7. For the held matched arm at its selected strength, score non-target predicate
   specificity and full m-gram movement toward source. A positive requires this
   full m-gram audit to be finite.

## Scores

Predicate control is the same closure fraction as exp 33:

```text
c(P) = [MSE(p_un, p_src) - MSE(p_P, p_src)] / [MSE(p_un, p_src) - MSE(p_full, p_src)]
```

`room <= ROOM_TOL` means `NO_PATCH_ROOM`. No-op is 0, full/own source delta is
the measured ceiling. Full m-gram movement is the analogous source-distribution
KL closure over the full `m=3` continuation distribution, using the full patch as
its denominator; if that denominator is absent, the branch is
`MGRAM_UNAUDITABLE`, not a positive. Specificity is max absolute non-target
predicate control over room-cleared non-targets.

## Per-Seed Verdicts

Exactly one branch per `(target, seed)` by the script's `classify_target`.

| branch | plain-language gloss | condition | routes to |
|---|---|---|---|
| `TARGET_VACUOUS` | target too flat on the eligible rows | `std_disc<VAR_MIN` or `std_held<VAR_MIN` | change target/subset |
| `LOW_MATCH_SUPPORT` | too few eligible high-difference rows or no non-self donor support | fewer than `24` eligible rows per position, or any eligible row lacks a non-self same-sign matched donor, opposite-sign mismatched donor, or non-self shuffled donor | increase sample or change target |
| `NO_PATCH_ROOM` | full/source interpolation cannot move the selected rows | room at either bin `<= ROOM_TOL` | change target/patch point |
| `OBS_EXACT_DRIFT` | observable endpoint p_phi is not calibrated | endpoint audit `> OE_BAND` | repair scoring before geometry |
| `DELTA_GATE_INVALID` | even own source-target deltas fail the predicate gate | `own_delta` control `< C_MIN` in either bin | L1 residual delta gate invalid for this target |
| `NO_MATCHED_DELTA_CONTROL` | matched observed deltas do not control on discovery | `matched_disc<C_MIN` | do not spend on learned I2 without a new diagnostic |
| `DISCOVERY_ONLY_DELTA` | matched deltas work only in-split or fail retention | held control `<C_MIN` or retention `<RETENTION_MIN` | position-specific/interchange overfit |
| `NONSPECIFIC_DELTA` | matched deltas move non-target predicates too much | specificity `> SPEC_MAX` | needs better matching/path before primitive claim |
| `MGRAM_UNAUDITABLE` | full m-gram replacement control has no finite denominator | held full m-gram closure is NaN | cannot certify predicate-specific movement |
| `BROAD_MGRAM_REPLACEMENT` | predicate movement is near full-distribution replacement | held full m-gram closure `> MGRAM_MAX` | not predicate-specific enough |
| `EXTRAPOLATED_DELTA_CONTROL` | matched deltas control only beyond interpolation scale | selected matched `alpha_disc>1` or `alpha_held>1` | direction signal only; not near-manifold success |
| `SHUFFLED_MATCHED_CONTROL` | matched deltas fail to beat observed-delta floors | matched minus max(mismatched, shuffled) `< C_MARGIN` on either bin | no evidence beyond no-information deltas |
| `MATCHED_DELTA_CONTROL` | matched observed deltas control with transfer and controls at interpolation strength | all gates above pass, including `alpha<=1` on both bins | near-manifold residual write signal exists; consider compact I2 |

Thresholds: `VAR_MIN=0.05`, `C_MIN=0.50`, `C_MARGIN=0.20`,
`RETENTION_MIN=0.50`, `SPEC_MAX=0.35`, `SPEC_ROOM_MIN=0.01`,
`MGRAM_MAX=0.85`, `INTERP_ALPHA_MAX=1.0`, `OE_BAND=0.10`,
`SEED_MAJORITY=3`.

## Multi-Seed Aggregation and Decision Precedence

A target aggregate is the branch in `>=3/4` seeds via `battery.majority_vote`,
else `SEED_UNSTABLE`. The top-level decision uses `battery.first_precedence`:

```text
OBS_EXACT_DRIFT > MATCHED_DELTA_CONTROL > BROAD_MGRAM_REPLACEMENT
  > MGRAM_UNAUDITABLE > EXTRAPOLATED_DELTA_CONTROL > NONSPECIFIC_DELTA
  > DISCOVERY_ONLY_DELTA > SHUFFLED_MATCHED_CONTROL
  > NO_MATCHED_DELTA_CONTROL > DELTA_GATE_INVALID > NO_PATCH_ROOM
  > LOW_MATCH_SUPPORT > TARGET_VACUOUS
```

The decision string is `<branch>(<targets>)`. A stable positive on either target
is sufficient to justify a compact follow-up on that target; per-target
aggregates remain printed so a failure on the other target is not promoted to a
success.

## Predictions

- **P1 (guards; enforced).** Config guard, I0 artifact gate, PairSet self-checks,
  `--selftest`, and `py_compile` pass.
- **P2 (eligibility; likely).** The top-35% high-difference subset leaves enough
  eligible rows per position in both bins, so `LOW_MATCH_SUPPORT` does not fire.
- **P3 (room/calibration; likely).** `own_delta` and full/source interpolation
  have predicate room and endpoint audit remains inside `OE_BAND`.
- **P4 (headline; uncertain).** `matched_delta` may or may not control the
  predicates. Both outcomes are useful: a positive justifies a compact I2
  approximation; a clean negative argues against an expensive learned pair run
  without a new patch-point/path diagnostic.
- **P5 (controls; enforced).** A positive must beat mismatched and shuffled
  observed-delta controls by `C_MARGIN`, pass non-target specificity, have a
  finite full m-gram audit, avoid near-full m-gram replacement, and use
  interpolation-scale matched strengths (`alpha <= 1`) on both bins.

## Confound Table — Load-Bearing Quantity

The headline positive turns on `matched_heldout`, the held predicate control of
the matched observed-delta arm.

| confound producing high `matched_heldout` | excluded by |
|---|---|
| own/source interpolation trivially works, but donor matching adds nothing | matched arm must beat mismatched and shuffled observed-delta floors by `C_MARGIN` |
| broad distribution replacement, not predicate-specific movement | non-target specificity gate plus finite full m-gram replacement gate |
| extrapolated direction works but interpolation-scale delta does not | `EXTRAPOLATED_DELTA_CONTROL` branch; positive requires `alpha <= 1` on both bins |
| position-specific donor overfit | held-out position bin and retention gate |
| exact labels leaked into matching | matching uses observable `p_un`, `p_src` only; exact truth is endpoint audit only |
| tiny denominator exaggerates control | full-patch room gate, finite m-gram audit, and high-difference eligible subset reported |

The headline negative turns on low `matched_disc` or failed held transfer.

| confound producing low matched control | excluded by |
|---|---|
| no residual delta can move the target | not concluded; `own_delta` is a higher-priority ceiling gate |
| wrong donor matching rule | not fully excluded; this is the registered finite matching rule |
| not enough high-difference examples or donor alternatives | per-position `LOW_MATCH_SUPPORT` gate plus non-self donor support check |
| no target room or observable/exact drift | higher-priority `NO_PATCH_ROOM` and `OBS_EXACT_DRIFT` gates |
| wrong strength | six-point dose curve over `alpha` |

Conclusion discipline: a negative means the registered matched observed-delta
rule failed. It does not prove no interchange intervention or no residual write
exists.

## Reliability Baselines for Thresholds

- `C_MIN=0.50`: ceiling is `own_delta`/full source interpolation on the same
  eligible rows; floors are `mismatched_delta` and `shuffled_delta`.
- `C_MARGIN=0.20`: positive matched control must exceed both observed-delta
  floors on discovery and held.
- `SPEC_MAX=0.35`: zero is ideal predicate specificity; non-target predicate
  closure above the threshold routes away from a primitive claim.
- `INTERP_ALPHA_MAX=1.0`: the observed-delta success claim is interpolation
  scale only. Control first appearing at `alpha>1` is reported as
  `EXTRAPOLATED_DELTA_CONTROL`, a direction signal rather than a positive.
- `MGRAM_MAX=0.85`: a near-full m-gram source replacement is too broad to count
  as predicate-specific control, even if the predicate moves. Finite full m-gram
  room is required; otherwise the branch is `MGRAM_UNAUDITABLE`.
- `OE_BAND=0.10`: inherited endpoint audit band; exp 33 observed much lower
  drift, so a breach is methodological.

## Measured-but-Unadjudicated

The script prints per-arm discovery/held selected alpha, held control, full
m-gram control, specificity, retention, total and per-position eligible counts,
room, and endpoint audit. The verdict reads only the quantities in
`classify_target`; dose curves are used only through the best strength per arm.
Donor identities and p_phi bins are implementation details of the registered
matching rule, not interpreted post hoc.

## Halt Conditions

The run halts if the checkpoint config differs from registered `pstack-L4`, the
I0 preflight artifact is missing or lacks the intervention `GO` route, or any
PairSet known-answer self-check fails.

## Non-goals / Scope Guard

- No learned read/write pair, no rank-k composition, no new process training.
- No claim about all pairs; this is scoped to the high predicate-difference
  eligible subset.
- No claim that matched deltas identify the mechanism; they are a feasibility
  probe for near-manifold residual writes.
- No real-LLM claim.
- Device choice is engineering/runtime only and is recorded in the output.

## Reviewable Failure Modes

- Expensive-but-ambiguous search creep: this preregistration avoids a learned
  optimizer as the load-bearing object.
- Broad label creep: branch names say matched observed deltas, not general
  writability.
- Positive from extrapolated deltas: separated into `EXTRAPOLATED_DELTA_CONTROL`
  and not counted as near-manifold success.
- Positive from full replacement: controlled by finite full m-gram and non-target
  gates.
- No-information donor success: controlled by non-self shuffled and opposite-sign
  mismatched observed deltas.
- Negative overreach: scoped to this donor rule, layer, horizon, positions,
  target subset, and checkpoint.
