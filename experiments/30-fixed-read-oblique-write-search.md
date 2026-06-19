# Experiment 30 — I1 fixed-read oblique write search on pstack — PRE-REGISTRATION

**Script:** `scripts/interventions/i1_fixed_read_write_search.py`.

**Status: pre-registered; NOT YET RUN.** This preregistration is the pause
point before the first claim-producing run. No result or conclusion should be
added until pre-registration review approves both this writeup and the script.

## Question

Experiment 29 showed that `phi1_next_closes` and `phi2_net_return` on
`pstack-L4` were linearly readable and exact-calibrated, but the registered
same-read/same-write rank-1 patch did not move their predicate probabilities.
That result typed a decode/control split for the tested patch class. It did not
show that the predicates are unwritable.

I1 asks the next narrower intervention-class question from
`INTERVENTION_CLASS_BENCHMARK.md`:

```text
Given the affine predicate readout from exp 29 as a fixed read, can a separate
write direction causally control the same predicate probability?
```

The experiment is about a fixed-read rank-1 oblique intervention class at L1 on
the existing `pstack-L4` checkpoint. It is not a predicate-CEGAR experiment, a
general claim about pstack predicates, or a proof that the affine readout is a
latent state variable.

## Registered Command

After preregistration review only:

```bash
uv run python scripts/interventions/i1_fixed_read_write_search.py \
  --outdir out/pstack-L4 | tee out/exp30_pstack-L4.txt
```

Review-only checks:

```bash
uv run python scripts/interventions/i1_fixed_read_write_search.py --selftest
uv run python -m py_compile interventions.py scripts/interventions/i1_fixed_read_write_search.py
```

The script halts unless the I0 artifact `out/i0_preflight_pstack-L4.txt`
contains `I1 ROUTING: GO`.

## Scope Indices

| index | value |
|---|---|
| process/checkpoint | `pstack-L4`, registered config from exp 29/I0 |
| patch point | residual stream L1, prefix-wide PairSet patch |
| horizon | `m=3` within-horizon completion distribution |
| target predicates | `phi1_next_closes`, `phi2_net_return` |
| control predicates | `phi3_all_neutral`, `phi4_first_matched` reported for vacuity/interpreter limits |
| read class | affine ridge readout from centered L1 residuals to observable model `p_phi` |
| intervention class | rank-1 oblique patch `P = c w^T / (c.w)`, fixed read `c`, searched write `w` |
| discovery split | positions `{10,18}`, 512 pairs per seed |
| held-out transfer split | positions `{26,34}`, 1024 pairs per seed |
| seeds | `400..403`, fresh relative to exp 29/I0 |
| exact oracle use | endpoint audit only; no direction, threshold, or selection uses exact labels |

## Design

For each seed and each target predicate:

1. Fit the fixed read `c_phi` by affine ridge regression from discovery
   residual rows to observable unpatched model `p_phi`.
2. Require the read to decode on discovery and held-out positions before a
   write failure is interpretable.
3. Build write candidates with the I0 library helpers:
   same-read/same-write baseline, CEGAR-core-constrained writes, predicate
   delta-stratified residual writes, random matched writes, and one learned
   write optimized with the read fixed.
4. Select write and strength on discovery by observable predicate control.
   Strength is swept over `{0, .25, .5, 1, 1.5, 2}`.
5. Evaluate the selected family on held-out positions, with specificity scored
   as maximum non-target predicate movement.
6. Audit source/target observable endpoints against exact predicate truth.

The learned write objective minimizes observable predicate MSE to the source
run on discovery pairs. It uses a fixed-read parameterization `w = c + u_perp`
so `c.w = 1` throughout. Exact process probabilities are not used in training.

## Scores

Predicate control is the I0/exp29 closure score:

```text
c_phi(P) =
  [MSE(p_un, p_src) - MSE(p_P, p_src)]
  /
  [MSE(p_un, p_src) - MSE(p_full, p_src)].
```

`NO_PATCH_ROOM` fires when the denominator is not positive. The no-op score is
near 0, the full-patch ceiling is near 1, and random writes are searched with
the same strength grid as non-random writes.

Specificity is:

```text
max over non-target registered predicates of |c_psi(P)|
```

using only predicates with finite full-patch room. Lower is better. This is a
predicate-level specificity check, not a full-distribution noninterference
theorem.

Exact endpoint audit is:

```text
max(mean |p_un_model - p_tgt_exact|,
    mean |p_src_model - p_src_exact|)
```

It audits observable endpoints only. It does not define exact truth for
off-manifold patched activations.

## Per-Seed Verdicts

For each `(target, seed)`, the script assigns exactly one branch:

| branch | condition | interpretation |
|---|---|---|
| `TARGET_VACUOUS` | discovery or held-out predicate std `< .05` | target is too flat for I1 |
| `FIXED_READ_NOT_DECODABLE` | discovery affine `R2 < .50` | fixed read not established |
| `FIXED_READ_NOT_TRANSPORTED` | discovery read decodes, held-out read `R2 < .50` | read is position-limited |
| `NO_PATCH_ROOM` | discovery or held-out full patch has no predicate room | target/patch point is non-diagnostic |
| `OBS_EXACT_DRIFT` | endpoint audit `> .10` on either split | observable endpoints cannot support intervention geometry |
| `DISCOVERY_ONLY_WRITE` | discovery control passes but held-out control `< .50` or retention `< .50` | write is position-entangled or overfit |
| `NO_FIXED_READ_WRITE_WORKS` | full-patch room and calibrated endpoints exist, but held-out best control `< .50` | fixed-read write class fails |
| `NONSPECIFIC_CONTROL` | held-out target control passes but specificity `> .35` | target movement is too broad |
| `RANDOM_MATCHED_CONTROL` | best non-random write fails to beat matched random by `.20` on discovery or held-out | search did not beat no-information writes |
| `SAME_READ_BASELINE_CONTROL` | best non-random write fails to beat same-read baseline by `.20` held-out | oblique write did not improve exp29-style baseline |
| `FIXED_READ_WRITE_CONTROL` | held-out target control `>= .50`, transfer retained, specific, exact-sound, and beats random/same-read margins | fixed-read oblique write succeeds for this target |

## Multi-Seed Aggregation

A target aggregate reproduces when the same per-seed branch appears in at
least `3/4` seeds; otherwise it is `SEED_UNSTABLE`.

Top-level decision precedence:

1. `OBS_EXACT_DRIFT(phi...)`
2. `FIXED_READ_WRITE_CONTROL(phi...)`
3. `POSITION_ENTANGLED_WRITE(phi...)`
4. `NONSPECIFIC_CONTROL(phi...)`
5. `RANDOM_MATCHED_CONTROL(phi...)`
6. `SAME_READ_BASELINE_CONTROL(phi...)`
7. `NO_FIXED_READ_WRITE_WORKS(phi...)`
8. `NO_PATCH_ROOM(phi...)`
9. `FIXED_READ_LIMIT(phi...)`
10. `TARGET_VACUOUS(phi...)`
11. `SEED_UNSTABLE`

## Predictions

- **P1 (guards; enforced).** I0 routes `GO`, config guard passes, PairSet
  self-checks pass, and the I1 selftest passes.
- **P2 (read continuity; likely).** The fixed affine reads for `phi1` and
  `phi2` decode on discovery and held-out positions. Failure is a read
  transport limit, not a write result.
- **P3 (room and calibration; likely).** Both targets have full-patch
  predicate room and endpoint audit `<= .10`, matching exp 29/I0.
- **P4 (headline; uncertain).** At least one target may admit a separate write
  that controls held-out predicate probability. A clean negative with room and
  calibration is equally informative: it routes away from fixed-read write
  search toward I2/read-pair or path/interchange follow-up.
- **P5 (controls; expected).** `phi3_all_neutral` remains vacuity-limited and
  `phi4_first_matched` remains interpreter-limited under the reported
  discovery/held-out read checks. If either becomes decodable and non-vacuous,
  it is reported as a control observation only, not added as a target.

## Interpretation Map

| decision | next action |
|---|---|
| `FIXED_READ_WRITE_CONTROL` | carry fixed-read oblique writes into the next richer intervention comparison |
| `POSITION_ENTANGLED_WRITE` | treat learned/delta/core writes as position-entangled; do not promote the primitive |
| `NONSPECIFIC_CONTROL` | add stronger specificity/full-distribution controls before using the intervention |
| `RANDOM_MATCHED_CONTROL` or `SAME_READ_BASELINE_CONTROL` | do not count the write search as evidence beyond controls |
| `NO_FIXED_READ_WRITE_WORKS` | fixed-read write parameterization failed despite room; route to I2 or path/interchange |
| `NO_PATCH_ROOM` | change target, patch point, or process before interpreting write geometry |
| `FIXED_READ_LIMIT` | the fixed affine read is not a stable causal read under this split |
| `OBS_EXACT_DRIFT` | repair predicate scoring/calibration before further intervention claims |

## Halt Conditions

The run halts if:

- the checkpoint config differs from the registered `pstack-L4` config;
- I0 did not route `GO`;
- PairSet known-answer self-checks fail.

The learned write arm is typed `LEARNED_DIVERGED` and cannot carry a positive
decision if its norm becomes non-finite or exceeds the registered guard
(`1e4`).

## Reviewable Failure Modes

- broad labels replacing the narrow construct: all verdicts are fixed-read,
  L1, `pstack`, `m=3`, position-split claims;
- learned write overfitting: separated by `DISCOVERY_ONLY_WRITE`;
- broad behavior replacement: separated by `NONSPECIFIC_CONTROL`;
- no-information search success: separated by `RANDOM_MATCHED_CONTROL`;
- exp29 baseline not improved: separated by `SAME_READ_BASELINE_CONTROL`;
- exact/observable mismatch: separated before any geometry claim by
  `OBS_EXACT_DRIFT`.
