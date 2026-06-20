# Experiment 30 — I1 fixed-read oblique write search on pstack — CONCLUDED

**Script:** `scripts/interventions/i1_fixed_read_write_search.py`.

**Status: concluded.** Canonical output: `out/exp30_pstack-L4.txt`.

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
max over included non-target registered predicates of |c_psi(P)|
```

where a non-target predicate is included only when its held-out full-patch
predicate room is at least `0.01`. Low-room non-targets are still reported as
controls, but are excluded from this closure-fraction specificity score because
their denominator would amplify tiny absolute marginal changes. Lower is
better. This is a predicate-level specificity check, not a full-distribution
noninterference theorem.

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
| `NONSPECIFIC_CONTROL` | held-out target control passes but included-predicate specificity `> .35` | target movement is too broad |
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

The headline decision is per-target-led: the highest-precedence reproduced
target branch is printed first, while the per-target aggregate table remains
load-bearing for secondary positives or negatives.

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
  discovery/held-out read checks. `phi3` is expected to be excluded from the
  closure-fraction specificity score unless its held-out full-patch room rises
  above `0.01`. If either control becomes decodable and non-vacuous, it is
  reported as a control observation only, not added as a target.

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
- low-room denominator amplification in specificity: controlled by the
  registered non-target room floor (`0.01`) and printed included/skipped sets;
- no-information search success: separated by `RANDOM_MATCHED_CONTROL`;
- exp29 baseline not improved: separated by `SAME_READ_BASELINE_CONTROL`;
- exact/observable mismatch: separated before any geometry claim by
  `OBS_EXACT_DRIFT`.

---

## Results

Registered command, with the implementation note below:

```bash
uv run python scripts/interventions/i1_fixed_read_write_search.py \
  --outdir out/pstack-L4 | tee out/exp30_pstack-L4.txt
```

Decision:

```text
FIXED_READ_LIMIT(phi1_next_closes,phi2_net_return)
```

Both target aggregates reproduced `FIXED_READ_NOT_TRANSPORTED` in all four
seeds:

| target | per-seed verdicts | aggregate |
|---|---|---|
| `phi1_next_closes` | 4/4 `FIXED_READ_NOT_TRANSPORTED` | `FIXED_READ_NOT_TRANSPORTED` |
| `phi2_net_return` | 4/4 `FIXED_READ_NOT_TRANSPORTED` | `FIXED_READ_NOT_TRANSPORTED` |

The blocking measured quantity was the held-out-position affine read `R2`.
Discovery reads decoded, but the same fixed affine reads did not transport
from positions `{10,18}` to `{26,34}`:

| target | discovery `R2` range | held-out `R2` range |
|---|---:|---:|
| `phi1_next_closes` | 0.53–0.64 | 0.06–0.24 |
| `phi2_net_return` | 0.65–0.72 | 0.16–0.26 |

The anti-vacuity and calibration gates were live. Held-out full-patch room was
positive on every seed (`phi1`: 0.0725–0.0807; `phi2`: 0.1008–0.1105), and
endpoint audit stayed within the registered `0.10` band (`phi1`: 0.009–0.010;
`phi2`: 0.007–0.008). Thus the failure is not `NO_PATCH_ROOM` and not
`OBS_EXACT_DRIFT`.

Write-arm results are reported in the output table but are not allowed to carry
a write verdict because the read-transport gate has higher precedence. Some
arms moved target predicates descriptively, but none can be interpreted as a
fixed-read write success under the registration:

- `phi1`: best held-out control was 0.47, 0.49, 0.12, 0.30 across seeds.
- `phi2`: best held-out control was 0.39, 0.37, 0.21, 0.33 across seeds.
- Same-read/same-write remained near zero, as expected from exp 29/I0.
- Random controls were sometimes comparable to or above the best non-random
  held-out score, especially for `phi2` seed 400.
- Specificity included `phi4_first_matched` and the other target predicate
  when their held-out room cleared `0.01`; `phi3_all_neutral` was skipped for
  low room on every reported target/seed.

Controls behaved as expected. `phi3_all_neutral` remained flat and
non-transported as a read (`std` about 0.014–0.016; held-out `R2` strongly
negative). `phi4_first_matched` was not a transported affine read under this
split (held-out `R2` from about -0.61 to -0.10).

### Implementation Note

The first execution attempt exposed a performance defect in the approved
script: it repeatedly evaluated full `V^3` completion distributions and
materialized residual outputs even though I1 only scores predicate
probabilities. That attempt was interrupted before producing a usable run
artifact. The script was then amended without changing the registered
estimand, candidate set, thresholds, PairSets, or verdict logic: it now computes
exact observable `p_phi` by summing only mask-true continuation probabilities
and avoids residual materialization in the scorer. The canonical output above
comes from the amended exact scorer.

## Conclusion

I1 does not answer whether a separate write direction can control
`phi1_next_closes` or `phi2_net_return` through the exp-29 affine reads,
because those fixed reads failed the registered held-out-position interpreter
gate. The correct typed result is a read-side limit:

```text
The exp-29 affine predicate readouts are discovery-position decoders, but not
transported fixed reads for the I1 position split. Under this registered
fixed-read/L1/pstack/m=3 setup, write failure is not adjudicated.
```

This narrows the next step. The right follow-up is not to claim that no
fixed-read write exists, and not to treat descriptive write movement as causal
control. The result routes to the I2-style read/write-pair question, or a
narrower read-transport repair, before carrying fixed-read oblique writes
forward as an intervention primitive.

**Resolved by exp 31 (read-transport atlas).** The held-out-position read
failure above was the off-diagonal transfer cell only; exp 31 measured the
in-place held-out diagonal this experiment never computed and found both
predicates *are* linearly readable in place at `{26,34}` at `R2` comparable to
discovery. So `FIXED_READ_NOT_TRANSPORTED` here is a **transport-of-read**
failure (the read direction is position-specific), not a representational
absence — the `FIXED_READ_LIMIT` verdict stands, but it routes to I2 with
position-conditioned reads, not to I4/depth. See
[experiments/31-read-transport-atlas.md](31-read-transport-atlas.md).
