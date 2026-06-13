# Experiment 23 — Oracle-withdrawal 1: measurement substrate — PRE-REGISTRATION

**Script:** `scripts/oracle_withdrawal/substrate.py` (with the new
`pstack` process in `processes.py`).

**Status: concluded.** Result review reported no findings.

## Question

Can the next target support hidden-oracle abstraction experiments without
smuggling oracle information into reference selection or verdict tuning?

This experiment is not an abstraction experiment. It is the measurement
substrate gate for `ORACLE_WITHDRAWAL.md`: build one richer exact process,
train a small model, and decide whether exact and sampled completion
measurement are stable enough to start hidden-oracle reference selection.

## Target

Primary target: `pstack`, a bounded probabilistic stack grammar added to
`processes.py`.

It is a pragmatic first target for oracle-withdrawal rehearsal:

- richer than Dyck-2 because a hidden mode biases both bracket openings
  and neutral terminals, so visible stack depth is not the full future
  state;
- still exact and cheap because the depth bound makes it an
  `HMMProcess`, so the existing training/cache/PairSet machinery applies;
- honest about scope: this is not a full unbounded PCFG or inside/outside
  parser.

Fallback policy: do not switch targets inside this experiment. If
`pstack` fails the registered substrate gates below, register either a
`pstack` repair or a Dyck/PCFG-hybrid fallback with the same gate types.
Target fallback is allowed only through those gates, before abstraction
results exist.

## Registered Commands

After pre-run review:

```bash
python3 train.py --process pstack --outdir out/pstack-L4 --layers 4 \
  --seq-len 40 --steps 8000 --m 3 --eval-seqs 1200 --burn-in 4 --seed 0
python3 scripts/oracle_withdrawal/substrate.py --outdir out/pstack-L4
```

Review-only self-test:

```bash
python3 scripts/oracle_withdrawal/substrate.py --selftest
```

## Oracle Discipline

Exact oracle use is allowed here only for measurement calibration:

- optimal-NLL competence yardstick;
- exact m-gram completion distributions;
- sampled-vs-exact estimator audit;
- full-patch ceiling/non-vacuity audit.

Exact oracle use is not allowed to choose proposal families, choose a
compact trusted reference, tune acceptance thresholds, or alter later
verdict branches after observable abstraction results. This experiment
has no compact-reference selection step.

## Registered Constants

| item | registered value |
|---|---|
| process | `pstack` |
| checkpoint | `out/pstack-L4` |
| model | `d_model=64`, `layers=4`, `seq_len=40`, `seed=0` |
| training budget | `8000` steps, batch default `256` |
| completion horizon | `m=3` (`6^3 = 216` outcomes) |
| registered positions | `ts = (10, 18, 26, 34)` |
| exact audit rows | `192` belief rows |
| sampled budgets | `64`, `256`, `1024` completions per row |
| sampled repeats | `4` |
| sampled confidence rule | one-sided `95%` nonparametric bootstrap upper bound over pooled row/repeat JS values, `2000` bootstrap resamples |
| PairSet size | `256` pairs from `800` sequences |
| stratum pool | `2000` sequences |

## Pre-Registered Gates

**P0 (process and sampler self-test; enforced).** `python3
scripts/oracle_withdrawal/substrate.py --selftest` must pass before the
canonical run. It
checks row stochasticity, exact m-gram normalization, conditional
sampling, and JS-divergence finiteness.

Failure = implementation defect; fix before training.

**P1 (model competence; gate).** The trained model's token NLL must be
within `0.030` nats/token of the exact optimal-NLL yardstick on the same
registered 1000-sequence competence sample. The script computes both the
model NLL and exact optimal NLL on that same sample; it does not subtract
the `config.json` training-time probe.

- `gap <= 0.030`: pass.
- `gap > 0.030`: no-go; do not start hidden-oracle reference selection.
  Register a training/target repair. This is a substrate failure, not a
  battery finding.

**P2 (exact runtime; gate).** Exact m-gram evaluation for `192` rows at
`m=3` must finish within `30` seconds on the run machine.

- pass: exact audit is affordable enough for this program;
- fail: no-go; either reduce the target/horizon through a reviewed repair
  or register the fallback target.

**P3 (sampled-vs-exact estimator; gate).** For each sampled budget, the
script pools JS divergences across the registered `192` rows and `4`
repeats, then computes one-sided `95%` nonparametric bootstrap upper
bounds (`2000` resamples) for the mean and p90 JS. At the largest sampled
budget (`1024` completions per row), those confidence bounds must
satisfy:

- mean-JS upper `95%` bound `<= 0.050`;
- p90-JS upper `95%` bound `<= 0.100`;
- mean JS must be non-increasing across budgets up to a 10% slack:
  `mean_js(next) <= 1.10 * mean_js(prev)`.

Failure = sampled measurement is not yet stable enough. Do not proceed
to hidden-oracle reference selection until the sampling policy or target
is repaired.

**P4 (observable stratum coverage; gate).** At the registered positions,
observable stack-depth strata `0..3` over `2000` sampled sequences must
each have at least `80` rows.

Failure = contrast-distribution substrate problem; repair PairSet
construction or target before abstraction experiments.

**P5 (full-patch non-vacuity; gate).** On the registered PairSet:

- unpatched observable KL gap `D0 >= 0.010`;
- full-patch relative gain `(D0 - Dfull) / D0 >= 0.50`.

The full patch is a ceiling/control only. Passing P5 does not select a
trusted compact reference. Failure means the residual-intervention target
is vacuous or too weak for this model/position/horizon setting.

**P6 (decision; deterministic).** If P0-P5 pass, print:

> GO: pstack measurement substrate is registered-usable for
> hidden-oracle reference selection.

and exit with status `0`.

If any gate fails, print:

> NO-GO: do not start hidden-oracle reference selection. Register a
> substrate repair or fallback target before abstraction experiments.

and exit with status `1`.

## Brittleness Criteria

For this experiment, the target is brittle if any of the following occur:

- process self-checks fail;
- training misses the competence gate;
- exact m-gram runtime exceeds the budget;
- sampled confidence/error policy fails P3;
- observable strata are too sparse for the registered PairSet plan;
- full-patch improvement is vacuous;
- PairSet self-checks fail.

Any brittleness result blocks the next oracle-withdrawal experiment until
reviewed.

## Adjudication

- **All gates pass:** preregister hidden-oracle reference selection on
  `pstack`.
- **P0 failure:** implementation bug; fix and re-review before training.
- **P1 failure:** training/target competence debt; repair model settings
  or target.
- **P2 failure:** exact audit too expensive; reduce horizon/target or
  register fallback.
- **P3 failure:** sampled-completion substrate not usable; repair sampling
  budget/error policy before oracle-withdrawal claims.
- **P4 failure:** PairSet/contrast-distribution miss; repair strata or
  target.
- **P5 failure:** vacuous intervention target; do not run reference
  selection.

## Scope & Local Assumptions

- This experiment does not test abstraction quality.
- This experiment does not select a trusted compact reference.
- Exact oracle access is measurement-calibration only.
- `pstack` is a bounded finite-state stack grammar, not a full PCFG.
- The registered horizon is `m=3`; any larger horizon is a future
  substrate question.
- The sampled estimator is calibrated for the registered process,
  positions, horizon, and budgets only.

## Expected Output Tables

The script prints:

- model competence gap;
- exact runtime;
- sampled-vs-exact JS table over budgets;
- observable depth-stratum counts;
- full-patch non-vacuity numbers (`D0`, `Dfull`, relative gain);
- final GO/NO-GO decision.

---

## Results

**P0-P5 all pass.** The `pstack` substrate is registered-usable for
hidden-oracle reference selection.

Run artifacts:

- `out/pstack-L4-train.txt`
- `out/exp23_pstack-L4.txt`
- `out/pstack-L4/config.json`

The checkpoint `out/pstack-L4/model.pt` is intentionally untracked: the
run is CPU/fixed-seed and reproducible from the registered training
command. `cache.npz` is also ignored per repository policy.

### Gate Results

| gate | result |
|---|---|
| P1 model competence | PASS: same-sample gap-to-optimal `+0.0027 <= 0.030` nats |
| P2 exact runtime | PASS: `192` rows, `216` outcomes, `0.08s <= 30s` |
| P3 sampled estimator | PASS: at `1024`, mean-JS upper 95% bound `0.0128 <= 0.050`, p90 upper 95% bound `0.0187 <= 0.100`, monotone |
| P4 observable strata | PASS: depth counts `2089, 1756, 934, 328`, min `328 >= 80` |
| P5 full-patch non-vacuity | PASS: `D0=5.0580`, `Dfull=0.0824`, relative gain `98.4% >= 50%` |

### Interpretation

The substrate gate did not just barely clear. Exact completion audit is
cheap at `m=3`, the trained model is competent by the same-sample
yardstick, and the residual intervention target is strongly non-vacuous.
The sampled estimator is clearly usable at the registered `1024`
completion budget.

The caution carried forward is budget-specific: `256` completions per row
has mean-JS upper bound `0.0530`, just above the registered `0.050`
threshold, while `1024` is comfortably inside the band. Later
sampled-verdict experiments should not quietly use `256` as if it had
passed the substrate policy.

### Decision

Proceed to hidden-oracle reference selection on `pstack`.

No abstraction quality, trusted compact reference, or battery transfer
claim is made here. Exact oracle access in this experiment was
measurement-calibration only.
