# Experiment 41 — Product-counter planted-carrier substrate gate — DESIGN/CALIBRATION DRAFT

**Script:** `scripts/product_counter/substrate.py`; analytic derivation:
`scripts/product_counter/derive_thresholds.py`.

**Status:** calibration/freeze review draft, with a product-counter instance
intended for freezing after review. This is not a clean blind preregistration:
the implementation has already been smoke-run during drafting, and those numbers
are recorded below as development verification. A future reviewed run is a
repeat of a deterministic frozen instance, not a first look. Only the concrete
product-counter regular-process oracle/mixed-carrier cell is a candidate for
freezing. The reusable schema remains provisional roadmap text.

## Motivation

This experiment separates two questions that transformer experiments usually
mix:

1. Is the diagnostic / intervention method well-posed?
2. Is the transformer residual stream hard to work with?

It introduces a deliberately small finite-state product-counter process with
known latent variables and planted carriers. The purpose is not to prove
anything about transformers. The purpose is to build a controlled substrate
where future intervention-class experiments can fail before they are tried on
transformers.

The process has hidden state:

```text
s_t = (a_t, b_t, c_t)
a_t in {0,1,2,3}
b_t in {0,1,2,3}
c_t in {0,1}
```

`a` is the primary target variable, `b` is a same-kind distractor, and `c` is a
known high-room out-of-bundle control. This directly addresses the Phase-3
`pstack` substrate defect: specificity could not be adjudicated because the
available out-of-bundle predicate was low-room and possibly coupled.

This is a substrate gate, not an abstraction experiment.

## Reusable Gate Schema

This provisional section is roadmap text, not a frozen methodology. This
experiment should leave behind a reusable substrate-gate shape, not only a
single toy, but exp 41 freezes only the concrete product-counter instance after
review. Future non-LLM process/carrier experiments should separate:

1. **Process design contract.** Register state variables, transition semantics,
   intended separable variables, intentionally coupled variables, the high-room
   out-of-bundle control, completion horizon, evaluated distribution, and oracle
   availability/cost.
2. **Carrier contract.** Register whether the carrier preserves completion
   semantics, whether variable decoding is in scope, whether it supports reads,
   writes, or both, whether its decoder is oracle/planted/learned-on-observables/
   learned-on-ground-truth, and what claim class that access level permits.
3. **Metric families.** Report completion faithfulness, target room, no-info or
   state-independent floor, oracle/reference ceiling, separability matrix,
   leakage/cross-drag, nuisance stability where relevant, and exact/sampled cost.
4. **Threshold policy.** Prefer relative thresholds of the form
   `effect >= floor + alpha * (ceiling - floor)` and
   `cross_drag <= beta * target_effect`. Product-counter uses absolute instance
   thresholds only because its analytic floor and ceiling are exact.
5. **Typed routing.** Future gates should prefer route labels such as
   `LOW_TARGET_ROOM`, `CONTROL_LOW_ROOM`, `NOT_DISSOCIABLE`,
   `UNEXPECTED_COUPLING`, `CARRIER_FAITHFULNESS_FAIL`, `TOO_EXPENSIVE`, and
   `READY_FOR_PLANTED_INTERVENTIONS`. The product-counter instance now emits
   local route labels, but those labels are not yet a global verdict ontology.

Reusable ladders this gate is meant to support:

- carrier ladder: oracle one-hot -> planted mixed linear carrier -> learned
  RNN/GRU carrier -> transformer residual stream;
- language ladder: finite product counter -> stack/Dyck variants ->
  PCFG/inside-outside exact languages -> sampled/no-oracle text regimes;
- metric ladder: exact completion agreement -> observable target movement ->
  separability/leakage matrix -> intervention specificity/cross-drag ->
  coherence under generation.

## What This Experiment Can Falsify

- The process has too little absolute observable room.
- The variables are not cleanly dissociable under exhaustive enumeration.
- `c` is not a strong enough high-room control.
- The registered policy leaks too much off-target movement.
- Exact `m=3` completion distributions are too slow for future work.
- The oracle carrier does not reproduce process completions.
- The planted mixed carrier construction or decoder fails to preserve state.
- The substrate is not suitable for future intervention-class experiments.

## What This Experiment Cannot Claim

- Transformer relevance, except as motivation.
- Abstraction success.
- Oracle-free read recovery.
- Intervention specificity.
- CEGAR success.
- Learned representation interpretability.

## Development / Review Commands

```bash
uv run python scripts/product_counter/derive_thresholds.py
uv run python scripts/product_counter/substrate.py --selftest
uv run python scripts/product_counter/substrate.py --carrier oracle --m 3 --seed 0
uv run python scripts/product_counter/substrate.py --carrier mixed --m 3 --seed 0 --kappa 100
```

The single aggregate command for a future post-review freeze run is:

```bash
uv run python scripts/product_counter/substrate.py --confirm --m 3 --seed 0 --kappa 100
```

The aggregate freeze verdict is the output of `--confirm`, which runs the selftest,
the oracle carrier panel, and the mixed carrier panel in one process and emits a
single route-bearing verdict. `--confirm` is in scope only for:

```text
m = 3
seed = 0
kappa = 100
d_hidden = 64
```

Any other `--confirm` setting routes `OUT_OF_SCOPE_CONFIG` and is not a reviewed
freeze result for exp 41. Single-carrier commands are development/review
panels only.

## Oracle Discipline

Exact hidden-state and exact `m`-gram access are allowed here only for:

- process construction;
- symbolic threshold derivation;
- substrate calibration;
- exact pair enumeration;
- carrier agreement checks;
- future-control-room audits.

Exact access is not allowed here to:

- claim an oracle-free read;
- tune thresholds after seeing results;
- select a compact abstraction reference;
- claim transformer transfer.

The mixed carrier uses planted-map access for decoding. That is calibration
access, not a learned read.

## Process

Vocabulary:

```text
A_PLUS, A_MINUS, B_PLUS, B_MINUS, C0, C1, NOISE
```

Transition semantics:

- `A_PLUS` increments `a` modulo 4.
- `A_MINUS` decrements `a` modulo 4.
- `B_PLUS` increments `b` modulo 4.
- `B_MINUS` decrements `b` modulo 4.
- `C0` sets `c = 0`.
- `C1` sets `c = 1`.
- `NOISE` leaves state unchanged.

## Registered Policy Constants

The softmax policy is registered through positive unnormalized weights. The
implementation stores probabilities, but these are exactly equivalent to logits
`log(weight)`.

Group weights:

```text
W_a = 1
W_b = 1
W_c = 6/5
W_n = 4/5
Z   = 4
```

State codes:

```text
u_a = (-3/5, -1/5,  1/5, 3/5)
u_b = (-3/5,  1/5, -1/5, 3/5)
u_c = (-3/4,  3/4)
```

Emission weights:

```text
A_PLUS  = W_a * (1 + u_a[a]) / 2
A_MINUS = W_a * (1 - u_a[a]) / 2
B_PLUS  = W_b * (1 + u_b[b]) / 2
B_MINUS = W_b * (1 - u_b[b]) / 2
C1      = W_c * (1 + u_c[c]) / 2
C0      = W_c * (1 - u_c[c]) / 2
NOISE   = W_n
```

All probabilities divide these weights by `Z`. The constant group masses make
one-step off-target leakage analytically zero:

```text
obs_a = P(A_PLUS) - P(A_MINUS) = (W_a / Z) * u_a[a]
obs_b = P(B_PLUS) - P(B_MINUS) = (W_b / Z) * u_b[b]
obs_c = P(C1)     - P(C0)      = (W_c / Z) * u_c[c]
```

This replaces the earlier high/low-bin sketch, which would have produced zero
own-room for within-bin value contrasts.

## Analytic Threshold Basis

`scripts/product_counter/derive_thresholds.py` uses SymPy to derive the exact
one-step observable margins before the reviewed freeze run.

Analytic own-room:

| target | ordered pairs | min cell count | mean own | min own | p10 own | p50 own | p90 own |
|---|---:|---:|---:|---:|---:|---:|---:|
| `a` | 96 | 8 | 1/6 = 0.1667 | 0.1000 | 0.1000 | 0.2000 | 0.3000 |
| `b` | 96 | 8 | 1/6 = 0.1667 | 0.1000 | 0.1000 | 0.2000 | 0.3000 |
| `c` | 32 | 16 | 9/20 = 0.4500 | 0.4500 | 0.4500 | 0.4500 | 0.4500 |

Analytic off-target movement is exactly zero for the registered one-step
observables. The runtime gate verifies this numerically against the implemented
HMM.

The floor/ceiling table below is the review-protocol baseline. For own-room and
high-room gates, the no-information floor is a state-independent observable: all
state contrasts move the observable by `0`. The product-counter ceiling is the
analytic positive margin induced by the registered policy. For off-target
leakage, the desired separable value is exactly `0`; a fully coupled bad
reference would have off-target movement on the same order as own movement
(`own/off ~= 1`), but that relative dominance check is descriptive here because
the product-counter instance registers the stricter analytic-zero gate.

| quantity | no-info / vacuous floor | registered positive / separable value | threshold |
|---|---:|---:|---:|
| mean own `a` | `0` | `0.1667` | `>= 0.10` |
| mean own `b` | `0` | `0.1667` | `>= 0.10` |
| mean own `c` | `0` | `0.4500` | `>= 0.30` |
| p10 own `a` | `0` | `0.1000` | `>= 0.05` |
| p10 own `b` | `0` | `0.1000` | `>= 0.05` |
| p10 own `c` | `0` | `0.4500` | `>= 0.30` |
| mean off-target movement | desired separable value `0` | analytic value `0` | `<= 1e-12` |
| synthetic equal-coupling bad reference | off/own `= 1`, own/off `= 1` | product-counter own/off `= infinite` | descriptive only |

Instance thresholds are set below the analytic positive margins but above the
vacuous floor:

| gate | threshold | analytic positive |
|---|---:|---:|
| mean own `a` | `>= 0.10` | `0.1667` |
| mean own `b` | `>= 0.10` | `0.1667` |
| mean own `c` | `>= 0.30` | `0.4500` |
| p10 own `a` | `>= 0.05` | `0.1000` |
| p10 own `b` | `>= 0.05` | `0.1000` |
| p10 own `c` | `>= 0.30` | `0.4500` |
| mean off-target movement | `<= 1e-12` | `0` |

These are utility gates, not universal constants. Any threshold change after
reviewed freeze results invalidates the run as a freeze result and must be an
amendment or new experiment.

The synthetic equal-coupling baseline is included only to anchor future
cross-drag reuse: a bad reference that copies target movement equally into an
off-target observable has off/own `= 1`. The product-counter instance uses the
stricter analytic-zero leakage gate instead of a relative cross-drag threshold.

## Registered Carriers

### Carrier 1: Oracle One-Hot Automaton

```text
h_t = e_{s_t}
```

The hidden vector is the 32-dimensional one-hot state vector. Its completion
distribution must exactly match the process completion distribution.

Gate:

```text
mean JS <= 1e-12
decode accuracy = 32/32
```

### Carrier 2: Planted Mixed Representation

Default:

```text
d_hidden = 64
kappa = 100
seed = 0
```

Construction:

```text
U = QR(seeded N(0,1) matrix in R^{64 x 32})
V = QR(seeded N(0,1) matrix in R^{32 x 32})
sigma_i = geomspace(1, 1/kappa, 32)
T = U diag(sigma) V^T
h_t = T e_{s_t}
```

Registered decoder:

```text
y = pinv(T) h_t
decoded_state = argmax(y)
```

Gate:

```text
rank(T) = 32
abs(condition_number(T) - kappa) / kappa <= 1e-10
max |pinv(T) T - I| <= 1e-10
decode accuracy = 32/32
mean JS <= 1e-10
```

The script also prints minimum planted-column separation as descriptive
geometry. It is not a GO/NO-GO gate in this experiment.

Descriptive future panels such as `kappa in {1,30,100,300}` may be useful, but
they are not part of this experiment's reviewed freeze verdict unless explicitly
registered by amendment before running them.

## Pre-Registered Predictions

P1. The process transition tensor is valid and exact `m`-gram distributions
normalize.

P2. Each variable has non-vacuous own-observable room.

P3. `c` has enough absolute room to serve as a high-room out-of-bundle control.

P4. Off-target one-step observable movement is zero to numerical precision for
each target contrast.

P5. Exact `m=3` computation is cheap enough for future experiments.

P6. The oracle carrier agrees with exact process completions to numerical
precision.

P7. The planted mixed carrier agrees with exact process completions under the
registered decoder.

## Confirmatory Scope

The only claim a passing reviewed freeze run can earn is:

```text
READY_FOR_PLANTED_INTERVENTIONS: the product-counter regular-process
oracle/mixed-carrier cell is ready for later planted-carrier intervention
experiments at m=3 under the registered one-step observables and planted
decoder.
```

This does not certify a coupled-variable substrate, nuisance-variable substrate,
learned carrier, intervention handle, stack-like dynamics, or general reusable
methodology. Those remain roadmap items.

## Gates

| gate | pass condition |
|---|---|
| process selftest | tensor shape `(7,32,32)`, row-stochastic marginal chain, token semantics, stationary normalization, all stationary masses `> 1e-12`, `m in {1,2,3}` normalization |
| analytic identity | implemented one-step observables match analytic formulas within `1e-12` |
| dissociability | expected ordered/value-cell counts exactly match `a: 96/12`, `b: 96/12`, `c: 32/2`; min exhaustive cell count `>= 1` |
| own-room | mean and p10 thresholds above |
| high-room control | mean `|delta obs_c|` for `c` contrasts `>= 0.30` |
| leakage | mean off-target movement `<= 1e-12`; own/off dominance is printed descriptively for future relative-threshold designs |
| exact runtime | all 32 one-hot beliefs, `m=3`, runtime `<= 10s` |
| oracle agreement | decode `32/32`, mean JS `<= 1e-12` |
| mixed agreement | rank `32`, condition relative error `<= 1e-10`, decode `32/32`, reconstruction error `<= 1e-10`, mean JS `<= 1e-10` |

## Route Labels

The executable emits exactly one route label by precedence:

| route | meaning |
|---|---|
| `OUT_OF_SCOPE_CONFIG` | `--confirm` was run outside the frozen instance settings |
| `HARNESS_FAIL` | selftest, normalization, or implementation guard failed |
| `NOT_DISSOCIABLE` | expected pair/value-cell enumeration failed |
| `LOW_TARGET_ROOM` | `a` or `b` own-room threshold failed |
| `CONTROL_LOW_ROOM` | `c` high-room control threshold failed |
| `LEAKAGE_FAIL` | analytic off-target zero gate failed |
| `TOO_EXPENSIVE` | exact `m=3` runtime exceeded budget |
| `CARRIER_FAITHFULNESS_FAIL` | oracle or mixed carrier agreement failed |
| `READY_FOR_PLANTED_INTERVENTIONS` | all product-counter instance gates passed |

## Confound Table

Load-bearing quantity: the GO/NO-GO substrate verdict, especially the
combination of own-room and off-target leakage.

| mechanism that could produce a clean GO number | excluded by? |
|---|---|
| The process is trivially separable only because the script read the analytic formulas rather than the HMM | excluded by numerical checks against `processes.product_counter()` transition matrices |
| Off-target movement is hidden by group normalization rather than measured | this is intentional and analytic; the conclusion is scoped to this controlled process, and the leakage table verifies the implemented HMM matches the analytic identity |
| Thresholds pass because they were tuned after seeing results | excluded only by freeze discipline; any post-result threshold/logit change is an amendment or new experiment |
| The planted mixed carrier passes only because the planted decoder has oracle access | not excluded; explicitly allowed and scoped as calibration access, not learned read recovery |
| `m=3` agreement hides longer-horizon failure | not excluded; claim is scoped to `m=3`, with longer horizons left to future gates |

## Measured But Not Verdict-Bearing

The script prints these descriptive quantities:

- p50/p90 own-room summaries;
- minimum planted-column separation;
- own/off dominance ratio;
- max JS, in addition to mean JS.

Only the gates above decide GO/NO-GO.

## Brittleness Criteria

- Any threshold failure is NO-GO.
- Any unregistered threshold change invalidates the run as a freeze result.
- Any policy-constant change after seeing results is an amendment or a new
  experiment.
- Any change to the planted decoder after seeing results is an amendment or a
  new experiment.
- Optional learned RNN/GRU carriers are deferred. Their failure does not affect
  this mandatory oracle/mixed substrate gate, but it must not be hidden if later
  added.

## Adjudication

The aggregate freeze command prints exactly one route and one of:

```text
GO: product-counter regular-process oracle/mixed-carrier cell is ready for planted-carrier intervention experiments.
```

or:

```text
NO-GO: product-counter instance is not ready for planted-carrier intervention experiments.
```

NO-GO is not a failed implementation. It is a useful negative result if it
prevents ambiguous later transformer experiments.

Because development runs already occurred, a reviewed run after freezing should
be described as a deterministic calibration/freeze result, not as a clean blind
preregistration result.

## Expected Output Tables

- state/token constants;
- pair counts;
- own-room metrics;
- leakage/dominance matrix;
- exact `m`-gram normalization/runtime;
- carrier-agreement divergences;
- route label;
- final GO/NO-GO decision.

## Development Verification

These runs occurred while drafting and are not the reviewed freeze run. The
sandbox used `UV_CACHE_DIR=.uv-cache` because the default
uv cache path is outside the writable sandbox.

| command | result |
|---|---|
| `uv run python scripts/product_counter/derive_thresholds.py` | PASS; SymPy derived the margins above |
| `uv run python scripts/product_counter/substrate.py --selftest` | PASS; max analytic observable error `2.776e-17` |
| `uv run python scripts/product_counter/substrate.py --carrier oracle --m 3 --seed 0` | development GO |
| `uv run python scripts/product_counter/substrate.py --carrier mixed --m 3 --seed 0 --kappa 100` | development GO |
| `uv run python scripts/product_counter/substrate.py --confirm --m 3 --seed 0 --kappa 100` | development route `READY_FOR_PLANTED_INTERVENTIONS`, development GO |
| `uv run python scripts/product_counter/substrate.py --confirm --m 2 --seed 1 --kappa 30 --d-hidden 40` | development route `OUT_OF_SCOPE_CONFIG`, development NO-GO |

Development run highlights:

| quantity | value |
|---|---:|
| exact `m=3` runtime, freeze oracle panel | `0.035929s` |
| exact `m=3` runtime, freeze mixed panel | `0.037191s` |
| max `m=3` normalization error | `4.441e-16` |
| oracle mean JS | `0` |
| mixed mean JS | `0` |
| mixed reconstruction error | `3.635e-15` |
| mixed rank | `32` |
| mixed condition number | `100` |
| mixed condition relative error | `2.984e-15` |

## Results

Empty until the reviewed freeze run.
