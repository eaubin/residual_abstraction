# Experiment 21 — Dyck robustness sweep (Phase 2, Block 3) — PRE-REGISTRATION

**Script:** `dyck_robustness.py` (on `battery.py` + `expcommon.py`).
**Status: pre-registered; NOT YET RUN. Pause here for review before the
first canonical run.**

**Question.** Exp 19 showed that the frozen battery reproduces the
Dyck-2 anchor and recalibrates cleanly at the standing horizon `mm=3`.
Exp 20 showed that the Block-2 matrix transfers at `mm=3` under one
adversarial coordinate setting, registered shifts, and prefix-balance
strata. Block 3 sweeps the remaining local indices named in `PHASE2.md`:

- completion horizon: `mm in {1,2,3,4}`;
- tolerance policy: `eps in {0.01,0.02,0.05,0.10}`;
- coordinate stress: `kappa in {30,100,300}`, `junk_seed=0`.

The purpose is not to discover a new abstraction. The discovered
exp-19 rank-4 core is treated as the trusted Dyck reference. This block
asks whether the Dyck transfer record is stable when the horizon,
acceptance tolerance, and adversarial conditioning are moved.

**What is NOT in scope.** Gradient read re-learning, checkpoint
selection, write widening, multiple junk draws, stronger distribution
shifts, and the Block-2 single-write probe. Exp 20 already scoped the
single-write probe result; this experiment does not try to turn it into
an existential statement about rank-1 directions.

## Design

**Model.** Existing `out/dyck2-L4` checkpoint only. The script halts
unless the config is exactly `dyck2`, 4 layers, `d_model=64`,
`seq_len=32`, `burn_in=4`, training `m=3`, seed 0.

**Patch point.** L1, inherited from exp 19.

**Core.** Reproduce the exp-19 rank-4 core by
`battery.cegar_loop(eps=0.05, k_max=12, eps_drop=0.01)`. Halt unless:

- `k* = 4`;
- discovery `c_obs = 98.5% +/- 2 pts`;
- nested exact `mm=3` staircase matches exp 19 within 2 pts:
  `{37.8%, 71.6%, 85.0%, 92.6%}`.

### Horizon Construction

`mm <= 3` uses the `m=3` chain and exact marginalization through
`kl_by_horizon`. `mm=4` uses new `m=4` PairSets with positions pinned to
`{8,16,24}`. The default PairSet position formula would shift positions
when `m` changes, so the script asserts:

- `m=3` and `m=4` groups are both `{8,16,24}`;
- same-seed `m=3` and `m=4` PairSets share pair indices;
- their prefix arrays are bit-identical;
- marginalizing an `m=3` chain to `mm=1` agrees with an independent
  `m=1` chain on the same small check set;
- exact m-gram marginalization and a synthetic marginal helper check
  pass.

Budget rule: stop at `mm=4`. Dyck has vocabulary size 4, so `m=4`
requires 256 continuations per pair, already 4x the exp-19/20 `m=3`
cost. `m=5` would require 1024 continuations per pair and is deferred
unless Block 3 exposes a horizon-local failure.

### Arms

**Arm A — Fixed-patch horizon matrix.**

Evaluate fixed patches at every `mm in {1,2,3,4}`:

| patch | source |
|---|---|
| full | identity patch |
| core | exp-19 discovered rank-4 core |
| PCA-4 | exp-19 PCA control |
| PLS-4 | exp-19 PLS echo control |
| rand-4 | exp-19 random control |
| emb-4 | token-embedding control |

Per `(patch, mm)`: observable closure, exact closure, calibration gap,
and rho against the core.

**Arm B — Benign tolerance staircases.**

Run `battery.cegar_staircase` for each horizon and
`eps in {0.01,0.02,0.05,0.10}`. For `mm=4`, the loop uses the `m=4`
PairSet; for `mm<=3`, it uses the pinned `m=3` PairSet.

**Arm C — Coordinate stress.**

For each `kappa in {30,100,300}`, build the exp-20 adversarial
transform with `P_c = core`, `junk_seed=0`, and junk perpendicular to
the core. For each kappa:

- reproduce the exp-20-style nearest-to-core z-write from the write
  pool;
- evaluate the `z-id` destructive comparator across all horizons;
- compute rho of `z-id` against the core across all horizons;
- run adversarial CEGAR accept-count staircases across `eps` and `mm`.

The adversarial arm intentionally does not construct learned reads.
It tests battery accept-counts and rho under coordinate stress, not read
optimization.

## Battery Members Exercised

| member | implementation in exp 21 |
|---|---|
| 1 observable closure | fixed-patch and z-id horizon matrices |
| 2 rho | core-vs-controls and core-vs-z-id across horizons |
| 5 accepted-cell calibration | obs/exact gap for all fixed/z-id cells with obs >= 20% |
| 6 CEGAR staircases | benign `k*(eps, mm)` and adversarial accept-counts |

Member 3 and member 4 were exercised in Block 2 and are not rerun here;
this block has no new held-out-position or distribution-shift arm.

## Pre-Registered Predictions

**P1 (anchor reproduction; ~95%).** The exp-19 core and nested
staircase reproduce within the thresholds above. Failure = halt; the
run is not Experiment 21.

**P2 (obs/exact calibration across horizons; ~85%).** Every fixed or
z-id cell with observable closure >= 20% has `|obs - exact| <= 0.10`.
If the worst gap is above 0.10 only at `mm=4`, record a horizon-local
calibration widening; if the gap appears at `mm<=3`, treat it as a
regression against exps 19/20.

**P3 (rho bands persist across horizons; ~80%).** At every horizon,
`full`, `PCA-4`, and `emb-4` remain equivalent to the core
(`rho <= 0.25`), while `PLS-4` and `rand-4` remain distinct
(`rho >= 0.50`). Failure with separation present but shifted thresholds
is recalibration; failure with flat rho is a battery-transfer failure.

**P4 (benign tolerance staircases; ~80%).** For every horizon,
`k*(eps)` is weakly decreasing as eps increases, and `k*(0.01) <= 8`.
For `mm in {2,3,4}`, `k*(0.05)` is between 3 and 5. `mm=1` is reported
descriptively: a smaller value would be a semantic-complexity finding,
not a failure.

**P5 (adversarial accept-counts; ~85%).** At `eps=0.05`, adversarial
accept-count is 0 for every `kappa x mm` cell. Failure means the
coordinate-stress no-false-accept record from Mess3/exp20 does not
transfer cleanly to the moved index.

**P6 (z-id remains behaviorally distinct; ~90%).** For every kappa,
the minimum z-id rho over horizons is >= 0.50. This checks that rho
separates destructive coordinate artifacts from the trusted core across
horizon and kappa.

**P7 (adversarial tolerance staircases; ~75%).** For every
`kappa x mm`, adversarial accept-count is weakly decreasing as eps
increases, and there are no acceptances for `eps >= 0.05`. If accepts
appear only at `eps=0.01`, record threshold sensitivity; if accepts
appear at `eps>=0.05`, P5/P7 fail together.

**P8 (validity/config/self-checks; enforced).** Validity gate,
registered config guard, standard PairSet self-checks, and horizon
self-checks pass. Failure = halt.

## Adjudication

- P1/P8 failure: halt; this is not a canonical run.
- P2 failure: determine whether the gap is horizon-local (`mm=4`) or a
  regression at previously-tested horizons.
- P3/P6 failure with shifted but separated bands: recalibrate. Failure
  with no separation: genuine battery-transfer failure.
- P4 lower benign dimension at `mm=1`: semantic-complexity finding, not
  a battery failure. `mm=2..4` outside the registered 3..5 band is a
  horizon-stability finding.
- P5/P7 failure: coordinate-stress no-false-accept record fails under
  the moved index; follow up with a dedicated κ/junk-draw experiment.

## Scope & Local Assumptions

- Process: Dyck-2 checkpoint `out/dyck2-L4`; no retraining.
- Horizon: `mm in {1,2,3,4}` only.
- Tolerance: `eps in {0.01,0.02,0.05,0.10}` only.
- Coordinate stress: `kappa in {30,100,300}`, `junk_seed=0` only.
- Reference patch: exp-19 discovered core. No T-aware clean construction
  is used.
- Proposal/interpreter class: linear CEGAR directions and linear
  projection/oblique patches only.
- Distribution: standard exp-19/20 PairSets at pinned positions
  `{8,16,24}`. No held-out-position or distribution-shift claim is made
  here.

## Reviewable Failure Modes

- Horizon-local calibration drift: observable score no longer tracks
  exact closure at `mm=4`.
- Horizon-local rho drift: rho separates at `mm=3` but not at `mm=1` or
  `mm=4`.
- Semantic staircase: the Dyck core needs fewer or more directions at
  moved horizons.
- Coordinate false acceptance: adversarial CEGAR accepts under a kappa
  or tolerance where the battery predicts rejection.
- Registration drift: `m=4` accidentally changes pair positions or
  pairs rather than only completion horizon; guarded by self-checks.

---

## Results

Not run. Pause for pre-run review.
