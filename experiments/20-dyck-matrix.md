# Experiment 20 — Interventional battery matrix (Phase 2, Block 2) — PRE-REGISTRATION

**Script:** `dyck_matrix.py` (on `battery.py` + `expcommon.py`).
**Status: concluded.**

**Question.** Exp 19 established the Dyck baseline: the battery
reproduces exp 7 exactly, all Mess3 thresholds transfer unchanged, and
the discovered rank-4 core closes 92.6% of the behavioral gap. This
experiment tests the battery under the conditions it was designed for —
adversarial coordinates, distribution shifts, and stratified evaluation
— and exercises member 4 (shift-retention), which was deferred from
Block 1.

The standing lens (PHASE2.md): the representation–oracle mismatch. The
battery is behavioral and never needed linear decode; Dyck is the
process where decode fails. Do the adversarial and robustness
instruments work where the oracle-based instruments broke?

**What is NOT in scope.** Horizon staircase (Block 3). κ-grading across
multiple κ values (Block 3). Multiple adversarial draws / T-robustness
sweep (Block 3). Tolerance staircases (Block 3). Full gradient-read
search with checkpoint selection and write widening (the Mess3 exp-16
protocol) — the gradient probe here is one write, one optimization, one
transport check.

## Design

**Model.** The existing `out/dyck2-L4` checkpoint. No retraining.

**Patch point.** L1 (exp 19's confirmed ℓ†).

**Core.** The exp-19 discovered rank-4 core, reproduced via
`battery.cegar_loop` (eps=0.05, k_max=12, eps_drop=0.01); assert
k\*=4, c_obs within 2 pts of 98.5%.

### Arms

**Arm A — Adversarial regime (κ=100).**
- T with P_c = exp-19 core (4-dim), junk_seed=0, κ=100.
- `battery.cegar_accept` in z-coordinates (the accept-count instrument).
- Destructive control: id patch (write=read in z-coords for the
  nearest-to-core pool write), pulled back through T.
- ρ of core vs z-id (member 2 in adversarial conditions).
- obs/exact calibration (member 5) on accepted adversarial cells.

**Arm B — Shift-retention (member 4, the Block 1 deferral).**
- *Position shift*: pair set at ts={10,14,22} (unseen positions,
  interpolated between the training grid {8,16,24}).
- *Depth-profile shift*: pair set with init_state = state (0,0) (stack
  depth 2, index 3 in the Dyck-2 state space). This shifts the input
  distribution toward deeper nesting.
- R for full, PCA-4, rand-4 on each shift, with the discovered core as
  the retention reference (R(core)=1 by construction).

**Arm C — Depth stratification (Dyck-specific).**
- For each eval pair: compute the bracket depth at the pair's position
  from the target sequence prefix (tokens 0,1 = open, tokens 2,3 =
  close; depth = cumulative opens minus closes).
- Partition into depth strata {0, 1, 2, 3}.
- Per-stratum exact closure for core, full, and controls.

**Arm D — Gradient read probe (compact).**
- Write pool (round 1, standard protocol), filtered to angle ≤ 15° of
  the 4-dim core. Pick the closest write.
- Gradient-learn the read via `expcommon.optimize_affine` (200 steps,
  lr=0.01, batch 64, adversarial=True).
- Evaluate the rank-1 oblique patch at standard and test positions.
- This is a single-write probe, not the full Mess3 protocol. Either
  outcome (rank-1-opaque core or position-entangled read) is
  informative.

### Pair sets

| set | seed | n | positions | init_state | purpose |
|---|---|---|---|---|---|
| disc | seed+111 | 400 | standard | — | discovery, obs refs |
| eval | seed+777 | 600 | standard | — | exact closures, ρ, depth strata |
| test | seed+443 | 400 | {10,14,22} | — | position-shift retention |
| shift-depth | seed+999 | 400 | standard | 3 (depth 2) | depth-profile retention |

### Patches in the matrix

| patch | rank | source | arm |
|---|---|---|---|
| full | 64 | identity | baseline |
| no-op | — | None | baseline |
| core | 4 | exp-19 cegar_loop | baseline |
| PCA-4 | 4 | exp-19 control | baseline |
| rand-4 | 4 | exp-19 control | baseline |
| z-id | 1 | oblique_patch(w,w) in z-coords, pulled back | A |
| learned-1 | 1 | gradient-optimized oblique_patch(c,w) | D |

### Battery members exercised

| member | what is measured | arm |
|---|---|---|
| 1 (obs closure) | Refs.obs on registered arm cells at mm=3 | all |
| 2 (ρ) | Exact.rho of core vs z-id | A |
| 3 (held-out gain) | Refs.obs on test and shift-depth sets | B |
| 4 (shift-retention) | R for each patch on each shift | B |
| 5 (P4 calibration) | calibration_gap on accepted cells including adversarial | A,B |
| 6 (CEGAR accept) | cegar_accept in z-coordinates at κ=100 | A |

## Scope & local assumptions

- Patch point L1 inherited from exp 19 (asserted via cegar_loop
  reproduction, not re-derived).
- m=3, mm=3 standing horizon.
- Core = battery.cegar_loop output with eps_drop=0.01 (full discovery
  loop, as in exp 19).
- κ=100 only (κ-grading is Block 3).
- junk_seed=0 only (T-robustness is Block 3).
- Shift-retention reference = discovered core (the Dyck-native
  "clean" reference; no T-aware construction since adversarial writes
  are not the core test here).
- Depth strata computed from target sequences at the pair's position
  (bracket depth = opens − closes in seq\[:t\]).

## Pre-registered predictions

**P1 (adversarial CEGAR accept=0; ~95%).** The miner's variance
dependence (exp 8) transfers to Dyck: CEGAR accept-count is 0 at
κ=100. The 4-dim core does not change the miner's failure mode — it is
a coordinate-system pathology, not a rank pathology.

**P2 (ρ separates in adversarial regime; ~90%).** Core-vs-z-id ρ ≥ 0.50
(z-id is behaviorally distinct from the core) and core-vs-full ρ ≤ 0.25
(the core is behaviorally equivalent to the full patch). ρ is an
oracle-free comparator operating on model outputs, so it should separate
regardless of the coordinate system the patch was constructed in.

**P3 (core position-shift gain retention ≥ 0.70; ~85%).** The core
retains at least 70% of its obs gain at unseen positions {10,14,22}
(gain retention = shift gain / base gain). Exp 19's val-set result
(+98.7% at {12,20}) strongly supports this. Note: this is the raw gain
ratio for the core; member-4 R (which normalizes by the reference) is 1
by construction since the core is the reference.

**P4 (core depth-shift gain retention ≥ 0.50; ~70%).** The core retains
at least half its obs gain under the depth-profile shift (init_state at
depth 2). Lower confidence than P3: this is a new type of shift, never
tested on either process. The core is behavioral, not depth-specific,
but the model's internal routing could vary by depth.

**P5 (depth uniformity; ~75%).** Core exact closure varies by ≤ 10 pts
across bracket-depth strata at mm=3. If the spread exceeds 10 pts,
record per-depth thresholds (this is a finding about Dyck-specific
routing, not a battery failure).

**P6 (obs/exact calibration in adversarial; ~90%).** Every accepted
cell (obs ≥ 20%) in the adversarial regime has |obs − exact| ≤ 0.10
(the Mess3 band, confirmed on Dyck at baseline in exp 19).

**P7 (rank-1 learned read > 20% at train; ~80%).** The best z-write
captures enough of the 4-dim core that a gradient-learned read exceeds
the acceptance threshold. Exp 19's marginal gain profile (43.6% for
direction 1) suggests rank-1 closure of 30–45% is reachable. If this
fails, the core is "rank-1-opaque" — a finding about 4-dim routing.

**P8 (position entanglement if P7 holds; ~70%).** The learned read's
closure at test positions {10,14,22} is below 20% (the Mess3
position-entanglement finding transfers). If this fails (the read
transports), that is a significant finding: Dyck's representation is
more position-generic than Mess3's.

## Adjudication

- **P1 failure**: investigate — the miner's coordinate sensitivity was
  established on Mess3; failure here would be process-specific.
- **P2 failure with separation at different thresholds**: recalibrate.
  P2 failure with no separation: escalate (genuine battery-transfer
  failure).
- **P3/P4 failure**: the core is distribution-sensitive — report as
  finding, follow up with dedicated shift experiment.
- **P5 failure**: record per-depth thresholds (finding, not failure).
- **P6 failure**: widen the calibration band for adversarial conditions.
- **P7 failure**: rank-1-opaque core (finding, not failure); skip P8.
- **P8 failure** (read transports): major finding — Dyck's
  representation is more position-generic. Follow-up experiment.

---

## Results

**P1–P6 all hold. P7 fails (rank-1-opaque core — informative finding).
P8 skipped.** Block 2 gates passed.

### P1 (adversarial CEGAR accept=0): HOLDS

Accept-count = 0 at κ=100. The miner's variance dependence (exp 8)
transfers to Dyck unchanged: the 4-dim core does not change the failure
mode. The nearest-to-core z-write is M2\*Sinv at 0.1° of the core.
z-id patch: obs=−0.8%, exact=−1.1% (fully destructive).

### P2 (ρ separates): HOLDS

| comparison | ρ | classification |
|---|---|---|
| core vs full | 0.0144 | equivalent (≤ 0.25) |
| core vs z-id | 1.0096 | distinct (≥ 0.50) |

Separation ratio 70×. Both Mess3 bands transfer. The has\_separation
flag is moot (P2 holds cleanly).

### P3 (core position-shift gain retention ≥ 0.70): HOLDS

Gain base=+98.5%, shift=+99.1%, retention=1.01. The core's gain
actually *improves slightly* at unseen positions {10,14,22} — far above
the 0.70 threshold.

### P4 (core depth-shift gain retention ≥ 0.50): HOLDS

Gain base=+98.5%, shift=+98.6%, retention=1.00. The depth-profile shift
(init\_state at depth 2) has no measurable effect on the core. This is
a stronger result than predicted (70% confidence, threshold 0.50).

### Arm B full table

| patch | base | pos-shift | R(pos) | depth-shift | R(depth) |
|---|---|---|---|---|---|
| full | +100.0% | +100.0% | 0.99 | +100.0% | 1.00 |
| core | +98.5% | +99.1% | 1.00 | +98.6% | 1.00 |
| pca | +99.6% | +99.7% | 0.99 | +99.5% | 1.00 |
| rand | +4.1% | +4.2% | 1.02 | +4.4% | 1.06 |

All working patches (full, core, pca) retain essentially 100% of their
gain under both shifts. Even the rand control retains its (small) gain.
The Dyck-2 model's 4-dim core is fully distribution-robust across
position and depth-profile shifts.

Calibration at shifted conditions (worst gap 0.073, all within 0.10):

| cell | obs | exact | gap |
|---|---|---|---|
| full/pos-shift | +100.0% | +92.7% | 0.073 |
| full/depth-shift | +100.0% | +95.8% | 0.042 |
| core/pos-shift | +99.1% | +91.9% | 0.072 |
| core/depth-shift | +98.6% | +94.4% | 0.042 |
| pca/pos-shift | +99.7% | +92.5% | 0.072 |
| pca/depth-shift | +99.5% | +95.3% | 0.042 |

### P5 (depth uniformity ≤ 10 pts): HOLDS

| patch | d=−2 | d=0 | d=2 |
|---|---|---|---|
| full | +93.5% | +94.0% | +92.6% |
| core | +91.8% | +93.2% | +91.3% |
| pca | +93.9% | +93.8% | +91.4% |
| rand | −1.5% | +3.1% | +6.0% |

Core spread: 1.9% (threshold 10%). Strata counts: d=−2: 121, d=0: 367,
d=2: 112 (relative prefix balance = opens − closes, not absolute
bracket depth as originally registered). The core's exact closure is
uniform across the observed prefix-balance strata.

### P6 (adversarial calibration ≤ 0.10): HOLDS

Core cell: obs=+98.5%, exact=+92.6%, gap=0.058. z-id not accepted
(obs < 20%). The Mess3 calibration band (0.10) holds in the adversarial
regime.

### P7 (rank-1 learned > 20% at train): FAILS — rank-1-opaque

obs(disc)=+19.7%, exact(eval)=+18.1%. Just below the 20% threshold.
ρ(core vs learned)=0.8691 (behaviorally distinct from the core). The
registered single-write rank-1 probe failed: the nearest-to-core
z-write (M2\*Sinv at 0.1°) with one gradient optimization does not
produce a behaviorally effective rank-1 oblique patch. This does not
establish that no rank-1 direction exists — the full read-search
protocol (multiple writes, checkpoint selection, write widening) was
explicitly out of scope (Block 3).

The loss curve (CE 8.23 → 6.88 → 7.01) shows the optimizer finding a
local minimum that doesn't correspond to a behaviorally effective read,
consistent with the adversarial-coordinate read-construction failures
from exps 13–16 on Mess3.

### P8 (position entanglement): SKIPPED

P7 did not hold — the rank-1 patch is below acceptance. Position
transport is not testable for a sub-threshold patch.

### Summary

The P1–P6 battery gates transfer to Dyck-2 under adversarial
coordinates, distribution shifts, and depth stratification — all Mess3
thresholds hold unchanged. The gradient read probe (P7) failed and
position-entanglement (P8) was not tested; these are findings about the
single-write rank-1 probe, not battery-transfer failures. The
discovered 4-dim core is:

- **Adversarially opaque** (CEGAR accept=0, z-id destructive, ρ
  separates 70×) — same as Mess3's 2-dim core.
- **Fully distribution-robust** (gain retention ≥ 0.99 under both
  position and depth-profile shifts) — stronger than predicted.
- **Depth-uniform** (1.9% spread across observed prefix-balance
  strata) — the core's closure does not vary with relative nesting.
- **Single-write rank-1 probe failed** (19.7%, just below threshold)
  — the registered probe (one write, one optimization) did not produce
  a behaviorally effective rank-1 patch. Whether a rank-1 direction
  exists is untested (full read-search is Block 3 scope).
