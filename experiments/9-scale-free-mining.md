# Experiment 9 — scale-free proposal mining — PRE-REGISTRATION

**Script:** `miners.py`. **Status: pre-registered; NOT YET RUN — results to
be appended below the marked line.**

**Question.** Experiment 8 split the interventional-discovery method claim:
behavioral *acceptance* produced no false confidence in any regime, but the
*proposal miner* (a weighted second-moment eigenvector — covariance
machinery) is variance-dependent and died at k\* = 0 in hostile
coordinates. Can the proposal map be made scale-free **while keeping the
behavioral acceptance rule unchanged** — recovering Experiment 6's plane
under Experiment 8's adversarial coordinates at small k\*, with the
observable/exact agreement (P4) finally tested on a nontrivial *accepted*
adversarial patch?

**Design basis** (FORMALISM.md §5): mining in coordinates whitened by the
discovery sample's own covariance is GL(d)-invariant in the ridgeless
limit — the repair is not a heuristic patch but the invariance principle
implemented, and the experiment *asserts* the invariance numerically as a
known-answer self-check.

## The miners (acceptance loop identical for all: eps_gain 0.05, eps_drop
0.01, k_max 8, observable closure on the discovery pairs)

- **M1 — covariance miner** (the incumbent, `discover.mined_direction`):
  eigenvector of Σᵢ wᵢ δᵢδᵢᵀ in raw regime coordinates; orthogonal-projector
  patch. Anchors: known-good benign (exp 6), known-fail adversarial (exp 8).
- **M2 — whitened miner** (the invariance repair): the same weighted
  second-moment eigenvector computed on Σ̂^{-1/2}-whitened prefix
  differences, where Σ̂ is the per-position-centered stream covariance at
  the patch point estimated from the registered basis sample
  (eigenvalue floor: 10⁻¹⁰·λ_max, registered); the interchange patch is the
  coordinate swap in whitened coordinates (raw-space
  P = Σ̂^{-1/2}UUᵀΣ̂^{1/2}, row convention). In the adversarial regime the
  miner sees only z and Σ̂_z = TᵀΣ̂T; the proposition says its induced
  stream-space patch is the same either way, up to ridge/finite-sample
  tolerance.
- **M3 — centered-weight miner** (the decorrelation alternative):
  eigenvector of Σᵢ (wᵢ − w̄) δᵢδᵢᵀ (largest algebraic eigenvalue), raw
  regime coordinates, orthogonal-projector patch. Robust through
  *decorrelation* — bulk delta energy that does not covary with behavioral
  failure cancels — rather than invariance; junk leakage scales with the
  covariance between junk energy and the weights, so this is NOT provably
  safe. Registered precisely because the outcome is uncertain (~50%
  credence either way): if it succeeds, decorrelation suffices without
  whitening; if it fails, invariance is necessary, not just sufficient.

**Omitted by design:** the "delta-ratio" miner (weighted vs unweighted
delta second moments, a generalized eigenproblem) — under this pairing
protocol (random same-position pairs) the unweighted delta second moment is
≈ 2Σ̂ in expectation, making it a near-duplicate of M2; noted rather than
run.

## Setting and protocol

The Experiment-6/8 setting, unchanged: `out/mess3-L4`, patch point L1,
prefix-wide scope, registered parameters (κ = 100; eps_gain 0.05; eps_drop
0.01; k_max 8; 400/600 disjoint discovery/evaluation pairs; basis sample
800 sequences; m = 3; three pooled positions). **Seed 0 is part of the
registration** (as in Experiment 8: T's construction depends on it).
Each miner runs in both regimes: **benign** (natural coordinates) and
**adversarial** (z = ρ·T, Experiment 8's registered T, rebuilt from the
in-run anchor reproduction of the Experiment-6 plane, which is asserted
against the recorded fixed point as before). M1's benign run *is* the
anchor. Evaluation as in Experiments 6/8: exact-target closures on the
disjoint evaluation pairs at each accepted k\*, per-position stability,
plane recovery against the Experiment-6 plane (dimension-matched: k\* = 2
AND largest principal angle, with the projection Frobenius distance as
characterization; the discovered stream-space subspace is read uniformly
as the row space of the induced raw patch matrix).

## Pre-registered predictions (thresholds fixed before the first run)

- **P1 (anchors reproduce).** M1-benign: k\* = 2 with exact closure ≥ 90%
  of full (Experiment 6). M1-adversarial: k\* = 0 (Experiment 8).
- **P2 (the repair).** M2-adversarial converges at **k\* = 2** with exact
  closure ≥ 90% of full and plane recovery (largest principal angle ≤ 15°).
- **P3 (no benign cost).** M2-benign exact closure ≥ M1-benign − 2 points
  at its accepted k\*.
- **P4 (nontrivial observable soundness — completes Experiment 8's
  matrix).** |c_obs − exact pooled m=3 closure| ≤ 0.10 for M2's
  adversarial accepted result, **which must be non-null (k\* > 0)**.
  Clarification (pre-run, from review): on a null result P4 is reported
  **NOT TESTED**, not HOLDS (a 0-vs-0 agreement would recreate exactly the
  Experiment-8 vacuity this prediction exists to close) and not FAILS
  (k\* = 0 is a miner failure — P2's verdict — not evidence of scoring
  unsoundness).
- **P5 (the invariance proposition, checked numerically).** M2's induced
  stream-space patch computed from z-data equals the one computed from
  x-data: relative Frobenius difference ≤ 0.05 at every accepted k, and
  first-direction subspace angle ≤ 5°. (Tolerances cover the ridge floor
  and finite samples; the ridgeless statement is exact.)
- **P6 (the decorrelation question, genuinely uncertain).**
  M3-adversarial exact closure ≥ 50% of full. Both outcomes are findings
  (see M3 above).
- **P7 (validity gate, enforced).** As in Experiments 5–8.

**Characterizations (no thresholds).** κ-sweep: M2-adversarial rerun at
κ ∈ {10, 1000} (k\*, c_obs, exact closure — probing where the ridge floor
starts to bite, motivated by Experiment 8's pls-z surprise); M3-benign;
principal-angle and projection-distance tables for all accepted subspaces.

## Failure modes this can newly exhibit

*Invariance gap* — P5 fails while P2 holds: the proposition's ridgeless
idealization is too far from the κ = 100 finite-sample reality; the repair
works but not for the registered reason (the theory note would need
weakening). *Repair insufficient* — P2 fails: whitening fixes the metric
but the weighted second moment still cannot see the causal plane through
the amplified-junk noise floor at this sample size; would push toward
behavioral-selection proposals (candidate generation scored by actual
closure gain) at higher compute. *Benign regression* — P3 fails: the
whitened metric distorts the benign-regime geometry enough to lose what
the covariance miner found; would mean proposal quality is
regime-dependent and no single miner dominates.

**Self-checks** (every invocation; `--selftest` exits after the standard
four): the Experiment-4/5 known-answer checks; the Experiment-8 anchor and
transform checks (real runs); loop invariants (unit-norm, inner-orthogonal
proposals; D₀ > D_full; c_obs(full) = 1); and a k=1 invariance probe run
*before* the adversarial loops. Clarification (pre-run): the probe aborts
only on *gross* violation (relative Frobenius > 0.5 — machinery-bug
level); P5's registered tolerance (0.05 / 5°) is a **verdict**, not an
abort, since "invariance gap: P5 fails while P2 holds" is a registered
failure mode that must remain observable.

**Enforcement.** Registered parameters, full model config, seed 0, and the
gate guarded in code exactly as in Experiment 8 (`--force-invalid` demotes
to a labeled exploratory run).

---

## Results: P1, P7 HOLD; P2, P3, P6 FAIL; P4 NOT TESTED — the invariance proposition is *confirmed* and the repair *fails anyway*

(Registered parameters, seed 0, gate +0.0024 PASS; anchors reproduced;
transform checks passed. Raw output `out/exp9_mess3-L4.txt`, figure
`out/mess3-L4/experiment9.png`.)

| miner / regime | k\* | c_obs | exact m=3 closure | vs exp-6 plane |
|---|---|---|---|---|
| M1 / benign (anchor) | 2 | 99.8% | 98.3% | 0.0° |
| M1 / adversarial | 0 | — | 0% | — |
| **M2 / benign** | **0** | — | **0%** | — |
| **M2 / adversarial** | **0** | — | **0%** | — |
| M3 / benign | 2 | 97.1% | 95.5% | 13.6° |
| M3 / adversarial | 0 | — | 0% | — |

**Finding 1: the invariance proposition is true — and it didn't help.**
The pre-loop probe found M2's induced stream-space patch *identical* from
x-data and z-data (relative Frobenius 0.0000, first-direction angle
0.00°), the M2 trajectories in the two regimes are step-for-step identical
(the rejected k=1 trial scored c_obs 1.0% in both), and the κ-sweep is the
proposition's fingerprint: **fixed point k\* = 0, c_obs = 0.0%, exact
0.0% at κ = 10, 100, and 1000, with the rejected first proposal scoring
an identical 1.0% at every κ** — the whitened miner literally cannot see
κ. (Correction from review: an earlier draft attributed the rejected
trial's 1.0% to the fixed point.) The theory note holds at machine
precision. The whitened miner mines the *same* direction everywhere; that
direction is just behaviorally useless everywhere, including in benign
coordinates (P3 fails: the repair costs everything even where the
incumbent works).

**P5 adjudication (an honest verdict-logic note).** The first run's code
printed P5 FAILS because its implementation treated the empty accepted-k
set as failure — the dual of the P4 vacuity loophole, caught pre-run for
P4 and missed for P5. On review the verdict was made three-way like P4's
(post-run code fix; core numbers diff-verified unchanged on the rerun):
the canonical output now reads **first-direction probe HOLDS (rel
0.0000, 0.00°); accepted-k clause NOT TESTED (no accepted k)**. The
*proposition* P5 was designed to test is confirmed by the probe, the
trajectory identity, and the κ-fingerprint. Standing rule extracted:
every quantifier over a run-dependent set gets the three-way treatment.

**Finding 2 (the sharp one): variance was never a bug in the miner — it
was the signal.** The characterization (computed and printed by miners.py
since review; an earlier draft cited untracked ad-hoc numbers against a
PCA proxy): M2's first mined direction is **not** an echo (85.9° from the
pls plane) — it sits *near* the causal plane (14.9°; M1's 0.0° reference
is by construction, since the causal plane is M1's own anchor output) —
yet its patch earns 1.0% closure where M1's first direction earned
54.3%. Two measured contributors: (a) *selection blur* — whitening
compresses the mining-matrix spectrum (λ1/mean drops 37.2 → 12.6, top
gap 1.56 → 1.35), costing the eigenvector its sharpness; (b) *weak
oblique coupling* — the whitened-coordinate swap reads through the
Σ^{-1/2}-weighted covector, which down-weights exactly the high-variance
directions where pair differences live, so even a near-causal written
direction transfers little. Their relative shares are not disentangled
here (Experiment 10 abandons correlational mining regardless). The
theoretical ceiling behind both: channel and echo carry the same
*correlational* information (the echo is a copy), so a coordinate-
invariant functional of unpatched deltas and pre-patch weights has no
principled causal signal once the coordinate-dependent variance contrast
is normalized away — M1's benign success was variance luck (now triply
confirmed), and M2 removes the luck without replacing the signal.

**Finding 3: decorrelation is regime-dependent (P6 fails, with a benign
consolation).** M3 (centered weights) succeeds benignly — k\* = 2, 95.5%
closure, plane within 13.6°, a second benign-capable miner discovered —
but dies adversarially at k\* = 0 with its first proposal actively harmful
(c_obs −10.0%). Per the registration's framing: invariance is necessary
(decorrelation alone fails the hostile regime) — and by Finding 2, *not
sufficient*. The conjunction is the result: nothing in the
correlational-mining design space satisfies both requirements.

**P4 remains NOT TESTED**, by its own three-way rule: no miner produced an
accepted adversarial patch. The nontrivial observable-soundness datum
still awaits a proposal map that survives hostile coordinates.

**Method implication (the program's next step, not registered here).**
Proposals must become *interventional*: the only invariant quantity that
distinguishes channel from echo is the behavioral response to actually
patching, so candidate directions must be selected by measured closure
gain (or by gradients of the behavioral divergence through the model with
respect to the patch direction — computable with autograd), not by any
statistic of the unpatched data. The acceptance rule has been the sole
causally-grounded component all along; Experiment 9 shows it cannot
delegate candidate generation to correlational statistics in any
coordinate system. That is Experiment 10.

**Status: CONCLUDED.**
