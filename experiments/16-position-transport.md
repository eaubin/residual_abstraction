# Experiment 16 — position transportability: can the training/selection protocol repair position entanglement? — PRE-REGISTRATION

**Script:** `heldout.py` (on `expcommon.py`). **Status: pre-registered;
NOT YET RUN — results to be appended below the marked line.**

**Question.** Exp 15 measured the failure: gradient-learned reads compute
the clean functional at their trained positions and invert at unseen ones
(ledger: pooled-CE position-genericity, falsified). One focused question:
**can learned reads be made position-transportable by changing the
training/selection protocol** — without touching the patch family, T, or
the optimizer?

**External proposal, evaluated.** Adopted: the single-question scope
(generality debts — T sweep, eps_gain staircase, m-staircase — explicitly
deferred; they audit different claims and would de-localize a failure
here); held-out-position validation as first-class; a second unseen
position set; small write widening to avoid another single-write
anecdote; the four-outcome map (registered below). **Modified, honesty
grounds:** the proposal suggests ρ and shift-retention R as selection/
acceptance criteria — but both reference the *clean patch*, which is
T-aware and firewalled as evaluation-only since exp 11; using them in
selection would leak oracle-adjacent information into discovery. The
honest first-class selection signal is **held-out-position observable
gain** (model-vs-model c_obs on pairs the optimizer never saw, at
positions it never trained on); ρ, R-style retentions, and per-position
EPR remain first-class as *evaluation* verdicts. **Added:** a scope
note — the unseen test positions are interior to the training range
(the protocol margin forbids anything outside [8, 24]), so this
experiment tests position-*interpolation*, not extrapolation.

## Design (deltas from the standard setting; everything else as exps 8–15)

**Position sets and seeds.** P_train = {8, 16, 24} (the standard
discovery set, seed+111, 400 pairs); **P_val = {12, 20}** (selection set:
a new *discovery-side* PairSet, seed+333, 400 pairs — observable use
only, beliefs never touched); **P_test = {10, 14, 22}** (final
evaluation only, unseen by both training and selection). Mixed-position
discovery set for arm B: seed+211, 400 pairs at {8, 12, 16, 20, 24}.
Eval sets (exact targets): seed+777 sequences at each of P_train, P_val,
P_test (the P_val eval set is exp-15's shift-A set, for comparability).

**Arm A — held-out checkpoint selection (the selection repair).** Per
write: the exp-14/15 optimizer unchanged (id init, registered single
init to cap budget — dual-init robustness was characterized in exps
13–14), checkpoints at step 0, 20, …, 200; **the selected read is the
checkpoint with the best observable gain on the P_val selection set.**
This is simultaneously generalization pressure (early stopping by
held-out positions) and a potential w1 rescue (its divergence is fast;
an early checkpoint may precede it).

**Arm B — mixed-position training (the objective repair).** w1 and w2
only: same optimizer, minibatches from the mixed-position discovery set,
final read (no checkpoint selection — B isolates whether position
diversity in the *objective* alone produces transportability; judged on
P_test).

**Write widening (arm A only).** w1 and w2 (reproduced, exp-12 rule) plus
the nearest-to-plane write ≤ 15° from each of two additional pool draws
(seeds seed+1, seed+2), deduplicated against already-selected writes at
|cos| > 0.999 of the back-mapped directions; empty slots recorded. T
fixed throughout (inherited single-T indexing, unchanged).

**Evaluation matrix.** Per write: clean patch, A-selected read, A-final
read (the exp-15-style entangled comparator — for w1/w2 the final
checkpoint reproduces the exp-15 read exactly, same seeds; train-gain
asserts are the P1 tripwires), and B-final (w1/w2). Each on the three
eval sets: exact closure, ρ (vs same-write clean), and position-t EPR
cells (selected reads). Own-retention gain_test/gain_train reported
(observable, no clean reference — the honest R-analogue).

## Pre-registered predictions

- **P1 (anchors + reproduction; ~90%).** Standard anchors; w1/w2
  reproduce (A-final train gains within 2 pts of −548.2% / +42.5%);
  reproduction failure halts (enforced, as exp 15).
- **P2 (headline — the selection repair; ~45%).** w2's A-selected read
  is **transportable**: observable gain ≥ 20% on P_val *and* exact
  closure gain ≥ 20% on P_test. Three-way: transportable / val-only
  (≥ 20% on P_val, < 20% on P_test — selection overfit to the selection
  positions) / not rescued (< 20% on P_val).
- **P3 (the objective repair; ~50%).** w2's B-final read reaches exact
  closure gain ≥ 20% on P_test. (Interpolation scope note applies.)
- **P4 (observable soundness; ~85%).** |observable − exact| ≤ 0.10 on
  every accepted (≥ 20%) read, per eval set where both are computed.
  NOT TESTED if none accepted.
- **P5 (equivalence consistency; gated on P2 or P3; ~60%).** Every
  transportable read has ρ ≤ 0.5 on P_test (transportability should come
  *with* behavioral proximity to clean, per exp 15's monotone ρ–transfer
  relationship). The interesting falsification: a transportable read
  that is behaviorally far from clean — a second, position-generic
  control distinct from the plane's.
- **P6 (write generality; gated on ≥ 1 widened write found; ~50%
  conditional).** At least half of all near-plane writes tested under
  arm A admit a transportable read (P2 criterion per write). NOT TESTED
  if both extra draws come up empty.
- **P7 (w1 rescue; ~30%).** w1's A-selected read reaches gain ≥ 20% on
  P_val. FAILS is informative either way: selection cannot rescue a bad
  optimization landscape.
- **P8 (validity gate, enforced).**

**Adjudication map (registered from the endorsed proposal).** (1) P2
holds → position entanglement was an objective/validation failure;
learned reads can be made transportable. (2) P2 holds with train gain
substantially below exp-15's +42.5% (reported, descriptive) → a real
tradeoff between local control and transported access. (3) P2, P3, P7
all fail → clean-plane access remains special; behavioral gradients find
statistical controls unless constrained more structurally than this.
(4) P6 mixed → write geometry/landscape remains central.

**Ledger rows (added with this registration).** Checkpoint selection /
triple split: 20-step granularity suffices; selection-on-P_val makes the
selection set a fit target — the unseen P_test bounds that, same
discipline as discovery/eval splits. Mixed-position training: position
diversity in minibatches is the operative variable (budget held fixed).
Position-interpolation scope: P_test ⊂ (8, 24) interior — transport
claims are interpolation claims; extrapolation is out of protocol reach.

**Self-checks** (standard set, plus): (i) checkpointing is inert — a
2-step micro-run with and without `checkpoint_every` returns bitwise
identical final reads; (ii) the three position sets construct exactly as
registered; (iii) the write-dedup rule rejects a duplicate of w1 and
accepts an orthogonal direction (synthetic).

**Enforcement.** Standard (registered params, full config, seed 0,
gate). Estimated runtime ~2–2.5 h (up to 6 optimization runs with 11
selection evals each + a ~14-patch × 3-set evaluation matrix).

---

*(Results to be appended here after the run.)*
