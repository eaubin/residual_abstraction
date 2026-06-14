# Experiment 25 — Oracle-withdrawal 3: does the exp-24 reference ambiguity matter downstream for ρ? — PRE-REGISTRATION

**Script:** `scripts/oracle_withdrawal/rho_invariance.py`.

**Status: pre-registered; NOT YET RUN. Pause here for review before the
canonical run.**

## Question

Exp 24 found a typed `REFERENCE_AMBIGUITY_CONFIRMED` on `pstack`: three
compact `k=4` candidates (`cegar`, `pca`, `delta`) tie at ~93% observable
closure and are **exact-audit-indifferent** (`0.926/0.923/0.923`, spread
`0.003`), yet `pca` separates (`10.4°/12.4°`) from a near-coincident
`{cegar, delta}` plane (`4.9°`). Because the oracle is indifferent among
them, there is no "right" reference to recover, so an oracle-free
*tie-break* would select a winner nothing audits — withdrawn in Exp 24.

By `ORACLE_WITHDRAWAL.md`'s definition, reference ambiguity is "differ
exactly **or** disagree on downstream ρ." These do not differ exactly, so
the live, auditable content is the ρ half:

> Do ρ-verdicts depend on which equally-good reference anchors them?

This has a ground truth — the verdicts agree or they do not. Exp 25 asks
it, with one prerequisite: the two-reference geometry must first be shown
to be a stable object rather than single-seed sampling noise.

**Firewall (carried from Exp 24, enforced in the verdict).** ρ-invariance
and unique-selection are *different claims*. A finding that ρ-verdicts
agree across references is **not** reported as "oracle-free selection
succeeded": the registered observable selection rule still did not yield a
unique reference (Exp 24's standing NO-GO). Exp 25 decides only whether
the ambiguity *matters downstream for ρ (member 2)*, never whether it was
resolved.

## FORMALISM & Ledger

Discharges the **under test (exp 25)** annotation on two §7 ledger rows
(the ρ-reference bet and the framing bet), both currently **scoped** on
the no-oracle half after Exp 24. The verdict returns here at conclusion.
Audited against the §6.1 verdict-predicate checklist (rule 8: every
selection rule registered; rule 4: equivalence claims name their metric —
here the ρ bands and the principal-angle clustering).

## Registered Command

```bash
python3 scripts/oracle_withdrawal/rho_invariance.py --outdir out/pstack-L4
```

Review-only self-test:

```bash
python3 scripts/oracle_withdrawal/rho_invariance.py --selftest
```

(Run via `uv run python ...` in this environment.)

## Oracle Discipline

Every observable verdict is computed and printed before any exact oracle is
read:

- Arm A's **structural gate** (G1 tie, G2 clustering) uses only principal
  angles between candidate subspaces and the Exp-24 observable selection —
  no oracle;
- Arm B's ρ-verdicts and the anchor-divergence metric are model-vs-model
  (`mean J(q_C, q_X) / mean J(q_C, q_un)`), computed through a local helper
  that never reads the process tables.

Per-seed exact closures are computed during Arm A but **quarantined** — not
read by any observable verdict — exactly as the Exp-24 PairSet carries
exact fields it does not read. The exact oracle is read only in the reveal
stage, after the Arm A structural verdict and the full Arm B ρ-table are
printed, for two things: the **premise check** (G3 — do the references
reproduce Exp-24's exact-indifference?) and **calibration** (are the ρ
bands correct on the known cases?). Neither selects a reference, sets a
band, nor tunes the invariance threshold.

## Registered Setting

Inherits the Exp 23/24 standard setting; deltas only.

| item | value |
|---|---|
| process/checkpoint | `pstack`, `out/pstack-L4`, `L1`, `m=3`, `mm=3` |
| candidate construction | reused unchanged from Exp 24 (`reference_selection.build_candidates`, by import; the frozen exp-24 script is not edited) |
| discovery positions | `ts=(10,18,26,34)`; held-out `ts=(12,20,28,36)` |
| eval pairs | `1024` (Exp-24 value, for strata coverage and ρ) |
| Arm A seeds | `8` (`seed = 0..7`), gate majority `6/8` |
| ρ bands | `<= 0.25` equivalent / `>= 0.5` distinct / between = indeterminate (battery member 2) |
| Arm B reference seed | `0` (the Exp-24 geometry) |
| Arm B anchors | distinct references `pca`, `cegar`; `delta` = within-cluster null |
| Arm B probes | `{cegar, pca, delta, emb, rand, full}` |
| anchor-divergence verdict | `d_cross <= d_null + 0.05`, `d_cross = max_X|ρ(pca,X)−ρ(cegar,X)|`, `d_null = max_X|ρ(cegar,X)−ρ(delta,X)|` |
| null validity | `max(ρ(cegar,delta), ρ(delta,cegar)) <= 0.25` (within-cluster pair ρ-equivalent) |
| clustering threshold | `10°` max principal angle (Exp-24 `AMBIG_ANGLE_MAX`) |
| exact-indifference band | tied exact-closure spread `<= 0.05` |
| ρ band calibration | `rand` `>= 0.5` (distinct) per anchor — `rand` only; ref-equivalence is `NULL_VALID` + invariance, not calibration |

## Arm A — Seed-Stability Gate

The gate splits into an **observable structural gate** (decides whether Arm
B runs) and an **exact premise check** (revealed later). For each of the
`8` seeds the script reruns the Exp-24 selection (same `build_candidates`,
strata guards, eligibility, tie rule) and records the eligible tied set,
the pairwise principal angles among `{cegar, pca, delta}`, the clean
`10°`-outlier identity, and — quarantined — the tied set's exact closures.
"Reproduces" = holds in `>= 6/8` seeds.

Observable structural gate. Two diagnostics, plus a **joint** per-seed flag
that is the operative criterion:

- **G1 tie (diagnostic).** The rule yields a multi-candidate compact tie
  (`>= 2` of `{cegar, pca, delta}` within the `0.03` margin).
- **G2 clustering (diagnostic).** The modal candidate is the clean `10°`
  outlier (`cegar–delta <= 10°`, the outlier's two angles `> 10°`); Exp 24:
  `pca`.
- **Joint (operative).** The *same* seed has both: a compact tie AND the
  modal outlier is a member of that tie. Counting G1 and G2 independently
  would let a seed credit the gate via a `cegar+delta` tie and a `pca`-
  outlier geometry that never co-occur; the joint flag matches the
  "two-reference ambiguity reproduces" claim exactly.

`STRUCTURAL_PASS` iff the joint flag holds in `>= 6/8` seeds → run Arm B.
Else the typed observable outcome is `SEED_UNSTABLE_TIE` (G1 fails) or
`SEED_UNSTABLE_CLUSTER` (tie holds but the joint flag does not): the
two-reference geometry is sampling-noise-level, ρ-invariance is moot, and
no exact oracle is read.

Exact premise check (reveal stage):

- **G3 exact-indifference reproduces.** The tied candidates' exact-closure
  spread is `<= 0.05` — below Exp-24's `0.10` "meaningfully better" margin,
  i.e. no "right" reference emerges. If G3 fails
  (`EXACT_INDIFFERENCE_BREAKS`), a right reference may exist after all and
  the auditable tie-break Exp 24 withdrew is **re-opened** — a typed
  reframing, not a failure.

Why `8` seeds / `6`-of-`8`: a coarse but honest separator. If the true
reproduction rate were `0.5` (noise), `P(>= 6/8) = 0.14`; if `0.85`
(stable), `P(>= 6/8) = 0.90`. The cost is ~`8×` an Exp-24 run; a finer
ensemble is not worth the compute.

**Disclosed feasibility peek (observable, seeds 0–2).** Before this
registration was finalized, an oracle-free check of the
`{cegar, pca, delta}` principal angles at seeds 0–2 was run: `cegar–delta`
= `4.9/3.3/4.9°`, `pca–cegar` = `10.4/13.9/13.7°`, `pca–delta` =
`12.4/14.6/16.8°`, clean outlier = `pca` in 3/3 (`k_ref=4` throughout). The
G2 clustering reproduced cleanly, and `pca`'s separation at seeds 1–2
(`~14–17°`) is more comfortable than the marginal seed-0 value (`10.4°`)
that first motivated the gate. This updates P2 upward; it changes a prior,
not a threshold — the decision rule is unchanged, and the full 8-seed gate
(including G1's tie, not peeked) and the exact premise check still run as
registered.

Scope of stability: pair sampling and basis draw (`240` seqs) at a **fixed
checkpoint** — exactly the single-seed concern Exp 24 flagged. Model-
retraining stability is a larger question, out of scope.

## Arm B — Downstream-ρ Invariance (conditional on `STRUCTURAL_PASS`)

At the reference seed (`0`), the script computes ρ(C, X) for every probe
`X` under three anchors: the two distinct references `pca` and `cegar`, and
`delta` — the *same* reference as `cegar` (`4.9°`) built by a different
method, hence a within-cluster null. ρ is observable; the whole table is
printed before any exact audit.

The verdict is a self-calibrated comparison, **not** a band threshold. A
band view alone is brittle — a probe at `ρ=0.24` under one anchor and
`0.26` under the other would "flip" on a `0.02` wiggle, which says more
about the probe sitting on a boundary than about anchor sensitivity (the
same threshold-marginality the Exp-24 review flagged). Instead:

- `d_null = max_X |ρ(cegar,X) − ρ(delta,X)|` — how much ρ-verdicts wobble
  between two *constructions of the same reference* (the noise floor);
- `d_cross = max_X |ρ(pca,X) − ρ(cegar,X)|` — divergence from the
  *cross-cluster* `10–12°` reference difference.

`RHO_ANCHOR_INVARIANT` iff `d_cross <= d_null + 0.05`: anchoring on the
outlier `pca` adds no more than a `0.05` slack beyond within-cluster
jitter. Otherwise `RHO_ANCHOR_SENSITIVE`, naming the worst probe. The
`0.05` slack is one fifth of the equivalence band (`0.25`) — small on the ρ
scale; band agreement is printed descriptively, not as the verdict.

`d_null` is only a valid noise floor if the within-cluster pair really is
ρ-equivalent. So a second observable check, **`NULL_VALID`**: the same
reference built two ways must read ρ-equivalent, `max(ρ(cegar,delta),
ρ(delta,cegar)) <= 0.25`. If it fails, ρ is splitting a behaviorally-
equivalent reference — the sharpest *load-bearing* case — and the floor is
meaningless; this routes to `AMBIGUITY_LOAD_BEARING`, not benign and not
miscalibration. (`NULL_VALID` is observable, printed with the ρ-table.)

Then the exact oracle is revealed:

- **Premise (G3).** Anchors exact-equivalent across seeds (above).
- **Calibration (rand only).** Under each anchor `C in {pca, cegar}`,
  `rand` must read ρ-distinct (`>= 0.5`) — the genuine band-threshold check
  (junk must look distinct; the bands transferred unchanged from Mess3 and
  Dyck). Failure = `RHO_MISCALIBRATED_ON_PSTACK`. Whether the *equivalent
  references* read mutually ρ-equivalent is deliberately **not** folded in
  here: that is the load-bearing question, carried by `NULL_VALID`
  (within-cluster) and the invariance metric (cross-cluster). Folding it
  into calibration would let the sharpest load-bearing finding —
  exact-equivalent references that ρ separates — be silently relabelled
  "recalibrate the bands."

The decision precedence (the verdict partition) is therefore: structural
gate → exact premise → band calibration (`rand`) → ρ splitting an
equivalent reference within-cluster (`NULL_VALID`) or cross-cluster
(invariance) → benign. An *under*-sensitive ρ (junk reads equivalent) is a
calibration problem; an *over*-sensitive ρ (equal refs read distinct) is
load-bearing — the priority keeps the two from colliding on one flag.

## Predictions

**P1 (substrate + self-checks; enforced).** Registered config, PairSet
self-checks, global non-degeneracy, and the Exp-24 strata guards hold at
every seed. Failure = typed substrate halt, not an Exp 25 result.

**P2 (structural seed-stability; ~75%).** `STRUCTURAL_PASS` (the joint
tie∧outlier flag in `>= 6/8`). The
disclosed observable peek found the G2 clustering clean in 3/3 (seeds 0–2),
with `pca` separating by `10–17°` — so the pre-peek "coin-flip" prior
(~55%, set by seed-0's marginality) was too low. The residual risk is the
remaining seeds and G1's tie reproducing under the full scoring (not
peeked). A gate failure is still an informative reframing of Exp 24's
geometry, not a code failure.

**P2b (exact premise; ~85%, given `STRUCTURAL_PASS`).** G3 reproduces — the
tied candidates stay exact-equivalent (Exp-24 spread `0.003 << 0.05`).
Failure (`EXACT_INDIFFERENCE_BREAKS`) would re-open the auditable tie-break.

**P3 (ρ anchor-invariance; ~65%, given premise).** `RHO_ANCHOR_INVARIANT`:
`d_cross <= d_null + 0.05`. Because `pca` and `cegar` are exact-equivalent,
anchoring on either *should* leave ρ-verdicts within the within-cluster
null; the `10–12°` gap could still make ρ's mean-level Jeffreys
anchor-sensitive — the interesting failure. Both outcomes are findings.

**P4 (ρ band calibration; ~90%, given premise).** `rand` reads ρ-distinct
under each anchor — the genuine band-threshold check (Dyck/Mess3
transferred the `0.5` distinct band; pstack is the open question). This is
deliberately narrow: ref-equivalence is *not* a calibration question here.

**P5 (decision; deterministic).**

- `STRUCTURAL_PASS ∧ premise ∧ rand-calibrated ∧ NULL_VALID ∧
  RHO_ANCHOR_INVARIANT` → `AMBIGUITY_DOWNSTREAM_BENIGN`: which equally-good
  reference anchors ρ does not change the equivalence verdicts beyond
  within-cluster jitter. **GO** to preregister Block 3 (battery transfer)
  under a *declared canonical anchor* (the interventionally-discovered
  `cegar` core; `delta` is subsumed, `pca` is a ρ-equivalent control).
  Firewall: this is "the ambiguity is downstream-benign **for ρ**," **not**
  "selection succeeded" — Exp 24's NO-GO on unique selection stands; member
  4 is checked in Block 3.
- `STRUCTURAL_PASS ∧ premise ∧ rand-calibrated ∧ (¬NULL_VALID ∨
  RHO_ANCHOR_SENSITIVE)` → `AMBIGUITY_LOAD_BEARING`: ρ separates
  behaviorally-equivalent references — within-cluster (`¬NULL_VALID`, the
  sharpest form) or cross-cluster (divergence beyond the null) — on an
  oracle-unadjudicable basis. **NO-GO**; the reference-selection failure
  propagates into the battery. Register the repair (e.g. a reference-robust
  ρ or a stronger selection rule) before transfer.
- `RHO_MISCALIBRATED_ON_PSTACK` → **NO-GO**; recalibrate member 2 on
  `pstack` before any ρ-based transfer claim.
- `SEED_UNSTABLE_TIE` / `SEED_UNSTABLE_CLUSTER` → **NO-GO**; the Exp-24
  two-reference geometry is sampling-noise-level. Battery transfer may then
  proceed under the single discovered `cegar` core without a genuine
  multiplicity, but that is a separate registration.
- `EXACT_INDIFFERENCE_BREAKS` → **HOLD-REFRAME**; a "right" reference may
  exist, re-opening the auditable tie-break Exp 24 withdrew. Register that
  tie-break as the next experiment.

## Scope & Local Assumptions

- Exp 25 tests invariance of **member 2 (ρ) only**. Member 4
  (shift-retention) is also reference-normalized; whether the ambiguity
  bites it is a residual, checked in Block 3 under the chosen anchor, not
  closed here. A benign ρ result is not a whole-battery invariance claim.
- Stability is over pair/basis **sampling** at a fixed checkpoint, not over
  model retraining.
- The ρ bands, the `0.05` anchor-divergence slack, the `10°` clustering
  threshold, the `0.05` indifference band, and the `6/8` majority are the
  registered indices; a different choice could move the verdict. The
  anchor-divergence verdict is *self-calibrated* against the within-cluster
  null `d_null`, so it is less threshold-sensitive than a raw band cutoff.
- `d_null` is a **single within-cluster pair** (`cegar`–`delta`) — a
  one-sample estimate of the noise floor. If that pair happens to give
  near-identical ρ, `d_null → 0` and the verdict reduces to the absolute
  `0.05` slack (a raw threshold); the slack caps that downside. The
  opposite pathology — `d_null` inflated because ρ cannot see the same
  reference as equivalent — is caught by `NULL_VALID`, which routes it to
  `AMBIGUITY_LOAD_BEARING` rather than a vacuous benign pass.
- ρ carries the member-2 mean-level caveat (BATTERY.md): equivalence is a
  mean-Jeffreys statement, not per-pair.
- Exact closure is evaluation-only; sampled completions are not used (exact
  model chain probabilities throughout, as in Exp 24).
- The result is indexed by `pstack`, `L1`, `m=3`, the registered positions,
  the Exp-24 candidate family, and these thresholds.

## Expected Output

The script prints, in order:

- the observable Arm-A table (per-seed tie flag, `{cegar,pca,delta}`
  pairwise angles, clean outlier) and the G1/G2/joint structural checks
  with their `k/8` counts and modal outlier;
- the `ARM_A_STRUCTURAL` outcome (and, on structural failure, the decision
  and exit — no exact read);
- on `STRUCTURAL_PASS`: the Arm-B ρ table (probe × the three anchors, with
  per-probe `d_cross`/`d_null`), the `d_cross`/`d_null` maxima, and the
  `RHO_ANCHOR_INVARIANT` and `NULL_VALID` lines — all before the exact
  reveal;
- the exact reveal: per-seed quarantined exact closures + G3 premise,
  anchor exact-equivalence, and per-anchor `rand` band calibration;
- `AUDIT_BRANCH`/`DECISION` lines per P5.

---

## Results

Not run. Pause for pre-run review of the gate, the ρ-invariance verdict,
and the code.
