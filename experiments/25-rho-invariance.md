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
the ambiguity *matters downstream*, never whether it was resolved.

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

Two observable stages are computed and **printed in full before any exact
oracle is read**:

- Arm A clustering: principal angles among the tied compact candidates
  across seeds (residuals/tokens/model behavior only);
- Arm B ρ-verdicts: ρ(C, X) bands for each probe `X` under each anchor
  `C`, and the observable anchor-invariance verdict.

ρ is model-vs-model (`mean J(q_C, q_X) / mean J(q_C, q_un)`, `battery`
`Exact.rho` / `jeffreys_rows`); the script computes it through a local
observable helper that never touches the process tables. The exact oracle
(`Exact.closure`, process m-grams) is read only in the audit stage, after
the Arm A clustering table and the Arm B ρ-verdict table with its
anchor-invariance line have been printed. Exact audit confirms the anchors
are exact-equivalent references and calibrates the ρ bands; it does not
choose the anchor, the probe set, or the bands.

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
| Arm B anchors | `C in {pca, cegar}` (+ `delta` as a same-cluster consistency check) |
| Arm B probes | `{cegar, pca, delta, emb, rand, full}` |
| clustering threshold | `10°` max principal angle (Exp-24 `AMBIG_ANGLE_MAX`) |
| exact-indifference band | tied exact-closure spread `<= 0.05` |
| ρ calibration known cases | references mutually `<= 0.25`; `rand` `>= 0.5` |

## Arm A — Seed-Stability Gate (prerequisite)

For each of the `8` seeds, the script reruns the Exp-24 selection pipeline
(same `build_candidates`, strata guards, eligibility, tie rule) and
records: the eligible tied set, the pairwise principal angles among
`{cegar, pca, delta}`, and their exact closures. The gate is the
conjunction of three pre-registered checks (each "reproduces" = holds in
`>= 6/8` seeds):

- **G1 tie reproduces.** The observable rule yields a multi-candidate tie
  (`>= 2` eligible compact candidates within the `0.03` margin).
- **G2 clustering reproduces.** The same single candidate is the `10°`
  outlier — `cegar–delta <= 10°` and the outlier's two angles `> 10°` —
  with the outlier identity stable (Exp 24: `pca`).
- **G3 exact-indifference reproduces.** The tied candidates' exact-closure
  spread is `<= 0.05` (no "right" reference emerges under audit).

Scope of stability: this varies the pair sampling and the basis draw
(`240` seqs) at a **fixed model checkpoint** — exactly the single-seed
concern Exp 24 flagged. Model-retraining stability is a larger question and
is out of scope.

Arm-A outcomes:

- `GATE_PASS` (G1∧G2∧G3): a stable two-reference geometry exists; run Arm
  B.
- `SEED_UNSTABLE_TIE` (G1 fails): the ambiguity itself is seed-fragile.
- `SEED_UNSTABLE_CLUSTER` (G1 holds, G2 fails): the `{pca | cegar,delta}`
  split is sampling noise; the ρ-invariance question is moot.
- `EXACT_INDIFFERENCE_BREAKS` (G1∧G2 hold, G3 fails): the tied candidates
  are *not* reliably exact-equivalent — a "right" reference may exist after
  all, which **re-opens the auditable tie-break** Exp 24 withdrew. This is
  a typed reframing, not a failure.

If the gate does not pass, Arm B is not run (it would be vacuous or
mis-posed); the script prints the typed Arm-A outcome and exits with the
corresponding decision.

## Arm B — Downstream-ρ Invariance (primary, conditional on `GATE_PASS`)

At the reference seed (`0`), for every probe `X` in the registered suite,
the script computes ρ(C, X) under each anchor `C in {pca, cegar}` on the
eval PairSet at horizon `mm=3`, and assigns the battery band
(equivalent / indeterminate / distinct). All ρ-verdicts and the
anchor-invariance line are printed **before** exact audit.

- **Primary (observable) — anchor invariance.** For each probe, is the ρ
  band identical under `C=pca` and `C=cegar`? `RHO_ANCHOR_INVARIANT` iff
  every probe's band agrees across the two anchors; otherwise
  `RHO_ANCHOR_SENSITIVE`, listing the probes whose band flips. (`delta`,
  same cluster as `cegar`, is also run as a `C` to confirm same-cluster
  anchors agree — a within-cluster sanity line, not part of the primary.)

Then the exact oracle is revealed:

- **Audit 1 — anchors are exact-equivalent.** Exact closure of `pca` and
  `cegar` (and `delta`) within the `0.05` indifference band, reconfirming
  neither anchor is privileged (so any ρ disagreement is an artifact, not a
  real difference the oracle would adjudicate).
- **Audit 2 — ρ calibration under each anchor.** Under each `C`, the
  reference probes are ρ-equivalent (`<= 0.25`) and `rand` is ρ-distinct
  (`>= 0.5`). Failure = `RHO_MISCALIBRATED_ON_PSTACK` (a battery member-2
  recalibration finding, distinct from anchor sensitivity).

## Predictions

**P1 (substrate + self-checks; enforced).** Registered config, PairSet
self-checks, global non-degeneracy, and the Exp-24 strata guards hold at
every seed. Failure = typed substrate halt, not an Exp 25 result.

**P2 (seed-stability gate; ~55%).** `GATE_PASS`. The honest prior is near
a coin flip: `pca`'s separation is marginal (`~2.4°` over the `10°` line)
at one seed, and the gate asks it to reproduce as a stable partition. A
gate failure is an informative reframing of Exp 24's geometry, not a code
failure.

**P3 (ρ anchor-invariance; ~60%, given `GATE_PASS`).** `RHO_ANCHOR_INVARIANT`.
Because `pca` and `cegar` are exact-equivalent, ρ *should* call the same
probes equivalent under either anchor; but a `10–12°` subspace difference
could make ρ's mean-level Jeffreys anchor-sensitive — the interesting
failure. Both outcomes are findings.

**P4 (ρ calibration; ~85%, given `GATE_PASS`).** The battery bands
(`0.25`/`0.5`) classify the known cases correctly under each anchor (Dyck
transferred them; pstack is the open question).

**P5 (decision; deterministic).**

- `GATE_PASS ∧ RHO_ANCHOR_INVARIANT ∧ calibrated` →
  `AMBIGUITY_DOWNSTREAM_BENIGN`: which equally-good reference anchors ρ
  does not change the battery's equivalence verdicts. **GO** to preregister
  Block 3 (battery transfer) under a *declared canonical anchor* (the
  interventionally-discovered `cegar` core; `delta` is subsumed, `pca` is a
  ρ-equivalent control). Firewall: this is "the ambiguity is downstream-
  benign," **not** "selection succeeded" — Exp 24's NO-GO on unique
  selection stands.
- `GATE_PASS ∧ RHO_ANCHOR_SENSITIVE` → `AMBIGUITY_LOAD_BEARING`: the
  reference choice flips battery verdicts on an oracle-unadjudicable basis.
  **NO-GO**; the reference-selection failure propagates into the battery —
  a sharper failure type. Register the repair (e.g. a reference-robust ρ or
  a stronger selection rule) before transfer.
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

- Stability is over pair/basis **sampling** at a fixed checkpoint, not over
  model retraining.
- The ρ bands, the `10°` clustering threshold, the `0.05` indifference
  band, and the `6/8` gate majority are the registered indices; a different
  choice could move `GATE_PASS`.
- ρ carries the member-2 mean-level caveat (BATTERY.md): equivalence is a
  mean-Jeffreys statement, not per-pair.
- Exact closure is evaluation-only; sampled completions are not used (exact
  model chain probabilities throughout, as in Exp 24).
- The result is indexed by `pstack`, `L1`, `m=3`, the registered positions,
  the Exp-24 candidate family, and these thresholds.

## Expected Output

The script prints, in order:

- per-seed Arm-A table (eligible tied set, `{cegar,pca,delta}` pairwise
  angles, exact closures) and the three gate checks with their `k/8`
  counts;
- the typed Arm-A outcome (and, on gate failure, the decision and exit);
- on `GATE_PASS`: the Arm-B ρ-verdict table (probe × anchor band) and the
  `RHO_ANCHOR_INVARIANT` / `RHO_ANCHOR_SENSITIVE` line — all before audit;
- the exact-audit block (anchor exact-equivalence; per-anchor calibration);
- `AUDIT`/`DECISION` lines per P5.

---

## Results

Not run. Pause for pre-run review of the gate, the ρ-invariance verdict,
and the code.
