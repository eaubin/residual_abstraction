# Experiment 28 — Oracle-withdrawal 6: consolidation

**Script:** — (consolidation; no run, like exp 22).

**Status: concluded.** Consolidates the oracle-withdrawal reference arc
(exps 23–27), promotes the `pstack` indices and new typed findings to
`BATTERY.md`, settles the FORMALISM ledger, and decides program disposition.

## The result, both halves (the headline — do not round up)

Across exps 23–27 the honest story has **two halves that must travel
together**:

- **Selection half — a typed NEGATIVE.** Observable diagnostics did **not
  uniquely earn** a reference. Exp 24 returned `REFERENCE_AMBIGUITY_CONFIRMED`
  (a tie among compact candidates, with an apparent distinct-subspace split);
  exp 25 showed that split is **seed-fragile sampling noise** (reproduces
  3/8). Net: observable selection ties among near-coincident estimates and
  cannot single out *the* reference. The anchor used downstream is therefore
  **declared by convention** (the interventionally-discovered `cegar` core),
  **not uniquely earned**.
- **Transfer half — a POSITIVE.** Under that *declared* anchor, the
  six-member battery transfers on `pstack` (exp 27,
  `BATTERY_TRANSFERS_WITH_RECALIBRATION`): all six members produce coherent,
  non-vacuous numbers, five clean, with one live recalibration (the lenient
  equivalent ρ band).

**The honest one-line summary:** *the hidden-oracle workflow yields a usable
six-member battery on a new process under a reference that was observably
selected but not uniquely earned.* The word "earned" (exp-27 headline)
overstates it — observable selection returned a typed non-uniqueness, and
that negative belongs in the program summary next to the transfer positive.
"Oracle-withdrawal works" is **not** the claim; "the workflow transfers
under a declared-by-convention anchor, while oracle-free unique selection
did not" is.

A second honesty note carried from exp 26/27: `pstack` did not turn out
meaningfully richer than Mess3 (near variance-mimicry, `cegar`≈`pca`
behaviorally), so the transfer is **system-level** — the protocol cleared
end-to-end — not new battery physics on a harder substrate.

## Per-experiment ledger (23–27)

| # | what it decided |
|---|---|
| 23 | `pstack` substrate is competent, exact-auditable, sampled-calibrated at 1024, stratified, non-vacuous (all gates pass). |
| 24 | Observable selection ties (`REFERENCE_AMBIGUITY_CONFIRMED` at seed 0); near-equal closure ≠ same reference — but driven by `pca`'s marginal 12.4° separation. |
| 25 | That two-reference split is **seed-fragile** (clean outlier 3/8, joint 2/8): one stable ~k=4 reference; the multiplicity was a marginal-seed artifact. |
| 26 | One coherent stable reference over 8 fresh seeds (`cegar` closure std 0.005); ρ separates the extremes (`BANDS_TRANSFER`, sep 0.830); `cegar`–`pca` `PARTIAL_MIMICRY` (~10°) but behaviorally inert at ρ. |
| 27 | Battery transfers under the declared `cegar` anchor (4 fresh seeds): M1/M3/M4/M5/M6 clean; M2 recalibrates — the **lenient equivalent band** (recalibrate 0.25→≈0.10, manual single-cell read). |

## Promotions to `BATTERY.md`

`pstack` is added as a **third transfer-layer process** (exps 23–27),
indexed by `L1`, `m=3`, the registered positions, the **declared** `cegar`
core as trusted reference, and 4–8 fresh seeds. Per-member transfer records
and the new typed findings are written into `BATTERY.md` (members 1–6
calibration column; failure-mode map; honest-residual). The new typed
findings:

- **Lenient equivalent ρ band (member-2 caveat, new).** On `pstack` the
  transferred `0.25` equivalent ceiling **over-accepts** an
  intermediate-strength, directionally-distinct patch (`emb`: ρ ≈0.18 at
  exact ≈0.78 vs the core's ≈0.93). The *distinct separation* transfers; the
  *equivalent ceiling* recalibrates to **≈0.10** (a **manual read** off the
  envelopes — estimates ≤0.044, `emb` ≥0.177 — not a script-validated
  threshold; one intermediate cell, needs a sweep to pin). ρ's
  magnitude-sensitivity remains the open exp-26 question.
- **Directional obs/exact recalibration (member-5 refinement, new).** When
  the obs/exact band recalibrates per-process, widen **only** the
  conservative side (`exact > obs`); **hold** the inversion side
  (`obs > exact`) at the transferred value — else recalibration re-loosens
  the over-trust guard. (On `pstack` M5 passed within `0.10` anyway,
  gap 0.026.)
- **Reference-selection non-uniqueness (a typed oracle-free outcome,
  new).** Observable selection can return a tie among near-coincident
  estimates whose apparent distinctness is seed-fragile; the resolution is a
  *declared* anchor, not an earned one. This is a typed reason oracle-free
  *selection* is not (yet) sufficient on its own — distinct from
  oracle-free *scoring/transfer*, which worked.

## FORMALISM ledger

Both §7 rows (the ρ-reference bet and the framing bet) are settled to the
consolidated result: on `pstack` the no-oracle reference is **declared, not
uniquely earned** (selection non-unique, exps 24/25), but it anchors a
**transferring** six-member battery (exp 27) with the lenient-band caveat —
the framing bet ("exact-toy adjudication calibrates oracle-free work") is
supported for *scoring/transfer* and returned a *typed negative for unique
selection*. LLM-scale reference selection remains **open**.

## Decision — program disposition

The oracle-withdrawal **reference arc closes** with this mixed typed result.
Per `ORACLE_WITHDRAWAL.md`'s closing bar, the program can honestly say:
*hidden-oracle reference selection does not uniquely earn a reference on
`pstack` (typed non-uniqueness), yet the workflow transfers a usable battery
under a declared anchor; the battery is updated with the new typed findings.*

Remaining units are **optional and lower-value on this substrate**:

- **Unit 4 (sampled-completion uncertainty).** A different axis (degrade the
  oracle to finite samples). Substrate-ready (exp 23 calibrated sampling at
  1024). Worth running only if the sampling stressor is itself the goal.
- **Unit 5 (proposal-family competition).** Its premise ("after reference
  selection is usable") is weakened by the selection non-uniqueness, and
  `pstack`'s near-mimicry means little discriminating power over PCA.

**Recommendation:** close the oracle-withdrawal program here on the mixed
result rather than grind further on a substrate that did not turn out
rich; treat units 4/5 as future work to revisit only on a richer process.
This is a user decision (close vs run unit 4); flagged at conclusion.

## Scope

This consolidation makes no new measurement. Every promoted number carries
its experiment's index (process, layer, horizon, positions, seeds, declared
reference, transferred/recalibrated thresholds). The `≈0.10` ρ-equivalent
recalibration is a manual single-cell read, not validated. The transfer is
system-level on a near-mimicry substrate. `pstack` claims do not extend to
other layers/horizons or to scale.

---

## Result

Consolidated. The reference arc's record is complete and honest in both
halves: oracle-free unique selection returned a typed negative; the
hidden-oracle workflow transferred a usable battery under a declared anchor.
