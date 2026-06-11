# Experiment 17 — T-robustness and the eps_gain staircase: is the adversarial story an artifact of one transform and one threshold? — PRE-REGISTRATION

**Script:** `tsweep.py` (on `expcommon.py`). **Status: pre-registered;
NOT YET RUN — results to be appended below the marked line.**

**Question.** Every adversarial conclusion since exp 8 is indexed by one
junk-plane draw and one κ (the ledger's oldest debt), and every
accept/reject claim by eps_gain = 0.05. Are the load-bearing conclusions
**T-generic** and **threshold-robust**?

**External proposal, evaluated.** Adopted: the split (T + eps now, m
deferred — the m-staircase changes the semantic target γ_m itself, a
different question from "is this an artifact of an arbitrary choice");
5 draws at κ = 100; the standard battery per draw rather than the full
historical sequence; staircases reported as tables, not single
thresholds. **Extended, one point:** the ledger debt is "one junk draw,
*one κ*" — and κ, unlike m, is the same artifact question — so a small
secondary κ arm is included (draw 1 at κ ∈ {30, 300}, descriptive
claims only). **Design choice beyond the proposal:** draw 0 *is* the
historical T, so its battery doubles as the reproduction tripwire
(gradient finals, clean/id gains, angles, D2 — asserted against
recorded exps 14–16 values; halt enforced on mismatch), and 4 new draws
carry the genericity claims.

## Design (deltas from the standard setting)

**Units.** A unit = (junk_seed, κ). Primary: junk seeds 0–4 at κ = 100.
Secondary: junk seed 1 at κ ∈ {30, 300}. T per unit via the registered
construction with the seeded junk draw; both transform checks asserted
per unit. **Shared pair sets across all units** (disc seed+111, ev_train
seed+777, disc_val seed+333 at {12, 20}) — the pairs are T-independent,
so the transform is the only varying factor.

**Per-unit battery** (the load-bearing core, each item tied to the
conclusion it audits):

1. *Adversarial CEGAR accept-count staircase* (audits exp 8's
   proposal-death + the eps policy): the frozen exp-6 loop run in the
   unit's z-view (proposals mined on z, patches scored after pullback —
   behavioral scoring is coordinate-free by §5), k_max = 4, at each
   eps ∈ {0.01, 0.02, 0.05, 0.10}; report accepted count and the first
   proposal's gain.
2. *Write pool, round 1* (audits exp 10's reachability): nearest-write
   angle; the two nearest ≤ 15° writes by the exp-12 rule.
3. *Read contrasts on the nearest write* (audits exps 10–12): id-read
   gain, clean-read gain, best spectral-grid gain (full α-grid, both
   writes).
4. *Gradient runs* (audits exps 13–16's landscape + entanglement): the
   registered optimizer (id init, 200 steps) on both writes; phenotype
   classified diverged (final train gain ≤ −100%) / converged (≥ +20%) /
   intermediate (reported, not classified). For converging reads:
   held-out val observable gain, ρ on ev_train vs the unit's clean
   patch, plane/junk/neutral decomposition, obs/exact pair.
5. *Clean D2 composition* (audits exp 12): exact closure vs full.

**Benign eps staircase** (T-independent, run once): the exp-6 anchor
loop at each eps in the grid; report k\*(eps) and the gain sequence.

**Output:** a per-unit summary table (units × {nearest angle, accept
counts, id/clean/spectral gains, phenotypes, converged-read val gain, ρ,
D2}) — the deliverable is tabular by design.

## Pre-registered predictions

- **P1 (draw-0 reproduction + anchors; ~90%).** Benign anchor; draw-0
  battery hits recorded values: nearest angles 1.1°/3.3° (±0.2°),
  id +1.0%, clean +51.3%, gradient finals −548.2%/+42.5% (±2 pts),
  D2 97.8% (±2 pts). Reproduction failure **halts** (enforced).
- **P2 (proposal death is T-generic; ~85%).** The adversarial loop
  accepts 0 directions at eps = 0.05 for all five κ=100 draws.
- **P3 (write reachability is T-generic; ~80%).** Every κ=100 draw's
  pool contains a write ≤ 5° from the plane.
- **P4 (the core contrasts are T-generic; ~70%).** For every κ=100
  draw: clean-read gain ≥ 40% on the nearest write; id-read gain ≤ 10%;
  best spectral gain ≤ 10%; clean D2 ≥ 90% of full.
- **P5 (landscape population; two clauses).** (a) ~70%: ≥ 1 diverging
  gradient run in at least 4 of 5 draws (the divergent phenotype is
  typical, not draw-0-specific). (b) ~60%: ≥ 1 converging read
  (≥ +20% train) somewhere across the five draws (the w2 phenotype
  exists generically). Always testable.
- **P6 (position entanglement is T-generic; gated on ≥ 1 converging
  read; ~75%).** Every converging read's held-out val gain < 20%.
- **P7 (observable soundness across transforms; ~85%).** For every
  patch in the battery with observable gain ≥ 20% (coverage, per pre-run
  review fix: clean, id, spectral-best, D2, and converged gradient
  patches — the full battery): |observable − exact| ≤ 0.10. The 5th
  consecutive P4-style test, now across transforms.
- **P8 (eps staircase; two clauses; ~70%).** (a) Benign k\*(eps) = 2
  for every eps in the grid (the benign discovery is
  threshold-robust). (b) Adversarial accepted count = 0 for every draw
  at every eps ≥ 0.02; eps = 0.01 is reported descriptively (noise
  acceptance at the bottom of the grid does not fail the clause).
- **P9 (validity gate, enforced).**
- **Secondary (κ arm, descriptive; registered expectation ~65%).** The
  *full* P4 contrasts — including the D2 clause (pre-run review fix; the
  first draft's check dropped it) — hold at κ = 30 and κ = 300 on
  draw 1; phenotypes and staircases reported.

## Scope & local assumptions

- Five draws sample T-space sparsely; genericity claims are per-draw
  threshold claims, not measure statements over transforms.
- The κ arm is secondary and descriptive (two points, one draw); the κ
  half of the single-T debt is discharged only at that strength.
- The eps grid {0.01, 0.02, 0.05, 0.10} brackets the frozen 0.05;
  staircase claims are grid-indexed.
- The per-unit battery is the load-bearing core; historical results not
  re-run (e.g., exp-15's shifts, exp-16's protocol arms) keep draw-0
  indexing.
- The adversarial loop here is the exp-6 acceptance rule applied in z
  (accept iff marginal gain ≥ eps, stop otherwise) — a registered
  simplification of exp-8's full protocol; accept-counts are what the
  claims use.
- Gradient runs: id init only, registered budget (endpoint comparability
  with exps 15–16); the phenotype thresholds (≤ −100% / ≥ +20%) leave an
  intermediate band that is reported, never silently classified.
- Shared pair sets isolate T as the only varying factor; draw-level
  variation therefore cannot be pair-sampling noise.

**Ledger updates (with this registration).** Single-T indexing →
**under test (exp 17)**; eps_gain tolerance policy → **under test
(exp 17)**. (No new global rows; the battery instruments are already
ledger-resident.)

**Self-checks** (standard set, plus): (i) `build_transform(junk_seed=0)`
equals the historical construction bitwise; (ii) distinct junk seeds
give distinct T's (pairwise non-equal asserted); (iii) the phenotype
classifier's boundaries on synthetic values (−1.0, −0.99, +0.19, +0.20).

**Enforcement.** Standard. Estimated runtime **~4–5 h** (7 units × 2
gradient runs ≈ 3 h + ~20 chain evaluations per unit + staircases).

---

*(Results to be appended here after the run.)*
