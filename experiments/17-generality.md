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

**Post-registration, mid-run note (two attempts — recorded in full).**
The first run crashed at unit (j=1, κ=300): the transform-check assert
used a fixed atol = 10⁻⁹, which float64 roundoff exceeds at κ = 300. A
first fix assumed a κ² roundoff law and also failed. The error was then
*measured* (max |err| 8.7×10⁻¹² / 1.2×10⁻⁹ / 8.2×10⁻⁸ at κ = 30/100/300
on synthetic draws): the pullback-product roundoff scales ~κ⁴, and the
historical 10⁻⁹ at κ = 100 had thin margin all along. Final tolerance:
bit-identical 10⁻⁹ at κ ≤ 100; the measured κ⁴ law with a 10× margin
above. An infrastructure artifact throughout — all five primary draws
and the κ=30 unit completed identically both times (deterministic under
the shared seeds); the run was restarted for a clean canonical log.

---

## Results: P1–P9 ALL HOLD (secondary κ expectation fails, informatively) — the adversarial core is T-generic at κ = 100, the thresholds were never load-bearing, and the gradient-pathology story is κ-GRADED: at κ = 30, learned reads transport

(Registered parameters, seed 0, gate +0.0024 PASS; draw-0 battery
reproduced recorded values exactly. Raw output `out/exp17_mess3-L4.txt`
— the per-unit summary table at its foot is the canonical deliverable.
Two infrastructure crashes en route, both the same transform-check
tolerance, documented in the mid-run note above.)

**Finding 1 — full T-genericity at κ = 100 (P2, P3, P4, P5, P8b).** All
five junk draws produce *identical* core numbers: accept-count 0 at
every eps, nearest write M2\*Sinv at 1.1°, id +1.0%, clean +51.3%,
D2 97.8%. The nearest write comes from the same candidate source
(M2\*Sinv) at the same angle with contrast numbers identical to the
decimal in every draw — consistent with exp 9's GL(d)-invariance
proposition (whitened mining is the one draw-independent candidate
source); vector identity itself was not asserted (review fix). What *does* vary with the junk draw is the
gradient landscape of the second write (3.3°): converged in 2 draws,
diverged in 2, intermediate in 1 — divergence is typical (≥ 1 per draw,
5/5; P5a) and the landscape is junk-draw-sensitive in detail. Where
reads converged, entanglement held (val −14.9%, −6.6%; P6).

**Finding 2 — the κ arm's discovery: the pathology is κ-graded, and at
κ = 30 behavioral-gradient discovery simply works.** At κ = 30 both
gradient runs converge (+52.5%/+51.5% train — *exceeding* the clean
read's +51.2%), **transport** (val +40.0%/+37.4%), and are nearly
clean-equivalent (ρ = 0.046/0.075) — by exps 15–16's own criteria these
are transportable members of the clean patch's equivalence class, found
honestly by the gradient. And they still carry **zero plane mass**
(junk 38%/26%, neutral 62%/74%): the zero-plane geometry that looked
pathological at κ = 100 is, at κ = 30, the geometry of *working,
position-generic* reads — geometry ≠ function, now confirmed in the
positive direction. At κ = 300 the opposite pole: both writes diverge
(including one at 0.1°, further supporting the nearer-plane-worse
landscape pattern). So the exps 13–16 story — divergence, position
entanglement, "clean-plane access is special" — is **a property of the
adversarial-conditioning regime, graded by κ**: absent at 30, mixed at
100, total at 300. Exp 16's review-driven "intrinsic *in the registered
geometry*" indexing now does real work: the thread's conclusions stand
at κ = 100 (now draw-generic), with the single κ = 300 probe supporting
the high-κ pole — re-indexed, not reopened (review fix: "κ ≥ 100" would
overstate two probe points into a measured range). The regime boundary κ\* is
unmeasured (a 2-point arm cannot locate it) — recorded as a scope note,
not a new debt.

**Finding 3 — the thresholds were never load-bearing (P8).** Benign
k\*(eps) = 2 across the whole grid (0.01–0.10); the adversarial loop
accepts nothing even at eps = 0.01 (first-proposal gains −1.4% to +0.8%,
below every tested threshold) — stronger than the registered clause,
which only required 0 at eps ≥ 0.02. The eps_gain debt is discharged on
the tested grid.

**Finding 4 — observable soundness, 18 for 18 across seven transforms
(P7).** Every battery patch over the 20% bar — clean, D2, and converged
gradient reads, across all five draws and both κ-arm units — tracks
observable to exact within 1.5 points. The 5th consecutive hold
(exps 13–17; review fix — the first draft miscounted) and by far the
widest replication; the instrument is now validated across the very
transform family it was built to survive.

**Finding 5 — the secondary κ expectation (registered ~65%) fails on
two informative counts.** At κ = 30, clean D2 = 85.2% < 90% of full —
a *pool-geometry* effect, not a composition failure: the κ=30 pool's
nearest writes sit at 6.7°/10.9°, and clean composition of
farther-from-plane writes simply caps lower. At κ = 300, best spectral
= +12.3% > 10%. Neither touches the primary claims; both are now on
record.

**Ledger effects.** Single-T indexing: **discharged in its draw half**
(T-draw-generic at κ = 100, five draws) and **scoped in its κ half**
(gradient-thread conclusions carry a κ index; measured at three
points). eps_gain: **supported** (discharged on the grid).
Position-locality of learned reads: **κ-graded** — at κ = 30 learned
reads transport. Observable scoring: held a 5th time, 18 cells.

**What remains.** The m-staircase (deferred by design); the
self-certification battery consolidation — strengthened by this run: ρ
flagged the κ=30 reads as clean-equivalent (0.05) exactly where they
transported, its first positive-direction validation; and the κ\*
regime boundary if the program ever needs it.

**Status: CONCLUDED.**
