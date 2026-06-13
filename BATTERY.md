# The frozen diagnostic battery — Mess3 calibration and Dyck transfer

**Scope statement, governing every claim below.** This document has two
layers:

1. **Calibration layer (Mess3, exps 1–18).** On Mess3-L4 at patch point
   L1, over the registered patch families (rank-1..k linear interchange,
   oblique (write, read) pairs and their compositions), the sampled
   regimes (benign; adversarial κ ∈ {30, 100, 300}, five junk draws at
   κ = 100), the registered position sets and evaluation distributions,
   and horizons mm ∈ {1, 2, 3, 4}: the diagnostics below predicted or
   tracked exact closure and transport behavior.
2. **Transfer layer (Dyck-2, exps 19–22).** On the registered Dyck-2
   checkpoint at L1, using the honestly discovered exp-19 rank-4 core as
   trusted reference, all six battery members transferred under their
   registered Dyck indices across exps 19–21. Those indices include
   baseline calibration, adversarial-coordinate checks, mild registered
   shifts, and signed prefix-balance strata where each was registered.
   Members 1, 2, 5, and 6 were additionally swept over horizons
   mm ∈ {1, 2, 3, 4}, eps ∈ {0.01, 0.02, 0.05, 0.10}, and
   κ ∈ {30, 100, 300} for `junk_seed = 0`.

**No claim here extends beyond these indices.** The calibration and
transfer records license carrying these instruments to settings without
ground truth only as evidence, not proof.

**Phase status.** Adopted at the exp-18 conclusion and updated at exp
22: Mess3 is appendix/debugging; Dyck-2 has served as the first transfer
process. Phase 2 closes with no new battery-transfer failure type. The
next program should move to a process class with weaker ground truth
rather than add another Dyck block, unless a review names a specific
unresolved Dyck debt.

## Members

| # | diagnostic | predicate (code home) | detects | calibration record | scope & caveats | ground truth needed? |
|---|---|---|---|---|---|---|
| 1 | **observable closure** (model-vs-model scoring) | c_obs(P) = (D₀ − D(P)) / (D₀ − D_full), D(P) = mean KL(q_src-run ‖ q_patched) over discovery pairs (`battery.Refs.obs`) | causal transfer, scored without ground truth | Mess3: 6 consecutive agreement holds (exps 13–18); exp 18 worst gap 0.017 over 52 cells. Dyck: exp 19 core `c_obs = 98.5%`, exact eval `92.6%`; exp 20 shifted gains ≥ 98.5%; exp 21 stable through mm=4 | needs a disjoint eval split; scale untested | no |
| 2 | **per-pair equivalence ratio ρ** | ρ(X) = mean J(q_C, q_X) / mean J(q_C, q_un), J = Jeffreys; bands ≤ 0.25 equivalent / ≥ 0.5 distinct (`battery.Exact.rho`, `battery.jeffreys_rows`) | behavioral equivalence to a trusted reference; separates working from destructive patches | Mess3: separator validated (exp 15), positive direction at κ = 30, horizon-stable (exp 18). Dyck: bands transfer with 69× separation at exp 19; exp 21 equivalent max 0.187, distinct min 0.998 through mm=4; z-id remains distinct across κ | the reference must be trusted; Dyck shows a discovered core can serve as that reference when exact checks validate it; mean-level caveat remains | no, given a trusted reference |
| 3 | **held-out-position gain** | c_obs on discovery-side pairs at positions excluded from training *and* selection (`battery.Refs.obs` on a held-out PairSet) | position-entangled statistical control (the exps-15/16 failure mode) | Mess3: flagged entanglement at κ = 100 and passed transported κ = 30 reads. Dyck: exp 19 core held-out-position gain `+98.7%`; exp 20 position shift gain `+99.1%` | tested positions are interior/interpolation-style; Dyck result covers the registered positions only | no |
| 4 | **shift-retention R** | R(X, s) = [gain_X(s)/gain_X(base)] / [gain_C(s)/gain_C(base)] under registered shifts, with competence (model-vs-exact NLL) and clean-gain guards (`battery.shift_retention`) | distribution-local control; fragility is decisive, robustness is *not* shift-immunity | Mess3: decisive in both directions (exp 15), guards passed. Dyck: exp 20 core retention `R = 1.00` under position and depth-profile shifts; guards passed; shifted worst obs/exact gap 0.073 | mild registered shifts only; needs the reference patch for normalization; Dyck shifts not rerun across mm | no, given a trusted reference |
| 5 | **accepted-cell calibration** (the P4 protocol) | acceptance = observable ≥ 20% per (patch, set) cell; check \|obs − exact\| ≤ 0.10 on every accepted cell, coverage audited symmetrically (§6.1 rule 8) (`battery.Exact.closure`, `battery.calibration_gap`) | objective hacking / scoring drift — never observed, including under gradient selection pressure | Mess3: harness behind member 1's record. Dyck: exp 19 worst gap 0.064; exp 20 shifted worst gap 0.073; exp 21 worst horizon gap 0.073 through mm=4 | **this is the calibration procedure itself** — its exact side does not transport; at scale it is replaced by the toy record + members 2–4 as consistency checks | **yes** (calibration-time only) |
| 6 | **CEGAR accept-count & staircases** | the frozen acceptance loop (accept iff marginal observable gain ≥ eps); k\*(eps), accept(eps), k\*(m) reported as staircases, never points (`battery.cegar_loop`, `battery.cegar_accept`, `battery.cegar_staircase`) | false-confidence discovery; threshold and horizon sensitivity | Mess3: acceptance sound under adversarial coordinates (exp 8), all draws (exp 17), all horizons (exp 18). Dyck: exp 19 staircase `5,4,4,3`; exp 21 same staircase at every mm=1..4; adversarial accept-counts zero for every registered κ×mm×eps cell | acceptance is sound but proposal-dependent; Dyck uses one junk draw, so no Dyck draw-genericity claim | no |

## Failure modes ↔ detectors

| failure mode (typed, with the experiment that named it) | detected by |
|---|---|
| objective hacking (exp 13's registered dual reading) | 1 + 5 — never materialized |
| position-entangled statistical control (exps 15–16) | 3, with 4 corroborating |
| distribution-local equivalence (exp 15) | 4 |
| junk-amplified destruction (exps 10–11) | 2 (ρ 5.3–14 vs ≤ 0.44) |
| proposal variance-dependence / false discovery (exp 8) | 6 + the §5 invariance principle |
| read-side geometry ≠ function (exps 13, 17) | not detectable geometrically — 2 and 3 are the *behavioral* instruments that replaced geometric read diagnostics (pooled EPR deprecated, exp 15) |
| single-write rank-1 probe failure (exp 20) | not a battery-member failure; record as a scoped negative probe unless a future rank-1 search with multiple writes/checkpoints is registered |

## What transports and what does not

**Usable without ground truth:** members 1, 3, 6 outright; 2 and 4 given
a trusted reference patch. **Oracle-dependent, calibration-time only:**
member 5 (exact belief-conditioned closure) and the T-aware clean
construction — on a new process class with known structure these run
again as the calibration harness; on a model without ground truth they
are exactly what the toy record substitutes for.

**The honest residual.** The program's framing bet (§7: "exact-toy
adjudication calibrates later oracle-free work") is now supported across
Mess3 calibration and one successful transfer process, Dyck-2, at the
strength of this document's scope statement and no further. Known limits
carried forward: scale; no-oracle reference selection; non-linear
interpreter classes; extrapolation beyond trained position ranges;
shifts stronger than the registered mild pairs; multiple Dyck junk
draws; and every claim's κ-grading where noted (the gradient-pathology
results are κ-indexed; Dyck Block 3 did not test gradient-read
transport).
