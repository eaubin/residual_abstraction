# The frozen diagnostic battery — Mess3 calibration consolidation (exps 1–18)

**Scope statement, governing every claim below.** On Mess3-L4 at patch
point L1, over the registered patch families (rank-1..k linear
interchange, oblique (write, read) pairs and their compositions), the
sampled regimes (benign; adversarial κ ∈ {30, 100, 300}, five junk
draws at κ = 100), the registered position sets and evaluation
distributions, and horizons mm ∈ {1, 2, 3, 4}: the diagnostics below
predicted or tracked exact closure and transport behavior. **No claim
here extends beyond these indices.** The calibration record is what
licenses carrying these instruments to settings without ground truth —
it is evidence, not proof, of transfer.

**Freeze.** Adopted at the exp-18 conclusion: Mess3 work from here on is
appendix/debugging only, unless a new process class exposes a failure
that needs a Mess3 back-check. The next program phase is a new process
class. (Dyck-2 already exists in-repo as the natural bridge — exp 7's
representation–oracle mismatch is the known open thread there.)

## Members

| # | diagnostic | predicate (code home) | detects | calibration record | scope & caveats | ground truth needed? |
|---|---|---|---|---|---|---|
| 1 | **observable closure** (model-vs-model scoring) | c_obs(P) = (D₀ − D(P)) / (D₀ − D_full), D(P) = mean KL(q_src-run ‖ q_patched) over discovery pairs (`expcommon.observable_refs`) | causal transfer, scored without ground truth | 6 consecutive agreement holds (exps 13–18); exp 18: 52 cells, worst gap 0.017, every horizon; exp 17: 18/18 across seven transforms; faithful over a ~550-pt outcome range (exp 14) | needs a disjoint eval split (selection-on-discovery is bounded only by that); scale untested | no |
| 2 | **per-pair equivalence ratio ρ** | ρ(X) = mean J(q_C, q_X) / mean J(q_C, q_un), J = Jeffreys; bands ≤ 0.25 equivalent / ≥ 0.5 distinct (`expcommon.jeffreys_rows`) | behavioral equivalence to a trusted reference; separates working from destructive patches (≥ 10–23×, monotone with transfer) | separator validated (exp 15); positive direction at κ = 30 — ρ ≈ 0.05 exactly where reads transported (exp 17); horizon-stable (exp 18) | the reference must be *trusted* (toys: the T-aware clean patch; at scale: the best-validated patch — trust must come from elsewhere); mean-level (frac_worse 32% even at ρ = 0.20) | no, given a trusted reference |
| 3 | **held-out-position gain** | c_obs on discovery-side pairs at positions excluded from training *and* selection | position-entangled statistical control (the exps-15/16 failure mode) | flagged entanglement at κ = 100 (reads invert, R −0.77/−0.41); passed the transported κ = 30 reads (+40%/+37%); horizon-stable (exps 15–18) | tested positions interior to the trained range — certifies interpolation only | no |
| 4 | **shift-retention R** | R(X, s) = [gain_X(s)/gain_X(base)] / [gain_C(s)/gain_C(base)] under registered shifts, with competence (model-vs-exact NLL) and clean-gain guards | distribution-local control; fragility is decisive, robustness is *not* shift-immunity | decisive in both directions (exp 15: position shift exposed entanglement while the prefix shift was robust); guards passed cleanly | mild registered shifts only; needs the reference patch for normalization; not rerun across m | no, given a trusted reference |
| 5 | **accepted-cell calibration** (the P4 protocol) | acceptance = observable ≥ 20% per (patch, set) cell; check \|obs − exact\| ≤ 0.10 on every accepted cell, coverage audited symmetrically (§6.1 rule 8) | objective hacking / scoring drift — never observed, including under gradient selection pressure | the harness behind member 1's record; six experiments of cells | **this is the calibration procedure itself** — its exact side does not transport; at scale it is replaced by the toy record + members 2–4 as consistency checks | **yes** (calibration-time only) |
| 6 | **CEGAR accept-count & staircases** | the frozen acceptance loop (accept iff marginal observable gain ≥ eps); k\*(eps), accept(eps), k\*(m) reported as staircases, never points | false-confidence discovery (none observed: zero junk acceptances over every regime, draw, horizon, and eps tested); threshold sensitivity | acceptance sound under adversarial coordinates (exp 8), all draws (exp 17), all horizons (exp 18); thresholds never load-bearing (exp 17) | acceptance is sound but *proposal-dependent* — the miner is a coordinate-sensitive heuristic (exp 8); soundness here = no false positives, not completeness | no |

## Failure modes ↔ detectors

| failure mode (typed, with the experiment that named it) | detected by |
|---|---|
| objective hacking (exp 13's registered dual reading) | 1 + 5 — never materialized |
| position-entangled statistical control (exps 15–16) | 3, with 4 corroborating |
| distribution-local equivalence (exp 15) | 4 |
| junk-amplified destruction (exps 10–11) | 2 (ρ 5.3–14 vs ≤ 0.44) |
| proposal variance-dependence / false discovery (exp 8) | 6 + the §5 invariance principle |
| read-side geometry ≠ function (exps 13, 17) | not detectable geometrically — 2 and 3 are the *behavioral* instruments that replaced geometric read diagnostics (pooled EPR deprecated, exp 15) |

## What transports and what does not

**Usable without ground truth:** members 1, 3, 6 outright; 2 and 4 given
a trusted reference patch. **Oracle-dependent, calibration-time only:**
member 5 (exact belief-conditioned closure) and the T-aware clean
construction — on a new process class with known structure these run
again as the calibration harness; on a model without ground truth they
are exactly what the toy record substitutes for.

**The honest residual.** The program's framing bet (§7: "exact-toy
adjudication calibrates later oracle-free work") is now supported at the
strength of this document's scope statement and no further. Known limits
carried forward: scale; non-linear interpreter classes (exp 7's
representation–oracle mismatch); extrapolation beyond trained position
ranges; shifts stronger than the registered mild pair; and every claim's
κ-grading where noted (the gradient-pathology results are κ-indexed;
transport was demonstrated at κ = 30, its absence at κ = 100/300).
