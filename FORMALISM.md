# Formalism — named objects, orders, invariance, verdict predicates

The project's dictionary: it names what the experiments measure so registrations
can say "this experiment estimates quantity Q under conditions C" instead of
re-deriving the setting. It introduces **no new claims**. This file is
definitional and changes rarely; per-experiment results live in `EXPERIMENTS.md`
and the writeups, status bets in `ASSUMPTIONS.md`.

## 1. Objects

| symbol | object | realized as |
|---|---|---|
| P | data-generating process | HMM: token-labeled matrices {T_s}, stationary prior π (`processes.py`) |
| μ | prefix distribution | stationary sampling of length-L sequences; positions t pooled per protocol |
| b(w) ∈ Δ(S) | belief state of prefix w | exact Bayes filter; *the process's* minimal sufficient statistic |
| γ_m(w) ∈ Δ(V^m) | completion kernel (concrete semantics), horizon m | exact m-step distribution; **linear** in b(w) |
| M, ρ_ℓ(w) ∈ R^d | model and residual map at stream point ℓ | trained transformer; stream entering block ℓ+1 (ℓ = "final" = pre-ln_f) |
| α : R^d → A | abstraction map | linear: subspace coordinates (rank-k projections), possibly in transformed coordinates |
| V | interpreter class | (a) fitted affine-softmax heads / k-NN — *decode* interpreters; (b) the model's own continuation after an interchange patch — the *causal* interpreter |
| D | divergence | KL(true‖predicted), per-pair rows or means; sym-KL for counterexample mining |
| m | horizon | m = 3 throughout (V^m outcomes); marginals give m = 1, 2 |
| τ | tolerance policy | always reported as curves/staircases plus a declared stopping rule; calibrated against the no-information baseline KL0 |

Two distinct sufficiency notions, separated empirically by Experiment 7:
**oracle sufficiency** (does α(ρ) determine γ_m? judged against b) vs
**model-state sufficiency** (does α(ρ) carry what *the model's own computation*
reads/propagates? judged by intervention). They coincide on Mess3 and dissociate
on Dyck (representation–oracle mismatch).

## 2. What each experiment estimated

Moved. Per-experiment quantities and verdicts live in `EXPERIMENTS.md` (index)
and the canonical `experiments/NN-*.md` writeups.

## 3. Orders on abstractions

- **Simulation preorder**: α ⪯ β iff α = h∘β for some allowed h (here: linear
  maps) — β refines α. Nested rank-k families are chains in this order;
  cross-family comparisons are generally not comparable, which is why the
  scalarizations below carry the comparison weight.
- **Behavioral risk preorder**: α ⪯_V β iff the optimal V-interpreter risk of β
  is ≤ that of α for every horizon/divergence in scope, over the evaluation
  distribution. Decode closure and interchange closure are scalarizations of this
  order, always indexed by (patch point ℓ, scope, horizon m, distribution μ,
  interpreter class V). An unindexed completeness claim is ill-formed in this
  house.

## 4. The three maps, kept separate

1. **Proposal map**: data → candidate α (PCA, X-whitened PLS, head-rows, the
   CEGAR miner). A *heuristic*.
2. **Abstract value**: the α itself — a subspace/patch, evaluable independently
   of how it was found.
3. **Interpreter**: what extracts predictions from α(ρ) — fitted probes
   (V-information) or the model's own downstream computation (causal).

A claim about an abstraction is only as coordinate-robust as the weakest map it
depends on, and the three must be audited separately. The failure taxonomy in §6
is organized by which map failed.

## 5. Invariance

**Principle.** A claim about the residual stream should state how it behaves
under invertible linear reparameterization z = T·ρ. Quantities that are invariant
by construction: the closure of a *given patch* (the pullback T⁻¹QQᵀT / row-form
TQQᵀT⁻¹ construction makes the behavioral scoring path coordinate-free), and
hence every acceptance verdict. Quantities that are not: anything ranked by
variance or raw covariance — PCA orderings, unwhitened PLS, the covariance CEGAR
miner, displaced-variance audits. Coordinate-dependent procedures are admissible
only as heuristics whose outputs are validated by invariant scores; where an
invariant procedure exists, prefer it.

**Proposition (whitened mining is GL(d)-invariant, ridgeless).** Let Σ be the
(population) covariance of the stream at the patch point and let the miner operate
on Σ^{-1/2}-whitened coordinates (weighted second-moment eigenvector of whitened
prefix differences), with the interchange patch defined as the coordinate swap in
whitened coordinates (raw-space patch P = Σ^{-1/2}UUᵀΣ^{1/2} in row convention).
Then for any invertible T, the miner applied to z = ρ·T with Σ_z = TᵀΣT produces
the *same* raw-space patch. *Sketch*: the whitened versions of ρ and z differ by
the orthogonal map O = Σ_z^{-1/2}TᵀΣ^{1/2}; weighted second moments transform by
conjugation with O, eigenvectors map by O, and the induced patches pull back
identically. Finite-sample and ridge-floor effects break this only at the
corresponding tolerance.

## 6. Verdict predicates (the failure taxonomy, as checkable conditions)

| verdict | checkable condition |
|---|---|
| domain coarseness | conflated pairs (abstract-space neighbors) with sym-KL(γ) > τ |
| interpreter incompleteness | head KL > τ while k-NN KL ≤ τ on the same α |
| proposal misalignment | identification R² high from full ρ, low from α's subspace |
| metric junk-domination | counterexamples persist under refinement while decode KL ≤ τ |
| vacuous tolerance | the KL0 baseline already passes τ |
| correlational-but-not-causal (echo) | decode closure high, interchange closure low, for the same α |
| state interference | per-step incremental closure < 0 under a full prefix-wide patch |
| representation–oracle mismatch | model behaviorally near-optimal while affine ρ→b R² low and decode k\* ≫ k_B |
| variance dependence (of a proposal map) | proposal succeeds in natural coordinates, fails under registered adversarial T, while acceptance verdicts stay correct |
| pullback off-manifold amplification | a "random control" patch closes ≪ 0 under an ill-conditioned pullback |

`battery.py` exposes these as helpers so live scripts emit machine-checkable
verdicts; this table is the specification.

### 6.1 Registration checklist for verdict predicates

The recurring registration defect is predicates that do not partition the outcome
space. Rules for future registrations (the rule numbers are a stable citation
anchor — code and the review protocol reference "§6.1 rule N"):

1. **Partition or label.** The verdict conditions for a prediction (and the
   branches of any adjudication rule) must be exhaustive and mutually exclusive
   over run-dependent outcomes, with **NOT TESTED** as the explicit residual —
   never a silent pass or silent fail.
2. **Quantifiers over run-dependent sets are three-way.** "For all accepted k…"
   must specify the verdict when the set is empty.
3. **Subspace claims carry dimension-parity checks.** Containment and
   principal-angle tests silently weaken when dimensions differ; require the
   dimension explicitly.
4. **Superlatives name their metric.** "Best candidate" is ill-formed when
   candidates are ranked along several axes; write "nearest-to-plane" or "best
   measured gain."
5. **Adjudication branches must not overlap.** If two branches can fire on the
   same run, the boundary was never registered.
6. **Audit the registration against this list before the first run**, and record
   any post-run discovery of a violation as a wording defect in the results —
   resolved on the registered *intent*, with the ambiguity on the record, never
   silently.
7. **Mechanism attributions in conclusions carry one of three labels** —
   *measured*, *consistent-with*, or *hypothesis* — and a consistent-with claim
   names the diagnostic that would settle it.
8. **Registration-to-code coverage is audited symmetrically**: every "per X" /
   "for each X" in the registration must be exercised for every X by the code,
   and every selection rule the code applies must be registered.
9. **Opposite-direction failures get separate branches; recalibrate ≠ fail.**
   When a tolerance/band can miss in two *opposite* ways (over- vs
   under-sensitive ρ; observable over- vs under-stating exact), those are
   distinct outcomes and must not collapse into one branch — and a per-process
   *recalibration* (widening a band on a clean envelope) is a **non-fail** state
   distinct from a *failure* (an inversion / over-trust no band can excuse).
   Recalibration is **directional**: widen only the safe side, hold the dangerous
   side. New live scripts use `battery.directional_tolerance_partition()` for
   this signed pass/recalibrate/fail split.

## 7. Assumption ledger

Moved to `ASSUMPTIONS.md` (the cross-experiment status register for global bets
and scope debts).

## 8. Standing claims and milestones

Moved. The standing equivalence-class finding and the linear-interpreter scope
note live in `ASSUMPTIONS.md`; the plain-language synthesis of where the program
stands is `SYNTHESIS.md`; superseded milestones are in git.
