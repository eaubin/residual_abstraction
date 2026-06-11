# Formalism — the named objects, what each experiment estimated, and the invariance principle

Consolidation of definitions that were previously distributed across README
prose and script docstrings. This file introduces **no new claims**; it
names what the experiments already measure, so that future registrations
can say "this experiment estimates quantity Q under conditions C" instead
of re-deriving the setting. Adopted after external review of Experiments
1–8 (the three-map separation and the invariance principle below were made
*empirical* by Experiment 8 before they were written down here — the
formalism is downstream of the findings, as it should be).

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
**model-state sufficiency** (does α(ρ) carry what *the model's own
computation* reads/propagates? judged by intervention). They coincide on
Mess3 and dissociate on Dyck (representation–oracle mismatch).

## 2. What each experiment estimated

| exp | named quantity | conditions index |
|---|---|---|
| 1 | decode-completeness curve KL(k) and its elbow; identification R² (affine, k-NN) | proposal family = PCA; V = fitted heads; (ℓ = final, m, τ) |
| 2 | k\*(τ) staircase per proposal family; identification at k\*; displaced-variance audit | families = {pca, X-whitened PLS, head-rows}; stopping rule declared |
| 3 | interchange closure of α at the readout: (gap − transfer)/(gap − floor) | patch point = final; V = model's own decoder; m = 1 |
| 4 | per-step incremental closure (persistence); state-coherence fraction | patch point = mid-stream; scopes pos/pre; m = 1..3 |
| 5 | the depth profile of 4: incremental closure and coherence per ℓ | all interior ℓ; full prefix-wide patch |
| 6 | closure of a *discovered* α (CEGAR, behavioral acceptance); observable-vs-exact agreement | discovery objective = observable model-vs-model KL; controls at matched k |
| 7 | all of 1–6 ported to a new P (Dyck-2); k_B (belief intrinsic dim); the oracle/model-state dissociation | registered rules k_B, ℓ† |
| 8 | the same quantities under an adversarial reparameterization z = T·ρ | T registered; discovery sees only z; patches pulled back |

## 3. Orders on abstractions

- **Simulation preorder**: α ⪯ β iff α = h∘β for some allowed h (here:
  linear maps) — β refines α. Nested rank-k families are chains in this
  order; cross-family comparisons generally are not comparable, which is
  why the scalarizations below carry the comparison weight.
- **Behavioral risk preorder**: α ⪯_V β iff the optimal V-interpreter risk
  of β is ≤ that of α for every horizon/divergence in scope, over the
  evaluation distribution. Decode closure and interchange closure are
  scalarizations of this order, always indexed by (patch point ℓ, scope,
  horizon m, distribution μ, interpreter class V). An unindexed
  completeness claim is ill-formed in this house.

## 4. The three maps, kept separate

1. **Proposal map**: data → candidate α (PCA, X-whitened PLS, head-rows,
   the CEGAR miner). A *heuristic*.
2. **Abstract value**: the α itself — a subspace/patch, evaluable
   independently of how it was found.
3. **Interpreter**: what extracts predictions from α(ρ) — fitted probes
   (V-information) or the model's own downstream computation (causal).

Findings sorted by which map failed: interpreter incompleteness (exp 1,
Z1R: k-NN succeeds where affine heads fail); proposal misalignment /
echoes (exps 2, 3, 6, 7: decode-supervised proposals find causally weak
copies); proposal variance-dependence with *sound acceptance* (exp 8).
Moral: a claim about an abstraction is only as coordinate-robust as the
weakest map it depends on, and the three must be audited separately.

## 5. Invariance

**Principle.** A claim about the residual stream should state how it
behaves under invertible linear reparameterization z = T·ρ. Quantities
that are invariant by construction: the closure of a *given patch* (the
pullback T⁻¹QQᵀT / row-form TQQᵀT⁻¹ construction of Experiment 8 makes the
behavioral scoring path coordinate-free), and hence every acceptance
verdict. Quantities that are not: anything ranked by variance or raw
covariance — PCA orderings, unwhitened PLS, the covariance CEGAR miner,
displaced-variance audits. Experiment 8 is the empirical instance:
acceptance survived T, the proposal miner did not. Coordinate-dependent
procedures are admissible only as heuristics whose outputs are validated
by invariant scores; where an invariant procedure exists, prefer it.

**Proposition (whitened mining is GL(d)-invariant, ridgeless).** Let Σ be
the (population) covariance of the stream at the patch point and let the
miner operate on Σ^{-1/2}-whitened coordinates (weighted second-moment
eigenvector of whitened prefix differences), with the interchange patch
defined as the coordinate swap in whitened coordinates (raw-space patch
P = Σ^{-1/2}UUᵀΣ^{1/2} in row convention). Then for any invertible T, the
miner applied to z = ρ·T with Σ_z = TᵀΣT produces the *same* raw-space
patch. *Sketch*: the whitened versions of ρ and z differ by the orthogonal
map O = Σ_z^{-1/2}TᵀΣ^{1/2}; weighted second moments transform by
conjugation with O, eigenvectors map by O, and the induced patches pull
back identically. Finite-sample and ridge-floor effects break this only at
the corresponding tolerance — which is what Experiment 9 registers and
measures.

## 6. Verdict predicates (the failure taxonomy, as checkable conditions)

| verdict | checkable condition | observed |
|---|---|---|
| domain coarseness | conflated pairs (abstract-space neighbors) with sym-KL(γ) > τ | exp 1 (Mess3 k=1) |
| interpreter incompleteness | head KL > τ while k-NN KL ≤ τ on the same α | exp 1 (Z1R k=1) |
| proposal misalignment | identification R² high from full ρ, low from α's subspace | exps 1–2 (Mess3 PCA) |
| metric junk-domination | counterexamples persist under refinement while decode KL ≤ τ | exp 1 revision notes |
| vacuous tolerance | the KL0 baseline already passes τ | guarded since exp 1 |
| correlational-but-not-causal (echo) | decode closure high, interchange closure low, for the same α | exps 3, 6, 7 (PLS) |
| state interference | per-step incremental closure < 0 under a full prefix-wide patch | exp 5 (Mess3 L3); absent on Dyck |
| representation–oracle mismatch | model behaviorally near-optimal while affine ρ→b R² low and decode k\* ≫ k_B | exp 7 |
| variance dependence (of a proposal map) | proposal succeeds in natural coordinates, fails under registered adversarial T, while acceptance verdicts stay correct | exp 8 |
| pullback off-manifold amplification | a "random control" patch closes ≪ 0 under an ill-conditioned pullback | exp 8 (rand-z −427%) |

A future refactor may encode these as predicates in a shared module so
scripts emit machine-checkable verdicts; until then this table is the
specification.
