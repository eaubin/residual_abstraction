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

### 6.1 Registration checklist for verdict predicates

Every defect external review has caught in registrations 8–10 was of one
kind: predicates that do not partition the outcome space. Rules for future
registrations, each with its precedent:

1. **Partition or label.** The verdict conditions for a prediction (and
   the branches of any adjudication rule) must be exhaustive and mutually
   exclusive over run-dependent outcomes, with **NOT TESTED** as the
   explicit residual — never a silent pass or silent fail. (Exp 8 P4:
   0-vs-0 agreement would have passed vacuously; Exp 9 P5: an empty
   accepted-k set failed vacuously — the same defect in both polarities.)
2. **Quantifiers over run-dependent sets are three-way.** "For all
   accepted k…" must specify what the verdict is when the set is empty.
   (Exp 9 P5.)
3. **Subspace claims carry dimension-parity checks.** Containment and
   principal-angle tests silently weaken when the dimensions differ;
   require the dimension explicitly. (Exp 8 P3: k\* > 2 containment
   loophole; Exp 10 P5: k\* = 1 cannot contain a plane.)
4. **Superlatives name their metric.** "Best candidate" is ill-formed
   when candidates are ranked along several axes; write
   "nearest-to-plane" or "best measured gain." (Exp 10 conclusion.)
5. **Adjudication branches must not overlap.** If two branches can fire
   on the same run, the boundary was never registered. (Exp 10
   trichotomy: outcome 2a's letter was satisfied by every 2b instance.)
6. **Audit the registration against this list before the first run**,
   and record any post-run discovery of a violation as a wording defect
   in the results — resolved on the registered *intent*, with the
   ambiguity on the record, never silently.

## 7. Assumption ledger

Constructions carry assumptions that procedures' honesty constraints do
not cover. The patch family's Euclidean assumption went unstated from
Experiment 3 until Experiment 10 falsified it — five experiments in which
"the obvious construction" was silently load-bearing. Each entry: the
construction, the assumption it leans on, and its current status.

| construction | implicit assumption | status |
|---|---|---|
| interchange patch = orthogonal projector ("minimal-norm edit", exp 3) | Euclidean metric of the working coordinates is meaningful — the patch's read covector equals its write direction | **falsified** for ill-conditioned coordinates (exp 10: read side junk-amplified ×κ; needs contamination ≲ κ⁻²); benign coordinates masked it for 7 experiments |
| per-position centering before PCA/PLS (exp 1 revision) | process stationarity; position content is completion-irrelevant | holds on these processes; would need restating for non-stationary data |
| pairing protocol (random same-position pairs) | the delta distribution is representative of behaviorally relevant contrasts; unweighted delta second moment ≈ 2Σ | held; made the delta-ratio miner redundant (exp 9, registered note) |
| whitening / precision constructions | sample covariance estimates the population Σ; ridge floor 10⁻¹⁰λ_max does not bite | held at κ ≤ 1000 (exp 9 invariance probe exact); restate at larger κ or smaller samples |
| observable scoring KL(q_src-run ‖ q_patched) | the model's own run is an adequate stand-in for the true completion kernel | held wherever testable (exps 6, 7: ≤ 6 points); **never yet tested on an accepted adversarial patch** (P4 chain, three experiments running) |
| validity-gate estimator | NLL estimator noise ≪ 0.005 threshold | violated at 400 sequences (exp 5 selftest caught it), fixed to 2000 token-weighted |
| optimal-NLL probe in train.py | 400-sequence filter estimate ≈ entropy rate | known-noisy (negative "gaps" on Z1R, dyck2); gate uses its own estimator, so benign — documented, not fixed |
| exact-chain evaluation | scores are deterministic given pair sets; no estimation noise inside selection | holds; selection-on-discovery overfitting still bounded only by disjoint evaluation (exp 10 P4-style gap is the measurement) |

Rule going forward: a new construction (patch family, pairing scheme,
estimator, composition rule) enters a registration together with its
ledger row — the assumption named, and the condition under which it would
be falsified.
