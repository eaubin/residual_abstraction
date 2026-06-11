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
| 9 | per-miner direction angle + closure gain in both regimes; the GL(d)-invariance proposition checked numerically (κ-fingerprint) | miners = {M1 cov, M2 whitened, M3 centered-weight}; acceptance frozen |
| 10 | per-candidate measured closure gain under patching + plane angle, 12-candidate pools, both regimes | uniform working-coordinate patch convention, judged after pullback |
| 11 | per-(write, read) closure gain; read-junk fraction; oblique composition closure | read menu {id, prec, cov}; conditioning guard 10⁶; same-T as exp 8 |
| 12 | same-write gain across the α-grid c ∝ Σ̂^{−α}w (equivariance × transfer); D2 composition anchor | α ∈ {0, …, 1}; eps_gain frozen 0.05; fixed write pair enters here |
| 13 | full-discovery gain of gradient-learned reads; observable-vs-exact agreement on optimized patches; plane/junk/neutral read decomposition | writes fixed (exp-12 pair); Adam registered; ⟨c,w⟩ = 1 by renormalization |
| 14 | the renorm-divergence feedback signature (instrumented trajectories); affine-slice learned-read gains; learned-composition closure; effective-plane-reading score EPR | same T and writes (deliberately — indexing inherited); ⟨c,w⟩ = 1 by construction; EPR on held-out eval deltas |
| 15 | per-pair equivalence ratio ρ(X) = mean J(clean, X)/mean J(clean, un); per-(t-group, position) EPR cells; relative retention R under two registered distribution shifts | reads = the reproduced exp-14 set; one write (w2) carries the verdicts; shifts guarded (competence, clean-gain ≥ 20%); targets stationary-frame |
| 16 | transportability of protocol-repaired learned reads: held-out-selected (arm A) and mixed-position-trained (arm B) gains on unseen positions; per-write transport verdicts | selection signal = observable c_obs on P_val (honest); ρ/EPR/retention evaluation-only; P_test interior to [8, 24] — interpolation, not extrapolation; T fixed |

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
7. **Mechanism attributions in conclusions carry one of three labels** —
   *measured*, *consistent-with*, or *hypothesis* — and a consistent-with
   claim names the diagnostic that would settle it. (Exp 12: neutral
   contamination presented as localized mechanism, demoted on review;
   exp 13: renormalization feedback presented as proven from per-minibatch
   losses, softened on review. Same defect twice ⇒ rule.)
8. **Registration-to-code coverage is audited symmetrically**: every
   "per X" / "for each X" in the registration must be exercised for every
   X by the code, and every selection rule the code applies must be
   registered. (Exp 13 pre-run review: dual inits registered per write,
   coded for one; stage-B selection coded but unregistered.)

## 7. Assumption ledger

Constructions carry assumptions that procedures' honesty constraints do
not cover. The patch family's Euclidean assumption went unstated from
Experiment 3 until Experiment 10 falsified it — five experiments in which
"the obvious construction" was silently load-bearing. Each entry: the
construction, the assumption it leans on, and its current status.

**Maintenance rule (added after a stale-row incident at the exp-13
conclusion): rows are updated in place — the status column always states
the current best knowledge, and history lives in git, not in the table.**
Appending "(update)" rows next to stale originals defeats the ledger's
purpose as a single current source.

| construction | implicit assumption | status |
|---|---|---|
| interchange patch = orthogonal projector ("minimal-norm edit", exp 3) | Euclidean metric of the working coordinates is meaningful — the patch's read covector equals its write direction | **falsified** for ill-conditioned coordinates (exp 10: read side junk-amplified ×κ; needs contamination ≲ κ⁻²); benign coordinates masked it for 7 experiments. Exp 11: a clean-read rank-1 patch verified causally (+51.3% vs +1.0% on the same write — linear interchange as a primitive is fine); the honest read menu {id, prec, cov} is insufficient (prec eliminates destruction but under-transfers everywhere); honest read *construction* is the open problem |
| per-position centering before PCA/PLS (exp 1 revision) | process stationarity; position content is completion-irrelevant | holds on these processes; would need restating for non-stationary data |
| pairing protocol (random same-position pairs) | the delta distribution is representative of behaviorally relevant contrasts; unweighted delta second moment ≈ 2Σ | held; made the delta-ratio miner redundant (exp 9, registered note) |
| whitening / precision constructions | sample covariance estimates the population Σ; ridge floor 10⁻¹⁰λ_max does not bite | held at κ ≤ 1000 (exp 9 invariance probe exact); restate at larger κ or smaller samples |
| observable scoring KL(q_src-run ‖ q_patched) | the model's own run is an adequate stand-in for the true completion kernel | **held under maximal selection pressure, four times** (exps 13–16 P4; ≤ 2.2 points on gradient-optimized adversarial patches; descriptively faithful across a ~550-point outcome range, and at positions the read was never trained or selected on — exp 16, sub-threshold cell, 2.3 points). Remaining caveats are scale and distribution, not concept |
| validity-gate estimator | NLL estimator noise ≪ 0.005 threshold | violated at 400 sequences (exp 5 selftest caught it), fixed to 2000 token-weighted |
| optimal-NLL probe in train.py | 400-sequence filter estimate ≈ entropy rate | known-noisy (negative "gaps" on Z1R, dyck2); gate uses its own estimator, so benign — documented, not fixed |
| exact-chain evaluation | scores are deterministic given pair sets; no estimation noise inside selection | holds; selection-on-discovery overfitting still bounded only by disjoint evaluation (exp 10 P4-style gap is the measurement) |
| fractional-precision reads c ∝ Σ̂^{−α}w (exp 12) | floored eigendecomposition (10⁻¹⁰·λ_max) of the *sample* covariance; and the registered impossibility note — no α equals the clean read unless the x-spectrum is flat on the relevant directions | **falsified as a sufficient family** (exp 12: adversarial gains flat ~+1.5% across the whole grid; low read-junk demonstrably does not buy transfer — 4–7% junk on the nearest write with no gain, while most near-plane writes shed junk only at α = 1; prec confirmed behaviorally equivariant, +1.4% = +1.4% same-write). *Leading hypothesis* for the obstacle: neutral-background read contamination — inferred, not yet decomposed (the plane/junk/neutral read decomposition is the exp-13 registered diagnostic). Clean-read *composition* exonerated (D2: 97.8%); the open object is a non-spectral read construction |
| differentiable-chain objective (exp 13) | minibatch gradients are unbiased estimates of the full-pair objective; float32 backprop through the 4-layer model is adequate. Verdict-protection: final scoring is always the full-pair non-differentiable evaluator, and the two code paths are asserted to agree (rel 10⁻⁴) before any optimization | held (regression link rel 1.7×10⁻⁸; w2 optimizations converged cleanly) |
| ⟨c,w⟩ = 1 by post-step renormalization (exp 13) | renormalization after each Adam step preserves descent | renormalization-feedback mechanism **refuted, measured twice** (exp 14): pre-renorm ⟨c,w⟩ median 1.0008/1.0012 — renormalization nearly inactive during the divergence — and the no-renorm affine parameterization diverges identically (−500.5%/−548.2%). The w1 divergence is a per-write *landscape* asymmetry (good basin escaped in ≤ 20 steps for the trajectory-logged renormalized runs, by step 50 for the affine runs — their logging floor; w2 descends from a −187% init to +42.5% through identical machinery). Current *hypothesis*: Adam per-coordinate steps vs κ-sharpened junk curvature; settling diagnostic: lr/optimizer sweep |
| gradient access to model weights (exp 13) | backprop through given weights is observable-legitimate (a reading of the network, like exp 3's unemb pullback). Falsified-if: a verdict ever depends on a quantity not computable from (weights, tokens, model outputs) | registered |
| single-T indexing (exps 8–13) | every adversarial result is indexed by ONE registered transform: one junk-plane draw (seed 0), κ = 100; generalization across draws/κ is assumed, not measured (exp 9's κ-sweep covered M2 only) | **untracked until now** (coverage-audit addition); a T-robustness sweep is owed before adversarial conclusions are treated as T-generic |
| anchor reproduction (exps 8–13) | the exp-6 loop reproduces deterministically (k\*=2, c_obs within 0.005 of 0.998) on every run — T's construction and all downstream results depend on it | held across six consecutive runs (asserted each time); fragile to library/numerics changes — the assert is the tripwire |
| fixed write-pair indexing (exps 12–16) | read-construction results are indexed by registered near-plane writes from seeded pool draws; write-generality is assumed, not measured | **partially discharged** (exp 16 widened to four writes: the landscape result gained population evidence — 3 of 4 diverge, nearest-plane worst — and the transport result is 0 of 4). Remaining index: one T, one pool family |
| eps_gain = 0.05 tolerance policy | the acceptance threshold, fixed at exp 6 for a different proposal regime, is treated as regime-independent; its sensitivity is unmeasured (the exp-11 +4.5% episode sat 0.5 pts under it) | **untracked until now**; per §1 conventions a staircase over eps_gain would index the affected conclusions properly |
| affine-slice read parameterization c = c₀ + (I − ŵŵᵀ)u (exp 14) | the constraint slice is searched in u-coordinates; Adam's per-coordinate scaling differs from exp-13's c-space, so endpoints are comparable across the two parameterizations but trajectories are not; assumes the slice (not the renormalized ray) is the right search space | behaves where exp 13 behaved (benign +52.2% ≥ id +51.3%; w2 +32.2%/+42.5% vs exp-13's +28.6%/+43.7%) and diverges where exp 13 diverged — **the parameterization is exonerated as the divergence cause**; the construction is sound but repairs nothing |
| effective-plane-reading score EPR = corr²(Δ·r, Δ·u_clean) (exp 14) | the clean-read functional on pooled held-out eval deltas is the right operationalization of "what the read computes"; on-distribution evidence only — high EPR does not promise off-distribution transfer | **deprecated — the pooled score is misleading by construction** (exp 14's refutation branch fired at pooled EPR 0.008/0.007, but exp 15 P3a resolved it as an aggregation artifact: per-position correlations are real and pooling sign-cancels them; the "deep reading" — closure gain without the clean functional — is closed). Use the per-cell EPR row below |
| per-pair equivalence ratio ρ(X) = mean J(C,X)/mean J(C,un) (exp 15) | Jeffreys divergence is the right symmetric comparison; the do-nothing distance is the right scale unit; the reference patch C is *trusted*, not oracular (here the T-aware clean patch — at LLM scale the reference would be the best-validated patch, so the construction transports) | **validated as a separator** (exp 15 P6: destructive 5.3/13.8 vs accepted ≤ 0.44, > 10×; descriptively monotone with transfer across the menu). Caveat: mean-level — frac_worse 32% even at ρ = 0.20. Oracle-free given a trusted reference → self-certification battery member |
| per-cell EPR (t-group × absolute position) (exp 15) | pooling by absolute position is the right disaggregation of exp-14's pooled score; ~200-row cells put the null corr² ≈ 0.005, far under the 0.2 threshold | **resolved the exp-14 puzzle** (exp 15 P3a): the pooled refutation was an aggregation artifact — both accepted reads' position-t cells above the 0.5 threshold (aff/id 0.85–0.93 throughout; best-α mixed, 0.83/0.53/0.91; id baseline 0.36–0.59; early-position cells trivially ≈ 0.99 for any read). Per-position EPR is the instrument; the pooled score is deprecated |
| registered distribution shifts (exp 15: positions {12, 20}; fixed initial state 0) | per-prefix targets stay in the stationary belief frame (the trained model's frame), so shifts move only the distribution over prefixes and the clean patch's meaning is unchanged; guards (model-vs-exact NLL gap ≤ 0.01 on shifted data; clean shifted gain ≥ 20%) make a too-destructive shift NOT TESTED rather than misread. Scope: these are *mild* shifts — robustness here does not establish shift-immunity; fragility here is decisive | **decisive** (exp 15: guards passed cleanly — gap +0.0001, clean 56.8%/46.9%; shift-A exposed position entanglement, learned reads inverting to R = −0.77/−0.41 while clean improved; shift-B robust 0.89/0.97). The mild-shift caveat is consumed: fragility was found. Shift-retention R → battery member |
| gradient-learned reads' position indexing (exps 13–16) | optimizing the pooled CE over a discovery position set yields a position-generic read | **falsified, and the protocol repairs falsified too** (exp 15: trained-position EPR high, inverts at unseen positions; exp 16: held-out checkpoint selection finds nothing to select, mixed-position training memorizes positions). The entanglement is intrinsic to behavioral-gradient discovery in this geometry; the clean-plane read is the only position-generic access found; the failure mode is detectable oracle-free (ρ, R, held-out-position gain). Read-construction thread concluded — four repair families falsified (exps 11, 12, 13–14, 16) |
| checkpoint selection / triple split (exp 16) | 20-step checkpoint granularity suffices; selection on P_val makes the selection set a fit target — bounded by the unseen P_test, the same discipline as discovery/eval splits | **resolved — nothing to select** (exp 16: no transportable checkpoint exists on any of 4 trajectories; the converging write's val gain is never positive, the rest diverge by step 20; granularity adequate, trajectories smooth at this scale) |
| mixed-position training (exp 16) | position diversity in minibatches is the operative variable (total optimizer budget held fixed); judged only on P_test | **falsified as a repair — position memorization** (exp 16: positive gain at every trained position, +27.7% on val ⊂ train, vs +3.0% at interpolated unseen positions; diversity widens the memorized support, does not produce the position-generic functional). Also pre-empts minimax-over-positions: coverage balance is not the failure, interpolation is |
| position-interpolation scope (exp 16) | P_test = {10, 14, 22} is interior to the training range [8, 24] (protocol margins forbid going outside) — transport verdicts are *interpolation* claims; extrapolation is out of reach in this setting | registered; exp 16 sharpened it — even *interpolation* fails for learned reads, so the scope caveat understates the locality |

Rule going forward: a new construction (patch family, pairing scheme,
estimator, composition rule) enters a registration together with its
ledger row — the assumption named, and the condition under which it would
be falsified.

## 8. Standing claims and named milestones

(This section replaced a point-by-point response to a 2026-06 external
program review: reactive content that dates quickly does not belong here —
only what binds future work. History in git.)

- **The equivalence-class claim** — adjudicated for its first instance
  (exp 15): equivalence exists but is **distribution-local**. The best
  learned read is mean-equivalent to the clean patch on the discovery
  position set (ρ = 0.20), yet both learned reads *invert* at unseen
  positions while the clean patch improves — patches are not
  interchangeable access to transported state. The plane remains the
  content and the only position-transportable access found; any future
  use of the claim carries its position index. (Exp 16: protocol repairs
  — held-out selection, mixed-position training — do not remove the
  locality; it is intrinsic to behavioral-gradient discovery here.)
- **Named milestones.** The generality / de-localization sweep — T draws,
  κ, write pool, eps_gain staircase (reported as k\*(tolerance) curves,
  not points), m-staircase — discharging the §7 index debts. Then the
  **self-certification battery**: oracle-free consistency signals
  validated against exact closure on the exps-10–15 patch zoo; the gate
  to any LLM-scale phase, where no exact adjudicator exists. Validated
  members: the equivalence ratio ρ (given a trusted reference patch),
  shift-retention R (exp 15), and held-out-position gain (exp 16) —
  together they caught and characterized the statistical-control failure
  mode with no oracle access.
- **Scope.** Abstractions here are linear by registered interpreter class
  (§1, §4) — a parameter of every claim, not a theory commitment;
  nonlinear charts (Dyck as the venue, after exp 7's
  representation–oracle mismatch) are named future work.
