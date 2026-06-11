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
| 17 | T-genericity of the adversarial core (per-draw battery: accept-count staircase, pool angle, id/clean/spectral contrasts, gradient phenotypes, D2) and eps staircases k\*(eps) / accept(eps) | units = (junk_seed, κ): draws 0–4 at κ = 100, draw 1 at κ ∈ {30, 300} secondary; shared pair sets isolate T; draw 0 = reproduction tripwire |

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

## 7. Assumption ledger — global bets and scope debts

Constructions carry assumptions that procedures' honesty constraints do
not cover; the patch family's Euclidean assumption went unstated for five
experiments before exp 10 falsified it. This ledger keeps the project's
hidden load-bearing ideas visible.

**Structure (adopted at exp 16, on review).** The ledger holds only
*global* items — bets and scope debts that span experiments.
Experiment-local assumptions (thresholds, split granularities, estimator
details) live in their registration under a one-line-per-item **"Scope &
local assumptions"** section, and are promoted here only when reused or
cross-experiment load-bearing. Statuses use a controlled vocabulary —
**open / supported / falsified / scoped / deprecated / under test
(exp N)** — one bolded word, then at most a sentence and a pointer; the
detail lives in the writeups. Rows are updated in place; history lives in
git, and resolving a question means updating *every* row that mentions
it, not just the newest.

### Live bets and debts

| assumption / bet | why it matters | status |
|---|---|---|
| observable model-vs-model scoring is a usable proxy for exact closure | the entire LLM-phase plan rests on it | **supported** — held under gradient selection pressure 4× (exps 13–16, ≤ 2.2 pts; descriptively also at positions the read never saw); residual caveats: scale, distribution |
| linear (rank-1..k) patches are an adequate intervention class for this phase | every closure number is indexed by the patch family | **scoped** — sufficient on Mess3 (clean D2 97.8%); exp 7's decode/control dissociation hints at limits; nonlinear charts are named future work (§8) |
| the clean / T-aware patch can serve as a *trusted reference* (for ρ) | the battery's ρ needs a reference; at LLM scale it must be the best-validated patch, not an oracle | **supported** on toys (exps 15–16); LLM-scale transport **open** |
| gradient access to model weights is observable-legitimate | separates reading the network from reading the oracle | **supported** (exps 13–16); falsified-if a verdict depends on a quantity not computable from (weights, tokens, outputs) |
| pairing protocol (random same-position pairs) represents behaviorally relevant contrasts | all discovery and closure quantities inherit it | **supported** (exp 9: made the delta-ratio miner redundant) |
| per-position centering before second-moment estimation | the stationarity assumption under everything spectral | **scoped** — holds on these processes; restate for non-stationary data |
| disjoint-split discipline bounds selection overfitting (exact-chain scores are deterministic) | discovery/eval — and, since exp 16, /test — separation | **supported** (exp-10-style gaps measured; exp-16 triple split) |
| anchor reproduction (k\* = 2, c_obs ≈ 0.998) | T's construction and every adversarial result ride it | **supported** across 8 consecutive runs; the assert is the tripwire |
| single-T indexing of all adversarial conclusions (exps 8–16) | one junk draw, one κ — T-genericity assumed, never measured | **under test (exp 17)** — 5 draws at κ = 100 + a 2-point κ arm; the oldest debt (since exp 8) |
| fixed-write indexing of read-construction results | write-genericity | **scoped** — partially discharged (exp 16: four writes, landscape result has population evidence); remaining index: one T, one pool family |
| eps_gain = 0.05 tolerance policy | accept/reject claims and k\* are threshold-indexed | **under test (exp 17)** — k\*(eps) and accept-count staircases over {0.01, 0.02, 0.05, 0.10}, benign + adversarial |
| position-locality of learned reads | any learned-read result transports only with its position index | **supported as a limitation** (exps 15–16: entanglement intrinsic in the registered geometry; the clean read is the only position-generic access found) |
| exact-toy adjudication calibrates later oracle-free work | the program's framing bet | **open until battery consolidation** — validated members so far: ρ, shift-retention R, held-out-position gain |

### Settled items (one line each; detail in the writeups)

- Euclidean patch read (read = write): **falsified** (exp 10; mechanism verified exp 11).
- Whitening / precision constructions: **scoped** — held at κ ≤ 1000 (exp 9).
- Fractional-precision read family c ∝ Σ̂^{−α}w: **falsified** as sufficient (exp 12).
- Differentiable-chain objective + torch/numpy regression link: **supported** (exps 13–16).
- Post-step renormalization as the divergence mechanism: **falsified**, measured twice (exp 14); the divergence is a per-write landscape asymmetry — population evidence exp 16 (3 of 4 near-plane writes diverge, nearest-plane worst).
- Affine-slice parameterization: **supported** (sound; exonerated as the divergence cause, exp 14).
- Pooled EPR: **deprecated** (exp 15: aggregation artifact; per-position EPR is the instrument).
- Registered distribution shifts (pair positions; fixed initial state): **decisive** (exp 15; guards passed cleanly; the mild-shift caveat is consumed).
- Position-genericity of gradient-learned reads, including the protocol repairs — held-out checkpoint selection (nothing to select at the registered 20-step granularity) and mixed-position training (position memorization): **falsified** (exps 15–16; interpolation-scope: even interpolation fails). Robust/minimax objectives: open option, not motivated as the immediate follow-up.
- Validity-gate estimator noise: **fixed** at exp 5 (2000 sequences, token-weighted).
- train.py optimal-NLL probe: **documented quirk** (the gate uses its own estimator).

Rule going forward (revised at exp 16): every registration carries a
"Scope & local assumptions" section, one line per item; a new *global*
construction or bet still enters with its ledger row; local items are
promoted only on reuse.

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
  locality; it is intrinsic to behavioral-gradient discovery *in the
  registered geometry* — single T, rank-1 patches, the registered
  optimizer and checkpoint grid, interpolation-only test.)
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
