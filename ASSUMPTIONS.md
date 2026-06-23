# Assumption ledger — global bets and scope debts

The project's hidden load-bearing ideas, kept visible. This is a **status
register**, not a log: it holds only *global* items — bets and scope debts that
span experiments. Experiment-local assumptions (thresholds, split granularities,
estimator details) live in each registration's "Scope & local assumptions"
section and are promoted here only when reused or cross-experiment load-bearing.

**Maintenance discipline.** One row per bet. Status is one bolded word from the
controlled vocabulary — **open / supported / falsified / scoped / deprecated /
under test (exp N)** — then at most a sentence and a pointer. Rows are settled at
**phase consolidations** (`END_OF_PHASE.md`), since a global bet rarely turns on
a single experiment; the only mid-phase edit is a back-annotation when a
conclusion directly revises a row (the propagate-grep safety net). Rows are
updated in place; the run-by-run history lives in the experiment writeups and
git, never in the cell. Resolving a question means updating *every* row that
mentions it.

## Live bets and debts

| assumption / bet | why it matters | status |
|---|---|---|
| observable model-vs-model scoring proxies exact closure | the whole LLM-phase plan rests on it | **supported** on toys (Mess3 6×, transferred to Dyck); residual caveat: scale. exps 13–22 |
| linear rank-1..k patches are an adequate intervention class | every closure number is indexed by the patch family | **scoped — falsified for pstack predicate targets** — sufficient on Mess3 (exp 7 hints at limits); but rank-1 residual oblique writes (same-read and position-conditioned fixed-read) do **not** control the `pstack` `phi1`/`phi2` predicates despite read room (exps 29/30/33), while matched near-manifold deltas move them (specificity un-adjudicable on pstack). Manifold/interchange interventions outrank rank-1 linear writes for these targets; nonlinear/manifold charts now have direct motivation, not just future work. exps 7, 10, 29–35 |
| a trusted reference patch can anchor ρ | the battery's ρ needs an anchor; at scale it must be the best-validated patch, not an oracle | **supported** on toys / **scoped** at no-oracle scale — the anchor is *declared by convention, not uniquely earned* (selection non-unique); the six-member battery still transfers under the declared core (lenient-equivalent-band caveat); LLM-scale selection **open**. exps 15–18, 24–28 |
| gradient access to model weights is observable-legitimate | separates reading the network from reading the oracle | **supported** — falsified-if a verdict depends on a quantity not computable from (weights, tokens, outputs). exps 13–18 |
| pairing protocol (random same-position pairs) represents behaviorally relevant contrasts | all discovery/closure quantities inherit it | **supported**. exp 9 |
| per-position centering before second-moment estimation | the stationarity assumption under everything spectral | **scoped** — holds on these processes; restate for non-stationary data |
| disjoint-split discipline bounds selection overfitting | discovery/eval/test separation | **supported**. exp 10, exp 16 (triple split) |
| anchor reproduction (k\*=2, c_obs ≈ 0.998) | T's construction and every adversarial result ride it | **supported** across 8 runs; the assert is the tripwire |
| T-indexing of adversarial conclusions | junk-draw and κ genericity | **supported (draw) / scoped (κ)** — draw-generic at κ=100; the gradient-thread conclusions are κ-graded (κ30 work, κ300 diverge); the κ\* boundary is unmeasured. exp 17 |
| fixed-write indexing of read-construction results | write-genericity | **scoped** — partially discharged (exp 16, four writes); remaining index: one T, one pool family |
| eps_gain = 0.05 tolerance policy | accept/reject and k\* are threshold-indexed | **supported** — exp 17 grid {0.01–0.10}; the threshold was never load-bearing for the CEGAR claims |
| position-locality of learned reads | learned-read results transport only with their position index | **scoped — κ-graded (Mess3); position-specific (pstack)** — entangled at κ=100, transports at κ=30 (Mess3, exps 15–17). On pstack the predicate *read* is genuinely position-specific: readable in place at held-out positions yet with a near-different direction (no single-position read carries; a milder shared subspace not excluded). exps 15–17, 31–32 |
| exact-toy adjudication calibrates later oracle-free work | the program's framing bet | **scoped-supported — consolidated in BATTERY.md** — Mess3 calibrated the battery, Dyck transferred all six members; open half is scale, where oracle-free *unique* selection returned a typed negative (declared-not-earned anchor) yet the battery still transfers under the declared core. exps 18, 22, 24–28 |
| m = 3 standing horizon | every claim is indexed by the completion horizon — the semantic target γ_m, not a nuisance parameter | **supported, scoped** — horizon-stable over mm ∈ {1–4} for tested regimes; historical arms keep their m=3 index. exp 18 |
| pstack `phi1`/`phi2` are facets of one coupled stack-state variable | decides whether exp-34/35 "non-specificity" is correct identification of one variable or broad replacement; a backward-design input for the next toy's predicate inventory | **open — valence uncertain** — causally coupled on pstack (directed near-manifold delta co-moves both, 4/4; cannot separate *one variable* from *broad replacement*, no high-room out-of-bundle control). exp 40 reproduces an *analogous*, now **asymmetric and depth-graded** coupling on a separable-by-construction Dyck-2 toy — consistent with representational (not a `pstack` resolution; different process); causal-vs-geometric (non-orthogonal directions) still open. exps 34–36, 40 |

## Settled items (one line each; detail in the writeups)

- Euclidean patch read (read = write): **falsified** (exp 10; mechanism verified exp 11).
- Whitening / precision constructions: **scoped** — held at κ ≤ 1000 (exp 9).
- Fractional-precision read family c ∝ Σ̂^{−α}w: **falsified** as sufficient (exp 12).
- Differentiable-chain objective + torch/numpy regression link: **supported** (exps 13–16).
- Post-step renormalization as the divergence mechanism: **falsified**, measured twice (exp 14); the divergence is a per-write landscape asymmetry (population evidence exp 16).
- Affine-slice parameterization: **supported** (exonerated as the divergence cause, exp 14).
- Pooled EPR: **deprecated** (exp 15: aggregation artifact; per-position EPR is the instrument).
- Registered distribution shifts (pair positions; fixed initial state): **decisive** (exp 15).
- Position-genericity of gradient-learned reads, including protocol repairs: **falsified at κ=100** (exps 15–16); transported at κ=30 (exp 17 — see the κ-graded live row).
- Validity-gate estimator noise: **fixed** at exp 5 (2000 sequences, token-weighted).
- train.py optimal-NLL probe: **documented quirk** (the gate uses its own estimator).

## Standing finding (the equivalence-class claim)

Equivalence between a learned read and the clean patch exists but is
**distribution-local** and **κ-graded**: at κ=100 the best learned read is
mean-equivalent on the discovery positions yet both learned reads *invert* at
unseen positions while the clean patch improves — patches are not interchangeable
access to transported state. Positive (transporting) instances appear at κ=30
(exp 17). The plane remains the only position-transportable access found; any use
of the claim carries its position index. (exps 15–17.)

**Scope.** Abstractions here are linear by registered interpreter class
(FORMALISM §1, §4) — a parameter of every claim, not a theory commitment.
