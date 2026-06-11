# Experiment 13 — fixed-write read optimization (the read-only gradient) — PRE-REGISTRATION

**Script:** `readopt.py`. **Status: pre-registered; NOT YET RUN — results
to be appended below the marked line.**

**Question.** Experiments 10–12 cornered the adversarial failure to one
object: the read covector. Writes are reachable from cheap pools (exp 10),
the (write, read) patch object is right and linear rank-1 interchange works
with a clean read (exp 11: +51.3% vs +1.0%, write fixed), clean-read
composition reaches the ceiling (exp 12 D2: 97.8%), and no Σ̂-spectral read
suffices (exp 12: flat ~+1.5% across the α-grid). Can the read covector be
**learned from behavioral signal** — gradient descent on the observable
closure objective, write held fixed?

**Provenance and ordering.** Per external review, the read-only gradient
goes before paired read/write pools: pools are expressive but would not
say whether a win came from better reads, better writes, or lucky
coupling; the gradient probes exactly the isolated object. The original
gradient objections were retired in order — selection sound (10), patch
object right (11), composition fine (12) — so a read-only gradient no
longer "optimizes a bad coupling."

**Honesty status of gradients (registered).** Backpropagation through the
model is observable-legitimate: the weights are given, and differentiating
through them is a reading of the network, not of hidden ground truth (the
same category as Experiment 3's unembedding pullback). The objective is
the model-vs-model KL on discovery pairs — fully observable. Beliefs
remain evaluation-only.

**The P4 dual reading (registered before the run).** This is
simultaneously the strongest observable-soundness test the program can
construct: gradient optimization applies *maximal* selection pressure on
the observable objective. If the held-out exact closure agrees with the
discovery-side observable closure on the optimized patch, that is the
strongest oracle-free soundness datum yet (after five consecutive
NOT TESTED verdicts). If they diverge, the divergence is itself a
first-class finding: **objective hacking** — the gradient found an
off-manifold read that mimics the source *run* without transferring causal
content — locating the limit of observable scoring. Neither outcome is a
mere failure.

## Design (all hyperparameters frozen here)

**Setting.** The Experiment-6/8–12 setting unchanged (mess3-L4, L1,
prefix-wide, κ = 100, 400/600 disjoint pairs, basis sample 800, m = 3,
seed 0, anchor + both transform checks). No acceptance loop — this is a
two-stage diagnostic, per review: single-write optimization first,
composition second.

**Writes (fixed, reproduced in-run by the registered exp-12 rule).** The
two nearest-to-plane round-1 adversarial writes from the registered write
pool (deterministic: same seeds) — in the recorded exp-12 run these are
M2\*Sinv at 1.1° and a raw random at 3.3°, exactly the D2 pair, so the
composition ceiling for stage B is already on record (97.8%).

**Optimization (per write, adversarial regime; benign sanity arm for the
pulled-back nearest write).** Read c ∈ R^d in working coordinates;
interchange constraint ⟨c, w⟩ = 1 enforced by renormalization after every
step (guard: abort-and-report if |⟨c, w⟩| collapses below 10⁻⁶ before
renormalization). Patch P = pullback(c·wᵀ), applied prefix-wide as
always. Objective: minimize the mean model-vs-model KL,
KL(q_src-run ‖ q_patched), over a per-step minibatch of discovery pairs —
differentiable through the exact m=3 chain (per-continuation log-prob
sums; probability floor 10⁻¹²; model parameters frozen). **Adam, lr 0.05,
200 steps, minibatch 64 pairs (seeded), torch seed 0.** Two registered
initializations per adversarial write: (a) the write's best-α read from
the Experiment-12 grid, (b) the id read — agreement is a robustness check,
disagreement a registered characterization of local optima. Benign arm
initializes at id.

**Evaluation per optimized read.** Full-discovery c_obs gain (the
optimizer's own scale, no minibatch noise); held-out exact closure of the
rank-1 patch (the verdict scale); and the **read decomposition** — the
stream-space read covector's squared-norm fractions on the causal plane /
the registered junk plane / the neutral remainder — applied to the
initialization, the optimized read, and the clean diagnostic read
(expected plane fraction ≈ cos²(write angle) for the latter; a sanity
anchor for the decomposition itself). This diagnostic turns Experiment
12's neutral-contamination hypothesis from inference into measurement.

**Stage B.** Compose the two adversarially-optimized (c, w) pairs by the
oblique composition; evaluate exact closure against the recorded D2
ceiling. *Selection rule (pre-run clarification from review — the first
draft left it implicit and the code composed best-α inits only):* per
write, the read with the **best final full-discovery gain across its
inits** enters the composition — the same selection P3 and P6b use, so
the headline single-write choice and the composition choice cannot
diverge. (The same review caught that the code ran the dual init only for
w1; both adversarial writes now get both registered inits, as the
registration always said.)

**Controls and ceilings (in-run).** Best-α read for each write (recorded
+1.5%); the clean diagnostic read (T-aware, labeled non-discoverable;
recorded +51.3%); D1/D2 re-evaluated as anchors.

## Pre-registered predictions (NOT TESTED residuals explicit)

- **P1 (anchors).** Exp-6 loop reproduces; both transform checks pass;
  D1 ≥ 40%; D2 exact ≥ 90% of full. Always testable.
- **P2 (benign sanity; ~85% credence).** The benign-optimized read's
  full-discovery gain ≥ the id read's gain − 5 points (the gradient does
  not regress where the Euclidean read is already right). Always testable.
- **P3 (headline; ~55% credence).** The adversarially-optimized read for
  the nearest write reaches full-discovery c_obs gain ≥ 40% (≈ 80% of its
  clean-read ceiling). FAILS includes divergence below the initialization.
- **P4 (observable soundness under maximal pressure).** For every
  optimized rank-1 patch with full-discovery gain ≥ 20%:
  |observable closure − held-out exact closure| ≤ 0.10. **NOT TESTED** if
  no optimized patch reaches 20%. (The dual reading above applies: FAILS
  here = objective hacking located, a first-class result.)
- **P5 (composition).** The composed optimized pair reaches exact closure
  ≥ 90% of full. **NOT TESTED** unless both single-write gains ≥ 20%.
- **P6 (the Experiment-12 hypothesis, measured; two clauses).**
  (a) The best-α initialization's stream read has neutral fraction ≥ 50%
  (~70% credence). (b) If P3 holds, the optimized read's plane fraction
  ≥ 50% (~70% credence); clause (b) **NOT TESTED** if P3 fails. A P3
  success with a *non*-plane-dominant optimized read would falsify the
  neutral-contamination account in an interesting way (the behavioral
  optimum would not be where the geometry says it should be).
- **P7 (validity gate, enforced).** As established.

## Failure modes this can newly exhibit

*Objective hacking* — P4 fails on an accepted-scale patch: the observable
objective is exploitable under gradient pressure; the program's oracle-free
scoring has a measured limit (and the LLM-phase plan needs a robustness
term, e.g. patched-vs-target KL regularization — for the follow-up, not
retrofitted here). *Local-optimum trap* — the two initializations converge
to materially different reads (registered characterization; would motivate
multi-start or annealed α-paths). *Optimization failure* — loss curves
plateau at the initialization: gradient signal through the chain is too
weak at this κ; would push toward paired pools or read parameterizations
with built-in structure. *Decomposition surprise* — P6(b)'s falsification
branch above.

**Ledger rows (FORMALISM §7, added with this registration).**
Differentiable-chain objective: assumes minibatch gradients are unbiased
estimates of the full-pair objective and float32 backprop through a
4-layer model is adequate (loss curves printed; final scoring is always
the full-pair non-differentiable evaluator, so optimizer numerics cannot
contaminate verdicts). Gradient honesty: model-weights access is
observable-legitimate; falsified-if: a verdict ever depends on a quantity
not computable from (weights, tokens, model outputs).

**Self-checks** (every invocation; `--selftest` exits after the standard
four): the standard four; anchor + both transform checks (real runs); the
differentiable chain must reproduce the non-differentiable evaluator —
for a fixed read, the torch-graph objective and `kl_rows`-based objective
must agree to 10⁻⁴ relative (regression link between the two code paths,
asserted before any optimization step).

**Enforcement.** Registered parameters (including all optimizer
hyperparameters), full config, seed 0, gate — as in Experiments 8–12.

---

## Results: P1, P2, P4, P6a, P7 HOLD; P3 FAILS; P5, P6b NOT TESTED — P4's six-experiment wait ends with the strong reading, and the clean-read picture itself is overturned

(Registered parameters, seed 0, gate +0.0024 PASS; anchor, transform
checks, and the torch/numpy objective regression link (rel 1.7×10⁻⁸)
passed. Raw output `out/exp13_mess3-L4.txt`. The UserWarning at the top of
the output is a cosmetic cast in a print statement; it touches no
computation.)

**Finding 1 — P4 holds, at last, with the strong reading.** Two optimized
patches cleared the 20% gain bar (adv/w2/best-α +28.6%, adv/w2/id +43.7%)
and their observable closures track held-out exact closures to **0.3 and
1.6 points** — under gradient optimization, the maximal selection pressure
the observable objective will face in this program. No objective hacking.
After five consecutive NOT TESTED verdicts, oracle-free interventional
scoring is now validated on nontrivial *adversarial, gradient-optimized*
patches. This was the program's single most important open assumption
(ledger §7, observable-scoring row) and it survived its hardest test.

**Finding 2 — the genuine discovery: a working read with ZERO causal-plane
mass.** The w2 (3.3° random write) gradient runs improved honestly from
the spectral baseline (best-α +4.4%) to **+43.7%** — and the learned
read's decomposition is **plane 0% / junk 49% / neutral 51%**. A read
covector with no mass on the causal plane transfers ~85% of what the
T-aware clean read achieves. P6b is formally NOT TESTED (P3 failed), but
this is its registered falsification branch materializing at sub-threshold
scale: the geometric picture — "a good read aligns with the plane" — is
too narrow. New leading hypothesis (inferred; the next registration should
make it measured): the read's job is *statistical, not geometric* — the
optimal read is the best linear predictor of the write-coordinate
difference, and it may legitimately concentrate mass on junk/neutral
directions whose activity is *correlated* with the plane coordinates (the
echo structure exps 3–9 mapped, now load-bearing in the read's favor).
This also retroactively reframes why spectral reads failed: not because
non-spectral geometry was needed, but because the predictor weighting is
data-specific in a way no Σ̂-power realizes.

**Finding 3 — P3 fails via a NEW typed failure: constraint-renormalization
instability.** Both w1 inits *diverged*: batch CE rose monotonically
(3.29 → 3.40) and the final reads are 100%-junk with gains −462%/−498%.
An optimizer ascending its own objective means the registered
constraint-handling — renormalize c ← c/⟨c, w⟩ after every Adam step — is
the defect: when steps shrink ⟨c, w⟩, renormalization rescales the whole
vector, amplifying off-plane components faster than descent reduces them;
a feedback runaway that the w2 geometry happened not to trigger (its loss
fell smoothly). This is a *parameterization* failure, not evidence about
read-learnability — w2 learned fine through the same machinery. The
registered repair candidate for the follow-up (not retrofitted here):
parameterize the affine slice directly, c = c₀ + v with v ⊥ w, making
⟨c, w⟩ = 1 hold by construction with no renormalization step to feed back.

**Finding 4 — P6a holds at 100%: Experiment 12's hypothesis is now a
measurement, for the inits.** The best-α spectral reads are 97–100%
*neutral* (w1: 0/0/100; w2: 0/3/97 plane/junk/neutral) — exactly the
neutral-contamination account, confirmed where it was claimed. The twist
is Finding 2: neutral mass is fatal for *spectral* reads but a learned
read can apparently exploit correlated neutral/junk mass deliberately.

**Remaining verdicts.** P2 holds (benign optimization +51.7% ≥ id's
+51.3% — no regression, stable dynamics). P5 NOT TESTED, correctly gated
by w1's divergence (stage B composed −486.5%, poisoned by w1; the gate
worked as designed). P1 anchors reproduced (D1 +51.3%, D2 97.8%).

**What the next registration inherits.** (1) The affine-slice
parameterization repair, expected to stabilize w1 and unlock P5/P6b.
(2) A *correlational-read diagnostic* to make Finding 2's hypothesis
measured: the covariance between the read functional ⟨s, Δ⟩ and the
plane-coordinates of Δ — "effective plane reading" — alongside the mass
decomposition, for learned, spectral, and clean reads. (3) The P4 result
upgrades the program's LLM-phase posture: behavioral scoring withstood
gradient pressure at toy scale; the remaining caveats are scale and
distribution, not concept.

**Status: CONCLUDED.**
