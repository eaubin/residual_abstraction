# Experiment 6 — interventional discovery: CEGAR on interchange closure — PRE-REGISTRATION

**Script:** `discover.py`. **Status: pre-registered; NOT YET RUN — results
to be appended below the marked line.**

**Question.** Experiments 3–5 established that every completions-supervised
proposal family tested finds *echoes* — decode-sufficient subspaces that are
causally weak (X-whitened PLS at L1: 3.3% closure in a stream whose full
patch closes 98.7%). Can a CEGAR loop whose *scoring is interventional* —
proposals tested by interchange closure at L1, counterexamples mined from
behaviorally-failed pairs — discover a low-dimensional causal abstraction
that approaches full-space patch closure, while avoiding the echo failure?

## Design

**Model and patch point.** The Experiment-5 model (`out/mess3-L4`, Mess3,
4 layers, validity-gated) at **L1** (input to block 2) — the depth where the
stream is state (Experiment 5: full-patch step-2/3 incremental closure
93.7%/91.0%), so the discovered abstraction is tested against the strongest
known causal target. Prefix-wide scope; the Experiment-4/5 evaluation
protocol carried forward unchanged (three positions t = 8/16/24 pooled,
600 evaluation pairs, exact m = 1..3 chains, per-position stability table).

**Honesty constraint (sharpened for discovery).** The loop's objective is
**fully observable**: behavioral divergence KL(q_source-run ‖ q_patched)
between *model* distributions over the m=3 chain — the score that exists on
a real LLM with no oracle. Exact belief-conditioned closures (as in
Experiments 3–5) are computed only at evaluation, on a disjoint pair sample.
Discovery pairs (n=400, own seed) and evaluation pairs (n=600, the
Experiment-5 seed, for comparability) never mix.

**The CEGAR loop (all parameters fixed here).**
- *State*: an orthonormal basis Q, starting empty (k=0 ⇒ unpatched).
- *Counterexample mining*: per-pair weights w_i = current observable
  divergence KL(q_src-run,i ‖ q_patched,i) — the pairs the current
  abstraction fails on, weighted by how badly.
- *Propose*: the top eigenvector of the w-weighted second moment of the
  **unexplained** prefix differences Δ⊥ = (I − QQᵀ)(r_src − r_tgt),
  accumulated over all prefix positions p ≤ t of all discovery pairs.
- *Test*: interchange transfer of the (k+1)-dim patch on the discovery
  pairs; observable closure c_obs = (D₀ − D)/(D₀ − D_full) where D₀ =
  unpatched divergence and D_full = full-space-patch divergence.
- *Accept / stop*: accept only if the marginal gain in c_obs is ≥
  **eps_gain = 0.05** (five points of observable closure per dimension);
  otherwise stop. Hard cap **k_max = 8**; *reaching k_max is reported as
  NON-CONVERGENT, a failure, never as success*.
- *Coarsen (junk guard)*: after stopping, iteratively drop any direction
  whose removal costs < **eps_drop = 0.01** of c_obs. k\* is the size after
  coarsening.

**The anti-triviality guards (the registered answer to "just grow k until
it's full-space").** Three independent mechanisms, all fixed pre-run:
(1) the marginal-gain rule makes dimension k+1 unacceptable unless it earns
five points of observable closure — as the remaining gap shrinks below five
points the loop *must* stop; (2) k_max = 8 ≪ d = 64, and hitting it is a
typed failure; (3) the success criterion P1 jointly requires small k\* AND
high closure — a large-k\* solution cannot satisfy it regardless of
closure. The full nested curve closure(k) for k = 1..k\* is reported
(staircase discipline), so a reader can move the threshold.

**Controls at matched k\*** (evaluation, exact-target closures):
- `pca` k\* and `pls` k\* — discovered at L1 on a separate
  discovery-sequence sample (the Experiment-5 protocol);
- `rand` k\* — no-information control;
- `full` — the known ceiling (Experiment 5: 98.7% pooled m=3 at L1);
- `emb` — the fixed architecture basis available at L1: the orthonormalized
  span of the token-embedding rows (≤ V = 3 dims, no fitting; the analogue
  of Experiment 3's unemb pullback at the input side). Characterization
  control, no threshold: it should carry token-identity content but not
  block-1's aggregated history.
- Principal angles between the discovered subspace and pca / pls / emb
  subspaces are reported as characterization (alignment with the variance
  basis is an allowed empirical outcome, not a target).

## Pre-registered predictions (thresholds fixed before the first run)

- **P1 (the headline, jointly anti-trivial).** The discovered abstraction
  reaches ≥ 90% of the full-space patch's pooled m=3 exact-target closure
  (i.e. ≥ ~0.89 absolute) at **k\* ≤ 4**.
- **P2 (echo avoidance).** Discovered closure at k\* exceeds the PLS k\*
  closure by ≥ 50 points (pooled m=3).
- **P3 (non-inferiority to the variance basis).** Discovered closure at k\*
  is within 2 points of PCA at matched k (PCA is right on this model
  because the causal content is high-variance; discovery must not lose to
  variance luck — and unlike PCA, its objective would survive a
  buried-belief regime).
- **P4 (control).** rand k\* closure ≤ 25% at every horizon.
- **P5 (validity gate, enforced in code).** Same gate and enforcement as
  Experiment 5 (token-weighted NLL on 2000 fresh sequences, gap ≤ 0.005;
  exit on failure; registered config mess3/4-layer/L1 required unless
  --selftest / --force-invalid).
- **P6 (convergence).** The loop terminates by its marginal-gain rule at
  k\* ≤ 4 (it does not hit k_max). Hitting k_max ⇒ typed NON-CONVERGENT.
- **P7 (observable-supervision soundness).** The observable discovery score
  and the exact-target evaluation agree: |c_obs(k\*, discovery pairs) −
  pooled m=3 exact closure(k\*, evaluation pairs)| ≤ 0.10. This is the
  claim that matters off-oracle: if it fails, ground-truth-free
  interventional discovery is unsound at this scale and the program's
  LLM-phase plan needs revision.

## Failure modes this can newly exhibit

*Echo rediscovery* — the loop converges onto a pls-like subspace with low
true closure: counterexample mining itself can be fooled (would falsify the
"interventional scoring fixes discovery" thesis directly). *Non-convergent*
— k_max reached without the marginal-gain rule firing: the causal content
at L1 is not low-dimensional in any basis reachable by greedy rank-1
refinement. *Variance mimicry* — discovered ≈ pca (small principal angles):
not a failure (P3 expects parity) but reported honestly; this experiment
cannot distinguish "interventional discovery works" from "variance was
right anyway" — that discrimination needs a buried-causal-content regime
and is explicitly out of scope here. *Observable-true divergence* — P7
fails: model-vs-model scoring does not track ground-truth closure.

**Self-checks** (every invocation; `--selftest` exits after them): the four
Experiment-4/5 known-answer checks at L1, plus loop invariants asserted at
every round (proposed direction unit-norm and orthogonal to the current
basis; c_obs(full) ≡ 1 by construction).

---

**Results to be appended below this line after the first run.**
