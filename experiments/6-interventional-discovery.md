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
Experiment-4/5 known-answer checks at L1; loop invariants asserted at every
round (proposed direction unit-norm and orthogonal to the current basis);
and asserted invariants of the observable scale (D₀ > D_full, and
c_obs(full) = 1 — guarding the closure denominator against code drift).

**Enforcement (review fix, pre-run).** The registered loop/protocol
parameters (eps_gain, eps_drop, k_max, pair counts, control-basis sample
size, m) are guarded in code exactly like the model config: overriding any
of them without `--force-invalid` refuses to run, and with it the output is
banner-labeled EXPLORATORY, not Experiment 6. (Seed is exempt: a different
seed is labeled a seed-robustness rerun of the registered design.) The
control-basis sample flag is named `--basis-seqs` to avoid the
reproducibility trap of sounding like it controls the discovery pairs —
the discovery/evaluation pair sets are fixed by the registered protocol.

---

## Results: P1–P7 ALL HOLD — interventional CEGAR finds the causal plane at k\* = 2

(Registered parameters, seed 0, gate +0.0024 PASS, self-checks and scale
invariants passed. Raw output `out/exp6_mess3-L4.txt`, figure
`out/mess3-L4/experiment6.png`.)

**The loop.** Two rounds and done: k=1 closes 54.3% of observable closure,
k=2 closes 99.8%, the k=3 candidate earns +0.1% and the marginal-gain rule
stops the loop; the coarsen pass removes nothing. The anti-triviality
guards were never stressed — the loop stopped four dimensions short of
k_max on its own.

**Evaluation (disjoint pairs, exact targets, pooled m=3):**

| basis | k | closure |
|---|---|---|
| full | 64 | 98.7% |
| **discovered** | **2** | **98.3%** |
| pca | 2 | 98.7% |
| pls | 2 | 2.7% |
| rand | 2 | 6.2% |
| emb (token-embedding span) | 2 | 69.7% |

Nested curve: k=1 → 53.6%, k=2 → 98.3%. Per-position stability ≤ ~1 point
everywhere.

**Finding 1 (P1, P2, P6): interventional scoring fixes discovery.** The
same greedy, completions-era CEGAR shape — propose from counterexamples,
test, stop at a declared margin — discovers a 2-dimensional subspace that
is causally equivalent to patching the entire 64-dim stream (98.3% vs
98.7%), where the decode-supervised proposal family found a 3% echo
(principal angles to the discovered plane: 86–87°, i.e. the echo is almost
orthogonal to the causal plane). The discovered k\* = 2 equals the
dimension of the belief simplex: the interventional loop recovers the same
minimal dimensionality that Experiment 2's decode-loop found, but this
time the subspace *is* the channel, not a copy of it.

**Finding 2 (P7, the load-bearing one for the program): oracle-free
scoring is sound here.** The loop never touched beliefs; its model-vs-model
objective (KL of patched behavior against the source *run*) agreed with the
exact belief-conditioned evaluation to 1.5 points (99.8% vs 98.3%, on
disjoint pairs). This is the soundness property the LLM phase needs, now
with one data point in its favor at toy scale.

**Finding 3 (declared characterization): variance mimicry occurred, as the
registration anticipated.** Principal angles between the discovered plane
and PCA's top-2: 3.3°, 3.6° — the loop, scored purely by intervention,
independently converged onto the variance plane. On this model that is the
*correct* answer (P3 parity, not superiority, was the registered claim, and
it held at 0.4 points). But it means this experiment cannot distinguish
"interventional discovery works" from "variance was right anyway" —
exactly the limitation declared in the scope notes. The discriminating
regime (causal content buried at low variance, where PCA must fail and the
interventional loop must not) is the natural Experiment 7.

**Finding 4 (emb control): the architecture basis at the input side is
partial.** The token-embedding span closes ~70% — substantial (current
token is most of what position t contributes at L1) but well short of the
discovered plane (one of its two directions is 56° away): block-1's
aggregated history content is causally necessary and lives outside the
embedding span. The input-side analogue of Experiment 3's unemb pullback
is *not* sufficient, unlike its output-side counterpart.

**Status: CONCLUDED.**
