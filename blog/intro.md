# Residual Streams as Abstract Domains

This repository is an experiment in making interpretability claims behave more
like engineering claims: indexed, testable, and allowed to fail in named ways.

The motivating question is simple to state. Given a language model that has
read some prefix of text, its internal state at that position, the **residual
stream**, determines what it will do next. What is the relationship between
that vector and the distribution of possible continuations?

For a technical reader outside AI, a useful analogy is program analysis. In
abstract interpretation, we replace a concrete program state with a simpler
abstract value, then ask what facts about the program are preserved. This repo
tries to adapt that idea to language models. The concrete meaning of a prefix
is not a set of possible states, but a probability distribution over futures.
The abstract value is not a hand-written interval or type lattice, but a vector
inside a transformer. So the classical guarantees do not transfer directly.
The point here is to keep the spirit of the method without pretending the old
theorems still apply unchanged.

## The Core Idea

For any prefix, there are three objects in play:

1. the prefix itself;
2. the model's residual vector after reading that prefix;
3. the probability distribution over future token sequences.

An abstraction is good when it preserves the third object. Two residual vectors
are equivalent, for the purposes of this project, if they induce nearly the
same distribution over continuations. That is a sufficiency claim: the
abstraction is a sufficient statistic for the future, up to some tolerance.

This immediately changes what "complete" means. In ordinary abstract
interpretation, completeness can be a yes/no property. Here it is a number:
how much the predicted completion distribution differs from the true one,
usually measured by KL divergence. It is also always relative to:

- a tolerance policy;
- a finite horizon, such as the next three tokens;
- a probe or interpreter class, such as linear projections followed by a
  softmax head;
- the distribution of prefixes being evaluated.

Leaving any of those indices out is overclaiming. A central habit of the repo
is to report curves and verdicts under declared conditions, not a single
context-free score.

## Why Use Toy Processes?

The repo starts with small synthetic sequence processes instead of real
language. That is deliberate. For these processes, the ideal state for
prediction is known exactly.

The main examples are:

- **Z1R**, a tiny process that emits `0`, then `1`, then a random bit, and
  repeats;
- **Mess3**, a three-state hidden Markov process whose belief states form a
  fractal inside a simplex;
- **Dyck-2**, a depth-bounded bracket language with stack-like state, now
  present as a concluded transfer test.

For each prefix, the code can compute the exact Bayesian **belief state**:
the posterior distribution over hidden states. This belief state is the
minimal sufficient statistic for the future. It is the gold standard against
which discovered residual abstractions can be checked.

That exactness is the reason the toys matter. The project is not trying to
make claims about toy automata. It is using them as calibration instruments:
small enough that every claim can be audited, but rich enough to expose failure
modes that would be easy to miss in a large model.

## What the Pipeline Does

The basic loop is:

1. Train a small transformer on one of the processes.
2. Cache residual vectors, exact belief states, and exact completion
   distributions for many held-out prefixes.
3. Propose a low-dimensional abstraction of the residual stream.
4. Test whether that abstraction predicts the completion distribution.
5. Mine counterexamples where it fails.
6. Refine, coarsen, or report a typed failure.

This is a CEGAR loop: counterexample-guided abstraction refinement. But the
counterexamples are behavioral. A bad pair is not just two vectors that are far
apart or close together geometrically. It is a pair of prefixes whose proposed
abstractions look the same while their future distributions differ.

The scripts split the work by stage:

- `train.py` trains the model and writes caches;
- `analysis.py` checks belief-state linearity and plots sufficiency curves;
- `refine.py` runs the first CEGAR-style refinement loop;
- `compare.py` compares proposal families such as PCA and completion-supervised
  PLS;
- `intervene.py`, `midstream.py`, `depth.py`, and `discover.py` move from
  correlational decoding tests to causal intervention tests;
- `dyck.py` ports the battery to a richer stack-like process;
- `adversarial.py`, `miners.py`, `candidates.py`, `patches.py`, `reads.py`,
  `readopt.py`, and `readaffine.py` stress-test whether the discovery
  machinery survives hostile coordinate systems and patch parameterizations.

The outputs in `experiments/` are not just result summaries. They are part of
the method: predictions are written down before runs, verdicts are typed, and
unexpected behavior is folded back into later diagnostics.

`FORMALISM.md` now consolidates the named quantities, the separation between
proposal maps, abstractions, and interpreters, and the verdict taxonomy that
the later experiments use.

## What Has Been Learned So Far

The first stage reproduced a known fact from the belief-state literature:
small transformers trained on these hidden-state processes often linearly
embed the Bayesian belief state in their residual stream. On Mess3, the full
residual predicts the exact belief state with held-out R^2 around 0.99.

But the early experiments also showed why "the information is linearly
present" is not the same as "we found the right abstraction." PCA initially
looked plausible because it is cheap and nested by dimension. On Mess3, it
found high-variance directions that mostly encoded current-token identity,
which is useful to the model's readout but not the same as the minimal
completion-relevant state. A completion-supervised, X-whitened PLS proposal
found the belief plane at dimension two. The typed lesson was **proposal
misalignment**: variance is not relevance.

The next surprise was sharper. A subspace can be excellent for decoding
future distributions and still not be the causal channel the model uses.
Interchange interventions at the readout showed that the PLS belief plane was
partly an echo: it contained completion-relevant information, but patching it
did not fully move the model's behavior. The model's own decoder relied more
on higher-variance directions. The typed lesson was
**correlational-but-not-causal**.

Mid-stream interventions then separated two more notions:

- a residual vector can be a good per-position summary;
- that does not mean it is a state that propagates coherently under
  autoregressive extension.

In the two-layer Mess3 model, patching an interior stream position mostly
affected the next prediction, but future positions re-derived the relevant
state from lower-level token information. In a four-layer Mess3 model, a depth
profile found that early layers behave much more like state, while late layers
behave more like summaries. Late patches can even make later predictions worse
than leaving the target run alone, a failure mode recorded as **state
interference**.

Experiment 6 then put interventional scoring inside the discovery loop. At
the early state-like layer of the four-layer Mess3 model, a CEGAR loop using
only observable model-vs-model behavioral divergence found a two-dimensional
causal abstraction that matched the full 64-dimensional patch almost exactly:
98.3% closure versus 98.7% for the full patch. The same setting gave only
2.7% closure for the decode-supervised PLS echo. This was the first major
positive result: at toy scale, oracle-free behavioral scoring can find a
compact causal abstraction.

Experiment 7 then moved the battery from Mess3 to Dyck-2, a bounded
stack-like bracket process with 15 hidden states. This produced the cleanest
dissociation so far. The model was behaviorally near-exact, but the old
belief-state calibration broke: the full residual predicted the exact belief
state with only R^2 = 0.66, and decode probes did not recover the
13-dimensional oracle statistic under the registered rule. The interventional
battery still worked. At the early state-like layer, observable CEGAR found a
four-dimensional causal core closing 92.6% of the behavioral gap, versus
93.6% for a full 64-dimensional patch. The new typed outcome was
**representation-oracle mismatch**: the model can route a compact state that
is behaviorally sufficient without linearly representing the oracle's minimal
sufficient statistic.

That success exposed a caveat. In both Mess3 and Dyck, the interventional
subspace was close to the high-variance PCA subspace. Experiments 8 and 9
constructed adversarial coordinates to test whether the discovery method was
really coordinate-robust. It was not. Behavioral acceptance remained honest:
it rejected bad proposals rather than reporting false success. But the
proposal miner itself was variance-dependent, so in hostile coordinates it
chased amplified junk and stopped at the null abstraction. A scale-free
whitened miner was invariant in the formal sense, but still failed
behaviorally. The typed lesson was **variance dependence of the proposal
map**, not unsoundness of the behavioral score.

Experiments 10 through 13 narrowed the remaining problem. Finite-candidate
interventional search could find directions near the right plane, but the
standard "orthogonal swap in the working coordinates" patch could still be
destructive under ill-conditioned pullbacks. The patch itself had to become
part of the abstraction: not just a write direction, but a write direction
plus a read covector. Hand-built read menus and spectral read searches did
not solve the adversarial case. Gradient-learned reads finally gave the
strongest oracle-free soundness check in the repo: after optimizing reads
directly against observable model-vs-model behavior, observable closure
tracked exact closure within a couple of percentage points. But the learned
reads also overturned the simple "clean geometric read" picture. Some useful
reads appear to be statistical predictors that exploit correlations outside
the original causal plane.

## Current State of the Repo

As of this snapshot, the tracked experiment notes mark Experiments 1 through
13 as concluded:

- Experiment 1 established the sufficiency-curve and refinement machinery.
- Experiment 2 separated PCA failure from genuine nonlinear belief geometry.
- Experiment 3 showed that decode-sufficient subspaces can be causally weak.
- Experiment 4 tested mid-stream persistence and found summary without
  coherent state propagation in the two-layer model.
- Experiment 5 produced a layer-by-layer state profile in a four-layer model.
- Experiment 6 used interventional CEGAR to discover a compact causal
  abstraction from observable behavior.
- Experiment 7 showed that the interventional battery transfers to Dyck even
  when linear oracle-belief calibration fails.
- Experiment 8 showed that the original interventional proposal miner was
  variance-lucky: acceptance was sound, but proposal generation was not
  coordinate-robust.
- Experiment 9 confirmed the intended invariance repair mathematically and
  numerically, then showed that invariance alone did not make the miner
  behaviorally useful.
- Experiment 10 moved to finite-candidate interventional search and located
  the next failure in the patch parameterization.
- Experiments 11 and 12 made the read side of a patch explicit, then showed
  that simple read menus and spectral read interpolation were still
  insufficient.
- Experiment 13 optimized read covectors directly against observable behavior,
  validating oracle-free scoring under stronger selection pressure while
  revealing that useful reads need not be clean geometric projections.

Experiment 14 is pre-registered. Its job is to settle the read-optimization
dynamics more cleanly: whether the earlier divergent read run was caused by
the renormalized parameterization, whether an affine-slice parameterization
repairs it, how learned reads compose, and how much they function as
statistical predictors of the effective causal plane.

## What This Is Not Claiming

This project is not a proof that we can recover human-interpretable concepts
from language models. It is not even trying to find "the" true internal state
of a model in an absolute sense.

The claim is narrower and more useful: for a declared distribution, horizon,
probe family, tolerance, and intervention protocol, we can measure how much
completion-relevant structure a proposed abstraction preserves, and we can
name the way it fails.

That scope matters. A concept that rarely changes the completion distribution
on the evaluation distribution will be invisible to this method, even if it is
semantically or normatively important. Passing these tests says "sufficient
for these futures under these conditions." It does not say "identical to a
human ontology," and it does not give off-distribution guarantees.

## The Longer Aim

The eventual target is real language models, where there is no exact belief
state and no oracle over all futures. What should transfer is not the toy
ground truth, but the discipline:

- always compute a no-information baseline;
- report capacity-indexed curves instead of isolated points;
- separate decoding from causal use;
- test coherence under generation, not only static sufficiency;
- test coordinate robustness rather than trusting variance-friendly regimes;
- treat the patch map itself as part of the abstraction, not an implementation
  detail;
- use proposal families supervised only on observables;
- keep a taxonomy of typed failures;
- treat surprising results as harness bugs until the harness survives.

The hoped-for deliverable is not "we found everything the model knows." It is
measured missability: for a given concept, distribution, probe class, horizon,
and tolerance, how much completion-relevant structure escapes the abstraction,
with error bars and named caveats.

That is a modest claim, but it is the kind of claim this repo is trying to make
checkable.
