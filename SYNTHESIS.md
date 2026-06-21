# Project Synthesis

This is the infrequently updated synthesis layer. It is meant to stand on its
own. It hides most experiment detail and states, in broad terms, what the
project has learned, what it still assumes, and what conclusions should not be
drawn.

Update this document at phase boundaries, consolidations, or branch-closing
results. Do not update it for every experiment or use it as a handoff. The
detailed experiment writeups remain the audit trail.

Last synthesis update: after experiment 31.

## The Core Question

The project studies whether a language model's hidden vectors contain compact
pieces that matter for what the model will predict next.

The basic test is behavioral. If we swap or replace a candidate part of the
hidden vector, does the model's distribution over future tokens change in the
way it should? If yes, that part is a plausible control handle. If it is merely
decodable, that only shows the information is present somewhere in the vector;
it does not show the model uses that direction causally.

The toy processes are not the subject. They are calibration environments where
the true future distribution is available, so the measurement method can be
checked before ground truth is removed.

## Established Lessons

1. Model-vs-model behavioral scores have been reliable on the toy systems
   tested so far.

   The project often scores an intervention by asking whether the patched model
   behaves more like a source run of the same model. On the toy systems, this
   observable score repeatedly agreed with exact ground-truth scoring. That
   supports using observable scores for discovery, while using exact truth only
   for audits when it is available.

2. Correlation and causal control separate.

   Several methods find directions that predict future-relevant quantities but
   do not control future behavior when patched. A decoded feature is not
   automatically a usable intervention handle.

3. Proposal generation is fragile; behavioral acceptance is more reliable.

   Methods that propose candidate directions from variance or covariance can
   fail under reparameterizations of the same hidden vectors. The important
   observed property is that behavioral acceptance tests tend to reject bad
   proposals rather than confidently accepting them.

4. The model's useful internal state need not match the true hidden state of the
   data-generating process.

   On stack-like toy data, the model's behavior was well captured by a compact
   discovered core even though decoding the process's exact hidden state was a
   poor guide. The model solved the task in its own coordinates. Internal-state
   claims should therefore be judged by behavior, not by identity with the
   oracle's representation.

5. The diagnostic battery transfers, but only under explicit indices.

   A small set of diagnostics was calibrated on one toy process and then moved
   to two others. It worked under the registered model checkpoints, layers,
   horizons, distributions, reference patches, and thresholds. This is evidence
   for the workflow, not a general theorem.

6. Oracle-free reference selection remains only partly solved.

   Some diagnostics need a trusted reference intervention. On the latest toy
   process, the workflow produced a usable declared reference, but did not
   uniquely earn that reference from observable behavior alone. The distinction
   matters: "we can proceed with a declared anchor" is supported there;
   "observable evidence uniquely selects the right anchor" is not.

7. Predicate readout and predicate control split.

   Some future-token properties are linearly readable from the hidden vector and
   calibrated against exact truth, but simple interventions still fail to
   control them. Predicate work must keep "information is present" separate from
   "this intervention can use it."

8. Read geometry can be position-specific.

   In the latest predicate work, the problem was not that the target properties
   were absent at the tested layer. They were readable at each position, but the
   read direction changed by position. This makes read transport a first-class
   issue for predicate interventions.

## Standing Assumptions

- Completion behavior on the evaluation distribution is the object being
  measured. Passing tests does not imply off-distribution validity or identity
  with human concepts.
- Linear residual interventions are a useful test class, but not guaranteed to
  be the right primitive. Current predicate work is explicitly testing that
  boundary.
- A trusted reference patch can anchor equivalence tests when it has been
  behaviorally validated. At larger scale, earning or declaring that reference
  remains a central unresolved problem.
- Exact ground truth is evaluation-only. Proposal fitting and intervention
  selection should use observable model behavior unless a control explicitly
  registers otherwise.
- The current stack-like toy process is a useful rehearsal substrate but not a
  rich new physics result. More work on it should change an intervention-class
  decision, not merely mine the substrate.

## Durable Implications

1. New abstraction claims should be indexed by distribution, horizon, tolerance,
   patch/interpreter class, and reference patch.

2. Observable diagnostics can guide discovery, but accepted results need
   calibration, baselines, and failure branches. A clean number without its
   branch is not a conclusion.

3. When a readout fails to transport, first distinguish representational absence
   from chart/transport failure. Do not treat a failed global read as evidence
   that the layer or target is wrong until the in-place read has been checked.

4. When residual-level linear interventions fail despite room and calibration,
   the next explanation should usually be one of: wrong read chart, off-manifold
   write, nonspecific distribution replacement, or wrong patch point. These
   should be separated by design, not narrated after the run.

5. A toy process should not become the default place to mine more phenomena. It
   is useful when it distinguishes intervention classes or validates workflow
   machinery; otherwise the program should move to a process designed around
   the next conceptual question.

## Intervention-Class Consequences

The predicate-intervention branch has not yet validated a primitive that can
stand for "we can manipulate the internal predicate variable." The tested and
candidate intervention classes should be interpreted in this order:

- Same-read/write rank-1 patches: tested negatively for the registered
  predicates; do not treat them as adequate by default.
- Fixed-read oblique writes: not adjudicated as a write-only question when the
  fixed read itself does not transport.
- Read/write pairs with position-valid reads: the next relevant residual-level
  test class.
- Matched activation deltas or interchange-style interventions: the natural
  next class if arbitrary linear read/write pairs fail despite room.
- Component/path localization: justified after residual-level tests create a
  specific localization question, or when residual interventions work but are
  too broad.

This ordering is not a schedule. It is a guard against over-interpreting one
failed primitive as a general absence claim.

## Known Limits

- The method measures what is load-bearing for completions on the tested
  distribution. Rare, adversarial, normative, or off-distribution concepts can
  remain invisible.
- Battery thresholds are process-indexed. Reusing them without recalibration is
  an error.
- Equivalence scores depend on the trusted reference and can be too lenient at
  intermediate strengths.
- Mean-level equivalence can hide pair-level failures; distribution-local
  equivalence is not transported state.
- The current intervention work has not yet validated a predicate-control
  primitive.

## Minimal Glossary

- Hidden vector / residual stream: the model's internal vector at a layer and
  position.
- Completion behavior: the model's probability distribution over the next few
  tokens.
- Abstraction: a proposed low-dimensional summary of the hidden vector.
- Patch / intervention: replacing part of one example's hidden vector with the
  corresponding part from another example, then measuring model behavior.
- Read: the direction or function used to measure a value in the hidden vector.
- Write: the direction used to change the hidden vector.
- Reference patch: a trusted intervention used as the comparison target for
  behavioral equivalence.
- Predicate: a Boolean property of short future continuations, such as whether
  the next token closes a bracket.
