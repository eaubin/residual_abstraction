# Residual Methods Overlap Note

This note records a future comparison point, not a literature review. The
project should revisit it once the experimental substrate is mature enough
that this repo and modern residual-manipulation methods can both operate on
the same model/process class.

## Why this matters

External mechanistic-interpretability work has moved quickly on how to
interpret and manipulate residual streams. The relevant families include:

- activation steering and representation engineering;
- diff-in-means directions and supervised linear probes;
- sparse autoencoder features and related dictionary-learning latents;
- ReFT / low-rank representation edits;
- activation patching and interchange interventions;
- transcoders, crosscoders, and circuit-tracing style replacement models.

These methods are not automatically answers to this project's question. They
are candidate proposal, intervention, or decomposition families whose claims
need to be scored against completion behavior.

## Project relation

This project's core object is not "an interpretable residual feature." It is
a residual-derived abstraction evaluated by how much completion-relevant
structure it preserves under declared:

- horizon;
- evaluation distribution;
- tolerance policy;
- interpreter/proposal class;
- intervention family.

Modern residual methods therefore enter the project as candidates to test,
not as authorities to inherit. An SAE latent, steering direction, probe
direction, ReFT direction, transcoder node, or circuit component can be
treated as a proposed abstraction. The battery then asks whether it preserves
the completion measure, where it fails, and whether the failure is typed as
domain coarseness, interpreter incompleteness, proposal misalignment,
metric junk-domination, vacuous tolerance, or some new registered failure.

## Complement and competition

The methods are complementary when they provide better candidate objects:
they may give proposal families that are more semantically meaningful, more
causal, or more scalable than PCA/PLS/random/hand-coded subspaces.

They compete when they claim that a discovered object is "the concept,"
"the state," or "the mechanism." This project should ask the stricter
question: does that object function as a sufficient summary for futures,
within the registered horizon/distribution/tolerance/probe setting?

Manipulability alone is not sufficiency. A direction can causally change
outputs without preserving all completion-relevant variation. Interpretability
alone is not sufficiency. A human-legible feature can be too coarse, too
fragile, or irrelevant to the completion measure. Conversely, an ugly or
uninterpretable subspace may be behaviorally sufficient.

## When to revisit

Do the serious overlap pass when all of the following are true:

1. The project has a model/process where residual interventions are
   nontrivial and external methods are applicable. Likely candidates are
   late Dyck/PCFG stages or the first TinyStories-scale model, not the
   current calibration-only point.
2. The harness can compare multiple proposal families under the same
   completion-measure battery.
3. There is a stable intervention API for patching, projecting, clamping,
   ablating, or steering residual-derived variables.
4. The experiment can pre-register exact acceptance, recalibration, and
   failure branches before running.

## Future comparison battery

At that point, run a registered comparison with shared data, shared horizons,
shared baselines, and shared verdict logic:

- baseline proposal families: PCA, PLS, random, existing CEGAR proposals;
- representation methods: diff-in-means, linear probes, SAE latents,
  ReFT-style directions, possibly transcoder/circuit nodes;
- intervention tests: activation patching, projection/ablation, steering,
  and interchange interventions where meaningful;
- score: completion divergence curves, no-information baselines, accepted
  cell calibration where oracle ground truth still exists, and typed
  failures otherwise.

The central question should be comparative:

> When treated as abstractions of the residual stream, do modern residual
> methods preserve future behavior better, worse, or differently than this
> project's current proposal families?

## Boundary condition

The honest pitch remains unchanged. Passing the battery can certify only
that an object functions as a sufficient summary under the registered probes,
distribution, horizon, tolerance, and interventions. It does not certify
identity with a human ontology and does not give off-distribution guarantees.

## Pointers

Representative external threads to revisit:

- Representation Engineering: A Top-Down Approach to AI Transparency
  (Zou et al., 2023).
- Causal Abstraction / interchange interventions
  (Geiger et al., 2021/2023).
- AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse
  Autoencoders (Wu et al., 2025).
- SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language
  Model Interpretability (Karvonen et al., 2025).
- Circuit tracing / transcoders / replacement-model approaches, especially
  work that yields manipulable graph nodes rather than only labeled features.

These pointers should be refreshed before the comparison is run; the point of
this note is to preserve the project-level relation, not to freeze the
literature snapshot.
