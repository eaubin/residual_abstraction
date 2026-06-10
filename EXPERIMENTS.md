# Experiment log — overview

One file per experiment in `experiments/`; this file is the index. The
conceptual framework is in `README.md`, the working norms in `AGENTS.md`.

| # | file | script(s) | question | verdict | status |
|---|---|---|---|---|---|
| 1 | [1-sufficiency.md](experiments/1-sufficiency.md) | analysis.py, refine.py | does completeness saturate at the belief dimension; can CEGAR discover it? | yes on Z1R (k=1, the *reachable* belief set); on Mess3 the top-k PCA subspace misses the belief plane — proposal misalignment vs curvature left open | concluded |
| 2 | [2-proposal-families.md](experiments/2-proposal-families.md) | compare.py | misalignment or curvature? | misalignment: X-whitened PLS k\*=2 hits the full-residual identification R² (0.9916 vs 0.9917); seed-stable | concluded |
| 3 | [3-readout-interventions.md](experiments/3-readout-interventions.md) | intervene.py | is the discovered subspace causally load-bearing at the readout? | no — correlational-but-not-causal (closure 63% / 0.7%); the causal channel is the LN-linearized unembedding pullback (100.0% closure at k=V, validated empirically) | concluded |
| 4 | [4-midstream-interventions.md](experiments/4-midstream-interventions.md) | midstream.py | do interventions persist mid-stream, through attention, over multi-token horizons? coherence as state? | pre-registered P1–P6 | running |

## Conventions

KL in nats, held-out unless stated; "affine R²" = held-out R² of an affine
map fit on the train split. Honesty constraint everywhere: proposal families
and probes are supervised on observables (completions) only — belief states
are evaluation ground truth, never training signal. Predictions are
pre-registered and committed before first runs; failed predictions are
reported as findings, not reworked.

## Artifacts

Current run artifacts live in `out/<process>/` (model.pt, config.json,
figures, logs — tracked in git; `cache.npz` is gitignored and regenerates
deterministically on CPU from the tracked model.pt). Per-experiment text
outputs: `out/exp<N>_<process>.txt`, tracked. Superseded runs and rejected
code live in `archive/` (gitignored).
