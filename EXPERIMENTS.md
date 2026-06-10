# Experiment log — overview

One file per experiment in `experiments/`; this file is the index. The
conceptual framework is in `README.md`, the working norms in `AGENTS.md`.

| # | file | script(s) | question | verdict | status |
|---|---|---|---|---|---|
| 1 | [1-sufficiency.md](experiments/1-sufficiency.md) | analysis.py, refine.py | does completeness saturate at the belief dimension; can CEGAR discover it? | yes on Z1R (k=1, the *reachable* belief set); on Mess3 the top-k PCA subspace misses the belief plane — proposal misalignment vs curvature left open | concluded |
| 2 | [2-proposal-families.md](experiments/2-proposal-families.md) | compare.py | misalignment or curvature? | misalignment: X-whitened PLS k\*=2 hits the full-residual identification R² (0.9916 vs 0.9917); seed-stable | concluded |
| 3 | [3-readout-interventions.md](experiments/3-readout-interventions.md) | intervene.py | is the discovered subspace causally load-bearing at the readout? | no — correlational-but-not-causal (closure 63% / 0.7%); the causal channel is the LN-linearized unembedding pullback (100.0% closure at k=V, validated empirically) | concluded |
| 4 | [4-midstream-interventions.md](experiments/4-midstream-interventions.md) | midstream.py | do interventions persist mid-stream, through attention, over multi-token horizons? coherence as state? | P1–P5 hold, P6 fails: per-step incremental closure 12.5%/0.0% at steps 2/3 — future positions re-derive state from raw tokens below the patch; the stream is a per-position *summary*, not propagated *state* | concluded |
| 5 | [5-depth-profile.md](experiments/5-depth-profile.md) | depth.py | at which depth, if any, does the stream function as state? (4-layer Mess3, patch every interior layer, per-step incremental closure profile) | P1–P6 all hold: state early (L1: 93.7%/91.0% incremental, 94.8% coherence), summary late — and at L3 incremental closure goes *negative* (−29.7%/−83.7%): mixed-provenance state corrupts (new typed failure: state interference) | concluded |
| 6 | [6-interventional-discovery.md](experiments/6-interventional-discovery.md) | discover.py | can a CEGAR loop scored by *observable* interchange closure at L1 discover a low-dim causal abstraction approaching full-patch closure, avoiding the PLS echo failure — without trivially growing k? | P1–P7 all hold: k\*=2 (= belief-simplex dim) at 98.3% vs full 98.7%; echo avoided (pls 2.7%, near-orthogonal); oracle-free score tracks exact closure to 1.5 points; converged to the variance plane (declared mimicry limit → experiment 8 is the discriminator) | concluded |
| 7 | [7-dyck.md](experiments/7-dyck.md) | dyck.py (+ processes.dyck2) | does the battery transfer to a stack process (depth-bounded Dyck-2, 15 states)? state-vs-bypass tension: attention can bracket-match from raw tokens | P3–P9 hold, P1–P2 fail *informatively*: the linear-belief calibration breaks (R² 0.66, decode k\*>12, k_B=13) on a behaviorally-exact model, while interventional CEGAR finds a 4-dim causal core closing 92.6% vs full 93.6% with oracle-free scoring sound (5.9 pts) — new typed outcome: representation–oracle mismatch; no state interference on Dyck; variance mimicry recurs (→ exp 8 still needed) | concluded |
| 8 | [8-adversarial-coordinates.md](experiments/8-adversarial-coordinates.md) | (none yet — deliberately uncoded until exp 7 concludes) | does interventional discovery survive adversarially ill-conditioned coordinates where PCA must fail? (closes exp 6's variance-mimicry limitation; constructed validation, exp-2 buried-cache tradition) | pre-registered P1–P5 | pre-registered, no code, not run |

## Conventions

KL in nats, held-out unless stated; "affine R²" = held-out R² of an affine
map fit on the train split. Honesty constraint everywhere: proposal families
and probes are supervised on observables (completions) only — belief states
are evaluation ground truth, never training signal. Predictions are
pre-registered and committed before first runs; failed predictions are
reported as findings, not reworked.

## Artifacts

Current run artifacts live in `out/<process>/` (model.pt, config.json,
figures, logs — tracked in git). `cache.npz` is gitignored; in a clean
checkout regenerate it with
`python3 train.py --process <p> --outdir out/<p> --cache-only`, which loads
the tracked model.pt and rebuilds the cache on CPU without touching the
checkpoint (deterministic; caches recorded from MPS runs match to float
tolerance — the tracked `out/exp*.txt` logs are the canonical numbers).
Per-experiment text outputs: `out/exp<N>_<process>.txt`, tracked.
Superseded runs and rejected code live in `archive/` (gitignored).
