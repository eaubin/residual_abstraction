# Experiment log — overview

One file per experiment in `experiments/`; this file is the index and the
per-experiment writeups are **canonical** — rows here are pointers, not
summaries (slimmed at exp 15; the old long-form rows are in git history).
Conceptual framework: `README.md`; working norms: `AGENTS.md`; named
objects, per-experiment quantities, verdict taxonomy, assumption ledger:
`FORMALISM.md`.

| # | file | script(s) | question | verdict | status |
|---|---|---|---|---|---|
| 1 | [1-sufficiency.md](experiments/1-sufficiency.md) | analysis.py, refine.py | does decode-completeness saturate at the belief dimension? | yes on Z1R (k\*=1); on Mess3 top-k PCA misses the belief plane | concluded |
| 2 | [2-proposal-families.md](experiments/2-proposal-families.md) | compare.py | proposal misalignment or representation curvature? | misalignment: X-whitened PLS k\*=2 matches full-residual identification | concluded |
| 3 | [3-readout-interventions.md](experiments/3-readout-interventions.md) | intervene.py | is the discovered subspace causally load-bearing at the readout? | no — correlational-not-causal; the causal channel is the unembedding pullback | concluded |
| 4 | [4-midstream-interventions.md](experiments/4-midstream-interventions.md) | midstream.py | do mid-stream interventions persist over multi-token horizons? | P6 fails: the stream is a per-position *summary*, not propagated *state* | concluded |
| 5 | [5-depth-profile.md](experiments/5-depth-profile.md) | depth.py | at which depth does the stream function as state? | state early (L1 ≈ 94% incremental), summary late; L3 negative = *state interference* | concluded |
| 6 | [6-interventional-discovery.md](experiments/6-interventional-discovery.md) | discover.py | can observable-scored CEGAR discover a low-dim causal abstraction? | yes: k\*=2 at 98.3% vs full 98.7%, oracle-free score sound to 1.5 pts — but converged to the variance plane (mimicry limit) | concluded |
| 7 | [7-dyck.md](experiments/7-dyck.md) | dyck.py | does the battery transfer to a stack process (Dyck-2)? | causal core found (4-dim, 92.6%); linear-belief calibration fails — *representation–oracle mismatch* | concluded |
| 8 | [8-adversarial-coordinates.md](experiments/8-adversarial-coordinates.md) | adversarial.py | does discovery survive adversarially ill-conditioned coordinates? | no: variance dependence exposed (miner dies, k\*=0); acceptance never false-confident | concluded |
| 9 | [9-scale-free-mining.md](experiments/9-scale-free-mining.md) | miners.py | can the proposal miner be made scale-free? | no: invariance confirmed exactly, but the invariant miner has no causal signal — correlational mining exhausted | concluded |
| 10 | [10-interventional-search.md](experiments/10-interventional-search.md) | candidates.py | does finite-candidate interventional search work? | selection works; *patch parameterization* fails — the read side is junk-amplified ×κ | concluded |
| 11 | [11-patch-parameterization.md](experiments/11-patch-parameterization.md) | patches.py | does a (write, read) menu {id, prec, cov} fix the patch? | read *concept* vindicated (+51.3% clean vs +1.0% id, same write); the menu is insufficient | concluded |
| 12 | [12-read-search.md](experiments/12-read-search.md) | reads.py | does some spectral read c ∝ Σ̂^{−α}w suffice? | no — flat ~+1.5% across the grid; D2 composition exonerated at 97.8% | concluded |
| 13 | [13-read-gradient.md](experiments/13-read-gradient.md) | readopt.py | can the read be learned from behavioral signal? | P4 holds at last (0.3/1.6 pts); w2's read works at +43.7% with **zero plane mass**; w1 diverges | concluded |
| 14 | [14-affine-reads.md](experiments/14-affine-reads.md) | readaffine.py | mechanism settle + affine repair + EPR diagnostic | both exp-13 hypotheses die: renorm feedback refuted (divergence = per-write landscape asymmetry); working reads score EPR ≈ 0; P4 holds again | concluded |
| 15 | [15-transport.md](experiments/15-transport.md) | transport.py | statistical control vs transported state (per-pair equivalence ρ, EPR cells, distribution shifts) | adjudicated: **position-entangled statistical control** — pooled EPR was the wrong aggregation (position-t 0.85–0.93); learned reads invert at unseen positions (R −0.77/−0.41) while clean improves; ρ separator validated; P4 3rd hold; both decisive instruments oracle-free → battery members | concluded |

## Conventions

KL in nats, held-out unless stated; "affine R²" = held-out R² of an affine
map fit on the train split. Honesty constraint everywhere: proposal families
and probes are supervised on observables (completions) only — belief states
are evaluation ground truth, never training signal. Predictions are
pre-registered and committed before first runs; failed predictions are
reported as findings, not reworked.

**Redundancy policy (adopted at exp 15).** The per-experiment writeup is
the single canonical record of a conclusion. Everything else points at
it: index rows above are one line; script docstring RESULTS sections are
≤ 3 lines plus the pointer; FORMALISM §7 ledger rows carry only the
*current status* of an assumption. Conclusions are written once, not four
times.

**Code policy (adopted at exp 15).** Concluded experiment scripts are
frozen records — they keep their inline machinery and are never
refactored. The living edge (exp 15 onward) imports the shared
scaffolding from `expcommon.py` (guards, gate, anchor, transform, write
reproduction, differentiable chain, affine optimizer, shared metrics), so
there is exactly one live copy; new scripts contain only
experiment-specific content. Registrations reference "the standard
setting" and "the standard guards/self-checks" instead of restating them,
and state only deltas.

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
