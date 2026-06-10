# Experiment 2 — proposal families, the tolerance staircase, the variance audit

**Script:** `compare.py`. **Status: CONCLUDED.** Raw outputs:
`out/exp2_{mess3,z1r}.txt`.

**Question.** Was the Mess3 identification gap of
[Experiment 1](1-sufficiency.md) (a) subspace misalignment — a PCA artifact
— or (b) a genuinely curved belief embedding?

**Pre-registered prediction.** A completion-supervised family reaches affine
abstraction→belief R² near the full-residual reference (~0.99) at k = 2 ⇒
(a); all families plateauing below it ⇒ (b).

## Answer: (a), decisively

Mess3, stopping rule = per-sample KL within 2 SE of the belief-oracle floor
OR ≥98% of the closable (KL0 − oracle) range closed:

| family | k* | affine R² at k* | k-NN R² at k* |
|---|---|---|---|
| pca | 8 | 0.9813 | 0.9892 |
| **pls (X-whitened)** | **2** | **0.9916** | **0.9955** |
| head row space | >10 | 0.9454 | 0.9728 |

The full-residual reference is 0.9917: at k = 2 the PLS subspace recovers
*all* of the linearly readable belief geometry. The model learned a flat,
affinely-identifiable belief plane; PCA was simply looking at the wrong two
directions.

**Naming note (post-hoc correction).** The winning family (`CompletionPLS`)
is *X-whitened cross-covariance*: residuals are whitened
(ridge-regularized), completion distributions are centered but **not**
variance-normalized — so it is not full cross-correlation/CCA, and earlier
"CCA-flavored" phrasing overstated it. Whitening X is the load-bearing part
(it is what fixed the scale-blindness of plain cross-covariance, which
failed the buried-belief validation cache; that rejected variant is
preserved in `archive/rejected_families.py`). Whitening Y as well is
untested.

**Why PCA fails (displaced-variance audit, Mess3).** PC1 and PC2 carry 88%
of (position-centered) residual variance and are R² ≈ 0.99 explained by
*current-token identity* — structure the unembedding path needs but which is
completion-redundant given the belief. Their belief R² (0.79/0.83) is
incidental: in Mess3 each hidden state preferentially emits its own symbol,
so the token plane partially carries belief — which is exactly why PCA
limped to 0.81 in Experiment 1 rather than failing cleanly.

**Z1R separates two senses of "found it."** PCA passes at k = 1
(behaviorally sufficient: three belief points embed injectively on a line;
k-NN R² 0.9998) while PLS at k = 2 recovers the affine simplex (affine R²
0.9806, k-NN 1.0000). "Sufficient for completions at this tolerance" and
"affinely equivalent to the belief simplex" are different claims and the
staircase k*(tol) shows where they part: at tolerances below ~0.01 nats only
PLS k = 2 reaches the oracle floor.

## Seed stability (seeds 0/1/2, full pipeline rerun per seed)

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Mess3 pls k* (affine R²) | 2 (0.9916) | 2 (0.9916) | 2 (0.9915) |
| Mess3 pca k* | 8 | 8 | 8 |
| Mess3 head k* | >10 | 3 | >10 |
| Z1R pls k* (affine R²) | 2 (0.9806) | 2 (0.9810) | 2 (0.9809) |
| Z1R pca k* | 1 | 1 | 1 |

## Typed findings

1. *Proposal misalignment resolved.* The Experiment-1 Mess3 gap was the
   proposal family, not the model. Variance is not relevance; X-whitened
   cross-covariance with completions is a working relevance ordering.
2. *Covariance inherits variance.* Plain (unwhitened) cross-covariance PLS
   failed the buried-belief validation cache exactly as theory predicts
   (scale-blind); whitening X fixed it.
3. *The decoder's fitted row space is not a reliable relevance basis.*
   HeadRowSpace underperforms and is seed-UNSTABLE (k* = 3 vs >10 across
   seeds): the 400-step numpy Adam fit on 64 standardized dims yields a
   noisy weight matrix whose singular directions mix signal with
   optimization noise — even though the decoder itself decodes fine.
   (Experiment 3 rehabilitates the *idea* at fit budget zero: the model's
   own unembedding rows.)
4. *Head-fit noise bounds resolution.* Within-family KL(k) curves are
   non-monotone at the ~1e-3 level (nested subspaces cannot truly get worse
   with k); differences at that scale are optimizer jitter, not structure.
   The k* and identification conclusions are far above this floor and are
   seed-stable.

## Code closeout

Removed a duplicate `completeness_kl_rows` that shadowed its twin and two
dead proposal-family drafts from abstraction.py; made analysis.py's elbow
verdict conditional; fixed compare.py's docstring mischaracterizing
CompletionPLS. Regression-checked: seed-0 rerun after cleanup reproduces the
recorded outputs exactly.
