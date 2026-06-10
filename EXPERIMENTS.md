# Experiment log

Conclusions per experiment, with the numbers that carry them. Conventions
throughout: KL in nats, held-out unless stated; "affine R^2" means held-out
R^2 of an affine map fit on the train split; proposal families and probes are
supervised on observables (completions) only — belief states are evaluation
ground truth, never training signal. Raw outputs for Experiment 2 live in
`out_exp2/` (gitignored; regenerate with `run_all.py` + `compare.py`).

---

## Experiment 1 (analysis.py + refine.py): sufficiency curves and the CEGAR loop

**Question.** Does quantified completeness (held-out KL from a rank-k
abstraction of the residual stream to the exact m=3 completion distribution)
saturate at the dimension of the known minimal sufficient statistic, and can
a counterexample-guided loop discover that rank without being told?

**Findings (kept; restated here for continuity).**

- Calibration replicated Shai et al.: affine residual→belief R^2 = 0.9917
  (Mess3), 0.9842 (Z1R); the Mess3 fractal is visible under the probe.
- Z1R: fixed point at k = 1 — *correct* for the realized semantics: the
  reachable synchronized belief set is three points, embeddable injectively
  in 1-D (k-NN identification R^2 0.9998 vs affine 0.4769). The complete
  shell is of the realized semantics, not the ambient simplex.
- Mess3: fixed point at k = 2 under the loop's tolerance, but identification
  was the open wound: affine abstraction→belief R^2 only 0.8104 inside the
  top-2 principal subspace vs 0.9917 from the full residual. Verdict cell:
  belief geometry linearly present in the residual, missing from the top-k
  PCA subspace — *proposal misalignment* ("variance is not relevance"), with
  the alternative hypothesis (genuinely curved embedding) left undecided.
- The PCA sufficiency curve's strict elbow sits at k = 8 on Mess3 — by the
  Experiment-2 stopping rule PCA needs k = 8 to match the belief-oracle
  floor. (analysis.py's verdict text now branches on this instead of
  declaring victory unconditionally; the old unconditional "matches the known
  minimal sufficient statistic" message was stale after Experiment 2.)

**Status: concluded**, superseded in interpretation by Experiment 2.

---

## Experiment 2 (compare.py): proposal families, the staircase, the variance audit

**Question.** Was the Mess3 identification gap (a) subspace misalignment — a
PCA artifact — or (b) a genuinely curved belief embedding?

**Pre-registered prediction.** A completion-supervised family reaches affine
abstraction→belief R^2 near the full-residual reference (~0.99) at k = 2 ⇒
(a); all families plateauing below it ⇒ (b).

**Answer: (a), decisively.** Mess3, stopping rule = per-sample KL within
2 SE of the belief-oracle floor OR ≥98% of the closable (KL0 − oracle) range
closed:

| family | k* | affine R² at k* | k-NN R² at k* |
|---|---|---|---|
| pca | 8 | 0.9813 | 0.9892 |
| **pls (whitened)** | **2** | **0.9916** | **0.9955** |
| head row space | >10 | 0.9454 | 0.9728 |

The full-residual reference is 0.9917: at k = 2 the PLS subspace recovers
*all* of the linearly readable belief geometry. The model learned a flat,
affinely-identifiable belief plane; PCA was simply looking at the wrong two
directions.

**Why PCA fails (displaced-variance audit, Mess3).** PC1 and PC2 carry 88%
of (position-centered) residual variance and are R² ≈ 0.99 explained by
*current-token identity* — structure the unembedding path needs but which is
completion-redundant given the belief. Their belief R² (0.79/0.83) is
incidental: in Mess3 each hidden state preferentially emits its own symbol,
so the token plane partially carries belief — which is exactly why PCA limped
to 0.81 in Experiment 1 rather than failing cleanly.

**Z1R separates two senses of "found it."** PCA passes at k = 1
(behaviorally sufficient: three belief points embed injectively on a line;
k-NN R² 0.9998) while PLS at k = 2 recovers the affine simplex (affine R²
0.9806, k-NN 1.0000). "Sufficient for completions at this tolerance" and
"affinely equivalent to the belief simplex" are different claims and the
staircase k*(tol) shows where they part: at tolerances below ~0.01 nats only
PLS k = 2 reaches the oracle floor.

**Seed stability (seeds 0/1/2, full pipeline rerun per seed).**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Mess3 pls k* (affine R²) | 2 (0.9916) | 2 (0.9916) | 2 (0.9915) |
| Mess3 pca k* | 8 | 8 | 8 |
| Mess3 head k* | >10 | 3 | >10 |
| Z1R pls k* (affine R²) | 2 (0.9806) | 2 (0.9810) | 2 (0.9809) |
| Z1R pca k* | 1 | 1 | 1 |

**Typed findings.**

1. *Proposal misalignment resolved.* The Experiment-1 Mess3 gap was the
   proposal family, not the model. Variance is not relevance; whitened
   cross-correlation with completions is a working relevance ordering.
2. *Covariance inherits variance.* Plain cross-covariance PLS failed the
   buried-belief validation cache exactly as theory predicts (scale-blind);
   whitening fixed it. The rejected variant is preserved in
   `archive/rejected_families.py`.
3. *The decoder's row space is not a reliable relevance basis.* HeadRowSpace
   underperforms and is seed-UNSTABLE (k* = 3 vs >10 across seeds): the
   400-step numpy Adam fit on 64 standardized dims yields a noisy weight
   matrix whose singular directions mix signal with optimization noise —
   even though the decoder itself decodes fine. Caution for the LLM phase:
   "use the unembedding/probe row space as the subspace" is a tempting
   default and it is not safe at modest fit budgets.
4. *Head-fit noise bounds resolution.* Within-family KL(k) curves are
   non-monotone at the ~1e-3 level (nested subspaces cannot truly get worse
   with k); differences at that scale are optimizer jitter, not structure.
   The k* and identification conclusions are far above this floor and are
   seed-stable.

**Code closeout (this commit).** Removed a duplicate `completeness_kl_rows`
that shadowed its twin and two dead proposal-family drafts (the rejected
covariance NIPALS PLS and a near-duplicate head-row class) from
abstraction.py; made analysis.py's elbow verdict conditional instead of
unconditionally claiming a match; fixed compare.py's docstring
mischaracterizing CompletionPLS as cross-covariance. Regression-checked:
seed-0 rerun after cleanup reproduces the recorded outputs exactly.

**Status: CONCLUDED.**

---

## Experiment 3 (intervene.py): the interventional upgrade — PRE-REGISTRATION

Committed before the first real run of intervene.py on trained models.

**Question.** Experiment 2's claim is correlational: the 2-D PLS subspace
suffices to *decode* completions, and identifies affinely with the belief
simplex. Is it *causally load-bearing* — does writing to it move behavior
the way moving the belief state would (causal abstraction / interchange
interventions, Geiger et al.)?

**Design (stage 1, declared scope).** Patch point: final-layer residual
(pre-ln_f) at position t — the readout point all Experiment 1–2 probes used;
the "rest of the network" from there is exactly ln_f + unembedding, so the
behavioral readout is the model's own decoder, not a fitted probe. For
position-matched prefix pairs (target, source):
`r' = r_tgt + QQᵀ(r_src − r_tgt)` — the minimal-norm edit making the
subspace readout equal the source's (asserted exactly at runtime). Score:
KL(p_src_true ‖ model(r')) against the source's exact belief-conditioned
next-token distribution; `closure = (gap − transfer)/(gap − floor)` where
floor = unpatched KL to target truth, gap = unpatched KL to source truth.
Declared horizon: m = 1. A final-layer patch cannot propagate through
attention to later positions; the persistent mid-stream patch over m ≥ 2
horizons is stage 2 and overlaps roadmap item #2 (coherence under
generation). Subspaces are discovered on the Experiment-2 cache under the
same honesty constraint (completions only); fresh evaluation prefixes are
sampled independently of both training and cache data.

**Conditions.** `pls` k=2 (the claim), `pca` k=2 (Mess3: mostly
current-token identity — discriminating control), `rand` k=2
(no-information control), `comp` = orthogonal complement of pls (all
d−k = 62 dimensions: the "junk precision" claim made causal).

**Pre-registered predictions (thresholds fixed before seeing any
intervention numbers).**

- **P1** pls k=2 closure ≥ 0.90 on both processes.
- **P2** (Mess3) pca k=2 closure ≤ pls closure − 0.05.
- **P3** complement leak = (KL(p_tgt‖q_comp) − floor)/(gap − floor) ≤ 0.05.
- **P4** rand k=2 closure ≤ 0.25.

**New typed failure modes this can exhibit.**
*Correlational-but-not-causal* — subspace readout swapped exactly, behavior
stays at target (redundant coding elsewhere). *Off-manifold breakage* —
patched behavior farther than the unpatched gap from both source and target.
*Complement leak* — readout-sufficiency did not imply causal localization.

**Validation.** Self-checks run before the experiment on every invocation:
no-op patch (source = target) must reproduce the unpatched distribution
bit-for-bit; full-space patch (Q = I) must reproduce the source's unpatched
distribution exactly; the projector must realize the interchange on
alpha_pls exactly.

**Status: pre-registered; results appended below.**

---

## Experiment 3 — results

Run at commit 5927602's pre-registration (4000 position-matched pairs, 1000
fresh evaluation sequences, seed 0; self-checks passed on every invocation).
Raw outputs: `out_exp2/exp3_{mess3,z1r}.txt`.

**Scoreboard against the pre-registration: P1 FAILED, P2 FAILED (reversed),
P3 FAILED, P4 held.** This is the experiment working as designed — the
failures are typed and the diagnosis is complete.

Mess3 (floor 0.00019, gap 0.01889): closure at k=2 —

| subspace | closure | KL(p_tgt‖patched) |
|---|---|---|
| pls k=2 | 63.2% | 0.00344 |
| pca k=2 | 81.6% | 0.01539 |
| rand k=2 | 12.7% | 0.00048 |
| complement of pls (62 dims) | leak 36.5% | 0.00701 |

Z1R (floor 0.00031, gap 2.78): pls k=2 closure **0.7%**, pca k=2 **100.0%**
(PC1+PC2 carry 98.5% of variance there — nearly a full-state swap),
complement leak 100.0%, rand 0.2%.

**Finding 1: CORRELATIONAL-BUT-NOT-CAUSAL is real.** The PLS k=2 subspace —
which decodes completions at the oracle floor and identifies affinely with
the belief simplex at R² 0.99 — is *not* the channel through which the
model's own decoder reads that information. The interchange swaps the
subspace readout exactly (asserted), yet behavior moves only partially
(Mess3) or not at all (Z1R). Decode-sufficiency under standardized probes is
**scale-blind**: whitening/standardization amplifies faint copies of the
belief geometry living in low-variance directions; the model's decoder
weights directions by their raw scale and reads the high-variance encoding.
The whitened-PLS family found a faithful but largely epiphenomenal *echo*.

**Finding 2: the causal effect is additive, not broken.** In every run,
closure(pls) + leak(complement) ≈ 100% (99.7% Mess3 across k=2/3/4/8; 100.7%
Z1R): the readout responds locally linearly, and the causal effect simply
splits between the subspace and its complement. The off-manifold-breakage
verdict never fired — patched behavior always lands between target and
source. (This diagnostic is now printed by intervene.py.)

**Finding 3 (post-hoc k-sweep): the PLS echo never catches up; PCA does.**
Mess3 pls closure is *flat* in k — 63.2% (k=2), 63.9% (k=3), 64.1% (k=4),
66.3% (k=8) — its deeper components are more echo, never the channel. PCA
reaches 98.2% by k=4. Z1R pls jumps 0.7% → 11.6% → 99.3% at k=2/4/8.
Decode-relevance ordering and causal-relevance ordering are *different
orderings* of the same residual stream.

**Finding 4 (post-hoc `unemb` family): the causal channel is exactly where
the architecture says it is.** At this patch point the decoder is
softmax(W_U·ln_f(r)), so to first order it can only read
span((I−11ᵀ/d)·diag(gain)·W_Uᵀ) — at most V dims. Patching that subspace at
k = V closes **100.0% on both processes** (Mess3 k=3: KL 0.00020 vs floor
0.00019; Z1R k=2). Truncating it hurts (Mess3 k=2: 59.7%) — the channel
genuinely uses all V dims. The discovered subspaces are now judged by their
overlap with this channel: the PLS belief plane overlaps it ~63%, the
high-variance token/PCA plane more.

**Method implications (what changes going forward).**

1. *Interventional scoring must enter the discovery loop, not just the
   evaluation.* Experiment 2's protocol optimizes decode-closure; this
   experiment shows that criterion cannot distinguish a causal channel from
   its echo. The CEGAR proposal-scoring step should add an interchange term.
2. *At readout patch points the architecture hands you the causal basis for
   free* (the unembedding pullback — no fitting, available on any real LLM).
   The honest restatement of Experiment 2's HeadRowSpace failure: the *idea*
   (decoder row space) was the causally right one; the 400-step fitted
   approximation of it was too noisy. Fit budget 0 — the model's own weights
   — beats fit budget 400 here.
3. *The interesting causal-discovery question therefore lives mid-stream*
   (stage 2), where no closed-form reading basis exists and later positions
   attend to the patched state: persistent patch, multi-token horizons,
   coherence under generation (roadmap item #2).
4. *Scope note for the belief-geometry literature*: "the residual stream
   linearly embeds the belief simplex" (Experiments 1–2, Shai et al.) and
   "the model uses that embedding" are separated by exactly this experiment.
   On these toys the used encoding is the high-variance one; the
   probe-found plane is partly a copy.

**Status: CONCLUDED (stage 1). Stage 2 (mid-stream persistent interchange,
m ≥ 2 horizons, coherence under generation) is the next experiment.**
