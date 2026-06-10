# Experiment 3 — interchange interventions at the readout

**Script:** `intervene.py`. **Status: CONCLUDED.** Raw outputs:
`out/exp3_{mess3,z1r}.txt`, `out/exp3_mess3_k3.txt`. Pre-registered at
commit 5927602 before the first run.

**Question.** [Experiment 2](2-proposal-families.md)'s claim is
correlational: the 2-D PLS subspace suffices to *decode* completions, and
identifies affinely with the belief simplex. Is it *causally load-bearing* —
does writing to it move behavior the way moving the belief state would
(causal abstraction / interchange interventions, Geiger et al.)?

## Design (declared scope)

Patch point: final-layer residual (pre-ln_f) at position t — the readout
point all Experiment 1–2 probes used; the "rest of the network" from there
is exactly ln_f + unembedding, so the behavioral readout is the model's own
decoder, not a fitted probe. For position-matched prefix pairs (target,
source): `r' = r_tgt + QQᵀ(r_src − r_tgt)` — the minimal-norm edit making
the subspace readout equal the source's (asserted exactly at runtime).
Score: KL(p_src_true ‖ model(r')) against the source's exact
belief-conditioned next-token distribution;
`closure = (gap − transfer)/(gap − floor)` where floor = unpatched KL to
target truth, gap = unpatched KL to source truth. Declared horizon: m = 1
(a final-layer patch cannot propagate through attention to later positions;
mid-stream patches over longer horizons are
[Experiment 4](4-midstream-interventions.md)). Subspaces are discovered on
the Experiment-2 cache under the same honesty constraint (completions
only); fresh evaluation prefixes are sampled independently.

**Conditions.** `pls` k=2 (the claim), `pca` k=2 (Mess3: mostly
current-token identity — discriminating control), `rand` k=2
(no-information control), `comp` = orthogonal complement of pls (all
d−k = 62 dimensions: the "junk precision" claim made causal).

**Pre-registered predictions.**
P1 pls k=2 closure ≥ 0.90 on both processes. P2 (Mess3) pca k=2 closure ≤
pls closure − 0.05. P3 complement leak ≤ 0.05. P4 rand k=2 closure ≤ 0.25.

**Validation.** Self-checks on every invocation: no-op patch reproduces the
unpatched distribution bit-for-bit; full-space patch (Q = I) reproduces the
source's unpatched distribution; the projector realizes the interchange on
alpha_pls exactly.

## Results: P1 FAILED, P2 FAILED (reversed), P3 FAILED, P4 held

(4000 position-matched pairs, 1000 fresh evaluation sequences, seed 0;
self-checks passed.) This is the experiment working as designed — the
failures are typed and the diagnosis is complete.

Mess3 (floor 0.00019, gap 0.01889), closure at k=2:

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

**Finding 4 (post-hoc `unemb` family): the causal channel is where a
first-order reading of the architecture predicts.** At this patch point the
decoder is softmax(W_U·ln_f(r)). The `unemb` family patches
span((I−11ᵀ/d)·diag(gain)·W_Uᵀ) — the unembedding rows pulled back through
a *linearized* LayerNorm. This is an approximation, not the exact reading
basis: the true LN Jacobian is input-dependent (per-sample 1/σ scaling and
a −x̂x̂ᵀ/d term are dropped). The claim therefore rests on the empirical
validation, which is strong: patching this fixed subspace at k = V closes
**100.0% on both processes** (Mess3 k=3: KL 0.00020 vs floor 0.00019; Z1R
k=2), i.e. on these residuals the linearization captures the whole channel.
Truncating it hurts (Mess3 k=2: 59.7%) — the channel genuinely uses all V
dims. The discovered subspaces are then judged by their overlap with it:
the PLS belief plane overlaps ~63%, the high-variance token/PCA plane more.

## Method implications

1. *Interventional scoring must enter the discovery loop, not just the
   evaluation.* Experiment 2's protocol optimizes decode-closure; this
   experiment shows that criterion cannot distinguish a causal channel from
   its echo. The CEGAR proposal-scoring step should add an interchange term.
2. *At readout patch points the architecture hands you a (first-order)
   causal basis for free* (the unembedding pullback — no fitting, available
   on any real LLM). The honest restatement of Experiment 2's HeadRowSpace
   failure: the *idea* (decoder row space) was the causally right one; the
   400-step fitted approximation of it was too noisy. Fit budget 0 — the
   model's own weights — beats fit budget 400 here.
3. *The interesting causal-discovery question therefore lives mid-stream*,
   where no closed-form reading basis exists and later positions attend to
   the patched state: [Experiment 4](4-midstream-interventions.md).
4. *Scope note for the belief-geometry literature*: "the residual stream
   linearly embeds the belief simplex" (Experiments 1–2, Shai et al.) and
   "the model uses that embedding" are separated by exactly this experiment.
   On these toys the used encoding is the high-variance one; the
   probe-found plane is partly a copy.
