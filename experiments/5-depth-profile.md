# Experiment 5 — a depth profile of patch persistence — PRE-REGISTRATION

**Script:** `depth.py`. **Status: pre-registered; NOT YET RUN — results to
be appended below the marked line.**

**Question.** Experiment 4 found that at the only interior point of a
2-layer model, the stream is a per-position *summary*, not propagated
*state*: a persistent full prefix-wide patch carried 12.5% / 0.0% of the
new information at steps 2 / 3, because future positions re-derive their
predictive state from raw token embeddings *below* the patch. That leaves
the obvious confound: with one block below the patch, "below the patch" is
almost the whole belief-synthesis pathway. On a deeper model the question
becomes a *profile*: *at which depth, if any, does the stream start to
function as state* — i.e. where is patched content carried forward because
the re-derivation pathway below the patch is too shallow to bypass it?

## Design

**Model.** Mess3, 4 layers, otherwise the recorded configuration
(d_model 64, seq_len 32, defaults): trained at run time into
`out/mess3-L4` via `python3 train.py --process mess3 --layers 4 --outdir
out/mess3-L4`. Training is part of the run, not of this registration; its
adequacy is gated (P5), not assumed.

**Patch points.** Input to block ℓ+1 for every interior ℓ ∈ {1, 2, 3}
(ℓ = 0, the embedding stream, is the exact-identity anchor — Experiment 4's
self-check #2 — and is not a measurement). Prefix-wide (`pre`) scope only:
Experiment 4 already established the pos/pre comparison; the profile
question is about depth.

**Conditions per layer.** full (identity), pls k=2, pca k=2, rand k=2 —
pls/pca/rand re-discovered *at each patch point* (same protocol and honesty
constraint as Experiment 4: per-position centering, X-whitened
cross-covariance with exact m=3 completions, completions-only supervision).

**Metrics.** Carried forward from Experiment 4 unchanged: three fixed
positions (t = 8/16/24) round-robin, pooled metrics with per-position
stability; exact m = 1..3 horizons by chain rule over all 27 continuations.
**Headline statistic: per-step incremental closure of full/pre** (the
Experiment-4 lesson — pooled multi-token closures inherit the ≈100% first
step). Coherence at t+1 (teacher-forced w*, full/pre, per layer) in the
**orthonormalized unemb-pullback coordinates as the registered basis**
(lesson from the Experiment-4 review: plain subspace distance; the
logit-weighted pullback and final-layer pls coordinates are reported as
secondary).

**Self-checks.** The four Experiment-4 checks, run at ℓ = 1: no-op bitwise;
layer-0 prefix swap reproduces the source run; pre-scope full patch m=1
matches the source's next-token distribution; causal-mask sanity.
`--selftest` exits after them and is runnable against the existing 2-layer
model (machinery validation only).

## Pre-registered predictions (thresholds fixed before training or running)

"Weakly decreasing" below means: never increases by more than 2 percentage
points from one layer to the next (declared slack for estimation noise).

- **P1 (profile direction).** Step-2 and step-3 incremental closures of
  full/pre are weakly decreasing in patch depth ℓ: the deeper the patch,
  the more re-derivation capacity sits below it.
- **P2 (state exists somewhere — the risky one).** At ℓ = 1, step-2
  incremental closure ≥ 50%: a single block below the patch is NOT enough
  to re-derive the belief state from raw tokens. If this fails low, the
  typed conclusion is *never-state*: on this process the stream at every
  interior depth is re-derivable summary, and coherence-as-state fails
  architecture-wide, not just at the last layer.
- **P3 (the scale lesson at every depth).** Pooled m=3 closure of pca k=2
  ≥ pls k=2 at every ℓ.
- **P4 (control).** rand k=2 pooled closure ≤ 25% at every ℓ and m.
- **P5 (validity gate, checked by depth.py itself).** The trained 4-layer
  model's NLL/token on a fresh sample of 2000 sequences is within 0.005
  nats of the exact entropy rate. (Sample size declared pre-run after the
  selftest showed a 400-sequence estimator has noise comparable to the
  threshold: the well-trained 2-layer model, train-time gap +0.0016, read
  +0.0050 on 400 sequences.) If the gate fails, the run is reported as
  invalid and the model retrained longer — results from an undertrained
  model are not interpreted. **Enforced in code** (review fix, pre-run):
  depth.py exits on a failed gate, and likewise refuses to produce
  Experiment-5 verdicts for any model other than the registered
  mess3/4-layer configuration; `--force-invalid` overrides both for
  explicitly exploratory runs only.
- **P6 (coherence tracks the profile).** The coherence fraction (full/pre,
  registered basis) is weakly decreasing in ℓ.

## Failure modes this can newly exhibit

*Never-state* — P2 fails low at every ℓ: no interior depth carries
necessary state; the "residual stream as state" picture fails globally on
this process/architecture. *Knee* — incremental closure high for ℓ ≤ ℓ*
then collapsing: a sharp depth at which the summary/state character flips
(the interesting positive outcome; localizes where belief synthesis
completes). *Non-monotonicity* — closure rising with depth would violate
the re-derivation account outright: suspect the harness before the science,
then look for attention routing that skips layers.

---

## Results: P1–P6 ALL HOLD — early layers are state, late layers are summary, and late patches *corrupt*

(4-layer Mess3 model trained into `out/mess3-L4`, validity gate
+0.0024 nats PASS; 600 pairs at t ∈ {8, 16, 24}, seed 0, self-checks
passed. Raw output `out/exp5_mess3-L4.txt`, figure
`out/mess3-L4/experiment5.png`. Patch layer Lℓ = input to block ℓ+1.)

The headline profile — per-step incremental closure of the full prefix-wide
patch, with the coherence fraction (registered orthonormalized-unemb basis)
alongside:

| patch | step 2 | step 3 | coherence at t+1 |
|---|---|---|---|
| L1 (input to block 2) | 93.7% | 91.0% | 94.8% |
| L2 (input to block 3) | 52.5% | 12.4% | 77.0% |
| L3 (input to block 4) | **−29.7%** | **−83.7%** | 26.8% |

**Finding 1 (P1, P2, P6 hold): the stream functions as state early and the
profile localizes where that stops.** At L1 the patched content persists
almost fully — one block below the patch cannot re-derive the belief state
from raw tokens, so future positions genuinely read the patched stream;
*never-state is refuted*. Persistence then falls monotonically: roughly
half the step-2 information survives a patch at L2, almost none of step-3;
by L3 the stream is re-derivable summary. The knee sits between L1 and L3:
belief synthesis is effectively complete (and bypassable) once two blocks
sit below the patch. Coherence tracks the same profile in all three bases
(registered and both secondaries agree within ~5 points everywhere).

**Finding 2 (beyond the registered failure modes): late patches don't just
fade — they actively corrupt. A new typed failure: STATE INTERFERENCE.**
At L3 the incremental closures are negative: the patched run predicts the
source's steps 2–3 *worse than the unpatched target run does*. Mechanism:
position t+j's own state is target-derived (blocks 1–3 see the raw target
prefix), while block 4's attention reads source-provenance states at
positions ≤ t — a mixed-provenance state that is less predictive than
either coherent run. The pre-registration anticipated never-state, a knee,
and non-monotonicity; monotone-decreasing-through-zero was not anticipated
and is the run's genuinely new observation. Arithmetic verified against
the pooled numbers (pooled m=2 closure 76.8% at L3 decomposes to exactly
−29.7% incremental).

**Finding 3 (P3, P4 hold): the scale lesson is now 5-for-5 across
experiments and depths.** The X-whitened PLS k=2 plane is causally near
empty at every depth (3.3% at L1, 8.7% at L2, 44.9% at L3, pooled m=1)
while PCA k=2 ≈ full (99.2% / 95.7% / 89.3%). At L1 this is especially
stark: the completion-supervised family finds a 3%-causal echo in a stream
whose top-2 *variance* plane carries essentially everything. Decode-
relevance ordering is reliably anti-causal on these models.

**Method implications.**
1. *"Residual stream as state" is a per-layer property with a measurable
   profile, not a property of the model.* Coherence-as-state certificates
   must name the layer; the per-step incremental closure is the right
   statistic and the depth profile is cheap.
2. *Interventions at late layers need an interference control.* Negative
   incremental closure means a patch can degrade behavior without any
   off-manifold breakage at step 1 — mixed-provenance state is a failure
   mode that single-step interchange scores (Experiment 3, and most
   activation-patching practice) cannot see at all.
3. *The discovery-family gap is now the program's central open problem*:
   every completions-supervised proposal family tested finds echoes, while
   the causal channel follows raw variance/architecture structure. The
   natural Experiment 6 candidates, in roadmap order: (a) make interchange
   closure itself the proposal-scoring objective in the CEGAR loop
   (interventional discovery, the exp-3 implication, now with L1 as the
   right patch point and a known ~94% ceiling for full-space transfer);
   (b) port the depth profile to a process with longer-range structure
   (Dyck/stack, per the roadmap), where re-derivation from raw tokens
   should be harder and the state region should extend deeper.

**Status: CONCLUDED.**
