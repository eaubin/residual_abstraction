# Experiment 4 — mid-stream persistent interventions and coherence — PRE-REGISTRATION

**Script:** `midstream.py`. **Status: pre-registered; results to be appended
below the marked line.**

**Provenance note.** This was trailed in
[Experiment 3](3-readout-interventions.md)'s scope declaration as "stage 2."
That label was sloppy: a scope exclusion is not a pre-registration, and a
follow-up with its own design, predictions, and failure modes is a new
experiment. It is numbered accordingly and pre-registered here, committed
before the first run of midstream.py. ("Stage 2" language has been retired.)

**Question.** Experiment 3 intervened at the readout, where the remaining
computation is a closed-form decoder and the architecture hands you the
causal basis. The open question is the one that matters for real models:
**mid-stream**, where later positions attend to the patched state and no
closed-form reading basis exists — (a) is the discovered subspace causally
load-bearing for the *downstream computation* (not just the readout)?
(b) does the intervention *persist* over autoregressive extension — the
coherence/bisimulation condition (roadmap item #2)?

## Design

**Process: Mess3 only.** The Z1R model has 1 layer — no interior stream
point; its readout point was Experiment 3. Declared, not discovered.

**Patch point.** The residual stream between block 1 and block 2 (input to
the final block), position-aligned pairs as in Experiment 3. A patch here is
*persistent*: block 2's keys/values at patched positions change, so every
later position reads it through attention.

**Amendment (pre-run, 2026-06-10, before any results existed).** The first
implementation evaluated a single fixed mid-sequence position, which is
narrower than "position-aligned intervention" as registered (flagged in
external review): with learned positional embeddings one position may be
unrepresentative. Amended design: pairs are evaluated at **three fixed
positions** spanning the usable band (t = 8, 16, 24 for the recorded
config), assigned round-robin; all P1–P6 verdicts use metrics **pooled**
across positions, and per-position closures at the longest horizon are
reported as a stability check. Thresholds unchanged. Two implementation
fixes from the same review, also pre-run: `--selftest` no longer touches
the gitignored cache (the coherence basis loads lazily), and
`train.py --cache-only` now exists so a clean checkout can regenerate
cache.npz from the tracked model.pt — making the artifact policy claim in
EXPERIMENTS.md actually true.

**Subspace discovery at the patch point.** The Experiment-2/3 subspaces live
in the final-layer stream and cannot be assumed to transfer; pls/pca/rand
k=2 bases are re-discovered at the mid-stream point on a fresh discovery
sample (per-position centering, X-whitened cross-covariance with exact m=3
completion distributions — completions-only supervision, beliefs stay
evaluation-only).

**Scopes.** `pos` — patch position t only (does a one-position edit survive
attention?); `pre` — patch all positions p ≤ t (swap the prefix's entire
subspace content).

**Horizons.** Exact m-step completion distributions for m = 1, 2, 3 via the
chain rule over all 27 continuations (one teacher-forced forward per
continuation; m=1,2 are marginals of the m=3 joint). Targets are the
source's exact belief-conditioned m-gram distributions; closure defined as
in Experiment 3, per horizon.

**Conditions.** {full (Q=I), pls k=2, pca k=2, rand k=2} × {pos, pre},
plus complement-of-pls (pre scope). Note one identity by construction:
*pre-scope full patch at m=1 equals the source's next-token behavior
exactly* (everything block 2 sees at position t is swapped), so it serves as
a self-check, not a result; its m ≥ 2 closure is a real measurement (later
positions also read the *unpatched* token embeddings through block 1 — the
shortfall from 100% measures how much completion-relevant information
bypasses the patch layer).

**Coherence (state-level).** Teacher-force the source's most likely next
token w*; compare the final-layer state at position t+1 across three runs —
patched-target, source, unpatched-target — in the Experiment-2 final-layer
PLS coordinates. Coherence holds for a pair when the patched run's state is
closer to the source run's than the unpatched run's is.

**Self-checks (every invocation).** (1) No-op patch reproduces unpatched
chain probabilities bit-for-bit. (2) Prefix-wide full swap at layer 0
(token+position embeddings) with a shared continuation reproduces the
source run's chain probabilities (validates the whole chain machinery
against a known answer). (3) Pre-scope full patch at the real patch point
matches the source's next-token distribution at m=1 (the identity above).
(4) Prefix states are independent of continuation tokens (causality of the
mask).

## Pre-registered predictions (thresholds fixed before the first run)

- **P1 (scope monotonicity).** pre-scope closure ≥ pos-scope closure − 0.02
  for every family and every horizon.
- **P2 (persistence decay).** pos-scope full-patch closure strictly
  decreases from m=1 to m=3: a one-position edit dilutes as later positions
  attend mostly elsewhere.
- **P3 (the Experiment-3 lesson persists mid-stream).** pca k=2 closure ≥
  pls k=2 closure for every scope and horizon: causal weight follows raw
  variance scale, not decode-relevance, at this patch point too.
- **P4 (control).** rand k=2 closure ≤ 0.25 everywhere.
- **P5 (layer bypass).** pre-scope full closure at m=3 lands in
  [0.80, 0.99]: below 1.00 because continuation positions re-aggregate
  prefix information through block 1 (whose inputs are unpatched token
  embeddings); above 0.80 because the block-2 stream carries most of the
  completion-relevant summary.
- **P6 (coherence).** Under the pre-scope full patch with teacher-forced
  w*, the patched run's t+1 state is closer to the source run's than the
  unpatched run's is in ≥ 90% of pairs. (Same statistic for the pls k=2
  patch reported as exploratory, no threshold.)

## Failure modes this can newly exhibit

*Attention bypass* — pos-scope patches wash out at m ≥ 2 (closure → 0):
single-position state is not where history is causally carried.
*Lower-path bypass* — pre-scope full closure ≪ 0.80 at m=3: the model
re-derives beliefs above the patch from raw token embeddings, and the
"stream as state" picture fails at this layer. *Incoherence* — behavior
transfers at m=1 but the t+1 state does not track the source (P6 fails):
the patched state is read once but not propagated as state.

---

**Results to be appended below this line after the first run.**
