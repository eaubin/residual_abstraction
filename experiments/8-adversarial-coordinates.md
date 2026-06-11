# Experiment 8 — adversarial coordinates: the variance-mimicry discriminator — PRE-REGISTRATION (design only)

**Script:** `adversarial.py` — written *after* Experiment 7 concluded, per
the registered ordering (the design below was frozen at commit 58abc69,
before any Dyck result existed, and is unchanged by the implementation).
**Status: pre-registered, coded, NOT RUN.**

**Implementation notes (post-Dyck, design unchanged).** (1) Experiment 6's
causal plane is not a stored artifact; adversarial.py reproduces it in-run
by re-executing the Experiment-6 loop deterministically and *asserting* the
recorded fixed point (k\* = 2, c_obs ≈ 99.8%) before constructing T — the
anchor doubles as a reproduction check. (2) T is symmetric with the
closed-form inverse T⁻¹ = I + (κ−1)P_c + (1/κ−1)P_j, so no numerical
inversion enters the pullback. (3) A mechanical consequence of the frozen
loop spec, noted in the code before running: the *proposal* step
(weighted second-moment eigenvector) is variance-driven even though
*acceptance* is behavioral, so in hostile coordinates the registered
failure mode "variance dependence exposed" may manifest as the loop
stopping at k\* = 0 (first junk proposal earns no behavioral gain); the
code handles that path explicitly and types it. (4) The two registered
transform checks need the reproduced plane and therefore run in real runs;
`--selftest` covers the standard four machinery checks.

**Question.** Experiment 6's declared limitation: the interventional CEGAR
loop converged onto PCA's top-2 plane (principal angles 3.3°/3.6°), so on
that model "interventional discovery works" and "variance was right anyway"
are indistinguishable. This experiment constructs the regime where they
must come apart: discovery performed in **adversarially ill-conditioned
coordinates** of the same stream, where variance actively misleads. PCA
must fail by construction; the interventional loop's objective is purely
behavioral and must not care.

**Why analysis-side coordinates rather than a modified model (worked out at
registration).** The surgical version — re-gauging the network so the
causal plane has low variance while behavior is unchanged — is *not exactly
achievable* in a pre-LN transformer: orthogonal gauges commute with
LayerNorm but preserve the variance spectrum (PCA still wins); the
non-orthogonal rescalings that bury variance break LN's per-sample
normalization and cannot be folded into static weights. The exact and
honest alternative is the analysis-side transform, in the tradition of
Experiment 2's buried-belief validation cache: all *discovery* operates on
z = T·(stream at L1), and every patch is pulled back through T⁻¹ before it
touches the model. The model and its behavior are bit-identical to
Experiment 6; only the coordinate system handed to the discovery procedures
is hostile.

**Scope honesty, stated up front.** This is a *constructed validation of
the method*, not natural-model evidence: T is built using Experiment 6's
known answer (that is what makes the test sharp, exactly as the buried
cache used a known belief plane). Passing it licenses "the interventional
loop's success does not depend on variance being benign," not "real models
bury their causal content." Natural evidence, if it exists, would come from
a process that dissociates variance from causal content on its own —
Experiment 7's principal-angle characterization is registered to look for
precisely that, and may partially pre-empt this experiment.

## Design (frozen)

**Setting.** The concluded Experiment-6 setting, unchanged: `out/mess3-L4`,
patch point L1, prefix-wide scope, registered protocol parameters
(eps_gain 0.05, eps_drop 0.01, k_max 8, 400/600 disjoint discovery/
evaluation pairs, three pooled positions, m = 3), same seeds. Unlike
Experiments 5–7, **the seed is part of the registration, not exempt as a
robustness knob** (pre-run clarification from review): T is constructed
from the seed-0 reproduction of the Experiment-6 plane, so a different
seed changes the transform itself — adversarial.py refuses nonzero seeds
without `--force-invalid` and labels such runs exploratory.

**The transform (registered construction rule).**
T = I − (1 − 1/κ)·P_c + (κ − 1)·P_j, with κ = 100, where P_c is the
orthogonal projector onto Experiment 6's discovered causal plane (k\*=2,
committed at 61cf1f8) and P_j the projector onto 2 fixed random directions
drawn orthogonal to it (seed 0). In z = T·stream coordinates the causal
plane's variance is suppressed ×10⁴ and junk variance amplified ×10⁴:
PCA-on-z's top components are junk by construction. T is invertible and
fixed before any run.

**Procedure.** Identical to Experiment 6 with one substitution: every
proposal family (pca, pls, rand, and the CEGAR loop's mined directions)
sees only z; a discovered subspace Q_z induces the stream-space patch via
the oblique pullback projector — T⁻¹ Q_z Q_zᵀ T acting on column vectors,
equivalently Δ ↦ Δ · (T Q_z Q_zᵀ T⁻¹) in the codebase's row-vector
convention where z = x·T with T symmetric (the two formulas are the same
map; stated in both conventions since the first draft used only the
column form and the code the row form) — so the *behavioral* scoring path
is unchanged. Self-checks
carried over, plus two new known-answer checks: (i) the pullback of the
full z-space (Q_z = I) must reproduce the full-space patch exactly;
(ii) the pullback of T·(exp-6 plane) must reproduce Experiment 6's
discovered-plane closure to float tolerance.

## Pre-registered predictions

- **P1 (PCA must fail).** pca-on-z at k = 2: exact-target pooled m=3
  closure ≤ 25%.
- **P2 (the loop must not care).** The CEGAR loop on z converges (below
  k_max) at k\* ≤ 4 with exact closure ≥ 90% of the full patch's — the
  Experiment-6 success bar, now with variance hostile.
- **P3 (it finds the same thing).** The pullback of the discovered z-plane
  is within 15° (largest principal angle) of Experiment 6's discovered
  plane, **and k\* = 2** — "the same thing" includes the dimension; with
  k\* > 2 principal angles only test containment and junk dimensions could
  ride along. (Pre-run clarification from review; a symmetric
  projection-distance is reported as characterization.)
- **P4 (oracle-free soundness, again).** |c_obs − exact closure| ≤ 0.10.
- **P5 (controls).** rand-on-z k=2 ≤ 25%; the validity gate and registered
  parameters enforced as in Experiments 5–7.

**Failure modes.** *Variance dependence exposed* — P2 fails: the loop's
proposal step (weighted second-moment eigenvector) is itself secretly
variance-driven, and the mined directions chase amplified junk; this would
falsify the method claim that Experiment 6 could not test. *Pullback
pathology* — the oblique projector's conditioning (κ² = 10⁴) amplifies
numerical error enough to corrupt patches; caught by new self-check (i).

---

## Results: P1, P4, P5 HOLD; P2, P3 FAIL — variance dependence exposed

(Registered parameters, seed 0, gate +0.0024 PASS; anchor reproduced
Experiment 6's loop exactly — k\*=2, c_obs 99.8% — before T was built;
transform checks passed; hostility confirmed: causal-plane variance share
in z = 4.8×10⁻⁵, pca-z top-2 at 89.9° from the causal plane. Raw output
`out/exp8_mess3-L4.txt`, figure `out/mess3-L4/experiment8.png`.)

**The headline: the registered failure mode fired, exactly as the
implementation notes anticipated.** In hostile coordinates the CEGAR loop's
first mined direction was amplified junk; the behavioral acceptance rule
correctly gave it no credit (gain −1.4% < eps_gain) and the loop stopped at
**k\* = 0**. P2 and P3 fail. The diagnosis is clean and splits the method
claim in two:

- *Scoring/acceptance is behavioral and sound.* The loop never accepted a
  junk direction, and P4 holds in the way that matters: the observable
  score reported the failure honestly (c_obs 0.0% vs exact 0.0%) — no
  false confidence. The same soundness that validated Experiments 6–7's
  successes correctly validated this failure.
- *Proposal generation is variance-dependent.* `mined_direction` is a
  weighted second-moment eigenvector — covariance machinery — and in z the
  mining matrix is junk-dominated by ~κ⁴. **Experiments 6 and 7's
  discovery successes leaned on benign variance for their proposals**,
  even though their acceptance never did. This is the fact Experiment 6
  could not test about itself, now established.

There is a pleasing symmetry with Experiment 2: "covariance inherits
variance" was exactly the lesson that killed the unwhitened PLS proposal
family there, and the CEGAR miner has now failed the same way at the
causal level. The natural repair is the same one — make the miner
scale-free (whiten the prefix differences before the eigenvector step, or
mine by weighted correlation with the behavioral divergence) — and is the
obvious candidate for the next registration, not smuggled into this one.

**Evaluation (exact targets, pooled m=3):** full 98.7%, exp-6 plane
98.3% (the known answer, for reference), pca-z **7.0%** (P1 holds — the
coordinates genuinely defeat variance ordering), disc-z 0.0% (k\* = 0),
and two surprises:

**Surprise 1: pls-z closes 85%.** X-whitened PLS — the 6-for-6 *echo*
family, 2.7% in natural coordinates on this very model — becomes
substantially causal in hostile coordinates. Hypothesis (unverified,
flagged for follow-up): whitening makes the family nearly invariant to
invertible linear maps *except through its ridge term*, which is scaled by
the largest singular value; T rescales the spectrum by ×κ/÷κ, moving which
small-variance directions survive the ridge floor — in x that selection
favored the echo, in z it favors the (now low-variance) causal plane.
Whatever the mechanism, it underlines the running theme: these proposal
families select by *scale-sensitive* criteria, and what they find is an
accident of the coordinate system.

**Surprise 2: rand-z is destructive, not uninformative.** A random 2-dim
z-subspace pulls back to closure **−427%** — behavior far worse than not
patching. The oblique pullback of a generic z-direction has components
amplified ×κ, so the patch is norm-uncontrolled and throws the stream off
the reachable manifold. P5 technically holds (the registered bound was
one-sided), but its *meaning* changed: in ill-conditioned coordinate
regimes the random control stops being a no-information baseline and
becomes an off-manifold-damage probe. New typed observation for the
taxonomy: *pullback off-manifold amplification*.

**What Experiment 8 licenses.** The constructed-validation goal is met,
with the polarity reversed from hope: the discrimination "interventional
discovery works" vs "variance was right anyway" is now resolved as —
*acceptance works; proposals were variance-lucky*. Method status after
experiments 6–8: behavioral interchange scoring is validated as an
acceptance criterion in three regimes (benign, new-process, adversarial);
the proposal miner is falsified outside benign variance and needs the
scale-free repair before the LLM phase, where coordinate conditioning is
not under our control.

**Status: CONCLUDED.**
