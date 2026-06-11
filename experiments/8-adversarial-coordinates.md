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
evaluation pairs, three pooled positions, m = 3), same seeds.

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
the pullback (the oblique projector T⁻¹ Q_z Q_zᵀ T applied to prefix
differences), so the *behavioral* scoring path is unchanged. Self-checks
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
  plane.
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

**Code and results to be added only after Experiment 7 concludes.**
