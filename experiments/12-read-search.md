# Experiment 12 — fractional-precision read search — PRE-REGISTRATION

**Script:** `reads.py`. **Status: pre-registered; NOT YET RUN — results to
be appended below the marked line.**

**Question.** Experiment 11 established that the (write, read) proposal
object is right, that a clean-read rank-1 patch transfers half the closable
gap from a single pool write (+51.3% vs +1.0%, write held fixed), and that
the fixed read menu {id, prec, cov} cannot realize a clean read honestly.
Does the read covector need to be *searched* rather than *constructed* —
and does the simplest one-parameter search space, fractional precision
powers, contain a good-enough point?

**Provenance.** Primary design per external review (fractional grid as the
single sharp hypothesis test; paired read/write pools deferred as
expressive-but-undiagnostic; same-write equivariance isolation registered
as a diagnostic table). Adopted with the additions below, which change how
the outcome must be interpreted.

## The impossibility note (registered up front so success cannot be
over-read)

The clean z-space read for a write w is the pullback of the
stream-Euclidean read: with z = x·T (T symmetric), the stream write is
u = T⁻¹w, the clean stream read is u itself, and its z-covector is
c_clean ∝ T⁻²w. The grid offers c_α ∝ Σ̂_z^{−α}w with Σ_z = TΣ_xT. On T's
eigenspaces: matching T⁻²'s κ-dependence (suppress junk reads by κ⁻²,
amplify plane reads by κ²) requires **α = 1** — but α = 1 also injects the
x-spectrum weighting Σ_x⁻¹, which is exactly the distortion Experiment 11
*measured* benignly (prec +16.1% vs id +54.3% on the same 0° write: GLS
controls away signal). α = 0 avoids the spectrum distortion but forfeits
the κ-cancellation entirely (Experiment 10's failure). The two
requirements pull on the same exponent, so **no α equals the clean read
unless the x-spectrum is flat on the relevant directions** — which it is
not (the causal plane is top-variance benignly). Therefore:

- If some intermediate α succeeds, the finding is "the α-tradeoff has a
  good-enough point at this κ and this spectrum" — an honest, z-only,
  one-parameter repair, but *not* a construction of the clean read, and
  its κ-robustness is an open question the characterization sweep probes.
- If all α fail, the conclusion is sharp: no Σ̂-spectral read suffices,
  and read search needs non-spectral structure (per-write interventional
  read refinement, or richer two-parameter mixtures — the registered
  follow-up direction, not smuggled in here).

## Design

**Setting.** The Experiment-6/8–11 setting unchanged (mess3-L4, L1,
prefix-wide, κ = 100, k_max 8, eps_drop 0.01, 400/600 disjoint pairs,
basis sample 800, m = 3, seed 0, anchor + both transform checks).
**eps_gain stays 0.05, restated explicitly**: Experiment 11's best
candidate missed it by 0.5 points, which is precisely the tuning
temptation a registration exists to block. The threshold does not move.

**Candidates: write pool × α-grid (up to 60 per round, survival printed).**
Writes: the Experiment-10/11 pool (now the module-level
`patches.write_pool`, imported — no third copy). Reads per write:
c_α ∝ Σ̂^{−α}w for **α ∈ {0, 0.25, 0.5, 0.75, 1}** (registered grid;
α = 0 is the id read, α = 1 is Experiment 11's prec — the grid nests both
failures), normalized ⟨c, w⟩ = 1; matrix powers by eigendecomposition with
the registered floor 10⁻¹⁰·λ_max. Composition, conditioning guard
(cond(WᵀC) ≤ 10⁶), per-candidate diagnostics (gain, write angle,
read-junk) all carried from Experiment 11. Ledger row (FORMALISM §7):
fractional powers assume the floored eigendecomposition of the *sample*
covariance; the impossibility note above is the registered interpretation
constraint.

**Diagnostic anchors (evaluation-side, labeled, not discoverable).**
- *Ceiling*: the known-plane patch (as before).
- *D1*: the stream-clean rank-1 patch of round 1's nearest-to-plane
  adversarial write — re-verifies Experiment 11's P6 (+51.3% there).
- *D2 (new — closes the composition escape hatch)*: the stream-orthogonal
  patch of the span of the **two** nearest-to-plane distinct round-1
  adversarial writes (both within 15°). If clean reads work at k = 1 but
  their composition cannot approach the ceiling at k = 2, composition —
  not read construction — becomes the suspect; one cheap evaluation
  settles it inside this run instead of spawning a follow-up.

**Same-write equivariance table (the Experiment-11 confound, fixed).**
Take round 1's nearest-to-plane adversarial write w_z and its pullback
w_x = unit(T⁻¹w_z). Evaluate the full α-grid for w_x in benign coordinates
and for w_z in adversarial coordinates, side by side — the same underlying
write in both regimes, so per-α differences now measure read behavior
(including the ridge gap at α = 1) rather than write differences.
Diagnostic table, no threshold.

## Pre-registered predictions (NOT TESTED residuals explicit, per
FORMALISM §6.1)

- **P1 (anchors).** Exp-6 loop reproduces; both transform checks pass;
  ceiling ≥ 90% of full; **D1 ≥ 40%** (Experiment 11's P6b reproduces).
  Always testable.
- **P2 (benign).** k\* ≤ 4 with exact closure ≥ 90% of full (the grid
  nests α = 0, which succeeded in Experiments 10–11). Always testable.
- **P3 (adversarial headline; declared credence ~50% — the impossibility
  note is why it is not higher).** k\* ≤ 4 with exact closure ≥ 90% of
  full, after pullback. FAILS includes k\* = 0.
- **P4 (observable soundness; fifth attempt).** |c_obs − exact| ≤ 0.10 on
  a non-null accepted adversarial patch; **NOT TESTED** if k\* = 0.
- **P5 (plane containment in writes).** k\* ≥ 2 and both plane directions
  within 15°; **FAILS** at k\* = 1; **NOT TESTED** at k\* = 0.
- **P6 (D2 composition).** Exact closure of the D2 anchor ≥ 90% of full;
  **NOT TESTED** if fewer than two distinct round-1 adversarial writes lie
  within 15° of the plane. Declared credence ~80% (the plane patch itself
  composes — D2 differs only by write imperfection).
- **P7 (validity gate, enforced).** As established.

**Adjudication partition (non-overlapping, decidable from the output).**
(i) P3 holds → a good-enough α exists; report which α values were
accepted and their read-junk; interpretation bounded by the impossibility
note. (ii) P3 fails ∧ P6 holds → spectral reads are insufficient while
clean-read composition works: read search needs non-spectral structure
(the registered follow-up). (iii) P3 fails ∧ P6 fails → composition is
implicated after all; if D1 *also* fails (Experiment 11's +51.3% does not
reproduce), suspect the harness before the science.

## Failure modes this can newly exhibit

*Grid-resolution artifact* — the good α lies between grid points; visible
as a unimodal gain-vs-α pattern peaking strictly inside an interval, and
addressable by a registered finer grid in a follow-up, not by post-hoc
interpolation here. *Per-write α heterogeneity* — different writes peak at
different α (the spectrum seen along each write differs); visible in the
per-candidate tables; would motivate per-write read refinement.
*Composition fragility under oblique stacking* — accepted α-pairs
interact: coarsen-pass drops or conditioning skips concentrated in later
rounds.

**Self-checks** (every invocation; `--selftest` exits after the standard
four): the standard four; anchor + both transform checks (real runs); loop
invariants (⟨c, w⟩ = 1 per candidate; writes unit and residualized;
D₀ > D_full; c_obs(full) = 1); α = 0 must reproduce the id read exactly
(matrix-power identity, asserted); the rank-1 α-grid patch at α = 1 must
equal Experiment 11's prec patch for the same write (regression link,
asserted).

**Enforcement.** Registered parameters (including the α-grid), full
config, seed 0, gate — as in Experiments 8–11.

---

**Results to be appended below this line after the first run.**
