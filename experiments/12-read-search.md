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

## Results: P1, P2, P6, P7 HOLD; P3 FAILS; P4, P5 NOT TESTED — branch (ii): no Σ̂-spectral read suffices; composition exonerated; the equivariance–transfer trade-off made visible

(Registered parameters, seed 0, gate +0.0024 PASS; anchor, both transform
checks, and both read-construction regression links passed. Raw output
`out/exp12_mess3-L4.txt`, figure `out/mess3-L4/experiment12.png`.)

**The same-write equivariance table — the run's central measurement**
(write held fixed: the 1.1° M2\*Sinv write and its pullback; k=1 gains):

| α | benign | adversarial |
|---|---|---|
| 0.00 | +51.3% | +1.0% |
| 0.25 | +51.4% | +1.5% |
| 0.50 | +48.1% | +1.5% |
| 0.75 | +13.3% | +1.5% |
| 1.00 | **+1.4%** | **+1.4%** |

Three conclusions read directly off it:

1. **Exp-11's confound is resolved: prec is behaviorally equivariant.** At
   α = 1 the same write earns +1.4% in both regimes, to the decimal. The
   apparent cross-regime prec gap in Experiment 11 (+31.7% vs +4.5%) was
   write-difference, as flagged — not a ridge gap. The ridge-gap worry is
   retired.
2. **Equivariance and transfer are anti-correlated across the grid in
   hostile coordinates.** α = 0 is maximally coordinate-dependent (benign
   +51.3%, adversarial +1.0%); α = 1 is equivariant and uniformly weak.
   The benign column shows pure spectrum-distortion cost (transfer
   survives to α = 0.5, collapses by α = 1 with no junk in play); the
   adversarial column is **flat at ~+1.5% across the entire grid** — the
   tradeoff has no interior optimum to find.
3. **The impossibility note's mechanism, refined by the data — measured
   and inferred parts separated (review fix).** *Measured*: the
   adversarial flatness is not simple junk-domination. On the nearest
   write (M2\*Sinv), read-junk falls to 4–7% by α ≥ 0.25 with no transfer
   gain; on the other near-plane writes junk suppression arrives only late
   (M3: 100/100/85% at α ≤ 0.75, 24% at α = 1, destruction −10% → +4.5%).
   So low read-junk demonstrably does not buy transfer. *Inferred — the
   leading hypothesis, not yet decomposed*: the missing ingredient is the
   causal read against the **neutral** background. T crushes the plane
   component of any z-covector ×κ⁻¹ while leaving the ~60 neutral
   directions untouched, and Σ̂_z^{−α} re-amplifies low-variance neutral
   reads by the inverse x-spectrum in step with plane reads, so neutral
   contamination stays competitive at every α. The output measures plane
   angle, junk fraction, and gain — it does **not** decompose the read
   covector into plane/junk/neutral fractions, so this account is
   hypothesis-grade; the follow-up registers the read-decomposition
   diagnostic that turns it into a measurement.

**P6 — composition exonerated, emphatically.** D2: the stream-clean
composition of the two nearest round-1 writes — at 1.1° *and 3.3°*, the
second being a raw random — reaches **+99.0% c_obs and 97.8% exact
closure** (ceiling 98.3%/98.7%). Two imperfect clean-read rank-1 patches
stack to the ceiling. Combined with D1 reproducing (+51.3%), every
component of the causal patch operator is now verified *except* an honest
construction of the read covector: the writes are reachable from the pool,
the patches compose, the acceptance rule selects correctly, the observable
score is sound where tested. The program's entire remaining adversarial
gap is one object: the clean read.

**Benign regime:** k\* = 2 at 98.5% exact (sources `dPCA1/a0.25`,
`M3/a0.00` — selection again by measured gain; α = 0.25 edged α = 0
benignly, consistent with the table's flat benign top). The loop's
adversarial best was again M3/a1.00 at +4.5% — the registered eps_gain
held and the threshold question did not recur.

**Adjudication: branch (ii)** — P3 fails ∧ P6 holds: spectral reads are
insufficient while clean-read composition works. Per the registration, the
sharp conclusion is that **read search needs non-spectral structure**. The
information that distinguishes the clean read is not in Σ̂ (any spectral
function re-weights the same eigenbasis); it is only in behavioral
response. Natural Experiment 13 candidates, in increasing power: (a)
paired read/write pools (the deferred expressive option — now justified,
since this experiment established *why* spectral fails); (b) per-write
read refinement — a small interventional search in read-space initialized
at the best α; (c) the long-deferred gradient, now properly scoped:
optimize the read covector alone (write fixed from the pool, behavioral
objective, initialized at best-α) — Experiments 10–12 have removed the
objections in order (selection sound; patch object right; composition
fine), so a read-only gradient no longer "optimizes the bad coupling."
P4 stands NOT TESTED after five experiments; it now waits on exactly one
construction.

**Ledger update (FORMALISM §7):** the fractional-precision row moves from
"registered, untested" to *falsified as a sufficient family* — with the
refinement that the obstacle is neutral-background read contamination, not
junk: the impossibility note's two-requirement tension is confirmed and
localized.

**Status: CONCLUDED.**
