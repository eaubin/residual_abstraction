# Experiment 14 — affine-slice read learning: settling the divergence mechanism, the repair, learned composition, and the statistical-read diagnostic — PRE-REGISTRATION

**Script:** `readaffine.py`. **Status: pre-registered; NOT YET RUN — results
to be appended below the marked line.**

**Question.** Experiment 13 left four open objects. (1) The w1 divergence
mechanism is labeled *consistent-with* renormalization feedback, not
measured — the settling diagnostics are named but not run (§6.1 rule 7
debt). (2) The affine-slice repair c = c₀ + v, v ⊥ w is registered but
untested. (3) The zero-plane working read (w2: +43.7% with plane mass 0%)
made "reads are statistical predictors, not geometric aligners" the
leading hypothesis — it needs the correlational diagnostic to become
measured. (4) P5 (learned composition) and P6b remain NOT TESTED, gated
behind w1. Experiment 14 is the discriminator for all four.

**External proposal, evaluated on its merits (working norm).** Codex
proposed framing this as a three-way discriminator (mechanism /
composition / what the zero-plane read does) rather than "repair expected
to stabilize w1". **Adopted:** the discriminator framing; the composition
test; the correlational diagnostic as a first-class prediction; the ≥ 40%
headline threshold (which coincides with what exp 13 already registered).
**Modified, one substantive point:** Codex reads "if w1 now learns under
affine-slice, the diagnosis is strongly supported." That inference is
weaker than it looks — the reparameterization removes the feedback loop
*and* changes the optimizer's geometry simultaneously, so a w1 success is
still only consistent-with. §6.1 rule 7 requires the *named settling
diagnostics*, and they are cheap: arm A re-runs the **original
renormalized parameterization** for w1, fully instrumented (deterministic
seeds — this is the recorded divergence, observed properly). The 2×2 of
(feedback signature present/absent) × (affine repairs/doesn't) is strictly
more informative than the repair alone. **Added, absent from the
proposal:** the inheritance note — this experiment deliberately stays on
the single registered T (κ = 100, junk seed 0) and the fixed write pair
(ledger rows: single-T indexing, fixed write-pair indexing). Discriminating
power first; the owed T-robustness / write-generality sweep is the natural
experiment 15, and every exp-14 conclusion remains indexed by this T and
these writes.

## Design (all hyperparameters frozen here)

**Setting.** The Experiment-6/8–13 setting unchanged: mess3-L4, patch at
L1, prefix-wide, κ = 100, 400/600 disjoint discovery/eval pairs, basis
sample 800, m = 3, seed 0, anchor + both transform checks, validity gate
enforced. Writes reproduced in-run by the registered exp-12 rule (the
recorded pair: M2\*Sinv at 1.1° = w1, raw random at 3.3° = w2). Optimizer
wherever one runs: Adam lr 0.05, 200 steps, minibatch 64, torch seed 0,
minibatch rng seed 0 — identical to exp 13. Full-objective logging every
20 steps (`log_every`, registered).

**Arm A — mechanism diagnostics (instrumented rerun, original
parameterization).** w1 only, both inits (best-α, id), the exact exp-13
optimizer including post-step renormalization c ← c/⟨c, w⟩. New
instrumentation: per step, ⟨c, w⟩ *before* renormalization and ‖c‖ after;
at steps 0, every 20, and 200: the full-discovery objective (all 400
pairs — the same quantity the optimizer subsamples) and the
plane/junk/neutral decomposition. **Registered feedback signature** (all
three required to upgrade the mechanism label to *measured*, evaluated per
run):

- (i) *ascent* — final full-discovery CE ≥ initial + 0.02 nats (the run
  climbs its own objective; settles what per-minibatch prints could not);
- (ii) *shrinkage bias* — median pre-renormalization ⟨c, w⟩ < 1 (steps
  systematically shrink the constraint, so renormalization systematically
  inflates);
- (iii) *runaway coupling* — ‖c‖ grows ≥ 10× from the init and the junk
  fraction at the last logged step exceeds the first.

**Arm B — the repair (affine slice).** Parameterize
c(u) = c₀ + (I − ŵŵᵀ)u with ŵ = w/‖w‖₂, c₀ = init/⟨init, w⟩, u
initialized at 0 — so the starting point equals exp 13's renormalized init
exactly, ⟨c, w⟩ = 1 holds by construction inside the computational graph,
and there are **no post-step operations at all** (nothing to feed back).
Adam on u, identical hyperparameters. Registered caveat (ledger row):
Adam's per-coordinate scaling makes u-space trajectories non-comparable to
exp-13's c-space trajectories; endpoints are the comparison. Runs: both
adversarial writes × both inits, plus the benign sanity arm (id init).

**Arm C — learned composition.** Best affine read per write (best final
full-discovery gain across its inits — the exp-13 stage-B rule), oblique
composition, exact closure vs full and vs the D2 anchor (recorded 97.8%).
Gated: both single-write gains ≥ 20%.

**Arm D — effective plane reading (EPR).** For a stream-space read
covector r and the write's clean stream read u (the normalized pulled-back
write), define

> EPR(r) = corr²( Δ·r , Δ·u )

over the pooled prefix rows Δ of the **held-out eval pairs** (every
position the patch touches, 0..t, all three t-groups; optimization saw
only discovery pairs). This asks "does the read compute the clean read's
functional on-distribution, regardless of where its mass sits?" — the
mass decomposition says where the norm lives, EPR says what it reads.
Scale-invariant by construction. Scored for, per write: the id read, the
best-α init, every arm-B learned read, the arm-A diverged reads (w1), and
the clean read (≡ 1 exactly — a plumbing check). Descriptive companion
(no verdict): R² of regressing Δ·r on the two plane coordinates Δ·Q_c.
Registered limitation (ledger row): EPR is on-distribution evidence only;
a high score does not promise off-distribution transfer.

## Pre-registered predictions (NOT TESTED residuals explicit)

- **P1 (anchors + no-regression; ~85%).** Exp-6 loop reproduces; both
  transform checks pass; D1 ≥ 40%; D2 exact ≥ 90% of full; the benign
  affine-optimized gain ≥ the benign id gain − 5 pts; **and** w2's best
  affine gain ≥ 20% (the known-good case survives the reparameterization).
  Always testable.
- **P2 (mechanism settle; two clauses).** (a) ~90%: both arm-A runs
  reproduce the divergence — final full-discovery gain ≤ −100%. Failure of
  (a) is a determinism breach: halt and investigate before trusting
  anything downstream. (b) gated on (a), ~60%: both diverging runs show
  the full three-part feedback signature → the mechanism label upgrades to
  **measured**; signature absent or partial → renormalization feedback is
  **demoted** (per-criterion outcome recorded; the divergence mechanism
  reopens). (b) NOT TESTED if (a) fails.
- **P3 (headline — the repair; ~55%).** w1's best affine gain ≥ 40%
  (the threshold exp-13 P3 registered: ≈ 80% of the +51.3% clean-read
  ceiling). Sub-clause **P3a (stability; ~75%)**: every adversarial affine
  run ends with full-discovery gain ≥ its init's gain − 5 pts (the repair
  at least removes the catastrophe). The most informative failure is
  P3a ∧ ¬P3 — optimization stable yet w1 still does not transfer: a
  genuine asymmetry between near-plane writes, and exp 13's
  parameterization account was incomplete.
- **P4 (observable soundness; ~80% given exp 13).** Every
  adversarially-optimized patch (arms A and B) with full-discovery gain
  ≥ 20% has |observable closure − held-out exact closure| ≤ 0.10.
  NOT TESTED if none reach 20%. The exp-13 dual reading stays in force
  (FAILS = objective hacking located, first-class).
- **P5 (learned composition; gated).** The composed best affine pair
  reaches exact closure ≥ 90% of full. NOT TESTED unless both writes'
  best gains ≥ 20%. ~50% unconditional, ~85% conditional on the gate.
- **P6 (decomposition generality; gated on w1 best affine gain ≥ 20%).**
  w1's learned read has plane fraction ≤ 20% (~60%). **Registered
  direction flip:** exp-13 P6b predicted plane ≥ 50% at 70% credence,
  written before Finding 2; the zero-plane discovery is now expected to
  generalize. The opposite outcome is equally informative — it would make
  the proxy-read phenomenon write-specific.
- **P7 (statistical-read hypothesis via EPR; gated on ≥ 1 adversarial
  affine read with gain ≥ 20%; ~70%).** Every such read has EPR ≥ 0.5 —
  the zero/low-plane read predicts the clean functional through correlated
  coordinates. **Registered falsification branch, first-class:** a working
  read (gain ≥ 20%) with EPR < 0.2 would refute the statistical-predictor
  account — transfer without computing the clean functional — and reopen
  what closure gain is measuring. Descriptive, no verdict: failed reads
  (best-α inits, diverged finals) are expected to score low.
- **P8 (validity gate, enforced).** As established.

**Interpretation map (registered).** Signature + repair + composition +
high EPR all land → the exp-13 account closes: the divergence was the
parameterization, proxy reads are an honest construction class, and the
program's adversarial gap is solved *on this T and write pair* (generality
= exp 15). Signature present but repair fails → renorm feedback real but
not the whole story; write asymmetry becomes the object. Signature absent
→ the consistent-with label was wrong; mechanism reopens with the
trajectory data in hand. Working reads with low EPR → the deepest
revision: closure gain without the clean functional.

## Failure modes this can newly exhibit

*Determinism breach* (P2a fails). *Repair-specific pathology* — the affine
slice introduces its own failure mode, caught by P1's benign/w2 clauses.
*Stable non-transfer* (P3a ∧ ¬P3) — write asymmetry. *Proxy-read
refutation* (P7's branch).

**Ledger rows (FORMALISM §7, added with this registration).**
Affine-slice parameterization: the constraint slice is searched in
u-coordinates; Adam scaling differs from exp-13's c-space, so endpoints
are comparable but trajectories are not; falsified-if known-good cases
regress (P1). EPR score: corr² on held-out eval deltas operationalizes
"computes the clean functional"; on-distribution only. **Inherited,
deliberately unchanged:** single-T indexing and fixed write-pair indexing
— every exp-14 conclusion is indexed by both (the sweep is experiment
15's candidate question).

**Self-checks** (every invocation; `--selftest` exits after them): the
standard four; anchor + both transform checks (real runs); the torch/numpy
objective regression link (rel 10⁻⁴, asserted before any optimization);
**new:** (i) affine construction — for random (w, c₀, u), the in-graph
float32 read satisfies |⟨c(u), w⟩ − 1| ≤ 10⁻⁴, and at u = 0 the read
equals the renormalized init bitwise; (ii) EPR plumbing — EPR of a read
against itself is 1 to 10⁻¹².

**Enforcement.** Registered parameters (including `log_every`), full
config, seed 0, gate — as in Experiments 8–13. Estimated runtime
~75–100 min (7 optimization runs + ~24 full-objective evaluations).

---

## Results: P1, P2a, P4, P8 HOLD; P2b FAILS; P3/P3a FAIL; P5/P6 NOT TESTED; P7 FAILS via its registered refutation branch — both exp-13 mechanism hypotheses die in one run

(Registered parameters, seed 0, gate +0.0024 PASS; anchor, transform
checks, torch/numpy regression link (rel 1.7×10⁻⁸), and the new
affine-construction and EPR plumbing checks all passed. Raw output
`out/exp14_mess3-L4.txt`.)

**Finding 1 — the renormalization-feedback mechanism is refuted, and the
refutation is *measured* (two independent ways).** P2a held exactly as
determinism predicts: the arm-A reruns land on exp-13's recorded gains to
the decimal (−462.2% / −498.1%). But the registered signature came back
(ascent True, shrinkage False, runaway False) in both runs: the median
pre-renormalization ⟨c, w⟩ is **1.0008 / 1.0012** — renormalization was
*nearly inactive* during the divergence, so it cannot have driven it. The
second, independent refutation is arm B: the affine-slice parameterization
— no renormalization, nothing to feed back, constraint exact by
construction — diverges **identically** for w1 (−500.5% / −548.2%, final
reads 100% junk, both inits). Exp-13's *consistent-with* label did
exactly its job: the named diagnostics ran and killed the hypothesis.
The affine repair itself is mechanically sound (benign +52.2% ≥ id
+51.3%; w2 endpoints +32.2%/+42.5% match exp-13's +28.6%/+43.7% within a
few points — P1 holds), but it repairs nothing, because the
parameterization was never the problem.

**Finding 2 — what the divergence actually is (measured trajectory, new
hypothesis labeled; timing claim scoped to the logging granularity —
review fix).** *Measured for the two renormalized runs* (arm-A trajectory
logs): the jump into a 100%-junk read happens within ≤ 20 steps
(full-discovery CE 3.289 → 3.411 by step 20), followed by a junk plateau
for the remaining 180 steps with CE drifting slowly *down*
(3.411 → 3.398 — the run is descending *within* the bad basin, not
feeding back). *Measured for the two affine runs* (batch CE logged only
at steps 1/50/…/200): baseline at step 1 (3.291), junk-plateau level by
step 50 (3.417 / 3.399), final reads 100% junk — the divergence happened
somewhere in steps 1–50; "≤ 20" is not established for them. Arm-B-style
trajectory logging is owed if affine runs recur. Meanwhile w2/id starts
at a *catastrophic* init (−187.4%, the raw destructive write) and Adam
**descends out of it** to +42.5% through identical machinery. So the asymmetry is in the per-write
objective landscape: w1's good basin is escaped immediately, w2's is
reachable even from a destructive start. *Hypothesis (not measured)*:
Adam's per-coordinate step normalization interacts with κ-sharpened
curvature along junk directions — the working-coordinate junk components
of c are amplified ×κ in the stream read, so the first few steps of size
~lr overshoot the narrow good basin for the nearest-plane write.
Settling diagnostics for a follow-up: lr/optimizer sweep (the cheap
discriminator), gradient-norm trajectories, basin width probes along
junk directions.

**Finding 3 — P7's registered refutation branch fired: working reads
score EPR ≈ 0.** w2's two affine-learned reads transfer +32.2% / +42.5%
(exact: +31.4% / +41.0%), yet their effective-plane-reading scores are
**0.008 / 0.007** — on the pooled held-out deltas, the learned functional
is linearly *uncorrelated* with the clean functional. The benign anchor
confirms the plumbing (learned benign read: EPR 0.976). Exp-13's
statistical-predictor hypothesis ("the read exploits echo correlations to
predict the plane coordinates") is therefore **refuted in its
pooled-linear form** — EPR was built to measure exactly that correlation
and found none. Two labeled interpretations remain. *(i) Operationalization
suspect (the ledger row's own stated assumption)*: EPR pools all prefix
rows (0..t, three t-groups); if the patch's causal effect rides a subset
of positions, per-position correlations could be substantial with the
pooled correlation diluted or sign-cancelled — position means cancel in Δ
by construction, but position-dependent covariance structure does not.
*Settling diagnostic (exp 15): per-position EPR, and EPR at the pair
position t only.* *(ii) The deep reading*: closure gain genuinely does not
require computing the clean functional — interchange transfer through a
rank-1 patch with a read that carries no (linear, pooled) plane
information. If (i) is excluded, this reopens what closure gain measures
— note P4 says observable and exact *agree* on these patches, so
whatever is transferring is real, not a scoring artifact.

**Finding 4 — no registered read-side diagnostic with a validated
threshold predicts transfer yet (review-softened from
"indistinguishable").** EPR genuinely fails to separate working from
catastrophic reads (0.008 vs 0.000 — both ≈ 0). The mass decomposition
*does* differ numerically: the working reads retain substantial neutral
mass (junk 54–69%, neutral 31–46%) while the destructive reads are 100%
junk — but that contrast rests on two runs of each kind, with no
registered threshold and no validation, so "neutral mass present" is a
*candidate* separator for a follow-up, not a diagnostic. Behavioral
measurement remains the only validated separator; a read-side observable
that predicts transfer is an explicit open object for the program.

**Finding 5 — P4 holds again (2nd consecutive, 7th experiment).** Both
patches over the 20% bar track observable-to-exact at 0.8 / 1.5 points.
Descriptively, agreement extends to the catastrophic scale (−500.5% vs
−498.2%; −462.2% vs −484.3%): oracle-free scoring is faithful across a
~550-point range of outcomes.

**Remaining verdicts.** P3a fails alongside P3 (the registered
"most informative failure" P3a ∧ ¬P3 did *not* materialize — the affine
runs are not stable-but-untransferring, they diverge outright). P5 and P6
NOT TESTED, correctly gated by w1. P8 gate passed.

**What the next registration inherits.** (1) The w1 question is now an
*optimization landscape* question: lr/optimizer sweep as the cheap
discriminator of Finding 2's hypothesis. (2) The EPR question needs the
position-resolved diagnostics before interpretation (i) vs (ii) can be
adjudicated — and the learned reads should be persisted as artifacts
(this run's reads are reproducible but were not saved; deterministic
reruns are ~15 min each). (3) Finding 4's open object: a read-side
observable that predicts transfer — with the neutral-mass contrast
(junk < 100%) as the first candidate to validate, and arm-B-style
trajectory logging wherever affine runs recur. (4) The
standing index debts are unchanged and now three deep: single-T, fixed
write pair, eps_gain staircase.

**Status: CONCLUDED.**
