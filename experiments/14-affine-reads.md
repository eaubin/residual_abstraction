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

*(Results to be appended here after the run.)*
