# Experiment 11 — patch parameterization as proposal space — PRE-REGISTRATION

**Script:** `patches.py`. **Status: pre-registered; NOT YET RUN — results
to be appended below the marked line.**

**Question.** Experiment 10 cleared interventional selection and indicted
the patch family: a candidate 1.1° from the causal plane earned +1.0%
because the working-coordinate orthogonal swap pairs every write direction
with a junk-amplified read covector. The proposal object was too small.
Here the candidate becomes a **(write direction, read covector) pair** —
the patch map is part of the search space — with generation still cheap
and fallible, selection still by measured closure gain, and acceptance
unchanged. Can the search find a *causal patch operator*, not just a
causal subspace, in hostile coordinates?

**Critical departures from the external review that prompted this design**
(recorded per house practice of not adopting suggestions wholesale):
(1) The suggested "stream-space orthogonal projector onto the pulled-back
direction" family **requires knowing T** and is therefore not constructible
by an honest z-only procedure — it is reclassified below as a *diagnostic
anchor*, evaluation-side only, alongside the known-plane patch. (2) The
suggested "low-rank u·vᵀ pairs from different source families" is subsumed
by the write-source × read-family grid and is dropped as redundant.
(3) The gradient deferral is endorsed, now with Experiment 10's sharper
reason: gradients through a fixed patch family would optimize the bad
read/write coupling.

## Design

**Setting.** The Experiment-6/8/9/10 setting unchanged (mess3-L4, L1,
prefix-wide, κ = 100, eps_gain 0.05, eps_drop 0.01, k_max 8, 400/600
disjoint pairs, basis sample 800, m = 3, seed 0 registered, anchor and
transform checks as before). Both regimes; success judged after pullback.

**Candidates: write × read grid (up to 36 per round, survival printed).**
Write directions: the Experiment-10 pool, reimplemented identically
(M1, M3, M2 under both back-mappings, delta-PCA top-2, isotropic /
covariance-shaped / precision-shaped randoms; per-round
`default_rng(seed + 1000 + round)`). Read covectors per write w, all
constructible from working-coordinate data:

| read family | covector | rationale |
|---|---|---|
| `id` | c = w | the Experiment-10 baseline (orthogonal swap); expected to fail adversarially |
| `prec` | c ∝ Σ̂⁻¹w | the GLS read — reads w's coordinate controlling for correlated junk. **Coordinate-equivariant in the ridgeless limit**: with Σ_z = TᵀΣT, the z-pair (w, Σ_z⁻¹w) pulls back to the x-pair (T⁻¹w, Σ_x⁻¹T⁻¹w). Implementation note (review fix, pre-run): each regime inverts **its own** ridge-regularized covariance (registered ridge 10⁻¹⁰·tr/d) — forming Σ̂_z⁻¹ by conjugating Σ̂_x⁻¹ would *enforce* the equivariance rather than test it; with the honest construction, any cross-regime behavioral difference of `prec` measures the ridge gap (the Experiment-9 P5 pattern). The mechanism candidate for P3. |
| `cov` | c ∝ Σ̂w | anti-rationale control: reads along amplified variance; expected bad everywhere |

All reads normalized to ⟨c, w⟩ = 1 (the interchange condition).

**Composition (new construction — ledger row per FORMALISM §7).** Accepted
pairs compose as the oblique projector P with column form
Pᵀ = W(CᵀW)⁻¹Cᵀ: the patch sets *all* accepted read-functionals to the
source's values, writing only in span(W). Assumption: CᵀW stays
well-conditioned; guard: candidates pushing cond(WᵀC) above 10⁶ are
skipped (skips printed). Falsified if legitimate candidates are
systematically skipped — would appear in the per-round tables.

**Per-candidate registered diagnostics** (the Experiment-10 style, extended
to the read side): source, read family, measured gain, write angle to the
causal plane (stream space, after pullback), and **read-junk fraction**
‖Q_jᵀ(stream read covector)‖/‖stream read covector‖ — the quantity
Experiment 10's mechanism says predicts destruction.

**Diagnostic anchors (evaluation-side, labeled, NOT discoverable).**
(a) The known-plane orthogonal patch (the Experiment-6 plane) — the
ceiling. (b) The stream-orthogonal patch of round 1's nearest-to-plane
write (requires T): quantifies "this write would have worked with a clean
read" — the direct verification of Experiment 10's read-side mechanism.

## Pre-registered predictions (each with its NOT TESTED residual, per
FORMALISM §6.1)

- **P1 (anchors).** Exp-6 loop reproduces; transform checks pass; the
  known-plane anchor patch closes ≥ 90% of full on evaluation pairs.
  (Always testable.)
- **P2 (benign).** Search converges at k\* ≤ 4 with exact closure ≥ 90% of
  full. (Always testable.)
- **P3 (adversarial headline; declared credence ~65%, resting on the
  `prec` equivariance argument).** Same bars after pullback: k\* ≤ 4,
  exact closure ≥ 90% of full. (Always testable; FAILS includes k\* = 0.)
- **P4 (observable soundness; fourth attempt).** |c_obs − exact| ≤ 0.10 on
  a non-null accepted adversarial patch; **NOT TESTED** if k\* = 0.
- **P5 (plane containment of the write subspace, adversarial).** k\* ≥ 2
  and both plane directions within 15° of span(pulled-back writes);
  **FAILS** at k\* = 1 (dimension parity); **NOT TESTED** at k\* = 0.
- **P6 (the read-side mechanism, verified directly).** Both clauses on
  round 1 of the adversarial regime: (a) the nearest-to-plane write's
  `id`-read gain < 5% (Experiment 10 reproduces), AND (b) the
  stream-orthogonal *diagnostic* patch of that same write gains ≥ 40% —
  the write was fine, the read was the problem. **NOT TESTED** if no
  round-1 write lies within 15° of the plane (pool-support failure
  instead). Declared credence ~75%.
- **P7 (validity gate, enforced).** As established.

**Trichotomy update (registered, non-overlapping per §6.1).** Adversarial
outcomes partition as: (i) some read family succeeds (P3 holds) — the
causal *patch operator* is discoverable; identify which family carried it;
(ii) P3 fails but P6 holds — clean-read patches work and none of the
registered read families realizes one honestly: the read-family menu is
too small, not the concept; (iii) P3 and P6 both fail — low-rank linear
interchange at L1 is itself too brittle under transformed coordinates
(the registered "deeper problem": linear patching as a primitive is
implicated, and the program revisits the patch construction at the
FORMALISM level). Membership is decidable from the tables in every case.

## Failure modes this can newly exhibit

*Composition fragility* — the oblique projector's conditioning guard
fires often, starving the search: the biorthogonal composition is the
wrong way to stack rank-1 interchanges. *Read-family insufficiency* —
trichotomy (ii). *Linear-primitive failure* — trichotomy (iii), the
deepest. *Equivariance gap* — `prec` behaves differently across regimes
beyond ridge tolerance: the equivariance argument has a flaw; would show
directly in the per-candidate tables.

**Self-checks** (every invocation; `--selftest` exits after the standard
four): the standard four; anchor + transform checks (real runs); loop
invariants (writes unit and orthogonal to accepted writes; ⟨c, w⟩ = 1 for
every candidate; D₀ > D_full; c_obs(full) = 1); rank-1 sanity — the
(w, id)-read candidate's patch must equal Experiment 10's orthogonal-swap
patch for the same w (regression link between the experiments).

**Enforcement.** Registered parameters, full config, seed 0, gate — as in
Experiments 8–10.

---

## Results: P1, P2, P6, P7 HOLD; P3 FAILS; P4, P5 NOT TESTED — partition branch (ii): the read **menu** is insufficient, the read **concept** is vindicated

(Registered parameters, seed 0, gate +0.0024 PASS, anchor and both
transform checks passed, no composition-guard skips. Raw output
`out/exp11_mess3-L4.txt`, figure `out/mess3-L4/experiment11.png`.)

**P6 — the read-side mechanism is now verified by direct intervention, not
arithmetic.** The same adversarial write (M2\*Sinv, 1.1° from the plane)
earns **+1.0%** under the id read and **+51.3%** under the stream-clean
diagnostic patch. Experiment 10's mechanism account was a derivation; this
is the controlled experiment: write held fixed, read varied, fifty points
of behavioral difference. Trichotomy branch (iii) — linear interchange
itself broken — is thereby **refuted**: a rank-1 linear patch transfers
half the closable gap when the read is right. The primitive is fine.

**The `prec` read did exactly half the repair.** Destruction is
*eliminated*: every catastrophic id/cov row turns harmless under prec
(same write: cov **−606.1%** → prec **+1.4%**; rand/id −187.4% → rand/prec
+2.2%), and the read-junk diagnostic predicts destruction exactly as the
mechanism says (3–24% junk under prec vs 99–100% under id/cov). But
transfer stays weak: the best adversarial candidate (M3/prec) earned
**+4.5%**, one twentieth of a point below the registered eps_gain — the
loop stopped honestly by its rule. (Proximity noted; the threshold is not
retuned post hoc.)

**Why prec under-transfers — and it is not an adversarial artifact.** The
benign table shows the same deficit with no junk in sight: on the *same
0.0° write with 0% read-junk*, id earns +54.3% and prec earns **+16.1%**.
The GLS covector's junk-safety is bought at the price of causal
alignment — controlling for correlated variance also controls away part
of the signal being read. The registered cross-regime ridge-gap
measurement is **confounded** in this run: benign and adversarial writes
from the same source are different vectors, so the prec gap (ben +31.7%
vs adv +4.5% for M3-source writes) mixes write differences with read
behavior — flagged rather than concluded; a clean equivariance isolation
needs the same pulled-back write evaluated in both regimes (folded into
the follow-up below).

**Benign regime:** unchanged from Experiment 10 — the search selected
dPCA1/id then M3/id (k\* = 2, 98.4%, plane contained at 0.3°/3.3°): where
the Euclidean read is right, interventional selection correctly prefers
it over prec. The `cov` control behaved exactly as registered
(anti-rationale: −606% is the most destructive patch in the program's
history).

**Adjudication.** P3 fails with P6 holding ⇒ registered branch **(ii)**:
clean-read patches exist and work; none of the honest read families
{id, prec, cov} realizes one in hostile coordinates. The proposal object
(write, read) is correct — Experiment 10's diagnosis stands — but the
read covector needs to join the *search*, not a fixed menu. The natural
Experiment 12: interventionally-scored read construction (e.g., a
fractional-precision grid c ∝ Σ̂^{-α}w with α selected by measured gain,
or paired read/write pools), with the same-write cross-regime equivariance
isolation included. The P6 diagnostic guarantees the target exists:
+51.3% at k = 1 from one write; composition should reach the ~98% ceiling
at k = 2 if an honest read construction gets there. P4 remains NOT TESTED
— fourth consecutive experiment — and now waits specifically on an honest
clean-read construction.

**Status: CONCLUDED.**
