# Experiment 7 — the Dyck/stack port: do the diagnostics transfer? — PRE-REGISTRATION

**Script:** `dyck.py` (+ `processes.dyck2`). **Status: pre-registered; NOT
YET RUN — results to be appended below the marked line.**

**Question.** Experiments 1–6 built and validated the battery on Mess3/Z1R.
This is the roadmap's "richer processes" step: port the battery to a
structurally different process — stack state, bracket matching, longer-range
constraints — and test whether the diagnostics, the typed findings (echo,
state-vs-summary, oracle-free discovery soundness), and the registered
rules transfer while the oracle is still exact.

**The registered tension.** The naive intuition says longer-range structure
⇒ more state carried in the stream. The opposing argument: bracket matching
is what attention does well from *raw tokens*, so the re-derivation bypass
that limited persistence on Mess3 could be *stronger* here, shrinking the
state region. The directional bets below (P3, and the L2/L3
characterization) put money on specific sides.

## Process (registered exactly; `processes.dyck2`)

Depth-bounded Dyck-2: vocabulary `( [ ) ]` = tokens 0–3; hidden state = the
stack (tuple of bracket types, length ≤ 3) ⇒ 15 states — still exactly an
HMM, so beliefs and m-gram completion distributions stay closed-form and
the whole pipeline reuses. Generation policy: at depth 0 open (forced),
type `(` w.p. 0.6 / `[` w.p. 0.4; at interior depths open w.p. 0.4 (same
type split), else emit the closer matching the stack top; at depth 3 close
(forced). The depth process is a periodic birth–death chain; the stationary
distribution is the standard unique left fixed point. Swapping Dyck ahead
of the roadmap's PCFG step is pragmatic (bounded Dyck is an HMM; PCFGs need
inside–outside machinery), and is hereby noted, not hidden.

## Model and protocol

4 layers, d_model 64, seq_len 32, m = 3 (V³ = 64 outcomes), trained at run
time: `python3 train.py --process dyck2 --layers 4 --outdir out/dyck2-L4`
(steps default 6000; the validity gate decides adequacy — if it fails,
retrain longer, never interpret). Evaluation protocol carried forward
unchanged: three positions t = 8/16/24 pooled with per-position stability,
600 evaluation / 400 discovery pairs (disjoint seeds), exact m = 1..3
chains, the four known-answer self-checks, registered parameters guarded in
code as in Experiment 6 (k_max raised to **12** since Dyck's belief
dimension may exceed Mess3's 2; hitting k_max remains NON-CONVERGENT, a
failure).

## Registered data-dependent rules (rules fixed now, values measured later)

- **k_B** (belief intrinsic dimension): minimal number of principal
  components of the train-split exact beliefs reaching 99% variance.
  Stage-A/C predictions are indexed to k_B rather than to a guessed number.
- **ℓ†** (the state layer, unknown a priori for Dyck): the interior layer
  maximizing step-2 incremental closure of the full prefix-wide patch
  (ties → smaller ℓ). Stage-C discovery runs at ℓ†.

## Stages

**A — calibration** (on the train.py cache): held-out affine
residual→belief R²; k_B; decode k\* for pls and pca under the Experiment-2
stopping rule. **B — depth profile** (Experiment-5 form, full/pre only):
per-step incremental closure and orthonormalized-unemb coherence at every
interior layer; ℓ† by the rule above. **C — interventional discovery**
(Experiment-6 form) at ℓ†: observable-objective CEGAR, controls
full/pca/pls/rand/emb at matched k\*, nested closure(k), per-position
stability, principal angles. Honesty as before: discovery never touches
beliefs; exact closures are evaluation-only on disjoint pairs.

## Pre-registered predictions (thresholds fixed before training or running)

- **P1 (calibration transfers).** Held-out affine residual→belief R² ≥
  0.95: the linear belief-geometry result extends to stack states.
- **P2 (decode dimension tracks the oracle).** Decode k\*(pls) ≤ k_B + 1
  under the Experiment-2 stopping rule.
- **P3 (state exists early — the directional bet).** Step-2 incremental
  closure of full/pre at L1 ≥ 50%: one block below the patch cannot
  re-derive stack state from raw tokens (one attention pass can count
  depth, but matching *nested* contents should need more).
- **P4 (profile shape).** Step-2 and step-3 incremental closures weakly
  decreasing in depth (2-point slack, as in Experiment 5).
- **P5 (the echo persists on Dyck).** The decode-supervised pls basis at
  matched k\* closes ≤ 50% of the full patch's pooled m=3 closure at ℓ†.
- **P6 (interventional discovery transfers).** The CEGAR loop converges
  (below k_max) at k\* ≤ k_B + 2 with exact-target closure ≥ 90% of the
  full patch's.
- **P7 (oracle-free soundness, the program's load-bearing claim).**
  |c_obs(k\*) − exact pooled m=3 closure(k\*)| ≤ 0.10, discovery vs
  evaluation pairs.
- **P8 (control).** rand k\* closure ≤ 25% at every horizon.
- **P9 (validity gate, enforced in code).** Gap-to-optimal ≤ 0.005 nats
  (token-weighted, 2000 fresh sequences); exit on failure.

**Characterizations (reported, no thresholds).** The Dyck profile at L2/L3
vs Mess3's (52.5% / −29.7% step-2 incremental — does the
attention-bypass argument show up as a *weaker* mid-depth profile here, and
does state interference recur?); k\* vs k_B; principal angles disc vs
pca/pls/emb (does variance mimicry recur on a stack process, or does Dyck
naturally dissociate variance from causal content — which would partially
pre-empt [Experiment 8](8-adversarial-coordinates.md)?).

## Failure modes this can newly exhibit

*Battery non-transfer* — calibration holds but the interventional stages
produce incoherent verdicts on a new process family: the diagnostics were
Mess3-tuned, not general. *Oracle-free unsoundness on richer state* — P7
fails here after holding on Mess3: model-vs-model scoring degrades as the
completion space grows (64 outcomes vs 27). *Synchronization starvation* —
Dyck beliefs synchronize slowly from stationary starts; if k_B is large and
decode/discovery k\* both chase it, burn-in = 4 may be inadequate (would
show as position-dependent closures; the stability table catches it).

---

## Results: P3–P9 HOLD, P1–P2 FAIL — the calibration breaks, the interventional battery doesn't

(Model trained into `out/dyck2-L4`; gate −0.0121 PASS — the negative sign
is sampling noise in train.py's 400-sequence optimal-NLL probe, a known
estimator artifact also seen on Z1R; the model is behaviorally
near-optimal, unpatched next-token KL to truth 0.0007 nats. Raw output
`out/exp7_dyck2-L4.txt`, training log `out/dyck2-L4-train.txt`, figure
`out/dyck2-L4/experiment7.png`.)

**The headline: a dissociation the program predicted in principle but had
never observed.** Stage A — the Shai-et-al-style calibration that anchored
Experiments 1–2 — fails on Dyck: held-out affine residual→belief R² is
**0.66** (P1), and decode k\* under the Experiment-2 stopping rule exceeds
12 for *both* pls and pca (P2) — not a proposal-family artifact this time;
the 13-dimensional sufficient statistic (k_B = 13) is simply not
linearly/low-dimensionally present in the residual of a model whose
*behavior* is essentially exact. Meanwhile stages B–C transfer completely:
the stream is state at L1 (step-2/3 incremental closure 88.1%/85.2%, P3),
the profile decays monotonically (P4), and the interventional CEGAR loop
converges at **k\* = 4** with exact-target closure **92.6% vs the full
64-dim patch's 93.6%** (P6), oracle-free score agreeing with exact
evaluation to 5.9 points (P7). Decode probing asks "is the oracle's
statistic linearly there?" — on Dyck, no. Interventional probing asks
"what does the model actually route and use?" — a 4-dimensional core that
carries, in KL terms, essentially everything that matters.

**A wrong hypothesis, caught by its own check (recorded per repo norms).**
The tempting explanation — k\* = 4 reflects the m=3 horizon needing less
than the full belief — is *false*: post-hoc, the exact m=3 completion
distributions are also 13-dimensional at 99% variance and the belief→mgram
map is full-rank (G = B·M fits with R² = 1.000, rank 15). The truncated
horizon does not collapse the statistic. What k\* = 4 measures is
KL-weighting plus the model's own routing: the distinctions beyond the
4-dim core either carry negligible completion KL or bypass L1 entirely
(the full patch itself only closes 93.6% — more lower-path bypass than
Mess3's 98.7%, consistent with the registered attention-bypass argument).

Depth profile (full/pre):

| layer | closure m=1/2/3 | incr step2/step3 | coherence |
|---|---|---|---|
| L1 | 100.0 / 96.1 / 93.6% | 88.1% / 85.2% | 85.7% |
| L2 | 100.0 / 80.1 / 71.6% | 39.0% / 42.4% | 66.8% |
| L3 | 100.0 / 73.8 / 62.7% | 19.6% / 25.1% | 59.0% |

**Characterizations.** (1) *The L2/L3 comparison*: Dyck sits below Mess3 at
L2 (39.0% vs 52.5% — more re-derivation through the lower path) but above
at L3 (19.6% vs −29.7%): **no state interference on Dyck** — Mess3's
negative closures are not universal. (2) *Variance mimicry recurs*
(discovered-vs-pca principal angles 0.8–8.6°; pca k=4 ≈ full): Dyck does
**not** naturally dissociate variance from causal content, so
[Experiment 8](8-adversarial-coordinates.md) remains necessary. (3) *The
echo is even more extreme here*: pls k=4 closes **0.2%** (vs full's 93.6%),
at 83–89° from the discovered subspace — decode-relevance ordering is
anti-causal on stack processes too (P5; the scale lesson is now 6-for-6).
(4) *The input-side architecture basis is strong on Dyck*: emb closes
83.9% (vs ~70% on Mess3) and shares ~3 of 4 directions with the discovered
basis — much of the causally-routed core is current-token/top-of-stack
content; the fourth discovered direction (83° from the emb span) carries
the aggregated-history remainder.

**Typed outcome for the taxonomy: REPRESENTATION–ORACLE MISMATCH.** A model
can be behaviorally sufficient without linearly embedding the oracle's
minimal sufficient statistic; calibration-stage failure (P1/P2-style) then
says nothing about the interventional stages, which measure the model's
*own* state. Method implication, sharpened by this run: on systems where
ground truth is unavailable (the LLM phase), the decode-calibration stage
was always going to be impossible — this run shows the battery's causal
core stands without it, and that its oracle-free scoring stayed sound
(P7) on a process where the oracle-aligned calibration broke.

**Status: CONCLUDED.**
