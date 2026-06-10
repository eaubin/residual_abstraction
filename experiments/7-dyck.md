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

**Results to be appended below this line after the first run.**
