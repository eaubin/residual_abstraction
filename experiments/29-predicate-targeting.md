# Experiment 29 — Predicate-targeting diagnostic (ORIGINAL_SIN reduction #3) — PRE-REGISTRATION

**Script:** `scripts/predicates/predicate_targeting.py`; library module
`predicates.py`.

**Status: pre-registered; NOT YET RUN. Pause here for review before the
canonical run.**

## Question

`ORIGINAL_SIN.md` names the first reduction to attack as **#3**: the
completion side currently has no named lattice — every claim is about the
full finite-horizon distribution scored by one KL/closure number. A
*within-horizon predicate* is a Boolean `φ: V^m → {0,1}` (a fixed mask over
the `216` continuations); its truth-probability `p_φ = Σ_c φ(c)·q(c)` is one
behavioral coordinate — a rank-1 abstraction of the completion distribution.

> Do completion predicates pick out non-PCA, *causal* residual structure on
> `pstack`, or does predicate-targeting collapse onto the variance/KL
> subspace the way exp 26 found `cegar ≈ pca`?

This is the cheap **diagnostic** that decides whether the predicate phase is
worth building. `ORIGINAL_SIN.md` itself flags `pstack` as near variance-
mimicry and says the eventual toy should be *designed backward* from a
predicate suite — so `ALIGNED` (likely) is **not a failure**: it is
evidence-not-vibes that the geometric payoff needs a richer process.
`DIVERGENT` would be the first daylight for the original idea with oracle
discipline intact. The experiment does **not** run a CEGAR-against-φ loop —
it decides whether that loop is worth building.

This is the first step of the `ORIGINAL_SIN.md` bridge (post-oracle-
withdrawal: exps 27/28 closed the reference arc cleanly), and the first
build under the new library-home rule — it uses library primitives
(`battery.cegar_loop`, `abstraction.PCAAbstraction`,
`discover.principal_angles_deg`) and the new reusable `predicates.py`, with
**no import from frozen experiment scripts**.

## Registered Command

```bash
python3 scripts/predicates/predicate_targeting.py --outdir out/pstack-L4
```

Review-only self-tests:

```bash
python3 predicates.py                                            # library
python3 scripts/predicates/predicate_targeting.py --selftest     # verdict logic
```

(Run via `uv run python ...`.)

## Oracle Discipline

Decode probes are fit on the **observable** model `p_φ` (honesty intact);
the patch movement and the principal angles are model-only. The exact
oracle enters once, evaluation-only: exact `p_φ*(belief) = Σ_c φ(c)·
mgram(belief)` for the obs/exact soundness datum. No threshold is tuned by
the oracle.

## The Predicates (registered, library `predicates.py`)

Boolean masks over `V^3` (`pstack` tokens: `0,1` open type-0/1; `2,3` close
type-0/1; `4,5` neutral), spanning behavioral salience so the
PCA-divergence test has teeth:

| φ | meaning | why |
|---|---|---|
| `phi1_next_closes` | `c[0] ∈ {2,3}` | common, high-variance — likely on the PCA axis |
| `phi2_net_return` | net depth change over the window `<= −1` | depth-flavored, structured |
| `phi3_all_neutral` | all three tokens neutral (`{4,5}³`) | the hidden-mode emission — pstack's "richer than Dyck" part, the divergence bait |
| `phi4_first_matched` | the first opened bracket is closed by a matching close inside the window | the smallest genuinely within-window temporal/binding predicate |

`phi3`/`phi4` are the divergence bait. Minimal `not/and/or` algebra is in
the library (`ORIGINAL_SIN.md` prerequisite); `phi4` is the one temporal
operator.

## Per-φ Measurements (per seed)

- **Decode (finds the candidate direction).** Affine ridge probe
  `p_φ ≈ w·r + b` from centered residuals → observable `p_φ`, held-out `R²`;
  `w_φ = w/|w|`. Also a **kNN `R²`** (`k=10`): linear low **and** kNN high =
  present-but-not-affinely-decodable, the exp-1 Z1R interpreter gap.
- **Intervene (validates causality).** Patch `r` along `w_φ`
  (source→target) and measure the **`p_φ`-closure** `c_φ(w)` = fraction of
  the `p_φ`-gap-to-source closed, against the **full-patch ceiling**
  (`c_φ(full)=1`) and a **random-direction floor** `c_φ(rand)`. Decode
  without causal control = `ECHO`.
- **Diverge (the headline).** Principal angles `∠(w_φ, PCA top-k)` and
  `∠(w_φ, cegar core)` — `k` matched to the core dimension. Large vs PCA =
  φ needs non-variance structure; **outside the core** = the full-distribution
  KL objective washed out a behaviorally-relevant direction (the
  `ORIGINAL_SIN` indictment) — *only if `w_φ` is causally validated* (an
  echo direction outside the core indicts nothing).
- **Soundness.** `|p_φ(model) − p_φ*(exact)|` per φ — the per-predicate
  M5/P4 analog; gates trust in the verdict.

## Typed Outcomes (clean partition, FORMALISM §6.1)

Per `(φ, seed)`, exactly one of:

- `VACUOUS` — `p_φ` std `< 0.05` (the predicate barely varies; the probe
  would fit noise — `phi3` is the live risk). Excluded from the verdict.
- `NOT_DECODABLE` — linear `R² < 0.50` **and** kNN `R² < 0.50` (φ absent or
  beyond this interpreter).
- `INTERPRETER_GAP` — linear `< 0.50`, kNN `≥ 0.50` (present, not affinely
  decodable; Z1R-like).
- `ECHO` — decode-sufficient but `c_φ(w) < 0.50` or within `0.20` of the
  random floor (decodes, doesn't causally control).
- `ALIGNED` — decode-sufficient, causal, and `w_φ` **inside** both PCA and
  core (`∠ ≤ 10°`).
- `DIVERGENT` — decode-sufficient, causal, and `w_φ` **outside both**
  (`∠ > 10°`).
- `MIXED` — causal but off one subspace and not the other.

**Multi-seed (mandatory).** The angles are the exact single-seed geometry
that proved seed-fragile in exps 24/25, so a φ's verdict must **reproduce in
`≥ 3/4` fresh seeds** (`300–303`); `DIVERGENT` additionally requires
obs/exact soundness (`oe ≤ 0.10`) in the majority, else `DIVERGENT_UNSOUND`.
No majority verdict → `UNSTABLE`.

Experiment decision: `DIVERGENT(φ…)` if any φ reproduces divergent-and-sound;
else `ALIGNED` if every causal φ is aligned; else `ECHO(φ…)` if the decodable
φ are echoes with no causal φ; else `INCONCLUSIVE`.

## Predictions

**P1 (substrate/self-checks; enforced).** Registered config, PairSet
self-checks at every seed; library + verdict self-tests pass.

**P2 (predicate non-vacuity; ~descriptive).** `phi1`/`phi2`/`phi4` vary
enough to decode; `phi3` (all-neutral) may be `VACUOUS` (rare) — reported,
not a failure.

**P3 (decode; ~75%).** The non-vacuous predicates are linearly decodable
(`R² ≥ 0.50`); an `INTERPRETER_GAP` on a structured φ (e.g. `phi4`) is a
real, interesting sub-outcome.

**P4 (the headline; ~`ALIGNED` likely).** On near-mimicry `pstack` the
causal predicate directions most likely sit inside PCA≈core → `ALIGNED`
(consistent with exp 26). `DIVERGENT` on `phi3`/`phi4` is the daylight case
and would be decisive. `ECHO` (decode ≠ causal at predicate granularity) is
the third live outcome.

**P5 (decision; deterministic).**

- `DIVERGENT(φ…)` → predicate-targeting reaches non-variance causal
  structure; **pre-register the CEGAR-against-φ loop** as the predicate
  phase's first claim-producing experiment.
- `ALIGNED` → the lattice adds no new residual geometry on `pstack`;
  **design the next toy backward from predicates** (`ORIGINAL_SIN.md`) — do
  not keep mining `pstack`. Evidence, not failure.
- `ECHO(φ…)` → the interventional criterion is the only honest one; any
  predicate-CEGAR must score causally, not by decode.
- `INCONCLUSIVE` → the geometry did not reproduce; report per-φ and treat
  as a substrate/probe limitation, not a result.

## Scope & Local Assumptions

- Indexed by `pstack`, `L1`, `m=3`, the registered positions, the four
  predicates, the linear/kNN interpreter classes, and the 4 fresh seeds;
  pair/basis sampling at a fixed checkpoint.
- A predicate is a *within-horizon* mask (no cross-window temporal operator
  beyond `phi4`'s within-`m` matching); the predicate *language* is
  deliberately tiny (`ORIGINAL_SIN.md`: do not start large).
- The `w_φ`-outside-core indictment holds only for *causally validated*
  directions; geometry alone proves nothing here (the program's oldest
  lesson).
- This is a diagnostic, not the predicate phase: no CEGAR-against-φ loop,
  no predicate-algebra composition claims, no new toy. `ALIGNED` is a
  redirect with evidence, not a failure.
- Exact closure/`p_φ*` is evaluation-only; sampled completions not used.

## Expected Output

Per seed: `k_core`, then a per-φ table (`p_φ` std, linear/kNN `R²`,
`∠(w_φ,PCA)`, `∠(w_φ,core)`, `c_φ(w)`, `c_φ(rand)`, obs/exact, and the
per-`(φ,seed)` verdict). Then the multi-seed aggregate per φ (the four
seed verdicts, soundness, reproduced verdict) and the experiment `DECISION`
per P5.

---

## Results

Not run. Pause for pre-run review of the predicate layer, the decode/
intervene/diverge partition, and the code.
