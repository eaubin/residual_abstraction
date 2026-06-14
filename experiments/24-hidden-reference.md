# Experiment 24 — Oracle-withdrawal 2: hidden-oracle reference selection — PRE-REGISTRATION

**Script:** `scripts/oracle_withdrawal/reference_selection.py`.

**Status: run complete; result-review pending. Headline: P2 fails
informatively — `REFERENCE_AMBIGUITY_CONFIRMED`, a typed NO-GO. Three
`k=4` candidates tie at ~93% closure and resolve into two distinct
references — `pca` versus a near-coincident `{cegar, delta}` pair; no
unique earned reference for battery transfer.**

## Question

Can an observable-only protocol select a compact trusted reference on
`pstack`, before exact oracle audit is revealed?

This is the first oracle-withdrawal abstraction experiment. Exp 23
established that the `pstack` substrate is competent, exact-auditable,
sampled-calibrated at `1024` completions, stratified, and non-vacuous.
Exp 24 now tests the load-bearing bridge to no-oracle work: whether the
reference used by rho and later battery members can be earned inside the
observable protocol rather than handed over by ground truth.

## FORMALISM & Ledger

This experiment begins to discharge two live ledger rows in
`FORMALISM.md` §7:

- "a trusted reference patch can anchor rho ... LLM-scale reference
  selection **open**" (the rho-reference bet) — Exp 24 puts *no-oracle*
  reference selection on a richer toy (`pstack`) **under test**;
- "exact-toy adjudication calibrates later oracle-free work ... Remaining
  open half: scale and no-oracle reference selection" (the framing bet) —
  Exp 24 tests the no-oracle reference-selection half.

Both rows are annotated **under test (exp 24)** in the ledger; the verdict
returns here at conclusion. The verdict logic is built as a partition over
the registered branches and is audited against the FORMALISM §6.1
verdict-predicate checklist (every selection rule the code applies is
registered — rule 8; equivalence claims name the metric that settles them
— rule 4).

## Registered Command

```bash
python3 scripts/oracle_withdrawal/reference_selection.py --outdir out/pstack-L4
```

Review-only self-test:

```bash
python3 scripts/oracle_withdrawal/reference_selection.py --selftest
```

## Oracle Discipline

Selection uses only:

- tokens;
- residual streams;
- model source-run and patched completion distributions;
- observable closure on discovery/evaluation/held-out PairSets;
- observable stack-depth strata.

Exact m-gram closure is revealed only after the observable selection
branch is fixed and printed. Exact closure audits the selected reference;
it does not participate in proposal construction, candidate ranking, or
selection thresholds.

Implementation note: the shared `PairSet` object stores exact fields by
library design, but Exp 24 selection functions do not read those fields.
The first exact read is inside the registered audit stage after
`OBSERVABLE_SELECTION` has been printed.

The full patch is a ceiling/control only. It is not a selectable
successful compact reference. If the protocol can only trust the full
residual, the outcome is the typed `FULL_PATCH_FALLBACK` branch.

## Registered Setting

| item | value |
|---|---|
| process/checkpoint | `pstack`, `out/pstack-L4` |
| patch layer | `L1` (`expcommon.LAYER`) |
| horizon | `m=3`, `mm=3` |
| discovery positions | `ts=(10,18,26,34)` |
| held-out positions | `ts=(12,20,28,36)` |
| discovery pairs | `320` from `900` sequences |
| eval pairs | `1024` from `900` sequences |
| held-out pairs | `512` from `900` sequences |
| observable basis sample | `240` sequences |
| CEGAR | `eps=0.05`, `eps_drop=0.01`, `k_max=10` |
| selectable dimension cap | `1 <= k <= 8` |
| stratum coverage floor | `>= 20` rows per observable depth `0..3` |
| stratum non-vacuity floor | recovery room `>= 5%` of global (`STRATUM_DENOM_MIN_FRAC`) |
| ambiguity subspace threshold | max principal angle `> 10 deg` ⇒ CONFIRMED (diagnostic only) |

Eval pairs were lifted from `512` to `1024` before the canonical run
(pre-run review, F3): at `512` the deepest observable stratum (depth 3,
`~4%` of positions per Exp 23) lands at `E~21` rows against the `20`-row
coverage floor — a coin-flip on silently dropping the most
completion-relevant stratum. At `1024` it sits at `E~42 +/- 6`, robustly
covered. Held-out pairs stay at `512`: `obs_held` is a global mean with no
stratified guard.

## Candidate References

All compact candidates are built without exact oracle labels.

| candidate | construction | selectable? |
|---|---|---|
| `cegar` | observable CEGAR loop on discovery PairSet | yes |
| `pca` | top residual-variance directions on observable basis rows | yes |
| `obs_pls` | PLS from residual rows to model source-run completion distributions | yes |
| `delta` | fixed weighted prefix-delta basis using observable unpatched error rows | yes |
| `emb` | token-embedding row space, truncated to matched dimension | yes |
| `rand` | random matched-dimension destructive control | no |
| `full` | full residual patch ceiling/control | no |

Candidate dimension is matched to the observable CEGAR dimension, clipped
to `1..8`, except `emb`, which is capped by vocabulary dimension. The
full patch is reported but cannot count as success.

## Observable Selection Rule

Before any candidate is scored, the script runs two typed substrate
guards on the stratified observable scale (printed, then halt-on-fail):

- `STRATA_TOO_SPARSE`: any observable depth stratum `0..3` has fewer than
  `20` rows on the eval PairSet — the deepest strata would be silently
  omitted from `strata_min`.
- `STRATA_VACUOUS`: a stratum's full-patch recovery room `d0 - df` is at
  or below `5%` of the global recovery room — its closure denominator is
  degenerate (the per-stratum analog of `discover.py`'s `D0 > D_full`
  invariant). A degenerate denominator could blow up or sign-flip
  `strata_min` and sink an otherwise-good candidate.

Either guard returns a typed `NO-GO` halt (status `1`); the miss surfaces
as itself rather than being absorbed into `NO_TRUSTWORTHY_REFERENCE`. The
per-stratum `d0`, `df`, and recovery room are printed for the record.

With the guards passed, for each candidate the script computes:

- `obs_eval`: observable closure on the registered eval PairSet;
- `obs_held`: observable closure on held-out positions;
- `strata_min`: minimum observable closure over the four guarded
  stack-depth strata (all covered and non-vacuous by construction);
- `score = min(obs_eval, obs_held, strata_min)`.

A compact candidate is eligible iff:

- `obs_eval >= 0.70`;
- `obs_held >= 0.60`;
- `strata_min >= 0.50`;
- it is selectable and has dimension at least `1`.

Selection:

- If no compact candidate is eligible and full patch `obs_eval >= 0.90`,
  branch = `FULL_PATCH_FALLBACK`.
- If no compact candidate is eligible and full patch is also weak,
  branch = `NO_TRUSTWORTHY_REFERENCE`.
- Otherwise choose the eligible candidate with largest `score`.
- If more than one eligible candidate is within `0.03` of the best
  score, branch = `REFERENCE_AMBIGUITY`.
- If exactly one candidate wins, branch = `SELECTED` and the selected
  name is printed before exact audit is computed.

## Exact Audit Branches

After observable selection is fixed, the script computes exact closure
for all non-full candidates on the eval PairSet.

If the observable branch is `SELECTED`, exact audit assigns:

- `REFERENCE_SELECTED_CORRECTLY` iff:
  - selected exact closure `>= 0.70`;
  - `abs(obs_eval - exact) <= 0.15`;
  - no other selectable compact candidate is better by more than `0.10`
    exact closure.
- `SELECTED_REFERENCE_TOO_WEAK` iff selected exact closure `< 0.70`.
- `OBS_EXACT_DISAGREEMENT` iff the selected candidate passes exact
  strength but `abs(obs_eval - exact) > 0.15`.
- `ORACLE_REVEAL_INVERSION` iff another selectable compact candidate is
  better by more than `0.10` exact closure.

Mixed selected-reference failures are classified in that priority order:
too-weak, then obs/exact disagreement, then oracle-reveal inversion.

If the observable branch is `REFERENCE_AMBIGUITY`, exact audit assigns:

- `REFERENCE_AMBIGUITY_CONFIRMED` iff tied candidates differ by more than
  `0.10` exact closure **or** span genuinely distinct subspaces (max
  pairwise principal angle `> 10 deg`) — the latter being the
  `ORACLE_WITHDRAWAL.md` "disagree on downstream rho" case, since
  near-equal exact closure does not imply the same reference;
- `REFERENCE_AMBIGUITY_BENIGN` only when tied candidates agree on **both**
  exact closure and subspace.

Naming the metric that settles the equivalence claim (FORMALISM §6.1 rule
4): the script prints pairwise principal angles for the tied candidates,
and prints `cegar`-vs-`pca` angles unconditionally so the variance-mimicry
lens (Exps 6/8) is on the record whether or not a tie occurs.

Both ambiguity sub-branches are a `NO-GO` for battery transfer (P2/P5: any
ambiguity blocks transfer and registers a follow-up). The
`CONFIRMED`/`BENIGN` split is a diagnostic label for the writeup, not a
separate decision; the `10 deg` threshold therefore refines the label, not
the transfer call.

Conclusion-framing note (binds the eventual writeup): a
`REFERENCE_AMBIGUITY_BENIGN` landing is *not* a selection failure. It means
the protocol had a usable reference — two equivalent names for one
subspace — and registered a tie-break follow-up out of conservatism. The
results writeup must say that plainly rather than report it as "no
reference found".

`FULL_PATCH_FALLBACK` and `NO_TRUSTWORTHY_REFERENCE` remain typed
no-reference outcomes after exact audit.

## Predictions

**P1 (registered substrate guard; enforced).** The script runs only on
the registered `pstack` config unless `--force-invalid` is passed, the
standard PairSet self-checks pass, and both stratified guards pass
(`STRATA_TOO_SPARSE`, `STRATA_VACUOUS` above), each a typed substrate halt
(status `1`) — not an Exp 24 reference-selection result; register a
substrate repair. Global non-degeneracy (`D0 > D_full`) is enforced as a
bare `assert`/tripwire (the `discover.py` idiom), not an expected typed
branch: given Exp 23's `98.4%` global recovery it will not fire, and if it
ever did the run should crash rather than be interpreted.

**P2 (observable protocol finds a compact reference; ~65%).** The
observable branch is `SELECTED`, not full-patch fallback, no-reference,
or ambiguity.

Failure is not a code failure. It is a typed reference-selection result
and blocks battery transfer until a focused follow-up is registered.

**P3 (exact audit confirms selection; ~60%).** Given P2 selection, exact
audit returns `REFERENCE_SELECTED_CORRECTLY`.

Failure branches:

- selected too weak: reference overtrust;
- obs/exact disagreement: oracle-reveal calibration problem;
- oracle-reveal inversion: observable ranking selected the wrong compact
  reference.

**P4 (full patch remains ceiling/control only; enforced).** The full
patch is reported but never selectable. If it is the only trustworthy
object, the outcome is `FULL_PATCH_FALLBACK`, not success.

**P5 (decision; deterministic).**

- If P2 and P3 hold: GO to preregister battery transfer with the earned
  reference.
- If P2 fails: do not run battery transfer; register the corresponding
  reference-selection follow-up.
- If P3 fails: do not run battery transfer; register the exact-audit
  failure branch as the next experiment or protocol repair.

## Scope & Local Assumptions

- This experiment selects at most one compact reference for `pstack`.
- It does not run the six-member battery.
- It does not compare proposal families as a research result; candidates
  here are reference-selection candidates only.
- Exact closure is evaluation-only.
- Sampled completions are not used here; Exp 23 established substrate
  sampling at `1024`, but this experiment uses exact model chain
  probabilities for observable scoring.
- The result is indexed by `pstack`, L1, `m=3`, the registered positions,
  candidate family list, and selection thresholds above.

## Expected Output

The script prints:

- per-stratum coverage and recovery-room table, and the strata-guard
  result (typed halt on failure);
- CEGAR `k*`, matched candidate dimension, and `cegar`-vs-`pca` principal
  angles;
- observable candidate table;
- the observable selection branch and selected/tied candidates;
- exact audit table after selection;
- exact audit branch, plus tied-candidate principal angles when the
  branch is `REFERENCE_AMBIGUITY`;
- GO/NO-GO decision.

---

## Results

Run artifact: `out/exp24_pstack-L4.txt`. Result-review pending; this
section is the conclusion draft. The checkpoint `out/pstack-L4/model.pt`
and `cache.npz` are untracked per repository policy: the run is
CPU/fixed-seed and reproducible from the Exp 23 registered training
command.

**Headline: P2 fails informatively — `REFERENCE_AMBIGUITY_CONFIRMED`, a
typed NO-GO.** The observable protocol found compact references but could
not single one out: three candidates (`cegar`, `pca`, `delta`) tie
observably at ~`0.93` closure and exactly (spread `0.003`), yet do not
collapse to a single plane — `pca` separates from a near-coincident
`{cegar, delta}` pair. The multiplicity is (at least) two distinct
references, not one under three names.

### Verdict fidelity

| prediction | registered | outcome |
|---|---|---|
| P1 substrate guard | enforced | **held** — self-checks pass; strata covered (depth 0–3 = `287/226/137/34`, min `34 >= 20`); all strata non-vacuous (recovery room `4.51–9.38`) |
| P2 observable selects | ~65% `SELECTED` | **failed (typed)** — `REFERENCE_AMBIGUITY`; `cegar`/`pca`/`delta` tied |
| P3 exact confirms | ~60%, **given P2 `SELECTED`** | **not reached** — conditional on a unique selection; not evaluated, counted neither pass nor fail |
| P4 full = ceiling only | enforced | **held** — full `obs_eval=1.000` by construction, never selectable, not selected |
| P5 decision | deterministic | **NO-GO** — register the `REFERENCE_AMBIGUITY` follow-up |

### What happened

CEGAR converged at `k*=4` (the same compact dimension Dyck's causal core
took). At matched dimension `4`, three observable candidates clear every
eligibility gate and tie within the `0.03` margin:

| candidate | obs_eval | obs_held | strata_min | score | exact |
|---|---|---|---|---|---|
| `pca` | 0.946 | 0.945 | 0.936 | **0.936** | 0.923 |
| `cegar` | 0.950 | 0.950 | 0.930 | 0.930 | 0.926 |
| `delta` | 0.947 | 0.947 | 0.926 | 0.926 | 0.923 |
| `emb` | 0.794 | 0.803 | 0.779 | 0.779 | 0.773 |
| `obs_pls` | 0.003 | 0.003 | 0.002 | 0.002 | 0.003 |
| `rand` | 0.055 | 0.040 | −0.016 | −0.016 | 0.055 |
| `full` | 1.000 | 1.000 | 1.000 | 1.000 | — |

`emb` sits a clear notch below; `obs_pls` collapses to ~`0` (the recurring
PLS echo — decode-sufficient, causally empty); `rand` is destructive
(strata_min `−0.016`, the deepest stratum punishing an off-manifold
projector). The full patch is the ceiling at `1.000` by construction and
is never selectable (P4).

Exact audit, revealed only after `OBSERVABLE_SELECTION` was printed, puts
the three tied candidates at near-identical closure: `cegar 0.926`,
`delta 0.923`, `pca 0.923` — spread `0.003`. **By exact-closure spread
alone this reads `BENIGN`.** The subspace diagnostic overturns it, but not
uniformly: `cegar` and `delta` are near-coincident (`4.9°`, the same plane
by the `10°` rule), while `pca` stands apart from that pair
(`pca–delta 12.4°`, `cegar–pca 10.4°`). The max tied principal angle
(`12.4°`) exceeds the `10°` threshold, so the branch is
`REFERENCE_AMBIGUITY_CONFIRMED` — driven entirely by `pca`'s separation,
not a three-way split.

### Interpretation

This is the variance-mimicry lens (Exps 6/8) landing on the **opposite
pole** from Mess3. On Mess3's 3-state belief simplex, `cegar ≈ pca ≈` the
variance plane (`3.3–3.6°` apart) — effectively one reference under
several names. On `pstack` (`S=30`, a hidden mode biasing both bracket
openings and neutral terminals), the tie does not collapse to one plane:
`cegar` and `delta` remain near-coincident (`4.9°`, the same plane by this
experiment's `10°` rule), but `pca` separates from that pair
(`10.4°`/`12.4°`). So at least two distinct 4-dim references each reproduce
~`92–93%` of source completion behavior — observably *and* exactly — and
the `CONFIRMED` verdict is driven entirely by `pca`'s separation.
**Near-equal closure does not imply the same reference.** This is exactly
the case the subspace criterion (F2) was added to catch, and here it is
load-bearing: without the angle test the protocol would have mislabeled a
real reference multiplicity as benign.

So the observable protocol cannot, on `pstack` at this setting, hand a
*unique* earned reference to ρ. It is not that no compact reference exists
— several do — but that neither the observable score nor exact closure
singles one out. That is a typed reason oracle-free reference selection is
not yet ready here.

Two design choices proved necessary rather than ornamental:

- **The F3 eval-pair bump was load-bearing for getting a result at all.**
  Depth-3 covered `34` rows at `1024` pairs; at the original `512` it
  would have been ~`17`, below the `20` floor, and the run would have
  halted `STRATA_TOO_SPARSE` before any selection. The deepest stratum is
  also where the controls separate most (`rand` strata_min `−0.016`).
- **The `0.03` tie margin did its job.** `pca` leads by `0.006`/`0.010`
  observably; a tighter margin would have spuriously "selected" `pca` over
  two exactly-as-good, genuinely-different references. The `0.003` exact
  spread confirms the tie is real, not a thresholding artifact.

### Decision

**NO-GO on battery transfer.** Per `ORACLE_WITHDRAWAL.md`'s surprise
triage this is a *follow-up-before-the-next-experiment* result (the
selected reference is ambiguous): ρ and every later battery member are
indexed by which reference anchors them, so the multiplicity must be
resolved before Block 3 (battery transfer). Register a focused
reference-ambiguity follow-up (Exp 25). Candidate directions for that
registration:

- an **observable tie-break** — does ρ-stability, held-out-position gain,
  or a horizon sweep separate `pca` from the near-coincident `{cegar,
  delta}` plane without the oracle? The live contest is two-way, not
  three: `cegar` and `delta` are already one reference by the `10°` rule.
- **accept multiplicity and index ρ by reference** — test whether the
  battery's verdicts agree across the two distinct references (`pca` vs the
  `{cegar, delta}` plane), a robustness reframing of the ambiguity;
- a **canonical-reference rule** — register the interventionally-discovered
  `cegar` core as the anchor with the others as controls (the LLM-posture
  choice, since at scale only the discovered core is available). Note
  `cegar` and `delta` coincide here, so this rule also resolves the
  observable-supervised `delta` candidate; only `pca` is left as a genuine
  alternative to adjudicate.

### Scope

Indexed by `pstack`, `L1`, `m=3`, the registered positions, the candidate
family, the `0.03` tie margin, the `0.70/0.60/0.50` eligibility
thresholds, and the `10°` ambiguity-angle threshold. A different tie
margin or candidate set could change whether the ambiguity *fires*; the
underlying *finding* — distinct compact references of near-equal closure
exist on `pstack` — is robust in that all three tied candidates also agree
under exact audit (spread `0.003`) while `pca` sits `10–12°` from the
`{cegar, delta}` plane. The `CONFIRMED` label is itself threshold-marginal:
the max tied angle (`12.4°`) clears the `10°` line by only ~`2.4°`, so
"distinct subspaces" here means modestly, not crisply, separated — and
`cegar–delta` (`4.9°`) fall on the *same* side as Mess3's variance plane.
The ambiguity-angle threshold is diagnostic only (both sub-branches are
NO-GO), so it did not affect the transfer decision, only the label; a
`10°` line drawn a few degrees higher would have read this same geometry
as `BENIGN`.
