# Experiment 35 — Joint stack-state separability diagnostic (design label 2b) on pstack-L4 — PRE-REGISTERED

**Script:** `scripts/interventions/i3b_joint_state_separability.py`.
**Output:** `out/exp35_pstack-L4.txt`.

**Status: concluded.** Pre-registration review completed; the claim-producing run
is in `out/exp35_pstack-L4.txt`. Results and conclusion are at the end of this
file.

**Decision:**

```text
BROAD_STATE_REPLACEMENT(stack_state_bundle)
```

with a load-bearing caveat: the out-of-bundle discrimination rests on `phi4`,
whose room is in the marginal band (flagged every seed). See the conclusion.

## Phase fit and why this is the next experiment

Exp 34 concluded `NONSPECIFIC_DELTA(phi1_next_closes,phi2_net_return)`: near-
manifold matched activation deltas *do* move each predicate (beating the
mismatched and shuffled floors, with held-out transfer and the `own_delta`
ceiling closing fully), but they co-move the other registered predicates, so
per-target specificity failed. Exp 34 scored the two stack-state predicates as
**mutually** non-target: when targeting `phi1`, a move in `phi2` counted against
specificity, and vice versa.

That scoring presupposes `phi1_next_closes` and `phi2_net_return` are separately
controllable variables. They may not be. Both are depth/closing readouts of the
same stack state, so a near-manifold residual move that changes one should change
the other — not as a leak, but because they are two facets of one variable. Under
that reading, exp 34's "nonspecific" verdict conflated *bundle-internal coupling*
(expected, benign) with *broad replacement* (the real specificity failure).

This experiment separates those two. It does **not** introduce a new optimizer,
a learned write, or a new process: it reuses the exact exp-34 near-manifold delta
machinery and only **re-adjudicates the same moves** under a bundle vs out-of-
bundle split. It is the cheap diagnostic that can *reinterpret* the whole arc
from exp 29 onward before any expensive learned I2 read/write-pair run. Per the
next-experiment bar in `INTERVENTION_CLASS_BENCHMARK.md`, its outcomes change the
carry-forward decision: a positive flips exp 34's negative into "the rank-1
oblique class is alive, the target decomposition was wrong" and routes to a cheap
joint-write confirmation; a negative confirms the broad-replacement reading and
routes to I4 patch-point or consolidation.

## Question

```text
On pstack-L4 at L1, m=3, for the exp-33/34 positions, do the exp-34 near-manifold
matched deltas — each built to move one bundle predicate — (a) co-move the other
bundle predicate toward source (joint stack-state control) while (b) sparing the
out-of-bundle predicate phi4_first_matched and avoiding broad full-m-gram
replacement, on held-out positions with transfer? I.e., is exp-34's nonspecific
co-movement bundle-internal (one writable stack-state variable) or broad (state
replacement)?
```

This is a target-decomposition diagnostic, not a learned read/write-pair
experiment and not a general interchange proof.

## Registered Command

After preregistration review only:

```bash
uv run python scripts/interventions/i3b_joint_state_separability.py \
  --outdir out/pstack-L4 | tee out/exp35_pstack-L4.txt
```

Review-only checks:

```bash
uv run python scripts/interventions/i3b_joint_state_separability.py --selftest
uv run python -m py_compile interventions.py intervention_eval.py \
  scripts/interventions/i3b_joint_state_separability.py
```

## Scope Indices

| index | value |
|---|---|
| process/checkpoint | `pstack-L4`, same registered config as exps 33/34 |
| patch point | residual stream L1, prefix-wide PairSet patch up to the scored position |
| horizon | `m=3` within-horizon completion distribution |
| bundle (joint target) | `phi1_next_closes`, `phi2_net_return` |
| out-of-bundle control | `phi4_first_matched` (within-window binding predicate) |
| excluded | `phi3_all_neutral` (vacuous, no room) |
| intervention class | exp-34 observed matched/own/mismatched/shuffled activation deltas; no covector, no learned write |
| discovery bin | positions `{10,18}`, 512 pairs/seed |
| held-out bin | positions `{26,34}`, 1024 pairs/seed |
| seeds | `600..603` (fresh relative to exps 30-34) |
| exact oracle use | endpoint audit only; no matching, donor, dose, split, or verdict uses exact labels |
| device | live accelerator when available; runtime detail only |

## Design

For each seed, for each bundle predicate `t in {phi1, phi2}`:

1. Build discovery and held PairSets (exp-33/34 positions and sizes); run
   `discover.self_checks` on both bins.
2. Build the exp-34 delta matching on `t`: eligibility is the top-35%
   `|p_src - p_un|` rows per position; donor arms are `own_delta`,
   `matched_delta` (same position, same sign, matched `p_un` bin, nearest
   magnitude, non-self), `mismatched_delta` (opposite sign), `shuffled_delta`
   (random non-self).
3. Compute observable endpoints for `phi1, phi2, phi4` on the **same** eligible
   rows, with exact endpoint audit.
4. Dose the four arms over `alpha in {0,.25,.5,1,1.5,2}` and select the best
   observable `t`-control per arm on each bin (identical to exp 34).
5. At the `matched_delta` arm's selected held-out dose, read the **same patched
   completion distribution** on three predicates: `t` itself (target control),
   the other bundle predicate (bundle co-movement, signed), and `phi4`
   (out-of-bundle control, magnitude). Also score full `m=3` m-gram closure.

A reported **read-geometry** block fits in-place affine reads for `phi1, phi2,
phi4` and prints `|cos|` between read directions: `cos(phi1,phi2)` is expected to
exceed `cos(phi1,phi4)` and `cos(phi2,phi4)` if the bundle framing holds. This is
corroborating context only; the verdict is interventional (the causal
`bundle_co` quantity is the load-bearing version of "are they bundled").

## Scores

Predicate control is the exp-33/34 closure fraction
`c(P) = [MSE(p_un,p_src) - MSE(p_P,p_src)] / room`, with `room <= ROOM_TOL`
giving `NO_PATCH_ROOM`. Full m-gram closure is the analogous source-distribution
KL closure over the full `m=3` continuation distribution. The three new
quantities, all read off the same matched-arm held-out patched distribution at
the target's selected dose:

- **target control** `c(t)` — the matched-on predicate (the exp-34 number).
- **bundle co-movement** `c(other)` — signed control of the other bundle
  predicate; the joint reading wants this positive and large. Read with **its
  own no-information floors**: the other predicate's control under the same
  arm's `mismatched_delta` and `shuffled_delta` moves. At these high-gap
  eligible rows an undirected same-position move already co-shifts both
  predicates (exp-34 shuffled floor on the matched-on predicate was 0.26-0.41),
  so coupling must be read as movement *beyond* that undirected floor.
- **out-of-bundle control** `|c(phi4)|` — the joint reading wants this small,
  with the `own_delta` full-replacement value (`|oob|` under literal source
  replacement, expected `~1`) printed as a sanity reference.

## Per-Seed Verdict

One joint verdict per seed from the two matched-on arms (`classify_seed`), in
the check order below. Gates require the condition for **both** arms. Drift is
checked before the room gates (room is uninterpretable under endpoint drift);
the structural preconditions (`TARGET_VACUOUS`, `LOW_MATCH_SUPPORT`,
`NO_PATCH_ROOM`) are calibration-independent and checked first.

| branch | plain-language gloss | condition |
|---|---|---|
| `TARGET_VACUOUS` | a bundle predicate too flat | `std < VAR_MIN` on either bin, either arm |
| `LOW_MATCH_SUPPORT` | too few eligible rows or missing donor support | `< MIN_ELIGIBLE_PER_BIN` per position, or donor support fails |
| `OBS_EXACT_DRIFT` | observable endpoints not calibrated | endpoint audit `> OE_BAND` |
| `NO_PATCH_ROOM` | bundle full/source interpolation cannot move | room `<= ROOM_TOL` either bin |
| `NO_OOB_ROOM` | phi4 has no full-patch room | `oob_room < OOB_ROOM_MIN`; separability test non-diagnostic |
| `DELTA_GATE_INVALID` | even own source-target deltas fail the gate | `own_delta` held control `< C_MIN` |
| `NO_JOINT_CONTROL` | matched deltas do not reliably move both bundle predicates | any `target` control `< C_MIN`, retention `< RETENTION_MIN`, or fails the `C_MARGIN` floor margin |
| `JOINT_STACK_VARIABLE` | one writable stack-state variable | coupled **and** out-of-bundle spared (below) |
| `BROAD_STATE_REPLACEMENT` | bundle moves but so does phi4 / the distribution | coupled, not spared |
| `SEPARABLE_PREDICATES` | target moves, phi4 spared, but the other bundle predicate does not follow | not coupled, spared |
| `NONJOINT_NONSPECIFIC` | move neither couples the bundle nor spares out-of-bundle | not coupled, not spared |

with, on held-out, for both arms:

```text
coupled    := bundle_co >= COUPLE_MIN
              and  (bundle_co - max(bundle_co_mismatched, bundle_co_shuffled))
                   >= C_MARGIN
oob_spared := |oob| <= OOB_MAX  and  mgram <= MGRAM_MAX
              and  (target_control - |oob|) >= SEP_MARGIN
```

The coupling floor is the load-bearing F1 fix: an absolute `COUPLE_MIN` alone
could be cleared by the generic same-position co-movement that even
no-information donors produce, which would both weaken a `JOINT` positive and
structurally suppress `SEPARABLE_PREDICATES` (it could never fire if shuffled
moves already pushed the other predicate past `COUPLE_MIN`). Requiring the
directed move to beat its undirected floors by `C_MARGIN` keeps all four
interventional cells reachable.

Thresholds: `VAR_MIN=0.05`, `C_MIN=0.50`, `C_MARGIN=0.20`,
`RETENTION_MIN=0.50`, `COUPLE_MIN=0.40`, `OOB_MAX=0.35`, `SEP_MARGIN=0.20`,
`OOB_ROOM_MIN=0.01`, `OOB_ROOM_MARGINAL=0.03`, `MGRAM_MAX=0.85`, `OE_BAND=0.10`,
`SEED_MAJORITY=3`. A `phi4` room in `[OOB_ROOM_MIN, OOB_ROOM_MARGINAL)` does not
change the branch but flags the seed: a `JOINT` positive resting on a
marginal-room seed is reported as direction-level (`|oob|` is a high-variance
closure fraction there), not a clean near-manifold result.

## Multi-Seed Aggregation and Decision

The decision is the per-seed verdict holding in `>=3/4` seeds via
`battery.majority_vote`, else `SEED_UNSTABLE`. The decision string is
`<branch>(stack_state_bundle)`. The precedence order (most to least decisive)
records how a split would be read:

```text
OBS_EXACT_DRIFT > NO_OOB_ROOM > JOINT_STACK_VARIABLE > BROAD_STATE_REPLACEMENT
  > SEPARABLE_PREDICATES > NONJOINT_NONSPECIFIC > NO_JOINT_CONTROL
  > DELTA_GATE_INVALID > NO_PATCH_ROOM > LOW_MATCH_SUPPORT > TARGET_VACUOUS
```

## Predictions

- **P1 (guards; enforced).** Config guard, I0 artifact gate, PairSet self-checks,
  `--selftest`, and `py_compile` pass.
- **P2 (room/calibration; likely).** Both bundle predicates retain exp-34 room
  and calibration; `phi4` clears `OOB_ROOM_MIN` (it cleared exp-34's `0.01`
  specificity floor). A `phi4` room below `OOB_ROOM_MIN` is a clean, honest
  `NO_OOB_ROOM` non-diagnostic; a room in `[OOB_ROOM_MIN, OOB_ROOM_MARGINAL)`
  does not block the run but downgrades a `JOINT` positive to direction-level.
- **P3 (joint control; likely).** Given exp 34, the matched deltas should move
  each bundle predicate over the floors, so `NO_JOINT_CONTROL` is unlikely.
- **P4 (coupling; uncertain).** Whether the `phi1`-matched delta also moves
  `phi2` toward source (and vice versa) is the first genuinely new measurement.
- **P5 (headline; uncertain).** Whether `phi4` is spared while the bundle moves
  is the load-bearing measurement. `JOINT_STACK_VARIABLE`,
  `BROAD_STATE_REPLACEMENT`, and `SEPARABLE_PREDICATES` are all live and each
  changes the next step.

## Confound Table — Load-Bearing Quantity

The headline positive turns on `oob_heldout` (held control of `phi4` under the
bundle-matched delta) being small while `bundle_co_heldout` is large, for both
arms.

| confound producing a spurious `JOINT_STACK_VARIABLE` | excluded by |
|---|---|
| the move is full source replacement (everything moves, trivially "joint") | `own_delta` is the replacement reference; matched arm must keep `mgram <= MGRAM_MAX` and beat mismatched/shuffled floors by `C_MARGIN` |
| high-gap eligible rows make *any* same-position delta co-move both predicates, so coupling is undirected, not bundle-specific | coupling must beat the other predicate's `mismatched_delta`/`shuffled_delta` co-movement floors by `C_MARGIN`, not just clear `COUPLE_MIN` (F1) |
| `phi4` happens to be flat/low-room so "spared" is vacuous | `NO_OOB_ROOM` gate below `OOB_ROOM_MIN`; `OOB_ROOM_MARGINAL` caveat band; `phi4` room printed |
| `phi4` is partly correlated with depth and is dragged a little | `SEP_MARGIN`: the target must beat `|phi4|` by `0.20`, not merely exceed it |
| coupling is an artifact of one position/seed | held-out position bin, `RETENTION_MIN`, and `>=3/4` seed majority |
| exact labels leaked into matching | matching uses observable `p_un`, `p_src`; exact truth is endpoint audit only |

The headline negatives are split so each is actionable:

| negative | what it means | excluded confound |
|---|---|---|
| `BROAD_STATE_REPLACEMENT` | move drags `phi4` / the full distribution | not a clean variable; genuine exp-34 non-specificity |
| `SEPARABLE_PREDICATES` | `phi2` does not follow a `phi1` move (and vice versa) | entanglement hypothesis rejected; targets independent |
| `NONJOINT_NONSPECIFIC` | move respects neither bundle nor out-of-bundle | broad and unstructured |

## Reliability Baselines for Thresholds

- `COUPLE_MIN=0.40`: the other bundle predicate must close at least 40% of its
  gap toward source *and* beat its own no-information co-movement floors
  (`mismatched_delta`/`shuffled_delta` on that predicate) by `C_MARGIN`. The
  absolute bar is below `C_MIN` because co-movement is secondary; the floor
  margin is what makes coupling mean "directed move beyond undirected." Ceiling
  is `own_delta` co-movement (`~1`, full replacement); floor is shuffled
  co-movement (`~0.3` by exp-34 symmetry), now measured, not assumed.
- `OOB_MAX=0.35`: the exp-34 `SPEC_MAX`, kept for continuity — `phi4` movement
  above it is not "spared."
- `SEP_MARGIN=0.20`: the `C_MARGIN` value — the bundle must beat the out-of-
  bundle predicate by a real margin, not a sliver.
- `MGRAM_MAX=0.85`: bounds away from literal full-distribution replacement;
  m-gram is a coarse broad-replacement guard, not the joint-vs-broad
  discriminator (`phi4` is). A clean stack-state write may legitimately move a
  fair amount of m-gram because the continuation distribution depends on stack
  state, so m-gram is used only as an upper bound, with `phi4` carrying the
  discrimination.
- `OOB_ROOM_MIN=0.01`: the exp-34 specificity floor `phi4` already cleared;
  below it the separability test is non-diagnostic (`NO_OOB_ROOM`).
- `OOB_ROOM_MARGINAL=0.03`: `predicate_control` divides by room, so `|oob|` is a
  high-variance closure fraction at small `phi4` room. A room in
  `[OOB_ROOM_MIN, OOB_ROOM_MARGINAL)` keeps the run but flags the seed and
  downgrades a `JOINT` positive to direction-level; the `own_delta` `|oob|`
  reference (`~1`) is printed as the replacement sanity anchor.
- `OE_BAND=0.10`: inherited endpoint audit band; exp 34 observed `~0.01`. Drift
  is checked before the room gates, since room is uninterpretable under drift.

## Measured-but-Unadjudicated

The script prints, per seed: the read-geometry cosines; per arm the selected
disc/held alpha, target control, bundle co-movement with its mismatched/shuffled
co-movement floors, out-of-bundle control with the `own_delta` replacement
reference, m-gram closure, own-delta ceiling, target-side mismatched/shuffled
floors, room, `phi4` room (with a marginal flag), audit, and retention. The
verdict reads only the quantities in `classify_seed`. The read-geometry cosines
and dose curves are corroborating context, not adjudicated.

Selected matched alphas are reported **per arm**, not pooled. Exp 34 selected
`alpha=2.0` for `phi2` and `alpha=1.5` for `phi1`, so the `phi2`-matched arm
will likely operate at `alpha>1`. The conclusion must attribute the caveat to
the specific arm: a `JOINT` verdict whose `phi2`-side coupling or sparing only
appears at extrapolation strength is direction-level **for that arm**, even if
the `phi1`-side arm is interpolation-scale. This is reported, not silently
promoted, and does not by itself change the branch.

## Halt Conditions

The run halts if the checkpoint config differs from registered `pstack-L4`, the
I0 preflight artifact is missing or lacks the intervention `GO` route, or any
PairSet known-answer self-check fails.

## Non-goals / Scope Guard

- No learned read/write pair, no rank-k composition, no new process training; the
  intervention object is exp 34's, re-adjudicated.
- No claim about all pairs; scoped to the high predicate-difference eligible
  subset.
- A single out-of-bundle predicate (`phi4`) carries the separability test, plus
  the m-gram bound. A clean positive would be strengthened by a richer toy with
  more separable predicates — that is an exit-gate question, not this experiment.
- No claim that a joint write *mechanism* is identified; a positive motivates a
  cheap rank-1 joint-write confirmation, not a mechanism claim.
- No real-LLM claim. Device choice is runtime-only and recorded in the output.

## Reviewable Failure Modes

- Vacuous "spared": guarded by `NO_OOB_ROOM`, the `OOB_ROOM_MARGINAL` caveat
  band, and the reported `phi4` room.
- Undirected coupling masquerading as bundling: the `coupled` axis carries its
  own `mismatched_delta`/`shuffled_delta` co-movement floors, so coupling means
  the directed move beats undirected co-movement by `C_MARGIN` — and
  `SEPARABLE_PREDICATES` stays reachable (F1).
- Replacement masquerading as joint: guarded by `own_delta` reference, m-gram
  bound, floor margins, and `SEP_MARGIN`.
- Coupling/sparing conflation: `coupled` and `oob_spared` are independent axes;
  the four interventional branches are their full cross-product.
- Single-control fragility: acknowledged scope limit; m-gram backstops `phi4`.
- Negative overreach: each negative is scoped to this donor rule, layer, horizon,
  positions, bundle, out-of-bundle control, and checkpoint.

## Results

Run artifact: `out/exp35_pstack-L4.txt` (completed on `device=mps`).

```text
DECISION: BROAD_STATE_REPLACEMENT(stack_state_bundle)
```

Per-seed verdict was `BROAD_STATE_REPLACEMENT` on all four seeds (600–603), both
matched-on arms. Every registered gate passed: self-checks held; both bundle
predicates were non-vacuous (`std` 0.23–0.29); eligibility and donor support held;
bundle room was present (`phi1` 0.18–0.20, `phi2` 0.24–0.26); endpoint audit
stayed `0.006–0.009` (≪ `OE_BAND`); the `own_delta` ceiling closed fully
(control 1.00, m-gram 1.00, `|oob|` 1.00). So the run reached the joint
adjudication.

Held-out quantities (ranges across the four seeds):

| quantity | phi1-matched arm | phi2-matched arm |
|---|---|---|
| target control | 0.67–0.70 | 0.76–0.79 |
| bundle co-movement | 0.63–0.72 | 0.63–0.67 |
| co-movement floor (max mis/shuf) | 0.22–0.30 | 0.14–0.35 |
| coupling margin (co − floor) | 0.41–0.44 | 0.27–0.53 |
| out-of-bundle control `\|c(phi4)\|` | 0.37–0.54 | 0.36–0.62 |
| `own_delta` `\|c(phi4)\|` (replacement ref) | 1.00 | 1.00 |
| m-gram closure | 0.50–0.52 | 0.49–0.55 |
| phi4 room | 0.025–0.029 (**marginal, flagged 4/4**) | 0.030–0.037 |
| selected alpha | 1.0–1.5 | 1.5–2.0 (extrapolated) |
| retention | 0.90–0.98 | 0.88–0.97 |

### What the run establishes

**The bundle is genuinely, directionally coupled.** `bundle_co` cleared
`COUPLE_MIN` *and* beat its no-information co-movement floors by `C_MARGIN` on
every seed, both arms (margins 0.27–0.53). The F1 fix did its job: this is
directed coupling, not the undirected same-position co-movement that the shuffled
floor (0.14–0.35) already produces. A directed near-manifold delta built to move
one bundle predicate co-moves the other toward source, beyond chance.

**The move is partial, not literal replacement.** m-gram closure was ~0.50 for
the matched delta versus 1.00 for `own_delta`; the matched move reproduces about
half the full continuation distribution, not all of it.

**Read geometry shows bundle structure that the write does not respect.**
`cos(phi1,phi2)` was 0.54–0.65 while `cos(phi*,phi4)` was mostly 0.10–0.27 (one
outlier 0.52). The two bundle reads are more aligned with each other than with
`phi4` — yet the matched delta still moved `phi4`. Read-level bundling did not
translate into causal sparing.

### Why the verdict is `BROAD_STATE_REPLACEMENT`

`oob_spared` failed on every seed: `|c(phi4)|` was 0.36–0.62, above the
`OOB_MAX=0.35` ceiling. The coupling axis passed but the sparing axis did not, so
the four-way classifier returned `BROAD_STATE_REPLACEMENT`. The bundle does move
*more* than `phi4` (the `SEP_MARGIN` target−`|oob|` check passed, ~0.15–0.34), but
`phi4` still moves above the absolute sparing ceiling.

### The load-bearing caveat

The verdict turns **entirely** on `phi4`: m-gram (0.50) is well under its bound
and the separation margin passed, so `BROAD` is driven solely by `|c(phi4)| >
0.35`. And `phi4`'s room is small — 0.025–0.037, roughly 6–8× smaller than the
bundle predicates' — placing the `phi1` arm in the `OOB_ROOM_MARGINAL` band on
all four seeds. `predicate_control` divides by room, so this closure fraction is
amplified and high-variance. In absolute marginal terms `phi4` moves ~0.07–0.10
(a real, non-trivial drag, not noise), but the fraction overstates it relative to
the higher-room bundle. This is exactly the F2 risk the pre-run review flagged,
materialising: the headline turned on an underpowered control.

### Conclusion

The carried-forward claim:

```text
BROAD_STATE_REPLACEMENT(stack_state_bundle)
```

On `pstack-L4`, at L1, `m=3`, positions `{10,18}->{26,34}`, the exp-34
near-manifold matched deltas **do not** redeem exp 34's `NONSPECIFIC` as clean
joint-variable control. `phi1` and `phi2` are causally coupled — a directed
delta co-moves both beyond no-information floors — but the same move also drags
the out-of-bundle binding predicate `phi4` above the sparing ceiling, so the move
is broad rather than predicate- or bundle-specific. The joint-variable rescue
(`JOINT_STACK_VARIABLE`) did not occur; nor did the clean independence reading
(`SEPARABLE_PREDICATES`), which the F1 floor kept reachable but which the data
did not select.

Two findings deserve to carry forward with different confidence. **High
confidence:** the bundle is real and directionally coupled, and the residual
move at L1 is broad-ish (it moves coupled and uncoupled predicates together),
consistent with exps 33/34 — do **not** treat rank-1 residual oblique writes as a
clean predicate-specific primitive for these targets. **Lower confidence:** the
*degree* to which `phi4` is dragged, because the one available out-of-bundle
predicate has too little room to adjudicate sparing cleanly at L1. The experiment
under-powers its own load-bearing axis.

Routing: the literal branch points to I4 patch-point (find a component/layer
where the bundle moves but an out-of-bundle predicate is spared) or
consolidation. But the marginal-room caveat reweights this: `pstack`'s registered
predicate inventory cannot settle the separability question, because it has no
high-room out-of-bundle control. That argues the more informative next move is the
**exit gate** — a richer toy designed with a separable, high-room out-of-bundle
predicate — rather than another residual-level probe on `pstack`, which would
re-inherit the same underpowered control. An I4 patch-point pass is defensible
only if it can be scored against a better-separated control than `phi4`.

This negative is scoped to this donor rule, layer (L1), horizon (`m=3`),
positions, the `{phi1,phi2}` bundle, the single `phi4` out-of-bundle control, and
the `pstack-L4` checkpoint. It does not prove no joint write exists, nor that the
bundle is unwritable at another patch point.
