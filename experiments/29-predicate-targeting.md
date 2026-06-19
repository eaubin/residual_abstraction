# Experiment 29 — predicate-targeting measurement pilot — CONCLUDED

**Script:** `scripts/predicates/predicate_targeting.py`; library module
`predicates.py`.

**Status: concluded.** Canonical output:
`out/exp29_pstack-L4.txt`.

## Question

`ORIGINAL_SIN.md` argues that the next conceptual reduction to attack is
the completion side: the current battery scores the full finite-horizon
distribution, while the original idea needs named abstractions of futures.
The cheapest first step is a *within-horizon predicate*: a Boolean mask
`phi: V^m -> {0,1}` over the existing `m=3` completion distribution. Its
truth probability

```text
p_phi(q) = sum_c phi(c) q(c)
```

is a rank-1 abstraction of the completion distribution. It adds a named
completion-side coordinate without changing the exact-oracle substrate.

This experiment asks one narrow question:

> Is the proposed predicate-targeting measurement stack non-vacuous and
> interpretable on `pstack`, and does it give routing evidence about whether
> predicate targets differ geometrically from the PCA/CEGAR directions already
> seen in exp 26?

This is not the predicate phase. It does not run CEGAR against a predicate.
It does not test predicate composition. It does not claim to find minimal
predicate-sufficient residual abstractions. It is a construct-validity pilot
for the measurement stack that a later predicate phase would use.

## Estimand

For each registered predicate `phi` and each fresh seed, estimate the
following object:

```text
the affine readout direction w_phi fitted from centered L1 residuals to
observable model p_phi, evaluated by:

1. held-out affine and kNN decode R^2;
2. whether a rank-1 Euclidean same-read/write interchange patch along w_phi
   causally moves model p_phi toward the source-run p_phi;
3. the principal angle of w_phi to the registered PCA_k and CEGAR_k
   subspaces;
4. source/target obs-exact p_phi agreement.
```

The ugly qualification is intentional. This experiment is about one probe
construction and one patch class. A positive result means:

```text
a registered affine predicate readout direction, under this rank-1 Euclidean
patch, causally moves predicate probability and has the measured geometry.
```

It does **not** mean:

- the minimal predicate-sufficient abstraction has that geometry;
- all predicates behave that way;
- a different read/write patch parameterization would agree;
- CEGAR-against-`phi` would necessarily find the same direction.

## Why This Experiment Is This Narrow

`pstack` is already known to be near variance-mimicry: exp 26 found one
stable behavioral reference with `cegar` close to `pca`, and exp 27 showed
the battery transfers under a declared conventional anchor. Therefore
`ALIGNED` is expected if coverage holds, and is not a failure. It would say
the current `pstack` substrate and the registered predicates, at the
registered coverage floor, do not expose new residual geometry under this
measurement stack.

`DIVERGENT` would be stronger routing evidence: a predicate direction that is
decode-sufficient, causally movable, exact-sound, and outside both PCA and
the CEGAR core. That would justify pre-registering a real CEGAR-against-phi
experiment. If no such daylight appears, the next move is probably a toy
designed backward from predicates, not more mining of `pstack`.

## Registered Command

```bash
python3 scripts/predicates/predicate_targeting.py --outdir out/pstack-L4
```

Review-only self-tests:

```bash
python3 predicates.py
python3 scripts/predicates/predicate_targeting.py --selftest
```

Run through `uv run python ...` in this environment.

## Oracle Discipline

All proposal and geometry quantities are observable/model-only:

- ridge and kNN probes are fit to model `p_phi(q_un)`;
- patches are scored against model source-run `p_phi(q_src)`;
- PCA and CEGAR directions use residuals and observable model behavior.

The exact oracle is used only for evaluation: source/target exact
`p_phi*(belief) = sum_c phi(c) mgram(belief)[c]`. It gates whether an
observable predicate verdict can support a geometry decision. No threshold,
probe, predicate, or direction is tuned using exact values.

## Registered Predicates

The library `predicates.py` defines four Boolean masks over `V^3` for
`pstack` (`0,1` opens; `2,3` matching closes; `4,5` neutral terminals).
They are not meant to be a rich predicate language. They are a small test
suite with one common predicate, two structured predicates, and one likely
rare predicate.

| predicate | definition | registered role |
|---|---|---|
| `phi1_next_closes` | first continuation token is `2` or `3` | common/high-variance control; likely aligned |
| `phi2_net_return` | net depth change over three tokens is `<= -1` | depth-flavored structured target |
| `phi3_all_neutral` | all three continuation tokens are neutral | hidden-mode/terminal target; may be rare or no-room |
| `phi4_first_matched` | first opened bracket inside the window is closed by matching type inside the window | minimal within-window temporal/binding target |

Minimal mask algebra (`not`, `and`, `or`) is self-tested in the library but
is not used for composition claims in this experiment.

## Design

**Seeds and data.** Four fresh pair/basis seeds `300..303`, fixed
checkpoint `out/pstack-L4`, patch point L1, `m=3`, positions
`{10, 18, 26, 34}`, discovery pairs `320`, evaluation pairs `1024`.

**Core and PCA comparators.** For each seed, recompute the standard CEGAR
core on the discovery PairSet (`eps=0.05`, `eps_drop=0.01`, `k_max=10`,
clipped to `k<=8`) and fit PCA on the evaluation residual train split.
Angles are measured from the rank-1 `w_phi` direction to PCA_k and CEGAR_k,
where `k` is the recomputed core dimension.

**Decode.** Fit an affine ridge readout from centered residuals to observable
target-run `p_phi`. Report held-out linear `R^2`. Also report kNN `R^2`
with `k=10`; linear failure plus kNN success is an `INTERPRETER_GAP`, not
absence of predicate information.

**Intervention.** Patch the target prefix toward the source prefix with the
rank-1 Euclidean projector `w_phi w_phi^T`. Score predicate closure:

```text
c_phi(P) =
  [MSE(p_un, p_src) - MSE(p_P, p_src)]
  /
  [MSE(p_un, p_src) - MSE(p_full, p_src)]
```

where all `p_*` are observable model predicate probabilities. If the
denominator is not positive, the predicate has `NO_PATCH_ROOM`: even the
full patch does not create enough predicate-level movement to interpret
rank-1 predicate geometry.

**Exact audit.** Compute mean absolute source/target endpoint drift:

```text
oe = max(
  mean |p_un_model - p_tgt_exact|,
  mean |p_src_model - p_src_exact|
)
```

This audits whether the observable predicate endpoints used by the
intervention are calibrated to exact predicate truth. It does not make any
claim about an exact truth for the off-manifold patched distribution.

## Per-Seed Verdicts

For each `(phi, seed)`, exactly one verdict is assigned:

| verdict | condition | meaning |
|---|---|---|
| `PREDICATE_VACUOUS` | `std(p_phi) < 0.05` | predicate barely varies on the eval set |
| `NOT_DECODABLE` | linear `R^2 < 0.50` and kNN `R^2 < 0.50` | predicate not recovered by tested interpreters |
| `INTERPRETER_GAP` | linear `R^2 < 0.50`, kNN `R^2 >= 0.50` | predicate present but not affinely decoded |
| `NO_PATCH_ROOM` | decode-sufficient, but full-patch denominator not finite | intervention geometry cannot be interpreted |
| `ECHO` | decode-sufficient, but `c_phi(w) < 0.50` or `c_phi(w)-c_phi(rand) < 0.20` | decodable but not causally moved by this patch |
| `ALIGNED` | causal and `angle(w,PCA_k) <= 10°`, `angle(w,core_k) <= 10°` | registered direction lies inside both comparators |
| `DIVERGENT` | causal and both angles `> 10°` | registered direction lies outside both comparators |
| `MIXED_GEOMETRY` | causal and off exactly one comparator | geometry differs from one reference but not the other |

`ALIGNED`, `DIVERGENT`, `MIXED_GEOMETRY`, and `ECHO` are headline states and
require exact endpoint soundness (`oe <= 0.10`) on the seeds that reproduce
that state. If not, the aggregate is `OBS_EXACT_DRIFT(phi)`.

## Multi-Seed Aggregation

A predicate-level aggregate reproduces only if one per-seed verdict holds in
at least `3/4` seeds. Soundness is checked on those same reproducing seeds,
not merely anywhere in the four-seed set.

Top-level decision precedence:

1. `OBS_EXACT_DRIFT(phi...)` — a reproduced headline state failed exact
   endpoint calibration. Do not use it for geometry routing.
2. `DIVERGENT(phi...)` — at least one exact-sound predicate direction is
   causal and outside both PCA and core.
3. `MIXED_GEOMETRY(phi...)` — at least one exact-sound predicate direction
   is causal and off exactly one comparator.
4. `ALIGNED(phi...)` — at least three registered predicates are exact-sound
   and causal-aligned, and every registered predicate that is neither
   `PREDICATE_VACUOUS` nor `NO_PATCH_ROOM` is aligned. The excluded
   predicates are reported in the per-phi aggregate and do not support
   geometry claims. The `>=3` coverage floor is required because `ALIGNED`
   routes future work away from more `pstack` mining.
5. `PARTIAL_ALIGNED(aligned=...; blocked=...)` — at least one predicate is
   exact-sound and aligned, but either fewer than three predicates align or
   another non-excluded predicate is `ECHO`, `INTERPRETER_GAP`,
   `NOT_DECODABLE`, or `SEED_UNSTABLE`. This is limited coverage, not a
   routing result.
6. `ECHO(phi...)` — reproduced decodable predicates do not causally move
   under this rank-1 patch class, and no causal predicate reproduces.
7. `INTERPRETER_LIMIT(phi...)` — predicates are not linearly decodable, or
   only kNN-decodable, under the tested interpreters.
8. `NO_PATCH_ROOM(phi...)` — decodable predicates have no full-patch
   predicate room; do not interpret their geometry.
9. `PREDICATE_VACUOUS(phi...)` — predicates do not vary enough to test.
10. `SEED_UNSTABLE(phi...)` — no verdict reproduces.

## Interpretation Map

| decision | interpretation | next action |
|---|---|---|
| `OBS_EXACT_DRIFT` | observable predicate endpoints do not match exact predicate truth | repair predicate scoring/calibration before geometry claims |
| `DIVERGENT` | the registered predicate readout/patch finds non-PCA/core causal geometry | pre-register CEGAR-against-phi |
| `MIXED_GEOMETRY` | predicate direction separates from one comparator but not both | inspect which comparator failed; do not claim full divergence |
| `ALIGNED` | at least three predicates align, and every non-excluded registered predicate adds no rank-1 geometry beyond PCA/core on `pstack` | move to a designed-backward toy; do not keep mining `pstack` with this pilot |
| `PARTIAL_ALIGNED` | some predicates align, but coverage is too low because too few predicates aligned or another non-excluded predicate is echo/interpreter-limited/unstable | report the limited coverage; do not use it to redirect the program |
| `ECHO` | predicate decodes but this patch does not causally control it | future predicate work must score interventions, not decode |
| `INTERPRETER_LIMIT` | tested interpreters do not recover the predicate | change interpreter class before causal geometry claims |
| `NO_PATCH_ROOM` | full patch does not move the predicate enough | change predicate, horizon, PairSet, patch point, or process |
| `PREDICATE_VACUOUS` | predicate is too rare/flat on this distribution | change predicate/horizon/distribution |
| `SEED_UNSTABLE` | geometry or verdict does not reproduce | increase stability checks or narrow the claim |

## Predictions

- **P1 (self-checks and substrate; enforced).** Registered config,
  PairSet self-checks, `predicates.py` self-test, and verdict self-test pass.
- **P2 (predicate variation; descriptive).** `phi1`, `phi2`, and `phi4`
  should vary enough to test; `phi3` may be `PREDICATE_VACUOUS` or
  `NO_PATCH_ROOM`.
- **P3 (decode; ~75%).** Non-vacuous predicates are likely linearly
  decodable. `INTERPRETER_GAP` on `phi4` would be an informative narrower
  finding.
- **P4 (geometry routing; `ALIGNED` likely if coverage holds).** On
  near-mimicry `pstack`, exact-sound causal predicate directions likely lie
  inside PCA/core. This routes to a designed-backward process only if at
  least three predicates align and every registered predicate that is neither
  `PREDICATE_VACUOUS` nor `NO_PATCH_ROOM` is aligned. If fewer than three
  predicates align, or if some align while others are
  echo/interpreter-limited/unstable, the result is `PARTIAL_ALIGNED`, a
  limited-coverage finding rather than a program-routing result.
  `DIVERGENT` is the daylight case. `ECHO` is also live.

## Scope And Local Assumptions

- Indexed by `pstack`, fixed checkpoint `out/pstack-L4`, L1, `m=3`,
  positions `{10,18,26,34}`, four registered predicates, affine/kNN
  interpreters, Euclidean rank-1 same-read/write patches, and seeds
  `300..303`.
- Predicate targets are within-horizon masks over `V^3`; no trajectory-level
  predicate and no predicate composition claim is tested.
- Geometry claims refer only to the fitted affine readout direction `w_phi`,
  not to a minimal sufficient predicate abstraction.
- The patch class is intentionally narrow and known to be a potential
  limitation from earlier read/write work.
- Exact predicate values are evaluation-only.

## Known Invalid Stronger Claims

Do not conclude from this experiment that:

- predicate abstractions in general align or diverge from PCA/core;
- `pstack` has no predicate-specific structure;
- a rank-1 affine readout is the minimal predicate abstraction;
- CEGAR-against-phi is unnecessary in general;
- trajectory-level predicates would behave like these within-horizon masks.

## Expected Output

For each seed, print `k_core` and per-predicate measurements:
`std(p_phi)`, linear/kNN `R^2`, angles to PCA/core, `c_phi(w)`,
`c_phi(rand)`, source/target obs-exact endpoint drift, and the per-seed
verdict. Then print per-predicate aggregates and the top-level decision.

---

## Results

Canonical command:

```bash
uv run python scripts/predicates/predicate_targeting.py --outdir out/pstack-L4
```

Output artifact: `out/exp29_pstack-L4.txt`.

**Result, in this experiment's local verdict vocabulary:** the top-level
decision string is

```text
ECHO(phi1_next_closes,phi2_net_return)
```

Treat that string as an exp-29-local summary of this script's verdict
partition, not as a project-level label. The names inside the parentheses are
the two registered within-horizon masks that reproduced the same local
condition on all four seeds:

- first continuation token is a close token;
- three-token net depth change is at most `-1`.

For those two masks, the affine readout clears the registered decode threshold
on every seed, and the observable/exact endpoint audit is sound by a wide
margin (`oe <= 0.010`, threshold `0.10`). But the registered rank-1 Euclidean
same-read/write patch along the fitted readout direction closes essentially
none of the predicate-probability gap to the source (`c_w` prints as `0.00`
or `-0.00` on every seed). In ordinary language: under this readout and patch
class, the model representation contains linearly decodable information about
these two predicate probabilities, but the corresponding rank-1 write does not
causally control those probabilities.

The other two registered masks do not support geometry claims. The all-neutral
three-token mask is flat on this distribution (`std ~= 0.015`, below the
registered `0.05` floor). The within-window first-match mask varies enough to
test (`std ~= 0.099..0.103`) but is not recovered by either tested interpreter
(`linR2 ~= 0.19..0.25`, `knnR2 ~= 0.20..0.27`, both below `0.50`).

The registered `ALIGNED` routing condition did not fire. This run therefore
does not say that the registered predicates add no rank-1 geometry beyond
PCA/core on `pstack`, and it does not license moving away from `pstack` on
the basis of alignment. The geometry angles are near `90°` for the decoded
predicate readouts, but because the registered intervention fails, those
angles are descriptive only; they are not `DIVERGENT` evidence and should not
trigger predicate-CEGAR by themselves.

## Conclusion

Experiment 29 validates a narrow measurement lesson and blocks the tempting
geometry shortcut. On `pstack` at L1, `m=3`, positions `{10,18,26,34}`, seeds
`300..303`, affine/kNN interpreters, and Euclidean rank-1 same-read/write
patches, predicate readout success is not enough to support a causal geometry
claim. Two registered completion predicates are linearly readable and
exact-calibrated, but the corresponding rank-1 writes do not move the
predicate probabilities. One predicate is distribution-vacuous; one is
interpreter-limited.

The result is informative because it prevents a bad next experiment: do not
take a decodable predicate direction, measure its angle to PCA/core, and treat
that as evidence about predicate-sufficient residual geometry. For this
measurement stack, intervention is not a confirmatory add-on; it is the gate
that decides whether the geometry measurement is meaningful.

What carries forward is not the local verdict name, nor the four local
predicate identifiers. What carries forward is the design constraint:
predicate-targeting experiments must separate (1) predicate variation, (2)
interpreter recovery, (3) observable/exact endpoint calibration, and (4)
causal control under the registered patch class before making geometry or
routing claims. A later predicate phase can change the predicate language,
horizon, process, interpreter, or patch class, but it should keep those
checks separate.

This result also argues against more unaimed `pstack` mining with the current
pilot. The failure mode is not "predicates are impossible" and not "`pstack`
has no predicate structure." It is narrower: these registered within-horizon
masks, on this near-mimicry substrate, under this affine-readout/rank-1-write
stack, do not produce causally interpretable predicate directions. The next
useful registration should either design a toy backward from predicates and
known causal directions, or explicitly revise the intervention/read-write
class before returning to predicate geometry on `pstack`.
