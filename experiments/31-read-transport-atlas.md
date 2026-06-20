# Experiment 31 — Read-transport atlas diagnostic on pstack-L4 (pre-I2) — CONCLUDED

**Script:** `scripts/interventions/read_transport_atlas.py`.

**Status: concluded.** Canonical output: `out/exp31_pstack-L4.txt`.
Diagnostic-only. This experiment produces **no** writability, controllability,
or intervention claim. It produces a typed *routing* decision about a
read/representational-geometry question, to run *before* committing to
benchmark step I2.

**Result:** `ATLAS(phi1_next_closes=POSITION_SPECIFIC_READ, phi2_net_return=POSITION_SPECIFIC_READ)`,
both targets 4/4 seeds, `positions_exchangeable=True` 4/4. See Results/Conclusion.

## Question

Exp 30 (I1) returned `FIXED_READ_LIMIT(phi1_next_closes,phi2_net_return)`: the
exp-29 affine predicate reads decode on discovery positions `{10,18}` but fail
the registered held-out-position `R2` gate at `{26,34}`, so I1 could not
adjudicate write failure. That verdict, `FIXED_READ_NOT_TRANSPORTED`, collapses
two different statements:

1. **(representational)** the predicate is not linearly readable at L1 on the
   held-out positions at all; or
2. **(transport-of-a-fixed-read)** the predicate *is* readable there, but a
   single global affine read fit only on `{10,18}` does not carry over.

Exp 30 only ever fit on discovery and tested on held-out, so it measured exactly
one off-diagonal transfer cell and never the in-place held-out diagonal. This
experiment asks the measurement exp 30 skipped:

```text
For each target predicate at L1 on pstack-L4, is the predicate linearly readable
IN PLACE at the held-out positions, and if so, is the readout direction shared
with the discovery positions or position-specific?
```

This is a read/representational-geometry question. **No interventions, no
patching, no learned writes, no CEGAR.** If any patch were added the experiment
would have to be re-scoped as an intervention experiment with the corresponding
baselines (see Non-goals).

## Registered Command

After preregistration review only:

```bash
uv run python scripts/interventions/read_transport_atlas.py \
  --outdir out/pstack-L4 | tee out/exp31_pstack-L4.txt
```

Review-only checks (allowed before approval):

```bash
uv run python scripts/interventions/read_transport_atlas.py --selftest
uv run python -m py_compile interventions.py \
  scripts/interventions/read_transport_atlas.py
```

## Scope Indices

| index | value |
|---|---|
| process/checkpoint | `pstack-L4`, registered config from exp 29/30 |
| measurement point | residual stream L1 target-run residual at the scored prefix position |
| horizon | `m=3` within-horizon completion distribution |
| target predicates | `phi1_next_closes`, `phi2_net_return` |
| control predicates | `phi3_all_neutral`, `phi4_first_matched` (read controls only) |
| read class | affine ridge readout from per-position-centered L1 residuals to observable model `p_phi` |
| grouped discovery bin | positions `{10,18}`, 512 pairs/seed (exp-30 construction) |
| grouped held-out bin | positions `{26,34}`, 1024 pairs/seed (exp-30 construction) |
| single-position bins | `{6,10,14,18,22,26,30,34}`, 512 pairs/seed each |
| seeds | `400..403`, fresh relative to exp 29/30 |
| exact oracle use | none; the atlas is observable-only (no patch endpoints to audit) |

The grouped bins reuse the exp-30 pair construction (same per-bin pair counts
and seed offsets) so the discovery→held-out off-diagonal cell is a tripwire
reproduction of exp 30. The single-position bins are the new dense atlas.

## Design (observational only)

For each seed in `{400..403}`:

1. Build the two grouped PairSets and the eight single-position PairSets at L1.
   Run `discover.self_checks` on each (known-answer guard). Grouped bins reuse
   the exp-30 pool seeds (`seed+111` disc, `seed+222` held); each single-position
   bin `t` uses the fresh independent pool seed `seed+700+t`, so the atlas never
   re-reads the grouped pools.
2. For each bin, run the **living** evaluator once on the unpatched target run
   via `discover.PairSet.run(model, None)` — which delegates to
   `midstream.chain_probs` — and reduce the returned `(n, V^m)` joint to
   observable `p_phi` per registered predicate via `predicates.obs_pphi`. No
   prefix patch is applied (`P=None`). The residual frame comes from the living
   helper `interventions.pairset_residual_frame`, and per-position centering
   from `abstraction.center_by_position` with the bin's train mask.

For each target predicate the atlas measures, per the proposal:

- **(1) In-place decodability per bin.** Fit the affine ridge read on a train
  half of the bin's pairs (per-position centered with the train mask, via
  `abstraction.center_by_position`) and score `R2` on the held-out half *of the
  same bin*. The headline separator is in-place `R2` at the grouped held bin
  `{26,34}` — the diagonal exp 30 never computed.
- **(2) Cross-position transfer matrix.** Fit the affine map on bin `i`'s train
  half, evaluate it on bin `j`'s test half, for all single-position `(i,j)`. The
  train/test split keeps the diagonal honest (fit and score never share rows),
  so the diagonal is genuine in-place decodability and the off-diagonal is read
  carry. Exp 30 measured exactly one off-diagonal cell.
- **(3) Direction similarity.** Cosine between unit read covectors `c_i`. High
  cosine with collapsing transfer `R2` implies scale/bias drift of a shared
  direction, not a different direction.
- **(4) Predicate distribution by position.** Mean and std of observable `p_phi`
  per single position, to check whether the positions are even exchangeable for
  this predicate (whether positions are a fair transfer axis).

### Read-prediction convention (difference from exp 30, on the record)

The atlas scores every `R2` — in-place and transfer — with the **full fitted
affine map `(w, b)`**: `yhat = Rc @ w + b`. Exp 30's `decode_heldout` applied
the *unit* covector `c = w/||w||` with the `w`-scale bias `b`
(`yhat = Rc @ c + b`), a scale mismatch that is harmless for the qualitative
"does it transport" verdict but changes the decimal transfer `R2`. The atlas
therefore reproduces exp 30's transfer failure **qualitatively** (transfer
`R2 < R2_MIN`), not to the decimal; the unit covector is used only for the
direction-cosine in measurement (3). This is registered so a result reviewer
does not read a decimal mismatch as a discrepancy.

## Measured Quantities and Thresholds

Per `(target, seed)`:

| symbol | meaning |
|---|---|
| `r2_inplace_disc` | in-place `R2` at grouped disc bin `{10,18}` (exp-30 reproduction) |
| `r2_inplace_held` | in-place `R2` at grouped held bin `{26,34}` (**key separator**) |
| `r2_shuffle_held` | label-shuffle `R2` floor at the held bin (fit against position-shuffled `p_phi`) |
| `r2_transfer_disc_held` | fit at disc, evaluate at held (the single exp-30 off-diagonal cell) |
| `cos_held_disc` | cosine between disc and held unit read covectors |
| `std_disc`, `std_held` | std of observable `p_phi` on each grouped bin (vacuity guard) |
| `pphi_pos_spread` | max−min of per-single-position mean `p_phi` (exchangeability) |

Registered thresholds (reusing exp 30 where applicable):

```text
R2_MIN        = 0.50   # "readable" / "transports" threshold (exp 30)
VAR_MIN       = 0.05   # p_phi std vacuity floor (exp 30)
FLOOR_MARGIN  = 0.10   # in-place R2 must beat its own shuffle floor by this
COS_SHARED    = 0.70   # read covectors counted as a shared direction
PDIST_MAX     = 0.15   # per-position p_phi mean spread above which positions
                       # are a poorly-chosen transfer axis
LAM           = 1e-2   # ridge penalty (exp 30)
SEED_MAJORITY = 3      # >=3/4 seeds for an aggregate, else SEED_UNSTABLE
```

## Per-Seed Verdicts

For each `(target, seed)` the script assigns exactly one branch:

| branch | condition | interpretation |
|---|---|---|
| `TARGET_VACUOUS` | `std_disc < VAR_MIN` or `std_held < VAR_MIN` | predicate too flat to read on a bin |
| `DISC_READ_FAILED` | `r2_inplace_disc < R2_MIN` | exp-30 premise (disc decodes) not reproduced; tripwire |
| `NOT_READABLE_LATE` | `r2_inplace_held < R2_MIN` or `r2_inplace_held − r2_shuffle_held < FLOOR_MARGIN` | predicate not linearly present at L1 at later positions |
| `SHARED_READ_SCALE_DRIFT` | readable late and `cos_held_disc >= COS_SHARED` | shared read direction; exp-30 negative is scale/bias drift |
| `POSITION_SPECIFIC_READ` | readable late and `cos_held_disc < COS_SHARED` | predicate readable late but the read direction is position-specific |

Separately, a boolean `positions_exchangeable = (pphi_pos_spread <= PDIST_MAX)`
is recorded per `(target, seed)`. It overlays the branch: when positions are not
exchangeable, the position transfer axis itself is suspect regardless of branch.

## Multi-Seed Aggregation

A target aggregate is the branch appearing in at least `3/4` seeds; otherwise
`SEED_UNSTABLE`. `positions_exchangeable` aggregates by the same majority rule.

The top-level decision is a routing string, not a positive/negative claim:

```text
ATLAS(phi1_next_closes=<verdict>, phi2_net_return=<verdict>)
```

with the per-target routing and the exchangeability overlay printed beneath it.
No precedence ordering implies one target's verdict dominates another's; both
target aggregates are load-bearing.

## Predictions

- **P1 (guards; enforced).** Config guard passes, PairSet known-answer
  self-checks pass, and the `--selftest` passes.
- **P2 (exp-30 reproduction tripwire; likely).** For both targets, the grouped
  disc in-place read decodes (`r2_inplace_disc >= R2_MIN`) and the disc→held
  transfer is low (`r2_transfer_disc_held < R2_MIN`), qualitatively reproducing
  exp 30. A `DISC_READ_FAILED` branch would mean the atlas is not measuring the
  exp-30 object and blocks interpretation of the rest.
- **P3 (headline; uncertain).** The in-place held-out `R2`. If high
  (`>= R2_MIN`, beating shuffle), the predicate *is* readable late and exp 30's
  negative is about transport, not representation. If low, the predicate is not
  linearly present at L1 at later positions. Either outcome is informative and
  routes differently.
- **P4 (direction; conditional on P3 high).** The disc/held covector cosine
  decides `SHARED_READ_SCALE_DRIFT` vs `POSITION_SPECIFIC_READ`.
- **P5 (exchangeability; report).** Per-position `p_phi` mean/std. If the base
  rate shifts sharply across positions (`pphi_pos_spread > PDIST_MAX`), positions
  are a poor transfer axis and the overlay fires.
- **P6 (controls; expected).** `phi3_all_neutral` stays vacuity-limited and
  `phi4_first_matched` stays non-decodable as a read on both grouped bins, and
  every label-shuffle floor sits near 0. Controls are reported, never promoted to
  targets.

## Interpretation / Routing Map

| aggregate | interpretation of the exp-30 negative | routes to |
|---|---|---|
| `SHARED_READ_SCALE_DRIFT` | scale/bias drift of a shared read | trivial per-position calibration, then re-ask I1 cleanly (write question reopens) |
| `POSITION_SPECIFIC_READ` | predicate readable late but read is position-specific | I2 with **position-conditioned** reads; re-run I1 with a transport-valid read |
| `NOT_READABLE_LATE` | predicate not linearly present at L1 at later positions | I4 / depth: patch point or layer is wrong there, not a read-freedom problem |
| `positions_exchangeable = False` (overlay) | positions are a poorly-chosen transfer axis | switch transfer axis (fresh-seed at matched positions) before further intervention work |
| `DISC_READ_FAILED` | exp-30 premise not reproduced | fix substrate/measurement before reinterpreting exp 30 |
| `SEED_UNSTABLE` | branch not reproduced across seeds | the routing question is not stably answerable on this substrate |

Three of the four substantive outcomes imply **I2 as currently sketched would be
premature or mis-targeted**: adding read freedom to "fix" a failure that has not
been characterized, possibly fighting a position-transport wall that read
freedom cannot address.

## Halt Conditions

The run halts if:

- the checkpoint config differs from the registered `pstack-L4` config; or
- any PairSet known-answer self-check fails.

There is no I0 routing gate: I0 routes the *intervention* benchmark, and this
diagnostic applies no patch, so the relevant preconditions are a correct
checkpoint and sound PairSets only.

## Reviewable Failure Modes

- broad labels replacing the narrow construct: every verdict is an L1 / `pstack`
  / `m=3` / position-split **readability** claim, never a writability claim;
- spurious high `R2` from few pairs or low variance: separated by the
  label-shuffle floor (`FLOOR_MARGIN`) and the `VAR_MIN` vacuity guard;
- treating positions as a fair transfer axis when they are not: separated by the
  `positions_exchangeable` overlay (`pphi_pos_spread`);
- conflating shared-direction scale drift with a different direction: separated
  by the covector cosine;
- mistaking the atlas for an intervention result: the script applies no patch
  and emits no control/specificity number.

## Non-goals / Scope Guard

- No claim about whether the predicate is writable or controllable.
- No new process training; existing `pstack-L4` only.
- No interventions; if any patch is added the experiment must be re-scoped as an
  intervention experiment with the corresponding baselines.
- MPS acceleration is deferred: the registered run uses the living CPU evaluator
  `midstream.chain_probs`; moving to MPS would require modifying that shared
  evaluator, which is out of scope for a diagnostic and would be its own reviewed
  change.

## Results

Registered command, run after preregistration approval:

```bash
uv run python scripts/interventions/read_transport_atlas.py \
  --outdir out/pstack-L4 | tee out/exp31_pstack-L4.txt
```

Decision (routing string, not a positive/negative claim):

```text
ATLAS(phi1_next_closes=POSITION_SPECIFIC_READ, phi2_net_return=POSITION_SPECIFIC_READ)
```

Both target aggregates reproduced `POSITION_SPECIFIC_READ` in all four seeds,
and `positions_exchangeable=True` in all four seeds:

| target | per-seed branch | aggregate | exchangeable |
|---|---|---|---|
| `phi1_next_closes` | 4/4 `POSITION_SPECIFIC_READ` | `POSITION_SPECIFIC_READ` | 4/4 `True` |
| `phi2_net_return` | 4/4 `POSITION_SPECIFIC_READ` | `POSITION_SPECIFIC_READ` | 4/4 `True` |

**The headline separator exp 30 never computed — in-place held-out `R2` — is
high.** Both predicates are linearly readable *in place* at the held-out
positions `{26,34}`, at `R2` comparable to the discovery bin, and far above
their own (negative) label-shuffle floors:

| target | in-place disc `R2` (exp-30 premise) | in-place held `R2` (**key**) | shuffle floor (held) |
|---|---:|---:|---:|
| `phi1_next_closes` | 0.546–0.644 | 0.555–0.642 | −0.072…−0.148 |
| `phi2_net_return` | 0.648–0.726 | 0.678–0.753 | −0.088…−0.155 |

**The exp-30 tripwire (P2) holds.** The single disc→held off-diagonal cell —
the only transfer exp 30 measured — collapses qualitatively below `R2_MIN`
(mostly negative) on every seed, while the disc in-place read decodes:

| target | disc→held transfer `R2` (per seed) |
|---|---|
| `phi1_next_closes` | 0.122, −0.317, −2.075, −0.523 |
| `phi2_net_return` | 0.165, −0.252, −1.147, −0.050 |

**The read direction is position-specific, not a shared direction with scale
drift.** The disc/held unit read-covector cosines sit near zero on every seed,
well below `COS_SHARED = 0.70`:

| target | cos(held, disc) (per seed) |
|---|---|
| `phi1_next_closes` | −0.028, −0.122, 0.235, 0.219 |
| `phi2_net_return` | 0.096, −0.096, 0.243, 0.141 |

The dense single-position transfer matrix confirms this at finer grain: every
diagonal cell is decodable in place (`R2 ≈ 0.49–0.83` across the eight positions
`{6,10,14,18,22,26,30,34}`), while essentially every off-diagonal cell is
strongly negative — a read fit at one position does not carry to any other.
This is read-direction position-specificity across the whole atlas, not just the
two grouped bins.

**Positions are an exchangeable transfer axis (P5).** Per-position mean `p_phi`
spread is small on every seed (`phi1`: 0.019–0.045; `phi2`: 0.022–0.044), all
below `PDIST_MAX = 0.15`, so the exchangeability overlay is `True`. The base
rate is stable across positions; the transfer axis itself is not the problem.

**Controls behaved as registered (P6).** `phi3_all_neutral` is vacuity-limited
(`std ≈ 0.015 < VAR_MIN`) and does not decode (in-place `R2` 0.17–0.33).
`phi4_first_matched` is non-decodable as a read on both grouped bins (in-place
`R2` 0.07–0.30 disc, 0.14–0.25 held — below `R2_MIN`). Every label-shuffle floor
sits near or below zero. Controls are reported, never promoted to targets.

All guards (P1) passed: `--selftest` and `py_compile` clean; the config guard
matched the registered `pstack-L4` config; and the PairSet known-answer
`self_checks` passed on all ten bins for all four seeds. No halt fired.

### Read-prediction convention (as registered)

Every `R2` above is scored with the full fitted affine map `yhat = Rc @ w + b`,
per the registered difference from exp 30. The disc→held transfer `R2` therefore
reproduces exp 30's transfer failure *qualitatively* (`< R2_MIN`), not to the
decimal; the unit covector is used only for the direction cosine. This was
registered so the decimal mismatch is not read as a discrepancy.

## Conclusion

The atlas answers the measurement exp 30 skipped. Exp 30's
`FIXED_READ_NOT_TRANSPORTED` verdict collapsed two statements — (1) the predicate
is not linearly readable late at all, vs (2) the predicate *is* readable late but
a single global affine read does not transport there. The in-place held-out
diagonal disambiguates them:

```text
On pstack-L4 at L1, m=3, both phi1_next_closes and phi2_net_return are linearly
readable IN PLACE at the held-out positions {26,34} at R2 comparable to the
discovery positions {10,18}, but the read direction is position-specific: the
disc and held read covectors are near-orthogonal and no single-position read
carries to another position. Positions are an exchangeable transfer axis.
```

So exp 30's negative is a **transport-of-read** failure, statement (2), not a
representational absence (statement (1) is rejected: the predicates *are* present
at L1 at later positions). This is a readability / representational-geometry
claim only — no writability or controllability is claimed or implied, and no
patch was applied.

Routing consequence, per the registered Interpretation/Routing Map:
`POSITION_SPECIFIC_READ` routes to **I2 with position-conditioned reads** (or a
transport-valid read fit before re-running I1), and the `exchangeable=True`
overlay means the transfer axis need not be switched first. It explicitly does
**not** route to I4/depth (the predicate is readable late, so the patch point /
layer is not wrong there). Carrying the exp-29 single global affine readout
forward into I2 as a fixed transport-valid read would be mis-targeted: a write
search using it would again fight the transport wall this diagnostic localized,
not a write-freedom limit.
