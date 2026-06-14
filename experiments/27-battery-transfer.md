# Experiment 27 — Oracle-withdrawal 5: battery transfer under the earned reference (Block 3) — PRE-REGISTRATION

**Script:** `scripts/oracle_withdrawal/battery_transfer.py`.

**Status: pre-registered; NOT YET RUN. Pause here for review before the
canonical run.**

## Question

Run the six-member diagnostic battery (`BATTERY.md`) on `pstack` using the
**earned** reference — the interventionally-discovered `cegar` core that
exps 24–26 established, selected and audited under the hidden-oracle
protocol — rather than an oracle-blessed reference. Do the six members
produce coherent, non-vacuous numbers, and do their thresholds hold or
recalibrate per-process (as Dyck's exp 19 recalibrated from Mess3)?

**What this experiment expects to learn, stated up front (honesty, per the
exp-26 handoff).** `pstack` did **not** turn out meaningfully richer than
Mess3: exp 26 found it near variance-mimicry (`cegar` ≈ `pca`, behaviorally
one reference), and the anchor here is **declared** (the `cegar` core by
convention), not *uniquely earned* — observable selection ties among
near-coincident estimates (exp 24/25). So the honest claim on offer is
**system-level**, not new battery physics:

> Does the end-to-end hidden-oracle workflow — reference earned-not-blessed,
> observable scoring, exact audit only at registered points — yield a usable
> six-member battery on a new process?

If exp 27 merely reproduces Dyck's transfer with a conventional anchor, it
says so. The value is in the *workflow* clearing under oracle discipline (or
failing in a typed way), not in a richer substrate it did not find.

## FORMALISM & Ledger

This is the experiment the two §7 ledger rows have been pointing at: the
ρ-reference bet (does an earned reference anchor ρ and the battery?) and the
framing bet (does exact-toy adjudication calibrate oracle-free work?). On
`pstack` the no-oracle reference is in hand and ρ-calibrated (exp 26); exp 27
is the transfer test. Audited against the §6.1 verdict-predicate checklist
(rule 8: every accept/recalibrate rule registered; rule 4: each member names
its metric and threshold).

## Registered Command

```bash
python3 scripts/oracle_withdrawal/battery_transfer.py --outdir out/pstack-L4
```

Review-only self-test:

```bash
python3 scripts/oracle_withdrawal/battery_transfer.py --selftest
```

(Run via `uv run python ...`.)

## Oracle Discipline

Observable members (M1 closure, M2 ρ, M3 held-out gain, M4 shift-retention,
M6 CEGAR staircase) are computed from model behavior only and printed
before any exact audit. The exact oracle is then revealed for **M5**
(accepted-cell obs/exact agreement) and for the per-process **threshold
recalibration** — the exp-19-precedent extension (verdict-threshold
recalibration in a dedicated step, frozen before the verdict), **not** the
substrate carve-out. The earned reference itself was selected observably in
exps 24–26; exp 27 does not re-select it with the oracle.

## The Earned Reference

The trusted reference `C` is the `cegar` core: the `k = k_ref` (≈4)
interventionally-discovered subspace from `reference_selection.build_candidates`,
re-derived per seed. It is **declared** as the anchor (exp-26 decision),
strong (exact closure ≈ 0.92, std 0.005 over 8 seeds) and ρ-coherent with
its near-coincident estimates. Members 2 and 4 use it as `C`; the full
patch is the ceiling/normalizer control.

## Registered Setting

Inherits the Exp 23–26 standard setting; deltas only.

| item | value |
|---|---|
| process/checkpoint | `pstack`, `out/pstack-L4`, `L1`, `m=3`, `mm=3` |
| reference / anchor | `cegar` core (`build_candidates`, declared) |
| seeds | **4 fresh**: `200..203` (disjoint from exps 24/25/26) |
| base positions | `ts=(10,18,26,34)`; held-out `ts=(12,20,28,36)` |
| eval / held pairs | `1024` / `1024` |
| shift (M4) | non-stationary `init_state` prefix-distribution shift; `C` = full patch normalizer |
| ρ bands (M2) | `<= 0.25` equivalent / `>= 0.5` distinct (exp-26 `BANDS_TRANSFER`) |
| intermediate probe | rank-2 truncation of the `cegar` core (known intermediate exact closure) |
| obs/exact band (M5) | transferred `0.10` (Mess3) / `0.073` (Dyck); recalibrate on pstack |
| eps grid (M6) | `{0.01, 0.02, 0.05, 0.10}` |

Four fresh seeds (not 8): exp 26 already established the core's stability
over 8 seeds (closure std `0.005`); exp 27's new content is M3–M6, for which
4 seeds give stability evidence while keeping the per-seed shift and
staircase tractable.

## The Six Members (each: measurement → transferred threshold → verdict)

**M1 — reference strength (observable closure).** `obs_eval` of the `cegar`
core. PASS iff closure `>= 0.70` (strong, non-vacuous reference). Confirms
exp 26.

**M2 — ρ separation around the earned reference.** Anchor ρ on `cegar`;
probe suite `{cegar(self), pca, delta, emb, rand, full, trunc2}`. PASS iff
the near-coincident estimates read **equivalent** (`<= 0.25`) and `rand`
reads **distinct** (`>= 0.5`) — the transferred bands. **Inherited check
(exp 26): the intermediate probe `trunc2` must not be over-accepted** — if
its exact closure is meaningfully below the core's (gap `> 0.10`) yet ρ
reads it equivalent (`<= 0.25`), flag `RHO_BAND_LENIENT` (the band values,
unvalidated at intermediate strength on pstack, are too loose).

**M3 — held-out-position gain.** The `cegar` core patched at the held-out
positions `(12,20,28,36)`; observable and exact closure. PASS iff held-out
closure `>= 0.70` and within `0.10` of base-position closure — no
discovery-position overfitting (the *candidate-reference overfitting*
failure type).

**M4 — shift-retention R.** `R = [gain_C-core(shift)/gain(base)] /
[gain_full(shift)/gain(base)]` under the registered `init_state` shift, with
**competence guard** (model-vs-exact NLL on the shifted distribution within
the exp-23 band) and **clean-gain guard** (the full patch still gains under
the shift). PASS iff guards hold and `R >= 0.80` (the core retains its gain
about as well as the ceiling; Dyck reference `R = 1.00`).

**M5 — accepted-cell obs/exact agreement.** For each accepted cell (the
core, and `trunc2`), `|obs_closure − exact_closure|`. PASS iff the gap is
within the recalibrated pstack band (transferred `0.10`/`0.073`); a cell
that observable accepts but exact rejects is the *observable/exact
inversion* failure type. The pstack obs/exact band is recalibrated and
reported here (exp-19 style).

**M6 — CEGAR staircase.** `k*(eps)` over `{0.01,0.02,0.05,0.10}` on the
discovery PairSet. PASS iff the staircase is weakly decreasing in `eps` and
`k*` lands in the expected range (≈4 at the registered `eps=0.05`),
reproducing the discovery instrument under the earned-reference protocol.

All members are run at each of the 4 seeds; results are reported per-seed
and aggregated (mean/range). A member's threshold is **recalibrated** (not
just failed) if its transferred value misses but the pstack envelope is
clean — reported as the registered pstack threshold, exp-19 style.

## Predictions

**P1 (substrate + self-checks; enforced).** Registered config, PairSet
self-checks, global non-degeneracy at every seed. Failure = typed substrate
halt, not an Exp 27 result.

**P2 (M1 reference strength; ~95%).** Core closure `>= 0.70` (exp 26: ≈0.92).

**P3 (M2 ρ separation; ~85%).** Estimates equivalent, `rand` distinct.
Adverse sub-outcome `RHO_BAND_LENIENT` (the inherited `emb`/`trunc2`
concern) is a real possibility and a finding, not a pass.

**P4 (M3 held-out gain; ~75%).** Held-out closure strong and within `0.10`
of base; failure = candidate-reference overfitting.

**P5 (M4 shift-retention; ~70%).** Guards hold and `R >= 0.80`; failure =
shift fragility of the earned reference.

**P6 (M5 obs/exact agreement; ~80%).** Gaps within the recalibrated band;
failure = observable/exact inversion (the most consequential for trusting
observable verdicts).

**P7 (M6 staircase; ~90%).** Coherent, weakly decreasing, `k* ≈ 4` at
`eps=0.05`.

**P_final (decision; deterministic).**

- All six PASS (with thresholds held or cleanly recalibrated, no
  `RHO_BAND_LENIENT`/inversion) → **BATTERY_TRANSFERS**: the hidden-oracle
  workflow yields a usable six-member battery on `pstack` under the earned
  reference — the system-level claim. Consolidate the oracle-withdrawal
  reference program.
- Any member fails in a typed way → **TYPED_BATTERY_FAILURE**, named by
  member and failure type (reference overtrust / candidate overfitting /
  obs-exact inversion / lenient band / shift fragility / contrast miss);
  register the repair. A typed failure is the informative outcome, not a
  defeat.

## Scope & Local Assumptions

- Indexed by `pstack`, `L1`, `m=3`, the registered positions, the **earned
  `cegar`** reference, the transferred/recalibrated thresholds, the
  `init_state` shift, and the 4 fresh seeds; pair/basis sampling at a fixed
  checkpoint.
- The reference is *declared* (one behavioral reference; not uniquely earned
  in the strong sense) — the claim is that the workflow produces a usable
  battery with it, not that selection was unique.
- The ρ band *values* remain Dyck-inherited; M2/M5 test them against the
  intermediate probe but `pstack` still offers limited intermediate
  ground-truth (one constructed `trunc2` cell).
- Exact closure is evaluation/calibration-only; sampled completions not used.
- The robustness sweep (multiple shifts, κ stress, horizon staircase) that
  Dyck spread over exps 20–21 is **out of scope** — this is the single
  transfer test, deliberately not a sprawling re-discovery of robustness
  physics (per the marginal-value framing).

## Expected Output

The script prints, in order:

- per-seed observable member table (M1 closure, M2 ρ suite incl. `trunc2`,
  M3 held-out closure, M4 R + guards, M6 staircase) — before the exact
  reveal;
- the exact reveal: M5 obs/exact gaps (incl. `trunc2`), the recalibrated
  pstack obs/exact band, and the `RHO_BAND_LENIENT` check;
- per-member PASS/recalibrate verdicts with thresholds;
- `DECISION`: `BATTERY_TRANSFERS` or `TYPED_BATTERY_FAILURE(member, type)`.

---

## Results

Not run. Pause for pre-run review of the member verdicts, the recalibration
framing, and the code.
