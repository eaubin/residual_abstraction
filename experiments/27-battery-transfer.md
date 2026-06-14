# Experiment 27 — Oracle-withdrawal 5: battery transfer under the earned reference (Block 3) — PRE-REGISTRATION

**Script:** `scripts/oracle_withdrawal/battery_transfer.py`.

**Status: concluded. Headline: `BATTERY_TRANSFERS_WITH_RECALIBRATION(M2_rho)`
— GO. The six-member battery transfers to `pstack` under the earned `cegar`
reference across 4 fresh seeds: M1/M3/M4/M5/M6 PASS clean, M2 recalibrates.
The live finding is M2's lenient equivalent band — `emb` (directionally
distinct, exact ≈0.78) reads ρ-equivalent (≈0.19) to the core (≈0.93), so
the `0.25` band over-accepts; recalibrate to ≈0.10 on `pstack`. The
system-level claim holds (the hidden-oracle workflow yields a usable
battery); the substrate stayed Dyck-like.**

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
| shift (M4) | `init_state = 22` = pstack `(mode 1, stack (0,0,0))` (mode-1 base 15 + stacks-index 7), a deep-stack + alternate-mode prefix shift; `C` = full patch normalizer |
| ρ bands (M2) | `<= 0.25` equivalent / `>= 0.5` distinct (exp-26 `BANDS_TRANSFER`) |
| intermediate probe | `emb` (the token-embedding subspace) — **directionally distinct** from the core, exact closure ≈ 0.77, ρ ≈ 0.18 (exp 26): the geometry the band-leniency check actually needs |
| obs/exact band (M5) | transferred `0.10` (Mess3) / `0.073` (Dyck); three-way split (transfer / recalibrate / inversion) |
| decision states | `BATTERY_TRANSFERS` / `..._WITH_RECALIBRATION(members)` / `TYPED_BATTERY_FAILURE(members)` |
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
probe suite `{cegar(self), pca, delta, emb, rand, full}`. PASS iff the
near-coincident estimates read **equivalent** (`<= 0.25`) and `rand` reads
**distinct** (`>= 0.5`) — the transferred bands. **Inherited check
(exp 26): the intermediate probe must not be over-accepted.** The probe is
`emb` — *directionally distinct* from the core, with intermediate exact
closure (≈0.77) — not a truncation of the core: a same-direction truncation
would test only magnitude and ρ reading it equivalent could be ρ working
correctly, so `emb` is the right geometry. If `emb`'s exact closure is
meaningfully below the core's (gap `> 0.10`) yet ρ reads it equivalent
(`<= 0.25`), flag `RHO_BAND_LENIENT` — but as a **`RECALIBRATE` (hint), not
a `FAIL`**: whether ρ *should* be magnitude-sensitive is the open exp-26
question, so this surfaces a possibly-lenient band, it does not condemn one.

**M3 — held-out-position gain.** The `cegar` core patched at the held-out
positions `(12,20,28,36)`; observable and exact closure. PASS iff held-out
closure `>= 0.70` and within `0.10` of base-position closure — no
discovery-position overfitting (the *candidate-reference overfitting*
failure type).

**M4 — shift-retention R.** `R = [gain_core(shift)/gain_core(base)] /
[gain_full(shift)/gain_full(base)]` on **raw KL gains** (not normalized
closure, which would make the full normalizer trivially 1), under the
registered `init_state = 22` shift, with **competence guard** (model-vs-exact
NLL on the shifted distribution within the exp-23 band) and **clean-gain
guard** (the full patch still gains under the shift). PASS iff guards hold
and `R >= 0.80`. **Single-shift caveat:** this is one registered shift
(Dyck used several, out of scope here), so a *fail* flags fragility under
**this** shift, not proven general shift-fragility; the reported
`gain_full` at base vs shift shows whether the shift actually moved the
distribution (a washed-out shift makes M4 vacuous, not a pass).

**M5 — accepted-cell obs/exact agreement (three-way directional).** For
each cell (core, `emb`, held-out core) the **signed** gap `obs − exact`.
The split, recalibrate-vs-fail done directionally (the partition fixed in
exp-25/26, here directional):

- `|gap| <= 0.10` everywhere → **PASS** (transfers).
- `gap > 0.10` somewhere — observable *overstates* exact (accepts what exact
  rejects) → **FAIL** `OBS_EXACT_INVERSION` (the dangerous direction; a band
  cannot be widened to excuse over-trust).
- otherwise `|gap| > 0.10` but only in the conservative direction
  (`exact > obs`, observable understates) → **RECALIBRATE**: report the
  pstack obs/exact band as the observed envelope (exp-19 per-process
  recalibration, pass-with-note). A clean pstack envelope above the
  transferred `0.10` is recalibrated, not failed.

The recalibrated band is reported **directionally** for the Block-3 handoff:
the conservative side (`exact > obs`) widens to the observed `max_abs`, but
the inversion side (`obs > exact`) is **held at `0.10`** — Block 3 must not
apply the recalibrated number symmetrically, or it would re-loosen the
over-trust guard this split exists to protect.

**M6 — CEGAR staircase.** `k*(eps)` over `{0.01,0.02,0.05,0.10}` on the
discovery PairSet (the frozen **accept-only** loop, `battery.cegar_staircase`).
PASS iff the staircase is weakly decreasing in `eps` and `k*` lands in the
expected range (`3 <= k <= 5` at `eps=0.05`). Note this is a *different
instrument* from the coarsen-based discovery loop that set `k_ref`, so
`k*(0.05)` need not equal `k_ref` — the band absorbs the difference; the
member checks the staircase is coherent, not that it reproduces `k_ref`.

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
`RHO_BAND_LENIENT` on `emb` (the inherited concern) is a real possibility —
it routes to **`RECALIBRATE` (pass-with-note)**, not fail, since ρ's
magnitude-sensitivity is the open exp-26 question; given exp-26's `emb`
(ρ 0.18 at exact 0.77) it may well fire.

**P4 (M3 held-out gain; ~75%).** Held-out closure strong and within `0.10`
of base; failure = candidate-reference overfitting.

**P5 (M4 shift-retention; ~70%).** Guards hold and `R >= 0.80`; failure =
shift fragility of the earned reference.

**P6 (M5 obs/exact agreement; ~80%).** `|gap| <= 0.10`, or a clean
conservative-direction envelope → `RECALIBRATE`. The fail is
`OBS_EXACT_INVERSION` (obs overstates exact) — the most consequential for
trusting observable verdicts.

**P7 (M6 staircase; ~90%).** Coherent, weakly decreasing, `k*` in
`[3,5]` at `eps=0.05`.

**P_final (decision; deterministic).**

- No member FAILs → **`BATTERY_TRANSFERS`** (or
  **`BATTERY_TRANSFERS_WITH_RECALIBRATION(members)`** if any member
  recalibrated its per-process threshold): the hidden-oracle workflow yields
  a usable six-member battery on `pstack` under the earned reference — the
  system-level claim. Consolidate the oracle-withdrawal reference program.
  A recalibrate state is a PASS carrying its registered pstack threshold,
  not a partial failure.
- Any member FAILs → **`TYPED_BATTERY_FAILURE(members)`**, named by member
  and failure type (reference overtrust / candidate overfitting / obs-exact
  inversion / shift fragility / contrast miss); register the repair. A typed
  failure is the informative outcome, not a defeat.

## Scope & Local Assumptions

- Indexed by `pstack`, `L1`, `m=3`, the registered positions, the **earned
  `cegar`** reference, the transferred/recalibrated thresholds, the
  `init_state` shift, and the 4 fresh seeds; pair/basis sampling at a fixed
  checkpoint.
- The reference is *declared* (one behavioral reference; not uniquely earned
  in the strong sense) — the claim is that the workflow produces a usable
  battery with it, not that selection was unique.
- The ρ band *values* remain Dyck-inherited; M2/M5 test them against the
  single intermediate probe `emb`, so `pstack` still offers limited
  intermediate ground-truth (one cell) — `RHO_BAND_LENIENT` is a hint.
- Exact closure is evaluation/calibration-only; sampled completions not used.
- The robustness sweep (multiple shifts, κ stress, horizon staircase) that
  Dyck spread over exps 20–21 is **out of scope** — this is the single
  transfer test, deliberately not a sprawling re-discovery of robustness
  physics (per the marginal-value framing).
- **Runtime:** heavy — 4 seeds × 4 PairSets (1024 pairs) + the 4-eps CEGAR
  staircase (which reruns the accept-only loop per eps) + per-seed exact and
  a shift PairSet. Expect a multi-hour run (exp 25 was ~2 h for 8 lighter
  seeds; this is fewer seeds but adds the staircase and the shift set).

## Expected Output

The script prints, in order:

- per-seed observable member table (M1 closure, M2 ρ suite incl. `emb`,
  M3 held-out closure, M4 R + shift competence, M6 `k*(0.05)`) — before the
  exact reveal;
- the exact reveal: per-seed exact closures (core, `emb`, held) and the
  observable closures they audit;
- per-member verdicts as `PASS` / `RECALIBRATE` / `FAIL` with thresholds;
- `DECISION`: `BATTERY_TRANSFERS`, `..._WITH_RECALIBRATION(members)`, or
  `TYPED_BATTERY_FAILURE(members)`.

---

## Results

Run artifact: `out/exp27_pstack-L4.txt`. Checkpoint `model.pt`/`cache.npz`
untracked per repository policy (CPU/fixed-seed, reproducible from the
Exp-23 training command).

**Headline: `BATTERY_TRANSFERS_WITH_RECALIBRATION(M2_rho)` — GO.** The
six-member battery transfers under the earned `cegar` reference; the only
non-clean member is M2's lenient equivalent band (a recalibrate-with-note,
the live finding). The system-level claim holds.

### Verdict fidelity

| prediction | registered | outcome |
|---|---|---|
| P1 substrate + self-checks | enforced | **held** — self-checks at all 4 seeds; no halt |
| P2 M1 closure | ~95% PASS | **PASS** — min core closure `0.948` |
| P3 M2 ρ separation | ~85%, lenient→RECALIBRATE | **RECALIBRATE** — bands hold (estimates ≤`0.044`, `rand` ≥`0.882`) but `emb` lenient (the predicted finding) |
| P4 M3 held-out | ~75% PASS | **PASS** — held-out `0.935`, max \|held−base\| `0.012` (no overfitting) |
| P5 M4 shift-retention | ~70% PASS | **PASS** — `R_min 0.989`, guards held (mild shift) |
| P6 M5 obs/exact | ~80% | **PASS** — max \|obs−exact\| `0.026` ≤ `0.10`, no inversion |
| P7 M6 staircase | ~90% PASS | **PASS** — `k*(0.05)=[4,4,4,4]`, weakly decreasing |
| P_final | deterministic | **GO** — `BATTERY_TRANSFERS_WITH_RECALIBRATION(M2_rho)` |

### What happened (per-seed, 4 fresh seeds)

| member | numbers |
|---|---|
| M1 closure | core `0.948–0.958` |
| M2 ρ \| `cegar` | `pca` `0.022–0.044`, `delta` `0.001–0.009` (equivalent); `rand` `0.882–0.960` (distinct); `emb` `0.177–0.205` (equivalent band) |
| M3 held-out | `0.935–0.950`; \|held−base\| ≤ `0.012` |
| M4 R | `0.989–1.007`; competence-on-shift `+0.010–0.011` (≤`0.030`) |
| M5 obs−exact | core `+0.022`, `emb` `+0.019`, held `+0.026` (all within band) |
| M6 k*(0.05) | `4` every seed |

### Interpretation

**The system-level claim holds.** End-to-end under hidden-oracle discipline
— reference earned-not-blessed (exps 24–26), observable scoring, exact
revealed only at the registered audit — the six-member battery produces
coherent, non-vacuous numbers on `pstack`, with five members clean. The
workflow yields a usable battery. As the marginal-value framing warned, the
substrate stayed Dyck-like: nothing here is new battery *physics*. Two
quantitative notes even sharpen that — M5 obs/exact agreement (`0.026`) is
*tighter* than Dyck's `0.073`, and M6's `k*=4` is rock-stable — so the
transferred thresholds had room to spare except where the live finding bites.

**The live finding (M2, carry forward): the equivalent ρ band is lenient on
`pstack`.** With the *directionally-distinct* intermediate probe the review
required (`emb`, not a core truncation), the exp-26 concern reproduces
cleanly: `emb` closes ≈`0.78` exact vs the core's ≈`0.93` (gap ≈`0.15`) and
points in a different subspace, yet ρ reads it equivalent (`0.177–0.205`,
inside the `0.25` band). So ρ at the mean-Jeffreys level cannot separate a
behaviorally-weaker, different-direction patch from the reference. This is
**not** a clean pass dressed up — it is a real member-2 caveat. The
actionable recalibration: the true estimates sit at ≤`0.044` and `emb` at
≥`0.177`, so a **recalibrated `pstack` equivalent band ≈ `0.10`** (between
them) would correctly admit the genuine estimates and exclude `emb`. Block 3
/ any ρ-based use on `pstack` should adopt `≈0.10`, not `0.25`. (Whether ρ
*should* be magnitude-sensitive at all remains the open exp-26 question; but
here a *different-direction* weaker patch reading equivalent makes the
leniency concrete, not just a magnitude artifact.)

**M5 directional handoff (item-1 guard).** M5 passed within the transferred
`0.10`, so no recalibration fired — but note the gaps are all in the
*inversion direction* (obs overstates exact by `0.019–0.026`), just small.
The band is **not** loosened; the `0.10` inversion guard stands for Block 3.

**M4 single-shift caveat.** R ≈ `1.0` under the one registered `init_state`
shift with competence degrading only `+0.01` — a *mild* shift the core
retains under, not proven general shift-robustness (Dyck used several; out
of scope). Read as "robust under this shift," not "shift-immune."

### Decision

**GO** — `BATTERY_TRANSFERS_WITH_RECALIBRATION(M2_rho)`. The oracle-withdrawal
reference program can consolidate (unit 6): the hidden-oracle workflow
delivers a usable six-member battery on `pstack` under an earned reference.
Two things consolidation must carry, not bury:

- the **lenient equivalent band** (recalibrate `pstack` ρ-equivalent to
  ≈`0.10`; the `0.25` transferred value over-accepts intermediate patches);
- the **directional M5 band** (any future recalibration widens only the
  conservative side; the `0.10` inversion guard is held).

### Scope

Indexed by `pstack`, `L1`, `m=3`, the registered positions, the earned
`cegar` reference, the transferred/recalibrated thresholds, the single
`init_state=22` shift, and the 4 fresh seeds (`200–203`); pair/basis
sampling at a fixed checkpoint. The reference is *declared* (one behavioral
reference, not uniquely earned). The ρ-band recalibration (`≈0.10`) rests on
the single intermediate cell `emb`; a broader intermediate-strength sweep
would pin it more tightly. The robustness sweep (multiple shifts, κ,
horizon) Dyck spread over exps 20–21 is out of scope. Exact closure was
used for M5 + recalibration only (exp-19-precedent extension), revealed
after the observable members.
