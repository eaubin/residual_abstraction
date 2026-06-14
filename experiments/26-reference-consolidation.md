# Experiment 26 — Oracle-withdrawal 4: pstack reference-and-ρ consolidation (multi-seed) — PRE-REGISTRATION

**Script:** `scripts/oracle_withdrawal/reference_consolidation.py`.

**Status: concluded. Headline: GO to Block 3. Across 8 fresh seeds `pstack`
has one coherent, stable ~k=4 reference; ρ separates the extreme known
cases cleanly (separation `0.830`, Dyck-like) so the transferred `0.25/0.5`
bands hold; `cegar`–`pca` is `PARTIAL_MIMICRY` (mean `10.0°`) — but that
geometric wobble does **not** bite behavioral ρ (estimates read equivalent
at `0.02–0.04`). `RHO_OVERSENSITIVE` did not fire.**

## Question

Exp 25 dissolved the exp-24 reference ambiguity (the two-reference split was
single-seed noise; `pstack` has one stable ~k=4 reference) but, by gating
three checks behind `STRUCTURAL_PASS`, never ran them. Two are genuinely
unfinished and block any ρ-based battery claim, independent of the dead
invariance framing:

1. **Is `pstack` a mimicry process?** Exp 24 (seed 0) said `cegar` and `pca`
   are distinct (interventional discovery buys something over PCA); exp 25's
   fresh seeds said mostly not. The tension deserves proper seed statistics,
   because it decides whether interventional CEGAR earns its keep on the
   richer substrate — the motivation for moving past Dyck.
2. **Does ρ separate the known cases on `pstack`?** The `0.25/0.5`
   equivalence/distinct bands (battery member 2) are Dyck-transferred and
   **never verified here**. No ρ-based battery-transfer claim can stand until
   ρ is checked against the exact-known cases on this process. Note the only
   known cases are *extremes* (near-coincident estimates ≈ equivalent,
   destructive `rand` ≈ distinct), so this validates ρ's **separation**, not
   the threshold *values* — `pstack` offers no intermediate known-case to pin
   `0.25/0.5` themselves (scope, Arm B).

This is a consolidation step inserted before Block 3 (battery transfer,
exp 27): establish the reference object and the ρ calibration on `pstack`,
**multi-seed from the start** — the durable lesson from exp 25 is that
single-seed geometry claims on `pstack` are unreliable.

The exp-24 record correction (one reference, near-coincident estimates —
not two distinct references) is a satisfied prerequisite: exp 24's writeup,
its `EXPERIMENTS.md` row, and the FORMALISM ledger were back-annotated at
exp-25 conclusion. This registration cites that corrected framing.

## Registered Command

```bash
python3 scripts/oracle_withdrawal/reference_consolidation.py --outdir out/pstack-L4
```

Review-only self-test:

```bash
python3 scripts/oracle_withdrawal/reference_consolidation.py --selftest
```

(Run via `uv run python ...` in this environment.)

## Oracle Discipline

Observable quantities are computed and printed before the exact reveal:

- Arm A reference geometry: principal angles among candidate subspaces
  (residuals/tokens/model weights only);
- Arm B ρ-verdicts: ρ(`cegar`, X) bands for each probe (a local observable
  helper, never reads the process tables).

The exact oracle is then revealed in two distinct roles, and the
ground-truth-discipline justification differs for each:

- **Audit (primary, plain evaluation use).** Confirming the reference
  estimates are exact-equivalent (coherence) and checking whether ρ
  separates the exact-known cases as the transferred bands require. This is
  exact-as-evaluation against ground truth — the always-permitted use; it
  tunes nothing.
- **Recalibration (contingency, a precedented threshold extension).** *If*
  the transferred bands fail to separate, the script lowers the distinct
  floor — i.e. tunes a verdict threshold using exact. `ORACLE_WITHDRAWAL.md`
  restricts threshold-tuning to the **substrate block**, and exp 26 is **not**
  the substrate block (exp 23 was). So this is a deliberate, stated extension
  of the discipline, resting on the **exp-19 precedent** (per-process verdict-
  threshold recalibration in a dedicated step, frozen before any abstraction
  verdict) — not on the substrate carve-out, which does not cover it.

The bands are **checked as transferred first**; recalibration is the
contingency, reported as the registered pstack bands. No Block-3 abstraction
verdict is computed here.

## Registered Setting

Inherits the Exp 23/24 standard setting; deltas only.

| item | value |
|---|---|
| process/checkpoint | `pstack`, `out/pstack-L4`, `L1`, `m=3`, `mm=3` |
| candidate construction | reused from Exp 24 (`reference_selection.build_candidates`, by import; frozen script not edited) |
| seeds | **8 fresh**: `100..107` (disjoint from exp 25's `0..7`) |
| discovery positions | `ts=(10,18,26,34)`; eval `ts=(10,18,26,34)` |
| eval pairs | `1024` |
| anchor | the interventionally-discovered `cegar` core (single declared anchor) |
| probes | `{cegar, pca, delta, emb, rand, full}` |
| ρ bands (under test) | `<= 0.25` equivalent / `>= 0.5` distinct (Dyck-transferred) |
| ρ separation floor | `rand_min − equiv_max >= 0.25` for a usable band (else `RHO_NONSEPARATING`) |
| mimicry threshold | `cegar`–`pca` max principal angle vs `10°` (Exp-24 `AMBIG_ANGLE_MAX`); Mess3 pole `3.3–3.6°`; random-subspace null = `rand`–`pca` |
| reference coherence | per-seed exact spread over `{cegar,pca,delta}` `<= 0.05`; all closures `>= 0.70`; `cegar` exact-closure std `<= 0.05` |

## Arm A — Reference Geometry (mimicry characterization)

For each seed, build the candidates and record the max principal angles
`cegar`–`pca`, `cegar`–`delta`, `pca`–`delta`, and the random-subspace null
`rand`–`pca`. Report across seeds: mean, std, observed `[min, max]`.

Typed verdict on the `cegar`–`pca` angle (the mimicry question), using the
observed range as conservative bounds:

- `MIMICRY` iff `max <= 10°` — `cegar` sits in the PCA plane at every seed;
  interventional discovery ≈ PCA on `pstack` (like Mess3, where `cegar`–`pca`
  was `3.3–3.6°`).
- `DISTINCT` iff `min > 10°` — `cegar` is reliably separated; CEGAR finds a
  causal core PCA misses.
- `PARTIAL_MIMICRY` otherwise — the angle straddles `10°` across seeds
  (exp-25's 8-seed peek suggested this: `6.9–13.9°`, mean `~8.5°`).

This is **characterization, not a gate** on Block 3 (the `cegar` core is a
well-defined construction per seed regardless). It answers, on the record,
whether interventional discovery buys anything over PCA here.

## Arm B — ρ Calibration and Reference Coherence (under the `cegar` anchor)

For each seed, anchor ρ on `cegar` and compute ρ(`cegar`, X) for every probe
(observable), then reveal exact closures.

**ρ-separation calibration (the prerequisite for Block 3).** *Scope (this
checks ρ's separation, not its threshold values).* The only known cases on
`pstack` are extremes: `rand` is destructive (must read **distinct**, ρ
`>= 0.5`); `pca` and `delta` are exact-equivalent estimates of the one
reference (must read **equivalent**, ρ `<= 0.25`). Nothing lands near `0.25`
or `0.5`, so a pass confirms ρ **separates the available extreme cases**, not
that the band *values* are correct — those stay Dyck-inherited (`pstack`
offers no intermediate known-case to pin the thresholds; `emb`, exact `0.77`,
is computed and reported but has no a priori band label, so it is
deliberately unused in the verdict). Report, over seeds, the worst
equivalent ρ (`equiv_max = max_seed max(ρ(cegar,pca), ρ(cegar,delta))`), the
worst distinct ρ (`rand_min = min_seed ρ(cegar,rand)`), and the separation
`rand_min − equiv_max` (Dyck reference: equivalent max `0.187`, distinct min
`0.998`).

The verdict splits the two **opposite** ρ failures (they must not collapse
into one band-widening GO — the conflation caught and fixed in exp-25
`decide()`); it is read only under `REFERENCE_COHERENT`, so exact already
says the estimates are equivalent:

- `BANDS_TRANSFER` — `equiv_max <= 0.25` and `rand_min >= 0.5` across all 8
  seeds: ρ separates the extremes as the transferred bands require.
- `RHO_OVERSENSITIVE` (`equiv_max > 0.25`) — ρ reads the **exact-equivalent**
  estimates as non-equivalent: the member-2 mean-Jeffreys failure. **NO-GO**;
  band-widening would paper over precisely the thing Block 3's ρ rests on.
- `BANDS_RECALIBRATE` (`equiv_max <= 0.25`, `rand_min < 0.5`, separation
  `>= 0.25`) — estimates equivalent, junk under the `0.5` floor but a usable
  gap remains: lower the distinct floor. The script **emits the registered
  pstack bands explicitly** (equivalent `<= 0.25`; distinct floor = midpoint
  of the observed envelope) as Block 3's input.
- `RHO_NONSEPARATING` (separation `< 0.25`) — no usable gap even between the
  extremes. **NO-GO**.

**Reference coherence (the single-reference claim, exact).** Across seeds:
the estimates `{cegar, pca, delta}` are exact-equivalent (per-seed closure
spread `<= 0.05`), all strong (closure `>= 0.70`), and the anchor is stable
(`cegar` exact-closure std `<= 0.05`).

- `REFERENCE_COHERENT` iff all three hold — "one reference, near-coincident
  estimates," confirming the exp-25 single-reference picture multi-seed.
- `REFERENCE_INCOHERENT` otherwise — the estimates are not one reference
  (exact-distinct or `cegar` unstable); this reopens reference selection (the
  `EXACT_INDIFFERENCE_BREAKS` scenario exp 25 flagged) and blocks Block 3.

## Predictions

**P1 (substrate + self-checks; enforced).** Registered config, PairSet
self-checks, and global non-degeneracy hold at every seed. Failure = typed
substrate halt, not an Exp 26 result.

**P2 (mimicry characterization; descriptive, expect `PARTIAL_MIMICRY`).**
The `cegar`–`pca` angle straddles `10°` across fresh seeds (exp-25 prior:
mean `~8.5°`, range `6.9–13.9°`). A clean `MIMICRY` or `DISTINCT` is
possible and equally reportable.

**P3 (ρ separates the extremes; ~75%).** `BANDS_TRANSFER`: ρ reads the
estimates equivalent (`<= 0.25`) and `rand` distinct (`>= 0.5`).
`BANDS_RECALIBRATE` (junk under the `0.5` floor, usable gap) is a real
possibility and a finding, not a failure. The sharp adverse outcome is
`RHO_OVERSENSITIVE` (ρ splits the exact-equivalent estimates) — a member-2
failure that **NO-GO**s Block 3 rather than being absorbed into band-widening;
this is the failure exp 25 was originally built to surface.

**P4 (reference coherence; ~85%).** `REFERENCE_COHERENT`: the estimates are
exact-equivalent and strong, `cegar` stable. Exp 25 showed the estimates are
one cluster; this is the exact, multi-seed confirmation.

**P5 (decision; deterministic).**

Coherence (exact, one reference) gates first; under coherence the
calibration verdict stands.

- `REFERENCE_COHERENT ∧ BANDS_TRANSFER` → **GO**: pre-register Block 3
  (exp 27) under the `cegar` core with the transferred bands (values
  Dyck-inherited; see scope). Report the mimicry verdict.
- `REFERENCE_COHERENT ∧ BANDS_RECALIBRATE` → **GO**: Block 3 with the
  emitted recalibrated `pstack` ρ bands.
- `REFERENCE_COHERENT ∧ RHO_OVERSENSITIVE` → **NO-GO**: ρ does not track
  exact equivalence on `pstack`; do not widen the band — investigate ρ
  (per-pair vs mean-Jeffreys) before any ρ-based transfer.
- `REFERENCE_COHERENT ∧ RHO_NONSEPARATING` → **NO-GO**: ρ cannot separate
  even the extremes; recalibrate the substrate/ρ construction.
- `REFERENCE_INCOHERENT` → **NO-GO**: the single-reference premise fails;
  reopen reference selection before any battery transfer.

## Scope & Local Assumptions

- Stability/statistics are over pair/basis **sampling** at a fixed
  checkpoint, not model retraining.
- The mimicry verdict is a characterization of `cegar` vs `pca` at `L1`,
  `k=k_ref`, `m=3`; it does not claim anything about other layers/horizons.
- ρ carries the member-2 mean-level caveat (BATTERY.md): equivalence is a
  mean-Jeffreys statement, not per-pair.
- This step calibrates **member 2 (ρ) only**. The full six-member
  per-process threshold recalibration is Block 3's job (as Dyck's exp 19
  did), not this step's.
- Exact closure is evaluation/calibration-only; sampled completions are not
  used (exact model chain probabilities throughout).
- Indexed by `pstack`, `L1`, `m=3`, the registered positions/candidate
  family, the `10°` mimicry threshold, the `0.25/0.5` bands under test, and
  the 8 fresh seeds.

## Expected Output

The script prints, in order:

- per-seed observable table (angles `cegar`–`pca` / `cegar`–`delta` /
  `pca`–`delta` / null `rand`–`pca`; ρ(`cegar`, X) for each probe) — before
  any exact reveal;
- Arm A aggregate (`cegar`–`pca` mean/std/range, null, Mess3 pole) and the
  mimicry verdict;
- Arm B observable aggregate (`equiv_max`, `rand_min`, separation);
- the exact reveal: per-seed exact closures, the reference-coherence verdict,
  and the calibration verdict (`BANDS_TRANSFER` / `BANDS_RECALIBRATE` /
  `RHO_OVERSENSITIVE` / `RHO_NONSEPARATING`), read only under coherence;
- `DECISION` per P5, and — on `BANDS_RECALIBRATE` — the explicit registered
  pstack bands (equivalent ceiling, recalibrated distinct floor) for Block 3.

---

## Results

Run artifact: `out/exp26_pstack-L4.txt`. Checkpoint `model.pt`/`cache.npz`
untracked per repository policy (CPU/fixed-seed, reproducible from the
Exp-23 training command).

**Headline: GO to Block 3.** `pstack` has one coherent, stable reference
across 8 fresh seeds; ρ separates the extreme known cases cleanly so the
Dyck bands transfer; the `cegar`–`pca` geometry is `PARTIAL_MIMICRY`, but
the ~10° gap does not translate into a behavioral (ρ) difference.

### Verdict fidelity

| prediction | registered | outcome |
|---|---|---|
| P1 substrate + self-checks | enforced | **held** — self-checks passed at all 8 seeds; no halt |
| P2 mimicry | descriptive, expect `PARTIAL_MIMICRY` | **`PARTIAL_MIMICRY`** — `cegar`–`pca` mean `10.0°`, range `[7.6, 16.5]` (straddles `10°`) |
| P3 ρ separation | ~75% `BANDS_TRANSFER` | **`BANDS_TRANSFER`** — `equiv_max 0.041`, `rand_min 0.871`, separation `0.830`; `RHO_OVERSENSITIVE` did not fire |
| P4 reference coherence | ~85% `REFERENCE_COHERENT` | **`REFERENCE_COHERENT`** — closures `0.91–0.94`, per-seed spread `<= 0.05`, `cegar` mean `0.925` std `0.005` |
| P5 decision | deterministic | **GO** — preregister Block 3 (exp 27) under the `cegar` core, transferred bands |

### What happened

Reference geometry (Arm A, max principal angles, deg):

| | mean | std | range |
|---|---|---|---|
| `cegar`–`pca` | 10.0 | 2.6 | [7.6, 16.5] |
| `cegar`–`delta` | ~4.3 | — | [2.4, 6.2] |
| null `rand`–`pca` | 87.8 | — | — |

`cegar` and `delta` are near-coincident (one plane); `pca` is the ~10°
outlier — the same structure exp 25 found, now with the typical separation
*at* the threshold (mean `10.0°`), confirming `PARTIAL_MIMICRY`. Against the
poles — Mess3 mimicry `3.5°` and a random `87.8°` — `cegar` and `pca` share
nearly all of their `k=4` directions but differ in ~one.

ρ under the `cegar` anchor (Arm B): the reference estimates read
**equivalent** every seed (`pca` `0.020–0.041`, `delta` `0.001–0.007`),
`rand` reads **distinct** (`0.871–0.996`), giving worst-case separation
`0.830` — comparable to Dyck's `0.81`. Exact audit confirms the estimates
are exact-equivalent and strong (coherence) and that ρ's bands classify the
extremes correctly (calibration).

### Interpretation

**Geometric near-distinctness ≠ behavioral distinctness.** `pca` sits ~10°
off the `cegar`/`delta` plane in subspace, yet ρ reads it as behaviorally
equivalent to `cegar` (`0.02–0.04`, deep in the equivalent band). So the
exp-24 seed-0 "two distinct references / CONFIRMED" was both
*geometrically* marginal (exp 25) and *behaviorally* void: the ~10° gap
carries no completion-behavior difference. This is the clean reconciliation
of the arc — the honest object is **one behavioral reference**, estimated
several near-coincident ways. It is also *consistent with* (but does not
measure) exp 25's shelved ρ-invariance: ρ was computed only under the
`cegar` anchor here, so the cross-cluster gap leaving the estimates
ρ-equivalent (`0.02–0.04`) suggests `pca`-anchored verdicts would have
agreed — but the `pca`-anchored ρ was never computed, so anchor-invariance
stays inferred, not established.

**Does interventional discovery buy anything over PCA on `pstack`?**
Marginally and geometrically only: CEGAR's core differs from the PCA plane
by ~one direction (~10°), more than Mess3's `3.5°` mimicry but far from
distinct — and that difference is behaviorally inert at the ρ level. So
`pstack`, despite being richer than Mess3, is still close to a
variance-mimicry process at `L1`/`k=4`/`m=3`: interventional discovery
earns little here beyond PCA. (This is the characterization promised; it
does not gate Block 3.)

**ρ calibrates, with a stated limit.** The Dyck `0.25/0.5` bands separate
the extremes with a `0.830` margin, so they transfer — but only their
*separation* is validated, not the threshold *values*. The lone
intermediate probe, `emb` (exact closure `0.77–0.81`), reads ρ `0.16–0.20`
— inside the equivalent band despite being behaviorally weaker than the
estimates (`0.92`). That is consistent with ρ measuring behavioral
*direction/proximity* rather than closure magnitude, but it is also an
*uncalibrated* hint that the equivalent band may be lenient at intermediate
strengths on `pstack`. With no ground-truth band label for `emb`, it stays
out of the verdict; a future intermediate-known-case probe could pin the
threshold values themselves.

### Decision

**GO.** `REFERENCE_COHERENT ∧ BANDS_TRANSFER`: preregister Block 3 (exp 27)
— battery transfer under the single interventionally-discovered `cegar`
core, with the transferred `0.25/0.5` ρ bands and the full six-member
per-process recalibration (deliberately left to Block 3, as Dyck's exp 19
did). The `PARTIAL_MIMICRY` finding is reported context, not a gate.

Two things Block 3 must inherit explicitly:

- **Lenient-equivalent-band concern (live, not a footnote).** `emb` reads
  ρ `0.18` at exact closure `0.77` — inside the `0.25` equivalent band
  despite being behaviorally weaker than the `0.92` reference. Block 3
  judges patches of varying strength, so it must include an
  intermediate-strength probe with known exact closure and verify ρ's
  equivalent band does not over-accept it; if it does, the equivalent band
  needs tightening on `pstack` (the band *values*, unvalidated here).
- **State the expected marginal value up front (honesty).** `pstack` came
  out near variance-mimicry (`cegar` ≈ `pca`, behaviorally one reference)
  and the anchor is *declared* (the `cegar` core by convention), not
  *uniquely earned* — observable selection ties among near-coincident
  estimates. So Block 3 is at risk of being a Dyck/Mess3 transfer re-run on
  a substrate that did not turn out meaningfully richer. Its registration
  must say plainly what it expects to learn beyond Dyck: the honest claim is
  a **system-level** one — that the end-to-end hidden-oracle workflow
  (earned-not-blessed reference, observable scoring, exact audit at
  registered points) yields a usable six-member battery on a new process —
  not the discovery of new battery failure physics. If Block 3 merely
  reproduces Dyck's transfer, it should say so rather than dress it up.

### Scope

Indexed by `pstack`, `L1`, `m=3`, the registered positions/candidate
family, the `10°` mimicry threshold, the `0.25/0.5` bands (separation
validated, values Dyck-inherited), and the 8 fresh seeds (`100–107`),
pair/basis sampling at a fixed checkpoint. The mimicry characterization is
specific to `cegar` vs `pca` at this layer/dimension/horizon; the ρ-band
*values* remain uncalibrated at intermediate strengths (no `pstack`
intermediate known-case). Exact closure was used for calibration/audit
only (per the exp-19-precedent extension stated in Oracle Discipline),
frozen here before any Block-3 verdict.
