# Experiment 26 — Oracle-withdrawal 4: pstack reference-and-ρ consolidation (multi-seed) — PRE-REGISTRATION

**Script:** `scripts/oracle_withdrawal/reference_consolidation.py`.

**Status: pre-registered; NOT YET RUN. Pause here for review before the
canonical run.**

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
2. **Do the ρ bands hold on `pstack`?** The `0.25/0.5` equivalence/distinct
   bands (battery member 2) are Dyck-transferred and **never verified here**.
   No ρ-based battery-transfer claim can stand until they are checked against
   the exact-known cases on this process.

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

The exact oracle is then revealed for **calibration only** — confirming the
reference estimates are exact-equivalent, and checking whether the
Dyck-transferred ρ bands classify the exact-known cases correctly. This is
the per-process calibration the program explicitly permits in a measurement
step (cf. exp 19's Dyck threshold recalibration; `ORACLE_WITHDRAWAL.md`
substrate carve-out): exact may calibrate estimator/verdict thresholds *in
a dedicated calibration step, frozen before any abstraction verdict*. The
bands are **checked as transferred first**; recalibration happens only if
they fail and is reported as the registered pstack bands. No Block-3
abstraction verdict is computed here.

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

**ρ-band calibration (the prerequisite for Block 3).** The known cases:
`rand` is destructive (must read **distinct**, ρ `>= 0.5`); `pca` and `delta`
are exact-equivalent estimates of the one reference (must read **equivalent**,
ρ `<= 0.25`). Report, over seeds, the worst equivalent ρ (`equiv_max =
max_seed max(ρ(cegar,pca), ρ(cegar,delta))`), the worst distinct ρ
(`rand_min = min_seed ρ(cegar,rand)`), and the separation `rand_min −
equiv_max` (Dyck reference: equivalent max `0.187`, distinct min `0.998`).

- `BANDS_TRANSFER` iff `equiv_max <= 0.25` and `rand_min >= 0.5` across all
  8 seeds — the Dyck bands hold on `pstack` as transferred.
- `BANDS_RECALIBRATE` otherwise — report the pstack separation and the
  recalibrated bands (the equivalent/distinct envelope); a finding, not a
  failure, and the registered pstack bands for Block 3.

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

**P3 (ρ-band calibration; ~75%).** `BANDS_TRANSFER`: the Dyck `0.25/0.5`
bands separate the known cases on `pstack` (`rand` distinct, estimates
equivalent). `BANDS_RECALIBRATE` is a real possibility — pstack is a
different process — and is a finding, not a failure.

**P4 (reference coherence; ~85%).** `REFERENCE_COHERENT`: the estimates are
exact-equivalent and strong, `cegar` stable. Exp 25 showed the estimates are
one cluster; this is the exact, multi-seed confirmation.

**P5 (decision; deterministic).**

- `REFERENCE_COHERENT ∧ BANDS_TRANSFER` → **GO**: pre-register Block 3
  (exp 27) battery transfer under the `cegar` core with the transferred
  bands. Report the mimicry verdict.
- `REFERENCE_COHERENT ∧ BANDS_RECALIBRATE` → **GO**: Block 3 with the
  registered recalibrated `pstack` ρ bands.
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
- the exact reveal: per-seed exact closures, the ρ-band calibration verdict,
  and the reference-coherence verdict;
- `DECISION` per P5.

---

## Results

Not run. Pause for pre-run review of the verdict logic, the calibration
framing, and the code.
