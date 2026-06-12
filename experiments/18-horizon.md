# Experiment 18 — the m-staircase: do the diagnostics and conclusions survive changing the completion horizon? — PRE-REGISTRATION

**Script:** `mstair.py` (on `expcommon.py`). **Status: pre-registered;
NOT YET RUN — results to be appended below the marked line.**

**Question.** Every claim in the program is indexed by the standing
horizon m = 3 (§1). The horizon is not a nuisance parameter — it *is*
the semantic target γ_m. Is the battery measuring something stable about
future-relevant structure, or something tuned to m = 3? This is the last
unswept primary index, and — registered intent, not binding — the last
major Mess3 calibration experiment unless it reveals a new failure type;
after it, a battery-freeze writeup and a change of process class.

**External proposal, evaluated.** Adopted: the m ∈ {1, 2, 3, 4} sweep
over the four representative regimes (benign; κ=100 draw 0, the
historical setting; κ=30, the positive-gradient regime; κ=300, the
high-pathology pole); the five outcome framings, registered below as the
adjudication map nearly verbatim; no new repairs, read families, or κ
points; κ\* explicitly deferred (regime cartography of one toy
construction). **Modified, two points.** (1) *Structural*: the m ≤ 3
staircase points are exact **marginalizations of the m = 3 chain**
(reshape-and-sum on the 27-outcome joint — the exps-4/5 `kl_by_horizon`
identity, extended here to the observable side and to ρ), so they cost
nothing new; **m = 4 is the only new compute** (81 continuations, ~3×
per evaluation). (2) *Separation of questions*: "gradient read gains
per m" conflates two things — (i) **diagnostics stability**: fixed
patches (the m=3-learned reads, reproduced once) evaluated across all
m — eval-only, the battery-calibration question; and (ii) **discovery
m-sensitivity**: re-learning with the objective built at a different
m — sampled, not swept (4 runs: κ ∈ {30, 100} × m ∈ {1, 4}). Conflating
them would make a ρ shift ambiguous between "horizon-local certificate"
and "different object found."

## Design (deltas from the standard setting)

**Horizons.** mm ∈ {1, 2, 3} from the m = 3 chain by marginalization
(exact identity, asserted in the selftest); mm = 4 from new m = 4
PairSets (disc/eval/val, same seeds 111/777/333) with **ts pinned to
{8, 16, 24}** (and {12, 20} for val) — the default ts formula shifts
positions with m, which would break cross-m comparability.

**Regimes and patch sets (arm A — diagnostics stability, eval-only).**
- *benign*: the plane projector P_c (k = 2) and the clean rank-1; the
  benign k\*(m) staircase.
- *κ=100, draw 0*: clean (both writes), the reproduced aff/w2/id learned
  read (assert +42.5% ± 2 pts at m = 3), id-z(w2) (the −187%
  destructive comparator), spectral-best(w2), clean D2.
- *κ=30, draw 1*: clean, **both reproduced transported reads** (assert
  +52.5%/+51.5% train, val +40.0%/+37.4%, ρ 0.046/0.075 at m = 3), id
  (the −38.1% comparator), spectral-best, clean D2.
- *κ=300, draw 1*: reduced set — clean, id, spectral-best, D2 (no
  converging reads exist there).

Per (patch, regime, mm): exact closure, observable closure, the
obs/exact gap, ρ_mm vs the same-write clean, and — for learned reads —
the held-out val gain (transport across horizons). Reproduction failure
at m = 3 **halts** (enforced, as exps 15–17).

**Arm B — discovery m-sensitivity.**
- CEGAR loops per regime per mm at the frozen eps = 0.05 (exp 17
  settled threshold-robustness; the staircase here is over m, not eps):
  benign k\*(mm); adversarial accept-counts.
- Re-learning: the registered optimizer with the objective built at
  m = 1 and m = 4, on the κ=100 w2 write and the κ=30 nearest write
  (id init, registered budget; 4 runs). Each resulting read evaluated
  at its own m and across the staircase (cross-horizon transfer of the
  learned object, descriptive).

## Pre-registered predictions

- **P1 (m = 3 reproduction + anchors; ~90%).** All recorded m = 3
  numbers reproduce within 2 pts (ρ within 0.02); anchors; halt
  enforced.
- **P2 (calibration across horizons — the battery's load-bearing test;
  ~80%).** For every (patch, regime, mm) with observable closure ≥ 20%,
  including mm = 4: |observable − exact| ≤ 0.10.
- **P3 (ρ separation persists; ~70%).** At every mm: max ρ_mm over the
  κ=30 transported reads ≤ 0.25, and min ρ_mm over the named destructive
  comparators (id-z at κ=100; id at κ=30) ≥ 10× that max.
- **P4 (transport is horizon-stable — the most state-like claim under
  test; ~60%).** Both κ=30 transported reads keep val gain ≥ 20% at
  every mm ∈ {1, 2, 3, 4}.
- **P5 (proposal death persists; ~85%).** Adversarial accept-count = 0
  at eps 0.05 for all three adversarial regimes at every mm.
- **P6 (benign dimension across horizons; ~75%).** Benign k\*(mm) = 2
  for mm ∈ {2, 3, 4}; mm = 1 reported descriptively — a *lower* k\* at
  m = 1 would be the expected semantic-complexity staircase, not a
  failure (registered reading).
- **P7 (re-learning at other horizons; ~55%).** The κ=30 re-learned
  reads (m = 1 and m = 4 objectives) converge and transport at their own
  horizon (train ≥ 20%, val ≥ 20%). The κ=100 runs' phenotypes and all
  cross-horizon evaluations are descriptive (no registered claim — the
  m = 3 phenotype's persistence is genuinely unknown).
- **P8 (validity gate, enforced).**

**Adjudication map (registered from the proposal).** (1) Diagnostics
hold across m → the self-certification battery is much more credible;
freeze it and move process class. (2) ρ works at m = 3 but fails at
m = 1 or 4 → ρ is a horizon-local certificate, not a general one.
(3) κ=30 transport holds only at some horizons → "learned reads
transport" is horizon-indexed, not state-like. (4) κ=100 pathology
persists across m → the adversarial story is strengthened. (5) Benign
k\* shifts with m → the expected staircase of semantic complexity, not
a failure.

## Scope & local assumptions

- mm ≤ 3 points are exact marginalizations of the m = 3 chain, not new
  measurements — one chain evaluation yields all three horizons.
- m = 4 PairSets pin ts = {8, 16, 24} / {12, 20} explicitly; the default
  ts formula would move pair positions with m.
- The re-learning arm samples discovery m-sensitivity at 4 points
  (κ ∈ {30, 100} × m ∈ {1, 4}); it does not sweep it.
- eps is fixed at 0.05 throughout (exp 17 settled threshold-robustness
  at m = 3; re-sweeping eps per m is out of scope).
- disc3 and disc4 share pairs and prefix arrays bitwise by construction
  (same seed, pinned ts) — asserted in the selftest; the CEGAR mining
  input is therefore m-independent, and only the weights and scoring
  change with mm. (Pre-run review fix: mm = 4 loops nonetheless mine
  from disc4-built views, aligning the code with this registration's
  letter.)
- κ=300 carries the reduced patch set; its staircase audits the
  fixed-patch diagnostics only.
- The battery-freeze intent is registered as intent, not as a binding
  conclusion of this experiment.

**Ledger updates (with this registration).** New live row: "m = 3
standing horizon (exps 1–17)" → **under test (exp 18)**.

**Self-checks** (standard set, plus): (i) the marginalization identity —
a small m = 3 chain run marginalized to mm = 1 equals an independent
m = 1 chain run on the same pairs to 10⁻⁶ (float32 forward; measured
2.9×10⁻⁸ — tolerance set from measurement, the exp-17 lesson), and the
exact m-gram tables to 10⁻¹²; (ii) m = 4 PairSet groups == [8, 16, 24]
(pinned ts); (iii) the marginal helper on a synthetic known table.

**Enforcement.** Standard. Estimated runtime **~3–4 h** (3 reproduction
runs at m = 3 + 4 re-learning runs, two of them at the 3× m = 4 cost +
the eval matrix with its single m = 4 column + 16 CEGAR loops).

---

*(Results to be appended here after the run.)*
