# Experiment 15 — statistical control vs transported state: per-pair equivalence, EPR localization, and distribution shift — PRE-REGISTRATION

**Script:** `transport.py`. **Status: pre-registered; NOT YET RUN — results
to be appended below the marked line.**

**Question.** Experiment 14 left the program's central object unexplained:
w2's learned reads transfer (+32.2% / +42.5%, observable = exact), carry
zero plane mass, and compute essentially none of the clean functional
(pooled EPR ≈ 0.008). Two live readings: **statistical control** — the
read exploits correlations specific to the discovery/eval distribution,
causing the right downstream behavior on-distribution without
transporting the underlying state variable; or **transported state** —
the learned patch is behaviorally equivalent to the clean patch, and the
current read-side diagnostics are simply the wrong instruments. This is
also the registered test of FORMALISM §8's **equivalence-class claim**
(every working patch is behaviorally equivalent *per pair* to a
clean-plane patch), the adjudication §8 assigned to this experiment, and
the carrier for exp-14's inherited debts (per-position EPR; a read-side
separator candidate; persisting learned reads).

**External proposal status.** The §8 framing and this design were
reviewed externally and endorsed; the reviewer's structure (three tests:
equivalence, EPR localization, shift robustness) and its warning — the
shift must not destroy baseline competence or change the clean patch's
meaning, else failure-under-shift is uninterpretable — are both adopted
as registered guards below. The five "most informative outcomes" are
registered verbatim as the adjudication map.

## Design (all hyperparameters frozen here)

**Setting.** Exp-6/8–14 unchanged: mess3-L4, L1, prefix-wide, κ = 100,
seed 0, 400/600 disjoint discovery/eval pairs, basis 800, m = 3, anchor +
transform checks, gate. Writes reproduced by the exp-12 rule. Optimizer
(for read reproduction only): the exp-14 affine-slice machinery,
identical hyperparameters (Adam lr 0.05, 200 steps, batch 64, torch
seed 0).

**Reads (reproduced in-run, persisted this time).** The two accepted
learned reads aff/w2/best-α and aff/w2/id (reproduction assert:
full-discovery gains within 2 points of the recorded +32.2% / +42.5%);
the destructive learned control aff/w1/id (assert: ≤ −100%); plus three
free comparators per write — the clean read (the reference), the best-α
spectral read, and the id-z read (w2's recorded −187.4% destructive
comparator). All reproduced reads are saved to
`out/<process>/exp15_reads.npz` (gitignored; deterministic regeneration —
the artifact gap exp 14 flagged).

**Machinery changes (backward-compatible, this registration).**
`PairSet` gains optional `ts` and `init_state` kwargs; defaults reproduce
the registered protocol bit-for-bit. `HMMProcess.sample` gains
`init_state`. **Registered subtlety:** beliefs (hence exact targets) are
*always* computed from the stationary prior — the frame the trained model
approximates — so a shift moves the distribution *over* prefixes, never
the per-prefix target; the clean patch's meaning is unchanged by
construction.

**Test 1 — per-pair equivalence.** For patch X and the same-write clean
reference patch C, on a given pair set: per-pair Jeffreys divergence
J_i(C, X) = ½[KL(q_C,i ‖ q_X,i) + KL(q_X,i ‖ q_C,i)], and the
**equivalence ratio**

> ρ(X) = mean_i J_i(C, X) / mean_i J_i(C, un)

— distance to clean in units of the do-nothing distance. Registered
bands: **equivalent** ρ ≤ 0.25; **partial** 0.25 < ρ < 0.5;
**distinct** ρ ≥ 0.5. Descriptive companions: both directional KL means,
per-pair quantiles, and frac_worse = the fraction of pairs with
J_i(C, X) > J_i(C, un) (equivalent-on-average can hide a divergent
subpopulation). ρ is computed for every read above, on every pair set.

**Test 2 — EPR localization (base eval set).** Per-cell EPR: for each
t-group g and each absolute position p ≤ t_g, corr²(Δ_p·r, Δ_p·u_clean)
over that cell's rows (~200 rows/cell; null corr² ≈ 0.005, so the 0.2
threshold is far above noise). Reported as the full curve per read;
"position-t EPR" = the p = t_g cells. The clean read must score 1.0 in
every cell (plumbing).

**Test 3 — distribution shift.** Two registered shifts, evaluated as
fresh PairSets with full exact targets:

- **Shift-A (positions):** eval pairs at ts = {12, 20} (the midpoints of
  the base grid {8, 16, 24}); same sequence distribution, seed +777.
- **Shift-B (prefix distribution):** sequences sampled with initial
  hidden state fixed to state 0, seed +779, base positions. Mess3 is
  state-symmetric, so one fixed state is representative.

Quantities per shift s and patch X: exact closure gain on s, and the
**relative retention** R(X, s) = [gain_X(s)/gain_X(base)] /
[gain_C(s)/gain_C(base)] — normalizing by the clean patch's own retention
so that overall difficulty changes cancel. **Guards (per the external
warning, NOT TESTED if violated):** the shift is declared too destructive
for shift-B if the model's NLL exceeds the *exact predictor's* NLL on the
same shifted sample by > 0.01 nats (both computed in the stationary
frame — the right competence comparison for off-stationary data); and
for either shift if the clean patch's shifted exact gain < 20%.
Registered scope caveat: these are *mild* shifts — robustness here does
not establish shift-immunity in general; fragility here is decisive in
the other direction.

## Pre-registered predictions (NOT TESTED residuals explicit)

- **P1 (anchors + reproduction; ~90%).** Exp-6 loop, both transform
  checks, torch/numpy regression link; D1 ≥ 40%, D2 ≥ 90% of full; the
  three optimization reruns reproduce (w2 gains within 2 pts of recorded;
  w1/id ≤ −100%). Always testable; a reproduction failure halts
  interpretation (determinism breach) — *enforced* (pre-run review fix:
  the run exits after the reproduction asserts with P2–P6 printed
  NOT TESTED; `--force-invalid` continues exploratorily).
- **P2 (per-pair equivalence, headline 1; ~45% — genuinely uncertain,
  that is the point).** Both accepted learned reads are **equivalent**
  (ρ ≤ 0.25) to the clean patch on the base eval set. Three-way by the
  registered bands; "distinct with good closure" is the registered
  refutation of the §8 equivalence-class claim *for this instance*.
- **P3 (EPR localization, headline 2; three-way).** (a) ~35%: some
  position-t cell ≥ 0.5 for an accepted learned read → exp-14's pooled
  refutation was an **aggregation artifact**; the statistical-predictor
  account revives position-resolved. (b) ~40%: all cells < 0.2 for both
  learned reads → the linear clean-functional story is refuted at every
  granularity tested. (c) else: partial localization, recorded
  per-cell. Always testable.
- **P4 (observable soundness; ~85%).** On the base set, every reproduced
  patch with gain ≥ 20%: |observable − exact| ≤ 0.10. NOT TESTED if none
  (cannot happen if P1 holds).
- **P5 (shift robustness, headline 3; gated per shift by the guards).**
  All four cells (2 accepted reads × 2 shifts): R ≥ 0.7 → HOLDS
  (transported-state reading; ~40%). Any cell ≤ 0.3 → the registered
  **statistical-control branch** (first-class, not a mere failure). Cells
  in (0.3, 0.7) → partial, recorded per-cell. NOT TESTED for a shift
  whose guard fails.
- **P6 (ρ as the read-side separator candidate; ~75%).** On the base
  set: min ρ over destructive reads ≥ 10 × max ρ over accepted learned
  reads — Finding 4's first candidate validated. (Note ρ needs a trusted
  reference patch, not an oracle: at LLM scale the reference is the
  best-validated patch, so the construction transports.)
- **P7 (validity gate, enforced).** As established.

**Adjudication map (registered verbatim from the endorsed proposal).**
(1) ρ low + R robust → the learned read is an honest alternate
implementation; current geometry diagnostics are inadequate; §8
equivalence-class claim confirmed for this instance. (2) ρ low on-base +
R fragile → statistical control; equivalence is distribution-local.
(3) ρ high (distinct) but closure good → closure gain is too coarse;
patches can transfer for different behavioral reasons; equivalence-class
claim refuted for this instance. (4) position-t EPR high → exp-14's
pooled EPR was the wrong aggregation (compatible with any of 1–3).
(5) EPR low everywhere + ρ low → the clean-functional story is too
narrow even position-resolved: behavioral equivalence without linear
functional agreement.

**Indexing (inherited, narrower than ever — deliberate).** Every verdict
is indexed by the single registered T, *one* working write (w2), and the
two shifts chosen. The generality sweep (exp-16 candidate) widens T,
writes, eps_gain, and m; this experiment buys adjudication depth first.

**Ledger rows (FORMALISM §7, added with this registration).**
Equivalence ratio ρ (Jeffreys; do-nothing normalization; trusted-
reference-patch assumption). Per-cell EPR (absolute-position pooling is
the right disaggregation; ~200-row cells). Registered shifts (stationary-
frame targets; mild-shift scope; competence + clean-gain guards).

**Self-checks** (every invocation; `--selftest` exits after them): the
standard four; anchor + transform checks and the torch/numpy regression
link (real runs); **new:** (i) `PairSet(ts=None)` reproduces groups
{8, 16, 24} and `ts={12, 20}` builds exactly those; (ii)
`sample(init_state=0)` first-token frequencies match the exact
state-0 emission law within 0.05 on 2000 sequences (default path
unchanged by code inspection: identical rng draws); (iii) J(q, q) = 0
and ρ(C) = 0 (both in the selftest — the ρ identity added as a pre-run
review fix); (iv) per-cell EPR of the clean read = 1 — the generic
identity in the selftest, and asserted in-run on every actual cell.

**Enforcement.** Registered parameters, full config, seed 0, gate — as
exps 8–14. Estimated runtime ~80–110 min (3 optimization reruns + 3
pair-set constructions with exact targets + ~8 patches × 3 sets of
evaluations).

**Post-registration, pre-run note.** The script was refactored onto the
new shared scaffolding module (`expcommon.py`) — pure code motion, no
design change; the standard selftests, the exp-14 regression selftest,
and (when run) the registered reproduction asserts are the verification.
Concluded scripts (exps 1–14) keep their inline copies as frozen records.

---

*(Results to be appended here after the run.)*
