# Residual Abstraction

This project investigates what a transformer's internal representation
(the "residual stream") carries about the distribution of text it will
produce next. The central question: **can we find a small, interpretable
summary of the residual stream that preserves everything the model needs
to predict future tokens under a given distribution, completion
horizon, and intervention family — and can we measure how much it
misses?**

The method is interventional, not correlational. We don't just ask "does
this subspace *correlate* with the future?" — we physically replace parts
of the representation and measure whether the model's predicted
completions change. This distinguishes genuinely load-bearing structure
from echoes that happen to live nearby in the representation but don't
drive computation.

## What we found

### Phase 1 — calibration on toy processes (experiments 1–18)

We trained small transformers on processes with known ground truth (hidden
Markov models: Mess3 and Dyck-2), so that every claim could be checked
exactly.

**The core result.** A 2-dimensional subspace of a 64-dimensional residual
stream carries ~98–99% of the model's completion-relevant information on
Mess3. This subspace is found automatically by a discovery loop that uses
only the model's own outputs (no access to the hidden ground truth), and
the loop's score agrees with the exact ground-truth score to within
1–2 percentage points across six consecutive experiments (exps 13–18),
with the most thorough check being experiment 18's 52-cell matrix
spanning four completion horizons and four adversarial regimes.

**Key findings along the way:**

- **Correlation is not causation.** Subspaces found by decode-supervised
  methods (PLS) close under 3% of the behavioral gap when used as
  interventions — they find "echoes" of the causal structure, not the
  structure itself. (PCA happens to land near the causal plane on Mess3,
  but this is variance mimicry, not a reliable signal — experiment 8's
  adversarial coordinates break it.) This failure (the "proposal
  misalignment" verdict) was one of the earliest and most consistent
  lessons.

- **The model's representation is not the oracle's.** On Dyck-2 (a
  bracket-matching process), the ground-truth sufficient statistic is
  13-dimensional, but the model's own computation routes through a
  4-dimensional core that captures 93% of completions. The model solves
  the problem its own way. Standard probing methods that ask "is the
  oracle's statistic linearly present?" fail here (R² = 0.66), while the
  interventional battery transfers cleanly. This dissociation — the
  "representation–oracle mismatch" — is the headline result from
  experiment 7 and the strongest argument for causal over correlational
  methods.

- **A calibrated diagnostic battery.** Six instruments that can be run
  without ground truth (see "The diagnostic battery" below), calibrated
  against known answers over 18 experiments. The calibration record is
  evidence, not proof, of transfer — it motivates carrying these
  instruments to models where ground truth is unavailable.

- **Adversarial stress-testing.** The battery's *acceptance verdicts*
  (no false positives) survive deliberate coordinate rotations designed
  to fool variance-based methods (experiments 8–9), five independent
  random draws of the adversarial transformation (experiment 17), and
  four completion horizons (experiment 18). Distribution shifts
  (experiment 15) tested patch retention, not acceptance — and
  decisively exposed position-entangled overfitting while the prefix
  shift was robust. The proposal miner itself is coordinate-sensitive
  and fails under hostile coordinates — but acceptance is sound: it
  rejects rather than false-accepts, which is the safety-relevant
  property.

- **A typed failure taxonomy.** Every diagnostic failure is classified:
  domain coarseness, interpreter incompleteness, proposal misalignment,
  echo (correlational-not-causal), variance dependence,
  position-entangled statistical control, representation–oracle mismatch.
  Each type has a named detector in the battery.

### Phase 2 — transfer to new processes (in progress)

The frozen battery from Phase 1 is now being ported to Dyck-2 (a
structurally different process) to test whether the calibration transfers.
This is the battery's first use on a process where the diagnostics were
not developed.

## The diagnostic battery

Six instruments, consolidated and frozen at experiment 18. Full
definitions with calibration records are in `BATTERY.md`; the library
implementation is in `battery.py`.

| # | Name | What it measures | Needs ground truth? |
|---|------|-----------------|-------------------|
| 1 | **Observable closure** | How much a patch closes the gap between unpatched and fully-patched model behavior, scored by the model against itself | No |
| 2 | **Equivalence ratio** (ρ) | Whether a candidate patch is behaviorally interchangeable with a trusted reference, measured by symmetric KL divergence | No (given a trusted reference) |
| 3 | **Held-out-position gain** | Observable closure on positions excluded from discovery — detects position-specific overfitting | No |
| 4 | **Shift-retention** (R) | Whether a patch's gain is preserved under distribution shifts — fragility is decisive, robustness is not shift-immunity | No (given a trusted reference) |
| 5 | **Accepted-cell calibration** (P4) | Agreement between the model-vs-model score and the exact ground-truth score on every accepted cell — the calibration procedure itself | Yes (calibration-time only) |
| 6 | **CEGAR accept-count** | The discovery loop's acceptance staircase — how many directions pass the marginal-gain threshold, and whether this is stable across thresholds | No |

## Glossary

Internal vocabulary used in the experiments and their plain-English
meanings:

| Internal term | Plain meaning |
|---|---|
| c_obs | Observable closure score — how much a patch restores source behavior, scored without ground truth |
| D2 | The rank-2 "clean" reference patch — both causal write directions (back-mapped from adversarial coordinates) composed |
| ρ (rho) | Equivalence ratio — behavioral similarity between a candidate patch and a trusted reference |
| R (retention) | Shift-retention — how well a patch's improvement survives a change of input distribution |
| P4 held / P4 protocol | The observable and exact scores agreed (the calibration check passed) |
| κ (kappa) | Adversarial conditioning strength — how hard the coordinate rotation tries to fool variance-based methods |
| CEGAR | Counterexample-guided abstraction refinement — the iterative discovery loop |
| k\* | The number of directions the CEGAR loop accepted before stopping |
| eps_gain | Marginal gain threshold — a new direction must improve the score by at least this much |
| m, mm | Completion horizon — how many future tokens the score considers |
| Mess3 | A 3-state hidden Markov process used as the primary calibration target |
| Dyck-2 | A depth-bounded bracket-matching process (the second calibration target) |
| T (transform) | An adversarial coordinate rotation that inflates junk directions and shrinks causal ones |
| write / read | The two sides of a patch: "write" projects the source's signal; "read" extracts it at the target |
| PairSet | A set of paired prefixes used for interventional evaluation |
| echo | A correlational-not-causal subspace — looks relevant by decode metrics but carries almost no causal signal |
| representation–oracle mismatch | The model solves the problem differently from the ground-truth sufficient statistic |

## Repository layout

| Path | Contents |
|---|---|
| `BATTERY.md` | The frozen diagnostic battery — definitions, calibration records, scope |
| `FORMALISM.md` | Named quantities, verdict predicates, assumption ledger |
| `EXPERIMENTS.md` | Experiment index (one-line pointers to per-experiment writeups) |
| `PHASE2.md` | Phase 2 plan and session handoff |
| `RESIDUAL_METHODS_NOTE.md` | Future checkpoint for comparing this method with residual steering, SAE, ReFT, and circuit-tracing approaches |
| `AGENTS.md` | Working norms and research commitments |
| `experiments/` | Per-experiment registrations and conclusions |
| `battery.py` | The battery as a reusable library (Refs, Exact, ρ, R, CEGAR) |
| `expcommon.py` | Shared experiment scaffolding (CLI, guards, model loading, transforms) |
| `backcheck.py` | Regression harness — reruns the battery and reproduces recorded numbers |
| `train.py` | Model training |
| `processes.py` | Data-generating processes (HMMs: Mess3, Z1R, Dyck-2) |
| `discover.py` | PairSet, CEGAR mining, self-checks |
| `model.py` | Transformer architecture |
| `out/` | Trained models, cached arrays, tracked experiment outputs |
| `out/exp*_*.txt` | Canonical experiment output logs (tracked in git) |
