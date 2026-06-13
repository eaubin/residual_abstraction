# Phase 2 — Dyck-2: porting the frozen battery (plan and handoff)

**Status when this was written.** Mess3 calibration closed at exp 18;
the diagnostic battery is frozen in `BATTERY.md`; the ledger (`FORMALISM.md`
§7) carries the live scope debts. Dyck-2 is the chosen next process
class. **A fresh session starts here**, then `BATTERY.md`, then
`experiments/7-dyck.md` (the existing Dyck results), then `AGENTS.md`
(standing norms). The prior conversation is disposable; everything
load-bearing is in these files.

## Process change (evaluated from the external proposal; adopted with corrections)

The Mess3 phase ran 18 experiments because it was inventing the protocol
while calibrating it. Phase 2 reuses known instruments, so the
experimental unit grows from "one question" to "one block":

Phase 2 uses the general `EXPERIMENT_REVIEW_PROTOCOL.md`; the bullets
below are Dyck/block-specific applications of that repository-level
review discipline.

- **One registration, one pre-run review, one run, one conclusion review
  — per block, not per predicate.** Block registrations carry coarser
  block-level predictions plus sub-predicate tables; sub-results are
  table rows, not headline experiments.
- **Registration means writeup plus runnable code.** The writeup records
  the goals, assumptions, scope, predictions, adjudication rules, and
  failure modes; the script implements the registered guards,
  self-checks, verdict predicates, and output tables. Review pauses
  happen after both pieces are committed and before the first run.
- **Reviews are dual-purpose.** Pre-run review evaluates the experiment
  design and implementation together. Post-run review evaluates the
  results and conclusion logic, and also checks for LLM-work creep,
  overclaims, hidden dependency growth, and maintainability regressions.
- **Failure handling inside a block:** a failed instrument is reported
  and scoped in the block conclusion. It spawns a dedicated follow-up
  experiment *only* if it is a genuinely new failure type (the exp-16
  reviewer's criterion, now policy). Known failure types are recorded
  against the taxonomy and the block proceeds.
- The §6.1 checklist, the ledger, scope indexing, halt-on-reproduction-
  failure, and "Scope & local assumptions" sections all carry over
  unchanged — they are cheap and they caught real errors ~15 reviews
  running.

## The blocks (5, including one that is not an experiment)

**Block 0 — communication + library (no run verdicts; reviewed as a
deliverable).** Two debts that should be paid before new science:
1. *The external account.* README.md is currently a 14-line stub —
   there is no document readable by someone outside the loop. Rewrite it
   as the outward-facing account of phases 1–2: the question, the
   method, the main findings in plain language, and a **glossary**
   mapping internal vocabulary to plain terms (c_obs → "behavioral
   closure score", D2 → "two-direction clean-reference patch", "P4
   held" → "observable and exact scores agreed", κ → "adversarial
   conditioning strength", …). Policy adopted: **two registers** —
   `experiments/`/`FORMALISM.md` stay internal-efficient; README is the
   external register, updated at phase boundaries only. This bounds
   vocabulary drift without taxing every writeup.
2. *The battery as a library.* Extract `battery.py`: the six BATTERY.md
   members as named functions (the table's "code home" column then
   points at real symbols), plus the CEGAR loop (still copy-pasted
   per-script), the closure/obs reference classes (exp 18's `Refs`/
   `Exact` are the cleanest versions — promote them), ρ/retention as
   named metrics, and small reporting-table helpers. `expcommon.py`
   remains the scaffolding layer; frozen scripts stay untouched.
   **Verification: a Mess3 back-check script that reruns the battery
   through the new library and reproduces the recorded exp-17/18
   numbers** — the frozen record becomes the regression harness for the
   library. Do not over-engineer: extraction is justified exactly as
   far as the back-check exercises it.

**Block 1 — Dyck baseline + threshold recalibration.** Port PairSets/
completion oracle (dyck2 already implements the HMMProcess interface;
exp 7 ran the early battery). References: full patch, no-op, and the
exp-7 discovered 4-dim causal core. Recalibrate every Mess3-calibrated
constant rather than inheriting it: the obs/exact band (**exp 7's
recorded agreement was ~5.9 points — Mess3's 0.10 band may be near its
limit on Dyck; if the gap is structural, that is a finding, not a
nuisance**), acceptance gains (set relative to reference-patch gain,
not absolute 20%), ρ bands re-derived against the new reference.

**Block 2 — the interventional battery matrix.** Clean/reference
patches, learned/optimized candidates, ρ, held-out slices, shifts — one
registered matrix. Dyck-native translations: the **trusted reference
for ρ is the honestly-discovered exp-7 core** (not a T-aware
construction — this is the LLM-posture rehearsal, the first time ρ's
reference comes from discovery); held-out slices stratify by **bracket
depth** as well as position; the distribution shift is a depth-profile
shift. Standing lens for the whole block: exp 7's
representation–oracle mismatch — the battery is behavioral and never
needed linear decode, so Dyck tests whether it works where decode
fails.

**Block 3 — robustness sweep.** Horizon staircase (mind the cost:
Dyck's vocabulary makes V^m grow faster than Mess3's 3^m — set the m
range from a measured budget, marginalization trick still applies),
tolerance staircases, and coordinate stress (the adversarial T
translates: P_c = the discovered core subspace, junk ⊥ it; a small
κ-grading check, since the Mess3 gradient-pathology results are
κ-indexed).

**Block 4 — consolidation.** Update BATTERY.md's scope statement with
the Dyck record (it currently says Mess3-only, by design). Decide:
move up the ladder (sampled ground truth / TinyStories-scale, per
AGENTS.md's roadmap) or chase a new failure type if Dyck exposed one.

## Translation table (what transfers, per the external review's tiers, corrected)

| tier | items | note |
|---|---|---|
| direct | PairSet construction, completion evaluation, observable closure, CEGAR loop, obs/exact calibration, ρ, held-out-slice gains, shift-retention, registration style, failure taxonomy, ledger discipline | `dyck2` already implements the process interface; exp 7 proved most of this runs |
| translate | adversarial T (P_c := discovered core), trusted reference (T-aware clean → discovered core), rank-1 oblique patching (core is 4-dim — rank-k), position tests (+ depth strata), gradient read optimization | translation is Block-2 design work, registered there |
| do **not** inherit | every numeric threshold (20%, 0.10 band, ρ ≤ 0.25, κ values), the 2-D-plane conclusions, plane/junk/neutral decompositions (recast on the 4-dim core), m = 3 cost assumptions | Block 1 re-derives; inheriting Mess3 constants silently would be the new phase's version of the eps_gain debt |

## Handoff inventory (what a fresh session needs)

Read, in order: this file → `EXPERIMENT_REVIEW_PROTOCOL.md` →
`BATTERY.md` → `experiments/7-dyck.md` → `AGENTS.md`; consult
`FORMALISM.md` §6.1/§7 when registering and `EXPERIMENTS.md` for the
index. Auto-memory carries the working norms (pre-register → review
pause → run → review; files over conversation; evaluate external reviews
on merits; one writeup per unit; conclusions written once). Known
recurring defect classes to self-check in every conclusion: claims
outrunning their index (granularity, quantifiers, two-point ranges),
stale ledger rows when a question resolves (grep *every* row that
mentions it), and summary ranges that silently cover only the best case.
