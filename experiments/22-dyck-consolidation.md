# Experiment 22 — Dyck battery consolidation (Phase 2, Block 4)

**Script:** none. This is the Phase-2 consolidation record and
documentation update, not a new model run.

**Status: concluded.** The knowledge consolidation is recorded here and
promoted into `BATTERY.md`, `FORMALISM.md`, `PHASE2.md`, and
`EXPERIMENTS.md`.

## Question

Phase 2 asked whether the frozen Mess3 diagnostic battery transfers to
Dyck-2, a process where the old linear-belief story had already broken
down (exp 7's representation-oracle mismatch). The consolidation
question is:

> What did Dyck teach us about the battery, what should be carried
> forward, and did the phase expose a new failure type that blocks moving
> to the next process class?

No new code is needed for that question. The inputs are the reviewed
Phase-2 block records:

| block | experiment | role in consolidation |
|---|---|---|
| 1 | exp 19 | baseline reproduction, threshold recalibration, first Dyck transfer check |
| 2 | exp 20 | adversarial matrix, shift-retention, prefix-balance strata, single-write probe |
| 3 | exp 21 | horizon, tolerance, and kappa robustness sweep |

## Consolidated Findings

**1. The behavioral battery transfers to Dyck under the registered
indices.** Exp 19 reproduced the exp-7 Dyck anchor through `battery.py`
(`k* = 4`, `c_obs = 98.5%`, exact eval closure `92.6%`) and found that
the Mess3 obs/exact band transferred unchanged: worst accepted-cell gap
`0.064 <= 0.10`. Rho separated working from destructive patches by
`69x`, and the held-out-position core gain was `+98.7%`.

**2. The transfer survives the Block-2 matrix.** Exp 20 found that P1-P6
all held under adversarial coordinates, registered mild shifts, and
prefix-balance strata. The discovered rank-4 core retained essentially
all gain under both registered shifts (`R >= 0.99` with guards passed),
had only `1.9%` spread across observed signed-balance strata, and kept
shifted calibration within the `0.10` band (`0.073` worst gap).

**3. Members 1, 2, 5, and 6 survive the Block-3 robustness grid.** Exp
21 found that the benign CEGAR eps staircase was identical across
`mm = 1..4` (`5,4,4,3`); fixed-patch obs/exact and rho cells stayed in
band across horizons; the worst accepted-cell gap was `0.073` at
`mm=4`; and adversarial accept-counts were zero for every registered
`kappa x mm x eps` cell. Members 3 and 4 were exercised in Block 2 and
were not rerun in Block 3.

**4. The single-write rank-1 probe failure is a scoped finding, not a
battery-transfer failure.** Exp 20 P7 failed at `19.7%` train closure
with rho `0.8691`; P8 was correctly skipped. This says the registered
one-write, one-optimization probe did not find a behaviorally effective
rank-1 oblique patch for the Dyck core. It does not show that no rank-1
direction exists, and it does not block the six battery members.

**5. The main conceptual update is stronger than a numeric transfer.**
Dyck was the first post-freeze check where the trusted rho reference was
the discovered core itself, not a privileged clean/T-aware construction.
The battery still worked where linear belief decoding was already known
to dissociate from behavioral control. That supports the program's
claim that the battery is measuring completion-relevant behavior rather
than allegiance to the old oracle representation.

## What Is Promoted

The following updates are now part of the repository record:

- `BATTERY.md` carries a Dyck transfer layer for the six members,
  separate from the Mess3 calibration scope.
- `FORMALISM.md` marks the exact-toy calibration bet as discharged for
  the first transfer process, while keeping scale and no-oracle settings
  open.
- `PHASE2.md` is marked concluded and records the decision to move to
  the next process class rather than chase a Dyck follow-up.
- `EXPERIMENTS.md` points to this file as the canonical Phase-2
  conclusion.

## What Is Not Promoted

- No claim of Dyck robustness beyond the specific registered Dyck-2
  checks each member was run on: the checkpoint, L1 patch point,
  standard evaluation distribution, registered positions/shifts,
  horizons `mm <= 4`, and the tested kappa/eps grid where applicable.
- No multiple-junk-draw genericity claim for Dyck. Exp 21 used
  `junk_seed = 0`.
- No conclusion that a rank-1 Dyck read is impossible. Only the
  registered single-write probe failed.
- No claim that rho can be trusted without a reference patch. The new
  lesson is narrower: a discovered, behaviorally validated core can
  serve as the reference on a toy process with exact checks.
- No off-distribution or scale guarantee.

## Decision

Phase 2 is complete. Dyck did not expose a new battery-transfer failure
type. The scoped result is:

> On the registered Dyck-2 setting, using the exp-19 discovered rank-4
> core as trusted reference, the frozen diagnostic battery's six members
> transfer under their registered Dyck indices. Members 1, 2, 5, and 6
> additionally transfer through the registered horizon/tolerance/kappa
> robustness grid.

The next work should move up the roadmap to the next process class with
weaker ground truth, carrying forward the same discipline: observable
closure, trusted-reference rho where available, held-out/distribution
checks, accept-count staircases, exact calibration only while an oracle
still exists, and typed failure branches rather than post-hoc stories.
