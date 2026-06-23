# Experiment 41 - Product-counter intervention-class discriminator - PROCEDURAL FAILURE / QUARANTINED PILOT

**Script:** `scripts/product_counter/substrate.py`; analytic threshold derivation:
`scripts/product_counter/derive_thresholds.py`.

**Status:** procedural failure. This artifact is quarantined and must not be
counted as a preregistered claim-producing experiment.

## Procedural Failure

Exp 41 missed the required review pause. Several design drafts, threshold
choices, substrate smoke checks, and route-bearing implementation changes
happened before a clean pre-registration was frozen and reviewed. The main route
was exposed during repair. That cannot be undone by better wording, stricter
tables, or a later deterministic rerun.

The correct disposition is:

- no exp-41 result is accepted;
- no exp-41 threshold or route is evidence for the research program;
- no exp-41 run can be promoted from "calibration" to "pre-registered";
- any revival needs a fresh experiment number, a reviewed writeup, and a
  committed runnable script before the first route-bearing run.

This file is retained only as a caged design pilot and failure record.

## Useful Core

The scientific question remains real. Exp 36 exited `pstack` because its
predicate/control substrate could not decide specificity. The product-counter
toy was intended to remove that excuse with exact finite-state ground truth:
`a`, `b`, and `c` are separable variables, `c` is a high-room out-of-bundle
control, and exact `m=3` completions are cheap.

The intended discriminator is:

```text
Do noncontextual variable handles become coherent state edits,
or do they only control linear readouts while leaving the decoded state invalid?
```

That is the right bigger-picture question for a follow-up. Exp 41 does not answer
it.

## Quarantined Design

The current runner implements a useful pilot matrix:

| arm | role | route relevance |
|---|---|---|
| substrate guards | finite product-counter identities, room, leakage, exact `m=3`, carrier faithfulness | required before any intervention interpretation |
| contextual exact edit | replace clean hidden state with the source hidden state | positive-control ceiling |
| value-mean delta | average planted hidden deltas for each value transition across off-target contexts | natural noncontextual handle |
| observable-min-norm delta | minimum hidden-norm delta matching the target observable change and zero off-target observable change | charitable readout-level handle |
| matched-norm random floors | deterministic no-information directions with matched norm | threshold/floor control |

The pilot separates contextual state edits from two noncontextual handle classes
and checks both signed linear decoded coordinates and simplex-projected decoded
coordinates. That is a better design shape than the earlier ambiguous substrate
gate draft, but it is not a registered experiment.

## Scope Of The Pilot Runner

The only fixed pilot configuration is:

```bash
uv run python scripts/product_counter/substrate.py --confirm --m 3 --seed 0 --kappa 100
```

Scope indices:

| index | value |
|---|---|
| process | product counter, `S=32`, `V=7` |
| state | `(a,b,c)`, `a,b in {0,1,2,3}`, `c in {0,1}` |
| target/distractor/control | `a`, `b`, `c` |
| horizon | exact `m=3` completion distributions |
| carrier | planted mixed linear carrier for interventions; oracle one-hot guard |
| mixed carrier | `d_hidden=64`, `kappa=100`, `seed=0` |
| decoded coordinate | `y = pinv(T) h`; linear arms may be signed; projected arms are simplex-valued |
| oracle access | construction, analytic derivation, exhaustive calibration, exact completions, planted decoder |

Any other `--confirm` setting routes `OUT_OF_SCOPE_CONFIG`.

## Pilot Outcome Labels

These labels describe the runner's pilot contract only. They are not exp-41
findings.

| route | condition |
|---|---|
| `OUT_OF_SCOPE_CONFIG` | command differs from the quarantined pilot scope |
| `HARNESS_FAIL` | process identity, normalization, analytic identity, or projection self-test fails |
| `NOT_DISSOCIABLE` | contrast counts or value-cell coverage differ from the expected product-counter structure |
| `LOW_TARGET_ROOM` | `a` or `b` lacks enough own-observable room |
| `CONTROL_LOW_ROOM` | `c` lacks high-room control signal |
| `LEAKAGE_FAIL` | analytic zero off-target movement fails |
| `TOO_EXPENSIVE` | exact `m=3` audit exceeds the cheap-ground-truth budget |
| `CARRIER_FAITHFULNESS_FAIL` | oracle or planted mixed carrier does not preserve completion behavior |
| `CEILING_FAIL` | contextual exact edit fails the positive-control scorer |
| `FLOOR_FAIL` | matched random directions pass intervention thresholds |
| `COHERENT_NONCONTEXTUAL_HANDLE` | a noncontextual arm passes both linear and projected coherence gates |
| `READOUT_ONLY_NONCONTEXTUAL_HANDLES` | noncontextual arms pass linear readout gates but fail simplex-coherent state checks with off-simplex evidence |
| `CONTEXTUAL_ONLY` | contextual exact works, but no noncontextual arm passes linear readout gates |
| `PROJECTED_NONCONTEXTUAL_FAIL` | projected noncontextual arms fail without the off-simplex diagnosis |

## Why The Pilot Still Fails As A Preregistration

The load-bearing quantities are the distinction between
`COHERENT_NONCONTEXTUAL_HANDLE`, `READOUT_ONLY_NONCONTEXTUAL_HANDLES`, and
`CONTEXTUAL_ONLY`, plus the guard predicates that make those routes meaningful.
Those quantities were inspected while the experiment was being repaired. The
thresholds and route table may now be sharper, but they were not fixed before
route exposure.

Unexcluded confounds for exp 41 itself:

| confound | consequence |
|---|---|
| non-blind threshold and route repair | route cannot be treated as prospective evidence |
| LLM-work creep across drafts | prose, tables, and code evolved after seeing pilot behavior |
| single planted carrier instance | even as a pilot, the design does not establish seed or `kappa` robustness |
| oracle/planted decoder access | transport to learned carriers remains a future question |

## Reuse Rules

Allowed reuse:

- the finite product-counter process;
- exact `m=3` audit machinery;
- the contextual ceiling, value-mean, observable-min-norm, and random-floor arms;
- the compact runner as a pilot harness.

Not allowed reuse:

- exp-41 as a positive, negative, or calibration result;
- exp-41 thresholds as already-earned gates;
- exp-41 route exposure as a "prediction";
- any conclusion about which intervention class is licensed.

## Required Clean Revival

A clean successor must be a new numbered experiment. Before its first
route-bearing run it must have:

- a reviewed writeup with fixed scope, thresholds, route table, and predictions;
- a committed runner that already implements every route predicate;
- a fresh untouched degree of freedom, such as held-out carrier seeds, held-out
  `kappa` values, or a newly prespecified product-counter variant;
- a result table that reports numeric arm summaries without post-hoc prose doing
  the adjudication.

## Results

No accepted result. Exp 41 is closed as a procedural failure and quarantined as a
design pilot.
