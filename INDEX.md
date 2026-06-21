# Doc index — what to read, and when

Read this to find the *one or two* docs your task needs. Do not bulk-read.
Per-turn loads should stay small: `AGENTS.md` + this file, then pull on demand.

## Always

| Doc | What it is |
|---|---|
| `AGENTS.md` | Behavioral rules for every task. Small by design. |
| `INDEX.md` | This map. |

## Read on demand (by task)

| Doc | Read it when… |
|---|---|
| `RESEARCH_PROGRAM.md` | designing an experiment or judging whether a claim fits the program — frame, roadmap, scope honesty. |
| `EXPERIMENT_REVIEW_PROTOCOL.md` | pre-registering or reviewing a claim-producing experiment. Canonical review standard. |
| `END_OF_PHASE.md` | closing a phase or branch — the consolidation checklist (promote, settle, propagate, archive the phase doc). |
| `AGENT_WORKFLOW.md` | running the worker/reviewer loop (`scripts/experiment_agent_loop.py`). |
| `FORMALISM.md` | you need a definition of a named object (objects, orders, the invariance proposition) or the §6/§6.1 verdict taxonomy and registration checklist. |
| `ASSUMPTIONS.md` | you need the current status of a global bet or scope debt (the cross-experiment assumption ledger). |
| `EXPERIMENTS.md` | you need the experiment log index. From there open only the specific `experiments/NN-*.md` your task touches. |
| `experiments/NN-*.md` | working on, extending, or reviewing experiment NN. This is the primary spec — prefer it over the aggregate docs. |
| `BATTERY.md` | touching the frozen diagnostic battery (Mess3 calibration, Dyck/pstack transfer) or its scope statement. |
| `SYNTHESIS.md` | you want the standalone, plain-language synthesis of what the project has learned (no exp-N detail, widely-accepted terms). |
| `README.md` | you need the outward project overview. |

## Archived (`docs/archive/`)

Historical / concluded / superseded. Preserved for provenance; not part of any
per-turn load. Read only if a task explicitly reaches back to that arc.

| Doc | Was |
|---|---|
| `docs/archive/HANDOFF.md` | re-entry pointer; self-declared outdated. |
| `docs/archive/ORIGINAL_SIN.md` | reflection on drift from the original idea. |
| `docs/archive/ORACLE_WITHDRAWAL.md` | the oracle-withdrawal program (exps 24–28, concluded). |
| `docs/archive/INTERVENTION_CLASS_BENCHMARK.md` | the Phase 3 intervention-class design map (exps 29–35, concluded at exp 36). |
| `docs/archive/PHASE2.md` | Dyck-2 battery port (concluded at exp 22). |
| `docs/archive/RESIDUAL_METHODS_NOTE.md` | future comparison-point note. |
| `docs/archive/REFERENCES.md` | bibliography. |
