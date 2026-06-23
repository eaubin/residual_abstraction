# End of phase — consolidation guidance

What to do when a **phase or research branch closes**. This is a phase-boundary
activity, not a per-experiment conclusion: it makes no new measurement and
usually has no script (precedent: exp 22 closed Phase 2, exp 28 closed the
oracle-withdrawal arc). A consolidation is reviewed like an experiment — the
`EXPERIMENT_REVIEW_PROTOCOL.md` standards apply — but it produces a record, not a
claim-bearing run.

Per-experiment conclusions have their own obligations (verdict fidelity, the
propagate-grep step) in `EXPERIMENT_REVIEW_PROTOCOL.md`. The steps below are the
*additional* once-per-phase writes.

## Trigger

A phase's question is answered, or a branch resolves (positively or as a typed
negative). If you are tempted to keep mining the same substrate without a new
diagnostic, that is a signal to consolidate and move, not to add experiments.

## Must-write checklist

1. **Promote stable findings to `BATTERY.md`.** New process indices, transferred
   members, and typed findings that outlive the phase go into the frozen battery
   under their registered indices. Phase-local detail stays in the writeups.
2. **Settle `ASSUMPTIONS.md`.** Move every affected ledger row to its final
   status for the phase (supported / scoped / falsified / …), each as
   status + one sentence + pointer. Run history stays in the writeups and git.
3. **Run the cross-doc propagate-grep.** Apply the mandatory grep step
   (`EXPERIMENT_REVIEW_PROTOCOL.md`, Result Review) across `EXPERIMENTS.md`,
   `ASSUMPTIONS.md`, `BATTERY.md`, `SYNTHESIS.md`, and prior writeups so no
   canonical record still asserts a quantity the phase revised.
4. **Archive the phase's design/plan doc.** When a phase closes, its design map
   moves to `docs/archive/` and its `INDEX.md` row moves to the archived
   section. Precedent: `PHASE2.md` and `ORACLE_WITHDRAWAL.md` were retired this
   way. A "live during phase X" doc must not linger in the root after X closes.
5. **State program disposition.** In the consolidation writeup, say plainly what
   is closed, what is optional, and what the next phase is. Update the roadmap in
   `RESEARCH_PROGRAM.md` if the phase order changed.

## Not required

- A new run or script — consolidation is bookkeeping over existing results.
- Updating `SYNTHESIS.md` on a fixed schedule — it is human-owned and updated
  asynchronously; refresh it at a phase boundary only if the standing synthesis
  actually moved.
