# Experiment 19 — Dyck baseline + threshold recalibration (Phase 2, Block 1) — PRE-REGISTRATION

**Script:** `dyck_baseline.py` (on `battery.py` + `expcommon.py`).
**Status: pre-registered (script committed); NOT YET RUN — results to
be appended below the marked line.**

**Question.** The frozen diagnostic battery (BATTERY.md) was calibrated
entirely on Mess3. Does it produce coherent, reproducible numbers on
Dyck-2 through the `battery.py` library, and what are the
process-appropriate thresholds? Exp 7 proved the instruments *run* on
Dyck (k\*=4, 92.6% exact closure, 5.9-point obs/exact gap) but used
inline code, predated the battery formalization, and never measured ρ
or retention. This experiment is the library's first use on a new
process, the reproduction gate for Phase 2, and the threshold
recalibration that Blocks 2–3 inherit.

**What is NOT in scope.** Adversarial coordinates, gradient-learned
reads, shift-retention, and held-out-position transport are Block 2–3
design work. This block establishes the baseline numbers and calibrates
the constants they need.

## Design

**Model.** The existing `out/dyck2-L4` checkpoint (4 layers, d_model 64,
seq_len 32, m=3, V=4). No retraining.

**Patch point.** L1 (exp 7's ℓ† by the registered argmax-step-2 rule;
reproduced here as a self-check, not re-derived).

**Pair sets.** Same seeds as exp 7 (disc: seed+111, 400 pairs; eval:
seed+777, 600 pairs; n_seqs=800). An additional val set (seed+333,
400 pairs, ts={12,20}) for the held-out-gain baseline.

**Reference patches.**
- *Full* (identity): the ceiling.
- *No-op* (None): the floor.
- *Discovered core* (rank-4): reproduced from the exp-7 CEGAR loop
  through `battery.cegar_loop` with eps=0.05, k_max=12.
- *Nested k=1..4*: the per-direction closure staircase.
- *PCA k=4, PLS k=4, rand k=4, emb k=4*: the exp-7 controls,
  reconstructed identically.

**Battery members exercised (sub-predicate table below).**

| member | what is measured on Dyck | Mess3 constant being recalibrated |
|---|---|---|
| 1 (obs closure) | Refs.obs on disc/eval sets for every reference patch, at mm=3 | — (no threshold to recalibrate; the score itself) |
| 2 (ρ) | Exact.rho of each non-reference patch against the discovered-core rank-4 as the trusted reference | ρ bands (Mess3: ≤0.25 equivalent, ≥0.5 distinct) |
| 3 (held-out gain) | Refs.obs on the val set for the discovered core and full patch | — (baseline measurement for Block 2) |
| 5 (P4 calibration) | calibration_gap on every (patch, set) cell with obs ≥ 20% | obs/exact band (Mess3: 0.10; exp 7 recorded 5.9 pts) |
| 6 (CEGAR) | cegar_loop at mm=3 reproducing k\*=4; cegar_staircase over eps ∈ {0.01, 0.02, 0.05, 0.10} | eps_gain (Mess3: 0.05; relative-gain acceptance) |

Member 4 (shift-retention) is deferred to Block 2 (requires a
Dyck-native shift design: depth-profile shift, not the Mess3
position/init-state shifts).

**Recalibration procedure.** Every Mess3-calibrated constant is
*measured*, not inherited. The measured values become the Block 2–3
registration inputs. Specifically:

1. **obs/exact band**: report the worst |obs − exact| gap across all
   accepted cells (obs ≥ 20%). If the gap exceeds 0.10, this is a
   *finding* (the Mess3 band does not transfer), not a failure — record
   the Dyck-appropriate band as the smallest round value that covers
   every cell.
2. **acceptance gain**: report each direction's marginal gain as a
   fraction of the full-patch obs closure (not absolute percentage
   points). Record the Dyck relative-gain profile for the eps staircase.
3. **ρ bands**: compute ρ for (discovered core vs full), (PCA vs full),
   (PLS vs full), (rand vs full), (emb vs full) with the discovered core
   as the reference. The destructive/equivalent separation — if it
   exists — sets the Dyck bands.

## Scope & local assumptions

- Patch point L1 inherited from exp 7 (not re-derived; the ℓ† assert
  is the reproduction check).
- m=3 standing horizon (the m-staircase is Block 3).
- The discovered core is the battery.cegar_loop output with eps_drop=0.01
  (the exp-6/7 full loop including coarsen, since this is discovery, not
  the accept-only staircase predicate).
- Pair-set construction identical to exp 7 (seeds, counts, positions).

## Pre-registered predictions

**P1 (exp-7 reproduction; ~95%).** The battery.cegar_loop reproduces
exp 7's CEGAR trajectory: k\*=4, c_obs within 2 pts of 98.5%, and the
nested closure staircase within 2 pts of {37.8, 71.6, 85.0, 92.6}%.
The ℓ† assert (L1) holds. Validity gate passes. Failure = halt.

**P2 (obs/exact agreement exists on Dyck; ~75%).** Every (patch, set)
cell with obs ≥ 20% has |obs − exact| ≤ 0.15. The wider band (vs
Mess3's 0.10) is the directional bet: exp 7's 5.9-point gap suggests
the agreement is real but looser, and 64 outcomes (vs 27) give the
model's softmax more room to diverge from the oracle's exact chain.
**If the worst gap exceeds 0.15, the band is set to the measured worst
+ 0.02 margin and this is a finding, not a nuisance.**

**P3 (ρ separates on Dyck; ~70%).** The discovered core's ρ against
itself (via full-vs-core) is ≤ 0.25 (the core is behaviorally close to
the full patch). PLS ρ ≥ 0.50 and rand ρ ≥ 0.50 (behaviorally
distinct from the core). **If the Mess3 bands hold on Dyck unchanged,
that is an unexpectedly strong transfer; if they don't, the measured
separation ratio sets the Dyck bands.**

**P4 (controls; ~95%).** rand k=4 exact closure ≤ 25% at every
horizon. PLS k=4 exact closure ≤ 5% (the echo on Dyck is even more
extreme than on Mess3, per exp 7's 0.2%). Full-patch closure within
2 pts of exp 7's 93.6%.

**P5 (val-set baseline; descriptive).** Discovered-core obs closure on
the val set reported. No threshold (this is a Block-2 input, not a
Block-1 verdict).

**P6 (eps staircase shape; ~80%).** k\*(eps) is weakly decreasing in
eps. k\*(0.01) ≤ 8 (the loop does not blow up at fine resolution).

**Adjudication.** P1 failure halts (the library doesn't reproduce
exp 7). P2 failure triggers the measured-band protocol (not a block
failure). P3 failure with separation present but at different thresholds
triggers band recalibration. P3 failure with no separation (ρ flat
across working and destructive patches) is a genuine battery-transfer
failure — escalate to a dedicated experiment. P4/P6 failure investigated
normally.

---

*Results to be appended below this line after the run.*
