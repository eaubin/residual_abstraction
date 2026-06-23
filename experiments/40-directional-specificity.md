# Experiment 40 — Directional specificity of facet intervention: can `depth` and `top_type` be steered independently? — DESIGN DRAFT

**Status: design draft.** Not yet pre-registered: the question, the construct
(steering vectors, the 2×2 dissociation matrix, the ceiling/floor references), and
the verdict partition are fixed below; magnitudes (`α`-ladder), thresholds,
positions, counts, and the steer support are deferred to the pre-registration
(`<TBD-prereg>`) and gated by a feasibility smoke (below). State-localization phase
— the **interventional dissociation** rung. It brings the phase's destination
question (specific intervention — the well-posed `pstack`/ICB failure) forward to
the directional level, and **supersedes L2 (exp 39, coarse same-vs-different-parts)
as the next step**: a descriptive co-location map is replaced by the interventional
dissociation it was a proxy for. L2 is deferred, not dropped (it localizes *where*
a specific intervention acts, worth doing only once one exists).

## Why this, not L2 (what exps 37 + 38 jointly settled)

37 and 38 already answer the *prior* form of L2's question — "is each facet one
thing or several":

- **`top_type`** = a **certified clean m=1 point-summary** (37: `delta` 1.0, floor
  0, oe 0). Point-readable, point-transportable.
- **graded depth** = **carried but `DISTRIBUTED`** (38: recency-weighted, spread,
  no small window saturates, no single position necessary; 4/4 seeds).

A point-feature and a distributed-feature almost cannot fully co-locate, so L2's
same-vs-different verdict is semi-foregone. More importantly, 38 answered only the
**spatial** axis (graded depth is spread *across positions*). It did **not** answer
the **directional/representational** axis: at fixed spatial support, is the
depth-carrying residual *direction* separable from the type-carrying direction?
Spatial distribution does not imply directional entanglement — depth could be spread
across positions yet live in a subspace orthogonal to type at every position. **That
directional axis is what governs intervention specificity, and it is unmeasured.**
This rung isolates it.

## The question (a typed fork, well-posed in both directions)

```text
At fixed (full) spatial support, are the depth-carrying and top_type-carrying
residual directions SEPARABLE — i.e. can each facet be steered to a source value
without dragging the other — or are they ENTANGLED in the representation?
```

- **SEPARABLE / SPECIFIC** → the two facets are independently manipulable: the
  `pstack` coupling exp 36 could not adjudicate was *substrate*, not law. The
  well-posed positive the whole intervention thread was built to reach.
- **ENTANGLED** → steering one facet drags the other even on a toy built to
  separate them: the coupling is **law-like**, not a toy artifact. Routes to
  characterizing the binding.

## Why a directional steer, and why full replacement is the wrong instrument

The specificity question is **tautological under the full-prefix replacement patch**
(38/37's instrument): a full residual interchange at positions `≤ t` transports the
m=1 prediction *exactly* (37's model guard, bit-for-bit). So a full depth-source
patch makes the model read the *source's* `top_type` by construction — and since the
graded-depth source is `top_type`-matched (`depth_triples`), the off-target facet is
preserved trivially, `drag ≡ 0` by the transport guard. Full replacement hands the
model the answer.

Teeth require a **sub-residual, additive** intervention that moves *only the facet
content* and leaves the rest of each position's residual (all the off-target content
and nuisance) **intact**. The minimal honest such move is a **rank-1-per-position
steering vector**: add `α · v_f` to the clean residual, where `v_f` is the
mean residual difference between observable-matched classes that differ in facet `f`.
Because the off-target content is *not* overwritten (unlike replacement), whether
steering along `v_f` disturbs the off-target *readout* is a genuine empirical
question — directional separability, not bookkeeping.

This is the methods step past single-position interchange the phase deferred to L3.
It is **directional**, not spatial: spatial support is fixed (full scored prefix,
taking 38's distributed result as given), so the rung does not re-open localization.

## Construct: steering vectors from observable-matched pairs (the new core)

Both vectors are defined **only** by the Dyck parser's labels on observed tokens
(never oracle beliefs or model internals) and validated by their **behavioral**
effect — they are steering directions with a measured effect, **not** asserted as
the model's intrinsic features (the phase's standing direction-honesty rule).

- **`v_depth`** — over `top_type`-matched, depth-`(lo→hi)` pairs (`depth_triples`,
  already built): `v_depth(p) = mean[ resid_source(p) − resid_clean(p) ]` per prefix
  position `p`. Adding `α·v_depth` pushes the stack-depth belief `lo → hi`.
- **`v_type`** — over depth-matched, type-`(0→1)` pairs (`facet_pairs("top_type")`,
  already built): `v_type(p) = mean[ resid_source(p) − resid_clean(p) ]`. Adding
  `α·v_type` pushes the top-of-stack type `0 → 1`.

Each is **rank-1 per position** and **additive** (it adds the matched-difference
direction; it does not replace the residual). The matched pairing cancels the
off-target facet and, over many pairs, the nuisance — leaving the facet-specific
direction. The steer is applied at the **full scored prefix** `[0..t]` (38's
distributed regime for depth; symmetric for type), so the spatial axis is held fixed
and only the directional axis varies.

## Discriminator: the 2×2 dissociation matrix, read against measured references

Apply each steer to clean; read **both** facet observables (all reuse L0/L1 scorers):

```text
                      reads →   depth (graded: cr_cond k)   top_type (m=1 type-frac)
   steer α·v_depth              TARGET   f_dd               drag     f_dt
   steer α·v_type               drag     f_td               TARGET   f_tt
```

- depth target/drag is read on the **graded conditional** `cr_cond` at the
  horizon (`k=1` depth 1v2, `k=2` 2v3) — the 38 instrument, not the m=1 coarse
  proxy; `top_type` target/drag is the **m=1 type-fraction** (`facet_observable`).
- Each entry is a transport fraction toward its facet's source value, **normalized
  to that facet's own gap** (so `f=1` = "moved as far as a genuine intervention on
  that facet would"). `f_dd, f_tt` are target transports; `f_dt, f_td` are drags.

**The magnitude confound is handled by a sweep, not a point.** A large enough steer
moves everything. So sweep `α` and read the **drag at matched target-transport**:
at the `α` where a steer first reaches a reference fraction of its own ceiling, is
the off-target drag below the drag bound? The reported object is the
**target-vs-drag trajectory** over the `α`-ladder, vs the random-direction
trajectory — the curve-against-reference discipline 38 established, not a single
cell.

### Ceiling / floor references (the protocol's required baselines)

- **Target ceiling** (per facet): the **full-prefix replacement** transport — 38's
  `f_full ≈ 0.83–0.97` for depth, 37's certified `≈1.0` m=1 transport for
  `top_type`. The steer's target transport is read as a fraction of this: *does a
  rank-1 additive steer recover the facet move a full replacement achieves?* If not,
  the facet has **no low-rank steerable handle** (`NO_HANDLE`) — for depth this would
  extend 38's `DISTRIBUTED` to "not even rank-1 steerable," a real finding.
- **Drag floor** (no-specificity reference): the off-target facet's **own gap** —
  the move *its* genuine steer/replacement produces. `drag → 1` means the steer moved
  the off-target as much as targeting it directly would (full entanglement).
- **Random-direction control** (the off-manifold floor): steer along a random
  direction of **matched per-position norm**; it must move *neither* facet above the
  floor. This is the analog of 38's random-placement control — it separates "the
  facet direction does something" from "any push of this size perturbs the readout."
- **Sign/monotonicity self-test:** `+α·v_depth` raises the graded conditional toward
  `hi`, `−α` lowers it; likewise `v_type` toward type-1. A direction that is not
  monotone in its own facet is not a facet direction (`HARNESS_FAIL`).

## Verdict (exhaustive; substantive space is 4, the rest are standard guards)

```text
HARNESS_FAIL  — a guard/self-test fails: α=0 not bit-exact vs clean; steer not the
                intended rank-1 splice; random-direction steer moves a facet above
                floor; a facet vector non-monotone in its own facet. Blocks all.
OBS_DRIFT     — observable-vs-oracle endpoint gap > OE_BAND under the steer
                (uninterpretable readout).
NO_HANDLE(f)  — facet f's own steer does not transport its target above the
                random-direction floor at any α: no low-rank steerable handle
                (for depth, extends 38's DISTRIBUTED to non-steerable rank-1).
SPECIFIC      — BOTH facets steer their target (f_dd, f_tt ≥ ceiling fraction) with
                LOW cross-drag (f_dt, f_td ≤ drag bound) at matched target-transport:
                independently manipulable → the well-posed positive (separability
                has an interventional mechanism).
ENTANGLED     — at least one steer transports its target but DRAGS the other above
                the drag bound at matched transport: coupling in the representation
                → characterize the binding.
MIXED         — one direction specific, the other NO_HANDLE or dragging (a typed
                middle; names which facet is the entangling/unsteerable one).
SEED_UNSTABLE — no ≥3/4 cross-seed majority; underpowered.
```

Per `(position, horizon)`; reduced to a per-facet-direction verdict by position
majority, then a cross-seed `≥3/4` majority, then the matrix-level headline by a
registered precedence (`HARNESS_FAIL > OBS_DRIFT > SEED_UNSTABLE > ENTANGLED >
MIXED > NO_HANDLE > SPECIFIC`, so `SPECIFIC` is the headline only when nothing
couples or fails) — the 38 reduction pattern (`majority_vote` / `first_precedence`).

## Falsifiability and routing

| outcome | reading | routes to |
|---|---|---|
| **SPECIFIC** | depth & type are independently steerable; the `pstack` coupling was substrate, not law | localize *where* each specific steer acts (the deferred L2, now well-motivated); or scale to a richer setting |
| **ENTANGLED** | the facets share a representational subspace even on a separating toy; coupling is law-like | characterize the binding — the coupling is a finding with a mechanism, not an artifact |
| **MIXED** | one facet steerable & specific, the other not | the asymmetry is located (e.g. type point-steerable, depth not rank-1 steerable) → refine the unsteerable one (directions/L3) |
| **NO_HANDLE(both)** | neither facet has a rank-1 steerable handle | low-rank intervention is the wrong tool here → multi-direction / subspace steer, or accept distribution |

## Feasibility precondition (smoke before pre-registration — gates the rung)

The load-bearing unknown is **whether a rank-1 additive diff-in-means steer
transports the target facet at all.** 38/37 used full-residual *replacement*; a
rank-1 *additive* steer is strictly weaker and may not move depth (38's distributed
carrying could have no low-rank handle). The smoke: on seed 700, does `α·v_depth`
move the graded conditional toward `hi` above the random-direction floor, and does
`α·v_type` move type-fraction toward 1, for some α on the ladder? **If neither facet
is steerable, the specificity question is vacuous** (`NO_HANDLE` both) and the rung
reframes to a multi-direction steer before pre-registration — exactly as 38 smoked
its ceiling before committing. The smoke writes the target ceiling and
random-direction floor the thresholds are read against (`out/exp40_steer_smoke.txt`).

## Confound table — load-bearing quantity (the cross-drag `f_dt` / `f_td`)

| mechanism producing low drag (apparent specificity) | excluded by? |
|---|---|
| genuine directional separability (the intended signal) | this is what the matrix is built to detect |
| **the steer is too weak to move anything** (low drag because low effect) | drag is read **at matched target-transport** (the α-sweep), never at fixed α; a weak steer fails the target-ceiling fraction first → `NO_HANDLE`, not `SPECIFIC` |
| **off-target gap is small** so drag looks low in absolute terms | drag is **normalized to the off-target facet's own gap**, not absolute movement |
| random-perturbation insensitivity: the readout is just robust to any push | the **random-direction control** at matched norm must move neither facet — if it also looks "specific", specificity is uninformative |

| mechanism producing high drag (apparent entanglement) | excluded by? |
|---|---|
| genuine representational coupling (the intended signal) | this is what the matrix is built to detect |
| **the matched-difference vector leaks off-target content** (pairs imperfectly matched, so `v_depth` carries residual type signal) | pairs are parser-matched on the off-target facet (`depth_triples`/`facet_pairs`); the leak is bounded by the **same-facet floor** (a `v_f` built from same-facet pairs must produce ≈0 target transport — a registered self-test) |
| **off-manifold push** drags the readout artifactually | the random-direction control bounds generic off-manifold drag at matched norm; the oracle endpoint audit (`OBS_DRIFT`) bounds readout validity; **bounded, not eliminated** (additive steers leave the manifold — stated as a verdict bound, as 38 bounded its mid-curve hybrids) |

**Claim bound (pre-committed).** `SPECIFIC` means *"separable by a rank-1 additive
steer at full spatial support"* — not "separable by any intervention" and not "live
in orthogonal subspaces" (a richer steer could still couple, or a lower-rank one
fail). `ENTANGLED` means *"a rank-1 steer of one facet drags the other"* — coupling
under *this* intervention class, not a proof of inseparability in principle.
`NO_HANDLE` means *"no rank-1 additive handle"*, never "not manipulable." A negative
never licenses "the facets cannot be intervened on independently."

## What carries (not re-derived) / reuse vs single-use

The user asked the split explicitly. Most new work is **reusable infrastructure**;
little is throwaway.

- **Reusable spine (new, the rung's core):** a matched-pair **diff-in-means
  steering-vector builder**, an **additive rank-1 steer** applier over a position
  set, the **2×2 dissociation / drag scorer**, and the **α-sweep + matched-transport
  reducer**. All are facet- and toy-agnostic — they are the actual ICB specificity
  harness, finally well-posed (a certified control facet + a transport instrument
  exist). Home: new primitives in `localize.py`; orchestration + verdict in the rung
  script `scripts/localization/exp40_directional_specificity.py`.
- **Already built (carried):** `depth_triples` / `facet_pairs` (the matched pairs),
  `facet_observable` (m=1 readouts), `cr_cond` (graded readout), `stream_to` /
  `make_patched_prefix` / `q_at` (residual cache + forward), the checkpoint contract
  and bit-exact guards, 38's `f_full` ceiling and 37's certified `top_type` gap.
- **Single-use (throwaway):** only the Dyck-2 verdict itself and its registered
  thresholds.

So the deferred-generalization problem is **not** repeated: the mechanism is nailed
on the clean toy with reusable instruments before any broadening — the reason this
phase exists.

## Self-tests (known-answer, before any model claim)

- `α=0` steer reproduces the unpatched clean readout bit-exact;
- the steer is the intended rank-1 additive splice at the named positions (no
  off-by-one, off-target positions untouched);
- a **same-facet** `v_f` (built from same-label pairs) produces ≈0 target transport
  (the difference vector is facet-carried, not generic);
- the **random-direction** steer at matched norm moves neither facet above floor;
- sign/monotonicity: `+α` and `−α·v_f` move facet `f` in opposite, correct
  directions, monotone in α.

## Non-goals

- No spatial localization / same-vs-different-parts map (that is the deferred L2,
  re-motivated only by a `SPECIFIC`/`MIXED` outcome). No head/direction *enumerator*
  (L3): this rung uses a single matched-difference direction per facet, not a swept
  granularity. No multi-direction/subspace steer unless the smoke returns
  `NO_HANDLE`. No claim that the steering vectors are the model's intrinsic features.
  No real-LLM claim; the vehicle is fixed to the Dyck-2 checkpoint.

## Open design points (fill at pre-registration)

- the `α`-ladder and the matched-transport reference level; the ceiling/drag-bound
  thresholds (anchored to the smoke's measured ceiling and random-direction floor,
  as 38's were to the planted locus); the steer support (full prefix `[0..t]` vs the
  38 window family); per-position vs pooled-direction steering; positions (L0's
  `{8,12,16,20}`, held-out split for the specificity claim); seed set; whether the
  depth target is read at `k=1`, `k=2`, or both.
