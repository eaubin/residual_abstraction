# Experiment 39 — L2 coarse localization: where do the facets live, same or different parts? — DESIGN DRAFT (DEFERRED behind exp 40)

**Status: design draft, DEFERRED — not a live pre-registration.** The strategic fork
at the foot of this draft (L2-as-drafted vs pivot to the intervention thread L1
opened) resolved toward the **interventional dissociation** (exp 40, directional
specificity): the descriptive same-vs-different-parts *co-location map* is replaced as
the next step by the *interventional* dissociation it was a proxy for. L2 is **not
dropped** — it localizes *where* a specific intervention acts, re-motivated by a
`DISSOCIATED`/`MIXED` outcome at exp 40. The body below is the unchanged L2 design;
review it as a deferred sketch, not against the pre-registration bar.

**Two construct blockers carried here, both already solved in exp 40 (review,
2026-06-22).** If L2 ever revives, it inherits these — do not run the body as-is:
(1) **Specificity must be cross-facet, not facet-vs-total.** close-readiness =
`q2+q3` (sum) and `top_type` = `q2/(q2+q3)` (ratio) are coupled projections of the
same close mass, so a unit that moves the close distribution generically scores as
"specific" against the *total* vocabulary and inflates SAME_PARTS. Score the
**sum-component vs ratio-component** of each unit's `(q2,q3)` effect instead. Exp 40
does this by construction (cross-facet drag + pure-sum/pure-ratio matched directions).
(2) **Isolate the write additively, not by residual replacement.** The residual is
cumulative (pre-LN `x += write`), so overwriting the post-sublayer residual imports
everything ≤ it; add `(source_write − clean_write)`. Same "add the difference, don't
replace" move as exp 40's additive steer — shared machinery when L2 revives.

**Status: design draft.** Not yet pre-registered: the question, units, and
discriminator are fixed below; thresholds, positions, counts, and the exact
patch-isolation choice are deferred to the pre-registration (`<TBD-prereg>`).
State-localization phase, **L2** — the phase's deferred **first localization claim**
(L0 = substrate gate, exp 37; L1 = propagation gate, exp 38). The block/head unit
enumerator — deferred at L0/L1 — is **earned here** (L1 returned `DISTRIBUTED`, not a
NO-GO), and is this rung's new reusable core.

## The question (the phase's core question, now mechanistic)

```text
For each completion-relevant facet, where does it live across the architecture-given
parts (per-layer attention vs MLP writes, by position) — and do `top_type` (a
"what") and close-readiness (a "where") occupy the SAME parts or DIFFERENT parts?
```

- **DIFFERENT parts** → separability has a *mechanism*: the `pstack` coupling exp 35
  could not adjudicate was substrate, not law. Routes to intervention at located parts.
- **SAME parts** → genuine coupling/binding into shared parts; characterize the binding.
- **NO_LOCUS / distributed** → no unit separates from the random-unit floor; the
  expected-difficulty outcome (superposition), located, not forced into a clean handle.

## What carries from L0/L1 (not re-derived)

- **Targets are settled.** `top_type` is the **certified clean substrate** (L0:
  `delta` 1.0, floor 0, oe 0). Close-readiness is the **m=1 depth summary**.
  **Graded depth is `DISTRIBUTED` (L1, exp 38)** — carried but not point-localized —
  so it is **not a clean-locus target here**; any graded-depth importance is read as
  spread, never forced onto a single unit. L2 localizes the two **m=1 summary
  facets** (`top_type`, close-readiness); the graded-depth localization question was
  answered at L1 (distributed) and its *where-along-the-prefix* refinement is the
  dynamics rung's, not L2's.
- **Pairs, observables, patch/joint machinery, the checkpoint contract** from
  `localize.py` (`facet_pairs`, `facet_observable`, `make_patched_prefix`/`q_at`,
  `require_expected_config`). The **same-depth / mismatched-source floors** and the
  **oracle endpoint audit** carry as the control discipline.

## Units: architecture-given block writes (the enumerator)

The model is **pre-LN** (`model.py`): each layer adds two terms into the residual,
`x += attn(ln1 x)` then `x += mlp(ln2 x)`. The architecture-given units are therefore

```text
unit = (layer ℓ ∈ {0,1,2,3}, sublayer s ∈ {attn, mlp}, position t)
```

— the additive **writes** into the residual stream. The enumerator caches, for clean
and source runs, the residual immediately after each sublayer write at each position,
and **interchange-patches one unit** (source write replaces clean at that `(ℓ,s,t)`),
then completes the forward pass and reads the facet observable. Granularity is one
knob (block now; head/direction deferred to L3, built only if a coarse map demands
it). **Open design point (prereg):** isolating the *write* (the additive term) vs
overwriting the *post-sublayer residual* at `(ℓ,t)` — the residual-vs-write patch
choice — with a self-test that the chosen isolation is exact and that summing
single-unit effects approximates the full-patch effect where writes are independent.

## Discriminator (importance × specificity, then same-vs-different)

Per facet `f` and unit `u`, two scores (both reuse L0/L1 observable scorers):

- **importance** `I_f(u)` = movement of the facet observable toward source under the
  single-unit interchange patch, normalized by the full-patch ceiling and read
  against the **random-unit floor** (a same-count patch at random units — the real
  baseline L0 deferred, now built for blocks);
- **specificity** `S_f(u)` = facet effect vs **total effect** (movement of the full
  next-token / m-gram distribution): a unit specific to `f` moves `f` but not
  everything; a broad unit moves both.

The **same-vs-different-parts** verdict compares the two facets' importance maps:
do the units that carry `top_type` and the units that carry close-readiness
**coincide** (same parts) or **separate** (different parts), above the random-unit
floor and with specificity. Overlap is the registered statistic (e.g. rank
agreement / Jaccard of the supra-floor units), with a permutation null.

## Verdict shape (exhaustive; to finalize at prereg)

```text
HARNESS_FAIL   — a guard/self-test fails (full patch != ceiling; planted-unit not recovered; no-op not bit-exact); blocks all
OBS_DRIFT      — estimator-vs-oracle endpoint gap too large
NO_LOCUS(f)    — no unit's importance clears the random-unit floor for facet f: distributed/superposed, located not forced
DIFFERENT_PARTS — both facets localize (supra-floor, specific) to LARGELY DISJOINT units: separability has a mechanism
SAME_PARTS      — both facets localize to LARGELY SHARED units: coupling/binding with a mechanism
MIXED           — one facet localizes, the other is NO_LOCUS, or partial overlap below the registered separation margin (a typed middle)
SEED_UNSTABLE   — no stable majority across seeds; underpowered
```

## Baselines and confounds (the protocol's two required artifacts)

- **Baselines (ceiling/floor for every threshold).** Ceiling = full-patch transport
  (already exact for m=1 by the L0 guard). Floor = **random-unit** patch (importance)
  and **mismatched-source** pair (no facet diff → ≈0). A **planted-unit** synthetic
  (a known unit made to carry the facet) is the importance ceiling and a self-test —
  the analog of L1's planted locus. The overlap statistic gets a **permutation null**.
- **Confound table — load-bearing quantity (the overlap / importance map).**
  - *Redundancy / backup* (the named phase confound): single-unit patches under-read
    importance if computation is duplicated → a unit can read NO_LOCUS while being
    used. **Not excluded by single-unit patching**; the registered response is a
    multi-unit / small-group patch where the single-unit map is ambiguous (only if
    the coarse map demands it). A `NO_LOCUS` is therefore bounded to "not
    *single-unit*-localizable," never "not used" — the L1 redundancy bound, recurring.
  - *Off-manifold* patch: interchange (not zero-ablation) keeps the write near the
    observed manifold; mismatched-source ≈0 is the check.
  - *Importance-without-specificity*: a unit that moves everything (high total effect)
    is not facet-localization — the specificity axis separates it (the exp-34/35
    broad-replacement lesson).
  - *Overlap inflated by a shared broad unit*: the overlap statistic is computed over
    **specific** supra-floor units, not raw importance, so a globally-important unit
    does not manufacture SAME_PARTS.

## Self-tests (known-answer, before any model claim)

- planted-unit: a synthetic where a known `(ℓ,s,t)` carries the facet must be
  recovered (importance peaks there, specificity flags it);
- no-difference (mismatched-source) → importance ≈ 0 at every unit;
- full-patch reaches the ceiling; no-op patch reproduces unpatched bit-exact;
- single-unit patches are exact reconstructions of the intended write splice;
- the overlap statistic returns ≈ chance under a label permutation.

## Reuse / consolidation

- `localize.py` gains the **unit enumerator** (cache sublayer writes; single-unit
  interchange patch; importance/specificity/overlap reducers), self-tested. L0/L1
  scripts frozen. The rung script `exp39_coarse_localization.py` orchestrates.
- The enumerator's one granularity knob is designed so block → head → direction is
  the same code; **only block is built here** (head/direction deferred to L3, the
  seam added when a coarse map demands it — the standing guardrail).

## Open design points (fill at pre-registration)

- residual-vs-write patch isolation; the registered positions (L0's `{8,12,16,20}`,
  held-out split for the separability claim); the supra-floor / overlap-separation /
  specificity thresholds (anchored to the planted-unit ceiling and random-unit floor,
  as L1's were to the planted locus); seed set; whether MLP and attention are scored
  jointly per layer or separately; the multi-unit redundancy follow-up trigger.

## Non-goals

- No head/direction enumerator yet (L3). No graded-depth single-locus claim (settled
  `DISTRIBUTED` at L1). No intervention/control claim at the located parts (that is
  the phase the DIFFERENT_PARTS route *graduates to*). No real-LLM claim.

---

**Strategic note (for the review pause).** Two live directions after L1's
`DISTRIBUTED`:
1. **L2 as drafted** — localize the *summary facets* (`top_type`, close-readiness)
   and answer the phase's same-vs-different-parts question. Concrete, well-posed
   (`top_type` is a certified substrate), and it is the phase's prescribed first
   claim.
2. **Pivot to the propagation/dynamics question L1 opened** — exp 38's most striking
   finding was *recency-weighted distributed* carrying of graded depth; the natural
   follow-up is *where along the prefix* the distributed carrying concentrates and
   *which heads move it forward* (the old L4 dynamics rung). This chases L1's positive
   result rather than the deferred localization claim.

I drafted (1) because it is the phase's registered next step and unblocks the
DIFFERENT/SAME-parts verdict that the whole phase was built to reach. But if the
recency-gradient finding is the more interesting thread to you, (2) is the live
alternative — worth deciding before this goes to pre-registration.
