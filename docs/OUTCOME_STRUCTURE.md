# Outcome structure — representing results beyond the verdict scalar

A methodological **lens**, not a registered rule. It records a decomposition worth
keeping in mind when designing or reviewing an experiment. The linear, severity-ranked
verdict reduction (`FORMALISM §6`, `first_precedence`) remains the registered reduction
until a deliberate convention amendment changes it; nothing here silently overrides it.

## The seam

One severity-ranked ladder is currently asked to do three different jobs at once:
aggregate replicates, enforce guard dominance, and **combine genuinely different
conditions** into one headline token. The first two are real total orders and fine. The
third is not: different conditions can yield *incomparable* outcomes, and their
disagreement is often itself the finding. Forcing them onto one axis discards exactly the
cross-condition structure the experiment was built to measure.

Exp 40 surfaced this: `k=1` was `SEED_UNSTABLE` (underpowered) and `k=2` a stable
`CROSS_DRAG`; the ladder ranks "underpowered" above "found coupling," so the headline
read `SEED_UNSTABLE` and buried the stable result. Most earlier experiments had a single
outcome axis, so the projection was lossless enough and the seam stayed hidden.

## Three layers, three structures

Don't make one structure carry all three. Pick the lightest fit per layer.

| layer | what it is | structure that fits |
|---|---|---|
| **the phenomenon** | the scientific question under test (e.g. "does moving the close-mass *sum* causally drag the type *ratio*, or are the difference-directions merely non-orthogonal?") | **causal / structural model** — the steer is a `do()`; the verdict is a coarse readout of which edges are present |
| **one experiment's outcome** | the joint per-(direction, horizon, …) result | **partial order / antichain** — report the maximal incomparable findings; do **not** collapse to a scalar across condition axes |
| **accumulated knowledge** | how ledger propositions sharpen / are refuted across experiments | **evidence graph** — a result is an *update* (an edge) on the proposition graph, not a standalone label |

A fourth structure sits behind routing: **value-of-information.** You never need to rank
*outcomes* — you need to rank the next *action* by how much it would resolve the live
open question. Incomparable outcomes are then a non-problem. The precedence ladder is a
frozen, hand-coded approximation of this argmax; worth using as a sanity check ("does this
priority actually correspond to resolving the most uncertainty?"), not worth building.

## Standing leanings (lightly operationalized)

1. **Reduce replicate axes; do not reduce condition axes to a scalar.** Seeds and
   positions are replicates — majority-reduce them and keep a power flag. Horizons (and
   any deliberately-varied condition) are not replicates — report them as a short
   configuration / antichain, and key routing on the configuration, not on a single
   most-severe token. The per-axis labels stay as useful, auditable atoms.
2. **Treat a result as a diff on the evidence ledger**, not a standalone verdict.
   `ASSUMPTIONS.md` is already a degenerate version of this; the faithful statement of an
   experiment's result is which proposition it sharpened/refuted, with the scope bound.

## Back of mind (not operationalized; reach for case-by-case)

3. **Model the phenomenon causally when the question itself is causal.** When a headline
   confound is "causal vs geometric/correlational" (exp 40's sum→ratio is the exemplar),
   a small structural model over the relevant coordinates is the right language, and the
   discrete verdict is derived from it. Heavyweight — used at the confound, not routinely.
4. **VOI as the north-star for routing** (above). Held as the thing the ladder
   approximates, not implemented.

## The lens (questions at design / review time)

- Which axes here are **replicates** (reduce) vs **conditions** (keep incomparable)?
- Does the single headline token hide a stable finding behind an underpowered or
  guard-tripped sibling condition? If so, report the configuration.
- Is the load-bearing confound a **causal vs geometric** question? If yes, a structural
  model — not another threshold — is what would settle it.
- What ledger proposition does this result *update*, and with what scope bound?
