# Experiment 45 — Facet-factor decodability and composition (type × color) on colored Dyck-2

**Status: CONCLUDED — `JOINT_OUTSIDE_SPAN` 4/4 (seeds 801–804); see Result
below.** The runnable script
(`scripts/predicate_sufficiency/exp45_facet_composition.py`) is committed; the
`--calibrate` artifact (burned seed 800, logged in `docs/SCOUTS.md`) is run and
its numbers are frozen into the constants. The seed-777 redirection is a **scout**
(burned, non-claim) and is intentionally **not** reproducibility-gated — its
generating script is not committed and need not be (review finding 5 is retired:
the claim-artifact reproducibility requirement applies to claim runs, not the
scout lane). The four claim checkpoints (`out/colored_dyck2-s801..804`) were
trained and the claim run is complete — see **Result** below.

Revision history. The **first** revision excluded the multiplicative
closing-gate confound (**gate-normalization** + a closes-orthogonal geometry) and
added the empirical composition floors (`--calibrate`). The **second** reframed
the conjunction axis: `ΔR²` is sound (for a true direct sum the AND ceiling caps
`R²_full` and `R²_span` equally and cancels), but the old `INTERACTION_PRESENT`
label conflated a *dedicated-linear* joint direction with genuine nonlinearity —
so the axis gates the joint on a kNN comparison and splits `JOINT_OUTSIDE_SPAN`
(linear) from `NOT_LINEARLY_DECODED` (nonlinear). The **third** (this review)
(a) wired the kNN gate into `cell_verdict` (it was computed but unread) and added
`NOT_DECODED` for the linear-and-kNN-both-fail case; (b) replaced the refuted
absolute/bracket separability cut with a **relative** cut (observed vs the
per-seed GT ceiling, margin `ANGLE_MARGIN`, guarded by `CEIL_MIN` →
`ANGLE_UNRESOLVED`), which moves the GT ceiling into the verdict as a registered
supervised-on-ground-truth reference; (c) added the **data-matched** `ΔR²` floor
(real `r_⊥`, estimated directions) governing `COMP_GAP`; and (d) dropped `TAU` as
a decode gate (a near-binary `psi` gives ~0.175 pooled-mean error even at
`R²≈0.77`; estimator soundness is the separate `OBS_DRIFT` audit). Calibration
(s800, burned) then froze `COMP_GAP = 0.0018` and confirmed `ANGLE_MARGIN = 15°` /
`CEIL_MIN = 60°`; it read `JOINT_OUTSIDE_SPAN` cleanly (`ΔR² = 0.225` vs the
~0.002 floor), which informs the credences below.

## Result (claim run, seeds 801–804, 2026-06-24)

**`JOINT_OUTSIDE_SPAN`, 4/4 seeds** (`out/exp45_colored_dyck2.txt`) — the burned
calibration reading replicated out-of-design. All guards passed every seed:
validity gate −0.019…−0.035 nats (converged), OBS_DRIFT 0.008–0.014 ≤ `OE_BAND`,
degeneracy `matches_*` coincident (≈0.814). What the cells measured, on colored
Dyck-2 at layer 2, determined-ctx prefixes `t ∈ {8,12,16}` pooled, affine-ridge
probe, m=3:

- **Marginals decodable; separability is the weak (by-construction) cell.**
  `psi_type`, `psi_color` linear-R² 0.79–0.86 (≫ `R2_MIN`). The readout angle
  **equals** the per-seed independent-pair (GT) ceiling to one decimal in every
  seed (86.8=86.8, 85.5=85.5, 79.6=79.6, 84.7=84.7). Read this carefully: on
  determined-ctx `psi ≈` the top's label, so `w_facet ≈ w_label` **by
  construction** (finding-6 caveat), which makes `observed ≈ ceiling`
  near-tautological — axis 1 is **not** an earned separability positive. Its only
  informative content is the *absence of a psi-vs-label divergence*: had the
  model's facet readouts been entangled while the labels were separable, the angle
  would have dropped below the ceiling; it did not. So "no entanglement" here means
  "no readout divergence from the labels," not a discovered clean geometry.
- **The conjunction is on a dedicated LINEAR direction outside the marginal
  span.** `psi_both` linear-R² 0.79–0.86 (decoded), but `ΔR² = R²_full − R²_span
  = 0.224–0.249` — ~125× the data-matched direct-sum floor (`COMP_GAP = 0.0018`).
  The joint is linearly stored, **not** a linear direct sum of the two marginal
  directions, and **not** a nonlinearity (the kNN gate did not fire: `R²_full ≥
  R2_MIN` for all four). *Floor caveat:* `data_directsum_dr2` plants a linear
  in-span signal (`Rp·ŵ_type + Rp·ŵ_color + noise`), not a Boolean AND, so it
  could in principle under-estimate the `ΔR²` an AND-target produces for a *true*
  direct sum under imperfect (R²≈0.8) marginal decoding. This is bounded, not
  load-bearing: if the residual is linear in `(type,color)` only, every linear
  functional of full `r_⊥` is a linear combination of those two bits (+noise), so
  `R²_full` is capped at the same AND-from-marginals ceiling as `R²_span` and
  `ΔR² ≈ 0` — the observed 0.24 (and the `joint_linear` selftest's 0.32) requires
  a component **outside** the `(type,color)` span, which the 125× margin makes
  unambiguous.
- **Compression.** `k* = 5` (full-m-gram sufficient subspace) vs rank-1 facet
  readouts — each named facet conditional needs one residual direction where the
  full completion needs five.
- **Premise-audit (descriptive).** Product-of-decoded-marginals vs `psi_both`:
  0.746–0.819, i.e. `≈ R²_full` — exactly the decode-noise ceiling, **not** the
  0.99 the registration mistakenly expected (the audit multiplies the R²≈0.80
  decoded marginals, not the clean labels). This is **not** a type ⊥ color
  independence violation; a value well *below* `R²_full` would be (see controls).

**Scope (what this is and isn't).** Correlational decodability in the
affine-ridge probe class (a V-information statement), on colored Dyck-2 at the
registered layer/positions/horizon; **binding is assumed, not tested** (forced
matching identifies facet-of-close with facet-of-top); `psi` is near-binary on
determined-ctx (effectively the top's facet label). **No causal claim** — whether
an interchange edit on the joint direction moves the conjunction independently of
the marginals, against the entanglement null, is the next rung. The GT-ceiling
angle is a registered supervised-on-ground-truth reference (it sets the
separability scale and the `CEIL_MIN` guard), not a decoded target.

**Routes to** the causal rung: characterize the residualized joint direction
`w_both ⟂ span` and test whether it moves the conjunction causally independent of
the marginal edits (the inverse-intervention goal with the entanglement null).
This does **not** resolve the `pstack` coupled-stack-state ledger row
(`ASSUMPTIONS.md`): different process, different facet pair, and correlational
only — that row is a *causal* question on a different toy.

## Phase fit and the scope debt this resolves

First **claim** of the completion-predicate phase (`docs/COMPLETION_PREDICATES.md`):
relate residual abstractions to **named** completion predicates (not the
full-horizon scalar), and measure **composition**. This is the original payload
the program deferred (`docs/archive/ORIGINAL_SIN.md`) — a relation between two
abstraction languages, one over residual states, one over named properties of
futures. It runs on **colored Dyck-2** because composition needs **two** binding
factors (type AND color); dyck2 has only one, and its depth/type entanglement is
already exhausted (exps 40/42/44). The machinery (graded predicate layer,
ctx-reader, colored_dyck2) was built and self-tested as non-claim infrastructure
(commits 7479a8d…a8d2180); this rung is the first to stake a claim on it.

## The predicate suite, the gate, and gate-normalization (load-bearing)

The intuitive "the next close **matches** the prefix top" is **degenerate for
composition**: colored-Dyck matching is grammar-forced, so a closing top matches
on **both** facets at once. The calibration scout (seed 777) confirmed
`matches_type` = `matches_color` = `matches_both` numerically (all exact-mean
0.811, identical) — a **design-redirecting** result, logged burned in
`docs/SCOUTS.md` (a scout: non-claim, burned, intentionally not
reproducibility-gated — finding 5 retired). So the suite uses **facet-VALUE**
predicates — the close that pops the seeded top
has a *specific* facet value — whose values key on **different** facets:

- `phi_type0(c; ctx)` — the close popping the seeded top has **type 0**.
- `phi_color0(c; ctx)` — …has **color 0**.
- `phi_both00 = phi_type0 ∧ phi_color0` — …has **type 0 AND color 0** (the joint).
- `phi_closes(c; ctx)` — …**pops the seeded top in-window** (any facet). This is
  the **gate**: `E_q[phi_closes] = p_close(x)` exactly.

**The multiplicative-gate problem (review finding 1), and its exclusion.** Each
facet score is a PRODUCT of the gate and a facet bit:

```text
E_q[phi_type0](x)  = p_close(x) · P(type=0 | closes, x)
E_q[phi_both00](x) = p_close(x) · P(type=0, color=0 | closes, x)
```

`p_close(x)` varies per prefix with depth. Left in, it confounds **all three**
load-bearing quantities: a product is non-linear in `r` even if `r` encodes the
facet cleanly (spurious `NOT_LINEARLY_DECODED`); `w_type0` and `w_color0` both
load on the shared `p_close` direction (deflated angle → spurious
`ENTANGLED_FACTORS`); and `phi_both00` is a triple product (out-of-span structure
forced by the gate, not by the factors → spurious `JOINT_OUTSIDE_SPAN`). Naming it
is not excluding it.

**Registered exclusion — gate-normalization.** The decode target is the
**gate-normalized conditional**

```text
psi_facet(x) = E_q[phi_facet](x) / E_q[phi_closes](x) = P_model(facet | closes, x)
```

restricted to prefixes with `E_q[phi_closes] ≥ CLOSE_FLOOR` (ratio stability;
calibration: determined tops close with mean ≈0.81, so most qualify). Division
removes `p_close` **by construction** — `psi` is the model's *conditional* facet
probability given the top closes, observable (both terms from the model's `q`).
The remaining structure `psi_both ≈ psi_type · psi_color` (type ⊥ color in the
generator) is now the **genuine composition subject**, not the nuisance gate: the
ΔR² test asks whether the (type,color) **joint** is recoverable from the span of
the two marginal directions (a linear direct sum) or needs a direction **outside**
that span — and, if outside, whether that direction is a dedicated *linear* joint
axis or a genuinely *nonlinear* feature (the kNN split in axis 2 below).
"Outside the marginal span" is **not** the same as "higher-order"; the two are
distinguished, not conflated. A residual-side **closes-orthogonal** geometry
(below) is the additional safeguard for the angle.

`matches_*` is **retained as a registered DEGENERACY control** (the three must
coincide; separation → `HARNESS_FAIL`).

## Scope of the construct (review finding 3 — what is and isn't tested)

Forced matching makes *facet-of-close ≡ facet-of-top*. So the suite decodes the
seeded **top's** (type, color) and their conjunction — a prefix feature read out
through the closing dynamics — **not the open↔close binding relation** (the two
are identified by the grammar and cannot be distinguished here). "Composition of
factors" is genuinely tested (the joint vs marginals question is real); **binding
itself is assumed, not tested**. Wording throughout is "facet factors," not
"binding."

**`psi` is near-binary, not graded (review finding 6).** Because prefixes are
determined-ctx, the seeded top's (type, color) is known exactly, so each
`psi_facet(x) = P_model(facet | closes, x)` collapses to ≈{0,1} per prefix
(softened only by the model's <0.005 imperfection) — effectively a **binary label
of the seeded top's facet**, not a graded conditional. The "facet conditional
given the top closes" framing reads as more graded than the construct is. This is
the root of the conjunction-axis treatment below: the joint of two near-binary
facets is a Boolean AND, so the composition question is specifically "is the AND
in the marginal span, outside it, or nonlinear."

## The question

```text
On colored Dyck-2, at the registered probe layer/positions: from which residual
directions are the gate-normalized facet conditionals psi_facet = P_model(facet |
top closes, x) (observable, from the model's own completions) decodable by a
bounded LINEAR probe — and do the two facet FACTORS (type, color) compose?
Specifically, in the closes-orthogonal complement: (a) are the type and color
readout directions SEPARABLE (angle near the independent-facet reference) or
ENTANGLED (angle below it), and (b) is the conjunction psi_both decodable from the
span of the two marginal directions (a linear DIRECT SUM), or does it need a
direction OUTSIDE that span — a dedicated LINEAR joint axis, or (only if linear
decoding fails but a richer probe succeeds) a genuinely NONLINEAR feature?
```

This is **correlational** sufficiency by **decodability** (a V-information
question in a fixed probe class), explicitly distinguished from causal
patch-sufficiency: per phase decision 2, correlational sufficiency runs **before**
the causal loop, and the on-manifold interchange test + the entanglement null
(decision 3) are the **next** rung, a registered non-goal here.

## Why decodability, and the compression reference (not a tautology)

Full-m-gram sufficiency **trivially** implies predicate sufficiency: if an
abstraction preserves the whole completion `q`, it preserves `E_q[phi] = mask·q`
for every `phi`. So "does the m-gram-sufficient abstraction stay sufficient for
predicates" is vacuously yes and is **not** the claim. The non-trivial questions
are **compression** (a scalar predicate should need far fewer residual dimensions
than the full `q`) and **composition geometry** (how the factor directions
relate). The full-m-gram `k*` (the sufficient-subspace dimension from the exp-6 /
`discover.cegar_loop` machinery, pinned to these checkpoints) is reported as the
**compression ceiling** the predicate readouts are measured against.

## Instrument

- **Vehicle:** fresh-seed `colored_dyck2` checkpoints (depth 3, V=8), the
  `train.py` config registered below. Validity gate (gap-to-optimal ≤ 0.005)
  must pass — the noise floor is only interpretable on a converged model.
- **Residual:** the stream after block `ℓ` (registered `PROBE_LAYER`), at
  position `t`, paired with the exact belief (for the ctx-reader / ground-truth
  controls only).
- **Observable target:** `E_q[phi]` per prefix from the model's completion `q`,
  computed as `q @ mask` where `q` is the splice-and-`chain_probs` joint (the
  script's `model_q`, chunked over prefixes; the continuation row order matches
  `predicates.continuations`/`graded_mask`, so this is `eq_obs_from_model` inlined
  to reuse one `q` across the whole facet suite + gate). Device-aware, m=3. This is
  the **observable** the probe predicts; ground truth (`eq_exact`) enters only as
  the calibration audit (`OBS_DRIFT`) and the registered ground-truth control.
- **Prefixes:** DETERMINED-ctx prefixes only (the ctx-reader is exact there;
  `predicates.ctx_along`), at `t ∈ {8,12,16}`, pooled. Inherited-top /
  belief-integrated prefixes are out of scope here (their observable semantics is
  the latent-toy question — `docs/COMPLETION_PREDICATES.md` design discipline).

## Probe class and the sufficiency criterion (calibration-grounded)

- **Probe class = affine ridge** (the registered bounded class; V-information is
  indexed to it). The target is the **gate-normalized** `psi_facet` (above), on
  prefixes with `E_q[phi_closes] ≥ CLOSE_FLOOR`. Fit `psi_facet ≈ w_facet · r + b`
  on the TRAIN split, score on a disjoint HELD-OUT split (decision 4). The fitted
  **`w_facet`** is the rank-1 residual abstraction for that facet.
- **Closes-orthogonal fits (review finding 1, geometry safeguard).** Fit the gate
  direction `w_closes` (ridge on `E_q[phi_closes]`), and **partial it out of `r`**
  (`r_⊥ = r − (r·ŵ_closes)ŵ_closes`) before the facet fits used for the angle/ΔR².
  Facet geometry is measured in `r_⊥`, so any residual `p_close` loading is gone
  in addition to the ratio normalization. The full-`r` geometry is reported as a
  robustness companion, the `r_⊥` geometry is the verdict.
- **kNN R²** (registered `k`) on the same `psi` target is the
  **present-but-not-linear** reference (exp-1/exp-29 interpreter gap): high kNN R²
  with low linear R² = present but not affine → richer-probe route, not "absent."
- **Decodable iff** held-out linear `R²_facet ≥ R2_MIN`. **The pooled-mean decode
  error `|psi_facet − decode|` is reported but does NOT gate** (implementation
  finding, flagged for the second review): because `psi` is near-binary on
  determined-ctx prefixes, a linear probe's pooled-mean error is ~0.15 even at
  `R²≈0.77` — the registered `TAU=0.03` conflated this with the *estimator* floor.
  Estimator soundness is the separate **OBS_DRIFT ≤ OE_BAND** check (calibration:
  drift 0.009, passes). The drift audit stays **pooled-mean only — never
  per-prefix** (calibration: per-prefix drift up to ~0.3; the verdict aggregates
  over prefixes, the OUTCOME_STRUCTURE replicate axis).

## Composition: two measured axes (in `r_⊥`, against calibrated floors)

Held-out split, gate-normalized targets, closes-orthogonal residual `r_⊥`:

1. **Marginal separability** — `angle(w_type0, w_color0)` in `r_⊥`, judged
   **relative to the per-seed independent-pair ceiling** (the GT `top.type` /
   `top.color` label decodes). `SEPARABLE` iff `observed ≥ GT_ceiling −
   ANGLE_MARGIN` (PROVISIONAL 15°), **guarded** by `GT_ceiling ≥ CEIL_MIN`
   (PROVISIONAL 60°); below the guard the geometry can't separate even
   known-independent factors and the cut is not well-posed → `ANGLE_UNRESOLVED`.
   *Why relative, not the original bracket nor a lone absolute cut.* The
   floor/ceiling-interpolation was **refuted by calibration**: the "entangled
   floor" (un-normalized facets in raw `r`) came out 82.7° — *above* the GT-pair
   ceiling 79.4° ≈ observed 79.4° — so the multiplicative gate does **not**
   deflate the angle and `floor + 0.5·(ceiling − floor)` inverts. A lone absolute
   cut is an arbitrary guess between 0 and the ceiling. The relative cut anchors
   on the per-seed ceiling instead, and the `CEIL_MIN` guard answers the
   "ceiling shrinks with entanglement" objection: if even the labels can't
   separate, don't adjudicate. **Caveat (finding 6 consequence):** on
   determined-ctx `psi ≈` the top's label, so `observed ≈ GT_ceiling` almost by
   construction and `SEPARABLE` is the *expected* read; the cut fires
   `ENTANGLED_FACTORS` only on genuine `psi`-vs-label readout divergence — a
   narrow but meaningful positive (the model entangles its own facet readouts
   even though the labels are separable). The raw-`r` entangled floor is reported
   as a diagnostic only. REVIEWER: confirm `ANGLE_MARGIN=15°`, `CEIL_MIN=60°`
   (exp-40/42 give no transferable angle floor — they measure drag /
   edit-transfer cosine; `cos 0.70 ≈ 45°` is a loose neighborhood check only).
2. **Conjunction availability** — fit `psi_both` (a) from the **2-D span**
   `[w_type0, w_color0]`, (b) from full `r_⊥` (linear), (c) by **kNN** on full
   `r_⊥`. `ΔR² = R²_full − R²_span`. For a true direct sum (only the two marginal
   axes present) the Boolean-AND ceiling caps `R²_full` and `R²_span` **equally**
   and cancels in the difference, so `ΔR² ≈ 0` — the clean-separable case is the
   *natural* direct-sum cell, not an unreachable one (checked on synthetic
   planted residuals: clean-separable `ΔR² ≈ 0`; a dedicated joint axis `ΔR² ≈
   0.32`). So `ΔR²` measures whether the joint sits **in** the marginal span
   (`≤ COMP_GAP`) or needs a direction **outside** it (`> COMP_GAP`); the
   outside-span case is then split into *linear* vs *nonlinear* by (b)/(c):
   - `R²_full ≥ R2_MIN` and `ΔR² > COMP_GAP` → the joint is on a **dedicated
     LINEAR direction** outside the span (`JOINT_OUTSIDE_SPAN`): linearly stored,
     just not a combination of the marginals — **not** a nonlinearity claim.
   - `R²_full < R2_MIN` but kNN `R²_both ≥ R2_MIN` → the joint is **not affinely
     decodable** but recoverable by a richer probe: genuinely higher-order
     (`NOT_LINEARLY_DECODED`, extended to the joint).

   `COMP_GAP` sits a registered margin above a `--calibrate`-measured **direct-sum
   noise floor**: the `ΔR²` of a true-direct-sum control conjunction at the **same
   positive count** (finite-sample `ΔR²` is slightly >0 from estimation noise even
   when the joint is in-span — confound row 2). This is the finite-sample floor,
   *not* an arithmetic-product floor: since type ⊥ color, the product of the
   recovered marginals reconstructs `psi_both` up to its decode-noise ceiling
   (≈ `R²_full`; realized 0.75–0.82 — *not* 0.99, see the premise-audit note)
   regardless of any out-of-span joint axis, so an excess-over-product floor would
   make the joint-present cell unreachable. The
   product is kept only as a descriptive premise-audit (below), never a threshold.
   The outside-span direction (residualized `w_both ⟂ span`) is recorded.

## Baselines and controls (anti-vacuity)

- **No-information floor:** `psi_facet` predicted by its train mean (`R²=0`); the
  decode must beat it.
- **Full-residual ceiling:** linear `R²` from the whole `r_⊥` — the best the
  probe class can do; `w_facet` is read here.
- **Degeneracy control:** `matches_type/color/both` must coincide (forced
  matching); if they *separate*, the instrument is suspect → `HARNESS_FAIL`.
- **Ground-truth facet directions (registered, eval-only — the separability
  ceiling):** belief-derived linear directions decoding the true `top.type` /
  `top.color` from `r` (Shai-style probe on ground-truth labels). Their pairwise
  angle is the **separability reference** for axis 1 (a known-independent pair in
  this residual); each is also compared to its `w_facet`. **Registered
  supervised-on-ground-truth reference (discipline note):** with the relative cut
  this GT-ceiling angle is now *in* the axis-1 verdict path — it sets the
  per-seed scale the observable `angle(w_type0, w_color0)` is judged against, and
  the `CEIL_MIN` guard reads it. It supplies the *reference*, never the decoded
  *target* (the readouts `w_facet` are fit on observable `psi` only), so the
  observable readouts stay observable; but unlike a pure diagnostic it gates the
  cell, which is why it is registered here as a supervised-on-ground-truth
  control rather than eval-only. (The raw-`r` entangled floor remains a pure
  diagnostic.)
- **Gate predicate `phi_closes`:** measured for every prefix (the normalizer and
  the `w_closes` partial-out direction); the `CLOSE_FLOOR` selection is reported.
  *Selection note (review finding 7):* `CLOSE_FLOOR` selects prefixes on
  `E_q[phi_closes]`, a **depth-correlated denominator**, so the pooled verdict may
  carry mild depth/facet selection bias; at the calibration mean ≈0.81 few
  prefixes drop, so the effect is expected small — but it is acknowledged, not
  assumed away.
- **Product-of-marginals premise-audit (descriptive print only, NO threshold):**
  the `R²` of `psî_type·psî_color` (the product of the **recovered, decoded**
  marginals `Rp·ŵ + b`) against `psi_both`. **Correction (result review):** the
  earlier "should sit at R²≈0.99" expectation was a conflation — the audit
  multiplies the *decoded* marginals (each only R²≈0.80 on `r_⊥`), not the true
  near-binary `psi`, so its ceiling is `≈ R²_full`, not 0.99. Realized 0.746–0.819
  (≈ `R²_full` 0.79–0.86) is therefore **exactly expected from decode-noise
  propagated through the product, and is NOT evidence of a type ⊥ color
  independence violation.** What *would* flag a violation is a value well *below*
  `R²_full` (the product of two clean marginals failing to reconstruct the joint).
  No numeric flag is implemented (and none is needed): it is a descriptive print,
  read against `R²_full`, not a gated threshold.
- **Prefix-free anchor `net_return`: REGISTERED BUT NOT IMPLEMENTED (gap caught at
  result review).** It is absent from `facet_masks()` / the script / the artifact.
  Not re-run: the claim seeds are burned and the headline does not depend on it.
  Its anti-vacuity role — "the method finds decode directions at all" — is covered
  by the no-information floor (train-mean `R²=0`, beaten), the facet decodes
  (R²≈0.80), and the GT-label decodes. The registration row is retained with this
  note rather than deleted, so the gap stays on the record.

## Verdict (per `(seed, layer-if-swept)`; prefixes/positions pooled; then ≥3/4 seeds)

```text
HARNESS_FAIL          — a guard fails: predicate-layer self-tests, the validity
                        gate, the OBS_DRIFT audit (|eq_obs − eq_exact| pooled >
                        OE_BAND), or the degeneracy control (matches_* separate).
                        Blocks all.
BASELINE_VACUOUS      — a facet psi has std < VAR_MIN across prefixes (no signal
                        to decode): PREDICATE_VACUOUS for that facet; report, do
                        not interpret its direction.
NOT_LINEARLY_DECODED  — a psi (type, color, OR the joint psi_both) has linear
                        R² < R2_MIN but kNN R² ≥ R2_MIN (on the GATE-NORMALIZED
                        target, so not a p_close artifact): present but not affine —
                        route to a richer probe class (the V-information ladder),
                        do not claim a direction. For the JOINT this is the
                        genuinely higher-order / nonlinear case. The kNN ≥ R2_MIN
                        conjunct is a GATE in `cell_verdict` (`present_split`), not
                        a printed diagnostic: without it an absent facet would be
                        mislabeled higher-order.
NOT_DECODED           — a psi has signal (std ≥ VAR_MIN) but BOTH linear and kNN
                        R² < R2_MIN: not recoverable from the residual at this
                        layer/probe at all — distinct from NOT_LINEARLY_DECODED
                        (present, nonlinear) and BASELINE_VACUOUS (the predicate
                        itself carries no signal). Routes to the per-layer profile,
                        not the probe ladder.
SEPARABLE_DIRECTSUM   — type & color decoded (R² ≥ R2_MIN), r_⊥ angle WITHIN
                        ANGLE_MARGIN of the GT ceiling (separable, relative cut;
                        ceiling ≥ CEIL_MIN), AND psi_both linearly decoded
                        (R²_full ≥ R2_MIN) and IN the marginal span (ΔR² ≤ COMP_GAP):
                        factors separable and the joint is a linear direct sum of
                        them — the clean-composition cell, measured.
ENTANGLED_FACTORS     — type & color decoded, ceiling ≥ CEIL_MIN, but r_⊥ angle
                        < GT_ceiling − ANGLE_MARGIN: the psi readout directions
                        overlap well below what the independent-pair geometry
                        permits — entangled factors (exp-40/42 redux on a new pair).
ANGLE_UNRESOLVED      — type & color decoded but GT_ceiling < CEIL_MIN: the residual
                        geometry can't separate even the ground-truth independent
                        labels, so the separability cut is not well-posed. Report;
                        route to the per-layer profile (try another layer), do not
                        force SEPARABLE/ENTANGLED.
JOINT_OUTSIDE_SPAN    — type & color decoded, psi_both linearly decoded from full
                        r_⊥ (R²_full ≥ R2_MIN) but OUTSIDE the marginal span
                        (ΔR² > COMP_GAP): the (type,color) joint occupies a
                        dedicated LINEAR direction not spanned by the marginals —
                        linearly stored, NOT a direct sum, and NOT a nonlinearity
                        claim (the genuinely nonlinear case routes to
                        NOT_LINEARLY_DECODED via the kNN gate). The central positive
                        for the composition cell.
SEED_UNSTABLE         — no ≥3/4 cross-seed majority on the headline cell.
```

### Routing (each outcome changes the next step)

| outcome | reading | routes to |
|---|---|---|
| `SEPARABLE_DIRECTSUM` | factors separable, compose linearly | the causal rung: do interchange edits on `w_color` move color without type? (the inverse-intervention goal, with the entanglement null) |
| `JOINT_OUTSIDE_SPAN` | the joint is on a dedicated **linear** axis outside the marginal span | characterize that direction; the joint is linearly available but not factor-composed — test whether it moves causally independent of the marginals next |
| `ENTANGLED_FACTORS` | type/color overlap in the residual | the entanglement is the subject (per the phase's entangled-regime deliverable); a clean color-only edit is not well-posed |
| `NOT_LINEARLY_DECODED` | present, not affine (for the joint: genuinely higher-order) | climb the V-information probe ladder before any direction claim |
| `NOT_DECODED` | psi has signal but neither linear nor kNN recovers it from this residual | per-layer profile / another layer; the facet isn't represented here at this probe class |
| `ANGLE_UNRESOLVED` | even the GT independent labels don't decode separably (ceiling < CEIL_MIN) | per-layer profile; the separability question isn't well-posed at this layer — don't adjudicate the angle |
| `BASELINE_VACUOUS` | predicate adds no decodable signal over its mean | enrich the suite/toy (the leash); the layer is validated but not earning its keep |

## Registered prediction (walled off from adjudication; credences never enter a predicate)

Frozen at this pre-run commit, informed by the **burned calibration seeds
(777, 800)** (their results read in full → excluded from the claim seeds). The
**test** is whether the headline cell replicates on the fresh out-of-design claim
seeds at ≥3/4. Calib s800 (burned, one seed) read `JOINT_OUTSIDE_SPAN` cleanly
(`ΔR² = 0.225` vs a ~0.002 floor; marginals separable at the ceiling, angle 79.4°),
which shifts mass onto that cell — see the note below.

| configuration | credence | what it would teach |
|---|---|---|
| `JOINT_OUTSIDE_SPAN` | ~0.60 | the joint occupies a dedicated **linear** direction outside the marginal span — the factors are decodable but the conjunction isn't their linear combination; the composition cell's central positive, **as a linear-storage claim** (the genuinely nonlinear sub-case is split off into `NOT_LINEARLY_DECODED` by the kNN gate, and the p_close artifact the first review flagged is removed by gate-normalization). Bumped from ~0.35: it is also the *generic* outcome for a 4-class (type,color) code that isn't a perfect 2-D grid — the marginals give 2 linear directions, separating the AND class generally needs a 3rd — and calib s800 confirmed it |
| `SEPARABLE_DIRECTSUM` | ~0.20 | the residual carries (type,color) as a clean separable, linearly-composing pair (joint in the marginal span) — routes straight to the causal inverse-intervention rung. Lowered from ~0.30: it requires the special 2-D-grid encoding, which calib did **not** show |
| `ENTANGLED_FACTORS` | ~0.10 | type/color overlap (exp-40/42 generalizes to a fresh factor pair) — entanglement is the subject. Lowered from ~0.20: calib angle sat *at* the independent-pair ceiling (no readout divergence) |
| `NOT_LINEARLY_DECODED` / `NOT_DECODED` / `BASELINE_VACUOUS` / `ANGLE_UNRESOLVED` (a marginal, the joint, or the angle axis) | ~0.10 | a psi isn't affinely decodable (joint: genuinely higher-order), isn't recoverable at all, is vacuous, or the geometry can't express separability — climb the probe ladder, try another layer, or enrich |

**Worth-running judgment:** yes — this is the first measurement relating residual
abstractions to *named* predicates with *composition*, on a 2-factor toy built
for it. Every substantive outcome routes differently (causal rung vs outside-span-direction
characterization vs entanglement subject vs probe-ladder), and the compression
reference (`k*` vs rank-1 readouts) plus the composition geometry are the
phase's central positive object regardless of which cell fires.

## Confound table — load-bearing quantities (decodability R²; angle(w_type,w_color); ΔR²)

| mechanism producing the reading | excluded by? |
|---|---|
| **multiplicative closing gate** `p_close(x)` — confounds R² (product non-linear), angle (both load on `p_close`), and ΔR² (triple product) | **EXCLUDED by gate-normalization** (`psi = E_q[phi_facet]/E_q[phi_closes]` divides `p_close` out by construction) **AND** the closes-orthogonal geometry (`w_closes` partialled out of `r` before the angle/ΔR² fits). Both are registered computations, not prose. Residual robustness: full-`r` geometry reported beside `r_⊥` |
| **probe overfit** — `w_facet` fits noise, R² spurious | held-out split (fit TRAIN, score HELD-OUT, split at the SEQUENCE level — finding 4); ridge `λ` registered; no-information floor must be beaten on held-out |
| **ΔR²-from-noise** — `ΔR²` > 0 because `psi_both` is just noisier (fewer joint positives), or because `w_type`/`w_color` are estimated and non-orthogonal on a non-gaussian residual — not from a real out-of-span direction | the `--calibrate` **data-matched direct-sum noise floor** (`data_directsum_dr2`): `ΔR²` of an in-span true-direct-sum joint built on the **real `r_⊥`** with the **estimated** marginal directions, so it sees the actual residual covariance; `COMP_GAP = max(planted, data) μ + 3σ`. The planted-gaussian floor alone under-estimates this and is kept only as the optimistic reference. `ΔR²` is full-vs-span at matched samples |
| **dedicated-linear-axis vs genuine-nonlinearity** — `ΔR² > COMP_GAP` could be a joint on a dedicated *linear* direction (still linearly decodable) OR a genuinely *nonlinear* joint (linear probe fails); the old `INTERACTION_PRESENT`/"higher-order" label conflated them | **the kNN-on-`psi_both` gate**: `R²_full ≥ R2_MIN` with large `ΔR²` → `JOINT_OUTSIDE_SPAN` (linear); `R²_full < R2_MIN` but kNN `R²_both ≥ R2_MIN` → `NOT_LINEARLY_DECODED` (nonlinear). The verdict never calls a linearly-decodable joint "higher-order." (Note: no AND-arithmetic confound — the Boolean-AND ceiling caps `R²_full` and `R²_span` equally and cancels in `ΔR²`) |
| **entanglement-as-artifact** — small angle from a degenerate or undertrained model, not real factor coupling | validity gate (converged model) + the degeneracy control (matches_* coincide) + ground-truth control (do the true facet directions ALSO overlap? if the labels are separable but the predicate readouts aren't, that's a probe issue, not entanglement) |
| **off-manifold / wrong layer** — the chosen layer doesn't represent the facets | per-layer profile (descriptive) + decodability gate (if no layer decodes a facet, that is itself reported, not forced) |
| **estimator drift** — `E_q[phi]` mis-measured | `OBS_DRIFT` audit vs `eq_exact` (calibration: corr 0.98–0.999, plumbing validated); pooled-mean tolerance above the measured floor |

## Self-tests / controls (known-answer, before any model claim)

- Predicate-layer self-tests (`predicates._selftest`): graded masks, exact vs
  observable estimators, the seeded ctx-reader + color-scramble discrimination,
  verdict branches — all passing.
- **Degeneracy control** computed first: `matches_*` coincide on the model
  (forced matching); separation → `HARNESS_FAIL` (suite/instrument bug).
- **No-information floor & full-residual ceiling** measured for every predicate;
  the decodable threshold is read against both (the protocol's ceiling/floor
  requirement for the load-bearing R²).
- **OBS_DRIFT** audit (`eq_obs` vs `eq_exact`) on the claim sample, pooled-mean ≤
  `OE_BAND`; `--calibrate` **re-confirms the noise floor on the gate-normalized
  facet suite** (the 777 floor was on the degenerate matches suite) and the script
  **asserts seeds 777 and 800 ∉ claim seeds** (both burned — see the seed note).
- **`--calibrate` emits the composition references + the COMP_GAP floor.** TWO
  direct-sum noise floors are reported. The **planted** floor (`ΔR²` for a
  true-direct-sum control on clean isotropic-gaussian orthogonal axes, matched to
  `n`, `d`, marginal rates, decodability) is the optimistic reference — the
  ≈0.002 the first calibration smoke gave came from it. The **data-matched** floor
  (`data_directsum_dr2`: an in-span joint built on the REAL `r_⊥` with the
  ESTIMATED, non-orthogonal `w_type`/`w_color`, so the finite-sample `ΔR²`
  reflects the actual residual covariance, not a gaussian idealization) is the one
  that governs: `COMP_GAP = max(planted, data) μ + k·σ`, **k = 3 frozen**. Calib
  s800: planted μ −0.0049/σ 0.0022, data-matched μ −0.0044/σ 0.0013 → `COMP_GAP =
  0.0018` (the data floor came out *tighter*, both ≈0, so on this residual an
  in-span joint genuinely gives `ΔR² ≈ 0` and the observed `ΔR² = 0.225` is an
  unambiguous out-of-span direction). Separability is a **relative** cut (the
  absolute/bracket formula was refuted — see axis 1); the entangled floor and GT
  ceiling are reported (the GT ceiling is the cut's reference, registered as a
  supervised-on-ground-truth control — see the controls section).
- Ridge / kNN / angle / ΔR² reducers unit-tested on synthetic planted residuals
  (a known separable pair, joint in the span → `SEPARABLE_DIRECTSUM`, `ΔR² ≈ 0`; a
  planted **dedicated linear joint axis** → `JOINT_OUTSIDE_SPAN`, `ΔR² ≈ 0.32`; a
  planted **nonlinear-only** joint, decodable by kNN but not linearly →
  `NOT_LINEARLY_DECODED`); the gate-normalization unit-tested to recover a planted
  facet bit from a `p_close·bit` target (the gate-confound exclusion has a
  known-answer test).

## Registered constants (to finalize at the freeze; calibration-derived where marked)

| knob | value | note |
|---|---|---|
| checkpoint | fresh `out/colored_dyck2-sNNN` per claim seed | `train.py --process colored_dyck2`; config below; **seed 777 (calib) is burned**, not a claim seed |
| train config | L4 d64 seq_len32 m3, steps 8000 | matches the calibration checkpoint; validity gate ≤ 0.005 |
| claim seeds | `{801, 802, 803, 804}` | 4 fresh out-of-design; ≥3/4 majority; calibration seed `800` burned |
| `PROBE_LAYER` | calibration-selected (default block 2 of 4) | per-layer profile descriptive; headline at the registered layer |
| read points `t` | `{8, 12, 16}` | determined-ctx prefixes pooled (calibration: ~180–200 determined per t per 250 seqs) |
| horizon `m` | `3` | **deviation from decision 6** (m=3 "too short"); justified by calibration — tops close fast, matches-mean 0.811 → non-vacuous. `--calibrate` must report per-`psi` std at m=3 vs `VAR_MIN`; 0.811 is also the finding-1 gate magnitude |
| predicates | `phi_type0`, `phi_color0`, `phi_both00`, **gate `phi_closes`**; degeneracy `matches_*`; ~~control `net_return`~~ (registered, **not implemented** — result-review gap; anti-vacuity carried by the no-info floor + facet/GT decodes) | facet-value suite, gate-normalized (NOT matches — degenerate) |
| `CLOSE_FLOOR` | `0.30` | keep prefixes with `E_q[phi_closes] ≥` this (ratio stability); calibration: mean ≈0.81, so most qualify |
| `TAU` (decode error) | `0.03` → **reported only, not a gate** | calibration: near-binary `psi` gives pooled-mean error ~0.175 even at `R²≈0.77`; `TAU` conflated this with the estimator floor. Decodability = `R²≥R2_MIN`; estimator soundness = `OBS_DRIFT≤OE_BAND` (drift 0.009). **Second-review item** |
| `R2_MIN` | `0.50` | linear-decodable cut (exp-29 precedent); kNN `R²` ≥ this = "present" |
| `ANGLE_MARGIN` / `CEIL_MIN` | **`15°` / `60°`** (calib-confirmed) | RELATIVE separability cut: `SEPARABLE` iff `observed ≥ GT_ceiling − ANGLE_MARGIN`, guarded by `GT_ceiling ≥ CEIL_MIN` (else `ANGLE_UNRESOLVED`). Replaces the refuted absolute/bracket cut; on determined-ctx `observed ≈ ceiling` so `SEPARABLE` is expected and `ENTANGLED` is the narrow psi-vs-label-divergence positive. Calib s800: ceiling 79.4° clears `CEIL_MIN` by 19°, `ceiling − observed = 0.01°`. |
| `COMP_GAP` (ΔR²) | **`0.0018`** (FROZEN, calib s800) | `= max(planted, data-matched) μ + 3σ` of the direct-sum **noise** floor; **k = 3 frozen**. Calib: planted μ −0.0049/σ 0.0022, data-matched μ −0.0044/σ 0.0013 → max μ+3σ = 0.0018. Observed `ΔR² = 0.225` is ~125× this, so the freeze is not headline-load-bearing; a claim-seed `ΔR²` in ~0.01–0.05 would warrant a per-seed floor recheck |
| `VAR_MIN` | `0.05` | predicate non-vacuity (std of `psi` across prefixes) |
| `OE_BAND` | `0.02` | `OBS_DRIFT` audit, pooled-mean |
| ridge `λ` / kNN `k` | `1e-2` / `10` | the bounded probe class |
| eval sequences / split | `≥ 3000` seqs, split **by sequence** 50/50 | yields ≥ ~3500 determined-ctx prefixes per split over `{8,12,16}` (reconciles finding 4: the ~600 figure was per-250-seqs); disjoint at the sequence level (no prefix leakage); `d=64` so each split ≫ 20·d |
| `SEED_MAJORITY` | `3` of 4 | cross-seed headline |

**Burned seeds (review finding 4 — neither is a claim seed; both excluded from
{801–804}, the script asserts this):**

- **777** — the on-disk calibration **checkpoint** (`out/colored_dyck2-calib`):
  plumbing (corr, per-prefix drift, determined-fraction) and the **degeneracy
  redirection** (`matches_type` = `matches_color` = `matches_both` = 0.811
  identical → switch to the facet-value suite). Logged burned in `docs/SCOUTS.md`.
- **800** — the `--calibrate` **threshold** seed (entangled-angle floor +
  separable-angle ceiling + direct-sum noise floor, on the facet-value suite, like
  exp-43).

(Thresholds marked "finalize from calibration" are set by the script's
`--calibrate` on burned seed 800; both 777 and 800 are excluded from the claim
seeds.)

## Reuse vs single-use

- **Import:** `predicates.py` (graded layer, ctx-reader, `eq_obs_from_model`,
  `ctx_along`, `eq_exact_seeded`), `processes.colored_dyck2`, `train.py`,
  `expcommon.load_model`/`validity_gate`, `midstream.chain_probs`,
  `discover` (`cegar_loop`/`mined_direction` for the `k*` reference;
  `principal_angles_deg`), `abstraction.PCAAbstraction`/`center_by_position`. The
  predicate-targeted readout follows exp-29's `predicate_targeting.py` approach.
- **Rung-specific** (`scripts/predicate_sufficiency/exp45_*.py`): the facet-value
  template `tmpl_next_close_is(facet, value)` and the gate `phi_closes`
  (seeded locate; `value=None` = "pops") — promote to `predicates.py` only if a
  second rung needs them; **gate-normalization** (`psi`) + `CLOSE_FLOOR`
  selection + the `w_closes` partial-out; the ridge/kNN decode + sequence-level
  held-out scoring; the angle/ΔR² composition reducers, the **kNN-on-`psi_both`
  gate** (linear-joint vs nonlinear-joint split), and their calibrated floors and
  registered placement formulas; `cell_verdict`; `--calibrate`.

## Non-goals

- **No causal claim.** Decodability is correlational (the info is THERE in the
  probe class). Whether an interchange edit on `w_color` *moves* color alone —
  with the entanglement null (decision 3) — is the **next** rung, explicitly
  deferred. `SEPARABLE_DIRECTSUM`/`JOINT_OUTSIDE_SPAN` are decode-geometry
  verdicts, not "I moved color alone."
- **Probe-class indexed.** All verdicts are relative to affine-ridge (+ kNN as
  the present reference). `NOT_LINEARLY_DECODED` is bounded to "not affine," never
  "not represented"; a higher probe class is the V-information ladder, not a
  refutation.
- **Toy-indexed.** Colored Dyck-2, the registered config/layer/positions/horizon.
  No transfer claim until the toy ladder runs it (the generalization lever).
- **No privileged decomposition / no inherited-top prefixes** (latent-observability
  is the latent-toy question). Ground-truth facet directions supply the
  *decode targets* nowhere — the readouts are fit on observable `psi` only — but
  with the relative axis-1 cut their pairwise angle is a **registered
  supervised-on-ground-truth reference** that sets the separability scale and the
  `CEIL_MIN` guard (see the controls section). The verdict's decoded content
  stays observable; the *reference scale* is ground-truth, registered as such.
```
