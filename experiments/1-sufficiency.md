# Experiment 1 — sufficiency curves and the CEGAR loop

**Scripts:** `analysis.py` (calibration + sufficiency curve), `refine.py`
(CEGAR loop). **Status: concluded**, superseded in interpretation by
[Experiment 2](2-proposal-families.md).

**Question.** Does quantified completeness (held-out KL from a rank-k
abstraction of the residual stream to the exact m=3 completion distribution)
saturate at the dimension of the known minimal sufficient statistic, and can
a counterexample-guided loop discover that rank without being told?

## Findings

- Calibration replicated Shai et al.: affine residual→belief R² = 0.9917
  (Mess3), 0.9842 (Z1R); the Mess3 fractal is visible under the probe
  (`out/mess3/belief_regression.png`).
- Z1R: fixed point at k = 1 — *correct* for the realized semantics: the
  reachable synchronized belief set is three points, embeddable injectively
  in 1-D (k-NN identification R² 0.9998 vs affine 0.4769). The complete
  shell is of the realized semantics, not the ambient simplex.
- Mess3: fixed point at k = 2 under the loop's tolerance, but identification
  was the open wound: affine abstraction→belief R² only 0.8104 inside the
  top-2 principal subspace vs 0.9917 from the full residual. Verdict cell:
  belief geometry linearly present in the residual, missing from the top-k
  PCA subspace — *proposal misalignment* ("variance is not relevance"), with
  the alternative hypothesis (genuinely curved embedding) left undecided.
- The PCA sufficiency curve's strict elbow sits at k = 8 on Mess3 — by the
  Experiment-2 stopping rule PCA needs k = 8 to match the belief-oracle
  floor. (analysis.py's verdict text now branches on k* instead of
  declaring victory unconditionally; the old unconditional "matches the
  known minimal sufficient statistic" message was stale after Experiment 2.)

## Lessons that fed forward

Two failure modes disambiguated and built into refine.py: *domain
coarseness* (conflated pairs with divergent completions) vs *interpreter
incompleteness* (k-NN decodes where the affine head cannot — the
V-information gap). Plus the positional confound (per-position centering),
tolerance calibration against the no-information baseline KL0, and the
junk-domination guard. See README.md "Revision note" for the full list.
