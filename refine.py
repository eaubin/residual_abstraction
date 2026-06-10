"""
refine.py — the outer loop: discovery, rejection, refinement of abstractions.

CONTEXT (see README.md). analysis.py *verifies* abstractions; this script
*discovers* one without being told the answer, by the loop the conversation
called the empirical analogue of the complete shell (Giacobazzi, Ranzato &
Scozzari, JACM 2000: the coarsest refinement of an abstract domain complete
for a given semantics), run CEGAR-style:

  PROPOSE   alpha_k = rank-k projection of the residual stream (start k = 1).
  TEST      quantified completeness: held-out mean KL between true completion
            distributions and those predicted from alpha_k(resid).
  MINE      incompleteness counterexamples: pairs of prefixes that alpha_k
            CONFLATES (nearest neighbors in abstract space) whose true
            completion measures nonetheless DIVERGE (large symmetric KL).
            These witnesses are the stochastic analogue of the concrete pairs
            a Galois-connection completeness proof would exhibit.
  REFINE    if such witnesses exist above tolerance -> k := k + 1.
  COARSEN   junk-precision check: if dropping the last retained dimension
            leaves completeness unchanged, the abstraction distinguishes
            prefixes the completion measure does not -> k := k - 1, stop.
  FIXED PT  neither move fires: the empirical complete shell at this probe
            capacity and tolerance.

On these processes the loop should stop at k = 2 (the belief simplex is 2-D),
and the recovered 2-D coordinates should be an affine image of the true
belief state — verified at the end by regression. That final check is the
'discovered abstraction == known sufficient statistic' punchline.

Honest scope notes baked into the design:
  * everything is relative to the prefix distribution sampled here; a
    property that rarely moves completions on-distribution produces no
    counterexamples and stays invisible (the safety-relevant caveat);
  * tests are correlational sufficiency tests; the interventional upgrade
    (interchange interventions, Geiger et al.'s causal abstraction) is the
    natural v2 and is intentionally out of scope.
"""

import argparse
import os

import numpy as np

from abstraction import (PCAAbstraction, affine_lstsq, center_by_position,
                         completeness_kl, knn_kl, knn_predict, mean_kl,
                         r2_score, sym_kl_rows)


def conflated_pairs(Z, n_query, rng):
    """For sampled query points, find their nearest neighbor in abstract
    space: pairs the abstraction treats as (nearly) the same prefix-state."""
    idx_q = rng.choice(len(Z), size=min(n_query, len(Z)), replace=False)
    d2 = ((Z[idx_q, None, :] - Z[None, :, :]) ** 2).sum(-1)
    d2[np.arange(len(idx_q)), idx_q] = np.inf      # exclude self
    return idx_q, d2.argmin(axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/z1r")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="nats of completion divergence we agree to ignore")
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--n-query", type=int, default=400)
    ap.add_argument("--subsample", type=int, default=4000,
                    help="points used for nearest-neighbor mining")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d = np.load(os.path.join(args.outdir, "cache.npz"))
    R, B, G = d["resid"], d["belief"], d["mgram"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(R))
    R, B, G = R[perm], B[perm], G[perm]
    n_tr = int(0.7 * len(R))
    tr, te = slice(None, n_tr), slice(n_tr, None)

    # Deconfound positional-embedding variance before PCA (see
    # abstraction.center_by_position). Old caches lack `pos`; degrade gracefully.
    if "pos" in d:
        train_mask = np.zeros(len(R), dtype=bool); train_mask[:n_tr] = True
        R = center_by_position(R, d["pos"][perm], train_mask)
        print("(residuals centered per position before PCA)")
    pca = PCAAbstraction(R[tr])

    sub = rng.choice(n_tr, size=min(args.subsample, n_tr), replace=False)
    log = []

    def say(s):
        print(s); log.append(s)

    say(f"=== refinement loop: {str(d['process'])} | tol = {args.tol} nats ===")

    def kl_at(k):
        return completeness_kl(pca(R[tr], k), G[tr], pca(R[te], k), G[te],
                               seed=args.seed)[0]

    # Tolerance calibration. args.tol is one number doing two jobs on
    # different scales: (a) pairwise sym-KL between conflated prefixes'
    # completion measures (the counterexample trigger) and (b) the decoder's
    # held-out MEAN KL. (b) must be judged against the unconditional baseline
    # KL0 — the marginal predictor that ignores the prefix entirely — or
    # 'meets tolerance' can be vacuously true for processes whose completions
    # vary mildly at this horizon. We therefore use tol for (a) and
    # min(tol, KL0/2) for (b): sufficiency must at least halve what there is
    # to know. (Discovered the hard way: a junk 1-D abstraction 'passed' a
    # mean-KL tolerance that the no-information baseline also passed.)
    KL0 = mean_kl(G[te], np.tile(G[tr].mean(axis=0), (len(G[te]), 1)))
    tol_kl = min(args.tol, 0.5 * KL0)
    say(f"unconditional baseline KL0 = {KL0:.5f}  ->  mean-KL tolerance "
        f"{tol_kl:.5f} (pair-divergence tolerance stays {args.tol})")
    if args.tol >= KL0:
        say("NOTE: requested tol exceeds the no-information baseline; the "
            "mean-KL criterion was auto-tightened to stay meaningful.")

    k, verdict, worst_hist = 1, None, []
    while True:
        # PROPOSE + TEST (probe class V: affine-softmax head)
        kl = kl_at(k)
        # MINE incompleteness witnesses among abstraction-conflated pairs
        Zs = pca(R[sub], k)
        qi, nn = conflated_pairs(Zs, args.n_query, rng)
        div = sym_kl_rows(G[sub][qi], G[sub][nn])
        worst = float(np.quantile(div, 0.95))
        say(f"[k={k}] held-out KL (affine-softmax) = {kl:.5f} | conflated-pair "
            f"completion divergence: median {np.median(div):.5f}, "
            f"q95 {worst:.5f}")

        # FAILURE MODE 1 — domain coarseness: alpha_k conflates prefixes whose
        # true completion measures diverge. The classical refinement trigger.
        # GUARD — metric junk-domination: if decode KL is already under tol
        # yet counterexamples persist undiminished across refinements, the
        # conflation lives in the abstraction's METRIC (nearest neighbors are
        # determined by dominant completion-irrelevant directions), and adding
        # PCA dimensions can never remove it. Refining forever would be the
        # loop chasing a defect of the proposal family, not of capacity.
        worst_hist.append(worst)
        stagnant = (kl <= tol_kl and len(worst_hist) >= 3
                    and worst > 0.9 * worst_hist[-3])
        if worst > args.tol and k < args.kmax and not stagnant:
            j = int(np.argmax(div))
            say(f"      COUNTEREXAMPLE: two prefixes alpha_{k} conflates have "
                f"sym-KL {div.max():.4f} between their true completion "
                f"distributions (true beliefs {np.round(B[sub][qi[j]], 3)} vs "
                f"{np.round(B[sub][nn[j]], 3)}).")
            say(f"      -> abstraction too coarse: REFINE  k := {k + 1}")
            k += 1
            continue
        if worst > args.tol and stagnant:
            while k > 1 and kl_at(k - 1) <= tol_kl:    # minimal sufficient k
                k -= 1
            verdict = (f"FIXED POINT (with metric caveat) at k = {k}: decoding "
                       "from the abstraction meets tolerance, but its metric "
                       "conflates behaviorally distinct prefixes because "
                       "dominant directions are completion-irrelevant — a "
                       "defect the PCA-prefix proposal family cannot excise. "
                       "Next iteration: supervised subspace proposals.")
            say(f"      counterexamples persist undiminished under refinement "
                f"while decode KL <= tol:")
            say(f"      {verdict}")
            break

        # FAILURE MODE 2 — probe-class incompleteness (the V-information gap):
        # the head can't reach tolerance, yet mining found nothing alpha_k
        # conflates. Disambiguate with a nonparametric decode: if k-NN succeeds,
        # the information IS in alpha_k(resid) and the affine-softmax class V
        # is what's too weak (e.g. Z1R at k=1: three clusters on a line are
        # injective but not affinely decodable to their completion measures).
        if kl > tol_kl:
            kl_nn = knn_kl(pca(R[tr], k), G[tr], pca(R[te], k), G[te],
                           seed=args.seed)
            say(f"      KL above tol but no conflated-pair counterexamples; "
                f"k-NN decode KL = {kl_nn:.5f}")
            if kl_nn <= tol_kl:
                say("      -> abstraction is SUFFICIENT; the probe class V "
                    "(affine-softmax) cannot read it: interpreter "
                    "incompleteness, not domain coarseness (V-information).")
                if k < args.kmax and (kl2 := kl_at(k + 1)) <= tol_kl:
                    say(f"      -> one extra dimension linearizes the decode "
                        f"(KL {kl2:.5f}): REFINE  k := {k + 1}")
                    k += 1
                    continue
                verdict = ("FIXED POINT (with V-class caveat) at k = "
                           f"{k}: sufficient for completions, but only a "
                           "richer interpreter than affine-softmax can decode "
                           "it. Enrich V or accept the nonparametric decode.")
            elif k < args.kmax:
                say("      -> insufficiency is diffuse (not visible to local "
                    f"pair mining): REFINE  k := {k + 1}")
                k += 1
                continue
            else:
                verdict = (f"stopped at kmax = {k} with KL {kl:.5f} > tol: "
                           "unresolved incompleteness at this capacity.")
            say(f"      {verdict}")
            break

        # SUCCESS — junk-precision check before declaring the fixed point.
        # Coarsen ITERATIVELY: drop trailing dimensions while completeness is
        # essentially unchanged (strict relative test). A looser "still under
        # tol" test would be inconsistent with the refine trigger and oscillate
        # — it would re-admit the very counterexamples that forced k upward.
        # Note the structural limit of the PCA-prefix proposal family: only
        # TRAILING junk can be excised; dominant-variance irrelevant directions
        # ahead of the relevant ones are not removable (see the identification
        # verdicts below for how that case is detected and what to do).
        while k > 1:
            kl_drop = kl_at(k - 1)
            if kl_drop <= kl + 0.1 * max(kl, 1e-4) + 1e-4:
                say(f"      dim {k} is junk precision (KL {kl_drop:.5f} "
                    f"without it) -> COARSEN  k := {k - 1}")
                k -= 1
                kl = kl_drop
            else:
                break
        verdict = (f"FIXED POINT at k = {k} — the empirical complete shell at "
                   "this probe capacity and tolerance.")
        say("      no counterexamples above tol, no junk precision:")
        say(f"      {verdict}")
        break

    # ----- identify the discovered abstraction with the known one -----------
    # Three numbers separate three very different explanations of an imperfect
    # identification (computed held-out; probes fit on the train split):
    #   r2_full : affine probe FULL residual -> belief.   Low => the residual
    #             itself doesn't (linearly) carry the beliefs: undertrained.
    #   r2_aff  : affine probe alpha_k(resid) -> belief.  The headline number.
    #   r2_nn   : k-NN decode alpha_k(resid) -> belief.   High while r2_aff is
    #             low => the abstraction embeds beliefs INJECTIVELY but
    #             nonlinearly (curved manifold / discrete clusters).
    # If r2_full is high but both abstraction numbers are low, the belief
    # geometry exists linearly in the residual yet not in the top-k principal
    # subspace: variance != relevance, and the PCA *proposal family* is the
    # bottleneck (next iteration: supervised subspace proposals, e.g. PLS or
    # the row space of the full-residual belief probe).
    Wb, b0, _ = affine_lstsq(R[tr], B[tr])
    r2_full = r2_score(B[te], R[te] @ Wb + b0)
    Ztr, Zte = pca(R[tr], k), pca(R[te], k)
    Wb, b0, _ = affine_lstsq(Ztr, B[tr])
    r2_aff = r2_score(B[te], Zte @ Wb + b0)
    r2_nn = r2_score(B[te], knn_predict(Ztr, B[tr], Zte, seed=args.seed))
    say("")
    say(f"Identification of the discovered {k}-D abstraction with the true "
        f"belief state (held-out R^2):")
    say(f"  affine, full residual -> belief : {r2_full:.4f}")
    say(f"  affine, abstraction  -> belief  : {r2_aff:.4f}")
    say(f"  k-NN,   abstraction  -> belief  : {r2_nn:.4f}")
    if r2_aff > 0.95 and r2_nn > 0.9:
        say("Reading: the outer loop, using only residuals and completion")
        say("divergences (never the hidden states), converged to an AFFINE")
        say("image of the belief simplex — the known minimal sufficient")
        say("statistic. Discovery matches theory in the strongest form.")
    elif r2_aff > 0.95:
        say("Reading: the abstraction CONTAINS an affine image of the belief")
        say("simplex, but its geometry is dominated by completion-irrelevant")
        say("directions (k-NN, which trusts the metric, fails while the affine")
        say("probe, which can ignore coordinates, succeeds). This is junk")
        say("precision the PCA-prefix proposal family cannot excise when the")
        say("irrelevant directions carry more variance than the relevant ones.")
        say("Next iteration: supervised subspace proposals (e.g. PLS, or the")
        say("row space of the full-residual probe) in place of PCA.")
    elif r2_nn > 0.95:
        say("Reading: the abstraction embeds the reachable belief set")
        say("INJECTIVELY but not affinely — a curved manifold or discrete")
        say("clusters (Z1R's three synchronized states are the extreme case).")
        say("Sufficiency holds; only the affine identification fails. This is")
        say("a fact about the model's representational geometry, not an error.")
    elif r2_full > 0.95:
        say("Reading: the belief geometry IS linearly present in the full")
        say("residual but NOT inside the top-k principal subspace: variance is")
        say("not relevance, and the PCA proposal family is the bottleneck.")
        say("Next iteration: supervised subspace proposals (e.g. PLS, or the")
        say("row space of the full-residual probe) in place of PCA.")
    else:
        say("Reading: even the full residual does not linearly encode the")
        say("belief state well. Most likely the model is undertrained for this")
        say("geometry — check the gap-to-optimal NLL from train.py and train")
        say("longer — though a strongly nonlinear residual code is possible.")
    say("Moral from the conversation either way: completeness is relative to")
    say("tolerance, probe class, and completion horizon m; tighten any of")
    say("these and the loop may refine further.")

    with open(os.path.join(args.outdir, "refine_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\nwrote {os.path.join(args.outdir, 'refine_log.txt')}")


if __name__ == "__main__":
    main()
