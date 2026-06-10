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
                         completeness_kl, knn_kl, sym_kl_rows)


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

    k, verdict = 1, None
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
        if worst > args.tol and k < args.kmax:
            j = int(np.argmax(div))
            say(f"      COUNTEREXAMPLE: two prefixes alpha_{k} conflates have "
                f"sym-KL {div.max():.4f} between their true completion "
                f"distributions (true beliefs {np.round(B[sub][qi[j]], 3)} vs "
                f"{np.round(B[sub][nn[j]], 3)}).")
            say(f"      -> abstraction too coarse: REFINE  k := {k + 1}")
            k += 1
            continue

        # FAILURE MODE 2 — probe-class incompleteness (the V-information gap):
        # the head can't reach tolerance, yet mining found nothing alpha_k
        # conflates. Disambiguate with a nonparametric decode: if k-NN succeeds,
        # the information IS in alpha_k(resid) and the affine-softmax class V
        # is what's too weak (e.g. Z1R at k=1: three clusters on a line are
        # injective but not affinely decodable to their completion measures).
        if kl > args.tol:
            kl_nn = knn_kl(pca(R[tr], k), G[tr], pca(R[te], k), G[te],
                           seed=args.seed)
            say(f"      KL above tol but no conflated-pair counterexamples; "
                f"k-NN decode KL = {kl_nn:.5f}")
            if kl_nn <= args.tol:
                say("      -> abstraction is SUFFICIENT; the probe class V "
                    "(affine-softmax) cannot read it: interpreter "
                    "incompleteness, not domain coarseness (V-information).")
                if k < args.kmax and (kl2 := kl_at(k + 1)) <= args.tol:
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
        # Coarsen ONLY if the dropped dimension changes completeness
        # essentially not at all (strict relative test). A looser "still under
        # tol" test would be inconsistent with the refine trigger and oscillate
        # — it would re-admit the very counterexamples that forced k upward.
        if k > 1:
            kl_drop = kl_at(k - 1)
            if kl_drop <= kl + 0.1 * max(kl, 1e-4) + 1e-4:
                say(f"      dim {k} is junk precision (KL {kl_drop:.5f} "
                    f"without it) -> COARSEN  k := {k - 1}")
                k -= 1
        verdict = (f"FIXED POINT at k = {k} — the empirical complete shell at "
                   "this probe capacity and tolerance.")
        say("      no counterexamples above tol, no junk precision:")
        say(f"      {verdict}")
        break

    # ----- identify the discovered abstraction with the known one -----------
    Wb, b0, _ = affine_lstsq(pca(R[tr], k), B[tr])
    Bhat = pca(R[te], k) @ Wb + b0
    r2 = 1.0 - ((B[te] - Bhat) ** 2).sum() / ((B[te] - B[te].mean(0)) ** 2).sum()
    say("")
    say(f"Identification: affine map from the discovered {k}-D abstraction to "
        f"the true belief state has held-out R^2 = {r2:.4f}.")
    if r2 > 0.95:
        say("Reading: the outer loop, using only residuals and completion")
        say("divergences (never the hidden states), converged to an affine")
        say("image of the belief simplex — the known minimal sufficient")
        say("statistic. Discovery matches theory.")
    else:
        say("Reading: the fixed point is completion-sufficient at this")
        say("tolerance/horizon, yet is NOT an affine image of the belief")
        say("simplex. This is informative, not a failure: sufficiency only")
        say("requires an INJECTIVE reparametrization of the reachable belief")
        say("set, which can be lower-dimensional and nonlinear. (Z1R is the")
        say("canonical case: after synchronization its reachable beliefs are")
        say("three discrete points, injectively embeddable in 1-D even though")
        say("the ambient simplex is 2-D.) Moral from the conversation:")
        say("completeness is always relative to tolerance, probe class, and")
        say("completion horizon m — tighten any of these and the loop may")
        say("refine further.")

    with open(os.path.join(args.outdir, "refine_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\nwrote {os.path.join(args.outdir, 'refine_log.txt')}")


if __name__ == "__main__":
    main()
