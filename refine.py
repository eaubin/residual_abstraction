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

from abstraction import (PCAAbstraction, affine_lstsq, completeness_kl,
                         sym_kl_rows)


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
    pca = PCAAbstraction(R[tr])

    sub = rng.choice(n_tr, size=min(args.subsample, n_tr), replace=False)
    log = []

    def say(s):
        print(s); log.append(s)

    say(f"=== refinement loop: {str(d['process'])} | tol = {args.tol} nats ===")

    k, history = 1, []
    while True:
        # PROPOSE + TEST
        kl, _ = completeness_kl(pca(R[tr], k), G[tr], pca(R[te], k), G[te],
                                seed=args.seed)
        # MINE incompleteness witnesses among abstraction-conflated pairs
        Zs = pca(R[sub], k)
        qi, nn = conflated_pairs(Zs, args.n_query, rng)
        div = sym_kl_rows(G[sub][qi], G[sub][nn])
        worst = float(np.quantile(div, 0.95))
        say(f"[k={k}] held-out KL = {kl:.5f} | conflated-pair completion "
            f"divergence: median {np.median(div):.5f}, q95 {worst:.5f}")
        history.append((k, kl, worst))

        if worst > args.tol and k < args.kmax:
            j = int(qi[np.argmax(div)])
            say(f"      COUNTEREXAMPLE: two prefixes alpha_{k} conflates have "
                f"sym-KL {div.max():.4f} between their true completion "
                f"distributions (true beliefs {np.round(B[sub][j], 3)} vs "
                f"{np.round(B[sub][nn[np.argmax(div)]], 3)}).")
            say(f"      -> abstraction too coarse: REFINE  k := {k + 1}")
            k += 1
            continue

        # COARSEN check: junk precision
        if k > 1:
            kl_drop, _ = completeness_kl(pca(R[tr], k - 1), G[tr],
                                         pca(R[te], k - 1), G[te],
                                         seed=args.seed)
            if kl_drop <= kl + 0.1 * max(kl, 1e-4) + 1e-4:
                say(f"      dim {k} is junk precision (KL {kl_drop:.5f} "
                    f"without it) -> COARSEN  k := {k - 1}")
                k -= 1
                kl = kl_drop
        say(f"      no counterexamples above tol, no junk precision:")
        say(f"      FIXED POINT at k = {k} — the empirical complete shell at "
            f"this probe capacity and tolerance.")
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
