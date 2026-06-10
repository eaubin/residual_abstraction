"""
analysis.py — calibration (Shai et al. replication) + sufficiency curves.

CONTEXT (see README.md). Two questions, both answered exactly because the
generating process is known:

(A) CALIBRATION — does the residual stream linearly embed the belief state?
    Affine regression residual -> exact belief, reported as R^2; for Mess3 the
    regressed beliefs are plotted in simplex coordinates next to the ground
    truth. Seeing the fractal reproduces the headline result of Shai et al.
    2024 and certifies that the rest of the pipeline is probing real structure.

(B) SUFFICIENCY CURVE — quantified completeness as a function of abstraction
    capacity. For each rank k, project residuals onto their top-k principal
    subspace (the abstraction alpha_k), fit a softmax head to the exact
    m-token completion distribution, and report held-out mean KL. Reference
    rows:
      - 'belief oracle': head fit on the true belief state. The belief state
        is a sufficient statistic by construction, so this is the noise/probe
        floor; any KL here reflects head capacity, not lost information.
      - 'full residual': k = d_model — what the whole stream knows.
    Expected signature: KL drops sharply until k = 2 (the dimension of the
    3-state belief simplex) and is flat after — i.e. the empirically optimal
    abstraction coincides with the known minimal sufficient statistic. This
    elbow is the experiment's main quantitative finding; refine.py then shows
    an outer loop *discovering* it without being told the answer.
"""

import argparse
import os

import numpy as np

from abstraction import (PCAAbstraction, affine_lstsq, center_by_position,
                         completeness_kl, mean_kl)

SQ3 = np.sqrt(3.0)


def simplex_xy(B):
    """Barycentric -> 2D coords of the 2-simplex (equilateral triangle)."""
    return np.stack([B[:, 1] + 0.5 * B[:, 2], (SQ3 / 2) * B[:, 2]], axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/z1r")
    ap.add_argument("--max-points-plot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d = np.load(os.path.join(args.outdir, "cache.npz"))
    R, B, G = d["resid"], d["belief"], d["mgram"]
    name = str(d["process"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(R))
    R, B, G = R[perm], B[perm], G[perm]
    n_tr = int(0.7 * len(R))
    tr, te = slice(None, n_tr), slice(n_tr, None)

    # Deconfound positional-embedding variance (large, completion-irrelevant
    # for stationary processes; otherwise it dominates the top PCs and a
    # single-bias affine probe can't remove it either). Old caches lack `pos`.
    centered = "pos" in d
    if centered:
        train_mask = np.zeros(len(R), dtype=bool); train_mask[:n_tr] = True
        R = center_by_position(R, d["pos"][perm], train_mask)

    print(f"=== analysis: {name} | {len(R)} positions, resid dim {R.shape[1]}, "
          f"{G.shape[1]} completion outcomes"
          f"{', per-position centered' if centered else ''} ===\n")

    # ----- (A) calibration: belief-state geometry in the residual stream ----
    W, b0, r2 = affine_lstsq(R[tr], B[tr])
    Bhat = R[te] @ W + b0
    r2_te = 1.0 - ((B[te] - Bhat) ** 2).sum() / \
        ((B[te] - B[te].mean(0)) ** 2).sum()
    print(f"(A) affine probe residual -> belief state: "
          f"R^2 train {r2:.4f}, held-out {r2_te:.4f}")
    print("    Interpretation: high R^2 = the residual stream linearly embeds")
    print("    the minimal sufficient statistic for completions (Shai et al.).")

    # ----- (B) sufficiency curve --------------------------------------------
    pca = PCAAbstraction(R[tr])
    dmodel = R.shape[1]
    ks = sorted(set([1, 2, 3, 4, 6, 8, 16, 32, dmodel]))
    ks = [k for k in ks if k <= dmodel]

    kl_oracle, _ = completeness_kl(B[tr], G[tr], B[te], G[te], seed=args.seed)
    print(f"\n(B) quantified completeness, held-out mean KL(true || predicted) "
          f"over next-{int(d['m'])}-token distributions [nats]:")
    print(f"    belief-state oracle (sufficiency floor): {kl_oracle:.5f}")

    curve = []
    for k in ks:
        kl, _ = completeness_kl(pca(R[tr], k), G[tr], pca(R[te], k), G[te],
                                seed=args.seed)
        curve.append(kl)
        tag = "  <- full residual" if k == dmodel else ""
        print(f"    rank-{k:>3} PCA abstraction: {kl:.5f}{tag}")

    # the elbow: smallest k within 10% (relative, floored) of full-residual KL
    full = curve[-1]
    tol = full + 0.1 * max(full, 1e-4) + 1e-4
    k_star = next(k for k, kl in zip(ks, curve) if kl <= tol)
    print(f"\n    Elbow: completeness saturates at k = {k_star}.")
    print("    Expected k = 2 — the dimension of the 3-state belief simplex.")
    if k_star <= 2:
        print("    Reading: the coarsest PCA abstraction of the residual stream")
        print("    that is (V-)sufficient for completions matches (or undercuts,")
        print("    if the REACHABLE belief set is lower-dimensional — Z1R's is")
        print("    three points) the known minimal sufficient statistic.")
    else:
        print("    Reading: PCA saturates LATE. Experiment 2 (compare.py)")
        print("    diagnosed this as a proposal-family artifact, not a fact")
        print("    about the model: the dominant variance is current-token")
        print("    identity, so the belief plane is smeared across deeper")
        print("    principal components, while a completion-supervised family")
        print("    (whitened PLS) saturates at k = 2. Variance is not relevance.")

    # ----- plots -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        npts = min(args.max_points_plot, len(B[te]))
        Bt, Bh = B[te][:npts], np.clip(Bhat[:npts], 0, 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, pts, title in (
            (axes[0], Bt, "ground-truth belief states"),
            (axes[1], Bh, "affine decode of residual stream"),
        ):
            xy = simplex_xy(pts)
            ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c=np.clip(pts, 0, 1),
                       linewidths=0)
            tri = np.array([[0, 0], [1, 0], [0.5, SQ3 / 2], [0, 0]])
            ax.plot(tri[:, 0], tri[:, 1], "k-", lw=0.6)
            ax.set_title(title); ax.set_aspect("equal"); ax.axis("off")
        fig.suptitle(f"{name}: belief geometry in the residual stream "
                     f"(held-out R^2 = {r2_te:.3f})")
        p1 = os.path.join(args.outdir, "belief_regression.png")
        fig.tight_layout(); fig.savefig(p1, dpi=160); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, curve, "o-", label="rank-k PCA of residual")
        ax.axhline(kl_oracle, ls="--", c="g", label="belief-state oracle")
        ax.axvline(2, ls=":", c="gray", label="dim(belief simplex) = 2")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("abstraction rank k")
        ax.set_ylabel("held-out mean KL [nats]")
        ax.set_title(f"{name}: completeness vs abstraction capacity")
        ax.legend(fontsize=8)
        p2 = os.path.join(args.outdir, "sufficiency_curve.png")
        fig.tight_layout(); fig.savefig(p2, dpi=160); plt.close(fig)
        print(f"\n    wrote {p1}\n    wrote {p2}")
    except Exception as e:  # matplotlib optional; numbers above are the result
        print(f"\n    (plotting skipped: {e})")


if __name__ == "__main__":
    main()
