"""
compare.py — Experiment 2: proposal families, the tolerance staircase, and
the displaced-variance audit.

CONTEXT (see README.md; this follows Experiment 1's findings). Experiment 1
ended with three open items, each of which this script addresses:

(1) PROPOSAL FAMILIES. The Mess3 verdict was "variance is not relevance":
    the top-2 PCA plane was only ~81% the belief plane (affine R^2) while the
    full residual was ~99%. Here PCA competes against two
    completion-supervised families — CompletionPLS (cross-covariance with
    completion distributions) and HeadRowSpace (row space of the fitted
    full-residual decoder). HONESTY CONSTRAINT: families may be supervised on
    completions ONLY (the observable concrete semantics), never on belief
    states, which remain strictly evaluation ground truth; supervising on
    beliefs would make the discovery circular.
    PRE-REGISTERED PREDICTION (written before running on real caches): on
    Mess3, a supervised family reaches affine abstraction->belief R^2 close
    to the full-residual ~0.99 at k = 2. If so, the Experiment-1 gap was
    subspace MISALIGNMENT (PCA artifact); if not, the residual's belief
    embedding is genuinely curved.

(2) THE TOLERANCE STAIRCASE. The KL0/2 calibration of Experiment 1 is a
    pragmatic policy, not a principled threshold, so the fixed point is
    policy-relative. The honest object is therefore not one complete shell
    but the function k*(tol) — the minimal capacity meeting each tolerance.
    We compute KL(k) once per family and report the whole staircase; any
    tolerance policy is then a vertical line the reader can move.

(3) A PRINCIPLED STOPPING RULE. As a non-arbitrary alternative: stop at the
    smallest k whose held-out per-sample KL is statistically
    indistinguishable from the belief-oracle floor (paired comparison,
    mean difference <= 2 standard errors). "Indistinguishable from knowing
    the minimal sufficient statistic" is a sufficiency claim with error bars
    rather than a tolerance choice.

(4) THE DISPLACED-VARIANCE AUDIT. If PCA's top directions are not belief
    directions, what are they? For each top principal component we report its
    variance share and how much of it is explained by (a) the current token
    and (b) the belief state. Hypothesis from Experiment 1: the dominant
    completion-redundant variance is current-token identity kept for the
    unembedding path. Requires caches written by the updated train.py (which
    stores `tok`); skipped gracefully otherwise.
"""

import argparse
import os

import numpy as np

from abstraction import (CompletionPLS, HeadRowSpace, PCAAbstraction,
                         affine_lstsq, center_by_position,
                         completeness_kl_rows, knn_predict, mean_kl, r2_score)


def identification(Z_tr, Z_te, Btr, Bte, seed):
    Wb, b0, _ = affine_lstsq(Z_tr, Btr)
    r2_aff = r2_score(Bte, Z_te @ Wb + b0)
    r2_nn = r2_score(Bte, knn_predict(Z_tr, Btr, Z_te, seed=seed))
    return r2_aff, r2_nn


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/z1r")
    ap.add_argument("--kmax", type=int, default=10)
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
    if "pos" in d:
        mask = np.zeros(len(R), dtype=bool); mask[:n_tr] = True
        R = center_by_position(R, d["pos"][perm], mask)

    print(f"=== Experiment 2: {name} | {len(R)} positions ===\n")

    # Belief-oracle floor with per-sample KLs (for the statistical rule).
    kl_oracle_rows = completeness_kl_rows(B[tr], G[tr], B[te], G[te],
                                          seed=args.seed)
    kl_oracle = float(kl_oracle_rows.mean())
    KL0 = mean_kl(G[te], np.tile(G[tr].mean(axis=0), (len(G[te]), 1)))
    print(f"unconditional baseline KL0 = {KL0:.5f} | belief-oracle floor = "
          f"{kl_oracle:.5f}\n")

    families = {
        "pca": PCAAbstraction(R[tr]),
        "pls": CompletionPLS(R[tr], G[tr]),
        "head": HeadRowSpace(R[tr], G[tr], seed=args.seed),
    }
    ks = list(range(1, min(args.kmax, R.shape[1]) + 1))

    # Full-residual reference (same for all families).
    Wb, b0, _ = affine_lstsq(R[tr], B[tr])
    r2_full = r2_score(B[te], R[te] @ Wb + b0)
    print(f"affine full residual -> belief (reference): R^2 = {r2_full:.4f}\n")

    curves, k_stat, ident = {}, {}, {}
    for fam_name, fam in families.items():
        rows_by_k = []
        for k in ks:
            rows_by_k.append(completeness_kl_rows(
                fam(R[tr], k), G[tr], fam(R[te], k), G[te], seed=args.seed))
        curves[fam_name] = [float(r.mean()) for r in rows_by_k]
        # Stopping rule: paired against the oracle floor, with an explicit
        # equivalence margin. Pure "within 2 SE of the oracle" is the honest
        # ideal but with thousands of samples it detects even nats-level dust;
        # we therefore accept k when the remaining gap to the oracle is
        # either statistically zero (2 SE) or has closed >= 98% of the
        # closable range (KL0 - oracle). The margin is declared, not hidden —
        # the staircase below shows every other policy.
        margin = 0.02 * max(KL0 - kl_oracle, 0.0)
        k_stat[fam_name] = None
        for k, rows in zip(ks, rows_by_k):
            diff = rows - kl_oracle_rows
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            if diff.mean() <= max(2 * se, margin):
                k_stat[fam_name] = k
                break
        # Identification at the stopping k, falling back to the best-KL k
        # (labeled) when the rule never fires within range.
        k_id = k_stat[fam_name] or ks[int(np.argmin(curves[fam_name]))]
        ident[fam_name] = (k_id, *identification(
            fam(R[tr], k_id), fam(R[te], k_id), B[tr], B[te], args.seed))

    # ----- report -------------------------------------------------------------
    header = "k    " + "".join(f"{f:>10}" for f in families)
    print("held-out mean KL by abstraction rank (lower = more complete):")
    print(header)
    for i, k in enumerate(ks):
        print(f"{k:<5}" + "".join(f"{curves[f][i]:>10.5f}" for f in families))

    print("\nstopping rule (per-sample KL within 2 SE of the belief-oracle "
          "floor, OR >= 98% of the closable KL0-to-oracle range closed):")
    for f in families:
        ks_str = str(k_stat[f]) if k_stat[f] else f">{ks[-1]}"
        kid, r2a, r2n = ident[f]
        print(f"  {f:>5}: k* = {ks_str:<4} | identification at k={kid}: "
              f"affine R^2 {r2a:.4f}, k-NN R^2 {r2n:.4f}")
    print("\nReading guide: if a supervised family's affine R^2 reaches the "
          f"full-residual reference ({r2_full:.3f}) where PCA's does not, the "
          "Experiment-1 identification gap was subspace MISALIGNMENT "
          "(variance != relevance), not curvature of the belief embedding. "
          "If all families plateau below it, the embedding is genuinely "
          "nonlinear.")

    # ----- tolerance staircase ------------------------------------------------
    tols = np.geomspace(max(kl_oracle, 1e-5), max(KL0, 2e-5), 12)
    print("\ntolerance staircase k*(tol) — the complete shell as a function "
          "of tolerance policy, not a canonical point:")
    print("tol      " + "".join(f"{f:>7}" for f in families))
    for t in tols:
        row = ""
        for f in families:
            kk = next((k for k, kl in zip(ks, curves[f]) if kl <= t), None)
            row += f"{(str(kk) if kk else '-'):>7}"
        print(f"{t:<9.5f}" + row)

    # ----- displaced-variance audit -------------------------------------------
    if "tok" in d:
        tok = d["tok"][perm]
        pca = families["pca"]
        Zt = pca(R[tr], min(8, R.shape[1]))
        onehot = np.eye(int(tok.max()) + 1)[tok[tr]]
        print("\ndisplaced-variance audit: what do the top principal "
              "components encode?")
        print("PC   var-share   R^2(current token)   R^2(belief)")
        for i in range(Zt.shape[1]):
            y = Zt[:, i:i + 1]
            _, _, r2t = affine_lstsq(onehot, y)
            _, _, r2b = affine_lstsq(B[tr], y)
            print(f"{i + 1:<5}{pca.var_share[i]:>9.3f}{r2t:>17.3f}"
                  f"{r2b:>15.3f}")
        print("Hypothesis under test: dominant completion-redundant variance "
              "is current-token identity (kept for the unembedding path).")
    else:
        print("\n(displaced-variance audit skipped: cache has no `tok` array;"
              " rerun train.py to regenerate)")

    # ----- plot ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for f in families:
            axes[0].plot(ks, curves[f], "o-", label=f)
        axes[0].axhline(kl_oracle, ls="--", c="g", label="belief oracle")
        axes[0].axhline(KL0, ls=":", c="r", label="no-info baseline")
        axes[0].set_yscale("log"); axes[0].set_xlabel("rank k")
        axes[0].set_ylabel("held-out mean KL [nats]")
        axes[0].set_title(f"{name}: completeness by proposal family")
        axes[0].legend(fontsize=8)
        for f in families:
            stair = [next((k for k, kl in zip(ks, curves[f]) if kl <= t),
                          np.nan) for t in tols]
            axes[1].step(tols, stair, where="post", label=f)
        axes[1].set_xscale("log"); axes[1].set_xlabel("tolerance [nats]")
        axes[1].set_ylabel("k*(tol)")
        axes[1].set_title("the complete shell as a staircase")
        axes[1].legend(fontsize=8)
        p = os.path.join(args.outdir, "experiment2.png")
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plotting skipped: {e})")


if __name__ == "__main__":
    main()
