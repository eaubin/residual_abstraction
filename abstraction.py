"""
abstraction.py — the probe family V and the completeness measure (numpy-only).

CONTEXT (see README.md): classical abstract interpretation asks whether an
abstraction is *complete* (no precision lost). With a stochastic concrete
semantics and bounded compute, that becomes two replacements:

  * the extractor class V is fixed and cheap — here, rank-k linear
    projections of the residual followed by an affine-softmax head. What such
    probes can recover is exactly V-information (Xu et al. 2020):
    "information usable under computational constraints";
  * completeness is real-valued — the held-out mean KL between the true
    completion distribution and the one predicted from the abstraction.
    KL = 0 within noise <=> the abstraction is a sufficient statistic for the
    completion kernel within class V.

These utilities are shared by analysis.py (sufficiency curves) and refine.py
(the CEGAR-style outer loop). They are deliberately torch-free so they run on
the cached arrays anywhere.
"""

import numpy as np


# ----- linear algebra probes --------------------------------------------------

def affine_lstsq(X, Y):
    """Closed-form affine regression X -> Y. Returns (W, b, R^2 overall)."""
    X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    Wb, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    pred = X1 @ Wb
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
    return Wb[:-1], Wb[-1], 1.0 - ss_res / ss_tot


class PCAAbstraction:
    """alpha_k: residual -> its coordinates in the top-k principal subspace.

    This is the *proposal* family of the experiment: cheap, ordered (k+1
    extends k), and capacity-indexed, so 'refine' and 'coarsen' are literally
    k+1 and k-1. The interesting question is at which k completeness
    saturates — for a 3-hidden-state process the answer should be k = 2,
    the dimension of the belief simplex.
    """

    def __init__(self, X):
        self.mu = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.Vt = Vt

    def __call__(self, X, k):
        return (X - self.mu) @ self.Vt[:k].T


# ----- the completeness measure ------------------------------------------------

def fit_softmax_head(Z, P, steps=400, lr=0.5, seed=0, l2=1e-6):
    """Fit q(.|z) = softmax(W z + c) to soft targets P by minimizing
    cross-entropy == KL(P || q) + const. Full-batch Adam in numpy; these
    problems are tiny (k <= 64 inputs, <= 27 outcomes).
    Returns (W, c)."""
    rng = np.random.default_rng(seed)
    n, k = Z.shape
    V = P.shape[1]
    W = 0.01 * rng.standard_normal((k, V))
    c = np.zeros(V)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mc = np.zeros_like(c); vc = np.zeros_like(c)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        logits = Z @ W + c
        logits -= logits.max(axis=1, keepdims=True)
        q = np.exp(logits); q /= q.sum(axis=1, keepdims=True)
        gL = (q - P) / n                       # d(mean CE)/d(logits)
        gW = Z.T @ gL + l2 * W
        gc = gL.sum(axis=0)
        for g, m, v, theta in ((gW, mW, vW, W), (gc, mc, vc, c)):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            theta -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return W, c


def head_predict(Z, W, c):
    logits = Z @ W + c
    logits -= logits.max(axis=1, keepdims=True)
    q = np.exp(logits)
    return q / q.sum(axis=1, keepdims=True)


def mean_kl(P, Q, eps=1e-12):
    """Mean KL(P || Q) in nats — the quantified incompleteness."""
    P = np.clip(P, eps, None); Q = np.clip(Q, eps, None)
    return float((P * (np.log(P) - np.log(Q))).sum(axis=1).mean())


def sym_kl_rows(P, Q, eps=1e-12):
    """Row-wise symmetric KL between paired distributions (for counterexample
    mining: 'how behaviorally different are these two prefixes really?')."""
    P = np.clip(P, eps, None); Q = np.clip(Q, eps, None)
    lp, lq = np.log(P), np.log(Q)
    return ((P - Q) * (lp - lq)).sum(axis=1)


def completeness_kl(Ztr, Ptr, Zte, Pte, **fit_kw):
    """Fit head on train split, return held-out mean KL (and the head)."""
    W, c = fit_softmax_head(Ztr, Ptr, **fit_kw)
    return mean_kl(Pte, head_predict(Zte, W, c)), (W, c)
