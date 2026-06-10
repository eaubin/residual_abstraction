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

    Known limitation, demonstrated by Experiment 1: PCA orders directions by
    VARIANCE, and variance is not relevance — completion-irrelevant structure
    with dominant variance misaligns the top-k subspace. Experiment 2
    (compare.py) pits it against the supervised families below.
    """

    def __init__(self, X, Y=None):                 # Y ignored: unsupervised
        self.mu = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.Vt = Vt
        self.var_share = S ** 2 / (S ** 2).sum()   # for the variance audit

    def __call__(self, X, k):
        return (X - self.mu) @ self.Vt[:k].T


class CompletionPLS:
    """Supervised proposal family #1: directions of maximal cross-CORRELATION
    between residuals and COMPLETION distributions (whitened PLS, i.e.
    CCA-flavored).

    HONESTY CONSTRAINT (Experiment 2): proposal families may be supervised on
    completions ONLY — the observable concrete semantics — never on belief
    states, which exist only because these are toy processes and which serve
    strictly as evaluation ground truth. Supervising on beliefs would make
    the 'discovery' circular.

    Why whitened: raw cross-COVARIANCE is scale-blind — a dominant-variance
    irrelevant direction contributes spurious finite-sample covariance
    ~ scale/sqrt(n) that can exceed a small-scale relevant signal (this
    failure was observed directly on the 'buried' test cache). Whitening X
    (ridge-regularized) makes direction selection depend on correlation with
    completions, not on variance. Directions are returned in whitened
    coordinates; ordered and nested like PCA, so the same refine/coarsen
    loop applies.
    """

    def __init__(self, X, Y, ridge=1e-3):
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        n = len(X)
        sing_sd = S / np.sqrt(n)                      # per-direction std
        self.whiten = Vt.T * (1.0 / (sing_sd + ridge * sing_sd.max()))
        Xw = Xc @ self.whiten                          # ~unit variance dirs
        C = Xw.T @ (Y - Y.mean(axis=0))
        Uc, _, _ = np.linalg.svd(C, full_matrices=False)
        self.U = Uc

    def __call__(self, X, k):
        return (X - self.mu) @ self.whiten @ self.U[:, :k]


class HeadRowSpace:
    """Supervised proposal family #2: the row space of the full-residual
    softmax head fit to completion distributions.

    Rationale: the fitted head's weight matrix W (d x V) reads exactly the
    directions a bounded affine-softmax decoder finds useful for predicting
    completions; its top singular directions are therefore a
    'relevance-ordered' basis by construction. Same honesty constraint as
    CompletionPLS: supervised on completions only.
    """

    def __init__(self, X, Y, seed=0):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        W, _ = fit_softmax_head((X - self.mu) / self.sd, Y, seed=seed)
        U, _, _ = np.linalg.svd(W, full_matrices=False)
        self.U = U

    def __call__(self, X, k):
        return ((X - self.mu) / self.sd) @ self.U[:, :k]


# ----- the completeness measure ------------------------------------------------

def fit_softmax_head(Z, P, steps=400, lr=0.5, seed=0, l2=1e-6, _retry=2):
    """Fit q(.|z) = softmax(W z + c) to soft targets P by minimizing
    cross-entropy == KL(P || q) + const. Full-batch Adam in numpy; these
    problems are tiny (k <= 64 inputs, <= 27 outcomes).
    If the final training loss fails to beat the constant-marginal solution,
    the run diverged (observed occasionally at lr=0.5 on some geometries) —
    retry at a lower learning rate. Returns (W, c)."""
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
    # divergence check: must do at least as well as predicting the marginal
    final = mean_kl(P, head_predict(Z, W, c))
    marginal = mean_kl(P, np.tile(P.mean(axis=0), (n, 1)))
    if final > marginal + 1e-6 and _retry > 0:
        return fit_softmax_head(Z, P, steps=2 * steps, lr=lr / 5, seed=seed,
                                l2=l2, _retry=_retry - 1)
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
    """Fit head on train split, return held-out mean KL (and the head).

    Coordinates are standardized (train-split mean/std) before fitting:
    PCA coordinates carry singular-value scales differing by orders of
    magnitude, which wrecks the conditioning of the head fit. This is an
    optimization detail only — it does not change what the abstraction
    distinguishes, so mining and k-NN elsewhere stay in raw coordinates."""
    mu = Ztr.mean(axis=0)
    sd = Ztr.std(axis=0) + 1e-8
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    W, c = fit_softmax_head(Ztr, Ptr, **fit_kw)
    return mean_kl(Pte, head_predict(Zte, W, c)), (W, c)


def knn_predict(Ztr, Ytr, Zte, K=10, max_train=4000, seed=0):
    """Nonparametric decode: predict targets as the mean over the K nearest
    neighbors in abstract space. Used both for completion distributions
    (knn_kl) and for belief identification in refine.py."""
    rng = np.random.default_rng(seed)
    if len(Ztr) > max_train:
        idx = rng.choice(len(Ztr), max_train, replace=False)
        Ztr, Ytr = Ztr[idx], Ytr[idx]
    preds = np.empty((len(Zte), Ytr.shape[1]), dtype=np.float64)
    for i in range(0, len(Zte), 512):
        z = Zte[i:i + 512]
        d2 = ((z[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
        nn = np.argpartition(d2, K, axis=1)[:, :K]
        preds[i:i + 512] = Ytr[nn].mean(axis=1)
    return preds


def knn_kl(Ztr, Ptr, Zte, Pte, K=10, max_train=4000, seed=0):
    """Nonparametric sufficiency check, returning held-out mean KL.

    Purpose (the V-information disambiguation): if the affine-softmax head's
    KL is high but this is low, the information IS present in alpha_k(resid)
    and the failure lies in the probe class V — interpreter incompleteness,
    not abstraction coarseness. Z1R at k=1 is the canonical case: three
    clusters on a line are injective (sufficient) but not affinely decodable.
    """
    return mean_kl(Pte, knn_predict(Ztr, Ptr, Zte, K, max_train, seed))


def r2_score(Y, Yhat):
    return 1.0 - ((Y - Yhat) ** 2).sum() / ((Y - Y.mean(axis=0)) ** 2).sum()


def center_by_position(R, pos, train_mask):
    """Subtract the (train-split) mean residual at each sequence position.

    Why: the residual stream carries positional embeddings whose variance is
    large and completion-irrelevant for stationary processes; raw PCA spends
    its top components on it, so 'top-k PCA' stops meaning 'top-k of the
    belief geometry'. Removing per-position means before PCA is the cheap
    deconfound (cf. how Shai et al. probe across positions).
    """
    Rc = R.astype(np.float64).copy()
    for p in np.unique(pos):
        m = pos == p
        src = m & train_mask if (m & train_mask).any() else m
        Rc[m] -= R[src].mean(axis=0)
    return Rc


# ============================================================================
# Experiment 2 additions: statistical stopping support for compare.py.
#
# CONTEXT (see README.md, "Experiment 2"): the Mess3 run produced the verdict
# "variance is not relevance" — the belief plane is linearly present in the
# full residual (R^2 0.99) but the top-2 PRINCIPAL subspace captures it only
# partially (R^2 0.81), because PCA orders directions by variance, not by
# completion-relevance. The remedy is the supervised proposal families above
# (CompletionPLS, HeadRowSpace).
#
# HONESTY CONSTRAINT (important for anyone extending this): proposal families
# may be supervised ONLY on observables — residuals and completion
# distributions. The belief state is hidden-process ground truth and is used
# for EVALUATION ONLY. Supervising proposals on beliefs would make the
# "discovery" circular. (On a real LLM there are no beliefs to peek at; the
# discipline here keeps the toy experiment a faithful rehearsal.)
#
# CLEANUP NOTE (Experiment 2 closeout): two earlier proposal-family drafts
# were removed from this module to archive/rejected_families.py —
# PLSAbstraction (NIPALS PLS2, covariance-based: scale-blind, failed the
# buried-belief validation cache; CompletionPLS whitens first precisely to
# fix this) and HeadRowAbstraction (near-duplicate of HeadRowSpace). A
# duplicate definition of completeness_kl_rows that shadowed this one was
# also removed. The lesson stays here so nobody reintroduces the covariance
# variant: covariance inherits variance, and variance is not relevance.
# ============================================================================

def kl_rows(P, Q, eps=1e-12):
    """Per-sample KL(P_i || Q_i) — needed for bootstrap/SEM-based stopping."""
    P = np.clip(P, eps, None); Q = np.clip(Q, eps, None)
    return (P * (np.log(P) - np.log(Q))).sum(axis=1)


def completeness_kl_rows(Ztr, Ptr, Zte, Pte, **fit_kw):
    """Like completeness_kl but returns the per-sample held-out KL vector,
    needed for paired statistical comparisons against the belief-oracle floor
    (Experiment 2's principled stopping rule)."""
    mu = Ztr.mean(axis=0)
    sd = Ztr.std(axis=0) + 1e-8
    W, c = fit_softmax_head((Ztr - mu) / sd, Ptr, **fit_kw)
    return kl_rows(Pte, head_predict((Zte - mu) / sd, W, c))
