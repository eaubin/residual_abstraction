"""
battery.py — the frozen diagnostic battery as a reusable library.

Extracts the six BATTERY.md members as named classes and functions,
promoted from the exp-18 (mstair.py) implementations which are the
cleanest versions. expcommon.py remains the experiment-scaffolding
layer; frozen scripts stay untouched.

Contents:
  Refs          — observable references for a pair set (member 1)
  Exact         — exact (ground-truth) closures for an eval set (member 5)
  jeffreys_rows — symmetric KL, used by rho (member 2)
  Exact.rho     — member 2 (per-pair equivalence ratio)
  shift_retention — member 4 (R)
  calibration_gap — member 5 (P4 protocol, obs/exact agreement)
  cegar_loop    — member 6 (benign CEGAR discovery)
  cegar_accept  — member 6 (adversarial accept-count)
  cegar_staircase — member 6 (k*(eps) curve)

Members 1 and 3 are both Refs.obs(); member 3 just uses a held-out
pair set, which is the caller's responsibility, not a distinct function.
"""

import numpy as np

from abstraction import kl_rows
from midstream import kl_by_horizon, marginal


def jeffreys_rows(qa, qb):
    """Per-row Jeffreys (symmetric KL) divergence between (n, C) distributions."""
    return 0.5 * (kl_rows(qa, qb) + kl_rows(qb, qa))


def _mnorm(q, V, mm, m_full):
    """Marginalize an (n, V^m_full) joint to horizon mm and renormalize."""
    qm = marginal(q, V, mm, m_full)
    return qm / np.clip(qm.sum(axis=1, keepdims=True), 1e-30, None)


class Refs:
    """Observable (model-vs-model) references for one pair set.

    Battery member 1 (observable closure) and member 3 (held-out-position
    gain) are both served by .obs(); member 3 simply passes a pair set
    built on held-out positions.

    Promoted from mstair.py (exp 18) — the horizon-aware version that
    supports the mm-staircase via kl_by_horizon.
    """

    def __init__(self, ps, model, d, m_full):
        self.ps = ps
        self.model = model
        self.d = d
        self.m_full = m_full
        self.q_src = ps.run(model, None, src_side=True)
        self.q_un = ps.run(model, None)
        self.q_full = ps.run(model, np.eye(d))
        self.V = ps.V

    def obs(self, q, mm):
        """Observable closure c_obs at horizon mm.

        c_obs(P) = (D0 - D(P)) / (D0 - D_full), where D is mean KL
        of q_src against q_patched at the marginalized horizon mm.
        """
        V, mf = self.V, self.m_full
        D0 = float(kl_by_horizon(self.q_un, self.q_src, V, mf)[mm].mean())
        Df = float(kl_by_horizon(self.q_full, self.q_src, V, mf)[mm].mean())
        Dq = float(kl_by_horizon(q, self.q_src, V, mf)[mm].mean())
        return (D0 - Dq) / (D0 - Df)


class Exact:
    """Exact (ground-truth) closures for one eval pair set.

    Battery member 5 (accepted-cell calibration / the P4 protocol) uses
    .closure() compared against Refs.obs(). Battery member 2 (rho) is
    served by .rho().

    Promoted from mstair.py (exp 18).
    """

    def __init__(self, ps, model, m_full):
        self.ps = ps
        self.m_full = m_full
        self.V = ps.V
        q0 = ps.run(model, None)
        self.floor = {mm: float(kl_by_horizon(q0, ps.p_tgt3, self.V,
                                              m_full)[mm].mean())
                      for mm in range(1, m_full + 1)}
        self.gap = {mm: float(kl_by_horizon(q0, ps.p_src3, self.V,
                                            m_full)[mm].mean())
                    for mm in range(1, m_full + 1)}
        self._q0 = q0

    def closure(self, q, mm):
        """Exact closure at horizon mm: how far q moves from unpatched
        toward the source's belief-conditioned completion distribution."""
        t = float(kl_by_horizon(q, self.ps.p_src3, self.V,
                                self.m_full)[mm].mean())
        return (self.gap[mm] - t) / (self.gap[mm] - self.floor[mm])

    def rho(self, qC, qX, mm):
        """Per-pair equivalence ratio: mean Jeffreys(C, X) / mean
        Jeffreys(C, unpatched), marginalized to horizon mm.

        rho <= 0.25: behaviorally equivalent to the reference C.
        rho >= 0.5: behaviorally distinct.
        """
        V, mf = self.V, self.m_full
        a = _mnorm(qC, V, mm, mf)
        b = _mnorm(qX, V, mm, mf)
        u = _mnorm(self._q0, V, mm, mf)
        return float(jeffreys_rows(a, b).mean()
                     / jeffreys_rows(a, u).mean())


def calibration_gap(obs_score, exact_score):
    """Battery member 5: the P4 protocol's obs/exact agreement check.

    Returns the absolute gap |obs - exact|. The caller checks against
    the registered threshold (0.10 for Mess3).
    """
    return abs(obs_score - exact_score)


def shift_retention(gain_X_shift, gain_X_base, gain_C_shift, gain_C_base):
    """Battery member 4: relative retention R under a distribution shift.

    R(X, shift) = [gain_X(shift) / gain_X(base)] / [gain_C(shift) / gain_C(base)]

    R ~ 1: X retains its gain under the shift as well as the clean
    reference does. R << 1 or R < 0: fragile or inverting.
    """
    retC = gain_C_shift / gain_C_base
    return (gain_X_shift / gain_X_base) / retC


def cegar_loop(model, ps, refs, d, eps, k_max, mm, eps_drop=None):
    """The frozen CEGAR accept-only loop (the battery member 6 predicate).

    This is the accept-only loop whose staircase record is the
    calibration evidence (exps 8–18). With eps_drop=None (default),
    it matches exactly the frozen predicate.

    Pass eps_drop (e.g. 0.01) to opt into the exp-6/7 full discovery
    loop, which adds a coarsen pass that drops directions whose removal
    costs less than eps_drop. That is a different instrument with
    different provenance.

    Returns (k_star, Q, c_obs_final): the number of accepted directions,
    the orthonormal basis, and the final observable closure.
    """
    from discover import mined_direction

    Q = np.zeros((d, 0))
    c_cur = 0.0
    q_cur = refs.q_un
    V = refs.V
    while Q.shape[1] < k_max:
        w_rows = kl_by_horizon(q_cur, refs.q_src, V, refs.m_full)[mm]
        v = mined_direction(ps, Q, w_rows)
        Q_try = np.hstack([Q, v[:, None]])
        q_try = ps.run(model, Q_try @ Q_try.T)
        c_try = refs.obs(q_try, mm)
        if c_try - c_cur < eps:
            break
        Q, q_cur, c_cur = Q_try, q_try, c_try
    if eps_drop is not None:
        # exp-6/7 coarsen pass (discover.py:329)
        changed = True
        while changed and Q.shape[1] > 1:
            changed = False
            for j in range(Q.shape[1]):
                Qj = np.delete(Q, j, axis=1)
                cj = refs.obs(ps.run(model, Qj @ Qj.T), mm)
                if c_cur - cj < eps_drop:
                    Q, c_cur, changed = Qj, cj, True
                    break
    return Q.shape[1], Q, c_cur


def cegar_accept(model, view, pull, ps, refs, mm, d, eps, k_max):
    """Battery member 6: adversarial CEGAR accept-count at a single eps.

    Like cegar_loop but patches go through `pull` (the adversarial
    pullback) and mining uses `view` (a ZView into the adversarial
    coordinates). Returns k* (the number of accepted directions).
    """
    from discover import mined_direction

    Q = np.zeros((d, 0))
    c_cur = 0.0
    q_cur = refs.q_un
    V = refs.V
    while Q.shape[1] < k_max:
        w_rows = kl_by_horizon(q_cur, refs.q_src, V, refs.m_full)[mm]
        v = mined_direction(view, Q, w_rows)
        Q_try = np.hstack([Q, v[:, None]])
        P = pull(Q_try @ Q_try.T)
        q_try = ps.run(model, P)
        c_try = refs.obs(q_try, mm)
        if c_try - c_cur < eps:
            break
        Q, q_cur, c_cur = Q_try, q_try, c_try
    return Q.shape[1]


def cegar_staircase(model, ps, refs, d, eps_grid, k_max, mm):
    """Battery member 6: k*(eps) staircase over a grid of thresholds.

    Returns {eps: k_star} for each eps in eps_grid.
    """
    return {eps: cegar_loop(model, ps, refs, d, eps, k_max, mm)[0]
            for eps in eps_grid}
