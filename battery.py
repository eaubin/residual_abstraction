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
  rho_obs       — observable member-2 rho, without an exact oracle
  directional_tolerance_partition
                — shared pass/recalibrate/fail split for signed tolerances
  CandidateConfig / build_candidates
                — forward home for the oracle-withdrawal candidate menu

Members 1 and 3 are both Refs.obs(); member 3 just uses a held-out
pair set, which is the caller's responsibility, not a distinct function.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from abstraction import kl_rows
from midstream import kl_by_horizon, marginal


@dataclass(frozen=True)
class CandidateConfig:
    """Registered knobs for the oracle-withdrawal candidate menu.

    The defaults match the pstack oracle-withdrawal arc (exps 24-27), but the
    caller supplies the process/model/pair set. Keeping these values explicit
    prevents the generic library helper from silently becoming "the exp-24
    script in disguise."
    """

    m: int = 3
    mm: int = 3
    ts_disc: tuple = (10, 18, 26, 34)
    basis_seqs: int = 240
    eps: float = 0.05
    eps_drop: float = 0.01
    k_max: int = 10
    min_dim: int = 1
    max_dim: int = 8
    layer: int = 1


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
        return rho_obs(self._q0, qC, qX, self.V, mm, self.m_full)


def rho_obs(q0, qC, qX, V, mm, m_full):
    """Observable member-2 rho without using exact process state.

    rho(X) = mean J(q_C, q_X) / mean J(q_C, q_unpatched), with all rows
    marginalized to horizon ``mm`` and renormalized. This is the canonical
    forward copy of the helper hand-copied in the oracle-withdrawal arc.
    """
    a = _mnorm(qC, V, mm, m_full)
    b = _mnorm(qX, V, mm, m_full)
    u = _mnorm(q0, V, mm, m_full)
    denom = jeffreys_rows(a, u).mean()
    if denom <= 0:
        return float("nan")
    return float(jeffreys_rows(a, b).mean() / denom)


def rho_band(r, equiv=0.25, distinct=0.50):
    """Classify rho under the standard equivalent/distinct bands."""
    if r <= equiv:
        return "equivalent"
    if r >= distinct:
        return "distinct"
    return "indeterminate"


def calibration_gap(obs_score, exact_score):
    """Battery member 5: the P4 protocol's obs/exact agreement check.

    Returns the absolute gap |obs - exact|. The caller checks against
    the registered threshold (0.10 for Mess3).
    """
    return abs(obs_score - exact_score)


def directional_tolerance_partition(gaps, band, *, dangerous="positive"):
    """Shared FORMALISM 6.1 rule-9 partition for signed tolerance misses.

    ``gaps`` are signed misses, such as obs - exact. If all absolute gaps are
    inside ``band``, status is PASS. If the dangerous side exceeds ``band``,
    status is FAIL. Otherwise the miss is only on the safe/conservative side
    and status is RECALIBRATE. The helper returns the extrema used to justify
    the branch so scripts can print their registered labels.
    """
    arr = np.asarray(list(gaps), dtype=float)
    if arr.size == 0:
        raise ValueError("directional_tolerance_partition needs at least one gap")
    if dangerous not in {"positive", "negative"}:
        raise ValueError("dangerous must be 'positive' or 'negative'")
    max_abs = float(np.max(np.abs(arr)))
    max_pos = float(np.max(arr))
    max_neg = float(np.max(-arr))
    if max_abs <= band:
        status = "PASS"
    elif (max_pos if dangerous == "positive" else max_neg) > band:
        status = "FAIL"
    else:
        status = "RECALIBRATE"
    return {
        "status": status,
        "max_abs": max_abs,
        "max_positive": max_pos,
        "max_negative": max_neg,
        "dangerous_excess": max_pos if dangerous == "positive" else max_neg,
        "safe_excess": max_neg if dangerous == "positive" else max_pos,
        "band": band,
        "dangerous": dangerous,
    }


def shift_retention(gain_X_shift, gain_X_base, gain_C_shift, gain_C_base):
    """Battery member 4: relative retention R under a distribution shift.

    R(X, shift) = [gain_X(shift) / gain_X(base)] / [gain_C(shift) / gain_C(base)]

    R ~ 1: X retains its gain under the shift as well as the clean
    reference does. R << 1 or R < 0: fragile or inverting.
    """
    retC = gain_C_shift / gain_C_base
    return (gain_X_shift / gain_X_base) / retC


def observable_completion_basis(model, proc, cfg, n_seqs, seed, cand_cfg=None):
    """Residual rows and observable model completions for candidate fitting.

    Labels are model chain probabilities under the unpatched run, not exact
    process m-grams. This is the oracle-free basis sample used by the
    oracle-withdrawal candidate menu.
    """
    import torch

    from abstraction import center_by_position
    from midstream import chain_probs, stream_to

    c = cand_cfg or CandidateConfig()
    rng = np.random.default_rng(seed)
    X = proc.sample(n_seqs, cfg["seq_len"], rng)
    S = stream_to(model, torch.from_numpy(X), c.layer).double().numpy()
    conts = np.asarray(list(product(range(proc.V), repeat=c.m)), dtype=np.int64)
    rows, labels, pos = [], [], []
    for t in c.ts_disc:
        Xc = np.repeat(X[:, None, :], len(conts), axis=1).copy()
        Xc[:, :, t + 1:t + 1 + c.m] = conts[None, :, :]
        q, _ = chain_probs(model, Xc, c.layer, None, t, c.m, proc.V)
        rows.append(S[:, t])
        labels.append(q)
        pos.append(np.full(len(X), t, dtype=np.int64))
    R = np.concatenate(rows)
    Q = np.concatenate(labels)
    P = np.concatenate(pos)
    Rc = center_by_position(R, P, np.ones(len(R), dtype=bool))
    return Rc, Q


def fixed_mined_basis(ps, refs, dim, mm):
    """Observable weighted prefix-delta basis without accept/coarsen logic."""
    from discover import mined_direction

    Q = np.zeros((ps.d, 0))
    weights = kl_by_horizon(refs.q_un, refs.q_src, refs.V, refs.m_full)[mm]
    for _ in range(dim):
        v = mined_direction(ps, Q, weights)
        Q = np.hstack([Q, v[:, None]])
    return Q


def _candidate_entry(name, Q, d, *, selectable=True, kind="candidate",
                     min_dim=1):
    from midstream import orthonormal

    if Q is None:
        return {"P": np.eye(d), "Q": None, "dim": d,
                "selectable": False, "kind": "ceiling"}
    if Q.shape[1] == 0:
        P = np.zeros((d, d))
    else:
        Q = orthonormal(Q)
        P = Q @ Q.T
    return {"P": P, "Q": Q, "dim": int(Q.shape[1]),
            "selectable": selectable and Q.shape[1] >= min_dim, "kind": kind}


def build_candidates(model, proc, cfg, disc, refs_d, seed, cand_cfg=None):
    """Build the standard oracle-withdrawal candidate menu.

    Returns ``(candidates, k_cegar, k_ref)``. ``candidates`` contains ``full``
    plus ``cegar``, ``pca``, ``obs_pls``, ``delta``, ``emb``, and ``rand``
    entries with projector ``P``, basis ``Q``, dimension, selectability, and
    kind. This is a forward library home for the machinery first implemented
    in exp 24; concluded scripts can remain frozen historical records.
    """
    import torch

    from abstraction import CompletionPLS, PCAAbstraction
    from midstream import orthonormal

    c = cand_cfg or CandidateConfig()
    d = cfg["d_model"]
    k_cegar, Qc, _ = cegar_loop(model, disc, refs_d, d, c.eps, c.k_max, c.mm,
                                eps_drop=c.eps_drop)
    k_ref = int(np.clip(k_cegar, c.min_dim, c.max_dim))

    Rb, Qobs = observable_completion_basis(model, proc, cfg, c.basis_seqs,
                                           seed + 555, c)
    pca = PCAAbstraction(Rb)
    pls = CompletionPLS(Rb, Qobs)
    with torch.no_grad():
        Wtok = model.tok.weight.double().numpy()

    bases = {
        "cegar": Qc[:, :k_ref],
        "pca": pca.Vt[:k_ref].T,
        "obs_pls": orthonormal(pls.whiten @ pls.U[:, :k_ref]),
        "delta": fixed_mined_basis(disc, refs_d, k_ref, c.mm),
        "emb": orthonormal(Wtok.T)[:, :min(k_ref, proc.V)],
        "rand": orthonormal(np.random.default_rng(seed).standard_normal(
            (d, k_ref))),
    }
    candidates = {"full": _candidate_entry("full", None, d)}
    for name, Q in bases.items():
        candidates[name] = _candidate_entry(
            name, Q, d, selectable=name != "rand", min_dim=c.min_dim,
            kind="destructive" if name == "rand" else "candidate")
    return candidates, k_cegar, k_ref


def max_principal_angle(qa, qb):
    """Largest principal angle in degrees between two orthonormal bases."""
    from discover import principal_angles_deg

    return float(principal_angles_deg(qa, qb).max())


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


def _selftest():
    q0 = np.array([[0.7, 0.3]])
    qC = np.array([[0.4, 0.6]])
    assert rho_obs(q0, qC, qC, 2, 1, 1) < 1e-12
    assert rho_band(0.10) == "equivalent"
    assert rho_band(0.30) == "indeterminate"
    assert rho_band(0.80) == "distinct"

    part = directional_tolerance_partition([0.02, -0.03], 0.10)
    assert part["status"] == "PASS"
    part = directional_tolerance_partition([0.12, -0.03], 0.10)
    assert part["status"] == "FAIL"
    part = directional_tolerance_partition([0.03, -0.12], 0.10)
    assert part["status"] == "RECALIBRATE"
    part = directional_tolerance_partition([0.03, -0.12], 0.10,
                                           dangerous="negative")
    assert part["status"] == "FAIL"

    e = np.eye(4)
    assert max_principal_angle(e[:, :2], e[:, :2]) < 1e-6
    assert abs(max_principal_angle(e[:, :2], e[:, 2:]) - 90.0) < 1e-6
    cfg = CandidateConfig()
    assert cfg.m == 3 and cfg.mm == 3 and cfg.min_dim <= cfg.max_dim
    print("battery selftest passed: rho_obs, bands, partition, angles")


if __name__ == "__main__":
    _selftest()
