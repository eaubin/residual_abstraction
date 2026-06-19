"""
interventions.py — intervention-class patch API and predicate-control scorer.

Living library home for the Phase 3 intervention-class benchmark
(INTERVENTION_CLASS_BENCHMARK.md, design steps I0-I4). New live scripts import
from here; concluded/frozen experiment scripts must not be imported back.

Patch convention. A patch is a d x d operator ``P`` consumed by
``discover.PairSet.run``: the patched prefix is ``pt + (pref_src - pt) @ P``,
so ``P`` maps the source-target residual delta to the write update applied to
the target prefix. The reductions are:

    P = None                 unpatched (target run)
    P = I_d                  full patch (-> source run)
    P = w w^T / (w.w)         same-read/same-write rank-1 (the exp-29 patch)
    P = c w^T / (c.w)         rank-1 oblique: read covector c, write vector w;
                              sets the read functional c to the source value
    P = C (C^T W)^{-T} W^T    rank-k oblique composition: sets every read
                              functional in C to its source value

Oracle discipline. Every operator here and every predicate-control score is
observable/model-only. Exact predicate truth (``predicates.exact_pphi``) enters
only through ``endpoint_audit``, which gates whether the observable endpoints
are calibrated; it never selects a direction, write, or strength.
"""

from dataclasses import dataclass

import numpy as np

# Read/write degeneracy floor: |c.w| (rank 1) or |det C^T W| (rank k) below
# this is a singular oblique patch (the write cannot move the read value).
RW_TOL = 1e-9
# Predicate-room floor: the full-patch denominator must exceed this for the
# closure score to be interpretable (matches exp-29's pphi_closure guard).
ROOM_TOL = 1e-9


class SingularReadWrite(ValueError):
    """The read covector(s) and write direction(s) cannot set the read value:
    c.w ~= 0 in rank 1, or C^T W is singular in rank k."""


# ---------------------------------------------------------------------------
# direction utilities
# ---------------------------------------------------------------------------

def unit(v):
    """Unit-normalize a vector; raise on a (near) zero vector."""
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < RW_TOL:
        raise ValueError("cannot normalize a zero-norm direction")
    return v / n


# ---------------------------------------------------------------------------
# patch API (d x d operators for PairSet.run)
# ---------------------------------------------------------------------------

def same_write_patch(w):
    """Same-read/same-write rank-1 patch (the exp-29 class).

    P = w w^T / (w.w). Reads the delta along w and writes back along the same
    direction, so the projected component of the delta is copied from source.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    nn = float(w @ w)
    if nn < RW_TOL:
        raise ValueError("same_write_patch needs a nonzero direction")
    return np.outer(w, w) / nn


def oblique_patch(c, w, tol=RW_TOL):
    """Rank-1 oblique patch: read with covector c, write along vector w.

    P = c w^T / (c.w). Constrained so the write moves the read value exactly:
    after the patch the read functional c equals its source value. Raises
    ``SingularReadWrite`` when c.w ~= 0 (the write is read-blind)."""
    c = np.asarray(c, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    cw = float(c @ w)
    if abs(cw) < tol:
        raise SingularReadWrite(f"c.w={cw:.3e} below tol={tol:.1e}")
    return np.outer(c, w) / cw


def oblique_compose(C, W, tol=RW_TOL):
    """Rank-k oblique composition: read covectors C (d,k), writes W (d,k).

    P = C (C^T W)^{-T} W^T. Sets every read functional in C to its source
    value simultaneously. Raises ``SingularReadWrite`` when C^T W is singular
    (no joint write can realize the reads)."""
    C = np.asarray(C, dtype=float)
    W = np.asarray(W, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    if W.ndim == 1:
        W = W[:, None]
    if C.shape != W.shape:
        raise ValueError(f"C {C.shape} and W {W.shape} must match")
    M = C.T @ W
    detM = float(np.linalg.det(M))
    if abs(detM) < tol:
        raise SingularReadWrite(f"det(C^T W)={detM:.3e} below tol={tol:.1e}")
    return C @ np.linalg.inv(M).T @ W.T


def scaled_patch(P, alpha):
    """Strength-scaled patch: alpha * P. alpha=0 is an exact no-op (the delta
    update vanishes), alpha=1 is the base patch; the dose curve is swept over
    alpha by the caller."""
    return float(alpha) * np.asarray(P, dtype=float)


def fixed_read_write_search(c, write_candidates, score_fn, tol=RW_TOL):
    """Hold a read covector c fixed; score each candidate write direction.

    ``write_candidates`` maps name -> write vector. For each, build the rank-1
    oblique patch (read c, write w) and call ``score_fn(P) -> float`` (higher is
    better; the caller closes over the model). Read-blind candidates (c.w ~= 0)
    are recorded as singular and skipped, not silently dropped. Returns
    ``(best_name, best_score, results)`` where ``results`` is name -> score
    (NaN for singular candidates)."""
    results = {}
    best_name, best_score = None, -np.inf
    for name, w in write_candidates.items():
        try:
            P = oblique_patch(c, w, tol=tol)
        except SingularReadWrite:
            results[name] = float("nan")
            continue
        s = float(score_fn(P))
        results[name] = s
        if s > best_score:
            best_name, best_score = name, s
    return best_name, best_score, results


# ---------------------------------------------------------------------------
# write-source constructors
# ---------------------------------------------------------------------------

def affine_readout(X, y, lam):
    """Ridge affine readout y ~ (X - mean) . w + b on centered features.

    Returns ``(w, b)``. This is the fitted predicate readout used by exp 29;
    its direction is the affine-read write source."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, ym = X.mean(0), float(y.mean())
    Xc = X - mu
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, float(ym - mu @ w)


def affine_read_direction(X, y, lam):
    """Unit affine-read write source: the normalized ridge readout direction."""
    w, _ = affine_readout(X, y, lam)
    return unit(w)


def core_directions(Q):
    """Core/PCA-basis write sources: the columns of an orthonormal basis Q
    (d,k) as a list of unit write directions."""
    Q = np.asarray(Q, dtype=float)
    return [Q[:, j] for j in range(Q.shape[1])]


def delta_direction(deltas, kind="mean"):
    """Source-target residual-delta write source.

    ``deltas`` is an (n,d) array of (source - target) residuals. ``kind='mean'``
    returns the unit mean delta; ``kind='top'`` returns the top right singular
    vector (dominant delta axis)."""
    D = np.asarray(deltas, dtype=float)
    if kind == "mean":
        return unit(D.mean(0))
    if kind == "top":
        _, _, Vt = np.linalg.svd(D - D.mean(0, keepdims=True),
                                 full_matrices=False)
        return unit(Vt[0])
    raise ValueError(f"unknown delta kind {kind!r}")


def random_direction(rng, d):
    """Random unit write/read direction."""
    return unit(rng.standard_normal(d))


# ---------------------------------------------------------------------------
# predicate-control scorer (factored from exp 29)
# ---------------------------------------------------------------------------

def predicate_room(p_un, p_src, p_full):
    """Full-patch predicate room: MSE(p_un,p_src) - MSE(p_full,p_src).

    This is the closure denominator. room > ROOM_TOL means the full/reference
    patch moves the predicate enough toward source for a rank-1 closure score
    to be interpretable; room <= ROOM_TOL is NO_PATCH_ROOM."""
    p_un, p_src, p_full = map(np.asarray, (p_un, p_src, p_full))
    d0 = float(((p_un - p_src) ** 2).mean())
    dfull = float(((p_full - p_src) ** 2).mean())
    return d0 - dfull


def predicate_control(p_un, p_src, p_P, p_full):
    """Fraction of the predicate gap to source closed by patch P.

    c(P) = [MSE(p_un,p_src) - MSE(p_P,p_src)] / room. Returns NaN when there is
    no full-patch room (the geometry of P cannot be interpreted). By
    construction c(full) ~= 1 and c(no-op) ~= 0."""
    p_un, p_src, p_P, p_full = map(np.asarray, (p_un, p_src, p_P, p_full))
    denom = predicate_room(p_un, p_src, p_full)
    if denom <= ROOM_TOL:
        return float("nan")
    d0 = float(((p_un - p_src) ** 2).mean())
    dP = float(((p_P - p_src) ** 2).mean())
    return (d0 - dP) / denom


def endpoint_audit(p_un_model, p_tgt_exact, p_src_model, p_src_exact):
    """Observable/exact endpoint drift: max mean-abs gap over both endpoints.

    oe = max( mean|p_un_model - p_tgt_exact|, mean|p_src_model - p_src_exact| ).
    Audits whether the observable predicate endpoints used by the intervention
    are calibrated to exact predicate truth. It says nothing about an exact
    truth for the off-manifold patched distribution."""
    oe_tgt = float(np.abs(np.asarray(p_un_model) - np.asarray(p_tgt_exact)).mean())
    oe_src = float(np.abs(np.asarray(p_src_model) - np.asarray(p_src_exact)).mean())
    return max(oe_tgt, oe_src)


@dataclass(frozen=True)
class ControlReport:
    """Bundle of the predicate-control diagnostics for one (target, family)."""

    room: float
    control: float          # c(P) for the tested patch
    control_random: float   # random-write floor at matched class
    control_full: float     # full/reference ceiling (~1 when room finite)
    endpoint_drift: float

    @property
    def has_room(self):
        return self.room > ROOM_TOL


# ---------------------------------------------------------------------------
# report schema (stable intervention-family table)
# ---------------------------------------------------------------------------

# The INTERVENTION_CLASS_BENCHMARK.md "Standard Measurement Table" columns.
STANDARD_COLUMNS = (
    "target", "read", "write", "patch_point", "room",
    "control", "specificity", "exact_audit", "transfer", "failure_branch",
)


def format_intervention_table(rows, columns=STANDARD_COLUMNS, missing="-"):
    """Render a stable aligned table for intervention-family comparisons.

    ``rows`` is a list of dicts keyed by column name; absent keys print as
    ``missing``. Numeric values are formatted to 3 decimals. Returns the table
    as a string so callers print one stable schema instead of ad-hoc prose."""
    def fmt(v):
        if v is None:
            return missing
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.3f}"
        return str(v)

    cells = [[fmt(r.get(c)) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c)
              for i, c in enumerate(columns)]
    line = lambda vals: "  ".join(v.ljust(widths[i]) for i, v in enumerate(vals))
    out = [line(list(columns))]
    out += [line(row) for row in cells]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# known-answer self-tests
# ---------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(0)
    d = 8

    # (1) oblique rank-1 sets the read functional exactly.
    c, w = rng.standard_normal(d), rng.standard_normal(d)
    P = oblique_patch(c, w)
    delta = rng.standard_normal((5, d))
    upd = delta @ P
    assert np.allclose(c @ upd.T, c @ delta.T), "rank-1 read not set exactly"

    # same-read/same-write is the c=w reduction of the oblique patch.
    assert np.allclose(same_write_patch(w), oblique_patch(w, w)), "swsw != c=w"
    # and it copies the along-w component of the delta.
    upd_w = delta @ same_write_patch(w)
    wu = unit(w)
    assert np.allclose(upd_w @ wu, delta @ wu), "swsw does not copy w-component"

    # (2a) singular rank-1: c orthogonal to w is read-blind.
    c0 = c - (c @ wu) * wu              # remove the w-component of c
    try:
        oblique_patch(c0, w)
        raise AssertionError("singular rank-1 not caught")
    except SingularReadWrite:
        pass

    # (1b) + (2b) rank-k oblique sets all reads; singular C^T W is caught.
    C = rng.standard_normal((d, 3))
    W = rng.standard_normal((d, 3))
    Pk = oblique_compose(C, W)
    updk = delta @ Pk
    assert np.allclose(C.T @ updk.T, C.T @ delta.T), "rank-k reads not set"
    Wsing = W.copy()
    Wsing[:, 1] = Wsing[:, 0]          # make C^T W rank-deficient generically
    Csing = C.copy()
    Csing[:, 1] = Csing[:, 0]
    try:
        oblique_compose(Csing, Wsing)
        raise AssertionError("singular rank-k not caught")
    except SingularReadWrite:
        pass

    # (3) strength zero is an exact no-op; strength one is the base patch.
    assert np.allclose(scaled_patch(P, 0.0), 0.0), "alpha=0 not a no-op"
    assert np.allclose(delta @ scaled_patch(P, 0.0), 0.0), "alpha=0 moves delta"
    assert np.allclose(scaled_patch(P, 1.0), P), "alpha=1 != base patch"

    # (4) full-patch room: no room -> control is NaN; with room, full ~1, no-op ~0.
    p_un = np.array([0.2, 0.8, 0.5, 0.1])
    p_src = np.array([0.9, 0.1, 0.4, 0.7])
    assert np.isnan(predicate_control(p_un, p_src, p_un, p_un)), \
        "no-room control not NaN"          # p_full == p_un -> room 0
    p_full = p_src.copy()                   # full patch reaches source
    assert predicate_room(p_un, p_src, p_full) > ROOM_TOL
    assert abs(predicate_control(p_un, p_src, p_full, p_full) - 1.0) < 1e-9, \
        "full patch not ~1"
    assert abs(predicate_control(p_un, p_src, p_un, p_full) - 0.0) < 1e-9, \
        "no-op not ~0"

    # endpoint audit takes the worse of the two endpoint drifts.
    oe = endpoint_audit([0.1, 0.2], [0.1, 0.2], [0.5, 0.5], [0.5, 0.6])
    assert abs(oe - 0.05) < 1e-12, "endpoint_audit wrong"

    # write-source constructors return unit directions; affine readout recovers
    # a planted linear target.
    Xr = rng.standard_normal((200, d))
    wtrue = rng.standard_normal(d)
    yr = Xr @ wtrue + 3.0
    wfit, bfit = affine_readout(Xr, yr, 1e-6)
    assert np.allclose(wfit, wtrue, atol=1e-2), "affine readout off"
    for direction in [affine_read_direction(Xr, yr, 1e-6),
                      delta_direction(rng.standard_normal((10, d))),
                      delta_direction(rng.standard_normal((10, d)), kind="top"),
                      random_direction(rng, d)]:
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-9, "non-unit source"
    assert len(core_directions(np.eye(d)[:, :3])) == 3

    # report schema is stable and fills missing cells.
    tbl = format_intervention_table(
        [{"target": "phi1", "write": "same", "control": 0.0, "room": 0.12}])
    assert tbl.splitlines()[0].split()[0] == "target"
    assert "phi1" in tbl and "0.000" in tbl and "-" in tbl

    print("interventions selftest passed: patch API, write sources, "
          "predicate-control scorer, report schema")


if __name__ == "__main__":
    _selftest()
