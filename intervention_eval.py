"""
intervention_eval.py — predicate-control evaluation harness (living library).

Promoted from the inline harness first written in exp 30
(scripts/interventions/i1_fixed_read_write_search.py) so the Phase-3
intervention-class experiments (I1', I2, I3, I4) share one live copy of *how a
patch's predicate control is scored*, per the AGENTS.md library-home rule: exp
30 is a frozen record and must not be imported from. The exp-30 script keeps its
own inline copy (frozen, never refactored); this module is the live edge.

Scope of this module: the model-eval harness that turns a patch operator into a
predicate-control number — fast exact ``p_phi`` evaluation under a patch, the
full/no-op/source endpoints with room + exact-audit, the strength dose curve,
the in-place affine read fit, and the specificity scorer. It is *not* the write
search or the verdict partition: write-candidate construction and the verdict
logic stay experiment-local because each experiment varies them.

Oracle discipline (inherited from interventions.py): exact predicate truth
enters only through ``endpoints``' endpoint audit; reads, writes, strengths, and
control scores are observable/model-only.
"""

import numpy as np
import torch

import interventions as IV
import predicates as P
from abstraction import center_by_position

# Eval batch for the q-only completion path (exp-30 value, kept identical).
CHAIN_BATCH = 4096


# ---------------------------------------------------------------------------
# fast exact p_phi under a patch
# ---------------------------------------------------------------------------

def chain_probs_only(model, X_cont, layer, prefix_state, t, m, V,
                     batch=CHAIN_BATCH):
    """Exact m-step continuation probabilities without materializing residuals.

    Execution-preserving fast path (from exp 30): the intervention scorers only
    consume completion probabilities, so this avoids the residual-output
    allocation in ``midstream.chain_probs``. ``prefix_state`` (n, t+1, d) is the
    already-patched prefix to splice in at ``layer``; ``None`` is the unpatched
    run. Returns (n, C)."""
    n, C, L = X_cont.shape
    flat_np = X_cont.reshape(n * C, L)
    ps = None
    if prefix_state is not None:
        ps = prefix_state.repeat_interleave(C, dim=0)
    out = np.empty((n * C,))
    dev = next(model.parameters()).device
    pos_all = torch.arange(L, device=dev)
    if ps is not None:
        ps = ps.to(dev)
    with torch.no_grad():
        for i in range(0, n * C, batch):
            sl = slice(i, min(i + batch, n * C))
            flat = torch.from_numpy(flat_np[sl]).to(dev)
            x = model.tok(flat) + model.pos(pos_all)
            for li, blk in enumerate(model.blocks):
                if li == layer and ps is not None:
                    x = x.clone()
                    x[:, :t + 1] = ps[sl]
                x = blk(x)
            probs = torch.softmax(model.head(model.ln_f(x)), dim=-1)
            rows = torch.arange(sl.stop - sl.start, device=dev)
            q = torch.ones(sl.stop - sl.start, dtype=probs.dtype, device=dev)
            for j in range(m):
                q *= probs[rows, t + j, flat[:, t + 1 + j]]
            out[sl] = q.cpu().double().numpy()
    return out.reshape(n, C)


def run_pphi(ps, model, Ppatch, mask, src_side=False):
    """Exact observable p_phi for a PairSet under patch ``Ppatch``.

    Evaluates only the mask-true continuations (the predicate's support), summed
    to ``p_phi`` per row. ``Ppatch=None`` is the unpatched/target run; a d x d
    operator is applied as ``pref + (pref_src - pref) @ Ppatch`` at the patch
    point, matching ``discover.PairSet.run`` / interventions.py conventions."""
    keep = np.asarray(mask, dtype=bool)
    out = np.empty(ps.n)
    for t, idx in ps.groups:
        Xfull = ps.Xc_src[t] if src_side else ps.Xc_tgt[t]
        X = Xfull[:, keep, :]
        pref = None
        if Ppatch is not None:
            pt = ps.pref_tgt[t].double().numpy()
            delta = ps.pref_src[t].double().numpy() - pt
            pref = torch.from_numpy(pt + delta @ Ppatch).float()
        out[idx] = chain_probs_only(model, X, ps.layer, pref, t, ps.m,
                                    ps.V).sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# endpoints: room + exact audit
# ---------------------------------------------------------------------------

def endpoints(ps, model, proc, mask, d):
    """Unpatched/source/full endpoints, full-patch room, and exact audit.

    Returns a dict with ``p_un, p_src, p_full`` (observable model p_phi),
    ``room`` (interventions.predicate_room), and ``oe`` (endpoint_audit against
    exact predicate truth). The exact endpoints are evaluation-only; they audit
    calibration and never select a direction/write/strength."""
    p_un = run_pphi(ps, model, None, mask)
    p_src = run_pphi(ps, model, None, mask, src_side=True)
    p_full = run_pphi(ps, model, np.eye(d), mask)
    _, _, _, b_tgt, b_src = IV.pairset_residual_frame(ps, d)
    exact_tgt = P.exact_pphi(b_tgt, mask, proc, ps.m)
    exact_src = P.exact_pphi(b_src, mask, proc, ps.m)
    return {
        "p_un": p_un, "p_src": p_src, "p_full": p_full,
        "room": IV.predicate_room(p_un, p_src, p_full),
        "oe": IV.endpoint_audit(p_un, exact_tgt, p_src, exact_src),
    }


# ---------------------------------------------------------------------------
# in-place affine read fit (position-conditioned read)
# ---------------------------------------------------------------------------

def fit_inplace_read(ps, model, mask, m, d, lam, rng):
    """Fit the affine predicate read IN PLACE on a bin, train/test split.

    Per-position-centered residuals -> observable p_phi via ridge affine
    regression on a random train half; ``r2`` scored on the held-out test half
    of the *same* bin (in-place decodability). Returns ``{c, b, r2, std}`` with
    ``c`` the unit read covector. This is the position-conditioned read object:
    a separate in-place fit per bin, never transported across positions (the
    exp-30 single-global-read transport failure is exactly what I1' repairs)."""
    R, _, pos, _, _ = IV.pairset_residual_frame(ps, d)
    Rc = center_by_position(R, pos, np.ones(ps.n, dtype=bool))
    y = run_pphi(ps, model, None, mask)
    perm = rng.permutation(ps.n)
    tr, te = perm[:ps.n // 2], perm[ps.n // 2:]
    w, b = IV.affine_readout(Rc[tr], y[tr], lam)
    return {
        "c": IV.unit(w),
        "b": b,
        "r2": IV.r2_score(y[te], Rc[te] @ w + b),
        "std": float(y.std()),
    }


# ---------------------------------------------------------------------------
# strength dose curve + control scoring
# ---------------------------------------------------------------------------

def dose_curve(control_at_alpha, alphas):
    """Best finite predicate control over a strength grid.

    ``control_at_alpha(alpha) -> float`` returns the closure-fraction control at
    strength ``alpha`` (NaN when undefined). Returns ``{alpha, control, curve}``
    with the strength maximizing finite control (all-NaN -> NaN). Pure (no
    model), so the dose logic is unit-tested without a checkpoint."""
    curve = [(a, float(control_at_alpha(a))) for a in alphas]
    finite = [(s, a) for a, s in curve if np.isfinite(s)]
    if not finite:
        return {"alpha": float("nan"), "control": float("nan"), "curve": curve}
    best_s, best_a = max(finite, key=lambda x: x[0])
    return {"alpha": best_a, "control": best_s, "curve": curve}


def patch_control(ps, model, mask, c, w, alpha, ep):
    """Closure-fraction control of the rank-1 oblique patch (c, w) at strength
    ``alpha``. ``alpha=0`` is the exact no-op (0 with room, else NaN); a
    read-blind (singular) write returns NaN."""
    if alpha == 0.0:
        return 0.0 if ep["room"] > IV.ROOM_TOL else float("nan")
    try:
        base = IV.oblique_patch(c, w)
    except IV.SingularReadWrite:
        return float("nan")
    p_patch = run_pphi(ps, model, IV.scaled_patch(base, alpha), mask)
    return IV.predicate_control(ep["p_un"], ep["p_src"], p_patch, ep["p_full"])


def score_write(ps, model, mask, c, w, ep, alphas):
    """Dose curve of one write ``w`` under fixed read ``c`` on a bin."""
    return dose_curve(
        lambda a: patch_control(ps, model, mask, c, w, a, ep), alphas)


# ---------------------------------------------------------------------------
# specificity (non-target predicate movement)
# ---------------------------------------------------------------------------

def specificity_predicates(masks, target, eps, room_min):
    """Split non-target predicates into specificity-included (full-patch room
    >= ``room_min``) and skipped-low-room. Low-room non-targets are reported as
    controls but excluded from the closure-fraction specificity score, whose
    denominator would amplify tiny absolute marginal changes."""
    included, skipped_low_room = [], []
    for name in masks:
        if name == target:
            continue
        room = eps[name]["room"]
        if np.isfinite(room) and room >= room_min:
            included.append(name)
        else:
            skipped_low_room.append(name)
    return included, skipped_low_room


def specificity(ps, model, masks, target, c, w, alpha, eps, spec_names):
    """Max absolute non-target predicate control under the selected patch.

    Lower is better: a specific intervention moves the target more than it moves
    other registered predicates. Predicate-level only — not a full-distribution
    noninterference theorem."""
    vals = []
    try:
        Pbase = IV.oblique_patch(c, w)
    except IV.SingularReadWrite:
        return float("nan")
    for name in spec_names:
        ep = eps[name]
        p_patch = run_pphi(ps, model, IV.scaled_patch(Pbase, alpha),
                           masks[name])
        ctl = IV.predicate_control(ep["p_un"], ep["p_src"], p_patch,
                                   ep["p_full"])
        if np.isfinite(ctl):
            vals.append(abs(float(ctl)))
    return max(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# known-answer self-tests (the pure pieces; model-eval paths need a checkpoint)
# ---------------------------------------------------------------------------

def _selftest():
    # dose_curve picks the max finite control and tolerates NaNs / all-NaN.
    dc = dose_curve(lambda a: {0.0: 0.0, 0.5: 0.3, 1.0: 0.7,
                               2.0: float("nan")}[a], (0.0, 0.5, 1.0, 2.0))
    assert dc["alpha"] == 1.0 and abs(dc["control"] - 0.7) < 1e-12, dc
    dc_nan = dose_curve(lambda a: float("nan"), (0.5, 1.0))
    assert np.isnan(dc_nan["control"]) and np.isnan(dc_nan["alpha"]), dc_nan

    # specificity_predicates: room floor includes only finite >= floor.
    eps = {
        "target": {"room": 0.10},
        "near_flat": {"room": 0.0004},
        "practical": {"room": 0.02},
        "nan_room": {"room": float("nan")},
    }
    inc, skip = specificity_predicates(eps, "target", eps, room_min=0.01)
    assert inc == ["practical"], (inc, skip)
    assert skip == ["near_flat", "nan_room"], (inc, skip)

    print("intervention_eval selftest passed: dose curve, specificity filter")


if __name__ == "__main__":
    _selftest()
