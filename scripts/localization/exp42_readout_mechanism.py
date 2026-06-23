"""exp42_readout_mechanism.py — NON-CLAIM pre-check #2 for exp 42 (guarded steer).

Why this exists. Pre-check #1 (geometry) measured v_depth's ENERGY in the type-DIFFERENCE
subspace and found it high at k=2. But that cannot say WHY steering depth drags top_type,
given v_depth is ~orthogonal to the mean type-readout direction (cos~0 there). Two mechanisms
produce the exp-40 drag, and they are the geometric-vs-representational fork itself:

  (curvature / GEOMETRIC)        the type readout is depth-INDEPENDENT; finite-alpha drag is
                                 nonlinearity that vanishes as alpha->0 -> a removable basis
                                 artifact (a better fixed basis separates the facets);
  (depth-conditional / REPR.)    the type READOUT DIRECTION rotates with depth, so steering
                                 depth moves into a region that reads type differently ->
                                 genuine coupling, removable by NO fixed basis.

Guard-and-steer (with ANY guard -- linear subspace or point-Jacobian) blends these: the guard
is fit at one operating point and the steer moves off it. This pre-check measures the two
mechanisms DIRECTLY, with the real readout estimated by FINITE DIFFERENCES of top_type through
chain_probs (no autodiff -- the exp-42 Jacobian guard's autodiff is deferred to its own review).
NON-CLAIM: no verdict, routes nothing. It tells us whether to redesign the exp-42 discriminator
around depth-conditionality before building the script.

Reports, per (position, horizon k = lo->hi):
  (b) READOUT ROTATION -- cos(g_lo, g_hi) of the type-readout gradient g_d at depths lo vs hi
      (~1 -> depth-independent/geometric; <<1 -> depth-conditional/representational), with
      cos(v_depth, g_lo) / cos(v_depth, g_hi) and cos(v_depth, g_hi - g_lo): does the depth
      steer align with the readout ROTATION (the representational signature) rather than the
      readout itself (which pre-check #1 said it does not)?
  (a) SMALL-ALPHA DRAG -- type drag(alpha) and depth transport(alpha) on a fine alpha-ladder,
      and the log-log slope p (drag ~ alpha^p; p~1 first-order coupling, p~2 pure curvature).

Ground-truth discipline: directions and depth/type labels come from the Dyck parser on observed
tokens (localize.depth_triples / stack_labels); the oracle is not used. Readout = the model's
own m=1 top_type observable. Honesty/scope as the exp-42 writeup.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize import (LAYER, apply_additive_steer, depth_triples,  # noqa: E402
                      facet_diff_vector, facet_observable, q_at, read_facet,
                      require_expected_config, stack_labels, transport_fraction)
from midstream import stream_to  # noqa: E402
from processes import PROCESSES  # noqa: E402
from expcommon import load_model  # noqa: E402

POSITIONS = (8, 12, 16, 20)
HORIZONS = {1: (1, 2), 2: (2, 3)}
DEPTHS = (1, 2, 3)                 # readout-gradient operating-point depths
N_SEQS = 6000
GRAD_N = 256                       # operating points per depth for the FD gradient
GAP_MIN = 0.10
FINE_ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)   # ladder overlapping exp-40's drag regime
                                            # (drag is ~0 below alpha~1; slope needs the regime)
EPS_FRAC = 0.10                    # FD step = EPS_FRAC * per-coord residual std at t


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


def loglog_slope(alphas, ys):
    """Slope p of log(y) ~ p*log(alpha) over finite, positive points (drag ~ alpha^p:
    p~1 first-order, p~2 curvature). nan if < 2 usable points."""
    a = np.asarray(alphas, float)
    y = np.asarray(ys, float)
    keep = np.isfinite(y) & (y > 0) & (a > 0)
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(a[keep]), np.log(y[keep]), 1)[0])


def fd_gradient(readout_fn, base, t, eps):
    """Central-difference gradient of the MEAN readout w.r.t. base[:, t, :].
    readout_fn: (resid (n, t+1, d)) -> (values (n,), defined_mask (n,)). Returns g (d,)
    = mean over examples (where defined at both +/-eps) of d<readout>/dx_t,i. Pure finite
    differences on the real readout; no autodiff. The model path passes a closure over
    q_at + facet_observable; the selftest passes a synthetic linear readout."""
    d = base.shape[2]
    g = np.zeros(d)
    for i in range(d):
        rp = base.clone(); rp[:, t, i] += eps
        rm = base.clone(); rm[:, t, i] -= eps
        vp, mp = readout_fn(rp)
        vm, mm = readout_fn(rm)
        msk = np.asarray(mp, bool) & np.asarray(mm, bool)
        if msk.sum():
            g[i] = float(np.mean((np.asarray(vp, float)[msk] - np.asarray(vm, float)[msk])
                                 / (2.0 * eps)))
    return g


def type_readout_closure(model, tok, t, V, m):
    """readout_fn for fd_gradient: residual patch -> (top_type m=1 observable, mask)."""
    def f(resid):
        return facet_observable(q_at(model, tok, t, m, V, prefix_state=resid), "top_type", V, m)
    return f


def readout_gradient_at_depth(model, Xe, t, depth, V, m, rng):
    """Type-readout gradient g (d,) at position t evaluated at depth-`depth` operating
    points (clean residuals of up to GRAD_N sequences whose label at t is `depth`, both
    types mixed -- the readout direction is depth-conditioned, type-agnostic). Computed on
    two disjoint HALVES so the cross-depth cos can be read against a noise floor: `rel` =
    cos(g_A, g_B) is the split-half reliability (~1 -> g well-estimated; low -> the gradient
    is noise-dominated and the rotation cos below is not trustworthy). Returns (g, rel,
    n_used). |g| can be small (type is a weak/flat readout at a single position) -- `rel`,
    not |g|, says whether the DIRECTION is real."""
    idx = [i for i in range(len(Xe)) if stack_labels(Xe[i], [t], m)[t][0] == depth]
    if len(idx) < 16:
        return None, float("nan"), 0
    idx = np.array(idx)[rng.permutation(len(idx))[:GRAD_N]]
    tok = Xe[idx]
    rc = stream_to(model, torch.from_numpy(tok), LAYER)
    base = rc[:, :t + 1].clone()
    eps = EPS_FRAC * float(base[:, t].std().clamp_min(1e-6))
    h = len(idx) // 2

    def grad(sl):
        return fd_gradient(type_readout_closure(model, tok[sl], t, V, m), base[sl], t, eps)

    gA, gB = grad(slice(0, h)), grad(slice(h, len(idx)))   # disjoint halves (~free: same total fwd)
    return 0.5 * (gA + gB), cos(gA, gB), len(idx)


def small_alpha_sweep(model, Xe, t, k, lo, hi, V, m, rng):
    """v_depth (lo->hi diff-in-means, top_type-matched) steered at FINE_ALPHAS over [0..t];
    returns (alphas, type_drag[], depth_transport[]) -- raw mean|delta top_type| drag and
    depth transport fraction. None if pairs too thin."""
    labels = {i: stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
    cd, chi, _ = depth_triples(labels, lo, hi, rng)
    if len(cd) < 64:
        return None
    cd, chi = cd[:GRAD_N], chi[:GRAD_N]
    tok_c = Xe[cd]
    rc_c = stream_to(model, torch.from_numpy(tok_c), LAYER)
    rc_s = stream_to(model, torch.from_numpy(Xe[chi]), LAYER)
    v_depth = facet_diff_vector(rc_c, rc_s, t)
    all_pos = np.arange(t + 1)
    jc = q_at(model, tok_c, t, m, V)
    js = q_at(model, Xe[chi], t, m, V)
    C_d, mC_d = read_facet(jc, "depth", V, m, k)
    S_d, mS_d = read_facet(js, "depth", V, m, k)
    C_tt, mC_tt = facet_observable(jc, "top_type", V, m)
    drag, trans = [], []
    for a in FINE_ALPHAS:
        j = q_at(model, tok_c, t, m, V,
                 prefix_state=apply_additive_steer(rc_c, v_depth, t, a, all_pos))
        P_d, mP_d = read_facet(j, "depth", V, m, k)
        P_tt, mP_tt = facet_observable(j, "top_type", V, m)
        trans.append(transport_fraction(P_d, C_d, S_d, GAP_MIN, valid=mC_d & mS_d & mP_d))
        msk = mC_tt & mP_tt
        drag.append(float(np.mean(np.abs(P_tt[msk] - C_tt[msk]))) if msk.sum() else float("nan"))
    return FINE_ALPHAS, drag, trans


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
    ap.add_argument("--seed", type=int, default=700)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true", help="2 positions, fewer seqs")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    require_expected_config(cfg)
    proc = PROCESSES[cfg["process"]]()
    model = load_model(args.outdir, cfg, proc)
    m, V = cfg["m"], proc.V
    rng = np.random.default_rng(args.seed)
    n_seqs = 3000 if args.dry else N_SEQS
    positions = (8, 16) if args.dry else POSITIONS
    Xe = proc.sample(n_seqs, cfg["seq_len"], rng)
    dev = next(model.parameters()).device
    print(f"=== exp42 readout-mechanism pre-check (NON-CLAIM) | device={dev} | "
          f"seed={args.seed} n_seqs={n_seqs} ===")
    print("(b) readout rotation: cos(g_lo,g_hi)~1 -> depth-INDEPENDENT (geometric); "
          "<<1 -> depth-CONDITIONAL (representational)")
    print("(a) small-alpha drag slope p: ~1 first-order coupling; ~2 pure curvature\n")

    for t in positions:
        # (b) readout gradients per depth (horizon-independent), then rotation per horizon
        g_by_d = {}
        for d in DEPTHS:
            g, rel, n_used = readout_gradient_at_depth(model, Xe, t, d, V, m, rng)
            if g is not None:
                g_by_d[d] = g
                print(f"t={t:2d} depth={d}: |g|={np.linalg.norm(g):.4f}  "
                      f"split-half rel={rel:+.3f} (n={n_used})")
        # depth steering directions (need labels once)
        labels = {i: stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
        for k, (lo, hi) in HORIZONS.items():
            line = f"t={t:2d} k={k} (depth {lo}->{hi}) | "
            if lo in g_by_d and hi in g_by_d:
                cd, chi, _ = depth_triples(labels, lo, hi, rng)
                if len(cd):
                    rc_c = stream_to(model, torch.from_numpy(Xe[cd[:GRAD_N]]), LAYER)
                    rc_s = stream_to(model, torch.from_numpy(Xe[chi[:GRAD_N]]), LAYER)
                    vd = facet_diff_vector(rc_c, rc_s, t)[t].numpy()
                    g_lo, g_hi = g_by_d[lo], g_by_d[hi]
                    line += (f"cos(g_lo,g_hi)={cos(g_lo, g_hi):+.3f}  "
                             f"cos(vd,g_lo)={cos(vd, g_lo):+.3f}  "
                             f"cos(vd,g_hi)={cos(vd, g_hi):+.3f}  "
                             f"cos(vd,g_hi-g_lo)={cos(vd, g_hi - g_lo):+.3f}  | ")
                else:
                    line += "thin depth pairs | "
            else:
                line += "missing depth gradient | "
            sw = small_alpha_sweep(model, Xe, t, k, lo, hi, V, m, rng)
            if sw is not None:
                _, drag, trans = sw
                p = loglog_slope(FINE_ALPHAS, drag)
                dstr = " ".join(f"{a:g}:{d:.3f}/{tr:+.2f}" for a, d, tr in
                                zip(FINE_ALPHAS, drag, trans))
                line += f"slope p={p:.2f}  drag/transp[a]: {dstr}"
            else:
                line += "thin steer pairs"
            print(line)
        print()


def _selftest():
    # cos / unit
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert abs(cos(e1, e1) - 1.0) < 1e-9 and abs(cos(e1, e2)) < 1e-9
    assert abs(cos(e1, -e1) + 1.0) < 1e-9

    # loglog_slope recovers the exponent: drag = c*alpha^p
    a = np.array([0.125, 0.25, 0.5, 1.0])
    assert abs(loglog_slope(a, 0.3 * a) - 1.0) < 1e-6           # first-order
    assert abs(loglog_slope(a, 0.3 * a ** 2) - 2.0) < 1e-6      # curvature
    assert np.isnan(loglog_slope(a, np.array([0.0, -1.0, np.nan, 0.0])))

    # fd_gradient on a synthetic linear-ish readout: top_type = sigmoid(w . x_t).
    # central difference recovers a vector parallel to w (cos ~ 1).
    torch.manual_seed(0)
    d, t = 6, 3
    w = torch.randn(d)
    base = torch.randn(8, t + 1, d) * 0.1            # near 0 -> well inside the linear regime

    def lin_readout(resid):
        z = resid[:, t] @ w
        return torch.sigmoid(z).numpy(), np.ones(len(resid), dtype=bool)

    g = fd_gradient(lin_readout, base, t, eps=1e-3)
    assert cos(g, w.numpy()) > 0.999, cos(g, w.numpy())
    # a coordinate w has zero weight on -> ~zero gradient component
    w2 = w.clone(); w2[0] = 0.0
    g2 = fd_gradient(lambda r: (torch.sigmoid(r[:, t] @ w2).numpy(),
                                np.ones(len(r), bool)), base, t, eps=1e-3)
    assert abs(g2[0]) < 1e-4 and cos(g2, w2.numpy()) > 0.999
    print("exp42 readout-mechanism pre-check selftest OK")


if __name__ == "__main__":
    main()
