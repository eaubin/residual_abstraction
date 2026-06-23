"""exp42_readout_mechanism.py — is exp-40's depth->top_type drag curvature or a
depth-conditional readout?

PRE-REGISTRATION script (writeup: experiments/42-readout-mechanism.md). Decides, on the
Dyck-2 checkpoint, WHY steering depth drags top_type — the geometric-vs-representational
fork exp 40 left open and exp-42's guard-and-steer draft could not cleanly separate (a
single-point guard is fit at one operating point and the steer moves off it). Two pre-checks
motivated this design: #1 (geometry) showed the drag is k-graded and v_depth is ~orthogonal
to the MEAN type direction; #2 (this script's seed-700 design run) showed the drag is
curvature-dominated with a modest depth-conditional readout rotation. This is the claim-grade,
4-seed test of that pattern.

Two mechanisms, measured DIRECTLY with the real readout (finite differences of top_type
through chain_probs; NO autodiff — that is deferred to its own review):
  (curvature / GEOMETRIC)       type readout depth-INDEPENDENT; finite-alpha drag is
                                nonlinearity, vanishing as alpha->0 -> separable to first order;
  (depth-conditional / REPR.)   the type READOUT DIRECTION rotates with depth.

NOTE (result review): the rotation cos does NOT cleanly separate these — a fixed nonlinear
readout's gradient also differs between depth clouds, so cos<1 follows from nonlinearity alone
(the slope p independently shows nonlinearity). The rotation is therefore descriptive, not
evidence of representational coupling; CURVATURE_W_ROTATION reads as "curvature, rotation not
distinguished from nonlinearity." See experiments/42-readout-mechanism.md (Reading / caveats).

Two axes per (position, horizon k=lo->hi):
  ROTATION   cos(g_lo, g_hi) of the FD type-readout gradient at depths lo vs hi, gated by
             split-half reliability (both depths' gradients must be reliably directed);
  DRAG SLOPE p = log-log slope of type drag vs alpha over the RISING prefix (drag ~ alpha^p;
             ~1 first-order coupling, >=2 curvature), with a NO_DRAG floor.

Ground-truth discipline: directions/labels from the Dyck parser (depth_triples / stack_labels);
the oracle is not used. Readout = the model's own m=1 top_type observable. Verdict thresholds
and the registered prediction are in the writeup (walled off from this code's adjudication).
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize import (LAYER, apply_additive_steer, depth_triples,  # noqa: E402
                      facet_diff_vector, facet_observable, q_at, read_facet,
                      require_expected_config, stack_labels, transport_fraction)
from midstream import stream_to  # noqa: E402
from processes import PROCESSES  # noqa: E402
from expcommon import load_model  # noqa: E402
from battery import majority_vote  # noqa: E402

# ---- registered scope -----------------------------------------------------
POSITIONS = (8, 12, 16, 20)
HORIZONS = {1: (1, 2), 2: (2, 3)}          # k -> (clean/lo depth, source/hi depth)
DEPTHS = (1, 2, 3)                         # readout-gradient operating-point depths
SEEDS = (700, 701, 702, 703)               # 700 = design pre-check seed; 701-703 the test
N_SEQS = 6000
GRAD_N = 256                               # operating points per depth for the FD gradient
GAP_MIN = 0.10
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)        # ladder overlapping exp-40's drag regime
EPS_FRAC = 0.10                            # FD step = EPS_FRAC * per-coord residual std at t
SEED_MAJORITY = 3                          # >=3/4 seeds for a stable per-k verdict

# ---- registered thresholds (gate cutoffs; principled, NOT tuned to seed 700) ----
REL_MIN = 0.80          # split-half reliability floor: both depths' gradient directions agree
ROT_LO = 0.70           # cos(g_lo,g_hi) <= this (> ~45deg rotation) -> depth-CONDITIONAL
ROT_HI = 0.90           # cos(g_lo,g_hi) >= this (< ~26deg rotation) -> depth-INDEPENDENT
P_LINEAR = 1.5          # drag slope <= this -> a first-order (linear-limit) drag component
DRAG_FLOOR = 0.02       # max drag below this -> NO_DRAG (no coupling to characterize here)

SUBSTANTIVE = ["DEPTH_CONDITIONAL", "FIRST_ORDER", "CURVATURE_W_ROTATION", "GEOMETRIC"]
GUARDS = ["RELIABILITY_FAIL", "NONMONOTONE", "NO_DRAG"]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


def loglog_slope(alphas, ys):
    """Slope p of log(y) ~ p*log(alpha) over finite positive points. nan if < 2."""
    a, y = np.asarray(alphas, float), np.asarray(ys, float)
    keep = np.isfinite(y) & (y > 0) & (a > 0)
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(a[keep]), np.log(y[keep]), 1)[0])


def drag_slope(alphas, drag):
    """log-log drag slope over the RISING prefix [0..argmax drag] — robust to the readout
    saturating / going undefined at the top of the ladder (the t=8 dropoff). nan if the
    rising prefix has < 2 positive points (-> NONMONOTONE)."""
    d = np.asarray(drag, float)
    if not np.isfinite(d).any() or np.nanmax(d) <= 0:
        return float("nan")
    imax = int(np.nanargmax(d))
    return loglog_slope(np.asarray(alphas)[:imax + 1], d[:imax + 1])


def fd_gradient(readout_fn, base, t, eps):
    """Central-difference gradient of the MEAN readout w.r.t. base[:, t, :].
    readout_fn: (resid (n, t+1, d)) -> (values (n,), defined_mask (n,)). Returns g (d,)
    = mean over examples (where defined at both +/-eps) of d<readout>/dx_t,i. Pure finite
    differences on the real readout; no autodiff."""
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
    """Type-readout gradient g (d,) at position t at depth-`depth` operating points (clean
    residuals of up to GRAD_N sequences whose label at t is `depth`, types mixed). Computed
    on two disjoint HALVES so `rel` = cos(g_A, g_B) is split-half reliability (~1 -> direction
    well-estimated; low -> noise-dominated, rotation cos untrustworthy). |g| can be small
    (type is a weak/flat readout at one position) -- `rel`, not |g|, validates the direction.
    Returns (g, rel, n_used)."""
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
    """v_depth (lo->hi diff-in-means, top_type-matched) steered at ALPHAS over [0..t]; returns
    (alphas, type_drag[], depth_transport[], v_depth_t) -- raw mean|delta top_type| and depth
    transport fraction, plus v_depth at position t (for the diagnostic cosines). None if thin."""
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
    for a in ALPHAS:
        j = q_at(model, tok_c, t, m, V,
                 prefix_state=apply_additive_steer(rc_c, v_depth, t, a, all_pos))
        P_d, mP_d = read_facet(j, "depth", V, m, k)
        P_tt, mP_tt = facet_observable(j, "top_type", V, m)
        trans.append(transport_fraction(P_d, C_d, S_d, GAP_MIN, valid=mC_d & mS_d & mP_d))
        msk = mC_tt & mP_tt
        drag.append(float(np.mean(np.abs(P_tt[msk] - C_tt[msk]))) if msk.sum() else float("nan"))
    return ALPHAS, drag, trans, v_depth[t].numpy()


# ---- verdict --------------------------------------------------------------
def cell_verdict(p, cos_rot, rel_lo, rel_hi, max_drag):
    """Per-(position, k) verdict from the rotation cos, the drag slope p, the two split-half
    reliabilities, and the max drag. Gates first (untrustworthy / no coupling), then reads
    the readout-rotation axis, then the drag-mechanism axis."""
    if not (np.isfinite(rel_lo) and np.isfinite(rel_hi) and min(rel_lo, rel_hi) >= REL_MIN):
        return "RELIABILITY_FAIL"
    if not np.isfinite(max_drag) or max_drag < DRAG_FLOOR:
        return "NO_DRAG"                       # no coupling to characterize at this cell
    if not np.isfinite(p):
        return "NONMONOTONE"
    if cos_rot <= ROT_LO:
        return "DEPTH_CONDITIONAL"             # large gradient rotation (NOT cleanly representational; see docstring NOTE)
    if p <= P_LINEAR:
        return "FIRST_ORDER"                   # drag has a linear-limit component
    if cos_rot < ROT_HI:
        return "CURVATURE_W_ROTATION"          # curvature drag + modest gradient rotation (not distinguished from nonlinearity)
    return "GEOMETRIC"                         # curvature drag + depth-independent readout


def reduce_positions(cells):
    """A (seed, k) verdict from its per-position cells: vote over SUBSTANTIVE cells (majority,
    else precedence-first present); if < 2 substantive cells, report the dominant GUARD."""
    sub = [c for c in cells if c in SUBSTANTIVE]
    if len(sub) < 2:
        g = [c for c in cells if c in GUARDS]
        return Counter(g).most_common(1)[0][0] if g else "SEED_UNSTABLE"
    n = len(sub)
    for lab in SUBSTANTIVE:                    # SUBSTANTIVE doubles as the precedence order
        if sub.count(lab) > n / 2:
            return lab
    for lab in SUBSTANTIVE:                    # no majority -> precedence-first present
        if lab in sub:
            return lab


# ---- main path ------------------------------------------------------------
def run(model, proc, cfg, seeds=SEEDS, n_seqs=N_SEQS, positions=POSITIONS):
    m, V = cfg["m"], proc.V
    per_seed = {k: [] for k in HORIZONS}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        Xe = proc.sample(n_seqs, cfg["seq_len"], rng)
        print(f"[seed {seed}]")
        g_pos, rel_pos = {}, {}
        for t in positions:
            g_by_d, rel_by_d = {}, {}
            for d in DEPTHS:
                g, rel, n_used = readout_gradient_at_depth(model, Xe, t, d, V, m, rng)
                if g is not None:
                    g_by_d[d], rel_by_d[d] = g, rel
            g_pos[t], rel_pos[t] = g_by_d, rel_by_d
        for k, (lo, hi) in HORIZONS.items():
            cells = []
            for t in positions:
                g_by_d, rel_by_d = g_pos[t], rel_pos[t]
                sw = small_alpha_sweep(model, Xe, t, k, lo, hi, V, m, rng)
                if lo not in g_by_d or hi not in g_by_d or sw is None:
                    cells.append("RELIABILITY_FAIL"); continue
                _, drag, trans, vd = sw
                cr = cos(g_by_d[lo], g_by_d[hi])
                p = drag_slope(ALPHAS, drag)
                md = float(np.nanmax(drag)) if np.isfinite(drag).any() else float("nan")
                v = cell_verdict(p, cr, rel_by_d[lo], rel_by_d[hi], md)
                cells.append(v)
                print(f"  t={t:2d} k={k} {v:20s} | cos(g_lo,g_hi)={cr:+.3f} "
                      f"rel[{rel_by_d[lo]:+.2f},{rel_by_d[hi]:+.2f}] p={p:5.2f} "
                      f"maxdrag={md:.3f} | cos(vd,g_lo)={cos(vd, g_by_d[lo]):+.2f} "
                      f"cos(vd,g_hi-g_lo)={cos(vd, g_by_d[hi] - g_by_d[lo]):+.2f}")
            kv = reduce_positions(cells)
            per_seed[k].append(kv)
            print(f"  -> k={k} (depth {lo}->{hi}): {kv}   cells={cells}")
        print()

    config = {f"k{k}": majority_vote(per_seed[k], threshold=SEED_MAJORITY,
                                     unstable="SEED_UNSTABLE") for k in HORIZONS}
    print(f"per-seed per-k: {per_seed}")
    print(f"\nCONFIGURATION (>=3/4 seed majority per horizon): {config}")
    return config


def _selftest():
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert abs(cos(e1, e1) - 1.0) < 1e-9 and abs(cos(e1, e2)) < 1e-9

    a = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    assert abs(loglog_slope(a, 0.3 * a) - 1.0) < 1e-6
    assert abs(loglog_slope(a, 0.3 * a ** 2) - 2.0) < 1e-6
    # drag_slope uses the RISING prefix: a quadratic rise then a dropoff -> ~2, not contaminated
    assert abs(drag_slope(a, np.array([0.01, 0.04, 0.16, 0.64, 0.001])) - 2.0) < 0.2
    assert np.isnan(drag_slope(a, np.zeros(5)))                # no rise -> NONMONOTONE

    # fd_gradient on a synthetic linear readout: recovers a vector parallel to w.
    torch.manual_seed(0)
    d, t = 6, 3
    w = torch.randn(d)
    base = torch.randn(8, t + 1, d) * 0.1
    g = fd_gradient(lambda r: (torch.sigmoid(r[:, t] @ w).numpy(), np.ones(len(r), bool)),
                    base, t, eps=1e-3)
    assert cos(g, w.numpy()) > 0.999
    # depth-conditional synthetic: two readout directions w_lo, w_hi -> cos(g_lo,g_hi)=cos(w)
    w_hi = torch.tensor([w[0], -w[1], w[2], w[3], w[4], w[5]])      # a rotation of w
    g2 = fd_gradient(lambda r: (torch.sigmoid(r[:, t] @ w_hi).numpy(), np.ones(len(r), bool)),
                     base, t, eps=1e-3)
    assert abs(cos(g, g2) - cos(w.numpy(), w_hi.numpy())) < 1e-3

    # cell_verdict: the registered logic, known inputs
    assert cell_verdict(p=4.0, cos_rot=0.95, rel_lo=0.95, rel_hi=0.95, max_drag=0.2) == "GEOMETRIC"
    assert cell_verdict(3.3, 0.85, 0.95, 0.95, 0.05) == "CURVATURE_W_ROTATION"
    assert cell_verdict(3.0, 0.60, 0.95, 0.95, 0.2) == "DEPTH_CONDITIONAL"
    assert cell_verdict(1.0, 0.95, 0.95, 0.95, 0.2) == "FIRST_ORDER"
    assert cell_verdict(4.0, 0.95, 0.70, 0.95, 0.2) == "RELIABILITY_FAIL"   # rel below floor
    assert cell_verdict(4.0, 0.95, 0.95, 0.95, 0.001) == "NO_DRAG"          # below drag floor
    assert cell_verdict(float("nan"), 0.95, 0.95, 0.95, 0.2) == "NONMONOTONE"

    # reduce_positions: majority, guard routing, precedence-first on ties
    assert reduce_positions(["GEOMETRIC"] * 3 + ["NO_DRAG"]) == "GEOMETRIC"
    assert reduce_positions(["CURVATURE_W_ROTATION"] * 3 + ["NO_DRAG"]) == "CURVATURE_W_ROTATION"
    assert reduce_positions(["NO_DRAG", "NO_DRAG", "GEOMETRIC"]) == "NO_DRAG"   # <2 substantive
    assert reduce_positions(["GEOMETRIC", "DEPTH_CONDITIONAL"]) == "DEPTH_CONDITIONAL"  # tie->prec
    assert majority_vote(["GEOMETRIC"] * 3 + ["CURVATURE_W_ROTATION"], threshold=3,
                         unstable="SEED_UNSTABLE") == "GEOMETRIC"
    assert majority_vote(["GEOMETRIC", "GEOMETRIC", "CURVATURE_W_ROTATION", "DEPTH_CONDITIONAL"],
                         threshold=3, unstable="SEED_UNSTABLE") == "SEED_UNSTABLE"
    print("exp42 readout-mechanism selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true", help="1 seed, fewer seqs, 2 positions")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest(); return
    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    require_expected_config(cfg)
    proc = PROCESSES[cfg["process"]]()
    model = load_model(args.outdir, cfg, proc)
    dev = next(model.parameters()).device
    print(f"=== Experiment 42: Dyck-2 readout mechanism (curvature vs depth-conditional) | "
          f"L{cfg['layers']} d{cfg['d_model']} | m={cfg['m']} | device={dev} ===")
    print("ROTATION cos(g_lo,g_hi): >=0.90 depth-INDEPENDENT, <=0.70 depth-CONDITIONAL; "
          "DRAG SLOPE p: <=1.5 first-order, higher curvature\n")
    if args.dry:
        print("** DRY runnability check — 1 seed, reduced seqs, 2 positions; NOT the registered run **\n")
        run(model, proc, cfg, seeds=(700,), n_seqs=3000, positions=(12, 16))
    else:
        run(model, proc, cfg)


if __name__ == "__main__":
    main()
