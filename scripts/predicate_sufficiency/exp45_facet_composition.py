"""
Experiment 45 — facet-factor decodability and composition (type x color)
on colored Dyck-2.

See experiments/45-predicate-sufficiency-composition.md for the registration.
This is the runnable artifact that experiment must be frozen beside.

The question (correlational, probe-class-indexed): on colored Dyck-2, at the
registered probe layer/positions, from which residual directions are the
GATE-NORMALIZED facet conditionals

    psi_facet(x) = E_q[phi_facet](x) / E_q[phi_closes](x)
                 = P_model(facet | the seeded top closes, x)

decodable by a bounded LINEAR probe, and do the two facet FACTORS (type, color)
compose? Two measured axes, in the closes-orthogonal complement r_perp:

  axis 1  marginal separability  angle(w_type0, w_color0) vs SEP_ANGLE.
  axis 2  conjunction availability  dR2 = R2_full - R2_span on psi_both, split
          into in-span / dedicated-linear-outside-span / genuinely-nonlinear by
          R2_full and a kNN gate.

Load-bearing design notes (from the second review):
  * The multiplicative closing gate p_close is divided out of the TARGET by the
    psi ratio, and partialled out of the RESIDUAL (r_perp) before the geometry.
  * dR2 needs NO arithmetic-product correction: for a true direct sum the
    Boolean-AND ceiling caps R2_full and R2_span equally and cancels, so
    dR2 ~ 0 (verified in selftest). dR2 > COMP_GAP therefore means the joint
    needs a direction OUTSIDE the marginal span; the kNN gate then splits a
    dedicated *linear* axis (JOINT_OUTSIDE_SPAN) from a genuinely *nonlinear*
    joint (NOT_LINEARLY_DECODED). "Outside the span" != "higher-order".
  * Thresholds are placed by FROZEN formulas off --calibrate references:
    SEP_ANGLE = floor + ALPHA*(ceiling - floor);  COMP_GAP = mu + KSIG*sigma of
    the direct-sum noise floor. The reference numbers come from burned seed 800.

Reuse posture: imports the predicate layer, processes, model loader, validity
gate, residual centering, chain_probs, and the k* (compression ceiling)
machinery from the library. The facet-VALUE templates + gate, gate-normalization
+ CLOSE_FLOOR selection + w_closes partial-out, the ridge/kNN decode with
sequence-level held-out scoring, the angle/dR2 reducers + kNN-on-psi_both gate,
cell_verdict, and --calibrate are rung-specific (promote to predicates.py only on
a second use).

Device note: heavy forward passes (residuals, chain_probs) run on the model's
accelerator via expcommon.load_model; never CPU for a claim run.
"""

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import predicates as P
from abstraction import center_by_position
from battery import Refs, cegar_loop
from discover import PairSet
from expcommon import load_model, validity_gate
from midstream import chain_probs, stream_to
from processes import PROCESSES
import torch

# ----- registration constants ------------------------------------------------
REGISTERED_CFG = {"process": "colored_dyck2", "seq_len": 32, "layers": 4,
                  "d_model": 64, "m": 3}
M = 3                                    # predicate horizon (decision-6 deviation;
#                                          justified by calibration: tops close fast)
MM = 3                                   # k* compression-ceiling horizon
TS = (8, 12, 16)                         # determined-ctx read points, pooled
PROBE_LAYER = 2                          # stream entering blocks[2] = after 2 of 4

CLAIM_SEEDS = (801, 802, 803, 804)       # fresh out-of-design; >=3/4 majority
CALIB_SEED = 800                         # --calibrate threshold seed (burned)
BURNED_CKPT_SEED = 777                   # on-disk calibration checkpoint (burned)
BURNED = (BURNED_CKPT_SEED, CALIB_SEED)

EVAL_SEQS = 3000                         # split by SEQUENCE 50/50
LAM = 1e-2                               # ridge on centered residuals
KNN_K = 10                               # present-but-not-affine reference

R2_MIN = 0.50                            # linear-decodable cut (exp-29 precedent)
VAR_MIN = 0.05                           # psi non-vacuity (std across prefixes)
TAU = 0.03                               # pooled-mean |psi - decode| on held-out
CLOSE_FLOOR = 0.30                       # keep prefixes with E_q[phi_closes] >=
OE_BAND = 0.02                           # OBS_DRIFT pooled-mean tolerance

# Composition thresholds: PROVISIONAL numbers, FROZEN placement formulas.
SEP_ANGLE = 45.0                         # = floor + ALPHA*(ceiling - floor)
COMP_GAP = 0.10                          # = mu + KSIG*sigma of the noise floor
ALPHA = 0.5                              # SEP_ANGLE interpolation (frozen)
KSIG = 3.0                               # COMP_GAP margin in sigmas (frozen)

SEED_MAJORITY = 3                        # of 4

# k* (compression ceiling) discovery, like exp-29.
PAIRS_DISC, PAIR_POOL = 320, 900
EPS, EPS_DROP, K_MAX, MAX_DIM = 0.05, 0.01, 10, 8


# ----- rung-specific predicate templates (facet-VALUE + gate) ----------------
# Seeded on the prefix top (local stack depth=1). The popping close is the one
# that first returns the seeded frame to depth 0. Colored-Dyck matching is
# grammar-forced, so on the true model the popping close carries the seeded
# top's (type,color) -- hence on DETERMINED-ctx prefixes psi_facet collapses to
# a near-binary label of the top's facet (review finding 6). Prefix-free given a
# non-empty seeded top (the caller restricts to determined, non-empty ctx), so
# the mask does not depend on the top's value -- the value check is fixed.

def tmpl_next_close_is(facet=None, value=None, V=8):
    """Facet-VALUE predicate (facet in {"type","color"}, value in {0,1}) or the
    gate (facet=None -> "the seeded top is popped in-window, any facet")."""
    is_close, decode = P._dyck_vocab(V)

    def phi(cont):
        depth = 1                              # the seeded prefix top
        for t in cont:
            if not is_close(t):
                depth += 1
                continue
            depth -= 1
            if depth == 0:                     # popped the seeded top
                if facet is None:
                    return 1.0                 # gate: it closes in-window
                ty, co = decode(t)
                val = ty if facet == "type" else co
                return float(val == value)
        return 0.0                             # top not closed within the window
    return phi


def facet_masks(V, m):
    """The registered facet-value suite + the gate, as length-V**m masks."""
    type0 = P.graded_mask(tmpl_next_close_is("type", 0, V), V, m)
    color0 = P.graded_mask(tmpl_next_close_is("color", 0, V), V, m)
    return {
        "phi_type0": type0,
        "phi_color0": color0,
        "phi_both00": P.mask_and(type0, color0),
        "phi_closes": P.graded_mask(tmpl_next_close_is(None, None, V), V, m),
    }


def matches_masks(V, m):
    """Degeneracy control: matches_{type,color,both} must coincide (forced
    matching). They depend on the top's value, so they are built per ctx."""
    return {f: P.tmpl_next_close_matches(f, V) for f in ("type", "color", "both")}


# ----- small reducers --------------------------------------------------------

def r2(y, yhat):
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0


def ridge_fit(X, y, lam=LAM):
    mu, ym = X.mean(0), y.mean()
    Xc = X - mu
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, float(ym - mu @ w)


def knn_r2(Xtr, ytr, Xte, yte, k=KNN_K):
    d2 = ((Xte ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None, :]
          - 2 * Xte @ Xtr.T)
    nn = np.argpartition(d2, min(k, Xtr.shape[0] - 1), axis=1)[:, :k]
    return r2(yte, ytr[nn].mean(1))


def angle_deg(u, v):
    """Angle between two vectors in degrees (0..90 after sign-folding)."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    c = abs(float(u @ v) / (nu * nv))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def partial_out(R, wdir):
    """Remove the component of every row along unit direction wdir."""
    return R - np.outer(R @ wdir, wdir)


def unit(w):
    n = np.linalg.norm(w)
    return w / n if n > 1e-12 else w


def decode_metrics(Rtr, Rte, ytr, yte):
    """Linear + kNN held-out R2 and pooled-mean decode error for one target."""
    w, b = ridge_fit(Rtr, ytr)
    pred = Rte @ w + b
    return {"w": w, "b": b, "lin_r2": r2(yte, pred),
            "knn_r2": knn_r2(Rtr, ytr, Rte, yte),
            "tau": float(np.abs(yte - pred).mean())}


def delta_r2(Rtr, Rte, wt, wc, ytr, yte):
    """R2_span (psi_both from the 2-D marginal span), R2_full (from all of R),
    and dR2 = R2_full - R2_span. Span features are projections onto the two
    marginal decode directions; the AND ceiling caps both fits equally."""
    Ftr = np.c_[Rtr @ wt, Rtr @ wc]
    Fte = np.c_[Rte @ wt, Rte @ wc]
    ws, bs = ridge_fit(Ftr, ytr)
    r2_span = r2(yte, Fte @ ws + bs)
    wf, bf = ridge_fit(Rtr, ytr)
    r2_full = r2(yte, Rte @ wf + bf)
    return r2_span, r2_full, r2_full - r2_span, wf


# ----- the heavy model-touching helpers (chunked, device-aware) --------------

def residuals_at(model, X, layer, batch=512):
    """Residual stream entering blocks[layer] for all sequences, chunked."""
    out = []
    for i in range(0, len(X), batch):
        S = stream_to(model, torch.from_numpy(X[i:i + batch]), layer)
        out.append(S.double().numpy())
    return np.concatenate(out, axis=0)               # (N, L, d)


def model_q(model, X_pref, t, m, V, layer, pref_batch=64):
    """Model completion joint q(w_1..w_m) at position t for each prefix row of
    X_pref, by splicing every continuation at t+1..t+m. Chunked over prefixes so
    the (b, V**m, L) splice array stays bounded."""
    conts = np.array(list(product(range(V), repeat=m)), dtype=np.int64)
    C = len(conts)
    out = np.empty((len(X_pref), C))
    for i in range(0, len(X_pref), pref_batch):
        chunk = X_pref[i:i + pref_batch]
        xc = np.repeat(chunk[:, None, :], C, axis=1).copy()
        xc[:, :, t + 1:t + 1 + m] = conts[None, :, :]
        q, _ = chain_probs(model, xc, layer, None, t, m, V)
        out[i:i + len(chunk)] = q
    return out                                        # (n, V**m)


def compute_kstar(model, proc, cfg, seed):
    """Full-m-gram sufficient-subspace dimension k* (the compression ceiling the
    rank-1 readouts are measured against), via the exp-6 CEGAR loop. Reported,
    not a verdict input; guarded so a failure here never blocks the cells."""
    d = cfg["d_model"]
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + 111, PAIR_POOL,
                   layer=PROBE_LAYER, ts=TS)
    refs = Refs(disc, model, d, MM)
    k_raw, _, _ = cegar_loop(model, disc, refs, d, EPS, K_MAX, MM,
                             eps_drop=EPS_DROP)
    return int(np.clip(k_raw, 1, MAX_DIM))


# ----- gather one seed's prefixes (pooled over positions) --------------------

def gather(model, proc, cfg, seed):
    """Returns a dict of pooled, determined-ctx, gate-passing prefixes:
    residual r, gate-normalized psi_{type,color,both}, the raw E_q[phi]
    (un-normalized, for the entangled floor + OBS_DRIFT), true facet labels,
    position, sequence id, and the exact-estimator audit values."""
    V, d, L = proc.V, cfg["d_model"], cfg["seq_len"]
    rng = np.random.default_rng(seed)
    X = proc.sample(EVAL_SEQS, L, rng)
    R_all = residuals_at(model, X, PROBE_LAYER)              # (N, L, d)
    beliefs = np.stack([proc.beliefs_along(row) for row in X])

    fmask = facet_masks(V, M)
    cols = {k: [] for k in ("r", "psi_type", "psi_color", "psi_both",
                            "eq_type", "eq_color", "eq_both", "eq_closes",
                            "lab_type", "lab_color", "pos", "seq",
                            "ex_type", "ex_color", "ex_both", "ex_closes")}
    # degeneracy control accumulates pooled means of matches_* (exact).
    mctl = matches_masks(V, M)

    deg = {f: [] for f in mctl}
    for t in TS:
        ctxs, _ = P.ctx_along(proc, beliefs[:, t])
        keep_idx = [i for i, c in enumerate(ctxs)
                    if c is not P.UNDETERMINED and c is not None]
        if not keep_idx:
            continue
        keep = np.array(keep_idx)
        q = model_q(model, X[keep], t, M, V, PROBE_LAYER)    # (nk, V**m)
        eq_closes = q @ fmask["phi_closes"]
        sel = eq_closes >= CLOSE_FLOOR                       # gate selection
        if not sel.any():
            continue
        keep, q, eq_closes = keep[sel], q[sel], eq_closes[sel]
        eq_type = q @ fmask["phi_type0"]
        eq_color = q @ fmask["phi_color0"]
        eq_both = q @ fmask["phi_both00"]
        bel = beliefs[keep, t]
        # exact estimator for the OBS_DRIFT audit (facet masks are ctx-
        # independent, so plain eq_exact is exact here).
        ex_closes = P.eq_exact(bel, fmask["phi_closes"], proc, M)
        ex_type = P.eq_exact(bel, fmask["phi_type0"], proc, M)
        ex_color = P.eq_exact(bel, fmask["phi_color0"], proc, M)
        ex_both = P.eq_exact(bel, fmask["phi_both00"], proc, M)
        tops = [proc.states[np.argmax(b)][-1] for b in bel]  # determined top

        cols["r"].append(R_all[keep, t])
        cols["psi_type"].append(eq_type / eq_closes)
        cols["psi_color"].append(eq_color / eq_closes)
        cols["psi_both"].append(eq_both / eq_closes)
        cols["eq_type"].append(eq_type)
        cols["eq_color"].append(eq_color)
        cols["eq_both"].append(eq_both)
        cols["eq_closes"].append(eq_closes)
        cols["lab_type"].append(np.array([float(tp[0] == 0) for tp in tops]))
        cols["lab_color"].append(np.array([float(tp[1] == 0) for tp in tops]))
        cols["pos"].append(np.full(len(keep), t))
        cols["seq"].append(keep)
        cols["ex_type"].append(ex_type)
        cols["ex_color"].append(ex_color)
        cols["ex_both"].append(ex_both)
        cols["ex_closes"].append(ex_closes)
        # degeneracy control: exact matches_* on the determined tops.
        for f, fn in mctl.items():
            vals, _ = P.eq_exact_seeded(proc, bel, fn, M)
            deg[f].append(vals)

    out = {k: np.concatenate(v) for k, v in cols.items() if v}
    out["deg"] = {f: float(np.concatenate(v).mean()) for f, v in deg.items()
                  if v}
    return out


# ----- per-seed measurement + verdict ----------------------------------------

def measure_seed(g, seed):
    """All cell measurements for one seed from its gathered prefixes."""
    pos, seq = g["pos"], g["seq"]
    Rc = center_by_position(g["r"], pos, np.ones(len(pos), dtype=bool))

    # sequence-level held-out split (no prefix leakage across positions).
    rng = np.random.default_rng(seed)
    uniq = np.unique(seq)
    rng.shuffle(uniq)
    train_seqs = set(uniq[:len(uniq) // 2].tolist())
    tr = np.array([s in train_seqs for s in seq])
    te = ~tr

    # gate direction (un-normalized p_close) fit on TRAIN, partialled out.
    w_closes, _ = ridge_fit(Rc[tr], g["eq_closes"][tr])
    wc_hat = unit(w_closes)
    Rp = partial_out(Rc, wc_hat)                         # r_perp

    m = {"n": int(len(pos)), "n_tr": int(tr.sum()), "n_te": int(te.sum()),
         "deg": g["deg"], "p_close_mean": float(g["eq_closes"].mean())}

    # OBS_DRIFT audit (observable vs exact), pooled-mean per predicate.
    drift = max(float(np.abs(g["eq_" + f] - g["ex_" + f]).mean())
                for f in ("type", "color", "both", "closes"))
    m["drift"] = drift

    # marginal decodes (verdict in r_perp; full-r reported as companion).
    facets = {}
    for f in ("type", "color"):
        y = g["psi_" + f]
        dm = decode_metrics(Rp[tr], Rp[te], y[tr], y[te])
        dm["std"] = float(y.std())
        dm["lin_r2_full"] = decode_metrics(Rc[tr], Rc[te], y[tr], y[te])["lin_r2"]
        facets[f] = dm
    m["facets"] = facets

    # axis 1: marginal-separability angle in r_perp.
    m["angle"] = angle_deg(facets["type"]["w"], facets["color"]["w"])
    m["angle_full_r"] = angle_deg(
        decode_metrics(Rc[tr], Rc[te], g["psi_type"][tr],
                       g["psi_type"][te])["w"],
        decode_metrics(Rc[tr], Rc[te], g["psi_color"][tr],
                       g["psi_color"][te])["w"])

    # axis 2: conjunction availability on psi_both, in r_perp.
    yb = g["psi_both"]
    r2_span, r2_full, dr2, _ = delta_r2(
        Rp[tr], Rp[te], facets["type"]["w"], facets["color"]["w"],
        yb[tr], yb[te])
    m["both"] = {"std": float(yb.std()), "r2_span": r2_span,
                 "r2_full": r2_full, "dr2": dr2,
                 "knn_r2": knn_r2(Rp[tr], yb[tr], Rp[te], yb[te]),
                 "tau": float(np.abs(yb[te] - (Rp[te] @ ridge_fit(
                     Rp[tr], yb[tr])[0] + ridge_fit(Rp[tr], yb[tr])[1])).mean())}

    # ground-truth separable-ceiling pair angle (eval-only, reported).
    gt_t, _ = ridge_fit(Rp[tr], g["lab_type"][tr])
    gt_c, _ = ridge_fit(Rp[tr], g["lab_color"][tr])
    m["gt_ceiling"] = angle_deg(gt_t, gt_c)

    # entangled-angle floor: un-normalized facets in RAW r (gate left in).
    ef_t, _ = ridge_fit(Rc[tr], g["eq_type"][tr])
    ef_c, _ = ridge_fit(Rc[tr], g["eq_color"][tr])
    m["entangled_floor"] = angle_deg(ef_t, ef_c)

    # premise-audit (descriptive, no verdict weight): product of recovered
    # marginals vs psi_both. ~0.99 confirms grammar-faithful type _|_ color.
    pt = Rp[te] @ facets["type"]["w"] + facets["type"]["b"]
    pc = Rp[te] @ facets["color"]["w"] + facets["color"]["b"]
    m["premise_audit_r2"] = r2(yb[te], pt * pc)
    return m


def decodable(fm):
    """A facet psi is linearly decodable iff R2 >= R2_MIN and the pooled-mean
    decode error clears TAU."""
    return fm["lin_r2"] >= R2_MIN and fm["tau"] <= TAU


def cell_verdict(m, sep_angle=SEP_ANGLE, comp_gap=COMP_GAP):
    """One headline cell from a seed's measurements. Guards (validity, drift,
    degeneracy) are applied by the caller as HARNESS_FAIL before this runs."""
    # vacuity / not-affinely-decodable on the MARGINALS first.
    for f in ("type", "color"):
        fm = m["facets"][f]
        if fm["std"] < VAR_MIN:
            return "BASELINE_VACUOUS"
        if not decodable(fm):
            return "NOT_LINEARLY_DECODED"     # incl. kNN-present (present-not-affine)
    # both marginals decoded -> composition.
    if m["angle"] < sep_angle:
        return "ENTANGLED_FACTORS"
    b = m["both"]
    if b["std"] < VAR_MIN:
        return "BASELINE_VACUOUS"
    if b["r2_full"] < R2_MIN:
        # joint not affinely decodable: genuinely higher-order iff kNN recovers.
        return "NOT_LINEARLY_DECODED"
    if b["dr2"] <= comp_gap:
        return "SEPARABLE_DIRECTSUM"
    return "JOINT_OUTSIDE_SPAN"


def seed_verdict(m, gates, sep_angle=SEP_ANGLE, comp_gap=COMP_GAP):
    """Fold the harness guards in, then the cell."""
    if not gates["validity"]:
        return "HARNESS_FAIL"
    if m["drift"] > OE_BAND:
        return "HARNESS_FAIL"
    deg = list(m["deg"].values())
    if deg and (max(deg) - min(deg)) > 1e-2:        # matches_* must coincide
        return "HARNESS_FAIL"
    return cell_verdict(m, sep_angle, comp_gap)


def aggregate(verdicts):
    counts = {v: verdicts.count(v) for v in set(verdicts)}
    top = max(counts, key=counts.get)
    if counts[top] < SEED_MAJORITY:
        return "SEED_UNSTABLE"
    return top


# ----- calibration (burned seed): emit floors, set thresholds ----------------

def planted_directsum_dr2(n_tr, n_te, d, p_type, p_color, target_marginal_r2,
                          rng):
    """dR2 for a KNOWN direct-sum conjunction: a residual that linearly carries
    the two facet bits on orthogonal axes and NO dedicated joint axis. The AND
    is not in r beyond the marginals, so dR2 is pure finite-sample noise. Noise
    is scaled to hit target_marginal_r2 so the floor matches the data's
    decodability. Returns one dR2 sample."""
    n = n_tr + n_te
    a = (rng.random(n) < p_type).astype(float)
    bcol = (rng.random(n) < p_color).astype(float)
    # noise sigma s.t. held-out R2(bit ~ bit+noise) ~ target: R2 = var/(var+s^2)
    va, vb = a.var(), bcol.var()
    s2a = va * (1 - target_marginal_r2) / max(target_marginal_r2, 1e-6)
    s2b = vb * (1 - target_marginal_r2) / max(target_marginal_r2, 1e-6)
    G = rng.standard_normal((2, d))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    R = (np.c_[a, bcol] @ G
         + np.c_[np.sqrt(s2a) * rng.standard_normal(n),
                 np.sqrt(s2b) * rng.standard_normal(n)] @ G)
    R += 0.01 * rng.standard_normal((n, d))         # isotropic floor
    yboth = a * bcol
    tr = slice(0, n_tr)
    te = slice(n_tr, n)
    wt, _ = ridge_fit(R[tr], a[tr])
    wcl, _ = ridge_fit(R[tr], bcol[tr])
    _, _, dr2, _ = delta_r2(R[tr], R[te], wt, wcl, yboth[tr], yboth[te])
    return dr2


def calibrate(model, proc, cfg, seed, reps=40):
    """Emit the two composition references on the burned seed and set the
    thresholds by the frozen formulas. Returns (sep_angle, comp_gap, detail)."""
    g = gather(model, proc, cfg, seed)
    m = measure_seed(g, seed)
    ceiling, floor = m["gt_ceiling"], m["entangled_floor"]
    sep_angle = floor + ALPHA * (ceiling - floor)

    # direct-sum noise floor matched to (n, d, marginal rates, decodability).
    marg_r2 = float(np.mean([m["facets"]["type"]["lin_r2"],
                             m["facets"]["color"]["lin_r2"]]))
    rng = np.random.default_rng(seed + 4500)
    dr2s = np.array([planted_directsum_dr2(
        m["n_tr"], m["n_te"], cfg["d_model"],
        float(g["lab_type"].mean()), float(g["lab_color"].mean()),
        max(marg_r2, 0.05), rng) for _ in range(reps)])
    mu, sigma = float(dr2s.mean()), float(dr2s.std())
    comp_gap = mu + KSIG * sigma
    detail = {"gt_ceiling": ceiling, "entangled_floor": floor,
              "marginal_r2": marg_r2, "noise_floor_mu": mu,
              "noise_floor_sigma": sigma, "observed_dr2": m["both"]["dr2"],
              "observed_angle": m["angle"],
              "type_tau": m["facets"]["type"]["tau"],
              "color_tau": m["facets"]["color"]["tau"],
              "type_lin_r2": m["facets"]["type"]["lin_r2"],
              "color_lin_r2": m["facets"]["color"]["lin_r2"],
              "both_r2_full": m["both"]["r2_full"],
              "both_r2_span": m["both"]["r2_span"],
              "both_knn_r2": m["both"]["knn_r2"],
              "premise_audit_r2": m["premise_audit_r2"], "drift": m["drift"]}
    return sep_angle, comp_gap, detail


# ----- printing --------------------------------------------------------------

def print_seed(seed, m, kstar, verdict):
    f = m["facets"]
    print(f"[seed {seed}] n={m['n']} (tr {m['n_tr']}/te {m['n_te']}) "
          f"k*={kstar} p_close~{m['p_close_mean']:.2f} drift={m['drift']:.4f}")
    print(f"  degeneracy matches_*={ {k: round(v,3) for k,v in m['deg'].items()} }")
    for name in ("type", "color"):
        d = f[name]
        print(f"  psi_{name:<5} std={d['std']:.3f} linR2(r_perp)={d['lin_r2']:.2f}"
              f" linR2(full)={d['lin_r2_full']:.2f} knnR2={d['knn_r2']:.2f}"
              f" tau={d['tau']:.3f}")
    b = m["both"]
    print(f"  psi_both  std={b['std']:.3f} R2_span={b['r2_span']:.2f} "
          f"R2_full={b['r2_full']:.2f} dR2={b['dr2']:+.3f} knnR2={b['knn_r2']:.2f}")
    print(f"  angle(r_perp)={m['angle']:.1f} (full_r {m['angle_full_r']:.1f}) "
          f"GT_ceiling={m['gt_ceiling']:.1f} entangled_floor={m['entangled_floor']:.1f}")
    print(f"  premise-audit R2(prod-of-marginals)={m['premise_audit_r2']:.3f} "
          f"-> {verdict}")


# ----- self-tests ------------------------------------------------------------

def _planted_residual(kind, n, d, rng, p=0.5, noise=0.01):
    """Synthetic residual with two facet bits + optional joint structure.
    kind in {separable, joint_linear, entangled}. Low default noise so the
    cell-logic self-test does not depend on the (tight) TAU decode gate."""
    a = (rng.random(n) < p).astype(float)
    b = (rng.random(n) < p).astype(float)
    ab = a * b
    G = rng.standard_normal((3, d))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    if kind == "entangled":
        # type and color share residual geometry: plant their axes ~20 deg apart
        # so both stay decodable but angle(w_type, w_color) < SEP_ANGLE.
        g1 = unit(G[0])
        gp = unit(G[1] - (G[1] @ g1) * g1)
        th = np.radians(20.0)
        g2 = np.cos(th) * g1 + np.sin(th) * gp
        R = np.outer(a, g1) + np.outer(b, g2)
    elif kind == "separable":
        R = np.outer(a, G[0]) + np.outer(b, G[1])
    elif kind == "joint_linear":
        R = np.outer(a, G[0]) + np.outer(b, G[1]) + np.outer(ab, G[2])
    else:
        raise ValueError(kind)
    R += noise * rng.standard_normal((n, d))
    return R, {"type": a, "color": b, "both": ab}


def _mk_measure(R, psi, seed):
    """Build a measure_seed-style dict from a planted residual (one position)."""
    n = len(R)
    pos = np.zeros(n, dtype=int)
    seq = np.arange(n)
    g = {"r": R, "pos": pos, "seq": seq,
         "psi_type": psi["type"], "psi_color": psi["color"],
         "psi_both": psi["both"],
         "eq_type": psi["type"], "eq_color": psi["color"],
         "eq_both": psi["both"], "eq_closes": np.ones(n),
         "ex_type": psi["type"], "ex_color": psi["color"],
         "ex_both": psi["both"], "ex_closes": np.ones(n),
         "lab_type": psi["type"], "lab_color": psi["color"],
         "deg": {"type": 0.8, "color": 0.8, "both": 0.8}}
    return measure_seed(g, seed)


def selftest():
    P._selftest()

    # reducers
    assert abs(r2(np.array([1., 2, 3]), np.array([1., 2, 3])) - 1.0) < 1e-12
    assert abs(angle_deg(np.array([1., 0]), np.array([0., 1.])) - 90.0) < 1e-6
    assert angle_deg(np.array([1., 0]), np.array([1., 0.])) < 1e-6

    # facet-VALUE templates + gate (hand-computed; V=8 colored Dyck, opens 0..3,
    # closes 4..7 with type=(c-4)//2, color=(c-4)%2). Seeded top at depth 1.
    gate = tmpl_next_close_is(None, None, 8)
    ty0 = tmpl_next_close_is("type", 0, 8)
    co0 = tmpl_next_close_is("color", 0, 8)
    assert gate((4, 0, 0)) == 1.0 and gate((0, 0, 0)) == 0.0     # close-4 pops top
    assert ty0((4, 0, 0)) == 1.0 and ty0((6, 0, 0)) == 0.0       # 4:type0 6:type1
    assert co0((4, 0, 0)) == 1.0 and co0((5, 0, 0)) == 0.0       # 4:color0 5:color1
    # an open before the popping close: depth 1->2, close pops inner not the top.
    assert gate((0, 4, 0)) == 0.0 and gate((0, 4, 4)) == 1.0
    fm = facet_masks(8, 3)
    assert np.array_equal(fm["phi_both00"],
                          P.mask_and(fm["phi_type0"], fm["phi_color0"]))

    # composition reducers on planted residuals (the verified constructions).
    rng = np.random.default_rng(0)
    N = 4000
    R, psi = _planted_residual("separable", N, 64, rng)
    ms = _mk_measure(R, psi, 1)
    assert abs(ms["both"]["dr2"]) < 0.08, ms["both"]["dr2"]        # dR2 ~ 0
    assert cell_verdict(ms, sep_angle=45.0, comp_gap=0.10) == "SEPARABLE_DIRECTSUM"

    R, psi = _planted_residual("joint_linear", N, 64, rng)
    mj = _mk_measure(R, psi, 1)
    assert mj["both"]["dr2"] > 0.15, mj["both"]["dr2"]            # joint outside span
    assert mj["both"]["r2_full"] >= R2_MIN
    assert cell_verdict(mj, sep_angle=45.0, comp_gap=0.10) == "JOINT_OUTSIDE_SPAN"

    R, psi = _planted_residual("entangled", N, 64, rng)
    me = _mk_measure(R, psi, 1)
    assert me["angle"] < 45.0, me["angle"]
    assert cell_verdict(me, sep_angle=45.0, comp_gap=0.10) == "ENTANGLED_FACTORS"

    # gate-normalization recovers a facet bit from a p_close*bit target: the raw
    # E_q[phi] = p_close*bit is a PRODUCT (not linear in r even when r encodes
    # the bit cleanly), but psi = E_q[phi]/p_close = bit divides p_close out and
    # is linearly recovered (finding-1 exclusion, known answer).
    p_close = 0.3 + 0.6 * rng.random(N)
    bit = (rng.random(N) < 0.5).astype(float)
    gb = unit(rng.standard_normal(64))
    Rb = np.outer(bit, gb) + 0.05 * rng.standard_normal((N, 64))
    eq = p_close * bit
    raw_r2 = r2(eq[N // 2:], Rb[N // 2:] @ ridge_fit(Rb[:N // 2], eq[:N // 2])[0]
                + ridge_fit(Rb[:N // 2], eq[:N // 2])[1])
    psi_bit = eq / p_close
    w, b = ridge_fit(Rb[:N // 2], psi_bit[:N // 2])
    norm_r2 = r2(psi_bit[N // 2:], Rb[N // 2:] @ w + b)
    assert norm_r2 > 0.9 and norm_r2 - raw_r2 > 0.1, (raw_r2, norm_r2)

    # planted direct-sum noise floor is small and positive-ish (finite sample).
    dr2 = planted_directsum_dr2(1500, 1500, 64, 0.5, 0.5, 0.6,
                                np.random.default_rng(3))
    assert abs(dr2) < 0.1, dr2

    # verdict aggregation
    assert aggregate(["SEPARABLE_DIRECTSUM"] * 3 + ["ENTANGLED_FACTORS"]) \
        == "SEPARABLE_DIRECTSUM"
    assert aggregate(["JOINT_OUTSIDE_SPAN", "SEPARABLE_DIRECTSUM",
                      "ENTANGLED_FACTORS", "JOINT_OUTSIDE_SPAN"]) \
        == "SEED_UNSTABLE"

    # harness guards fold into HARNESS_FAIL
    assert seed_verdict(ms, {"validity": False}) == "HARNESS_FAIL"
    bad = dict(ms); bad["drift"] = 0.5
    assert seed_verdict(bad, {"validity": True}) == "HARNESS_FAIL"
    bad2 = dict(ms); bad2["deg"] = {"type": 0.8, "color": 0.5, "both": 0.8}
    assert seed_verdict(bad2, {"validity": True}) == "HARNESS_FAIL"

    print("exp45 selftest passed: facet-value templates + gate (hand-computed), "
          "gate-normalization recovery, composition reducers on planted "
          "separable/joint-linear/entangled residuals (dR2~0 vs >0; the kNN "
          "split), direct-sum noise floor, aggregation, and harness guards.")


# ----- main ------------------------------------------------------------------

def load_cfg(outdir):
    with open(os.path.join(outdir, "config.json")) as f:
        return json.load(f)


def check_cfg(cfg):
    return [(k, cfg.get(k), v) for k, v in REGISTERED_CFG.items()
            if cfg.get(k) != v]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help=f"emit composition floors on burned seed {CALIB_SEED}")
    ap.add_argument("--calib-outdir", default="out/colored_dyck2-calib",
                    help="checkpoint for --calibrate (burned)")
    ap.add_argument("--ckpt-fmt", default="out/colored_dyck2-s{seed}",
                    help="per-claim-seed checkpoint dir format")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    proc = PROCESSES["colored_dyck2"]()

    if args.calibrate:
        cfg = load_cfg(args.calib_outdir)
        mism = check_cfg(cfg)
        if mism:
            print("HALT: wrong calibration checkpoint config:", mism)
            return 1
        model = load_model(args.calib_outdir, cfg, proc)
        print(f"=== exp45 --calibrate (BURNED seed {CALIB_SEED}) ===")
        sep_angle, comp_gap, detail = calibrate(model, proc, cfg, CALIB_SEED)
        for k, v in detail.items():
            print(f"  {k:<18} {v:.4f}")
        print(f"\n  SEP_ANGLE = floor + {ALPHA}*(ceiling-floor) = {sep_angle:.2f} deg")
        print(f"  COMP_GAP  = mu + {KSIG}*sigma                = {comp_gap:.4f}")
        print(f"\n  Freeze these into the constants; seeds {BURNED} are burned, "
              f"not claim seeds.")
        return 0

    # claim run: fresh out-of-design seeds, >=3/4 majority.
    print(f"=== Experiment 45: facet composition | colored_dyck2 | "
          f"layer={PROBE_LAYER} | t in {TS} | seeds {CLAIM_SEEDS} ===")
    print(f"SEP_ANGLE={SEP_ANGLE} COMP_GAP={COMP_GAP} R2_MIN={R2_MIN} "
          f"TAU={TAU} CLOSE_FLOOR={CLOSE_FLOOR}\n")
    for s in BURNED:
        assert s not in CLAIM_SEEDS, f"burned seed {s} in claim seeds"

    verdicts = []
    for seed in CLAIM_SEEDS:
        outdir = args.ckpt_fmt.format(seed=seed)
        cfg = load_cfg(outdir)
        mism = check_cfg(cfg)
        if mism:
            print(f"HALT: wrong checkpoint config for seed {seed}:", mism)
            return 1
        model = load_model(outdir, cfg, proc)
        _, passed = validity_gate(model, proc, cfg, seed)
        g = gather(model, proc, cfg, seed)
        m = measure_seed(g, seed)
        try:
            kstar = compute_kstar(model, proc, cfg, seed)
        except Exception as e:                       # reported, never a verdict
            kstar = f"(k* failed: {e})"
        v = seed_verdict(m, {"validity": passed})
        verdicts.append(v)
        print_seed(seed, m, kstar, v)
        print()

    decision = aggregate(verdicts)
    print(f"verdicts {list(zip(CLAIM_SEEDS, verdicts))}")
    print(f"\nDECISION ({SEED_MAJORITY}/4 majority): {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
