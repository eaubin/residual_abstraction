"""exp40_directional_specificity.py — directional specificity of facet intervention.

PRE-REGISTRATION script (writeup: experiments/40-directional-specificity.md). Decides,
on the Dyck-2 checkpoint, whether the depth- and top_type-carrying residual DIRECTIONS
are separable at fixed (full) spatial support — the interventional dissociation that is
the well-posed pstack/ICB specificity question. 37+38 settled the SPATIAL axis (graded
depth is spread across positions); this rung isolates the DIRECTIONAL axis.

Construct: a rank-1-per-position additive diff-in-means STEERING vector per facet, from
observable-matched pairs (localize.facet_diff_vector / apply_additive_steer):
  - v_depth from top_type-matched depth (lo->hi) pairs -> a PURE-SUM direction;
  - v_type  from depth-matched type (0->1) pairs       -> a PURE-RATIO direction.
Full residual REPLACEMENT is tautological for specificity (the m=1 transport guard
preserves any matched label), so the steer is sub-residual/additive: it adds the facet
direction and leaves the off-target content intact, making cross-facet drag a real
measurement. The (q2,q3) coupling (close-readiness = q2+q3 sum, top_type = q2/(q2+q3)
ratio) is handled by construction: drag is the CROSS-component (does a pure-sum push
move the ratio, or vice versa), read against the other facet's OWN observable — never
facet-vs-total.

Discriminator: the 2x2 matrix (target transport x cross-drag), read at MATCHED target-
transport over an alpha-sweep, vs a matched-norm RANDOM-DIRECTION floor and the full-
replacement CEILING. Thresholds are in-run relative (fraction of own ceiling, margin
over own random floor) — no gating smoke; the rank-1-handle question is the NO_HANDLE
verdict.

Verdicts (substantive 4; rest standard guards):
  DISSOCIATED — both directions move their own facet (>= ceiling fraction, above the
                random-direction floor) and NOT the other (cross-drag <= bound over the
                random floor) at matched transport: the 2x2 is diagonal-dominant.
  CROSS_DRAG  — at least one direction moves its target but ALSO drags the other.
  MIXED       — one direction DISSOCIATED, the other NO_HANDLE (a DRAGS direction
                routes to CROSS_DRAG by precedence, never MIXED).
  NO_HANDLE   — neither direction moves its facet above the random floor: no rank-1
                additive handle (for depth, extends 38's DISTRIBUTED to non-steerable).
  HARNESS_FAIL / OBS_DRIFT / SEED_UNSTABLE — see the verdict block below.

Scope: verdicts are about the matched-difference direction, added at full support, at
matched transport — not a uniqueness claim and not "separable by any intervention".

Ground-truth discipline: pairs are built from the Dyck parser on observed tokens; the
oracle is used only for the endpoint calibration audit (OBS_DRIFT). Honesty and the
claim scope are in the writeup.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize import (LAYER, apply_additive_steer, cr_cond,  # noqa: E402
                      depth_triples, exact_joint, facet_diff_vector,
                      facet_observable, facet_pairs, floor_pairs,
                      make_patched_prefix, q_at, random_matched_direction,
                      require_expected_config, stack_labels, transport_fraction)
from midstream import marginal, stream_to  # noqa: E402
from processes import PROCESSES  # noqa: E402
from expcommon import load_model, validity_gate  # noqa: E402
from battery import majority_vote, first_precedence  # noqa: E402

# ---- registered scope -----------------------------------------------------
POSITIONS = (8, 12, 16, 20)            # L0's interior positions
SEEDS = (700, 701, 702, 703)           # 4 seeds, as L0/L1
N_SEQS = 6000                          # sequences per seed
HORIZONS = {1: (1, 2), 2: (2, 3)}      # k -> (clean/lo depth, source/hi depth)
ALPHAS = (0.5, 1.0, 2.0, 4.0)          # steer-magnitude ladder (v ~ a full diff)
GAP_MIN = 0.10                         # only score pairs with a real target gap
R_RAND = 4                             # matched-norm random-direction draws (averaged)
MIN_PAIRS = 256                        # min pairs per (position, facet) BEFORE fit/eval split
EVAL_CAP = 400                         # cap eval rows per cell (compute budget)
SEED_MAJORITY = 3                      # >=3/4 seeds for a stable verdict

# ---- registered thresholds (gate cutoffs; relative to in-run references) ---
REF_FRAC = 0.50        # target must reach this fraction of its full-replacement ceiling
HANDLE_MARGIN = 0.15   # ...and beat its matched random-direction transport by this (a handle)
DRAG_BOUND = 0.15      # cross-drag may exceed the random-direction drag by at most this
OE_BAND = 0.10         # max target-endpoint estimator-vs-oracle gap (else OBS_DRIFT)
OFF_DEF_MIN = 0.80     # min off-target-readout definedness retained under the steer at
                       # alpha* (else the drag readout is attrited -> OBS_DRIFT)
SELFTEST_FLOOR = 0.15  # a same-facet v_f must move its own facet <= this (~0 transport)

# DISSOCIATED is lowest severity: the headline only when nothing drags or fails.
PRECEDENCE = ["HARNESS_FAIL", "OBS_DRIFT", "SEED_UNSTABLE", "CROSS_DRAG",
              "MIXED", "NO_HANDLE", "DISSOCIATED"]


# ---- facet readouts -------------------------------------------------------
def read_facet(q, facet, V, m, k):
    """(value, defined_mask) for a facet on a completion joint q. depth is the GRADED
    conditional cr_cond at horizon k (38's instrument); top_type is the m=1 type-
    fraction (defined only where close mass clears CLOSE_MASS_MIN)."""
    if facet == "depth":
        val = cr_cond(q, V, m, k)
        return val, np.isfinite(val)
    return facet_observable(q, "top_type", V, m)


def drag_fraction(P, C, gap, valid):
    """Off-target movement |P - C| under the steer, normalized by the off-target
    facet's OWN between-class gap (the move a genuine intervention on it produces).
    There is no source for the off-target (matched pairing), so ANY movement is drag."""
    keep = np.asarray(valid, bool) & np.isfinite(P) & np.isfinite(C)
    if keep.sum() == 0 or not np.isfinite(gap) or gap < 1e-9:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(P, float)[keep] - np.asarray(C, float)[keep])) / gap)


# ---- one direction's alpha-sweep (target trajectory + cross-drag) ---------
def direction_sweep(model, proc, *, v, clean_tok, src_tok, rc_clean, rc_src, t, k,
                    tgt_facet, off_facet, V, m, rng):
    """Sweep alpha for one steering direction v applied to `clean_tok` at FULL support
    [0..t]. Reads the TARGET facet (transport toward `src_tok`'s value) and the
    OFF-target facet (RAW |P-C| drift from clean), plus a matched-norm random-direction
    control at every alpha. Returns the target trajectories, the RAW drag trajectories
    (`_raw_drag` / `_raw_drag_rand`, normalized later by `_renorm_drag` once the off-
    target facet's own gap is known), the full-replacement ceiling, the target gap, and
    the target-endpoint oracle gap (OBS_DRIFT)."""
    all_pos = np.arange(t + 1)
    jc = q_at(model, clean_tok, t, m, V)                       # unpatched clean
    js = q_at(model, src_tok, t, m, V)                         # source (target oracle)
    C_t, mC_t = read_facet(jc, tgt_facet, V, m, k)
    S_t, mS_t = read_facet(js, tgt_facet, V, m, k)
    C_o, mC_o = read_facet(jc, off_facet, V, m, k)             # off-target clean baseline
    tgt_valid = mC_t & mS_t

    # full-replacement ceiling on the SAME eval pairs (38's f_full / 37's m=1 transport)
    ps_full = make_patched_prefix(rc_clean, rc_src, t, all_pos)
    P_full, mP_full = read_facet(q_at(model, clean_tok, t, m, V, prefix_state=ps_full),
                                 tgt_facet, V, m, k)
    ceiling = transport_fraction(P_full, C_t, S_t, GAP_MIN, valid=tgt_valid & mP_full)

    # target-endpoint calibration vs the exact oracle (audit only)
    So, mSo = read_facet(exact_joint(proc, src_tok, t, m), tgt_facet, V, m, k)
    cal = np.isfinite(S_t) & np.isfinite(So) & mS_t & mSo
    oe = float(np.mean(np.abs(S_t[cal] - So[cal]))) if cal.sum() else float("nan")

    n_off = int(mC_o.sum())                                    # off-target rows defined clean
    f_tgt, f_tgt_rand, raw_drag, raw_drag_rand, off_def = [], [], [], [], []
    for a in ALPHAS:
        j = q_at(model, clean_tok, t, m, V,
                 prefix_state=apply_additive_steer(rc_clean, v, t, a, all_pos))
        Pt, mPt = read_facet(j, tgt_facet, V, m, k)
        Po, mPo = read_facet(j, off_facet, V, m, k)
        f_tgt.append(transport_fraction(Pt, C_t, S_t, GAP_MIN, valid=tgt_valid & mPt))
        raw_drag.append(drag_fraction(Po, C_o, 1.0, mC_o & mPo))   # RAW mean|P-C|
        # off-target definedness retained under the steer (attrition guard): of the rows
        # defined clean, the fraction still defined under the steer. A steer that pushes
        # the off-target readout UNDEFINED (rather than moving its value) would otherwise
        # drop those rows from `drag` and look falsely specific.
        off_def.append(float((mC_o & mPo).sum() / n_off) if n_off else float("nan"))
        ftr, rdr = [], []
        for _ in range(R_RAND):
            vr = random_matched_direction(v, rng)
            jr = q_at(model, clean_tok, t, m, V,
                      prefix_state=apply_additive_steer(rc_clean, vr, t, a, all_pos))
            Ptr, mPtr = read_facet(jr, tgt_facet, V, m, k)
            Por, mPor = read_facet(jr, off_facet, V, m, k)
            ftr.append(transport_fraction(Ptr, C_t, S_t, GAP_MIN, valid=tgt_valid & mPtr))
            rdr.append(drag_fraction(Por, C_o, 1.0, mC_o & mPor))   # RAW mean|P-C|
        f_tgt_rand.append(float(np.nanmean(ftr)))
        raw_drag_rand.append(float(np.nanmean(rdr)))
    gap = float(np.mean(np.abs((S_t - C_t)[tgt_valid & (np.abs(S_t - C_t) >= GAP_MIN)]))) \
        if (tgt_valid & (np.abs(S_t - C_t) >= GAP_MIN)).sum() else float("nan")
    return {"alphas": ALPHAS, "f_tgt": f_tgt, "f_tgt_rand": f_tgt_rand,
            "_raw_drag": raw_drag, "_raw_drag_rand": raw_drag_rand, "ceiling": ceiling,
            "off_def": off_def, "oe": oe, "gap": gap}


# ---- verdicts -------------------------------------------------------------
def _nz(x):
    return x if np.isfinite(x) else 0.0


def direction_subverdict(dr):
    """('OBS_DRIFT'|'NO_HANDLE'|'SPECIFIC'|'DRAGS', alpha_index|None) for one direction.
    alpha* = the smallest alpha whose target transport reaches REF_FRAC of the ceiling
    AND beats the matched random direction by HANDLE_MARGIN; drag is read THERE."""
    if not np.isfinite(dr["oe"]) or dr["oe"] > OE_BAND:
        return ("OBS_DRIFT", None)
    ceil = dr["ceiling"]
    if not np.isfinite(ceil) or ceil <= 0:
        return ("NO_HANDLE", None)
    hit = [i for i, (ft, fr) in enumerate(zip(dr["f_tgt"], dr["f_tgt_rand"]))
           if np.isfinite(ft) and ft >= REF_FRAC * ceil and (ft - _nz(fr)) >= HANDLE_MARGIN]
    if not hit:
        return ("NO_HANDLE", None)
    i = hit[0]
    od = dr["off_def"][i]                       # off-target readout attrited at alpha*?
    if not np.isfinite(od) or od < OFF_DEF_MIN:
        return ("OBS_DRIFT", i)                 # drag readout uninterpretable (attrition)
    excess = _nz(dr["f_drag"][i]) - _nz(dr["f_drag_rand"][i])
    return (("SPECIFIC" if excess <= DRAG_BOUND else "DRAGS"), i)


def matrix_verdict(d_sub, t_sub):
    """The 2x2 verdict at one (position, horizon) from the two direction subverdicts."""
    ds, ts = d_sub[0], t_sub[0]
    if ds == "OBS_DRIFT" or ts == "OBS_DRIFT":
        return "OBS_DRIFT"
    if ds == "NO_HANDLE" and ts == "NO_HANDLE":
        return "NO_HANDLE"
    if ds == "DRAGS" or ts == "DRAGS":
        return "CROSS_DRAG"
    if ds == "SPECIFIC" and ts == "SPECIFIC":
        return "DISSOCIATED"
    return "MIXED"            # one direction specific, the other NO_HANDLE


def reduce_positions(verdicts):
    """A horizon's verdict from its per-position matrix verdicts: OBS_DRIFT if any;
    else the position-majority verdict (most-severe wins ties via the check order);
    else MIXED (the typed middle)."""
    if not verdicts:
        return "SEED_UNSTABLE"
    if "OBS_DRIFT" in verdicts:
        return "OBS_DRIFT"
    n = len(verdicts)
    for lab in ("CROSS_DRAG", "DISSOCIATED", "NO_HANDLE"):
        if verdicts.count(lab) > n / 2:
            return lab
    return "MIXED"


# ---- per-(position, horizon) evaluation -----------------------------------
def _split_resid(model, tok, fit, ev):
    """Stream the residual cache once for the union of fit/eval token rows."""
    rc = stream_to(model, torch.from_numpy(tok), LAYER)
    return rc[fit], rc[ev]


def eval_cell(model, proc, Xe, t, k, lo, hi, V, m, rng, min_pairs):
    """Build both steering directions at (position t, horizon k) on a FIT split, then
    score the 2x2 on a held-out EVAL split. Returns (matrix_verdict, d_sub, t_sub,
    d_sweep, t_sweep) or None if pairs are too thin."""
    labels = {i: stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
    cd, chi, _ = depth_triples(labels, lo, hi, rng)           # top_type-matched, depth lo->hi
    ct, cs = facet_pairs(labels, "top_type", rng, len(Xe),    # depth-matched, ORIENTED 0->1
                         oriented=True)                        # (F1: unoriented cancels v_type)
    if min(len(cd), len(ct)) < min_pairs:
        return None
    # fit/eval split (the steering direction is FIT on one half, scored on the other)
    def split(n):
        h = n // 2
        ev = np.arange(h, min(n, h + EVAL_CAP))
        return np.arange(h), ev
    df, de = split(len(cd))
    tf, te = split(len(ct))

    rc_cd_f, rc_cd_e = _split_resid(model, Xe[cd], df, de)
    rc_chi_f, rc_chi_e = _split_resid(model, Xe[chi], df, de)
    rc_ct_f, rc_ct_e = _split_resid(model, Xe[ct], tf, te)
    rc_cs_f, rc_cs_e = _split_resid(model, Xe[cs], tf, te)

    v_depth = facet_diff_vector(rc_cd_f, rc_chi_f, t)         # pure-sum direction
    v_type = facet_diff_vector(rc_ct_f, rc_cs_f, t)           # pure-ratio direction

    # off-target gaps: each direction's target gap is the OTHER's off-target gap.
    # depth direction drags top_type -> normalize by the type gap; vice versa.
    d_sweep = direction_sweep(
        model, proc, v=v_depth, clean_tok=Xe[cd][de], src_tok=Xe[chi][de],
        rc_clean=rc_cd_e, rc_src=rc_chi_e, t=t, k=k, tgt_facet="depth",
        off_facet="top_type", V=V, m=m, rng=rng)
    t_sweep = direction_sweep(
        model, proc, v=v_type, clean_tok=Xe[ct][te], src_tok=Xe[cs][te],
        rc_clean=rc_ct_e, rc_src=rc_cs_e, t=t, k=k, tgt_facet="top_type",
        off_facet="depth", V=V, m=m, rng=rng)
    # fill the off-target gaps now that both target gaps are known, then re-score drag
    d_sweep = _renorm_drag(d_sweep, off_gap=t_sweep["gap"])
    t_sweep = _renorm_drag(t_sweep, off_gap=d_sweep["gap"])

    d_sub = direction_subverdict(d_sweep)
    t_sub = direction_subverdict(t_sweep)
    return matrix_verdict(d_sub, t_sub), d_sub, t_sub, d_sweep, t_sweep


def _renorm_drag(sw, off_gap):
    """Re-normalize the drag trajectory by the off-target facet's gap (known only after
    both directions' target gaps are computed). The sweep stored RAW |P-C| means under
    `_raw_drag`; here we divide by off_gap. Drag fractions are recomputed in place."""
    if not np.isfinite(off_gap) or off_gap < 1e-9:
        sw["f_drag"] = [float("nan")] * len(sw["alphas"])
        sw["f_drag_rand"] = [float("nan")] * len(sw["alphas"])
        return sw
    sw["f_drag"] = [d / off_gap if np.isfinite(d) else float("nan") for d in sw["_raw_drag"]]
    sw["f_drag_rand"] = [d / off_gap if np.isfinite(d) else float("nan")
                         for d in sw["_raw_drag_rand"]]
    return sw


# ---- harness guards -------------------------------------------------------
def model_guards(model, proc, cfg, m, V):
    """no-op additive steer (alpha=0) is bit-exact vs unpatched; the full-replacement
    patch reproduces source m=1 (the 37/38 wiring sanity for the patch path)."""
    rng = np.random.default_rng(0)
    Xe = proc.sample(64, cfg["seq_len"], rng)
    resid = stream_to(model, torch.from_numpy(Xe), LAYER)
    zero_v = torch.zeros(resid.shape[1], resid.shape[2], device=resid.device)
    perm = rng.permutation(len(Xe))
    for t in POSITIONS:
        q0 = q_at(model, Xe, t, m, V)
        noop = q_at(model, Xe, t, m, V,
                    prefix_state=apply_additive_steer(resid, zero_v, t, 0.0, np.arange(t + 1)))
        if not np.allclose(q0, noop, atol=1e-6):
            print(f"  GUARD FAIL: no-op additive steer not bit-exact at t={t}"); return False
        full = make_patched_prefix(resid, resid[perm], t, np.arange(t + 1))
        qf = q_at(model, Xe, t, m, V, prefix_state=full)
        if not np.allclose(marginal(qf, V, 1, m), marginal(q_at(model, Xe[perm], t, m, V),
                                                            V, 1, m), atol=1e-6):
            print(f"  GUARD FAIL: full patch m=1 != source m=1 at t={t}"); return False
    return True


def _steer_transport(model, tok_c, tok_s, rc_c, v, t, alpha, pos, facet, V, m, k):
    """Target transport toward `tok_s`'s facet value under an alpha-steer of `v` on the
    clean tokens `tok_c` (read on the registered facet observable)."""
    C, mC = read_facet(q_at(model, tok_c, t, m, V), facet, V, m, k)
    S, mS = read_facet(q_at(model, tok_s, t, m, V), facet, V, m, k)
    P, mP = read_facet(q_at(model, tok_c, t, m, V,
                            prefix_state=apply_additive_steer(rc_c, v, t, alpha, pos)),
                       facet, V, m, k)
    return transport_fraction(P, C, S, GAP_MIN, valid=mC & mS & mP)


def _sum_ratio_delta(model, tok_c, rc_c, v, t, alpha, pos, V, m):
    """(dsum, dratio): mean |delta| of the m=1 close-readiness SUM (q2+q3) and the
    top_type RATIO (q2/(q2+q3)) under an alpha-steer of `v`. The (q2,q3) self-test:
    v_depth (pure-sum) must move sum >> ratio; v_type (pure-ratio) the reverse."""
    jc = q_at(model, tok_c, t, m, V)
    jp = q_at(model, tok_c, t, m, V,
              prefix_state=apply_additive_steer(rc_c, v, t, alpha, pos))
    sC, msC = facet_observable(jc, "depth", V, m)     # close-readiness = q2+q3 (the SUM)
    sP, msP = facet_observable(jp, "depth", V, m)
    rC, mrC = facet_observable(jc, "top_type", V, m)   # type-0 frac = q2/(q2+q3) (RATIO)
    rP, mrP = facet_observable(jp, "top_type", V, m)
    ms, mr = msC & msP, mrC & mrP
    dsum = float(np.mean(np.abs(sP[ms] - sC[ms]))) if ms.sum() else float("nan")
    drat = float(np.mean(np.abs(rP[mr] - rC[mr]))) if mr.sum() else float("nan")
    return dsum, drat


def direction_guards(model, proc, cfg, m, V):
    """The writeup's registered direction self-tests, at the first position / seed 700:
      (1) sign/monotonicity for BOTH facets: +alpha moves the facet toward source MORE
          than -alpha (a non-monotone direction is not a facet direction);
      (2) same-facet floor: a v_f built from SAME-label pairs (depth_triples' src_lo;
          floor_pairs for type) moves its own facet <= SELFTEST_FLOOR (the diff vector
          is facet-carried, not a generic push — the high-drag leak control);
      (3) the (q2,q3) decomposition: v_depth moves the SUM more than the RATIO and
          v_type the RATIO more than the SUM (so a non-zero drag is a real cross-
          component, not an off-target leak)."""
    rng = np.random.default_rng(700)
    Xe = proc.sample(N_SEQS, cfg["seq_len"], rng)
    t, (lo, hi) = POSITIONS[0], HORIZONS[1]
    pos = np.arange(t + 1)
    labels = {i: stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
    cd, chi, clo = depth_triples(labels, lo, hi, rng)             # lo->hi ; lo->lo floor
    ct, cs = facet_pairs(labels, "top_type", rng, len(Xe), oriented=True)   # 0->1
    ftc, fts = floor_pairs(labels, "top_type", rng, len(Xe))     # same type, diff depth
    if min(len(cd), len(ct), len(ftc)) < MIN_PAIRS:
        print(f"  GUARD: thin pairs at t={t} "
              f"(depth={len(cd)} type={len(ct)} type_floor={len(ftc)})"); return False

    def cache(idx):
        return stream_to(model, torch.from_numpy(Xe[idx]), LAYER)
    rc_cd, rc_chi, rc_clo = cache(cd), cache(chi), cache(clo)
    rc_ct, rc_cs, rc_ftc, rc_fts = cache(ct), cache(cs), cache(ftc), cache(fts)
    hd, ht, hf = len(cd) // 2, len(ct) // 2, len(ftc) // 2
    v_depth = facet_diff_vector(rc_cd[:hd], rc_chi[:hd], t)       # pure-sum
    v_type = facet_diff_vector(rc_ct[:ht], rc_cs[:ht], t)         # pure-ratio
    v_dfloor = facet_diff_vector(rc_cd[:hd], rc_clo[:hd], t)      # same depth -> ~0 dir
    v_tfloor = facet_diff_vector(rc_ftc[:hf], rc_fts[:hf], t)     # same type  -> ~0 dir
    de = np.arange(hd, min(len(cd), hd + EVAL_CAP))
    te = np.arange(ht, min(len(ct), ht + EVAL_CAP))
    dc, ds_, hc, hs = Xe[cd][de], Xe[chi][de], Xe[ct][te], Xe[cs][te]

    ok = True
    # (1) sign / monotonicity, both facets
    dp = _steer_transport(model, dc, ds_, rc_cd[de], v_depth, t, +1.0, pos, "depth", V, m, 1)
    dm = _steer_transport(model, dc, ds_, rc_cd[de], v_depth, t, -1.0, pos, "depth", V, m, 1)
    tp = _steer_transport(model, hc, hs, rc_ct[te], v_type, t, +1.0, pos, "top_type", V, m, 1)
    tm = _steer_transport(model, hc, hs, rc_ct[te], v_type, t, -1.0, pos, "top_type", V, m, 1)
    sign_ok = (np.isfinite(dp) and np.isfinite(dm) and dp > dm
               and np.isfinite(tp) and np.isfinite(tm) and tp > tm)
    print(f"  sign t={t}: depth f(+1)={dp:+.3f} f(-1)={dm:+.3f} | "
          f"type f(+1)={tp:+.3f} f(-1)={tm:+.3f} -> {'OK' if sign_ok else 'FAIL'}")
    ok &= sign_ok
    # (2) same-facet floor: a same-label direction barely moves its own facet
    df = abs(_steer_transport(model, dc, ds_, rc_cd[de], v_dfloor, t, 1.0, pos, "depth", V, m, 1))
    tf = abs(_steer_transport(model, hc, hs, rc_ct[te], v_tfloor, t, 1.0, pos, "top_type", V, m, 1))
    floor_ok = np.isfinite(df) and np.isfinite(tf) and max(df, tf) <= SELFTEST_FLOOR
    print(f"  same-facet floor: depth={df:.3f} type={tf:.3f} (<= {SELFTEST_FLOOR}) -> "
          f"{'OK' if floor_ok else 'FAIL'}")
    ok &= floor_ok
    # (3) (q2,q3) decomposition: v_depth pure-sum, v_type pure-ratio
    dsd, drd = _sum_ratio_delta(model, dc, rc_cd[de], v_depth, t, 1.0, pos, V, m)
    dst, drt = _sum_ratio_delta(model, hc, rc_ct[te], v_type, t, 1.0, pos, V, m)
    decomp_ok = (np.isfinite(dsd) and np.isfinite(drd) and dsd > drd
                 and np.isfinite(dst) and np.isfinite(drt) and drt > dst)
    print(f"  (q2,q3): v_depth dsum={dsd:.3f} dratio={drd:.3f} | "
          f"v_type dsum={dst:.3f} dratio={drt:.3f} -> {'OK' if decomp_ok else 'FAIL'}")
    ok &= decomp_ok
    return ok


# ---- main path ------------------------------------------------------------
def run(model, proc, cfg, seeds=SEEDS, n_seqs=N_SEQS, min_pairs=MIN_PAIRS):
    m, V = cfg["m"], proc.V
    print("[guards] no-op additive bit-exact; full patch m=1 = source m=1")
    if not model_guards(model, proc, cfg, m, V):
        print("  -> HARNESS_FAIL\n"); return "HARNESS_FAIL"
    print("  -> OK")
    print("[direction guards: sign/monotonicity (both facets), same-facet floor, "
          "(q2,q3) decomposition]")
    if not direction_guards(model, proc, cfg, m, V):
        print("  -> HARNESS_FAIL (a registered direction self-test failed)\n")
        return "HARNESS_FAIL"
    print("  -> OK\n")

    per_seed = {k: [] for k in HORIZONS}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        Xe = proc.sample(n_seqs, cfg["seq_len"], rng)
        print(f"[seed {seed}]")
        for k, (lo, hi) in HORIZONS.items():
            verdicts, summary = [], []
            for t in POSITIONS:
                cell = eval_cell(model, proc, Xe, t, k, lo, hi, V, m, rng, min_pairs)
                if cell is None:
                    continue
                mv, d_sub, t_sub, d_sw, t_sw = cell
                verdicts.append(mv)
                summary.append((t, mv, d_sub, t_sub, d_sw, t_sw))
            kv = reduce_positions(verdicts)
            per_seed[k].append(kv)
            _print_horizon(k, lo, hi, summary, kv)
        print()

    agg = {f"k{k}": majority_vote(per_seed[k], threshold=SEED_MAJORITY,
                                  unstable="SEED_UNSTABLE") for k in HORIZONS}
    label, _ = first_precedence(agg, PRECEDENCE)
    print(f"per-horizon aggregate: {agg}")
    print("per-horizon routing:")
    for k in HORIZONS:
        print(f"  k={k}: {agg[f'k{k}']}")
    print(f"\nDECISION (highest-severity across horizons): {label}")
    return label


def _print_horizon(k, lo, hi, summary, kv):
    print(f"  k={k} (depth {lo} vs {hi}) -> {kv}")
    for t, mv, d_sub, t_sub, d_sw, t_sw in summary:
        print(f"    t={t:2d} {mv:11s} | "
              f"depth_dir[{d_sub[0]:9s} ceil={d_sw['ceiling']:+.2f} "
              f"tgt={_at(d_sw,'f_tgt',d_sub)}/rnd{_at(d_sw,'f_tgt_rand',d_sub)} "
              f"drag={_at(d_sw,'f_drag',d_sub)}/rnd{_at(d_sw,'f_drag_rand',d_sub)} "
              f"odef={_at(d_sw,'off_def',d_sub)} oe={d_sw['oe']:.3f}] "
              f"type_dir[{t_sub[0]:9s} ceil={t_sw['ceiling']:+.2f} "
              f"tgt={_at(t_sw,'f_tgt',t_sub)}/rnd{_at(t_sw,'f_tgt_rand',t_sub)} "
              f"drag={_at(t_sw,'f_drag',t_sub)}/rnd{_at(t_sw,'f_drag_rand',t_sub)} "
              f"odef={_at(t_sw,'off_def',t_sub)} oe={t_sw['oe']:.3f}]")


def _at(sw, key, sub):
    """The trajectory value at the chosen alpha* (or the max-alpha value if none)."""
    i = sub[1] if sub[1] is not None else -1
    v = sw[key][i]
    return f"{v:+.2f}" if np.isfinite(v) else "  nan"


# ---- self-tests (pure, no checkpoint) -------------------------------------
def _selftest():
    # subverdict: a clean handle with low drag -> SPECIFIC
    spec = {"oe": 0.01, "ceiling": 0.9, "f_tgt": [0.2, 0.5, 0.8, 0.9],
            "f_tgt_rand": [0.0, 0.05, 0.1, 0.2], "f_drag": [0.0, 0.03, 0.1, 0.2],
            "f_drag_rand": [0.0, 0.02, 0.05, 0.1], "off_def": [1.0, 1.0, 1.0, 1.0]}
    assert direction_subverdict(spec) == ("SPECIFIC", 1), direction_subverdict(spec)
    # off-target readout attrited at the chosen alpha (alpha*=1) -> OBS_DRIFT, not SPECIFIC
    assert direction_subverdict(dict(spec, off_def=[1.0, 0.5, 1.0, 1.0])) == ("OBS_DRIFT", 1)
    # same handle but big cross-drag -> DRAGS
    drags = dict(spec, f_drag=[0.0, 0.5, 0.6, 0.7])
    assert direction_subverdict(drags)[0] == "DRAGS"
    # target never reaches REF_FRAC*ceiling above the random floor -> NO_HANDLE
    noh = dict(spec, f_tgt=[0.1, 0.15, 0.2, 0.25])
    assert direction_subverdict(noh) == ("NO_HANDLE", None)
    # target reaches level but only by matching the random direction -> NO_HANDLE
    norand = dict(spec, f_tgt_rand=[0.2, 0.5, 0.8, 0.9])
    assert direction_subverdict(norand) == ("NO_HANDLE", None)
    # endpoint drift -> OBS_DRIFT
    assert direction_subverdict(dict(spec, oe=0.5)) == ("OBS_DRIFT", None)
    # no ceiling (facet not transportable even by full replacement) -> NO_HANDLE
    assert direction_subverdict(dict(spec, ceiling=float("nan"))) == ("NO_HANDLE", None)

    # matrix verdict
    S, D, N, O = ("SPECIFIC", 1), ("DRAGS", 1), ("NO_HANDLE", None), ("OBS_DRIFT", None)
    assert matrix_verdict(S, S) == "DISSOCIATED"
    assert matrix_verdict(S, D) == "CROSS_DRAG"
    assert matrix_verdict(S, N) == "MIXED"
    assert matrix_verdict(N, N) == "NO_HANDLE"
    assert matrix_verdict(O, S) == "OBS_DRIFT"
    assert matrix_verdict(D, N) == "CROSS_DRAG"      # a drag anywhere -> CROSS_DRAG

    # position reducer
    assert reduce_positions(["DISSOCIATED", "DISSOCIATED", "MIXED"]) == "DISSOCIATED"
    assert reduce_positions(["CROSS_DRAG", "DISSOCIATED", "MIXED"]) == "MIXED"
    assert reduce_positions(["CROSS_DRAG", "CROSS_DRAG", "DISSOCIATED"]) == "CROSS_DRAG"
    assert reduce_positions(["OBS_DRIFT", "DISSOCIATED"]) == "OBS_DRIFT"
    assert reduce_positions([]) == "SEED_UNSTABLE"

    # cross-seed reduction + precedence
    assert majority_vote(["DISSOCIATED"] * 3 + ["MIXED"], threshold=3,
                         unstable="SEED_UNSTABLE") == "DISSOCIATED"
    assert first_precedence({"k1": "DISSOCIATED", "k2": "CROSS_DRAG"},
                            PRECEDENCE)[0] == "CROSS_DRAG"
    assert first_precedence({"k1": "DISSOCIATED", "k2": "DISSOCIATED"},
                            PRECEDENCE)[0] == "DISSOCIATED"
    assert first_precedence({"k1": "SEED_UNSTABLE", "k2": "DISSOCIATED"},
                            PRECEDENCE)[0] == "SEED_UNSTABLE"

    # drag_fraction: mean|P-C|/gap over valid rows; gap<=0 -> nan
    assert abs(drag_fraction(np.array([0.5, 0.7]), np.array([0.1, 0.1]), 1.0,
                             np.array([True, True])) - 0.5) < 1e-9
    assert np.isnan(drag_fraction(np.array([0.5]), np.array([0.1]), 0.0, np.array([True])))
    print("exp40 selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true",
                    help="tiny runnability check (1 seed, few seqs, 1 position); "
                         "NOT the registered run")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest(); return
    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    require_expected_config(cfg)
    proc = PROCESSES[cfg["process"]]()
    model = load_model(args.outdir, cfg, proc)
    gap, passed = validity_gate(model, proc, cfg, 0)
    if not passed:
        print("HALT: validity gate failed."); sys.exit(1)
    device = next(model.parameters()).device
    print(f"=== Experiment 40: Dyck-2 directional specificity | L{cfg['layers']} "
          f"d{cfg['d_model']} | m={cfg['m']} | device={device} ===\n")
    if args.dry:
        global POSITIONS
        POSITIONS = (12,)
        print("** DRY runnability check — 1 seed, reduced seqs, 1 position; "
              "NOT the registered run **\n")
        run(model, proc, cfg, seeds=(700,), n_seqs=3000, min_pairs=64)
    else:
        run(model, proc, cfg)


if __name__ == "__main__":
    main()
