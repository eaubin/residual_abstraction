"""
Oracle-withdrawal experiment 3: does the exp-24 reference ambiguity matter
downstream for rho?

Exp 24 found three compact k=4 references (cegar/pca/delta) that tie
observably and are exact-audit-indifferent, with pca separated ~10-12 deg
from a near-coincident {cegar,delta} plane. Because the oracle is
indifferent among them, an oracle-free tie-break is unfalsifiable. The
auditable question (ORACLE_WITHDRAWAL.md: "differ exactly OR disagree on
downstream rho") is the rho half: do rho-verdicts depend on which
equally-good reference anchors them?

Arm A is a *structural* seed-stability gate (observable): does the
two-reference geometry reproduce across pair/basis sampling? Arm B
(conditional) measures rho anchor-divergence against a self-calibrated
null: cegar and delta are the SAME reference built two ways (4.9 deg), so
their rho-verdict spread is the within-cluster noise floor; the question is
whether anchoring on pca (the cross-cluster outlier) adds divergence beyond
that floor. The exact oracle is read only after every observable verdict is
printed, to (i) confirm the references reproduce exp-24's exact-
indifference [premise check] and (ii) calibrate the rho bands.

Firewall: rho-invariance is NOT selection success. Exp 24's NO-GO on
unique selection stands; Exp 25 decides only whether the ambiguity matters
downstream for rho (member 2). Member 4 (also reference-normalized) is a
residual checked in Block 3 under the chosen anchor.

Candidate construction is reused unchanged from Exp 24 by import; the
frozen exp-24 script is not edited.
"""

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from battery import Exact, Refs, jeffreys_rows
from discover import PairSet, self_checks
from expcommon import LAYER, load_model
from midstream import marginal
from processes import PROCESSES
from scripts.oracle_withdrawal.reference_selection import (
    AMBIG_ANGLE_MAX, M, MM, PAIR_POOL, PAIRS_DISC, PAIRS_EVAL, PAIRS_HELD,
    TS_DISC, TS_HELD, build_candidates, build_strata_layout,
    evaluate_observable, max_principal_angle, require_registered_config,
    select_observable, strata_guard,
)

K_SEEDS = 8
GATE_MAJORITY = 6
ARM_B_SEED = 0
EXACT_INDIFF_BAND = 0.05         # "no right reference"; < exp-24's 0.10 best-margin
RHO_EQUIV = 0.25                 # battery member-2 bands
RHO_DIST = 0.50
RHO_ANCHOR_SLACK = 0.05          # tolerated rho-divergence above the within-cluster null
COMPACT = ("cegar", "pca", "delta")
PROBES = ("cegar", "pca", "delta", "emb", "rand", "full")
ANCHORS = ("pca", "cegar")       # the two distinct references
NULL_ANCHOR = "delta"            # same cluster as cegar -> within-cluster null


def _mnorm(q, V, mm, m_full):
    qm = marginal(q, V, mm, m_full)
    return qm / np.clip(qm.sum(axis=1, keepdims=True), 1e-30, None)


def rho_obs(q0, qC, qX, V, mm, m_full):
    """Observable per-pair equivalence ratio (battery member 2), computed
    without touching the process oracle: mean J(q_C, q_X) / mean
    J(q_C, q_unpatched), marginalized to horizon mm."""
    a = _mnorm(qC, V, mm, m_full)
    b = _mnorm(qX, V, mm, m_full)
    u = _mnorm(q0, V, mm, m_full)
    return float(jeffreys_rows(a, b).mean() / jeffreys_rows(a, u).mean())


def rho_band(r):
    if r <= RHO_EQUIV:
        return "equivalent"
    if r >= RHO_DIST:
        return "distinct"
    return "indeterminate"


def clean_outlier(a_cd, a_pc, a_pd, thr):
    """The single candidate far (> thr) from both others while the other
    two are close (<= thr); None if there is no clean 2-cluster split."""
    if a_pc > thr and a_pd > thr and a_cd <= thr:
        return "pca"
    if a_pc > thr and a_cd > thr and a_pd <= thr:
        return "cegar"
    if a_pd > thr and a_cd > thr and a_pc <= thr:
        return "delta"
    return None


def run_seed(model, proc, cfg, seed, keep_arm_b):
    """Rerun the Exp-24 selection at one seed. Observable fields (tie,
    angles, outlier) drive the structural gate; exact closures are computed
    but QUARANTINED (read only at the reveal stage, for the premise check).
    Returns a summary and, for the reference seed, the Arm-B bundle."""
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + 111, PAIR_POOL,
                   layer=LAYER, ts=TS_DISC)
    eval_ps = PairSet(model, proc, cfg, PAIRS_EVAL, M, seed + 222, PAIR_POOL,
                      layer=LAYER, ts=TS_DISC)
    held_ps = PairSet(model, proc, cfg, PAIRS_HELD, M, seed + 333, PAIR_POOL,
                      layer=LAYER, ts=TS_HELD)
    self_checks(model, eval_ps, LAYER, M, proc.V)

    refs_d = Refs(disc, model, disc.d, M)
    refs_e = Refs(eval_ps, model, eval_ps.d, M)
    refs_h = Refs(held_ps, model, held_ps.d, M)
    strata, global_denom = build_strata_layout(refs_e, eval_ps)
    assert global_denom > 0, "observable scale degenerate: D0 <= D_full"
    ok, gbranch, gdetail = strata_guard(strata, global_denom)
    if not ok:
        return {"seed": seed, "halt": (gbranch, gdetail)}, None

    candidates, k_cegar, k_ref = build_candidates(model, proc, cfg, disc,
                                                  refs_d, seed)
    rows, q_eval = evaluate_observable(model, candidates, refs_e, refs_h,
                                       eval_ps, held_ps, strata)
    selection = select_observable(rows)

    angles = {pair: max_principal_angle(candidates[pair[0]]["Q"],
                                        candidates[pair[1]]["Q"])
              for pair in combinations(COMPACT, 2)}
    outlier = clean_outlier(angles[("cegar", "delta")],
                            angles[("pca", "cegar")],
                            angles[("pca", "delta")], AMBIG_ANGLE_MAX)
    tied_compact = [n for n in selection["tied"] if n in COMPACT]

    # QUARANTINED exact closures (not read until the reveal stage).
    exact = Exact(eval_ps, model, M)
    exact_cl = {n: exact.closure(q_eval[n], MM) for n in COMPACT}
    if len(tied_compact) >= 2:
        cls = [exact_cl[n] for n in tied_compact]
        exact_spread = max(cls) - min(cls)
    else:
        exact_spread = 0.0

    summary = {
        "seed": seed, "halt": None,
        "branch": selection["branch"], "tied": selection["tied"],
        "tied_compact": tied_compact, "tie_ge2": len(tied_compact) >= 2,
        "angles": angles, "outlier": outlier, "k_ref": k_ref,
        "_exact_cl": exact_cl, "_exact_spread": exact_spread,  # quarantined
    }
    bundle = None
    if keep_arm_b:
        bundle = {"q_eval": q_eval, "q_un": refs_e.q_un, "V": proc.V,
                  "exact": exact}
    return summary, bundle


def evaluate_structural(summaries):
    """Observable gate. The operative criterion is a per-seed JOINT flag —
    the modal outlier is both the clean 10-deg outlier AND a member of the
    tied compact set that seed — so a seed cannot credit the gate via a
    cegar+delta tie and a pca-outlier geometry that do not co-occur. G1
    (tie) and G2 (outlier==modal) are reported as diagnostics. No exact
    use. Pass requires the joint flag in >= GATE_MAJORITY of K_SEEDS."""
    g1 = sum(s["tie_ge2"] for s in summaries)
    outliers = [s["outlier"] for s in summaries if s["outlier"] is not None]
    modal = max(set(outliers), key=outliers.count) if outliers else None
    g2 = sum(s["outlier"] == modal and modal is not None for s in summaries)
    joint = sum(s["tie_ge2"] and modal is not None
                and s["outlier"] == modal and modal in s["tied_compact"]
                for s in summaries)
    if g1 < GATE_MAJORITY:
        branch = "SEED_UNSTABLE_TIE"
    elif joint < GATE_MAJORITY:
        branch = "SEED_UNSTABLE_CLUSTER"
    else:
        branch = "STRUCTURAL_PASS"
    return branch, {"G1_tie": g1, "G2_cluster": g2, "joint": joint}, modal


def exact_premise(summaries):
    """Exact premise check (reveal stage): G3 exact-indifference reproduces.
    If it fails, a 'right' reference may exist and the auditable tie-break
    Exp 24 withdrew is re-opened."""
    g3 = sum(s["_exact_spread"] <= EXACT_INDIFF_BAND for s in summaries)
    return g3 >= GATE_MAJORITY, g3


def arm_b(bundle):
    """Observable rho anchor-divergence vs the within-cluster null."""
    q_eval, q_un, V = bundle["q_eval"], bundle["q_un"], bundle["V"]
    rho, bands = {}, {}
    for C in (*ANCHORS, NULL_ANCHOR):
        for X in PROBES:
            r = rho_obs(q_un, q_eval[C], q_eval[X], V, MM, M)
            rho[(C, X)] = r
            bands[(C, X)] = rho_band(r)
    cross = {X: abs(rho[("pca", X)] - rho[("cegar", X)]) for X in PROBES}
    null = {X: abs(rho[("cegar", X)] - rho[(NULL_ANCHOR, X)]) for X in PROBES}
    d_cross = max(cross.values())
    d_null = max(null.values())
    invariant = d_cross <= d_null + RHO_ANCHOR_SLACK
    arg_cross = max(cross, key=cross.get)
    band_flips = [X for X in PROBES
                  if bands[("pca", X)] != bands[("cegar", X)]]
    # Null validity: the within-cluster pair (same reference, two
    # constructions) must itself be rho-equivalent, else d_null is an
    # invalid noise floor AND rho is splitting a behaviorally-equivalent
    # reference (the sharpest load-bearing case, not a band miscalibration).
    null_valid = max(rho[("cegar", NULL_ANCHOR)],
                     rho[(NULL_ANCHOR, "cegar")]) <= RHO_EQUIV
    return {"rho": rho, "bands": bands, "cross": cross, "null": null,
            "d_cross": d_cross, "d_null": d_null, "invariant": invariant,
            "arg_cross": arg_cross, "band_flips": band_flips,
            "null_valid": null_valid}


def arm_b_audit(bundle, bands):
    """Exact audit: anchors exact-equivalent at the reference seed, and the
    rho band CALIBRATION known-case check. Calibration here is ONLY the
    rand band (a genuine threshold check: junk must read distinct). Whether
    exact-equivalent references read mutually rho-equivalent is the
    load-bearing question, handled by null_valid (within-cluster) and the
    invariance metric (cross-cluster) — NOT folded into calibration."""
    exact, q_eval = bundle["exact"], bundle["q_eval"]
    exact_cl = {X: exact.closure(q_eval[X], MM) for X in PROBES}
    cls = [exact_cl[n] for n in COMPACT]
    anchors_spread = max(cls) - min(cls)
    anchors_equiv = anchors_spread <= EXACT_INDIFF_BAND
    calib = {C: bands[(C, "rand")] == "distinct" for C in ANCHORS}
    return exact_cl, anchors_spread, anchors_equiv, calib, all(calib.values())


def selftest():
    assert rho_band(0.1) == "equivalent" and rho_band(0.6) == "distinct"
    assert rho_band(0.4) == "indeterminate"
    rng = np.random.default_rng(0)
    q0 = rng.dirichlet(np.ones(8), size=50)
    qC = rng.dirichlet(np.ones(8), size=50)
    assert rho_obs(q0, qC, qC, 8, 1, 1) < 1e-9        # probe == anchor -> 0
    qX = rng.dirichlet(0.1 * np.ones(8), size=50)
    assert rho_obs(q0, qC, qX, 8, 1, 1) > 0.0

    assert clean_outlier(4.9, 10.4, 12.4, 10.0) == "pca"     # exp-24 geometry
    assert clean_outlier(11.0, 11.0, 4.0, 10.0) == "cegar"   # cegar far from both
    assert clean_outlier(2.0, 3.0, 4.0, 10.0) is None        # all one cluster
    assert clean_outlier(20.0, 30.0, 40.0, 10.0) is None     # all distinct

    def _summ(tie, outlier, spread, tied_compact=None):
        if tied_compact is None:
            tied_compact = list(COMPACT) if tie else []
        return {"tie_ge2": tie, "outlier": outlier, "_exact_spread": spread,
                "tied_compact": tied_compact}
    s = [_summ(True, "pca", 0.01)] * 7 + [_summ(False, None, 0.0)]
    branch, checks, modal = evaluate_structural(s)
    assert branch == "STRUCTURAL_PASS" and modal == "pca" and checks["joint"] == 7
    assert exact_premise(s)[0]
    s = [_summ(True, "pca", 0.01)] * 5 + [_summ(False, None, 0.0)] * 3
    assert evaluate_structural(s)[0] == "SEED_UNSTABLE_TIE"
    s = [_summ(True, "pca", 0.01)] * 4 + [_summ(True, "cegar", 0.01)] * 2 \
        + [_summ(True, None, 0.01)] * 2
    assert evaluate_structural(s)[0] == "SEED_UNSTABLE_CLUSTER"  # joint 4 < 6
    # joint catches tie & outlier that do NOT co-occur: G1=6, modal pca
    # counted 5x, but pca is tied in only 3 of those seeds.
    s = ([_summ(True, "pca", 0.01)] * 3
         + [_summ(True, None, 0.01, tied_compact=["cegar", "delta"])] * 3
         + [_summ(False, "pca", 0.01, tied_compact=[])] * 2)
    branch, checks, _ = evaluate_structural(s)
    assert branch == "SEED_UNSTABLE_CLUSTER"
    assert checks["G1_tie"] == 6 and checks["joint"] == 3
    s = [_summ(True, "pca", 0.20)] * 8
    assert evaluate_structural(s)[0] == "STRUCTURAL_PASS"     # structural ok
    assert not exact_premise(s)[0]                            # but premise breaks

    # arm_b metric: cross-cluster divergence judged against the null + slack
    def _bundle(rho_map):
        # synthesize q_eval-independent: monkeypatch via closure not needed;
        # test the metric on a hand-built rho dict by calling the math.
        return rho_map
    # invariant when cross ~ null; sensitive when cross >> null
    rho_inv = {}
    for X in PROBES:
        rho_inv[("pca", X)] = 0.30
        rho_inv[("cegar", X)] = 0.31
        rho_inv[("delta", X)] = 0.29        # null ~0.02, cross ~0.01
    d_cross = max(abs(rho_inv[("pca", X)] - rho_inv[("cegar", X)])
                  for X in PROBES)
    d_null = max(abs(rho_inv[("cegar", X)] - rho_inv[("delta", X)])
                 for X in PROBES)
    assert d_cross <= d_null + RHO_ANCHOR_SLACK
    rho_sen = dict(rho_inv)
    rho_sen[("pca", "emb")] = 0.80          # pca sees emb very differently
    d_cross_s = max(abs(rho_sen[("pca", X)] - rho_sen[("cegar", X)])
                    for X in PROBES)
    assert d_cross_s > d_null + RHO_ANCHOR_SLACK

    # decide(): known-answer cases pinning every branch and the priority.
    # signature: (structural, premise_ok, null_valid, invariant, calibrated)
    assert decide("STRUCTURAL_PASS", True, True, True, True) == \
        "AMBIGUITY_DOWNSTREAM_BENIGN"
    assert decide("STRUCTURAL_PASS", True, True, False, True) == \
        "AMBIGUITY_LOAD_BEARING"                 # cross-cluster sensitive
    assert decide("STRUCTURAL_PASS", True, False, True, True) == \
        "AMBIGUITY_LOAD_BEARING"                 # rho splits same ref (null)
    assert decide("STRUCTURAL_PASS", True, True, True, False) == \
        "RHO_MISCALIBRATED_ON_PSTACK"            # rand not distinct
    assert decide("STRUCTURAL_PASS", False, True, True, True) == \
        "EXACT_INDIFFERENCE_BREAKS"
    assert decide("SEED_UNSTABLE_TIE", True, True, True, True) == \
        "SEED_UNSTABLE_TIE"
    # priority: under-sensitive rho (rand not distinct) is calibration, even
    # if refs also look split — recalibrate the band before reading verdicts.
    assert decide("STRUCTURAL_PASS", True, False, False, False) == \
        "RHO_MISCALIBRATED_ON_PSTACK"
    print("selftest passed: rho bands, clustering, gates, arm-B, and decide")


def decide(structural, premise_ok, null_valid, invariant, calibrated):
    """Verdict partition (the most 6.1-sensitive logic). Priority:
    structural (observable) -> exact premise -> band calibration (rand) ->
    rho splitting an equivalent reference, within (null) or cross
    (invariance) cluster -> benign. Calibration is rand-only: an
    under-sensitive rho (junk reads equivalent) is a band problem; an
    over-sensitive rho (equal refs read distinct) is load-bearing, and the
    two must not collide on one flag."""
    if structural != "STRUCTURAL_PASS":
        return structural
    if not premise_ok:
        return "EXACT_INDIFFERENCE_BREAKS"
    if not calibrated:
        return "RHO_MISCALIBRATED_ON_PSTACK"
    if not null_valid or not invariant:
        return "AMBIGUITY_LOAD_BEARING"
    return "AMBIGUITY_DOWNSTREAM_BENIGN"


def print_decision(decision):
    print("\nDECISION:")
    if decision == "AMBIGUITY_DOWNSTREAM_BENIGN":
        print("  GO: the reference ambiguity is downstream-benign for rho "
              "(member 2); preregister Block 3 under the declared canonical "
              "cegar-core anchor. Member 4 (reference-normalized) is checked "
              "there. NOTE: selection non-uniqueness (Exp 24 NO-GO) STANDS — "
              "this is not selection success.")
    elif decision == "AMBIGUITY_LOAD_BEARING":
        print("  NO-GO: rho anchor-divergence exceeds the within-cluster "
              "null; the unauditable reference choice moves equivalence "
              "verdicts. The reference-selection failure propagates into the "
              "battery — register a reference-robust repair before transfer.")
    elif decision == "RHO_MISCALIBRATED_ON_PSTACK":
        print("  NO-GO: rho mis-calibrated on pstack (known cases "
              "misclassified); recalibrate battery member 2 before any "
              "rho-based transfer claim.")
    elif decision == "EXACT_INDIFFERENCE_BREAKS":
        print("  HOLD-REFRAME: tied candidates are not reliably "
              "exact-equivalent; a 'right' reference may exist — register the "
              "auditable tie-break Exp 24 withdrew.")
    else:
        print("  NO-GO: the exp-24 two-reference geometry is "
              "sampling-noise-level; rho-invariance is moot. Battery transfer "
              "under the single discovered cegar core is a separate "
              "registration.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/pstack-L4")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force-invalid", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    if not require_registered_config(cfg, args.force_invalid):
        return 1
    proc = PROCESSES[cfg["process"]]()
    model = load_model(args.outdir, cfg, proc)

    print("=== Experiment 25: downstream-rho invariance ===")
    print(f"target={proc.name} outdir={args.outdir} m={M} LAYER={LAYER}")
    print(f"seeds={K_SEEDS} (gate majority {GATE_MAJORITY}/{K_SEEDS}); "
          f"arm-B reference seed={ARM_B_SEED}")
    print("Exact oracle is read only after observable verdicts are printed.\n")

    summaries, bundle = [], None
    print("[Arm A — observable structural gate]")
    print("seed branch              tie  a(c,d) a(p,c) a(p,d) outlier")
    for seed in range(K_SEEDS):
        summ, b = run_seed(model, proc, cfg, seed,
                           keep_arm_b=(seed == ARM_B_SEED))
        if summ["halt"] is not None:
            gbranch, gdetail = summ["halt"]
            print(f"\nHALT (seed {seed}): {gbranch} — {gdetail}")
            print("  NO-GO: substrate guard failed; not an Exp 25 result.")
            return 1
        summaries.append(summ)
        if seed == ARM_B_SEED:
            bundle = b
        a = summ["angles"]
        print(f"  {seed}  {summ['branch']:<18} "
              f"{'Y' if summ['tie_ge2'] else 'n'}   "
              f"{a[('cegar','delta')]:>5.1f}  {a[('pca','cegar')]:>5.1f}  "
              f"{a[('pca','delta')]:>5.1f}  {str(summ['outlier'])}")

    structural, checks, modal = evaluate_structural(summaries)
    print(f"\nstructural checks (k/{K_SEEDS}, need {GATE_MAJORITY}): "
          f"G1_tie={checks['G1_tie']} G2_cluster={checks['G2_cluster']} "
          f"joint={checks['joint']} (operative; modal outlier={modal})")
    print(f"ARM_A_STRUCTURAL: {structural}")

    if structural != "STRUCTURAL_PASS":
        print_decision(decide(structural, False, False, False, False))
        return 0

    # ---- Arm B: observable rho-verdicts, printed before exact audit ------
    b = arm_b(bundle)
    print("\n[Arm B — observable rho-verdicts (before exact audit)]")
    print("probe       rho|pca  rho|cegar rho|delta | |d_pca-cegar| "
          "|d_cegar-delta|")
    for X in PROBES:
        print(f"{X:<10} {b['rho'][('pca',X)]:>7.3f}  "
              f"{b['rho'][('cegar',X)]:>8.3f}  "
              f"{b['rho'][(NULL_ANCHOR,X)]:>8.3f} |  "
              f"{b['cross'][X]:>10.3f}    {b['null'][X]:>11.3f}")
    print(f"\nd_cross (pca vs cegar, max) = {b['d_cross']:.3f} @ {b['arg_cross']}")
    print(f"d_null  (cegar vs delta, max within-cluster) = {b['d_null']:.3f}")
    print(f"RHO_ANCHOR_INVARIANT: {b['invariant']} "
          f"(d_cross <= d_null + {RHO_ANCHOR_SLACK})")
    print(f"NULL_VALID (within-cluster cegar~delta rho-equivalent): "
          f"{b['null_valid']}")
    if b["band_flips"]:
        print("  (descriptive) band flips pca vs cegar: "
              + ",".join(b["band_flips"]))

    # ---- exact reveal: premise check + audit -----------------------------
    premise_ok, g3 = exact_premise(summaries)
    print("\n[exact reveal — premise check + audit]")
    print("per-seed exact closures (cegar/pca/delta) and tied spread:")
    for s in summaries:
        ec = s["_exact_cl"]
        print(f"  seed {s['seed']}: {ec['cegar']:.3f}/{ec['pca']:.3f}/"
              f"{ec['delta']:.3f}  spread={s['_exact_spread']:.3f}")
    print(f"G3 exact-indifference reproduces: {premise_ok} "
          f"({g3}/{K_SEEDS} seeds spread <= {EXACT_INDIFF_BAND})")

    exact_cl, anchors_spread, anchors_equiv, calib, calibrated = \
        arm_b_audit(bundle, b["bands"])
    print("anchors exact-equivalent at reference seed: "
          f"{anchors_equiv} (spread {anchors_spread:.3f})")
    print(f"rho band calibration per anchor (rand distinct): "
          f"{calib} -> {calibrated}")

    decision = decide(structural, premise_ok, b["null_valid"], b["invariant"],
                      calibrated)
    print(f"\nAUDIT_BRANCH: {decision}")
    print_decision(decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
