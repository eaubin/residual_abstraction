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

Arm A is a seed-stability gate: does the two-reference geometry reproduce
across pair/basis sampling, or is it single-seed noise? Arm B (conditional
on the gate) anchors rho on pca and on cegar in turn and tests whether the
battery's equivalence verdicts are invariant to the anchor.

Firewall: rho-invariance is NOT selection success. Exp 24's NO-GO on
unique selection stands; Exp 25 decides only whether the ambiguity matters
downstream.

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
    REGISTERED_CFG, TS_DISC, TS_HELD, build_candidates, build_strata_layout,
    evaluate_observable, max_principal_angle, require_registered_config,
    select_observable, strata_guard,
)

K_SEEDS = 8
GATE_MAJORITY = 6
ARM_B_SEED = 0
EXACT_INDIFF_BAND = 0.05
RHO_EQUIV = 0.25
RHO_DIST = 0.50
COMPACT = ("cegar", "pca", "delta")
PROBES = ("cegar", "pca", "delta", "emb", "rand", "full")
ANCHORS = ("pca", "cegar")


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
    """Rerun the Exp-24 selection at one seed; return a light summary and,
    for the reference seed, the heavy bundle Arm B consumes."""
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

    exact = Exact(eval_ps, model, M)
    exact_cl = {n: exact.closure(q_eval[n], MM) for n in COMPACT}
    tied = selection["tied"]
    tied_compact = [n for n in tied if n in COMPACT]
    if len(tied_compact) >= 2:
        cls = [exact_cl[n] for n in tied_compact]
        exact_spread = max(cls) - min(cls)
    else:
        exact_spread = 0.0

    summary = {
        "seed": seed,
        "halt": None,
        "branch": selection["branch"],
        "tied": tied,
        "tie_ge2": len(tied_compact) >= 2,
        "angles": angles,
        "outlier": outlier,
        "exact_cl": exact_cl,
        "exact_spread": exact_spread,
        "k_ref": k_ref,
    }
    bundle = None
    if keep_arm_b:
        bundle = {"q_eval": q_eval, "q_un": refs_e.q_un, "V": proc.V,
                  "exact": exact}
    return summary, bundle


def evaluate_gate(summaries):
    """G1 tie, G2 clustering (stable outlier identity), G3 exact
    indifference; each must hold in >= GATE_MAJORITY of K_SEEDS."""
    g1 = sum(s["tie_ge2"] for s in summaries)
    outliers = [s["outlier"] for s in summaries if s["outlier"] is not None]
    modal = max(set(outliers), key=outliers.count) if outliers else None
    g2 = sum(s["outlier"] == modal and modal is not None for s in summaries)
    g3 = sum(s["exact_spread"] <= EXACT_INDIFF_BAND for s in summaries)
    checks = {"G1_tie": g1, "G2_cluster": g2, "G3_exact_indiff": g3}
    p1 = g1 >= GATE_MAJORITY
    p2 = g2 >= GATE_MAJORITY
    p3 = g3 >= GATE_MAJORITY
    if not p1:
        branch = "SEED_UNSTABLE_TIE"
    elif not p2:
        branch = "SEED_UNSTABLE_CLUSTER"
    elif not p3:
        branch = "EXACT_INDIFFERENCE_BREAKS"
    else:
        branch = "GATE_PASS"
    return branch, checks, modal


def arm_b(bundle):
    q_eval, q_un, V = bundle["q_eval"], bundle["q_un"], bundle["V"]
    rho, bands = {}, {}
    for C in (*ANCHORS, "delta"):
        for X in PROBES:
            r = rho_obs(q_un, q_eval[C], q_eval[X], V, MM, M)
            rho[(C, X)] = r
            bands[(C, X)] = rho_band(r)
    flips = [X for X in PROBES
             if bands[("pca", X)] != bands[("cegar", X)]]
    invariant = len(flips) == 0
    same_cluster_flips = [X for X in PROBES
                          if bands[("cegar", X)] != bands[("delta", X)]]
    return rho, bands, flips, invariant, same_cluster_flips


def arm_b_audit(bundle, bands):
    exact, q_eval = bundle["exact"], bundle["q_eval"]
    exact_cl = {X: exact.closure(q_eval[X], MM) for X in PROBES}
    cls = [exact_cl[n] for n in COMPACT]
    anchors_spread = max(cls) - min(cls)
    anchors_equiv = anchors_spread <= EXACT_INDIFF_BAND
    calib = {}
    for C in ANCHORS:
        ref_ok = all(bands[(C, r)] == "equivalent" for r in COMPACT)
        rand_ok = bands[(C, "rand")] == "distinct"
        calib[C] = ref_ok and rand_ok
    calibrated = all(calib.values())
    return exact_cl, anchors_spread, anchors_equiv, calib, calibrated


def decide(gate_branch, invariant, calibrated):
    if gate_branch != "GATE_PASS":
        return gate_branch
    if not calibrated:
        return "RHO_MISCALIBRATED_ON_PSTACK"
    return "AMBIGUITY_DOWNSTREAM_BENIGN" if invariant \
        else "AMBIGUITY_LOAD_BEARING"


def selftest():
    assert rho_band(0.1) == "equivalent"
    assert rho_band(0.6) == "distinct"
    assert rho_band(0.4) == "indeterminate"
    # a probe equal to its anchor has rho 0 (equivalent); an orthogonal-ish
    # destructive probe has large rho (distinct).
    rng = np.random.default_rng(0)
    q0 = rng.dirichlet(np.ones(8), size=50)
    qC = rng.dirichlet(np.ones(8), size=50)
    assert rho_obs(q0, qC, qC, 8, 1, 1) < 1e-9
    qX = rng.dirichlet(0.1 * np.ones(8), size=50)
    assert rho_obs(q0, qC, qX, 8, 1, 1) > 0.0

    assert clean_outlier(4.9, 10.4, 12.4, 10.0) == "pca"     # exp-24 geometry
    # cegar far from both others (a_pc, a_cd > thr), pca-delta the close pair
    assert clean_outlier(11.0, 11.0, 4.0, 10.0) == "cegar"
    assert clean_outlier(2.0, 3.0, 4.0, 10.0) is None        # all one cluster
    assert clean_outlier(20.0, 30.0, 40.0, 10.0) is None     # all distinct

    def _summ(tie, outlier, spread):
        return {"tie_ge2": tie, "outlier": outlier, "exact_spread": spread}
    # 7/8 stable -> GATE_PASS
    s = [_summ(True, "pca", 0.01)] * 7 + [_summ(False, None, 0.0)]
    branch, checks, modal = evaluate_gate(s)
    assert branch == "GATE_PASS" and modal == "pca"
    # tie fails (only 5/8) -> SEED_UNSTABLE_TIE
    s = [_summ(True, "pca", 0.01)] * 5 + [_summ(False, None, 0.0)] * 3
    assert evaluate_gate(s)[0] == "SEED_UNSTABLE_TIE"
    # tie ok, clustering flips (outlier identity not modal in >=6) -> CLUSTER
    s = ([_summ(True, "pca", 0.01)] * 3 + [_summ(True, "cegar", 0.01)] * 3
         + [_summ(True, None, 0.01)] * 2)
    assert evaluate_gate(s)[0] == "SEED_UNSTABLE_CLUSTER"
    # tie + cluster ok, exact spread breaks -> EXACT_INDIFFERENCE_BREAKS
    s = [_summ(True, "pca", 0.20)] * 8
    assert evaluate_gate(s)[0] == "EXACT_INDIFFERENCE_BREAKS"

    assert decide("GATE_PASS", True, True) == "AMBIGUITY_DOWNSTREAM_BENIGN"
    assert decide("GATE_PASS", False, True) == "AMBIGUITY_LOAD_BEARING"
    assert decide("GATE_PASS", True, False) == "RHO_MISCALIBRATED_ON_PSTACK"
    assert decide("SEED_UNSTABLE_TIE", True, True) == "SEED_UNSTABLE_TIE"
    print("selftest passed: rho bands, clustering, gate, and decision")


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
    print("Exact oracle is read only in the audit stage.\n")

    summaries, bundle = [], None
    print("[Arm A — seed-stability gate]")
    print("seed branch              tie  a(c,d) a(p,c) a(p,d) outlier  "
          "exact c/p/d            spread")
    for seed in range(K_SEEDS):
        summ, b = run_seed(model, proc, cfg, seed, keep_arm_b=(seed == ARM_B_SEED))
        if summ["halt"] is not None:
            gbranch, gdetail = summ["halt"]
            print(f"\nHALT (seed {seed}): {gbranch} — {gdetail}")
            print("  NO-GO: substrate guard failed; not an Exp 25 result.")
            return 1
        summaries.append(summ)
        if seed == ARM_B_SEED:
            bundle = b
        a = summ["angles"]
        ex = summ["exact_cl"]
        print(f"  {seed}  {summ['branch']:<18} "
              f"{'Y' if summ['tie_ge2'] else 'n'}   "
              f"{a[('cegar','delta')]:>5.1f}  {a[('pca','cegar')]:>5.1f}  "
              f"{a[('pca','delta')]:>5.1f}  {str(summ['outlier']):<7} "
              f"{ex['cegar']:.3f}/{ex['pca']:.3f}/{ex['delta']:.3f}  "
              f"{summ['exact_spread']:>6.3f}")

    gate_branch, checks, modal = evaluate_gate(summaries)
    print(f"\ngate checks (k/{K_SEEDS}, need {GATE_MAJORITY}): "
          f"G1_tie={checks['G1_tie']} "
          f"G2_cluster={checks['G2_cluster']} (modal outlier={modal}) "
          f"G3_exact_indiff={checks['G3_exact_indiff']}")
    print(f"ARM_A_OUTCOME: {gate_branch}")

    if gate_branch != "GATE_PASS":
        print("\nDECISION:")
        if gate_branch == "EXACT_INDIFFERENCE_BREAKS":
            print("  HOLD-REFRAME: tied candidates are not reliably "
                  "exact-equivalent; a 'right' reference may exist — register "
                  "the auditable tie-break Exp 24 withdrew.")
        else:
            print("  NO-GO: the exp-24 two-reference geometry is "
                  "sampling-noise-level; rho-invariance is moot. Battery "
                  "transfer under the single discovered cegar core is a "
                  "separate registration.")
        return 0

    # ---- Arm B: observable rho-verdicts, printed before exact audit ------
    rho, bands, flips, invariant, sc_flips = arm_b(bundle)
    print("\n[Arm B — observable rho-verdicts (before exact audit)]")
    print("probe       rho|pca  band|pca      rho|cegar band|cegar   "
          "anchor-agree")
    for X in PROBES:
        rp, rc = rho[("pca", X)], rho[("cegar", X)]
        agree = "yes" if bands[("pca", X)] == bands[("cegar", X)] else "FLIP"
        print(f"{X:<10} {rp:>7.3f}  {bands[('pca',X)]:<12} "
              f"{rc:>8.3f}  {bands[('cegar',X)]:<12} {agree}")
    print(f"\nRHO_ANCHOR_INVARIANT: {invariant}")
    if flips:
        print("  anchor-sensitive probes (pca vs cegar): " + ",".join(flips))
    if sc_flips:
        print("  WARNING same-cluster (cegar vs delta) flips: "
              + ",".join(sc_flips))

    # ---- exact audit (oracle revealed) -----------------------------------
    exact_cl, anchors_spread, anchors_equiv, calib, calibrated = \
        arm_b_audit(bundle, bands)
    print("\n[exact audit — revealed after observable rho-verdicts]")
    for X in PROBES:
        print(f"{X:<10} exact_closure={exact_cl[X]:.3f}")
    print(f"anchors exact-equivalent: {anchors_equiv} "
          f"(spread {anchors_spread:.3f} <= {EXACT_INDIFF_BAND})")
    print(f"rho calibrated per anchor: {calib} -> {calibrated}")

    decision = decide(gate_branch, invariant, calibrated)
    print(f"\nAUDIT_BRANCH: {decision}")
    print("\nDECISION:")
    if decision == "AMBIGUITY_DOWNSTREAM_BENIGN":
        print("  GO: the reference ambiguity is downstream-benign for rho; "
              "preregister Block 3 (battery transfer) under the declared "
              "canonical cegar-core anchor. NOTE: selection non-uniqueness "
              "(Exp 24 NO-GO) STANDS — this is not selection success.")
    elif decision == "AMBIGUITY_LOAD_BEARING":
        print("  NO-GO: rho-verdicts flip with the anchor on an "
              "oracle-unadjudicable basis; the reference-selection failure "
              "propagates into the battery. Register a reference-robust "
              "repair before transfer.")
    else:
        print("  NO-GO: rho mis-calibrated on pstack (known cases "
              "misclassified); recalibrate battery member 2 before any "
              "rho-based transfer claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
