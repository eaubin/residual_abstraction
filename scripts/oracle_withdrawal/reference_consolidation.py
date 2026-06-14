"""
Oracle-withdrawal experiment 4: pstack reference-and-rho consolidation.

Exp 25 dissolved the exp-24 reference ambiguity (one stable ~k=4 reference)
but gated three checks behind STRUCTURAL_PASS that never ran. This step runs
the two that are genuinely unfinished and block any rho-based battery claim,
MULTI-SEED from the start (the durable lesson from exp 25 is that single-seed
geometry on pstack is unreliable):

  Arm A  is pstack a mimicry process? cegar-vs-pca principal-angle statistics
         over fresh seeds (MIMICRY / PARTIAL_MIMICRY / DISTINCT).
  Arm B  do the Dyck-transferred rho bands (0.25/0.5) hold on pstack, and are
         the reference estimates exact-coherent (one reference)?

Observable quantities (angles, rho) are computed and printed before the exact
reveal. The exact oracle is used for calibration only (the per-process
calibration the program permits in a measurement step, cf. exp 19); bands are
checked as transferred first, recalibrated only if they fail. No Block-3
abstraction verdict is computed here. Candidate construction is reused from
exp 24 by import; the frozen exp-24/25 scripts are not edited.
"""

import argparse
import json
import os
import sys
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
    M, MM, PAIR_POOL, PAIRS_DISC, PAIRS_EVAL, TS_DISC, build_candidates,
    max_principal_angle, require_registered_config,
)

SEEDS = tuple(range(100, 108))          # 8 fresh seeds, disjoint from exp 25
ANCHOR = "cegar"
PROBES = ("cegar", "pca", "delta", "emb", "rand", "full")
COMPACT = ("cegar", "pca", "delta")
MIMICRY_ANGLE = 10.0                     # exp-24 AMBIG_ANGLE_MAX
MESS3_MIMICRY = 3.5                      # reference pole (cite)
RHO_EQUIV = 0.25                         # battery member-2 bands (under test)
RHO_DIST = 0.50
EXACT_MIN = 0.70                         # estimates must be strong
EXACT_INDIFF_BAND = 0.05                 # per-seed exact spread over estimates
CEGAR_STD_MAX = 0.05                     # anchor stability across seeds


def _mnorm(q, V, mm, m_full):
    qm = marginal(q, V, mm, m_full)
    return qm / np.clip(qm.sum(axis=1, keepdims=True), 1e-30, None)


def rho_obs(q0, qC, qX, V, mm, m_full):
    """Observable per-pair equivalence ratio (battery member 2) without the
    process oracle: mean J(q_C,q_X) / mean J(q_C,q_unpatched)."""
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


def run_seed(model, proc, cfg, seed):
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + 111, PAIR_POOL,
                   layer=LAYER, ts=TS_DISC)
    eval_ps = PairSet(model, proc, cfg, PAIRS_EVAL, M, seed + 222, PAIR_POOL,
                      layer=LAYER, ts=TS_DISC)
    self_checks(model, eval_ps, LAYER, M, proc.V)

    refs_d = Refs(disc, model, disc.d, M)
    refs_e = Refs(eval_ps, model, eval_ps.d, M)
    candidates, _, k_ref = build_candidates(model, proc, cfg, disc, refs_d,
                                            seed)
    Q = {n: candidates[n]["Q"] for n in COMPACT}
    angles = {
        ("cegar", "pca"): max_principal_angle(Q["cegar"], Q["pca"]),
        ("cegar", "delta"): max_principal_angle(Q["cegar"], Q["delta"]),
        ("pca", "delta"): max_principal_angle(Q["pca"], Q["delta"]),
        ("rand", "pca"): max_principal_angle(candidates["rand"]["Q"], Q["pca"]),
    }

    q_un = refs_e.q_un
    q_eval = {X: eval_ps.run(model, candidates[X]["P"]) for X in PROBES}
    rho = {X: rho_obs(q_un, q_eval[ANCHOR], q_eval[X], proc.V, MM, M)
           for X in PROBES}

    # Exact is needed every seed for calibration (not wasted); quarantined in
    # the sense that observable angles/rho are printed before it is read.
    exact = Exact(eval_ps, model, M)
    exact_cl = {X: exact.closure(q_eval[X], MM) for X in PROBES}
    return {"seed": seed, "angles": angles, "rho": rho, "exact_cl": exact_cl,
            "k_ref": k_ref}


def aggregate_arm_a(summaries):
    cp = np.array([s["angles"][("cegar", "pca")] for s in summaries])
    cd = np.array([s["angles"][("cegar", "delta")] for s in summaries])
    pd = np.array([s["angles"][("pca", "delta")] for s in summaries])
    null = np.array([s["angles"][("rand", "pca")] for s in summaries])
    cp_min, cp_max = float(cp.min()), float(cp.max())
    if cp_max <= MIMICRY_ANGLE:
        verdict = "MIMICRY"
    elif cp_min > MIMICRY_ANGLE:
        verdict = "DISTINCT"
    else:
        verdict = "PARTIAL_MIMICRY"
    return {"cp": cp, "cd": cd, "pd": pd, "null": null, "verdict": verdict,
            "cp_min": cp_min, "cp_max": cp_max}


def aggregate_arm_b_obs(summaries):
    equiv = [max(s["rho"]["pca"], s["rho"]["delta"]) for s in summaries]
    rand = [s["rho"]["rand"] for s in summaries]
    equiv_max = max(equiv)
    rand_min = min(rand)
    return {"equiv_max": equiv_max, "rand_min": rand_min,
            "separation": rand_min - equiv_max}


def calibration_verdict(equiv_max, rand_min):
    if equiv_max <= RHO_EQUIV and rand_min >= RHO_DIST:
        return "BANDS_TRANSFER"
    return "BANDS_RECALIBRATE"


def coherence_verdict(summaries):
    spreads = [max(s["exact_cl"][n] for n in COMPACT)
               - min(s["exact_cl"][n] for n in COMPACT) for s in summaries]
    cegar_cl = np.array([s["exact_cl"]["cegar"] for s in summaries])
    strong = all(s["exact_cl"][n] >= EXACT_MIN
                 for s in summaries for n in COMPACT)
    indiff = all(sp <= EXACT_INDIFF_BAND for sp in spreads)
    stable = float(cegar_cl.std()) <= CEGAR_STD_MAX
    verdict = "REFERENCE_COHERENT" if (strong and indiff and stable) \
        else "REFERENCE_INCOHERENT"
    return {"verdict": verdict, "spreads": spreads, "strong": strong,
            "indiff": indiff, "stable": stable,
            "cegar_mean": float(cegar_cl.mean()),
            "cegar_std": float(cegar_cl.std())}


def decide(coherent, calibrated):
    if not coherent:
        return "REFERENCE_INCOHERENT"
    return "GO_TRANSFER" if calibrated else "GO_RECALIBRATE"


def selftest():
    assert rho_band(0.1) == "equivalent" and rho_band(0.6) == "distinct"
    assert rho_band(0.4) == "indeterminate"
    rng = np.random.default_rng(0)
    q0 = rng.dirichlet(np.ones(8), size=40)
    qC = rng.dirichlet(np.ones(8), size=40)
    assert rho_obs(q0, qC, qC, 8, 1, 1) < 1e-9
    assert rho_obs(q0, qC, rng.dirichlet(0.1 * np.ones(8), size=40),
                   8, 1, 1) > 0.0

    def _s(cp, cd, pd, nul, cl):
        return {"angles": {("cegar", "pca"): cp, ("cegar", "delta"): cd,
                           ("pca", "delta"): pd, ("rand", "pca"): nul},
                "rho": {}, "exact_cl": cl}
    cl = {"cegar": 0.92, "pca": 0.92, "delta": 0.92, "emb": 0.77,
          "rand": 0.05, "full": 1.0}
    # mimicry verdict
    assert aggregate_arm_a([_s(3.0, 3.0, 3.0, 80, cl)] * 3)["verdict"] == \
        "MIMICRY"
    assert aggregate_arm_a([_s(20, 4, 22, 80, cl)] * 3)["verdict"] == "DISTINCT"
    assert aggregate_arm_a([_s(8, 4, 9, 80, cl), _s(13, 4, 14, 80, cl)]
                           )["verdict"] == "PARTIAL_MIMICRY"

    # calibration
    assert calibration_verdict(0.18, 0.99) == "BANDS_TRANSFER"
    assert calibration_verdict(0.30, 0.99) == "BANDS_RECALIBRATE"  # equiv high
    assert calibration_verdict(0.18, 0.40) == "BANDS_RECALIBRATE"  # dist low

    # coherence
    def _cs(cl):
        return {"exact_cl": cl}
    coh = coherence_verdict([_cs({"cegar": 0.92, "pca": 0.93, "delta": 0.91,
                                  "emb": 0, "rand": 0, "full": 1})] * 4)
    assert coh["verdict"] == "REFERENCE_COHERENT"
    incoh = coherence_verdict([_cs({"cegar": 0.92, "pca": 0.70, "delta": 0.91,
                                    "emb": 0, "rand": 0, "full": 1})] * 4)
    assert incoh["verdict"] == "REFERENCE_INCOHERENT"          # spread 0.22
    # cegar unstable across seeds -> incoherent
    unstable = coherence_verdict([
        _cs({"cegar": 0.95, "pca": 0.95, "delta": 0.95, "e": 0, "rand": 0,
             "full": 1}),
        _cs({"cegar": 0.80, "pca": 0.80, "delta": 0.80, "e": 0, "rand": 0,
             "full": 1})])
    assert unstable["verdict"] == "REFERENCE_INCOHERENT"

    assert decide(True, True) == "GO_TRANSFER"
    assert decide(True, False) == "GO_RECALIBRATE"
    assert decide(False, True) == "REFERENCE_INCOHERENT"
    print("selftest passed: rho, mimicry, calibration, coherence, decide")


def print_decision(decision, calib):
    print("\nDECISION:")
    if decision == "GO_TRANSFER":
        print("  GO: reference coherent and Dyck rho bands transfer to "
              "pstack; preregister Block 3 (exp 27) battery transfer under "
              "the cegar core with the transferred 0.25/0.5 bands.")
    elif decision == "GO_RECALIBRATE":
        print("  GO: reference coherent but the Dyck rho bands do not "
              "cleanly separate on pstack; register the recalibrated pstack "
              f"bands (separation {calib['separation']:.3f}) and proceed to "
              "Block 3 (exp 27) with them.")
    else:
        print("  NO-GO: reference estimates are not exact-coherent (not one "
              "reference, or cegar unstable); reopen reference selection "
              "before any battery transfer.")


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

    print("=== Experiment 26: pstack reference-and-rho consolidation ===")
    print(f"target={proc.name} outdir={args.outdir} m={M} LAYER={LAYER}")
    print(f"seeds={SEEDS} anchor={ANCHOR}")
    print("Exact oracle is read only after observable angles/rho are "
          "printed (calibration use).\n")

    summaries = [run_seed(model, proc, cfg, s) for s in SEEDS]

    print("[observable per-seed table — before exact reveal]")
    print("seed a(c,p) a(c,d) a(p,d) | rho(cegar,X): pca   delta  emb    "
          "rand   full")
    for s in summaries:
        a, r = s["angles"], s["rho"]
        print(f" {s['seed']:>3}  {a[('cegar','pca')]:>5.1f} "
              f"{a[('cegar','delta')]:>5.1f} {a[('pca','delta')]:>5.1f} | "
              f"{r['pca']:>15.3f} {r['delta']:>5.3f} {r['emb']:>5.3f} "
              f"{r['rand']:>5.3f} {r['full']:>5.3f}")

    a = aggregate_arm_a(summaries)
    print(f"\n[Arm A — reference geometry] cegar-pca: mean "
          f"{a['cp'].mean():.1f} std {a['cp'].std():.1f} range "
          f"[{a['cp_min']:.1f}, {a['cp_max']:.1f}] deg; "
          f"null(rand-pca) mean {a['null'].mean():.1f}; Mess3 pole "
          f"{MESS3_MIMICRY}")
    print(f"MIMICRY_VERDICT: {a['verdict']}")

    b = aggregate_arm_b_obs(summaries)
    print(f"\n[Arm B — observable rho under cegar] equiv_max "
          f"{b['equiv_max']:.3f} (<= {RHO_EQUIV}?), rand_min "
          f"{b['rand_min']:.3f} (>= {RHO_DIST}?), separation "
          f"{b['separation']:.3f}")

    print("\n[exact reveal — calibration only]")
    print("seed exact_closure: cegar  pca    delta  emb    rand   full")
    for s in summaries:
        e = s["exact_cl"]
        print(f" {s['seed']:>3}             {e['cegar']:.3f}  {e['pca']:.3f}  "
              f"{e['delta']:.3f}  {e['emb']:.3f}  {e['rand']:.3f}  "
              f"{e['full']:.3f}")
    calib = calibration_verdict(b["equiv_max"], b["rand_min"])
    calib_d = {"separation": b["separation"]}
    print(f"CALIBRATION_VERDICT: {calib}")
    coh = coherence_verdict(summaries)
    print(f"reference coherence: strong={coh['strong']} indiff={coh['indiff']}"
          f" cegar_stable={coh['stable']} (cegar closure mean "
          f"{coh['cegar_mean']:.3f} std {coh['cegar_std']:.3f})")
    print(f"COHERENCE_VERDICT: {coh['verdict']}")

    decision = decide(coh["verdict"] == "REFERENCE_COHERENT",
                      calib == "BANDS_TRANSFER")
    print(f"\nAUDIT_BRANCH: {decision}  (mimicry={a['verdict']}, "
          f"calibration={calib})")
    print_decision(decision, calib_d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
