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
SEP_MIN = 0.25                           # usable rho separation between extremes
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


def calibration_verdict(equiv_max, rand_min, separation):
    """Called only under REFERENCE_COHERENT, so exact says the estimates are
    equivalent and `equiv_max` reflects whether OBSERVABLE ρ agrees. Splits
    the two opposite ρ failures (cf. exp-25 decide()), which must not collapse
    into one band-widening GO:

      RHO_OVERSENSITIVE   ρ reads exact-equivalent estimates as non-equivalent
                          (equiv_max > 0.25) — the member-2 mean-Jeffreys
                          failure; band-widening would paper over it. NO-GO.
      BANDS_TRANSFER      estimates equivalent AND rand distinct as transferred.
      BANDS_RECALIBRATE   estimates equivalent, rand under 0.5 but a usable gap
                          remains — lower the distinct floor (per-process).
      RHO_NONSEPARATING   no usable gap even between the extreme cases. NO-GO.
    """
    if equiv_max > RHO_EQUIV:
        return "RHO_OVERSENSITIVE"
    if rand_min >= RHO_DIST:
        return "BANDS_TRANSFER"
    if separation >= SEP_MIN:
        return "BANDS_RECALIBRATE"
    return "RHO_NONSEPARATING"


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


def decide(coherent, calib):
    """Coherence (exact, one reference) gates first; then the calibration
    verdict stands. GO set: {BANDS_TRANSFER, BANDS_RECALIBRATE}. NO-GO:
    {REFERENCE_INCOHERENT, RHO_OVERSENSITIVE, RHO_NONSEPARATING}."""
    if not coherent:
        return "REFERENCE_INCOHERENT"
    return calib


def recalibrated_floor(equiv_max, rand_min):
    """Block-3 distinct-band floor when recalibrating: the midpoint of the
    observed envelope (equivalent stays <= 0.25, which held)."""
    return round((equiv_max + rand_min) / 2.0, 3)


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

    # calibration: the two opposite failures must NOT collapse to one GO
    assert calibration_verdict(0.18, 0.99, 0.81) == "BANDS_TRANSFER"
    assert calibration_verdict(0.30, 0.99, 0.69) == "RHO_OVERSENSITIVE"  # splits equiv
    assert calibration_verdict(0.18, 0.45, 0.27) == "BANDS_RECALIBRATE"  # gap usable
    assert calibration_verdict(0.18, 0.40, 0.22) == "RHO_NONSEPARATING"  # no gap
    assert recalibrated_floor(0.18, 0.46) == 0.32

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

    assert decide(True, "BANDS_TRANSFER") == "BANDS_TRANSFER"
    assert decide(True, "RHO_OVERSENSITIVE") == "RHO_OVERSENSITIVE"
    assert decide(False, "BANDS_TRANSFER") == "REFERENCE_INCOHERENT"
    print("selftest passed: rho, mimicry, calibration, coherence, decide")


def print_decision(decision, equiv_max, rand_min):
    print("\nDECISION:")
    if decision == "BANDS_TRANSFER":
        print("  GO: reference coherent and ρ separates the extreme known "
              "cases as the transferred 0.25/0.5 bands require; preregister "
              "Block 3 (exp 27) under the cegar core. The transferred bands "
              "are retained (their VALUES stay Dyck-inherited — pstack has no "
              "intermediate known-case to pin the thresholds; see scope).")
    elif decision == "BANDS_RECALIBRATE":
        floor = recalibrated_floor(equiv_max, rand_min)
        print("  GO: reference coherent; ρ separates the extremes but rand "
              f"reads below 0.5 (rand_min {rand_min:.3f}). Registered pstack "
              f"bands for Block 3: equivalent <= {RHO_EQUIV} (holds, "
              f"equiv_max {equiv_max:.3f}), distinct >= {floor} (recalibrated "
              f"distinct floor, was 0.5).")
    elif decision == "RHO_OVERSENSITIVE":
        print("  NO-GO: ρ reads the exact-equivalent reference estimates as "
              f"non-equivalent (equiv_max {equiv_max:.3f} > {RHO_EQUIV}) — a "
              "member-2 mean-Jeffreys failure. Block 3's ρ cannot be trusted; "
              "do NOT widen the band. Investigate per-pair vs mean-level ρ "
              "before any ρ-based transfer.")
    elif decision == "RHO_NONSEPARATING":
        print("  NO-GO: ρ does not usefully separate even the extreme known "
              f"cases (separation < {SEP_MIN}); no band can be drawn. "
              "Recalibrate the substrate/ρ construction before transfer.")
    else:  # REFERENCE_INCOHERENT
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
    coh = coherence_verdict(summaries)
    print(f"reference coherence: strong={coh['strong']} indiff={coh['indiff']}"
          f" cegar_stable={coh['stable']} (cegar closure mean "
          f"{coh['cegar_mean']:.3f} std {coh['cegar_std']:.3f})")
    print(f"COHERENCE_VERDICT: {coh['verdict']}")
    # Calibration is read only under coherence (so equiv_max reflects whether
    # observable rho agrees with exact-known equivalence, not incoherence).
    calib = calibration_verdict(b["equiv_max"], b["rand_min"],
                                b["separation"]) \
        if coh["verdict"] == "REFERENCE_COHERENT" else "n/a"
    print(f"CALIBRATION_VERDICT: {calib}")

    decision = decide(coh["verdict"] == "REFERENCE_COHERENT", calib)
    print(f"\nAUDIT_BRANCH: {decision}  (mimicry={a['verdict']})")
    print_decision(decision, b["equiv_max"], b["rand_min"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
