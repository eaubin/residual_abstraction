"""
dyck_baseline.py — Experiment 19: Dyck baseline + threshold recalibration
(Phase 2, Block 1).

CONTEXT (see experiments/19-dyck-baseline.md, the pre-registration,
committed before the first run). The frozen diagnostic battery
(BATTERY.md) was calibrated entirely on Mess3. This experiment is the
battery library's first use on a new process (Dyck-2): it reproduces
exp 7's anchor numbers through battery.py, exercises members 1/2/3/5/6,
and measures the process-appropriate thresholds that Blocks 2–3 inherit.

Member 4 (shift-retention) is deferred to Block 2 (requires a
Dyck-native shift design).

Run:  python3 dyck_baseline.py --outdir out/dyck2-L4

RESULTS (see experiments/19-dyck-baseline.md): NOT YET RUN.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from abstraction import (CompletionPLS, PCAAbstraction, center_by_position,
                         kl_rows)
from battery import (Refs, Exact, calibration_gap, cegar_loop,
                     cegar_staircase)
from discover import PairSet, principal_angles_deg, self_checks
from expcommon import LAYER, load_model, validity_gate
from midstream import kl_by_horizon, marginal, orthonormal, stream_to
from processes import PROCESSES

SEED = 0
M = 3
MM = 3
K_MAX = 12
EPS = 0.05
EPS_DROP = 0.01
EPS_GRID = (0.01, 0.02, 0.05, 0.10)
TS_VAL = (12, 20)

# Exp-7 recorded numbers (reproduction targets for P1).
EXP7 = {
    "k_star": 4,
    "c_obs": 0.985,
    "full_exact_m3": 0.936,
    "nested": [0.378, 0.716, 0.850, 0.926],
}


def l_dagger_check(model, proc, cfg, m):
    """Reproduce the exp-7 ℓ† rule: argmax over interior layers of step-2
    incremental closure (ties -> smaller layer). Assert L1."""
    L, burn, V, d = cfg["seq_len"], cfg["burn_in"], proc.V, cfg["d_model"]
    interior = list(range(1, cfg["layers"]))
    ms = list(range(1, m + 1))

    evs = {l: PairSet(model, proc, cfg, 600, m, SEED + 777, 800, layer=l)
           for l in interior}
    ev1 = evs[interior[0]]
    q0 = ev1.run(model, None)
    rows_f = kl_by_horizon(q0, ev1.p_tgt3, V, m)
    rows_g = kl_by_horizon(q0, ev1.p_src3, V, m)
    floor = {mm: float(rows_f[mm].mean()) for mm in rows_f}
    gapm = {mm: float(rows_g[mm].mean()) for mm in rows_g}

    inc = {}
    for l in interior:
        qp = evs[l].run(model, np.eye(d))
        rows_t = kl_by_horizon(qp, ev1.p_src3, V, m)
        cl = {mm: (gapm[mm] - float(rows_t[mm].mean()))
              / (gapm[mm] - floor[mm]) for mm in ms}
        trf = {mm: gapm[mm] - cl[mm] * (gapm[mm] - floor[mm]) for mm in ms}
        inc[l] = {mm: ((gapm[mm] - gapm[mm - 1]) - (trf[mm] - trf[mm - 1]))
                  / ((gapm[mm] - gapm[mm - 1])
                     - (floor[mm] - floor[mm - 1])) for mm in ms[1:]}
    l_dag = min(interior, key=lambda l: (-inc[l][2], l))
    print(f"[l†] depth profile step-2 incremental:")
    for l in interior:
        print(f"  L{l}: " + " / ".join(f"step {mm}: {inc[l][mm]:.1%}"
                                        for mm in ms[1:]))
    print(f"[l†] argmax -> L{l_dag}")
    return l_dag


def build_control_bases(model, proc, cfg, Q, k_star):
    """Build the exp-7 control bases at the discovered rank."""
    L, burn, d, V = cfg["seq_len"], cfg["burn_in"], cfg["d_model"], proc.V
    rng_d = np.random.default_rng(SEED + 555)
    Xd = proc.sample(800, L, rng_d)
    Sd = stream_to(model, torch.from_numpy(Xd), LAYER).double().numpy()
    keep = np.arange(burn, L - 1)
    Gd = np.concatenate([proc.mgram_table(proc.beliefs_along(row)[keep], M)
                         for row in Xd])
    Rd = center_by_position(Sd[:, keep].reshape(-1, d),
                            np.tile(keep, len(Xd)),
                            np.ones(len(Xd) * len(keep), dtype=bool))
    pls_c = CompletionPLS(Rd, Gd)
    rng = np.random.default_rng(SEED)
    with torch.no_grad():
        W_tok = model.tok.weight.double().numpy()
    return {
        "pca": PCAAbstraction(Rd).Vt[:k_star].T,
        "pls": orthonormal(pls_c.whiten @ pls_c.U[:, :k_star]),
        "rand": orthonormal(rng.standard_normal((d, k_star))),
        "emb": orthonormal(W_tok.T)[:, :min(k_star, V)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    assert proc.name == "dyck2", f"expected dyck2, got {proc.name}"
    V, d = proc.V, cfg["d_model"]

    model = load_model(args.outdir, cfg, proc)
    gap, passed = validity_gate(model, proc, cfg, SEED)
    if not passed:
        print("HALT: validity gate failed.")
        sys.exit(1)

    # ----- l† reproduction (P1 gate) ------------------------------------------
    l_dag = l_dagger_check(model, proc, cfg, M)
    assert l_dag == LAYER, (f"l† = L{l_dag}, expected L{LAYER} "
                            "(exp-7 reproduction failed)")
    print(f"[l†] L{LAYER} confirmed — exp-7 reproduction OK\n")

    # ----- pair sets (same seeds as exp 7) ------------------------------------
    disc = PairSet(model, proc, cfg, 400, M, SEED + 111, 800, layer=LAYER)
    ev = PairSet(model, proc, cfg, 600, M, SEED + 777, 800, layer=LAYER)
    val = PairSet(model, proc, cfg, 400, M, SEED + 333, 800, layer=LAYER,
                  ts=TS_VAL)
    self_checks(model, ev, LAYER, M, V)

    print(f"=== Experiment 19: Dyck baseline | {proc.name} | "
          f"L{cfg['layers']} d{d} | S={proc.S} V={V} | m={M} mm={MM} ===\n")

    # ----- member 6: CEGAR discovery (with coarsen, eps_drop=0.01) ------------
    refs_d = Refs(disc, model, d, M)
    refs_e = Refs(ev, model, d, M)       # eval-set refs for cross-check
    refs_v = Refs(val, model, d, M)
    exact = Exact(ev, model, M)

    print("[CEGAR] discovery loop (battery.cegar_loop, eps_drop=0.01):")
    k_star, Qc, c_obs_disc = cegar_loop(model, disc, refs_d, d, EPS, K_MAX,
                                         MM, eps_drop=EPS_DROP)
    print(f"  k* = {k_star}, c_obs = {c_obs_disc:.1%}")

    # P1 reproduction checks
    p1_k = k_star == EXP7["k_star"]
    p1_c = abs(c_obs_disc - EXP7["c_obs"]) <= 0.02
    print(f"  P1 check: k*={k_star} (expect {EXP7['k_star']}) — "
          f"{'OK' if p1_k else 'FAIL'}")
    print(f"  P1 check: c_obs={c_obs_disc:.1%} (expect {EXP7['c_obs']:.1%} "
          f"± 2pts) — {'OK' if p1_c else 'FAIL'}")
    if not (p1_k and p1_c):
        print("HALT: P1 reproduction failed — battery.cegar_loop does not "
              "reproduce exp 7.")
        sys.exit(1)

    # ----- nested closure staircase (P1 continued) ----------------------------
    print("\n[nested] closure staircase (exact, eval set, mm=3):")
    nested = []
    for k in range(1, k_star + 1):
        Qk = Qc[:, :k]
        q_k = ev.run(model, Qk @ Qk.T)
        cl_k = exact.closure(q_k, MM)
        nested.append(cl_k)
        exp7_k = EXP7["nested"][k - 1]
        ok = abs(cl_k - exp7_k) <= 0.02
        print(f"  k={k}: {cl_k:.1%} (expect {exp7_k:.1%} ± 2pts) — "
              f"{'OK' if ok else 'FAIL'}")

    # ----- control bases ------------------------------------------------------
    controls = build_control_bases(model, proc, cfg, Qc, k_star)

    # ----- reference patches: build all projections ---------------------------
    Pc = Qc @ Qc.T
    patches = {"full": np.eye(d), "disc": Pc}
    for name, B in controls.items():
        patches[name] = B @ B.T

    # ----- member 1 (obs closure) + member 5 (exact closure) ------------------
    print("\n[member 1+5] observable and exact closures at mm=3:")
    print(f"  {'patch':>6}  {'k':>3}  {'obs(disc)':>10}  {'obs(eval)':>10}  "
          f"{'exact':>10}  {'gap':>6}")
    obs_scores, exact_scores, gaps = {}, {}, {}
    for name, P in patches.items():
        k = d if name == "full" else (k_star if name != "emb"
                                      else min(k_star, V))
        q_d = disc.run(model, P)
        q_e = ev.run(model, P)
        obs_d = refs_d.obs(q_d, MM)
        obs_e = refs_e.obs(q_e, MM)
        ex = exact.closure(q_e, MM)
        g = calibration_gap(obs_e, ex)
        obs_scores[name] = (obs_d, obs_e)
        exact_scores[name] = ex
        gaps[name] = g
        print(f"  {name:>6}  {k:>3}  {obs_d:>+10.1%}  {obs_e:>+10.1%}  "
              f"{ex:>+10.1%}  {g:>6.3f}")

    # no-op floor
    q_noop_e = ev.run(model, None)
    obs_noop = refs_e.obs(q_noop_e, MM)
    ex_noop = exact.closure(q_noop_e, MM)
    print(f"  {'no-op':>6}  {'-':>3}  {'-':>10}  {obs_noop:>+10.1%}  "
          f"{ex_noop:>+10.1%}  {'-':>6}")

    # ----- member 5: P4 calibration gap (obs >= 20% cells) --------------------
    print("\n[member 5] P4 calibration gaps (obs >= 20% cells):")
    accepted_cells = []
    worst_gap = 0.0
    for name in patches:
        obs_e = obs_scores[name][1]
        if obs_e >= 0.20:
            g = gaps[name]
            worst_gap = max(worst_gap, g)
            accepted_cells.append((name, obs_e, exact_scores[name], g))
            tag = "OK" if g <= 0.15 else "WIDE"
            print(f"  {name}: obs={obs_e:+.1%} exact={exact_scores[name]:+.1%}"
                  f" gap={g:.3f} — {tag}")
    print(f"  worst gap: {worst_gap:.3f}")
    if worst_gap <= 0.10:
        print("  -> Mess3 band (0.10) transfers to Dyck")
    elif worst_gap <= 0.15:
        dyck_band = round(worst_gap + 0.02, 2)
        print(f"  -> Mess3 band does not transfer; Dyck band: {dyck_band}")
    else:
        dyck_band = round(worst_gap + 0.02, 2)
        print(f"  -> obs/exact gap exceeds P2 threshold; Dyck band: "
              f"{dyck_band}")

    # ----- member 2: rho (discovered core as reference) -----------------------
    print("\n[member 2] equivalence ratio rho (reference = discovered core, "
          "mm=3):")
    q_core_e = ev.run(model, Pc)
    rhos = {}
    for name, P in patches.items():
        if name == "disc":
            continue  # skip self
        q_e = ev.run(model, P)
        r = exact.rho(q_core_e, q_e, MM)
        rhos[name] = r
        if name == "full":
            label = "(core-vs-full, expect equivalent)"
        elif name in ("pls", "rand"):
            label = "(expect distinct)"
        else:
            label = ""
        print(f"  {name:>6}: rho = {r:.4f} {label}")
    # separation check
    equiv = [n for n, r in rhos.items() if n == "full"]
    destruct = [n for n in ("pls", "rand") if n in rhos]
    if equiv and destruct:
        max_equiv = max(rhos[n] for n in equiv)
        min_destruct = min(rhos[n] for n in destruct)
        print(f"  separation: max equiv rho = {max_equiv:.4f}, "
              f"min destructive rho = {min_destruct:.4f}")

    # ----- member 3: held-out gain (val set) ----------------------------------
    print("\n[member 3] held-out-position gain (val set, ts={12,20}, mm=3):")
    q_core_v = val.run(model, Pc)
    q_full_v = val.run(model, np.eye(d))
    obs_core_val = refs_v.obs(q_core_v, MM)
    obs_full_val = refs_v.obs(q_full_v, MM)
    print(f"  discovered core: {obs_core_val:+.1%}")
    print(f"  full:            {obs_full_val:+.1%}")

    # ----- member 6: eps staircase --------------------------------------------
    print("\n[member 6] CEGAR eps staircase (mm=3):")
    staircase = cegar_staircase(model, disc, refs_d, d, EPS_GRID, K_MAX, MM)
    for eps, k in staircase.items():
        print(f"  eps={eps:.2f}: k* = {k}")
    weakly_dec = all(staircase[EPS_GRID[i + 1]] <= staircase[EPS_GRID[i]]
                     for i in range(len(EPS_GRID) - 1))

    # ----- marginal gain profile (recalibration input) ------------------------
    print("\n[recalibration] per-direction marginal gains as fraction of full:")
    obs_full_d = refs_d.obs(disc.run(model, np.eye(d)), MM)
    q_prev = refs_d.q_un
    c_prev = 0.0
    for k in range(1, k_star + 1):
        Qk = Qc[:, :k]
        q_k = disc.run(model, Qk @ Qk.T)
        c_k = refs_d.obs(q_k, MM)
        gain = c_k - c_prev
        frac = gain / obs_full_d
        print(f"  direction {k}: gain {gain:+.1%} "
              f"(= {frac:.1%} of full-patch obs)")
        c_prev = c_k

    # ----- principal angles ---------------------------------------------------
    print("\n[angles] discovered vs controls (deg):")
    for name in ("pca", "pls", "emb"):
        ang = principal_angles_deg(Qc, controls[name])
        print(f"  disc vs {name:>4}: "
              + ", ".join(f"{a:.1f}" for a in ang))

    # ===== verdicts ===========================================================
    print("\n" + "=" * 60)
    print("VERDICTS\n")

    # P1: exp-7 reproduction
    nested_ok = all(abs(nested[i] - EXP7["nested"][i]) <= 0.02
                    for i in range(len(EXP7["nested"])))
    p1 = p1_k and p1_c and nested_ok
    print(f"P1 (exp-7 reproduction): {'HOLDS' if p1 else 'FAILS'}")
    if not p1:
        print("  HALT — the library does not reproduce exp 7.")

    # P2: obs/exact agreement
    p2 = worst_gap <= 0.15
    print(f"P2 (obs/exact <= 0.15): worst gap {worst_gap:.3f} — "
          f"{'HOLDS' if p2 else 'FAILS (measured-band protocol)'}")

    # P3: rho separates
    p3_equiv = rhos.get("full", 999) <= 0.25
    p3_pls = rhos.get("pls", 0) >= 0.50
    p3_rand = rhos.get("rand", 0) >= 0.50
    p3 = p3_equiv and p3_pls and p3_rand
    print(f"P3 (rho separates): full={rhos.get('full', '?'):.4f} "
          f"pls={rhos.get('pls', '?'):.4f} rand={rhos.get('rand', '?'):.4f}"
          f" — {'HOLDS' if p3 else 'FAILS (band recalibration)'}")

    # P4: controls
    p4_rand = exact_scores["rand"] <= 0.25
    p4_pls = exact_scores["pls"] <= 0.05
    p4_full = abs(exact_scores["full"] - EXP7["full_exact_m3"]) <= 0.02
    p4 = p4_rand and p4_pls and p4_full
    print(f"P4 (controls): rand={exact_scores['rand']:.1%} "
          f"pls={exact_scores['pls']:.1%} "
          f"full={exact_scores['full']:.1%} — "
          f"{'HOLDS' if p4 else 'FAILS'}")

    # P5: val-set baseline (descriptive)
    print(f"P5 (val baseline, descriptive): core={obs_core_val:+.1%} "
          f"full={obs_full_val:+.1%}")

    # P6: eps staircase
    p6_dec = weakly_dec
    p6_fine = staircase[0.01] <= 8
    p6 = p6_dec and p6_fine
    print(f"P6 (eps staircase): weakly decreasing={p6_dec}, "
          f"k*(0.01)={staircase[0.01]} <= 8 — {'HOLDS' if p6 else 'FAILS'}")

    print(f"\n{'=' * 60}")
    all_hold = p1 and p4 and p6  # P2/P3 have recalibration protocols, not halt
    if all_hold:
        print("Block 1 gates passed.")
    else:
        print("Block 1 gate FAILED — see verdicts above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
