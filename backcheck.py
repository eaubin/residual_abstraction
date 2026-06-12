"""
backcheck.py — Mess3 regression harness for battery.py.

Reruns the diagnostic battery through the new library and reproduces
recorded exp-17/18 numbers. The frozen output files are the regression
targets; this script exercises every battery.py export at least once.

Exercised battery.py symbols:
  Refs          — observable closure (member 1) + held-out gain (member 3)
  Exact         — exact closure (member 5) + rho at both poles (member 2)
  calibration_gap — P4 obs/exact agreement (member 5)
  shift_retention — member 4 (interface confirmation; the formula is
                    trivial — member 4 has no end-to-end exercise here)
  cegar_loop    — benign CEGAR (member 6)
  cegar_accept  — adversarial accept-count (member 6)
  cegar_staircase — k*(eps) curve (member 6)
  jeffreys_rows — used internally by rho

Representative subset from exp 18: anchor reproduction (k*=2,
c_obs=99.8%), benign staircase (k*=2 at mm=3), adversarial accept-count
(0 at kappa=100), observable/exact closures and rho on the clean D2 and
id patches (equivalent and destructive poles), held-out val gain on the
clean patch.
From exp 17: benign eps staircase (k*=2 at all eps).

Run: python3 backcheck.py --outdir out/mess3-L4   (~10-15 min)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from adversarial import ZView
from battery import (Refs, Exact, calibration_gap, shift_retention,
                     cegar_loop, cegar_accept, cegar_staircase)
from discover import PairSet, principal_angles_deg, self_checks
from expcommon import (LAYER, build_transform, load_model, validity_gate,
                       basis_covariance, adversarial_regime, oblique_patch,
                       orthonormal)
from midstream import kl_by_horizon
from patches import write_pool
from processes import PROCESSES

SEED = 0
M = 3
MM = 3
EPS = 0.05
KAPPA = 100.0
TS_STD = (8, 16, 24)
TS_VAL = (12, 20)
EPS_GRID = (0.01, 0.02, 0.05, 0.10)

# Recorded numbers from the tracked output files (tolerances chosen to
# match the experiment scripts' own reproduction checks).
EXPECT = {
    "anchor_k": 2,
    "anchor_c_obs": 0.998,
    "full_exact_mm3": 0.987,
    "benign_k_mm3": 2,
    "benign_eps_staircase": {0.01: 2, 0.02: 2, 0.05: 2, 0.10: 2},
    "adv_accept_k100_mm3": 0,
    # from exp-18 staircase table (mm=3 column)
    "clean_obs_mm3": 0.51,
    "clean_exact_mm3": 0.51,
    "D2_obs_mm3": 0.99,
    "D2_exact_mm3": 0.98,
    "id_obs_mm3": 0.01,
    "id_exact_mm3": 0.01,
    "id_rho_mm3": 0.99,
    "idw2_rho_mm3": 5.31,
    # val gain on clean at mm=3 (from mstair output, benign Pc ~ +100%)
    "Pc_obs_mm3": 1.00,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/mess3-L4")
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    assert proc.name == "mess3", f"back-check is for mess3, got {proc.name}"
    V, d = proc.V, cfg["d_model"]

    model = load_model(args.outdir, cfg, proc)
    gap, passed = validity_gate(model, proc, cfg, SEED)
    assert passed, f"validity gate failed: {gap}"

    # ----- pair sets (same seeds as exp 17/18) ----------------------------------
    disc = PairSet(model, proc, cfg, 400, M, SEED + 111, 800, layer=LAYER)
    ev = PairSet(model, proc, cfg, 600, M, SEED + 777, 800, layer=LAYER)
    val = PairSet(model, proc, cfg, 400, M, SEED + 333, 800, layer=LAYER,
                  ts=TS_VAL)
    self_checks(model, ev, LAYER, M, V)

    # ----- build battery objects -------------------------------------------------
    refs_d = Refs(disc, model, d, M)
    refs_v = Refs(val, model, d, M)
    exact = Exact(ev, model, M)
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        match = abs(got - want) <= tol
        tag = "OK" if match else "FAIL"
        print(f"  {name}: {got:.4f} (expect {want}, tol {tol}) — {tag}")
        if not match:
            ok = False

    print("=== battery.py back-check against exp-17/18 recorded numbers ===\n")

    # ----- member 1: observable closure (Refs.obs) + member 6: CEGAR loop -------
    print("[anchor] benign CEGAR loop:")
    k_star, Qc, c_obs_anchor = cegar_loop(model, disc, refs_d, d, EPS, 8, MM)
    check("anchor k*", k_star, EXPECT["anchor_k"], 0)
    check("anchor c_obs", c_obs_anchor, EXPECT["anchor_c_obs"], 0.005)

    # ----- member 1: observable closure on patches ------------------------------
    print("\n[staircase] observable + exact closures at mm=3:")
    Pc = Qc @ Qc.T

    # full patch
    q_full_e = ev.run(model, np.eye(d))
    cl_full = exact.closure(q_full_e, MM)
    check("full exact mm=3", cl_full, EXPECT["full_exact_mm3"], 0.005)

    # benign Pc
    q_Pc_d = disc.run(model, Pc)
    obs_Pc = refs_d.obs(q_Pc_d, MM)
    check("Pc obs mm=3", obs_Pc, EXPECT["Pc_obs_mm3"], 0.02)

    # adversarial regime (kappa=100, draw 0)
    Sig_x = basis_covariance(model, proc, cfg, SEED, 800)
    T, Tinv, Qj = build_transform(Qc, d, KAPPA, junk_seed=0)
    rg, Sig_z = adversarial_regime(disc, T, Tinv, Sig_x)
    w0w = kl_by_horizon(refs_d.q_un, refs_d.q_src, V, M)[M]
    pool = write_pool(rg, np.zeros((d, 0)), w0w, 1, d, SEED)
    back_u = lambda w: (lambda u: u / np.linalg.norm(u))(rg.back(w))
    angled = sorted((principal_angles_deg(back_u(w)[:, None], Qc)[0],
                     src, w) for src, w in pool)
    near = [(a, s, w) for a, s, w in angled if a <= 15.0][:2]
    assert len(near) >= 2, "fewer than two near-plane writes"
    (a1, s1, w1), (a2, s2, w2) = near[0], near[1]
    u1, u2 = back_u(w1), back_u(w2)
    pb = lambda c, w: rg.pull(oblique_patch(c[:, None], w[:, None]))

    # clean patch (rank-1, write 1)
    P_clean = np.outer(u1, u1)
    q_clean_d = disc.run(model, P_clean)
    q_clean_e = ev.run(model, P_clean)
    obs_cl = refs_d.obs(q_clean_d, MM)
    ex_cl = exact.closure(q_clean_e, MM)
    check("clean obs mm=3", obs_cl, EXPECT["clean_obs_mm3"], 0.02)
    check("clean exact mm=3", ex_cl, EXPECT["clean_exact_mm3"], 0.02)

    # D2 (clean rank-2 composition)
    U = orthonormal(np.column_stack([u1, u2]))
    P_D2 = U @ U.T
    q_D2_d = disc.run(model, P_D2)
    q_D2_e = ev.run(model, P_D2)
    obs_d2 = refs_d.obs(q_D2_d, MM)
    ex_d2 = exact.closure(q_D2_e, MM)
    check("D2 obs mm=3", obs_d2, EXPECT["D2_obs_mm3"], 0.02)
    check("D2 exact mm=3", ex_d2, EXPECT["D2_exact_mm3"], 0.02)

    # id patch (write=read=w1 in z-coordinates)
    P_id = pb(w1 / float(w1 @ w1), w1)
    q_id_d = disc.run(model, P_id)
    q_id_e = ev.run(model, P_id)
    obs_id = refs_d.obs(q_id_d, MM)
    ex_id = exact.closure(q_id_e, MM)
    check("id obs mm=3", obs_id, EXPECT["id_obs_mm3"], 0.02)
    check("id exact mm=3", ex_id, EXPECT["id_exact_mm3"], 0.02)

    # id-z(w2): destructive patch (write=read=w2 in z-coordinates)
    P_idw2 = pb(w2 / float(w2 @ w2), w2)
    q_idw2_e = ev.run(model, P_idw2)

    # ----- member 2: rho (Exact.rho) — equivalent and destructive poles ---------
    print("\n[rho] per-pair equivalence ratio:")
    q_clean2_e = ev.run(model, np.outer(u2, u2))
    rho_id = exact.rho(q_clean_e, q_id_e, MM)
    check("id rho mm=3 (destructive, same-write clean ref)", rho_id,
          EXPECT["id_rho_mm3"], 0.05)
    rho_idw2 = exact.rho(q_clean2_e, q_idw2_e, MM)
    check("id-z(w2) rho mm=3 (destructive pole)", rho_idw2,
          EXPECT["idw2_rho_mm3"], 0.50)

    # ----- member 3: held-out-position gain (Refs.obs on val set) ---------------
    print("\n[held-out] val-set observable closure:")
    q_clean_v = val.run(model, P_clean)
    obs_val = refs_v.obs(q_clean_v, MM)
    print(f"  clean val obs mm=3: {obs_val:+.1%} (descriptive; confirms "
          f"Refs works on a held-out pair set)")

    # ----- member 4: shift_retention (interface confirmation) -------------------
    print("\n[shift-retention] interface confirmation:")
    R = shift_retention(0.40, 0.51, 0.45, 0.51)
    R_expect = (0.40 / 0.51) / (0.45 / 0.51)
    assert abs(R - R_expect) < 1e-12, f"shift_retention formula: {R} != {R_expect}"
    print(f"  R(0.40, 0.51, 0.45, 0.51) = {R:.4f} — OK")

    # ----- member 5: calibration gap (P4 protocol) -----------------------------
    print("\n[P4] obs/exact calibration gaps on accepted cells:")
    for name, obs_s, ex_s in [("clean", obs_cl, ex_cl),
                               ("D2", obs_d2, ex_d2)]:
        g = calibration_gap(obs_s, ex_s)
        tag = "OK" if g <= 0.10 else "FAIL"
        print(f"  {name}: gap {g:.3f} — {tag}")
        if g > 0.10:
            ok = False

    # ----- member 6: CEGAR staircase + adversarial accept -----------------------
    print("\n[CEGAR] benign k* at mm=3:")
    check("benign k* mm=3", cegar_loop(model, disc, refs_d, d, EPS, 8, MM)[0],
          EXPECT["benign_k_mm3"], 0)

    print("\n[CEGAR] benign eps staircase (exp-17 numbers):")
    staircase = cegar_staircase(model, disc, refs_d, d, EPS_GRID, 8, MM)
    for eps, k in staircase.items():
        want = EXPECT["benign_eps_staircase"][eps]
        tag = "OK" if k == want else "FAIL"
        print(f"  eps {eps}: k* = {k} (expect {want}) — {tag}")
        if k != want:
            ok = False

    print("\n[CEGAR] adversarial accept-count (kappa=100, draw 0, mm=3):")
    adv_view = ZView(disc, T)
    adv_k = cegar_accept(model, adv_view, rg.pull, disc, refs_d, MM, d,
                         EPS, 4)
    check("adv accept k100 mm=3", adv_k, EXPECT["adv_accept_k100_mm3"], 0)

    # ----- summary ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    if ok:
        print("BACK-CHECK PASSED — all recorded numbers reproduced.")
    else:
        print("BACK-CHECK FAILED — see FAIL lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
