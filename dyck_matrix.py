"""
dyck_matrix.py — Experiment 20: interventional battery matrix
(Phase 2, Block 2).

CONTEXT (see experiments/20-dyck-matrix.md). Exp 19 established the
Dyck baseline. This experiment tests the battery under adversarial
coordinates, distribution shifts, and depth stratification, and
exercises member 4 (shift-retention), deferred from Block 1.

Run:  uv run python3 dyck_matrix.py --outdir out/dyck2-L4

RESULTS (see experiments/20-dyck-matrix.md): P1–P6 hold, P7 fails
(single-write rank-1 probe failed), P8 skipped. Block 2 gates passed.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from adversarial import ZView
from battery import (Refs, Exact, calibration_gap, shift_retention,
                     cegar_loop, cegar_accept)
from discover import PairSet, principal_angles_deg, self_checks
from expcommon import (LAYER, load_model, validity_gate, basis_covariance,
                       build_transform, adversarial_regime,
                       make_torch_objective, optimize_affine,
                       oblique_patch, orthonormal)
from midstream import kl_by_horizon
from patches import write_pool
from processes import PROCESSES

SEED = 0
M = 3
MM = 3
K_MAX = 12
EPS = 0.05
EPS_DROP = 0.01
KAPPA = 100.0
TS_TEST = (10, 14, 22)
INIT_DEPTH2 = 3          # state index of stack (0,0) in dyck2(depth=3)

# Exp-19 reproduction targets.
EXP19 = {"k_star": 4, "c_obs": 0.985}


def bracket_depths(ev):
    """For each eval pair, compute bracket depth at its position."""
    depths = np.zeros(ev.n, dtype=int)
    for t, idx in ev.groups:
        if t == 0:
            depths[idx] = 0
        else:
            seqs = ev.Xe[ev.a[idx], :t]
            depths[idx] = np.sum(seqs < 2, axis=1) - np.sum(seqs >= 2, axis=1)
    return depths


def per_depth_closure(ev, model, q, depths, mm):
    """Per-depth-stratum exact closure at horizon mm."""
    q0 = ev.run(model, None)
    kl_src = kl_by_horizon(q, ev.p_src3, ev.V, M)[mm]
    kl_un_src = kl_by_horizon(q0, ev.p_src3, ev.V, M)[mm]
    kl_un_tgt = kl_by_horizon(q0, ev.p_tgt3, ev.V, M)[mm]
    results = {}
    for d_val in sorted(np.unique(depths)):
        mask = depths == d_val
        gap = float(kl_un_src[mask].mean())
        floor = float(kl_un_tgt[mask].mean())
        t = float(kl_src[mask].mean())
        if gap > floor + 1e-6:
            results[d_val] = (gap - t) / (gap - floor)
        else:
            results[d_val] = float('nan')
    return results


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

    # ----- reproduce exp-19 core (gate) ---------------------------------------
    disc = PairSet(model, proc, cfg, 400, M, SEED + 111, 800, layer=LAYER)
    ev = PairSet(model, proc, cfg, 600, M, SEED + 777, 800, layer=LAYER)
    self_checks(model, ev, LAYER, M, V)

    refs_d = Refs(disc, model, d, M)
    exact_e = Exact(ev, model, M)

    print("[anchor] reproduce exp-19 core:")
    k_star, Qc, c_obs_disc = cegar_loop(model, disc, refs_d, d, EPS, K_MAX,
                                         MM, eps_drop=EPS_DROP)
    print(f"  k* = {k_star}, c_obs = {c_obs_disc:.1%}")
    p1_ok = (k_star == EXP19["k_star"]
             and abs(c_obs_disc - EXP19["c_obs"]) <= 0.02)
    if not p1_ok:
        print("HALT: exp-19 core reproduction failed.")
        sys.exit(1)
    print("  reproduction OK\n")

    Pc = Qc @ Qc.T
    print(f"=== Experiment 20: Dyck battery matrix | {proc.name} | "
          f"L{cfg['layers']} d{d} | κ={KAPPA} ===\n")

    # ===== ARM A: Adversarial regime ==========================================
    print("=" * 60)
    print("ARM A: Adversarial regime (κ=100)\n")

    Sig_x = basis_covariance(model, proc, cfg, SEED, 800)
    T, Tinv, Qj = build_transform(Qc, d, KAPPA, junk_seed=0)
    rg, Sig_z = adversarial_regime(disc, T, Tinv, Sig_x)

    # Member 6: CEGAR accept-count
    adv_view = ZView(disc, T)
    adv_k = cegar_accept(model, adv_view, rg.pull, disc, refs_d, MM, d,
                         EPS, K_MAX)
    print(f"[member 6] adversarial CEGAR accept-count: {adv_k}")

    # Build z-id destructive patch (nearest-to-core write from pool)
    w0w = kl_by_horizon(refs_d.q_un, refs_d.q_src, V, M)[MM]
    pool = write_pool(rg, np.zeros((d, 0)), w0w, 1, d, SEED)
    back_u = lambda w: (lambda u: u / np.linalg.norm(u))(rg.back(w))
    angled = sorted((principal_angles_deg(back_u(w)[:, None], Qc)[0],
                     src, w) for src, w in pool)
    near = [(a, s, w) for a, s, w in angled if a <= 15.0]
    print(f"  write pool: {len(pool)} candidates, {len(near)} within 15° "
          "of core")
    assert len(near) >= 1, "no near-core writes in pool"
    a_best, s_best, w_best = near[0]
    print(f"  nearest write: {s_best} at {a_best:.1f}°")

    # z-id patch: write=read in z-coords, pulled back
    P_zid = rg.pull(oblique_patch(
        w_best[:, None] / float(w_best @ w_best), w_best[:, None]))
    q_zid_d = disc.run(model, P_zid)
    q_zid_e = ev.run(model, P_zid)
    obs_zid = refs_d.obs(q_zid_d, MM)
    ex_zid = exact_e.closure(q_zid_e, MM)
    print(f"  z-id: obs={obs_zid:+.1%} exact={ex_zid:+.1%}")

    # Core in adversarial context (same patch, just for table completeness)
    q_core_d = disc.run(model, Pc)
    q_core_e = ev.run(model, Pc)
    obs_core = refs_d.obs(q_core_d, MM)
    ex_core = exact_e.closure(q_core_e, MM)

    # Member 2: ρ (core vs z-id, and core vs full for equivalent reference)
    rho_zid = exact_e.rho(q_core_e, q_zid_e, MM)
    q_full_e = ev.run(model, np.eye(d))
    rho_full = exact_e.rho(q_core_e, q_full_e, MM)
    print(f"\n[member 2] rho:")
    print(f"  core vs full: {rho_full:.4f} (equivalent reference)")
    print(f"  core vs z-id: {rho_zid:.4f}")

    # Member 5: P4 calibration on adversarial cells
    print("\n[member 5] adversarial calibration gaps:")
    adv_cells = [("core", obs_core, ex_core), ("z-id", obs_zid, ex_zid)]
    for name, obs, ex in adv_cells:
        if obs >= 0.20:
            g = calibration_gap(obs, ex)
            print(f"  {name}: obs={obs:+.1%} exact={ex:+.1%} gap={g:.3f}")

    # ===== ARM B: Shift-retention =============================================
    print("\n" + "=" * 60)
    print("ARM B: Shift-retention (member 4)\n")

    # Build shifted pair sets
    test_ps = PairSet(model, proc, cfg, 400, M, SEED + 443, 800,
                      layer=LAYER, ts=TS_TEST)
    shift_depth = PairSet(model, proc, cfg, 400, M, SEED + 999, 800,
                          layer=LAYER, init_state=INIT_DEPTH2)

    refs_test = Refs(test_ps, model, d, M)
    refs_shift = Refs(shift_depth, model, d, M)

    # Build control bases (same as exp 19)
    from abstraction import PCAAbstraction, center_by_position
    from midstream import stream_to
    rng_d = np.random.default_rng(SEED + 555)
    Xd = proc.sample(800, cfg["seq_len"], rng_d)
    Sd = stream_to(model, torch.from_numpy(Xd), LAYER).double().numpy()
    keep = np.arange(cfg["burn_in"], cfg["seq_len"] - 1)
    Rd = center_by_position(Sd[:, keep].reshape(-1, d),
                            np.tile(keep, len(Xd)),
                            np.ones(len(Xd) * len(keep), dtype=bool))
    rng = np.random.default_rng(SEED)
    controls = {
        "pca": PCAAbstraction(Rd).Vt[:k_star].T,
        "rand": orthonormal(rng.standard_normal((d, k_star))),
    }

    # Compute gains at each condition
    patches_for_R = {
        "full": np.eye(d),
        "core": Pc,
        "pca": controls["pca"] @ controls["pca"].T,
        "rand": controls["rand"] @ controls["rand"].T,
    }

    print("[member 4] gains and retention:")
    print(f"  {'patch':>6}  {'base':>8}  {'pos-shift':>10}  {'R(pos)':>8}  "
          f"{'depth-shift':>12}  {'R(depth)':>9}")

    gains_base, gains_pos, gains_depth = {}, {}, {}
    for name, P in patches_for_R.items():
        gains_base[name] = refs_d.obs(disc.run(model, P), MM)
        gains_pos[name] = refs_test.obs(test_ps.run(model, P), MM)
        gains_depth[name] = refs_shift.obs(shift_depth.run(model, P), MM)

    for name in patches_for_R:
        if name == "core":
            R_pos, R_depth = 1.0, 1.0  # reference
        else:
            R_pos = shift_retention(gains_pos[name], gains_base[name],
                                    gains_pos["core"], gains_base["core"])
            R_depth = shift_retention(gains_depth[name], gains_base[name],
                                     gains_depth["core"], gains_base["core"])
        print(f"  {name:>6}  {gains_base[name]:>+8.1%}  "
              f"{gains_pos[name]:>+10.1%}  {R_pos:>8.2f}  "
              f"{gains_depth[name]:>+12.1%}  {R_depth:>9.2f}")

    # Member 5: calibration at shifted conditions
    exact_test = Exact(test_ps, model, M)
    exact_shift = Exact(shift_depth, model, M)
    print("\n[member 5] shift calibration gaps (obs >= 20% cells):")
    shift_worst_gap = 0.0
    for name, P in patches_for_R.items():
        for setname, ps, refs, ex in [
            ("pos-shift", test_ps, refs_test, exact_test),
            ("depth-shift", shift_depth, refs_shift, exact_shift),
        ]:
            q = ps.run(model, P)
            obs_val = refs.obs(q, MM)
            ex_val = ex.closure(q, MM)
            if obs_val >= 0.20:
                g = calibration_gap(obs_val, ex_val)
                shift_worst_gap = max(shift_worst_gap, g)
                tag = "OK" if g <= 0.10 else "WIDE"
                print(f"  {name}/{setname}: obs={obs_val:+.1%} "
                      f"exact={ex_val:+.1%} gap={g:.3f} — {tag}")
    print(f"  worst shift gap: {shift_worst_gap:.3f}")

    # ===== ARM C: Depth stratification ========================================
    print("\n" + "=" * 60)
    print("ARM C: Depth stratification\n")

    depths = bracket_depths(ev)
    unique_d = sorted(np.unique(depths))
    counts = {d_val: int(np.sum(depths == d_val)) for d_val in unique_d}
    print(f"[depth strata] pair counts: "
          + ", ".join(f"depth {d_val}: {counts[d_val]}" for d_val in unique_d))

    print(f"\n[member 1+5] per-depth exact closure at mm=3:")
    print(f"  {'patch':>6}  " + "  ".join(f"{'d=' + str(d_val):>8}"
                                           for d_val in unique_d))
    depth_closures = {}
    for name, P in [("full", np.eye(d)), ("core", Pc),
                    ("pca", controls["pca"] @ controls["pca"].T),
                    ("rand", controls["rand"] @ controls["rand"].T)]:
        q = ev.run(model, P)
        cl = per_depth_closure(ev, model, q, depths, MM)
        depth_closures[name] = cl
        print(f"  {name:>6}  " + "  ".join(f"{cl.get(d_val, float('nan')):>+8.1%}"
                                            for d_val in unique_d))
    core_depths = depth_closures["core"]
    if len(core_depths) >= 2:
        core_vals = [v for v in core_depths.values() if not np.isnan(v)]
        core_spread = max(core_vals) - min(core_vals) if core_vals else 0
        print(f"  core spread: {core_spread:.1%}")

    # ===== ARM D: Gradient read probe =========================================
    print("\n" + "=" * 60)
    print("ARM D: Gradient read probe\n")

    # Torch objective for adversarial read learning
    q_src_d = refs_d.q_src
    torch_obj = make_torch_objective(model, disc, T, Tinv, q_src_d, LAYER)

    w = w_best  # nearest-to-core z-write from arm A
    u = back_u(w)
    print(f"[probe] write: {s_best}, angle to core: {a_best:.1f}°")

    # Gradient-learn the read (200 steps, adversarial)
    c0 = w.copy()
    c_learned = optimize_affine(torch_obj, disc.n, d, lr=0.01, steps=200,
                                batch=min(64, disc.n), seed=SEED, w=w,
                                init_c=c0, adversarial=True,
                                label="read-probe", print_every=50)

    # Evaluate rank-1 oblique patch at train positions
    P_learned = rg.pull(oblique_patch(c_learned[:, None], w[:, None]))
    q_learned_d = disc.run(model, P_learned)
    q_learned_e = ev.run(model, P_learned)
    obs_learned_train = refs_d.obs(q_learned_d, MM)
    ex_learned_train = exact_e.closure(q_learned_e, MM)
    print(f"\n  train: obs(disc)={obs_learned_train:+.1%} "
          f"exact(eval)={ex_learned_train:+.1%}")

    # Evaluate at test positions (position transport)
    obs_learned_test = None
    if obs_learned_train >= 0.20:
        q_learned_test = test_ps.run(model, P_learned)
        obs_learned_test = refs_test.obs(q_learned_test, MM)
        ex_learned_test = exact_test.closure(q_learned_test, MM)
        print(f"  test:  obs={obs_learned_test:+.1%} "
              f"exact={ex_learned_test:+.1%}")
        if obs_learned_train > 1e-6:
            R_learned = obs_learned_test / obs_learned_train
            print(f"  retention (obs test/train): {R_learned:.2f}")
    else:
        print("  train closure < 20% — probe failed; skipping transport")

    # ρ of learned patch vs core
    rho_learned = exact_e.rho(q_core_e, q_learned_e, MM)
    print(f"  rho (core vs learned): {rho_learned:.4f}")

    # ===== VERDICTS ===========================================================
    print("\n" + "=" * 60)
    print("VERDICTS\n")

    # P1: adversarial accept = 0
    p1 = adv_k == 0
    print(f"P1 (adversarial accept=0): accept={adv_k} — "
          f"{'HOLDS' if p1 else 'FAILS'}")

    # P2: ρ separates (core vs z-id ≥ 0.50, core vs full ≤ 0.25)
    p2 = rho_zid >= 0.50 and rho_full <= 0.25
    has_separation = rho_zid > rho_full * 2  # some separation exists
    print(f"P2 (rho separates): core-vs-full={rho_full:.4f} "
          f"core-vs-zid={rho_zid:.4f} — "
          f"{'HOLDS' if p2 else 'FAILS'}")
    if not p2:
        if has_separation:
            print("  (separation exists but thresholds differ — recalibrate)")
        else:
            print("  (no separation — genuine battery-transfer failure)")

    # P3: core position-shift gain retention ≥ 0.70
    core_pos_ret = gains_pos["core"] / gains_base["core"] if gains_base["core"] > 1e-6 else 0
    p3 = core_pos_ret >= 0.70
    print(f"P3 (core position-shift): gain base={gains_base['core']:+.1%} "
          f"shift={gains_pos['core']:+.1%} retention={core_pos_ret:.2f} — "
          f"{'HOLDS' if p3 else 'FAILS'}")

    # P4: core depth-shift gain retention ≥ 0.50
    core_depth_ret = gains_depth["core"] / gains_base["core"] if gains_base["core"] > 1e-6 else 0
    p4 = core_depth_ret >= 0.50
    print(f"P4 (core depth-shift): gain base={gains_base['core']:+.1%} "
          f"shift={gains_depth['core']:+.1%} retention={core_depth_ret:.2f} — "
          f"{'HOLDS' if p4 else 'FAILS'}")

    # P5: depth uniformity ≤ 10pts
    p5 = core_spread <= 0.10
    print(f"P5 (depth uniformity): spread={core_spread:.1%} — "
          f"{'HOLDS' if p5 else 'FAILS (record per-depth thresholds)'}")

    # P6: adversarial calibration ≤ 0.10 (conditional on P1)
    if adv_k == 0:
        # P1 holds (accept=0): no CEGAR-accepted patches to calibrate;
        # check the fixed adversarial cells (core, z-id) instead
        adv_accepted = [(n, o, e) for n, o, e in adv_cells if o >= 0.20]
        p6 = all(calibration_gap(o, e) <= 0.10 for _, o, e in adv_accepted)
        print(f"P6 (adversarial calibration): ", end="")
        if adv_accepted:
            gaps_str = ", ".join(f"{n}={calibration_gap(o,e):.3f}"
                                 for n, o, e in adv_accepted)
            print(f"{gaps_str} — {'HOLDS' if p6 else 'FAILS'}")
        else:
            print("no accepted cells (trivially holds)")
            p6 = True
    else:
        # P1 failed (accept>0): CEGAR accepted patches exist but
        # cegar_accept doesn't return them — cannot calibrate the full
        # accepted set, so P6 is not testable
        p6 = None
        print(f"P6 (adversarial calibration): NOT TESTED — {adv_k} "
              "CEGAR-accepted patches exist but cegar_accept does not "
              "return them; calibration requires the accepted patch set")

    # P7: rank-1 learned > 20% at train
    p7 = obs_learned_train >= 0.20
    print(f"P7 (rank-1 learned > 20%): {obs_learned_train:+.1%} — "
          f"{'HOLDS' if p7 else 'FAILS (probe failed)'}")

    # P8: position entanglement (if P7)
    if p7 and obs_learned_test is not None:
        p8 = obs_learned_test < 0.20
        print(f"P8 (position-entangled): test={obs_learned_test:+.1%} — "
              f"{'HOLDS' if p8 else 'FAILS (read transports!)'}")
    else:
        p8 = None
        print("P8 (position-entangled): skipped (P7 did not hold)")

    # Block gate
    print(f"\n{'=' * 60}")
    gate = p1 and (p6 is True)  # P2-P5, P7-P8 are findings, not gates
    if gate:
        print("Block 2 gates passed.")
    else:
        print("Block 2 gate FAILED — see verdicts above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
