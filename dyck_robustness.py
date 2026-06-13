"""
dyck_robustness.py — Experiment 21: Dyck robustness sweep
(Phase 2, Block 3).

CONTEXT (see experiments/21-dyck-robustness.md). Exp 19 recalibrated
the battery on Dyck-2; exp 20 exercised the interventional matrix at the
standing horizon mm=3 and one adversarial setting. This block sweeps the
remaining local indices: horizon mm <= 4, CEGAR tolerance eps, and a
small kappa grading of adversarial coordinates.

Run after pre-run review:
  uv run python3 dyck_robustness.py --outdir out/dyck2-L4

This is a pre-registration deliverable. Do not run for canonical results
until the writeup and this script have both been reviewed.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from abstraction import CompletionPLS, PCAAbstraction, center_by_position
from adversarial import ZView
from battery import Refs, Exact, calibration_gap, cegar_loop, cegar_staircase
from discover import PairSet, principal_angles_deg, self_checks
from expcommon import (LAYER, adversarial_regime, basis_covariance,
                       build_transform, load_model, oblique_patch,
                       orthonormal, validity_gate)
from midstream import kl_by_horizon, marginal, stream_to
from patches import write_pool
from processes import PROCESSES

SEED = 0
M3 = 3
M4 = 4
MM = (1, 2, 3, 4)
TS_STD = (8, 16, 24)
K_MAX = 12
EPS = 0.05
EPS_DROP = 0.01
EPS_GRID = (0.01, 0.02, 0.05, 0.10)
KAPPAS = (30.0, 100.0, 300.0)
JUNK_SEED = 0

EXPECTED_CFG = {
    "process": "dyck2",
    "seq_len": 32,
    "burn_in": 4,
    "d_model": 64,
    "layers": 4,
    "m": 3,
    "seed": 0,
}

EXP19 = {
    "k_star": 4,
    "c_obs": 0.985,
    "nested_exact_m3": (0.378, 0.716, 0.850, 0.926),
}


def require_expected_config(cfg):
    """Halt if this is not the canonical Dyck checkpoint."""
    mismatches = [(k, cfg.get(k), v) for k, v in EXPECTED_CFG.items()
                  if cfg.get(k) != v]
    if mismatches:
        print("HALT: wrong checkpoint config for canonical exp 21 run.")
        for key, got, want in mismatches:
            print(f"  {key}: got {got!r}, expected {want!r}")
        sys.exit(1)


def build_control_bases(model, proc, cfg, k_star):
    """Reconstruct the exp-19 fixed control bases."""
    L, burn, d = cfg["seq_len"], cfg["burn_in"], cfg["d_model"]
    rng_d = np.random.default_rng(SEED + 555)
    Xd = proc.sample(800, L, rng_d)
    Sd = stream_to(model, torch.from_numpy(Xd), LAYER).double().numpy()
    keep = np.arange(burn, L - 1)
    Gd = np.concatenate([proc.mgram_table(proc.beliefs_along(row)[keep], M3)
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
        "emb": orthonormal(W_tok.T)[:, :min(k_star, proc.V)],
    }


def exp21_self_checks(model, proc, cfg):
    """Checks specific to the horizon sweep."""
    V = proc.V
    p3 = PairSet(model, proc, cfg, 12, M3, 12345, 30, layer=LAYER,
                 ts=TS_STD)
    p4 = PairSet(model, proc, cfg, 12, M4, 12345, 30, layer=LAYER,
                 ts=TS_STD)
    p1 = PairSet(model, proc, cfg, 12, 1, 12345, 30, layer=LAYER,
                 ts=TS_STD)
    assert [t for t, _ in p3.groups] == list(TS_STD), "m=3 ts not pinned"
    assert [t for t, _ in p4.groups] == list(TS_STD), "m=4 ts not pinned"
    assert (p3.a == p4.a).all() and (p3.b == p4.b).all(), "pair identity"
    assert all(torch.equal(p3.pref_src[t], p4.pref_src[t])
               and torch.equal(p3.pref_tgt[t], p4.pref_tgt[t])
               for t, _ in p3.groups), "prefix identity across m"
    q3, q1 = p3.run(model, None), p1.run(model, None)
    q3m1 = marginal(q3, V, 1, M3)
    q3m1 = q3m1 / np.clip(q3m1.sum(axis=1, keepdims=True), 1e-30, None)
    q1n = q1 / np.clip(q1.sum(axis=1, keepdims=True), 1e-30, None)
    assert np.abs(q3m1 - q1n).max() < 1e-6, "chain marginal identity"
    assert np.abs(marginal(p3.p_src3, V, 1, M3) - p1.p_src3).max() < 1e-12, \
        "exact m-gram marginal identity"
    syn = np.arange(1.0, 1.0 + V ** 2)[None, :]
    assert np.allclose(marginal(syn, V, 1, 2),
                       syn.reshape(1, V, V).sum(axis=2)), "marginal helper"
    print("exp21 self-checks passed: pinned ts, pair/prefix identity, "
          "chain marginal, exact marginal, synthetic marginal")


def assert_pairset_identity(label, p3, p4):
    """Assert m=3 and m=4 PairSets differ only in continuation horizon."""
    g3 = [t for t, _ in p3.groups]
    g4 = [t for t, _ in p4.groups]
    assert g3 == list(TS_STD), f"{label}: m=3 ts not pinned"
    assert g4 == list(TS_STD), f"{label}: m=4 ts not pinned"
    assert (p3.a == p4.a).all() and (p3.b == p4.b).all(), \
        f"{label}: pair identity"
    assert all(torch.equal(p3.pref_src[t], p4.pref_src[t])
               and torch.equal(p3.pref_tgt[t], p4.pref_tgt[t])
               for t, _ in p3.groups), f"{label}: prefix identity"


def weakly_decreasing(values_by_eps):
    return all(values_by_eps[EPS_GRID[i + 1]] <= values_by_eps[EPS_GRID[i]]
               for i in range(len(EPS_GRID) - 1))


def build_zid_patch(model, proc, cfg, disc, refs, Qc, d, kappa, sig_x):
    """Build the exp-20 z-id destructive comparator for one kappa."""
    V = proc.V
    T, Tinv, _ = build_transform(Qc, d, kappa, junk_seed=JUNK_SEED)
    rg, _ = adversarial_regime(disc, T, Tinv, sig_x)
    w0w = kl_by_horizon(refs.q_un, refs.q_src, V, M3)[M3]
    pool = write_pool(rg, np.zeros((d, 0)), w0w, 1, d, SEED)
    back_u = lambda w: (lambda u: u / np.linalg.norm(u))(rg.back(w))
    angled = sorted((principal_angles_deg(back_u(w)[:, None], Qc)[0],
                     src, w) for src, w in pool)
    near = [(a, s, w) for a, s, w in angled if a <= 15.0]
    assert near, f"no near-core writes for kappa={kappa}"
    angle, source, w = near[0]
    P = rg.pull(oblique_patch(w[:, None] / float(w @ w), w[:, None]))
    return {
        "T": T,
        "Tinv": Tinv,
        "rg": rg,
        "angle": angle,
        "source": source,
        "patch": P,
    }


def adv_accept_count(model, disc, refs, T, pull, d, eps, mm):
    """Adversarial CEGAR accept-count for a specific horizon and eps."""
    from battery import cegar_accept

    view = ZView(disc, T)
    return cegar_accept(model, view, pull, disc, refs, mm, d, eps, K_MAX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
    ap.add_argument("--selftest", action="store_true",
                    help="run guards and horizon self-checks, then exit")
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    require_expected_config(cfg)
    proc = PROCESSES[cfg["process"]]()
    assert proc.name == "dyck2", f"expected dyck2, got {proc.name}"
    V, d = proc.V, cfg["d_model"]

    model = load_model(args.outdir, cfg, proc)
    gap, valid = validity_gate(model, proc, cfg, SEED)
    if not valid:
        print("HALT: validity gate failed.")
        sys.exit(1)

    disc3 = PairSet(model, proc, cfg, 400, M3, SEED + 111, 800,
                    layer=LAYER, ts=TS_STD)
    ev3 = PairSet(model, proc, cfg, 600, M3, SEED + 777, 800,
                  layer=LAYER, ts=TS_STD)
    disc4 = PairSet(model, proc, cfg, 400, M4, SEED + 111, 800,
                    layer=LAYER, ts=TS_STD)
    ev4 = PairSet(model, proc, cfg, 600, M4, SEED + 777, 800,
                  layer=LAYER, ts=TS_STD)
    assert_pairset_identity("disc", disc3, disc4)
    assert_pairset_identity("eval", ev3, ev4)
    self_checks(model, ev3, LAYER, M3, V)
    self_checks(model, ev4, LAYER, M4, V)
    exp21_self_checks(model, proc, cfg)
    if args.selftest:
        return

    print(f"=== Experiment 21: Dyck robustness sweep | {proc.name} | "
          f"L{cfg['layers']} d{d} | mm={list(MM)} | "
          f"eps={list(EPS_GRID)} | kappa={list(KAPPAS)} ===\n")

    refs3, refs4 = Refs(disc3, model, d, M3), Refs(disc4, model, d, M4)
    exact3, exact4 = Exact(ev3, model, M3), Exact(ev4, model, M4)

    print("[anchor] reproduce exp-19 core:")
    k_star, Qc, c_obs_disc = cegar_loop(model, disc3, refs3, d, EPS, K_MAX,
                                         M3, eps_drop=EPS_DROP)
    print(f"  k*={k_star}, c_obs={c_obs_disc:.1%}")
    p1_core = (k_star == EXP19["k_star"]
               and abs(c_obs_disc - EXP19["c_obs"]) <= 0.02)
    if not p1_core:
        print("HALT: exp-19 core reproduction failed.")
        sys.exit(1)
    Pc = Qc @ Qc.T

    nested = []
    for k in range(1, k_star + 1):
        q = ev3.run(model, Qc[:, :k] @ Qc[:, :k].T)
        nested.append(exact3.closure(q, M3))
    p1_nested = all(abs(nested[i] - EXP19["nested_exact_m3"][i]) <= 0.02
                    for i in range(len(EXP19["nested_exact_m3"])))
    print("  nested exact mm=3: " + ", ".join(f"k={i + 1}:{v:.1%}"
                                             for i, v in enumerate(nested)))
    if not p1_nested:
        print("HALT: exp-19 nested staircase reproduction failed.")
        sys.exit(1)
    print("  reproduction OK\n")

    controls = build_control_bases(model, proc, cfg, k_star)
    patches = {
        "full": np.eye(d),
        "core": Pc,
        "pca": controls["pca"] @ controls["pca"].T,
        "pls": controls["pls"] @ controls["pls"].T,
        "rand": controls["rand"] @ controls["rand"].T,
        "emb": controls["emb"] @ controls["emb"].T,
    }

    print("[fixed patches] horizon matrix:")
    fixed = {}
    q_core_e3 = ev3.run(model, Pc)
    q_core_e4 = ev4.run(model, Pc)
    for name, P in patches.items():
        qd3, qe3 = disc3.run(model, P), ev3.run(model, P)
        qd4, qe4 = disc4.run(model, P), ev4.run(model, P)
        cells = []
        for mm in MM:
            if mm <= M3:
                obs = refs3.obs(qd3, mm)
                ex = exact3.closure(qe3, mm)
                rho = exact3.rho(q_core_e3, qe3, mm)
            else:
                obs = refs4.obs(qd4, M4)
                ex = exact4.closure(qe4, M4)
                rho = exact4.rho(q_core_e4, qe4, M4)
            gap = calibration_gap(obs, ex)
            fixed[(name, mm)] = {
                "obs": obs,
                "exact": ex,
                "rho": rho,
                "gap": gap,
            }
            cells.append(f"{obs:+.0%}/{ex:+.0%}/g{gap:.2f}/r{rho:.2f}")
        print(f"  {name:>5}: " + " | ".join(cells))
    print()

    print("[fixed patches] accepted-cell calibration gaps by horizon:")
    for mm in MM:
        cells = []
        for name in patches:
            cell = fixed[(name, mm)]
            if cell["obs"] >= 0.20:
                cells.append(f"{name}={cell['gap']:.3f}")
        print(f"  mm={mm}: " + (", ".join(cells) if cells else "none"))
    print()

    print("[benign CEGAR] eps staircases by horizon:")
    benign_stair = {}
    for mm in MM:
        if mm <= M3:
            st = cegar_staircase(model, disc3, refs3, d, EPS_GRID, K_MAX, mm)
        else:
            st = cegar_staircase(model, disc4, refs4, d, EPS_GRID, K_MAX, M4)
        benign_stair[mm] = st
        print(f"  mm={mm}: " + ", ".join(f"eps={e:.2f}:k*={st[e]}"
                                         for e in EPS_GRID))
    print()

    print("[adversarial] kappa grading:")
    adv = {}
    sig_x = basis_covariance(model, proc, cfg, SEED, 800)
    for kappa in KAPPAS:
        z = build_zid_patch(model, proc, cfg, disc3, refs3, Qc, d, kappa,
                            sig_x)
        # A separate view for m=4 is required because PairSet owns prefixes.
        rg4, _ = adversarial_regime(disc4, z["T"], z["Tinv"],
                                    sig_x)
        qz_d3, qz_e3 = disc3.run(model, z["patch"]), ev3.run(model, z["patch"])
        qz_d4, qz_e4 = disc4.run(model, z["patch"]), ev4.run(model, z["patch"])
        zcells = {}
        for mm in MM:
            if mm <= M3:
                obs = refs3.obs(qz_d3, mm)
                ex = exact3.closure(qz_e3, mm)
                rho = exact3.rho(q_core_e3, qz_e3, mm)
            else:
                obs = refs4.obs(qz_d4, M4)
                ex = exact4.closure(qz_e4, M4)
                rho = exact4.rho(q_core_e4, qz_e4, M4)
            gap = calibration_gap(obs, ex)
            zcells[mm] = {"obs": obs, "exact": ex, "rho": rho, "gap": gap}
        acc = {}
        for mm in MM:
            acc[mm] = {}
            for eps in EPS_GRID:
                if mm <= M3:
                    acc[mm][eps] = adv_accept_count(
                        model, disc3, refs3, z["T"], z["rg"].pull, d, eps, mm)
                else:
                    acc[mm][eps] = adv_accept_count(
                        model, disc4, refs4, z["T"], rg4.pull, d, eps, M4)
        adv[kappa] = {"zid": zcells, "acc": acc, "source": z["source"],
                      "angle": z["angle"]}
        print(f"  kappa={kappa:.0f}: nearest {z['source']} at "
              f"{z['angle']:.1f} deg")
        print("    z-id: " + " | ".join(
            f"mm={mm}:{zcells[mm]['obs']:+.0%}/{zcells[mm]['exact']:+.0%}"
            f"/g{zcells[mm]['gap']:.2f}/r{zcells[mm]['rho']:.2f}"
            for mm in MM))
        accepted = [f"mm={mm}:{zcells[mm]['gap']:.3f}" for mm in MM
                    if zcells[mm]["obs"] >= 0.20]
        print("    accepted gaps: "
              + (", ".join(accepted) if accepted else "none"))
        for mm in MM:
            print("    accept mm=" + str(mm) + ": " + ", ".join(
                f"eps={eps:.2f}:{acc[mm][eps]}" for eps in EPS_GRID))
    print()

    # ----- verdicts -----------------------------------------------------------
    print("=" * 60)
    print("VERDICTS\n")

    p1 = p1_core and p1_nested
    print(f"P1 (exp-19 reproduction): {'HOLDS' if p1 else 'FAILS'}")

    accepted_cells = [(key, cell) for key, cell in fixed.items()
                      if cell["obs"] >= 0.20]
    accepted_cells += [((f"kappa={kappa:.0f}/z-id", mm), cell)
                       for kappa, vals in adv.items()
                       for mm, cell in vals["zid"].items()
                       if cell["obs"] >= 0.20]
    worst_key, worst_cell = max(
        accepted_cells,
        key=lambda item: calibration_gap(item[1]["obs"], item[1]["exact"]))
    worst_gap = calibration_gap(worst_cell["obs"], worst_cell["exact"])
    p2_strict = worst_gap <= 0.10
    p2_mm4_only = (not p2_strict
                   and all(calibration_gap(cell["obs"], cell["exact"]) <= 0.10
                           for key, cell in accepted_cells
                           if key[1] in (1, 2, 3)))
    p2 = p2_strict or p2_mm4_only
    print(f"P2 (obs/exact calibration across horizons): "
          f"{len(accepted_cells)} cells, worst {worst_gap:.3f} at "
          f"{worst_key} — ", end="")
    if p2_strict:
        print("HOLDS")
    elif p2_mm4_only:
        print("HORIZON-LOCAL WIDENING (mm=4 only)")
    else:
        print("FAILS (regression at mm<=3)")

    equiv_names = ("full", "pca", "emb")
    distinct_names = ("pls", "rand")
    p3_strict = True
    p3_has_separation = True
    for mm in MM:
        equiv_max = max(fixed[(name, mm)]["rho"] for name in equiv_names)
        distinct_min = min(fixed[(name, mm)]["rho"] for name in distinct_names)
        ok = equiv_max <= 0.25 and distinct_min >= 0.50
        separated = distinct_min > equiv_max * 2
        p3_strict = p3_strict and ok
        p3_has_separation = p3_has_separation and separated
        print(f"P3 mm={mm}: equiv max={equiv_max:.3f}, "
              f"distinct min={distinct_min:.3f} — "
              f"{'ok' if ok else ('separated' if separated else 'FLAT')}")
    p3 = p3_strict or p3_has_separation
    if p3_strict:
        print("P3 (rho bands across horizons): HOLDS")
    elif p3_has_separation:
        print("P3 (rho bands across horizons): RECALIBRATE "
              "(separation present, bands shifted)")
    else:
        print("P3 (rho bands across horizons): FAILS "
              "(no behavioral separation)")

    p4_shape = all(weakly_decreasing(benign_stair[mm]) for mm in MM)
    p4_fine = all(benign_stair[mm][0.01] <= 8 for mm in MM)
    p4_core = all(3 <= benign_stair[mm][0.05] <= 5 for mm in (2, 3, 4))
    p4 = p4_shape and p4_fine and p4_core
    print(f"P4 (benign eps staircases): weakly decreasing={p4_shape}, "
          f"k*(0.01)<=8={p4_fine}, k*(0.05) in [3,5] for mm=2..4="
          f"{p4_core} — {'HOLDS' if p4 else 'FAILS'}")

    p5 = all(vals["acc"][mm][0.05] == 0 for vals in adv.values()
             for mm in MM)
    print(f"P5 (adversarial accept=0 at eps=0.05 for all kappa/mm): "
          f"{'HOLDS' if p5 else 'FAILS'}")

    p6_strict = True
    p6_has_separation = True
    for kappa, vals in adv.items():
        rho_min = min(vals["zid"][mm]["rho"] for mm in MM)
        ok = rho_min >= 0.50
        separated = rho_min > 0.25
        p6_strict = p6_strict and ok
        p6_has_separation = p6_has_separation and separated
        print(f"P6 kappa={kappa:.0f}: z-id rho min={rho_min:.3f} — "
              f"{'ok' if ok else ('separated' if separated else 'FLAT')}")
    p6 = p6_strict or p6_has_separation
    if p6_strict:
        print("P6 (z-id remains behaviorally distinct): HOLDS")
    elif p6_has_separation:
        print("P6 (z-id remains behaviorally distinct): RECALIBRATE "
              "(separation present, bands shifted)")
    else:
        print("P6 (z-id remains behaviorally distinct): FAILS "
              "(no behavioral separation)")

    p7_shape = all(weakly_decreasing(vals["acc"][mm])
                   for vals in adv.values() for mm in MM)
    p7_no_hi = all(vals["acc"][mm][eps] == 0
                   for vals in adv.values() for mm in MM
                   for eps in EPS_GRID if eps >= 0.05)
    p7 = p7_shape and p7_no_hi
    print(f"P7 (adversarial tolerance staircases): "
          f"weakly decreasing={p7_shape}, no accepts for eps>=0.05="
          f"{p7_no_hi} — {'HOLDS' if p7 else 'FAILS'}")

    print(f"P8 (validity/config/self-checks): {'HOLDS' if valid else 'FAILS'}")

    print(f"\n{'=' * 60}")
    all_hold = p1 and p2 and p3 and p4 and p5 and p6 and p7 and valid
    if all_hold:
        if p2_strict and p3_strict and p6_strict:
            print("Block 3 registered gates passed.")
        else:
            print("Block 3 registered gates passed with typed "
                  "recalibration/widening branches.")
    else:
        print("Block 3 gate FAILED — see verdicts above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
