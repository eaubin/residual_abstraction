"""
reads.py — Experiment 12: fractional-precision read search.

CONTEXT (see experiments/12-read-search.md — the pre-registration carries
the full impossibility derivation; the short form: the clean z-read for a
write w is c_clean ∝ T⁻²w, the grid offers Σ̂_z^{−α}w, and matching the
κ-dependence needs α = 1 while avoiding the x-spectrum distortion needs
α = 0 — no α is exactly clean, so success means "a good-enough tradeoff
point exists", never "the clean read was constructed").

Experiment 11 showed the read menu {id, prec, cov} does half the repair
(prec eliminates destruction, under-transfers everywhere). Here the read
becomes a one-parameter SEARCH: per write w, reads c_α ∝ Σ̂^{−α}w for
α ∈ {0, 0.25, 0.5, 0.75, 1} (α = 0 is id, α = 1 is prec — the grid nests
both known failures), ranked with the writes by measured closure gain;
acceptance frozen, eps_gain explicitly restated at 0.05 against the
+4.5%-proximity temptation.

New diagnostic anchors: D1 re-verifies exp-11's stream-clean rank-1
(+51.3% there); D2 composes the TWO nearest-to-plane round-1 adversarial
writes with clean reads — closing the "composition is the suspect" escape
hatch inside this run. The same-write equivariance table fixes exp-11's
confound: the same pulled-back write evaluated across the α-grid in both
regimes.

Run: python3 reads.py --outdir out/mess3-L4   (~2 h: up to 60 chain
evaluations per round)
`--selftest` runs the standard four machinery checks and exits.

RESULTS (see experiments/12-read-search.md): P1/P2/P6/P7 HOLD, P3 FAILS,
P4/P5 NOT TESTED. Branch (ii): no Sigma-spectral read suffices. The
same-write equivariance table is the central measurement: adversarial
gains flat ~+1.5% across the whole grid (no interior optimum); benign
column shows pure spectrum-distortion cost (+51% through alpha=0.5,
collapsing to +1.4% at alpha=1); prec confirmed behaviorally equivariant
(+1.4% = +1.4% on the same write — exp-11's apparent ridge gap was
write-difference). Mechanism refined: junk suppression succeeds by
alpha >= 0.75 but the causal read never appears — the obstacle is
NEUTRAL-background read contamination (T crushes plane reads x kappa^-1
while leaving ~60 neutral directions untouched; spectral re-amplification
lifts neutral low-variance reads at the same time). D2 exonerates
composition at 97.8% (clean-read patches at 1.1 and 3.3 deg stack to the
ceiling). The program's remaining adversarial gap is one object: an
honest non-spectral clean-read construction (exp 13).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from abstraction import center_by_position, kl_rows
from adversarial import ZView
from discover import (PairSet, mined_direction, principal_angles_deg,
                      self_checks)
from midstream import kl_by_horizon, orthonormal, stream_to
from miners import sqrt_and_inv
from model import GPT, GPTConfig
from patches import oblique_patch, write_pool
from processes import PROCESSES

LAYER = 1
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)      # registered grid (module constant,
                                          # deliberately not a CLI argument)
REGISTERED = {"kappa": 100.0, "k_max": 8, "eps_gain": 0.05,
              "eps_drop": 0.01, "pairs_disc": 400, "pairs_eval": 600,
              "basis_seqs": 800, "m": 3}
COND_MAX = 1e6


def mat_power(S, p, floor=1e-10):
    """S^p by eigendecomposition with the registered eigenvalue floor."""
    w, Vv = np.linalg.eigh(S)
    w = np.maximum(w, floor * w.max())
    return (Vv * w ** p) @ Vv.T


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/mess3-L4")
    ap.add_argument("--kappa", type=float, default=REGISTERED["kappa"])
    ap.add_argument("--k-max", type=int, default=REGISTERED["k_max"])
    ap.add_argument("--eps-gain", type=float, default=REGISTERED["eps_gain"])
    ap.add_argument("--eps-drop", type=float, default=REGISTERED["eps_drop"])
    ap.add_argument("--pairs-disc", type=int,
                    default=REGISTERED["pairs_disc"])
    ap.add_argument("--pairs-eval", type=int,
                    default=REGISTERED["pairs_eval"])
    ap.add_argument("--basis-seqs", type=int,
                    default=REGISTERED["basis_seqs"])
    ap.add_argument("--m", type=int, default=REGISTERED["m"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force-invalid", action="store_true")
    args = ap.parse_args(argv)

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    registered_cfg = (proc.name == "mess3" and cfg["layers"] == 4
                      and cfg["seq_len"] == 32 and cfg["d_model"] == 64
                      and cfg["burn_in"] == 4)
    if not registered_cfg and not args.selftest and not args.force_invalid:
        print("Experiment 12 is registered for the Experiment-6/8-11 "
              "setting (mess3 / 4 layers / d_model 64 / seq_len 32 / "
              f"burn_in 4); this config is {cfg['process']} L{cfg['layers']}"
              f" d{cfg['d_model']} T{cfg['seq_len']} b{cfg['burn_in']}. "
              "Use --selftest or --force-invalid.")
        return
    overridden = [k for k, v in REGISTERED.items() if getattr(args, k) != v]
    if overridden and not args.selftest and not args.force_invalid:
        print(f"Experiment 12 parameters are registered; overridden: "
              f"{overridden}. Use --force-invalid for an exploratory run.")
        return
    if overridden:
        print(f"NOTE: EXPLORATORY RUN — non-registered parameters "
              f"{overridden}; verdicts below are NOT Experiment 12.\n")
    if args.seed != 0 and not args.selftest and not args.force_invalid:
        print(f"Experiment 12 registers seed 0; got {args.seed}. "
              "Use --force-invalid.")
        return
    if args.seed != 0:
        print(f"NOTE: EXPLORATORY RUN — seed {args.seed} != 0.\n")

    L, burn, V, m = cfg["seq_len"], cfg["burn_in"], proc.V, args.m
    d = cfg["d_model"]
    model = GPT(GPTConfig(vocab=V, seq_len=L, d_model=d,
                          n_layers=cfg["layers"]))
    model.load_state_dict(torch.load(os.path.join(args.outdir, "model.pt"),
                                     map_location="cpu"))
    model.eval()

    # ----- validity gate (P7, enforced) ---------------------------------------
    Xg = proc.sample(2000, L, np.random.default_rng(args.seed + 999))
    with torch.no_grad():
        tot, cnt = 0.0, 0
        for i in range(0, len(Xg), 256):
            logits = model(torch.from_numpy(Xg[i:i + 256]))
            tgt = torch.from_numpy(Xg[i:i + 256, 1:]).reshape(-1)
            tot += F.cross_entropy(logits[:, :-1].reshape(-1, V), tgt,
                                   reduction="sum").item()
            cnt += tgt.numel()
    gap_opt = tot / cnt - cfg["optimal_nll"]
    p7 = gap_opt <= 0.005
    print(f"validity gate: gap-to-optimal {gap_opt:+.4f} nats — "
          f"{'PASS' if p7 else 'FAIL'}\n")
    if not p7 and not args.selftest and not args.force_invalid:
        print("exiting: validity gate failed.")
        return

    disc = PairSet(model, proc, cfg, args.pairs_disc, m, args.seed + 111, 800,
                   layer=LAYER)
    ev = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777, 800,
                 layer=LAYER)
    self_checks(model, ev, LAYER, m, V)
    if args.selftest:
        return

    print(f"=== Experiment 12: fractional-precision read search | "
          f"{proc.name} | patch L{LAYER} | kappa = {args.kappa:g} | "
          f"alpha grid {ALPHAS} ===\n")

    # ----- observable refs -----------------------------------------------------
    q_src_d = disc.run(model, None, src_side=True)
    q_un_d = disc.run(model, None)
    q_full_d = disc.run(model, np.eye(d))
    D0 = float(kl_rows(q_src_d, q_un_d).mean())
    Dfull = float(kl_rows(q_src_d, q_full_d).mean())
    assert D0 > Dfull
    c_obs = lambda q: (D0 - float(kl_rows(q_src_d, q).mean())) / (D0 - Dfull)
    assert abs(c_obs(q_full_d) - 1.0) < 1e-12

    # ----- anchor: the frozen exp-6 loop ---------------------------------------
    print("[anchor] frozen Experiment-6 loop:")
    Q = np.zeros((d, 0))
    q_cur, c_cur = q_un_d, 0.0
    while Q.shape[1] < args.k_max:
        v = mined_direction(disc, Q, kl_rows(q_src_d, q_cur))
        Q_try = np.hstack([Q, v[:, None]])
        q_try = disc.run(model, Q_try @ Q_try.T)
        if c_obs(q_try) - c_cur < args.eps_gain:
            break
        Q, q_cur, c_cur = Q_try, q_try, c_obs(q_try)
        print(f"  k={Q.shape[1]}: c_obs {c_cur:.1%}")
    Qc = Q
    assert Qc.shape[1] == 2 and abs(c_cur - 0.998) < 0.005
    Pc = Qc @ Qc.T
    print(f"[anchor] reproduced: k* = 2, c_obs = {c_cur:.1%}\n")

    kap = args.kappa
    rng0 = np.random.default_rng(0)
    Gj = rng0.standard_normal((d, 2))
    Gj -= Qc @ (Qc.T @ Gj)
    Qj = orthonormal(Gj)
    T = np.eye(d) - (1 - 1 / kap) * Pc + (kap - 1) * (Qj @ Qj.T)
    Tinv = np.eye(d) + (kap - 1) * Pc + (1 / kap - 1) * (Qj @ Qj.T)
    assert np.allclose(T @ Tinv, np.eye(d), atol=1e-9)
    Qzc = orthonormal(T @ Qc)
    assert np.allclose(T @ (Qzc @ Qzc.T) @ Tinv, Pc, atol=1e-9)
    print("transform checks passed (T·Tinv = I; pull(T·plane) = plane)\n")

    rng_b = np.random.default_rng(args.seed + 555)
    Xb = proc.sample(args.basis_seqs, L, rng_b)
    Sb = stream_to(model, torch.from_numpy(Xb), LAYER).double().numpy()
    keep = np.arange(burn, L - 1)
    Rb = center_by_position(Sb[:, keep].reshape(-1, d), np.tile(keep, len(Xb)),
                            np.ones(len(Xb) * len(keep), dtype=bool))
    Sig_x = np.cov(Rb.T)
    Sig_z = T @ Sig_x @ T

    class Regime:
        def __init__(self, name, view_raw, view_wht, S, Sinv, pows, pull,
                     back, fwd):
            self.name, self.view_raw, self.view_wht = name, view_raw, view_wht
            self.S, self.Sinv, self.pows = S, Sinv, pows
            self.pull, self.back, self.fwd = pull, back, fwd

    Sx, Sx_inv = sqrt_and_inv(Sig_x)
    Sz, Sz_inv = sqrt_and_inv(Sig_z)
    pows_x = {a: mat_power(Sig_x, -a) for a in ALPHAS if a > 0}
    pows_z = {a: mat_power(Sig_z, -a) for a in ALPHAS if a > 0}
    # registered self-checks of the power construction
    assert np.allclose(mat_power(Sig_x, 0.0), np.eye(d), atol=1e-9), \
        "alpha = 0 must be the identity (id read)"
    ridge_inv = np.linalg.solve(Sig_z + 1e-10 * np.trace(Sig_z) / d
                                * np.eye(d), np.eye(d))
    w_probe = Qzc[:, 0]
    P_grid = np.outer(pows_z[1.0] @ w_probe / (pows_z[1.0] @ w_probe
                                               @ w_probe), w_probe)
    P_prec = np.outer(ridge_inv @ w_probe / (ridge_inv @ w_probe @ w_probe),
                      w_probe)
    rel = np.linalg.norm(P_grid - P_prec) / np.linalg.norm(P_prec)
    assert rel < 1e-3, ("alpha = 1 patch deviates from exp-11's prec patch "
                        f"beyond regularizer tolerance: rel {rel:.2e}")
    print("read-construction checks passed (alpha=0 = id; alpha=1 = exp-11 "
          f"prec to rel {rel:.1e})\n")

    regimes = {
        "ben": Regime("ben", disc, ZView(disc, Sx_inv), Sx, Sx_inv, pows_x,
                      lambda P: P, lambda v: v, lambda c: c),
        "adv": Regime("adv", ZView(disc, T), ZView(disc, T @ Sz_inv),
                      Sz, Sz_inv, pows_z, lambda P: T @ P @ Tinv,
                      lambda v: Tinv @ v, lambda c: T @ c),
    }

    def reads_for(rg, w):
        out = [("a0.00", w.copy())]
        for a in ALPHAS[1:]:
            c = rg.pows[a] @ w
            ip = float(c @ w)
            if abs(ip) > 1e-12:
                out.append((f"a{a:.2f}", c / ip))
        return out

    def search(rg):
        W = np.zeros((d, 0))
        C = np.zeros((d, 0))
        q_cur, c_cur = q_un_d, 0.0
        sources, converged, round1 = [], False, []
        rnd = 0
        while W.shape[1] < args.k_max:
            rnd += 1
            wts = kl_rows(q_src_d, q_cur)
            print(f"  [{rg.name}] round {rnd} (current c_obs {c_cur:.1%}):")
            best = None
            for wsrc, w in write_pool(rg, W, wts, rnd, d, args.seed):
                sw = rg.back(w)
                w_ang = principal_angles_deg(
                    (sw / np.linalg.norm(sw))[:, None], Qc)[0]
                if rnd == 1:
                    round1.append((w_ang, wsrc, w))
                for rfam, c in reads_for(rg, w):
                    assert abs(float(c @ w) - 1.0) < 1e-9
                    W_try = np.hstack([W, w[:, None]])
                    C_try = np.hstack([C, c[:, None]])
                    if np.linalg.cond(W_try.T @ C_try) > COND_MAX:
                        print(f"      {wsrc:>9}/{rfam}: SKIPPED "
                              "(conditioning)")
                        continue
                    P = rg.pull(oblique_patch(C_try, W_try))
                    gain = c_obs(disc.run(model, P)) - c_cur
                    sc = rg.fwd(c)
                    rj = float(np.linalg.norm(Qj.T @ sc)
                               / np.linalg.norm(sc))
                    print(f"      {wsrc:>9}/{rfam}: gain {gain:+7.1%}, "
                          f"w-angle {w_ang:5.1f} deg, read-junk {rj:4.0%}")
                    if best is None or gain > best[0]:
                        best = (gain, wsrc, rfam, w, c)
            gain, wsrc, rfam, w, c = best
            if gain < args.eps_gain:
                print(f"  [{rg.name}] best gain {gain:+.1%} < eps_gain -> "
                      "STOP")
                converged = True
                break
            W = np.hstack([W, w[:, None]])
            C = np.hstack([C, c[:, None]])
            q_cur = disc.run(model, rg.pull(oblique_patch(C, W)))
            c_cur += gain
            sources.append(f"{wsrc}/{rfam}")
            print(f"  [{rg.name}] accept {wsrc}/{rfam} -> k = {W.shape[1]}, "
                  f"c_obs {c_cur:.1%}")
        if not converged:
            print(f"  [{rg.name}] k_max reached: NON-CONVERGENT")
        changed = True
        while changed and W.shape[1] > 1:
            changed = False
            for j in range(W.shape[1]):
                Wj, Cj = np.delete(W, j, axis=1), np.delete(C, j, axis=1)
                cj = c_obs(disc.run(model, rg.pull(oblique_patch(Cj, Wj))))
                if c_cur - cj < args.eps_drop:
                    print(f"  [{rg.name}] coarsen: dropped pair {j + 1} "
                          f"({sources[j]})")
                    W, C, c_cur, changed = Wj, Cj, cj, True
                    sources.pop(j)
                    break
        print(f"  [{rg.name}] fixed point: k* = {W.shape[1]}, c_obs = "
              f"{c_cur:.1%}, sources = {sources}\n")
        return W, C, c_cur, sources, converged, sorted(round1)

    results = {}
    for rname in ("ben", "adv"):
        print(f"[search/{rname}]:")
        results[rname] = search(regimes[rname])

    # ----- diagnostic anchors D1/D2 + same-write equivariance table ----------
    round1_adv = results["adv"][5]
    near = [(a, s, w) for a, s, w in round1_adv if a <= 15.0]
    if near:
        a1, s1, w1 = near[0]
        u1 = regimes["adv"].back(w1)
        u1 = u1 / np.linalg.norm(u1)
        g_d1 = c_obs(disc.run(model, np.outer(u1, u1)))
        print(f"D1 (stream-clean rank-1, nearest write {s1} at {a1:.1f} "
              f"deg): gain {g_d1:+.1%}")
    else:
        g_d1 = None
        print("D1: NOT TESTED — no round-1 adversarial write within 15 deg")
    if len(near) >= 2:
        a2, s2, w2 = near[1]
        u2 = regimes["adv"].back(w2)
        U = orthonormal(np.column_stack([u1, u2 / np.linalg.norm(u2)]))
        P_d2 = U @ U.T
        g_d2 = c_obs(disc.run(model, P_d2))
        print(f"D2 (stream-clean composition of {s1} + {s2} at "
              f"{a1:.1f}/{a2:.1f} deg): c_obs {g_d2:+.1%}")
    else:
        P_d2, g_d2 = None, None
        print("D2: NOT TESTED — fewer than two round-1 adversarial writes "
              "within 15 deg")

    if near:
        wx = regimes["adv"].back(w1)
        wx = wx / np.linalg.norm(wx)
        print("\nsame-write equivariance table (write held fixed across "
              "regimes; k=1 gains):")
        print("  alpha    benign     adversarial")
        for a in ALPHAS:
            row = []
            for rg, wv in ((regimes["ben"], wx), (regimes["adv"], w1)):
                c = wv.copy() if a == 0 else rg.pows[a] @ wv
                ip = float(c @ wv)
                if abs(ip) < 1e-12:
                    row.append(float("nan"))
                    continue
                P = rg.pull(oblique_patch((c / ip)[:, None], wv[:, None]))
                row.append(c_obs(disc.run(model, P)))
            print(f"  {a:5.2f}  {row[0]:+8.1%}   {row[1]:+8.1%}")
    print()

    # ----- evaluation -----------------------------------------------------------
    q0_e = ev.run(model, None)
    rows_fe = kl_by_horizon(q0_e, ev.p_tgt3, V, m)
    rows_ge = kl_by_horizon(q0_e, ev.p_src3, V, m)
    floor_e = {mm: float(rows_fe[mm].mean()) for mm in rows_fe}
    gap_e = {mm: float(rows_ge[mm].mean()) for mm in rows_ge}
    ms = list(range(1, m + 1))

    def true_closure(P):
        rows_t = kl_by_horizon(ev.run(model, P), ev.p_src3, V, m)
        cl = {mm: (gap_e[mm] - float(rows_t[mm].mean()))
              / (gap_e[mm] - floor_e[mm]) for mm in ms}
        return cl

    print("evaluation (exact targets; pooled m=1..3):")
    cl_full = true_closure(np.eye(d))
    print(f"  {'full':>12} (k=64)  "
          + "  ".join(f"m={mm}: {cl_full[mm]:>6.1%}" for mm in ms))
    cl_plane = true_closure(Pc)
    print(f"  {'plane-anchor':>12} (k=2)  "
          + "  ".join(f"m={mm}: {cl_plane[mm]:>6.1%}" for mm in ms))
    cl_d2 = true_closure(P_d2) if P_d2 is not None else None
    if cl_d2:
        print(f"  {'D2-anchor':>12} (k=2)  "
              + "  ".join(f"m={mm}: {cl_d2[mm]:>6.1%}" for mm in ms))
    closures, plane_ang = {}, {}
    for rname, (W, C, c, srcs, conv, _) in results.items():
        if W.shape[1] == 0:
            closures[rname] = {mm: 0.0 for mm in ms}
            print(f"  {rname:>12} (k=0)   closure 0 by definition")
            continue
        P = regimes[rname].pull(oblique_patch(C, W))
        closures[rname] = true_closure(P)
        Wb = np.column_stack([regimes[rname].back(W[:, j])
                              for j in range(W.shape[1])])
        plane_ang[rname] = principal_angles_deg(Qc, orthonormal(Wb))
        print(f"  {rname:>12} (k={W.shape[1]})  "
              + "  ".join(f"m={mm}: {closures[rname][mm]:>6.1%}"
                          for mm in ms)
              + "   | plane-in-writes "
              + "/".join(f"{a:.1f}" for a in plane_ang[rname]) + " deg")

    # ----- verdicts -------------------------------------------------------------
    print("\nverdicts:")
    p1 = cl_plane[m] >= 0.90 * cl_full[m] and g_d1 is not None \
        and g_d1 >= 0.40
    print(f"  P1 anchors (ceiling >= 90%; D1 >= 40%): plane "
          f"{cl_plane[m]:.1%}, D1 "
          + (f"{g_d1:+.1%}" if g_d1 is not None else "NOT TESTED")
          + f" — {'HOLDS' if p1 else 'FAILS'}")
    Wb_, _, cb, sb, convb, _ = results["ben"]
    p2 = convb and 0 < Wb_.shape[1] <= 4 and \
        closures["ben"][m] >= 0.90 * cl_full[m]
    print(f"  P2 benign (k* <= 4, >= 90% of full): k*={Wb_.shape[1]}, "
          f"{closures['ben'][m]:.1%} — {'HOLDS' if p2 else 'FAILS'}")
    Wa, _, ca, sa, conva, _ = results["adv"]
    p3 = conva and 0 < Wa.shape[1] <= 4 and \
        closures["adv"][m] >= 0.90 * cl_full[m]
    print(f"  P3 adversarial (k* <= 4, >= 90% of full, after pullback): "
          f"k*={Wa.shape[1]}, {closures['adv'][m]:.1%} — "
          f"{'HOLDS' if p3 else 'FAILS'}")
    if Wa.shape[1] == 0:
        p4 = False
        print("  P4 observable/exact on adversarial: NOT TESTED — no "
              "accepted patch")
    else:
        p4 = abs(ca - closures["adv"][m]) <= 0.10
        print(f"  P4 observable/exact on adversarial (k*={Wa.shape[1]}): "
              f"c_obs {ca:.1%} vs exact {closures['adv'][m]:.1%} — "
              f"{'HOLDS' if p4 else 'FAILS'}")
    if Wa.shape[1] >= 2:
        p5 = max(plane_ang["adv"]) <= 15.0
        print(f"  P5 plane containment (k* >= 2, both <= 15 deg): "
              + "/".join(f"{a:.1f}" for a in plane_ang["adv"])
              + f" — {'HOLDS' if p5 else 'FAILS'}")
    elif Wa.shape[1] == 1:
        p5 = False
        print("  P5 plane containment: FAILS — k* = 1 cannot contain the "
              "2-D plane")
    else:
        p5 = False
        print("  P5 plane containment: NOT TESTED — no accepted adversarial "
              "subspace")
    if cl_d2 is None:
        p6 = False
        print("  P6 D2 composition: NOT TESTED — fewer than two near-plane "
              "round-1 writes")
    else:
        p6 = cl_d2[m] >= 0.90 * cl_full[m]
        print(f"  P6 D2 composition >= 90% of full: {cl_d2[m]:.1%} — "
              f"{'HOLDS' if p6 else 'FAILS'}")
    print(f"  P7 validity gate: {'HOLDS' if p7 else 'FAILS'}")
    print(f"\nsource attribution: benign {sb}; adversarial {sa}")

    # ----- plot -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        names = ["full", "plane"] + (["D2"] if cl_d2 else []) + list(results)
        vals = [cl_full[m], cl_plane[m]] + ([cl_d2[m]] if cl_d2 else []) \
            + [closures[r][m] for r in results]
        ax.bar(range(len(names)), vals)
        ax.axhline(0.9 * cl_full[m], ls="--", c="r", label="90% of full")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(f"exact closure, pooled m={m}")
        ax.set_title(f"{proc.name}: fractional-precision read search "
                     f"(kappa={args.kappa:g})")
        ax.legend(fontsize=8)
        p = os.path.join(args.outdir, "experiment12.png")
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plotting skipped: {e})")


if __name__ == "__main__":
    main()
