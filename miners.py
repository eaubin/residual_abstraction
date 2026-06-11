"""
miners.py — Experiment 9: scale-free proposal mining.

CONTEXT (see experiments/9-scale-free-mining.md, the pre-registration, and
FORMALISM.md §5, the design basis). Experiment 8 split the discovery method:
behavioral ACCEPTANCE produced no false confidence anywhere, but the
covariance PROPOSAL miner is variance-dependent and died (k* = 0) in
adversarial coordinates. This experiment keeps the acceptance loop frozen
and compares three proposal miners in both coordinate regimes of the same
Experiment-6/8 setting:

  M1  covariance miner (incumbent; benign run doubles as the exp-6 anchor,
      adversarial run as the exp-8 reproduction)
  M2  whitened miner — the invariance repair: the same weighted
      second-moment eigenvector, computed on stream-covariance-whitened
      prefix differences, patching by coordinate swap in whitened
      coordinates. FORMALISM.md's proposition says its induced stream-space
      patch is reparameterization-invariant (ridgeless); P5 asserts this
      numerically.
  M3  centered-weight miner — the decorrelation alternative: weights w - w̄
      so bulk delta energy uncorrelated with behavioral failure cancels.
      Not provably safe; registered at ~50% credence either way.

Implementation notes (pre-run): the pre-adversarial invariance check aborts
only on GROSS violation (relative Frobenius > 0.5 — machinery-bug level);
P5's registered tolerance (0.05 / 5 deg) is a VERDICT, not an abort, since
"invariance gap: P5 fails while P2 holds" is a registered failure mode that
must be observable. Discovered stream-space subspaces are read uniformly as
the row space of the induced raw patch matrix (top right-singular vectors),
which is convention-proof across orthogonal and oblique patches. P5 "at
every accepted k" is implemented on nested slices of the post-coarsen
bases (coarsening is expected to drop nothing; if it does, the comparison
covers the surviving prefix and says so).

Run: python3 miners.py --outdir out/mess3-L4
`--selftest` runs the standard four machinery checks and exits.
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
from model import GPT, GPTConfig
from processes import PROCESSES

LAYER = 1

REGISTERED = {"kappa": 100.0, "k_max": 8, "eps_gain": 0.05,
              "eps_drop": 0.01, "pairs_disc": 400, "pairs_eval": 600,
              "basis_seqs": 800, "m": 3}
EIG_FLOOR = 1e-10                 # registered eigenvalue floor (x lambda_max)


def sqrt_and_inv(Sig):
    w, V = np.linalg.eigh(Sig)
    w = np.maximum(w, EIG_FLOOR * w.max())
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


def patch_rowspace(P, k):
    """The stream-space subspace a (possibly oblique) patch writes into:
    top-k right-singular vectors of P (row convention: x' = x + delta @ P)."""
    _, _, Vt = np.linalg.svd(P)
    return Vt[:k].T


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
        print("Experiment 9 is registered for the Experiment-6/8 setting "
              "(mess3 / 4 layers / d_model 64 / seq_len 32 / burn_in 4); "
              f"this config is {cfg['process']} L{cfg['layers']} "
              f"d{cfg['d_model']} T{cfg['seq_len']} b{cfg['burn_in']}. "
              "Use --selftest or --force-invalid.")
        return
    overridden = [k for k, v in REGISTERED.items() if getattr(args, k) != v]
    if overridden and not args.selftest and not args.force_invalid:
        print(f"Experiment 9 parameters are registered; overridden: "
              f"{overridden}. Use --force-invalid for an exploratory run.")
        return
    if overridden:
        print(f"NOTE: EXPLORATORY RUN — non-registered parameters "
              f"{overridden}; verdicts below are NOT Experiment 9.\n")
    if args.seed != 0 and not args.selftest and not args.force_invalid:
        print(f"Experiment 9 registers seed 0 (T's anchor depends on it); "
              f"got {args.seed}. Use --force-invalid.")
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

    # ----- validity gate (P7, enforced as in experiments 5-8) ----------------
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

    print(f"=== Experiment 9: scale-free mining | {proc.name} | patch "
          f"L{LAYER} | kappa = {args.kappa:g} | miners M1/M2/M3 x "
          "benign/adversarial ===\n")

    # ----- observable refs ----------------------------------------------------
    q_src_d = disc.run(model, None, src_side=True)
    q_un_d = disc.run(model, None)
    q_full_d = disc.run(model, np.eye(d))
    D0 = float(kl_rows(q_src_d, q_un_d).mean())
    Dfull = float(kl_rows(q_src_d, q_full_d).mean())
    assert D0 > Dfull
    c_obs = lambda q: (D0 - float(kl_rows(q_src_d, q).mean())) / (D0 - Dfull)
    assert abs(c_obs(q_full_d) - 1.0) < 1e-12
    w0 = kl_rows(q_src_d, q_un_d)               # k=0 mining weights, shared

    def cegar(mine, patch_of, label):
        """The frozen acceptance loop; only the proposal map varies."""
        Q = np.zeros((d, 0))
        q_cur, c_cur = q_un_d, 0.0
        converged = False
        print(f"  [{label}] k=0: c_obs 0.0%")
        while Q.shape[1] < args.k_max:
            v = mine(Q, kl_rows(q_src_d, q_cur))
            assert abs(np.linalg.norm(v) - 1) < 1e-9 and (
                Q.shape[1] == 0 or np.abs(Q.T @ v).max() < 1e-9)
            Q_try = np.hstack([Q, v[:, None]])
            q_try = disc.run(model, patch_of(Q_try))
            c_try = c_obs(q_try)
            gain = c_try - c_cur
            if gain < args.eps_gain:
                print(f"  [{label}] k={Q_try.shape[1]}: c_obs {c_try:.1%} "
                      f"(gain {gain:+.1%} < eps_gain) -> STOP, revert")
                converged = True
                break
            Q, q_cur, c_cur = Q_try, q_try, c_try
            print(f"  [{label}] k={Q.shape[1]}: c_obs {c_cur:.1%} "
                  f"(gain {gain:+.1%}) -> accept")
        if not converged:
            print(f"  [{label}] k_max reached: NON-CONVERGENT")
        changed = True
        while changed and Q.shape[1] > 1:
            changed = False
            for j in range(Q.shape[1]):
                Qj = np.delete(Q, j, axis=1)
                cj = c_obs(disc.run(model, patch_of(Qj)))
                if c_cur - cj < args.eps_drop:
                    print(f"  [{label}] coarsen: dropped direction {j + 1}")
                    Q, c_cur, changed = Qj, cj, True
                    break
        print(f"  [{label}] fixed point: k* = {Q.shape[1]}, "
              f"c_obs = {c_cur:.1%}\n")
        return Q, c_cur, converged

    # ----- M1-benign = the exp-6 anchor ---------------------------------------
    print("[M1/benign] (doubles as the Experiment-6 anchor):")
    ident = lambda Q: Q @ Q.T
    Qc, c6, conv6 = cegar(lambda Q, w: mined_direction(disc, Q, w),
                          ident, "M1/ben")
    assert conv6 and Qc.shape[1] == 2 and abs(c6 - 0.998) < 0.005, \
        "failed to reproduce Experiment 6's fixed point"
    Pc = Qc @ Qc.T

    # ----- T (the exp-8 registered construction) ------------------------------
    kap = args.kappa
    rng0 = np.random.default_rng(0)
    Gj = rng0.standard_normal((d, 2))
    Gj -= Qc @ (Qc.T @ Gj)
    Qj = orthonormal(Gj)
    T = np.eye(d) - (1 - 1 / kap) * Pc + (kap - 1) * (Qj @ Qj.T)
    Tinv = np.eye(d) + (kap - 1) * Pc + (1 / kap - 1) * (Qj @ Qj.T)
    assert np.allclose(T @ Tinv, np.eye(d), atol=1e-9)
    pull_adv = lambda Pz: T @ Pz @ Tinv
    assert np.allclose(pull_adv(np.eye(d)), np.eye(d), atol=1e-9)
    assert np.allclose(pull_adv(orthonormal(T @ Qc) @ orthonormal(T @ Qc).T),
                       Pc, atol=1e-9)
    print("transform checks passed (pull(I) = I, pull(T·plane) = plane)\n")

    # ----- whitening (registered floor) ---------------------------------------
    rng_b = np.random.default_rng(args.seed + 555)
    Xb = proc.sample(args.basis_seqs, L, rng_b)
    Sb = stream_to(model, torch.from_numpy(Xb), LAYER).double().numpy()
    keep = np.arange(burn, L - 1)
    Rb = center_by_position(Sb[:, keep].reshape(-1, d), np.tile(keep, len(Xb)),
                            np.ones(len(Xb) * len(keep), dtype=bool))
    Sig_x = np.cov(Rb.T)
    Sig_z = T @ Sig_x @ T                       # T symmetric; z = x @ T
    Sx, Sx_inv = sqrt_and_inv(Sig_x)
    Sz, Sz_inv = sqrt_and_inv(Sig_z)

    # miner factories per regime: (mine, patch_of) pairs
    view_adv = ZView(disc, T)
    view_w_x = ZView(disc, Sx_inv)
    view_w_z = ZView(disc, T @ Sz_inv)
    miners = {
        ("M1", "adv"): (lambda Q, w: mined_direction(view_adv, Q, w),
                        lambda Q: pull_adv(Q @ Q.T)),
        ("M2", "ben"): (lambda Q, w: mined_direction(view_w_x, Q, w),
                        lambda Q: Sx_inv @ Q @ Q.T @ Sx),
        ("M2", "adv"): (lambda Q, w: mined_direction(view_w_z, Q, w),
                        lambda Q: pull_adv(Sz_inv @ Q @ Q.T @ Sz)),
        ("M3", "ben"): (lambda Q, w: mined_direction(disc, Q, w - w.mean()),
                        ident),
        ("M3", "adv"): (lambda Q, w: mined_direction(view_adv, Q,
                                                     w - w.mean()),
                        lambda Q: pull_adv(Q @ Q.T)),
    }

    # ----- pre-adversarial invariance probe (gross-violation abort only) ----
    v_x = miners[("M2", "ben")][0](np.zeros((d, 0)), w0)
    v_z = miners[("M2", "adv")][0](np.zeros((d, 0)), w0)
    P1x = miners[("M2", "ben")][1](v_x[:, None])
    P1z = miners[("M2", "adv")][1](v_z[:, None])
    rel1 = np.linalg.norm(P1x - P1z) / np.linalg.norm(P1x)
    ang1 = float(principal_angles_deg(patch_rowspace(P1x, 1),
                                      patch_rowspace(P1z, 1))[0])
    print(f"invariance probe (M2, k=1): relative Frobenius {rel1:.4f}, "
          f"first-direction angle {ang1:.2f} deg")
    assert rel1 <= 0.5, ("gross invariance violation — machinery bug level; "
                         "aborting before adversarial loops")
    print()

    # ----- the loops ------------------------------------------------------------
    results = {("M1", "ben"): (Qc, c6, conv6)}
    for key in (("M1", "adv"), ("M2", "ben"), ("M2", "adv"),
                ("M3", "ben"), ("M3", "adv")):
        mine, patch_of = miners[key]
        print(f"[{key[0]}/{key[1]}]:")
        results[key] = cegar(mine, patch_of, "/".join(key))

    def patch_of_key(key, Q):
        return ident(Q) if key == ("M1", "ben") else miners[key][1](Q)

    # ----- evaluation (exact targets, disjoint pairs) --------------------------
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
        bp = {t: (float(rows_ge[m][idx].mean())
                  - float(rows_t[m][idx].mean()))
              / (float(rows_ge[m][idx].mean())
                 - float(rows_fe[m][idx].mean()))
              for t, idx in ev.groups}
        return cl, bp

    print("evaluation (exact targets; pooled m=1..3 | per-position at m=3):")
    closures, angles = {}, {}
    cl_full, bp = true_closure(np.eye(d))
    print(f"  {'full':>8} (k=64)  "
          + "  ".join(f"m={mm}: {cl_full[mm]:>6.1%}" for mm in ms)
          + "   | " + "  ".join(f"t={t}: {bp[t]:.1%}" for t, _ in ev.groups))
    for key, (Q, c, conv) in results.items():
        name = "/".join(key)
        if Q.shape[1] == 0:
            closures[key] = {mm: 0.0 for mm in ms}
            print(f"  {name:>8} (k=0)   closure 0 by definition (loop "
                  "accepted nothing)")
            continue
        P = patch_of_key(key, Q)
        cl, bp = true_closure(P)
        closures[key] = cl
        sub = patch_rowspace(P, Q.shape[1])
        ang = principal_angles_deg(sub, Qc)
        pd = float(np.linalg.norm(sub @ sub.T - Pc))
        angles[key] = (ang, pd)
        print(f"  {name:>8} (k={Q.shape[1]})  "
              + "  ".join(f"m={mm}: {cl[mm]:>6.1%}" for mm in ms)
              + "   | " + "  ".join(f"t={t}: {bp[t]:.1%}"
                                    for t, _ in ev.groups)
              + f"   | vs exp6-plane: max angle {max(ang):.1f} deg, "
              f"proj-dist {pd:.3f}")

    # ----- P5: invariance at every accepted k (nested slices) ----------------
    Qb, Qa = results[("M2", "ben")][0], results[("M2", "adv")][0]
    k_common = min(Qb.shape[1], Qa.shape[1])
    rels = []
    for k in range(1, k_common + 1):
        Pb = miners[("M2", "ben")][1](Qb[:, :k])
        Pa = miners[("M2", "adv")][1](Qa[:, :k])
        rels.append(np.linalg.norm(Pb - Pa) / np.linalg.norm(Pb))
    print("\nP5 invariance, M2 benign-vs-adversarial induced patches, "
          "relative Frobenius by k: "
          + ", ".join(f"k={k + 1}: {r:.4f}" for k, r in enumerate(rels)))

    # ----- verdicts -------------------------------------------------------------
    print("\nverdicts:")
    m1b, m1a = results[("M1", "ben")], results[("M1", "adv")]
    p1 = (m1b[0].shape[1] == 2
          and closures[("M1", "ben")][m] >= 0.90 * cl_full[m]
          and m1a[0].shape[1] == 0)
    print(f"  P1 anchors (M1-ben k*=2 & >=90% full; M1-adv k*=0): "
          f"{'HOLDS' if p1 else 'FAILS'}")
    Q2a, c2a, conv2a = results[("M2", "adv")]
    p2 = (conv2a and Q2a.shape[1] == 2
          and closures[("M2", "adv")][m] >= 0.90 * cl_full[m]
          and ("M2", "adv") in angles
          and max(angles[("M2", "adv")][0]) <= 15.0)
    print(f"  P2 repair (M2-adv k*=2, >=90% full, plane <=15 deg): "
          f"k*={Q2a.shape[1]}, closure {closures[('M2', 'adv')][m]:.1%}"
          + (f", max angle {max(angles[('M2', 'adv')][0]):.1f} deg"
             if ("M2", "adv") in angles else "")
          + f" — {'HOLDS' if p2 else 'FAILS'}")
    p3 = closures[("M2", "ben")][m] >= closures[("M1", "ben")][m] - 0.02
    print(f"  P3 no benign cost (M2-ben >= M1-ben - 2pts): "
          f"{closures[('M2', 'ben')][m]:.1%} vs "
          f"{closures[('M1', 'ben')][m]:.1%} — {'HOLDS' if p3 else 'FAILS'}")
    p4 = abs(c2a - closures[("M2", "adv")][m]) <= 0.10
    print(f"  P4 observable/exact on M2-adv: c_obs {c2a:.1%} vs exact "
          f"{closures[('M2', 'adv')][m]:.1%} — {'HOLDS' if p4 else 'FAILS'}"
          + ("" if Q2a.shape[1] else "  (null output — nontrivial test "
             "requires an accepted patch)"))
    p5 = bool(rels) and max(rels) <= 0.05 and ang1 <= 5.0
    print(f"  P5 invariance (rel-Frob <= 0.05 at every accepted k; "
          f"first-direction <= 5 deg): max rel {max(rels) if rels else float('nan'):.4f}, "
          f"angle {ang1:.2f} deg — {'HOLDS' if p5 else 'FAILS'}")
    p6 = closures[("M3", "adv")][m] >= 0.50 * cl_full[m]
    print(f"  P6 decorrelation (M3-adv >= 50% of full): "
          f"{closures[('M3', 'adv')][m]:.1%} — {'HOLDS' if p6 else 'FAILS'}")
    print(f"  P7 validity gate: {'HOLDS' if p7 else 'FAILS'}")

    # ----- characterization: kappa sweep for M2-adv ---------------------------
    print("\nkappa sweep (M2-adversarial; characterization, no thresholds):")
    for kap_s in (10.0, 1000.0):
        T_s = np.eye(d) - (1 - 1 / kap_s) * Pc + (kap_s - 1) * (Qj @ Qj.T)
        Tinv_s = np.eye(d) + (kap_s - 1) * Pc + (1 / kap_s - 1) * (Qj @ Qj.T)
        Sig_zs = T_s @ Sig_x @ T_s
        Szs, Szs_inv = sqrt_and_inv(Sig_zs)
        view_s = ZView(disc, T_s @ Szs_inv)
        patch_s = lambda Q: T_s @ (Szs_inv @ Q @ Q.T @ Szs) @ Tinv_s
        Qs, cs, _ = cegar(lambda Q, w: mined_direction(view_s, Q, w),
                          patch_s, f"M2/adv k={kap_s:g}")
        cl_s = (true_closure(patch_s(Qs))[0][m] if Qs.shape[1] else 0.0)
        print(f"  kappa={kap_s:g}: k*={Qs.shape[1]}, c_obs {cs:.1%}, "
              f"exact {cl_s:.1%}")

    # ----- plot -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        names = ["/".join(k) for k in results]
        vals = [closures[k][m] for k in results]
        ax.bar(range(len(names)), vals)
        ax.axhline(cl_full[m], ls="--", c="g", label="full")
        ax.axhline(0.9 * cl_full[m], ls="--", c="r", label="90% of full")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(
            [f"{n}\nk={results[k][0].shape[1]}"
             for n, k in zip(names, results)], fontsize=8)
        ax.set_ylabel(f"exact closure, pooled m={m}")
        ax.set_title(f"{proc.name}: proposal miners x regimes "
                     f"(kappa={args.kappa:g})")
        ax.legend(fontsize=8)
        p = os.path.join(args.outdir, "experiment9.png")
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plotting skipped: {e})")


if __name__ == "__main__":
    main()
