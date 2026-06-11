"""
adversarial.py — Experiment 8: adversarial coordinates — the
variance-mimicry discriminator.

CONTEXT (see experiments/8-adversarial-coordinates.md; the design was
frozen at commit 58abc69, BEFORE Experiment 7 ran; this code was written
after Experiment 7 concluded, per the registered ordering, with the design
unchanged). Experiments 6 and 7 both ended with the declared limitation:
the interventional CEGAR loop converged onto (nearly) the PCA plane, so
"interventional discovery works" and "variance was right anyway" were
indistinguishable. This experiment manufactures the regime where they must
come apart: all DISCOVERY operates in adversarially ill-conditioned
coordinates z = T·(stream at L1) of the concluded Experiment-6 setting,
while every patch is pulled back through T^-1 before touching the model —
the model and its behavior are bit-identical to Experiment 6; only the
coordinate system handed to the discovery procedures is hostile.

THE TRANSFORM (registered construction rule): T = I − (1 − 1/κ)·P_c +
(κ − 1)·P_j with κ = 100, P_c the projector onto Experiment 6's discovered
causal plane (k* = 2; reproduced in-run deterministically and asserted
against the recorded fixed point), and P_j the projector onto 2 fixed
random directions drawn orthogonal to it (seed 0). T is symmetric with the
closed-form inverse T^-1 = I + (κ − 1)·P_c + (1/κ − 1)·P_j; in z the
causal plane's variance is suppressed ×κ² and junk variance amplified ×κ².

THE PULLBACK: a z-space orthonormal basis Q_z induces the stream-space
patch matrix P_x = T·Q_z·Q_zᵀ·T^-1 (oblique), so that the patched stream's
z-coordinates receive exactly the orthogonal interchange in z and the
z-complement is untouched. Note the causal plane itself is T-invariant as a
subspace (T acts on it as 1/κ · I), so a loop that finds the causal content
should discover Q_z ≈ the same plane.

A MECHANICAL CONSEQUENCE OF THE FROZEN DESIGN, stated before running: the
loop's PROPOSAL step (mined_direction: weighted second-moment eigenvector
of prefix differences) is variance-driven even though its ACCEPTANCE is
behavioral. In hostile coordinates the mined matrix is dominated by
amplified junk (×κ⁴ relative to the causal plane), so the registered
failure mode "variance dependence exposed" (P2 fails, possibly with the
loop stopping at k* = 0 when the first junk proposal earns no behavioral
gain) is a live outcome the code handles explicitly — it would falsify the
part of the method claim that Experiments 6–7 could not test.

SELF-CHECKS: the four standard known-answer checks (run in --selftest and
in real runs), plus the two registered transform checks which need the
reproduced causal plane and therefore run in real runs only:
(i) the pullback of the full z-space is exactly the identity patch;
(ii) the pullback of T·(exp-6 plane) is exactly the exp-6 plane's patch
(T commutes with P_c). Loop invariants (unit-norm, z-orthogonal proposals;
c_obs(full) = 1; D0 > D_full) asserted as in Experiment 6.

Run: python3 adversarial.py --outdir out/mess3-L4
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from abstraction import (CompletionPLS, PCAAbstraction, center_by_position,
                         kl_rows)
from discover import (PairSet, mined_direction, principal_angles_deg,
                      self_checks)
from midstream import kl_by_horizon, orthonormal, stream_to
from model import GPT, GPTConfig
from processes import PROCESSES

LAYER = 1

REGISTERED = {"kappa": 100.0, "k_max": 8, "eps_gain": 0.05,
              "eps_drop": 0.01, "pairs_disc": 400, "pairs_eval": 600,
              "basis_seqs": 800, "m": 3}


class ZView:
    """A PairSet view whose prefix arrays live in z = T·stream coordinates —
    exactly what mined_direction needs, nothing else."""

    def __init__(self, ps, T):
        Tt = torch.from_numpy(T)            # T symmetric: z_row = x_row @ T
        self.groups, self.d = ps.groups, ps.d
        self.pref_src = {t: ps.pref_src[t].double() @ Tt for t in ps.pref_src}
        self.pref_tgt = {t: ps.pref_tgt[t].double() @ Tt for t in ps.pref_tgt}


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
        print("Experiment 8 is registered for the Experiment-6 setting "
              "(mess3 / 4 layers / d_model 64 / seq_len 32 / burn_in 4); "
              f"this config is {cfg['process']} L{cfg['layers']} "
              f"d{cfg['d_model']} T{cfg['seq_len']} b{cfg['burn_in']}. "
              "Use --selftest or --force-invalid.")
        return
    overridden = [k for k, v in REGISTERED.items() if getattr(args, k) != v]
    if overridden and not args.selftest and not args.force_invalid:
        print(f"Experiment 8 parameters are registered; overridden: "
              f"{overridden}. Use --force-invalid for an exploratory run.")
        return
    if overridden:
        print(f"NOTE: EXPLORATORY RUN — non-registered parameters "
              f"{overridden}; verdicts below are NOT Experiment 8.\n")
    if args.seed != 0:
        print(f"NOTE: seed {args.seed} != 0 — a seed-robustness rerun.\n")

    L, burn, V, m = cfg["seq_len"], cfg["burn_in"], proc.V, args.m
    d = cfg["d_model"]
    model = GPT(GPTConfig(vocab=V, seq_len=L, d_model=d,
                          n_layers=cfg["layers"]))
    model.load_state_dict(torch.load(os.path.join(args.outdir, "model.pt"),
                                     map_location="cpu"))
    model.eval()

    # ----- validity gate (enforced, as in experiments 5-7) -------------------
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
    gate = gap_opt <= 0.005
    print(f"validity gate: gap-to-optimal {gap_opt:+.4f} nats — "
          f"{'PASS' if gate else 'FAIL'}\n")
    if not gate and not args.selftest and not args.force_invalid:
        print("exiting: validity gate failed.")
        return

    disc = PairSet(model, proc, cfg, args.pairs_disc, m, args.seed + 111, 800,
                   layer=LAYER)
    ev = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777, 800,
                 layer=LAYER)
    self_checks(model, ev, LAYER, m, V)
    if args.selftest:
        return

    print(f"=== Experiment 8: adversarial coordinates | {proc.name} | "
          f"patch L{LAYER} | kappa = {args.kappa:g} | "
          f"{args.pairs_disc} discovery / {args.pairs_eval} evaluation "
          "pairs ===\n")

    # ----- observable refs (stream space; identical to Experiment 6) ---------
    q_src_d = disc.run(model, None, src_side=True)
    q_un_d = disc.run(model, None)
    q_full_d = disc.run(model, np.eye(d))
    D0 = float(kl_rows(q_src_d, q_un_d).mean())
    Dfull = float(kl_rows(q_src_d, q_full_d).mean())
    assert D0 > Dfull, "observable scale degenerate"
    c_obs = lambda q: (D0 - float(kl_rows(q_src_d, q).mean())) / (D0 - Dfull)
    assert abs(c_obs(q_full_d) - 1.0) < 1e-12

    # ----- reproduce Experiment 6's causal plane (T's construction input) ----
    def cegar(view, pull):
        """The registered loop: proposals mined in `view` coordinates,
        accepted by behavioral closure of the pulled-back patch."""
        Q = np.zeros((d, 0))
        q_cur, c_cur = q_un_d, 0.0
        print(f"    k=0: c_obs {c_cur:.1%}")
        converged = False
        while Q.shape[1] < args.k_max:
            v = mined_direction(view, Q, kl_rows(q_src_d, q_cur))
            assert abs(np.linalg.norm(v) - 1) < 1e-9 and (
                Q.shape[1] == 0 or np.abs(Q.T @ v).max() < 1e-9)
            Q_try = np.hstack([Q, v[:, None]])
            q_try = disc.run(model, pull(Q_try))
            c_try = c_obs(q_try)
            gain = c_try - c_cur
            if gain < args.eps_gain:
                print(f"    k={Q_try.shape[1]}: c_obs {c_try:.1%} "
                      f"(gain {gain:+.1%} < eps_gain) -> STOP, revert")
                converged = True
                break
            Q, q_cur, c_cur = Q_try, q_try, c_try
            print(f"    k={Q.shape[1]}: c_obs {c_cur:.1%} "
                  f"(gain {gain:+.1%}) -> accept")
        if not converged:
            print(f"    k_max = {args.k_max} reached: NON-CONVERGENT")
        changed = True
        while changed and Q.shape[1] > 1:
            changed = False
            for j in range(Q.shape[1]):
                Qj = np.delete(Q, j, axis=1)
                cj = c_obs(disc.run(model, pull(Qj)))
                if c_cur - cj < args.eps_drop:
                    print(f"    coarsen: dropped direction {j + 1}")
                    Q, c_cur, changed = Qj, cj, True
                    break
        return Q, c_cur, converged

    print("[anchor] reproducing the Experiment-6 loop (stream coordinates):")
    Q_c, c6, conv6 = cegar(disc, lambda Q: Q @ Q.T)
    assert conv6 and Q_c.shape[1] == 2 and abs(c6 - 0.998) < 0.005, \
        "failed to reproduce Experiment 6's recorded fixed point"
    print(f"[anchor] reproduced: k* = 2, c_obs = {c6:.1%} "
          "(matches the recorded 99.8%)\n")

    # ----- build T (registered construction; junk directions seed 0) ---------
    kap = args.kappa
    rng0 = np.random.default_rng(0)
    Gj = rng0.standard_normal((d, 2))
    Gj -= Q_c @ (Q_c.T @ Gj)
    Q_j = orthonormal(Gj)
    Pc, Pj = Q_c @ Q_c.T, Q_j @ Q_j.T
    T = np.eye(d) - (1 - 1 / kap) * Pc + (kap - 1) * Pj
    Tinv = np.eye(d) + (kap - 1) * Pc + (1 / kap - 1) * Pj
    assert np.allclose(T @ Tinv, np.eye(d), atol=1e-9)
    pull = lambda Qz: T @ Qz @ Qz.T @ Tinv

    # registered transform checks (i) and (ii)
    assert np.allclose(pull(np.eye(d)), np.eye(d), atol=1e-9), \
        "check (i): pullback of the full z-space != identity patch"
    Qz_c = orthonormal(T @ Q_c)
    assert np.allclose(pull(Qz_c), Pc, atol=1e-9), \
        "check (ii): pullback of T·(exp-6 plane) != exp-6 plane's patch"
    print("transform checks passed: pull(I) = I, pull(T·plane) = plane "
          "(T commutes with P_c)\n")

    # ----- coordinate-hostility audit + control bases in z -------------------
    rng_d = np.random.default_rng(args.seed + 555)
    Xd = proc.sample(args.basis_seqs, L, rng_d)
    Sd = stream_to(model, torch.from_numpy(Xd), LAYER).double().numpy()
    keep = np.arange(burn, L - 1)
    Gd = np.concatenate([proc.mgram_table(proc.beliefs_along(row)[keep], m)
                         for row in Xd])
    Rd = center_by_position(Sd[:, keep].reshape(-1, d), np.tile(keep, len(Xd)),
                            np.ones(len(Xd) * len(keep), dtype=bool))
    Zd = Rd @ T
    var_c = float(((Zd @ Q_c) ** 2).sum() / (Zd ** 2).sum())
    pca_z = PCAAbstraction(Zd)
    ang_pca_c = principal_angles_deg(pca_z.Vt[:2].T, Q_c)
    print(f"coordinate hostility: causal-plane variance share in z = "
          f"{var_c:.2e}; pca-z top-2 vs causal plane angles = "
          + ", ".join(f"{a:.1f}" for a in ang_pca_c) + " deg\n")

    # ----- the experiment: CEGAR in z ------------------------------------------
    print("[z] CEGAR in adversarial coordinates:")
    Qz, c_z, conv_z = cegar(ZView(disc, T), pull)
    k_star = Qz.shape[1]
    print(f"[z] fixed point: k* = {k_star}, c_obs = {c_z:.1%}\n")

    # ----- evaluation (exact targets; controls at the registered k = 2) ------
    pls_z = CompletionPLS(Zd, Gd)
    rng = np.random.default_rng(args.seed)
    bases_z = {
        "pca-z": pca_z.Vt[:2].T,
        "pls-z": orthonormal(pls_z.whiten @ pls_z.U[:, :2]),
        "rand-z": orthonormal(rng.standard_normal((d, 2))),
    }
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

    conds = [("full", np.eye(d), d), ("exp6-plane", Pc, 2)]
    if k_star > 0:
        conds.append(("disc-z", pull(Qz), k_star))
    conds += [(f, pull(Bz), Bz.shape[1]) for f, Bz in bases_z.items()]
    print("evaluation (disjoint pairs, exact targets; pooled m=1..3 | "
          "per-position at m=3):")
    closures = {}
    for name, P, kk in conds:
        cl, bp = true_closure(P)
        closures[name] = cl
        print(f"  {name:>10} (k={kk})  "
              + "  ".join(f"m={mm}: {cl[mm]:>6.1%}" for mm in ms)
              + "   | " + "  ".join(f"t={t}: {bp[t]:.1%}"
                                    for t, _ in ev.groups))
    if k_star == 0:
        closures["disc-z"] = {mm: 0.0 for mm in ms}
        print("  (disc-z: k* = 0 — the loop accepted nothing; closure 0 by "
              "definition)")

    if k_star > 0:
        print(f"\nnested closure(k) of disc-z (pooled m={m}):")
        for k in range(1, k_star + 1):
            Qk = Qz[:, :k]
            cl, _ = true_closure(pull(Qk))
            print(f"  k={k}: {cl[m]:.1%}")
        back = orthonormal(Tinv @ Qz)
        ang = principal_angles_deg(back, Q_c)
        print("\npulled-back disc-z plane vs exp-6 causal plane, principal "
              "angles: " + ", ".join(f"{a:.1f}" for a in ang) + " deg")
    else:
        ang = None

    # ----- verdicts (experiments/8-adversarial-coordinates.md) ---------------
    print("\nverdicts:")
    p1 = closures["pca-z"][m] <= 0.25
    print(f"  P1 pca-on-z (k=2) <= 25%: {closures['pca-z'][m]:.1%} — "
          f"{'HOLDS' if p1 else 'FAILS'}")
    p2 = conv_z and 0 < k_star <= 4 and \
        closures["disc-z"][m] >= 0.90 * closures["full"][m]
    print(f"  P2 loop converges at k* <= 4 with >= 90% of full: k*={k_star},"
          f" disc-z {closures['disc-z'][m]:.1%} vs full "
          f"{closures['full'][m]:.1%} — {'HOLDS' if p2 else 'FAILS'}"
          + ("" if p2 else "  (typed: VARIANCE DEPENDENCE EXPOSED — the "
             "proposal step, not the scoring, chases amplified junk)"))
    p3 = ang is not None and max(ang) <= 15.0
    print(f"  P3 pulled-back plane within 15 deg of exp-6 plane: "
          + (f"max angle {max(ang):.1f} deg — " if ang is not None
             else "no plane discovered — ")
          + ("HOLDS" if p3 else "FAILS"))
    p4 = abs(c_z - closures["disc-z"][m]) <= 0.10
    print(f"  P4 observable/exact agreement <= 10 points: c_obs {c_z:.1%} vs "
          f"exact {closures['disc-z'][m]:.1%} — {'HOLDS' if p4 else 'FAILS'}")
    p5 = closures["rand-z"][m] <= 0.25 and gate
    print(f"  P5 rand-z <= 25% and gate: rand-z {closures['rand-z'][m]:.1%}, "
          f"gate {'PASS' if gate else 'FAIL'} — {'HOLDS' if p5 else 'FAILS'}")

    # ----- plot ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(closures)
        ax.bar(range(len(names)), [closures[f][m] for f in names])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.axhline(0.9 * closures["full"][m], ls="--", c="r",
                   label="P2 bar (90% of full)")
        ax.axhline(0.25, ls=":", c="gray", label="P1/P5 bar")
        ax.set_ylabel(f"exact-target closure, pooled m={m}")
        ax.set_title(f"{proc.name}: discovery under adversarial coordinates "
                     f"(kappa={args.kappa:g}), k*={k_star}")
        ax.legend(fontsize=8)
        p = os.path.join(args.outdir, "experiment8.png")
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plotting skipped: {e})")


if __name__ == "__main__":
    main()
