"""
patches.py — Experiment 11: patch parameterization as proposal space.

CONTEXT (see experiments/11-patch-parameterization.md, the pre-registration,
and FORMALISM.md §6.1/§7, which this registration is the first to apply in
full). Experiment 10 cleared interventional selection and indicted the
patch family: the working-coordinate orthogonal swap pairs every write
direction with a junk-amplified read covector, wasting candidates that sat
1.1 deg from the causal plane. Here the candidate is a (write direction,
read covector) PAIR — the patch map joins the search space — generation
stays cheap and fallible (the exp-10 write pool x three read families),
selection stays by measured closure gain, acceptance stays frozen.

Read families per write w (all constructible from working-coordinate
data): id (c = w; the exp-10 baseline), prec (c ∝ Σ̂⁻¹w; the GLS read,
provably coordinate-equivariant — the z-pair (w, Σ̂_z⁻¹w) pulls back to
the x-pair of the pulled-back write), cov (c ∝ Σ̂w; anti-rationale
control). Reads normalized to <c, w> = 1.

Composition (ledger row in the registration): accepted pairs form the
oblique projector with row-convention patch P = C (WᵀC)⁻¹ Wᵀ — all
accepted read-functionals set to the source's values, writing only in
span(W); candidates pushing cond(WᵀC) > 1e6 are skipped, skips printed.

Diagnostic anchors (evaluation-side, labeled, NOT discoverable: they use
T or the known plane): the known-plane orthogonal patch (ceiling), and
the stream-orthogonal patch of round 1's nearest-to-plane write — the
direct verification of exp-10's read-side mechanism (P6).

Run: python3 patches.py --outdir out/mess3-L4   (~45-60 min: up to 36
chain evaluations per round)
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
from miners import sqrt_and_inv
from model import GPT, GPTConfig
from processes import PROCESSES

LAYER = 1

REGISTERED = {"kappa": 100.0, "k_max": 8, "eps_gain": 0.05,
              "eps_drop": 0.01, "pairs_disc": 400, "pairs_eval": 600,
              "basis_seqs": 800, "m": 3}
COND_MAX = 1e6                    # registered composition guard


def oblique_patch(C, W):
    """Row-convention patch P = C (WᵀC)⁻¹ Wᵀ: sets every accepted
    read-functional to the source's value, writing only in span(W)."""
    return C @ np.linalg.solve(W.T @ C, W.T)


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
        print("Experiment 11 is registered for the Experiment-6/8/9/10 "
              "setting (mess3 / 4 layers / d_model 64 / seq_len 32 / "
              f"burn_in 4); this config is {cfg['process']} L{cfg['layers']}"
              f" d{cfg['d_model']} T{cfg['seq_len']} b{cfg['burn_in']}. "
              "Use --selftest or --force-invalid.")
        return
    overridden = [k for k, v in REGISTERED.items() if getattr(args, k) != v]
    if overridden and not args.selftest and not args.force_invalid:
        print(f"Experiment 11 parameters are registered; overridden: "
              f"{overridden}. Use --force-invalid for an exploratory run.")
        return
    if overridden:
        print(f"NOTE: EXPLORATORY RUN — non-registered parameters "
              f"{overridden}; verdicts below are NOT Experiment 11.\n")
    if args.seed != 0 and not args.selftest and not args.force_invalid:
        print(f"Experiment 11 registers seed 0; got {args.seed}. "
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

    print(f"=== Experiment 11: patch parameterization as proposal space | "
          f"{proc.name} | patch L{LAYER} | kappa = {args.kappa:g} | "
          "write-pool x {id, prec, cov} reads ===\n")

    # ----- observable refs -----------------------------------------------------
    q_src_d = disc.run(model, None, src_side=True)
    q_un_d = disc.run(model, None)
    q_full_d = disc.run(model, np.eye(d))
    D0 = float(kl_rows(q_src_d, q_un_d).mean())
    Dfull = float(kl_rows(q_src_d, q_full_d).mean())
    assert D0 > Dfull
    c_obs = lambda q: (D0 - float(kl_rows(q_src_d, q).mean())) / (D0 - Dfull)
    assert abs(c_obs(q_full_d) - 1.0) < 1e-12

    # rank-1 regression link to exp-10: id-read patch == orthogonal swap
    w_chk = np.zeros(d); w_chk[0] = 1.0
    assert np.allclose(oblique_patch(w_chk[:, None], w_chk[:, None]),
                       np.outer(w_chk, w_chk), atol=1e-12), \
        "rank-1 id-read patch != orthogonal swap"

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
    print("transform checks passed\n")

    rng_b = np.random.default_rng(args.seed + 555)
    Xb = proc.sample(args.basis_seqs, L, rng_b)
    Sb = stream_to(model, torch.from_numpy(Xb), LAYER).double().numpy()
    keep = np.arange(burn, L - 1)
    Rb = center_by_position(Sb[:, keep].reshape(-1, d), np.tile(keep, len(Xb)),
                            np.ones(len(Xb) * len(keep), dtype=bool))
    Sig_x = np.cov(Rb.T)
    Sig_z = T @ Sig_x @ T

    class Regime:
        def __init__(self, name, view_raw, view_wht, S, Sinv, Sig, Sig_inv,
                     pull, back, fwd):
            self.name, self.view_raw, self.view_wht = name, view_raw, view_wht
            self.S, self.Sinv, self.Sig, self.Sig_inv = S, Sinv, Sig, Sig_inv
            self.pull, self.back, self.fwd = pull, back, fwd

    Sx, Sx_inv = sqrt_and_inv(Sig_x)
    Sz, Sz_inv = sqrt_and_inv(Sig_z)
    _, Sig_x_inv = Sig_x, np.linalg.solve(Sig_x + 1e-10 * np.trace(Sig_x)
                                          / d * np.eye(d), np.eye(d))
    Sig_z_inv = Tinv @ Sig_x_inv @ Tinv
    regimes = {
        "ben": Regime("ben", disc, ZView(disc, Sx_inv), Sx, Sx_inv,
                      Sig_x, Sig_x_inv, lambda P: P, lambda v: v,
                      lambda v: v),
        "adv": Regime("adv", ZView(disc, T), ZView(disc, T @ Sz_inv),
                      Sz, Sz_inv, Sig_z, Sig_z_inv,
                      lambda P: T @ P @ Tinv,
                      lambda v: Tinv @ v,        # write pullback to stream
                      lambda c: T @ c),          # read covector to stream
    }

    def write_pool(rg, W, weights, rnd):
        """The exp-10 write pool, reimplemented identically (registered)."""
        rng = np.random.default_rng(args.seed + 1000 + rnd)
        ones = np.ones_like(weights)
        cands = [("M1", mined_direction(rg.view_raw, W, weights)),
                 ("M3", mined_direction(rg.view_raw, W,
                                        weights - weights.mean()))]
        Ww = orthonormal(rg.Sinv @ W) if W.shape[1] else np.zeros((d, 0))
        u2 = mined_direction(rg.view_wht, Ww, weights)
        cands += [("M2*S", rg.S @ u2), ("M2*Sinv", rg.Sinv @ u2)]
        p1 = mined_direction(rg.view_raw, W, ones)
        cands.append(("dPCA1", p1))
        cands.append(("dPCA2", mined_direction(
            rg.view_raw, orthonormal(np.hstack([W, p1[:, None]])), ones)))
        for i in range(2):
            cands.append(("rand", rng.standard_normal(d)))
        for i in range(2):
            cands.append(("randS", rg.S @ rng.standard_normal(d)))
        for i in range(2):
            cands.append(("randSinv", rg.Sinv @ rng.standard_normal(d)))
        out = []
        for src, v in cands:
            if W.shape[1]:
                v = v - W @ (W.T @ v)
            n = np.linalg.norm(v)
            if n > 1e-8:
                out.append((src, v / n))
        if len(out) < len(cands):
            print(f"      (write pool: {len(out)}/{len(cands)} survive "
                  "residualization)")
        return out

    def reads_for(rg, w):
        """The registered read families, normalized to <c, w> = 1."""
        out = [("id", w.copy())]
        for fam, M in (("prec", rg.Sig_inv), ("cov", rg.Sig)):
            c = M @ w
            ip = float(c @ w)
            if abs(ip) > 1e-12:
                out.append((fam, c / ip))
        return out

    def search(rg):
        W = np.zeros((d, 0))
        C = np.zeros((d, 0))
        q_cur, c_cur = q_un_d, 0.0
        sources, converged = [], False
        rnd, p6_data = 0, None
        while W.shape[1] < args.k_max:
            rnd += 1
            wts = kl_rows(q_src_d, q_cur)
            print(f"  [{rg.name}] round {rnd} (current c_obs {c_cur:.1%}):")
            best, nearest = None, None
            for wsrc, w in write_pool(rg, W, wts, rnd):
                sw = rg.back(w)
                w_ang = principal_angles_deg(
                    (sw / np.linalg.norm(sw))[:, None], Qc)[0]
                for rfam, c in reads_for(rg, w):
                    assert abs(float(c @ w) - 1.0) < 1e-9, "<c,w> != 1"
                    W_try = np.hstack([W, w[:, None]])
                    C_try = np.hstack([C, c[:, None]])
                    if np.linalg.cond(W_try.T @ C_try) > COND_MAX:
                        print(f"      {wsrc:>9}/{rfam:<4}: SKIPPED "
                              "(composition conditioning)")
                        continue
                    P = rg.pull(oblique_patch(C_try, W_try))
                    gain = c_obs(disc.run(model, P)) - c_cur
                    sc = rg.fwd(c)
                    rj = float(np.linalg.norm(Qj.T @ sc)
                               / np.linalg.norm(sc))
                    print(f"      {wsrc:>9}/{rfam:<4}: gain {gain:+7.1%}, "
                          f"w-angle {w_ang:5.1f} deg, read-junk {rj:4.0%}")
                    if best is None or gain > best[0]:
                        best = (gain, wsrc, rfam, w, c)
                    if rnd == 1 and rfam == "id" and (
                            nearest is None or w_ang < nearest[0]):
                        nearest = (w_ang, wsrc, w, gain)
            if rnd == 1:
                p6_data = nearest
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
        return W, C, c_cur, sources, converged, p6_data

    results = {}
    for rname in ("ben", "adv"):
        print(f"[search/{rname}]:")
        results[rname] = search(regimes[rname])

    # ----- P6 diagnostic anchor (labeled; uses T) ------------------------------
    p6_ang, p6_src, p6_w, p6_idgain = results["adv"][5]
    if p6_ang <= 15.0:
        u = regimes["adv"].back(p6_w)
        u = u / np.linalg.norm(u)
        g_diag = c_obs(disc.run(model, np.outer(u, u)))
        print(f"P6 diagnostic: nearest round-1 adversarial write {p6_src} "
              f"({p6_ang:.1f} deg): id-read gain {p6_idgain:+.1%}; "
              f"stream-orthogonal (T-aware, diagnostic-only) gain "
              f"{g_diag:+.1%}\n")
    else:
        g_diag = None
        print(f"P6 diagnostic: NOT TESTED — no round-1 write within 15 deg "
              f"(nearest {p6_ang:.1f} deg)\n")

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
        bp = {t: (float(rows_ge[m][idx].mean())
                  - float(rows_t[m][idx].mean()))
              / (float(rows_ge[m][idx].mean())
                 - float(rows_fe[m][idx].mean()))
              for t, idx in ev.groups}
        return cl, bp

    print("evaluation (exact targets; pooled m=1..3 | per-position at m=3):")
    cl_full, bp = true_closure(np.eye(d))
    print(f"  {'full':>11} (k=64)  "
          + "  ".join(f"m={mm}: {cl_full[mm]:>6.1%}" for mm in ms))
    cl_plane, _ = true_closure(Pc)
    print(f"  {'plane-anchor':>11} (k=2)  "
          + "  ".join(f"m={mm}: {cl_plane[mm]:>6.1%}" for mm in ms)
          + "   (diagnostic ceiling)")
    closures, plane_ang = {}, {}
    for rname, (W, C, c, srcs, conv, _) in results.items():
        if W.shape[1] == 0:
            closures[rname] = {mm: 0.0 for mm in ms}
            print(f"  {rname:>11} (k=0)   closure 0 by definition")
            continue
        P = regimes[rname].pull(oblique_patch(C, W))
        cl, bp = true_closure(P)
        closures[rname] = cl
        Wb = np.column_stack([regimes[rname].back(W[:, j])
                              for j in range(W.shape[1])])
        sub = orthonormal(Wb)
        ang = principal_angles_deg(Qc, sub)
        plane_ang[rname] = ang
        print(f"  {rname:>11} (k={W.shape[1]})  "
              + "  ".join(f"m={mm}: {cl[mm]:>6.1%}" for mm in ms)
              + "   | " + "  ".join(f"t={t}: {bp[t]:.1%}"
                                    for t, _ in ev.groups)
              + "   | plane-in-writes angles "
              + "/".join(f"{a:.1f}" for a in ang) + " deg")

    # ----- verdicts -------------------------------------------------------------
    print("\nverdicts:")
    p1 = cl_plane[m] >= 0.90 * cl_full[m]
    print(f"  P1 anchors + plane-anchor ceiling >= 90% of full: "
          f"{cl_plane[m]:.1%} — {'HOLDS' if p1 else 'FAILS'}")
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
        print(f"  P5 plane containment in writes (k* >= 2, both <= 15 deg): "
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
    if g_diag is None:
        p6 = False
        print("  P6 read-side mechanism: NOT TESTED — no near-plane round-1 "
              "write")
    else:
        p6 = p6_idgain < 0.05 and g_diag >= 0.40
        print(f"  P6 read-side mechanism (id-read < 5%, stream-orthogonal "
              f">= 40%): {p6_idgain:+.1%} / {g_diag:+.1%} — "
              f"{'HOLDS' if p6 else 'FAILS'}")
    print(f"  P7 validity gate: {'HOLDS' if p7 else 'FAILS'}")
    print(f"\nsource attribution: benign {sb}; adversarial {sa}")

    # ----- plot -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        names = ["full", "plane"] + list(results)
        vals = [cl_full[m], cl_plane[m]] + [closures[r][m] for r in results]
        ax.bar(range(len(names)), vals)
        ax.axhline(0.9 * cl_full[m], ls="--", c="r", label="90% of full")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(f"exact closure, pooled m={m}")
        ax.set_title(f"{proc.name}: (write, read) patch search "
                     f"(kappa={args.kappa:g})")
        ax.legend(fontsize=8)
        p = os.path.join(args.outdir, "experiment11.png")
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plotting skipped: {e})")


if __name__ == "__main__":
    main()
