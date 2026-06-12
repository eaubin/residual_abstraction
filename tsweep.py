"""
tsweep.py — Experiment 17: T-robustness and the eps_gain staircase.

CONTEXT (see experiments/17-generality.md). Every adversarial conclusion
since exp 8 rides one junk-plane draw and one kappa; every accept/reject
claim rides eps_gain = 0.05. Units = (junk_seed, kappa): primary draws
0-4 at kappa=100 (draw 0 = the historical T, asserted against recorded
exps 14-16 values — halt enforced); secondary draw 1 at kappa in
{30, 300} (descriptive). Per-unit battery: adversarial CEGAR
accept-count staircase over eps in {0.01, 0.02, 0.05, 0.10}; write-pool
nearest angle; id/clean/spectral read contrasts; gradient runs on the
two nearest writes (phenotype diverged/converged/intermediate; val gain,
rho, decomposition for converging reads); clean D2 composition. Benign
eps staircase once. Shared pair sets across units isolate T as the only
varying factor.

Run: python3 tsweep.py --outdir out/mess3-L4   (~4-5 h)
`--selftest` adds: junk_seed-equivalence, distinct-T, and phenotype-
boundary checks.

RESULTS (see experiments/17-generality.md): P1-P9 all hold. T-generic at
kappa=100 (5 draws, identical contrasts, accept 0 at every eps);
thresholds never load-bearing; P7 18/18 cells <= 1.5 pts across seven
transforms. Kappa arm: the gradient pathology is KAPPA-GRADED — at
kappa=30 learned reads transport (val +40%/+37%, rho ~0.05, zero plane
mass); at kappa=300 everything diverges. Exps 13-16 conclusions stand at
kappa=100 (draw-generic), with the kappa=300 probe supporting the
high-kappa pole.
"""

import os

import json
import numpy as np

from discover import mined_direction, self_checks
from expcommon import (LAYER, PairSet, adversarial_regime, alpha_grid,
                       alpha_powers, basis_covariance, build_transform,
                       decompose, jeffreys_rows, kl_rows, load_model,
                       make_torch_objective, oblique_patch, observable_refs,
                       optimize_affine, orthonormal, principal_angles_deg,
                       regression_link, reproduce_anchor, standard_guard,
                       standard_parser, validity_gate, write_pool)
from midstream import kl_by_horizon
from processes import PROCESSES

REGISTERED = {"kappa": 100.0, "lr": 0.05, "steps": 200, "batch": 64,
              "pairs_disc": 400, "pairs_eval": 600, "basis_seqs": 800,
              "m": 3}
DRAWS = (0, 1, 2, 3, 4)                  # primary units at kappa = 100
KAPPA_ARM = ((1, 30.0), (1, 300.0))      # secondary, descriptive
EPS_GRID = (0.01, 0.02, 0.05, 0.10)
K_MAX_ADV = 4
SEED_VAL = 333                           # exp-16's selection positions
TS_VAL = (12, 20)
RECORDED = {"a1": 1.1, "a2": 3.3, "id_w1": 0.010, "clean_w1": 0.513,
            "grad_w1": -5.482, "grad_w2": 0.425, "d2": 0.978}


def phenotype(g):
    return ("diverged" if g <= -1.00 else
            "converged" if g >= 0.20 else "intermediate")


def exp17_self_checks(d=16):
    rng = np.random.default_rng(3)
    Qc_s = orthonormal(rng.standard_normal((d, 2)))
    Ta, _, _ = build_transform(Qc_s, d, 100.0)
    Tb, _, _ = build_transform(Qc_s, d, 100.0, junk_seed=0)
    assert np.array_equal(Ta, Tb), "junk_seed=0 != historical default"
    Tc, _, _ = build_transform(Qc_s, d, 100.0, junk_seed=1)
    assert not np.allclose(Ta, Tc), "distinct junk seeds gave equal T"
    assert [phenotype(g) for g in (-1.0, -0.99, 0.19, 0.20)] == \
        ["diverged", "intermediate", "intermediate", "converged"], \
        "phenotype boundaries"
    print("junk-seed, distinct-T, and phenotype-boundary checks passed")


def main(argv=None):
    args = standard_parser(REGISTERED).parse_args(argv)
    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    if not standard_guard(args, cfg, proc, "Experiment 17", REGISTERED):
        return
    V, m, d = proc.V, args.m, cfg["d_model"]
    model = load_model(args.outdir, cfg, proc)

    gap_opt, p9 = validity_gate(model, proc, cfg, args.seed)
    if not p9 and not args.selftest and not args.force_invalid:
        print("exiting: validity gate failed.")
        return

    disc = PairSet(model, proc, cfg, args.pairs_disc, m, args.seed + 111,
                   800, layer=LAYER)
    ev = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777,
                 800, layer=LAYER)
    self_checks(model, ev, LAYER, m, V)
    exp17_self_checks()
    if args.selftest:
        return

    print(f"=== Experiment 17: T-robustness + eps staircase | {proc.name} "
          f"| patch L{LAYER} | draws {list(DRAWS)} at kappa 100 + "
          f"{list(KAPPA_ARM)} ===\n")

    # ----- shared scaffolding (T-independent) -----------------------------------
    q_src_d, q_un_d, c_obs = observable_refs(model, disc, d)
    Qc = reproduce_anchor(model, disc, q_src_d, q_un_d, c_obs, d)
    Sig_x = basis_covariance(model, proc, cfg, args.seed, args.basis_seqs)
    w0w = kl_rows(q_src_d, q_un_d)
    disc_val = PairSet(model, proc, cfg, args.pairs_disc, m,
                       args.seed + SEED_VAL, 800, layer=LAYER, ts=TS_VAL)
    _, _, c_obs_val = observable_refs(model, disc_val, d)
    q0_ev = ev.run(model, None)
    floor_ev = float(kl_by_horizon(q0_ev, ev.p_tgt3, V, m)[m].mean())
    gap_ev = float(kl_by_horizon(q0_ev, ev.p_src3, V, m)[m].mean())

    def cl(P):
        rows = kl_by_horizon(ev.run(model, P), ev.p_src3, V, m)
        return (gap_ev - float(rows[m].mean())) / (gap_ev - floor_ev)

    cl_full = cl(np.eye(d))
    print(f"full-patch exact closure (shared): {cl_full:.1%}\n")

    def cegar(view, pull, eps, k_max):
        """The exp-6 acceptance rule (accept iff marginal gain >= eps) on
        an arbitrary coordinate view with pullback scoring."""
        Q = np.zeros((d, 0))
        q_cur, c_cur, fg = q_un_d, 0.0, None
        while Q.shape[1] < k_max:
            v = mined_direction(view, Q, kl_rows(q_src_d, q_cur))
            Q_try = np.hstack([Q, v[:, None]])
            q_try = disc.run(model, pull(Q_try @ Q_try.T))
            if fg is None:
                fg = c_obs(q_try)
            if c_obs(q_try) - c_cur < eps:
                break
            Q, q_cur, c_cur = Q_try, q_try, c_obs(q_try)
        return Q.shape[1], fg

    # benign eps staircase (T-independent; once)
    print("[benign] eps staircase (frozen exp-6 loop):")
    benign_k = {}
    for eps in EPS_GRID:
        benign_k[eps], _ = cegar(disc, lambda P: P, eps, 8)
        print(f"  eps {eps:.2f}: k* = {benign_k[eps]}")
    print()

    # ----- the per-unit battery ---------------------------------------------------
    units = [(j, 100.0) for j in DRAWS] + [list(x) for x in KAPPA_ARM]
    units = [(int(j), float(k)) for j, k in units]
    rows = []
    p7_pairs = []

    for j, kap in units:
        label = f"T(j={j}, k={kap:g})"
        print(f"----- unit {label} -----")
        T, Tinv, Qj = build_transform(Qc, d, kap, junk_seed=j)
        rg, Sig_z = adversarial_regime(disc, T, Tinv, Sig_x)
        pows_z = alpha_powers(Sig_z)
        tobj = make_torch_objective(model, disc, T, Tinv, q_src_d)

        accept = {}
        for eps in EPS_GRID:
            accept[eps], fg = cegar(rg.view_raw, rg.pull, eps, K_MAX_ADV)
        print(f"  adversarial accept-count staircase: "
              + ", ".join(f"eps {e:.2f}: {accept[e]}" for e in EPS_GRID)
              + f"; first proposal gain {fg:+.1%}")

        pool = write_pool(rg, np.zeros((d, 0)), w0w, 1, d, args.seed)
        back_u = lambda w: (lambda u: u / np.linalg.norm(u))(rg.back(w))
        angled = sorted((principal_angles_deg(back_u(w)[:, None], Qc)[0],
                         src, w) for src, w in pool)
        near = [(a, s, w) for a, s, w in angled if a <= 15.0][:2]
        a_near = angled[0][0]
        print(f"  pool: nearest {angled[0][1]} at {a_near:.1f} deg; "
              f"{len(near)} write(s) <= 15 deg")

        regression_link(tobj, model, disc, rg, q_src_d, angled[0][2])
        gain = lambda c, w: c_obs(disc.run(model, rg.pull(
            oblique_patch(c[:, None], w[:, None]))))
        w_n = angled[0][2]
        c_id = w_n / float(w_n @ w_n)
        g_id = gain(c_id, w_n)
        u_n = back_u(w_n)
        g_clean = c_obs(disc.run(model, np.outer(u_n, u_n)))
        sb = None
        for _, _, w in (near if near else angled[:1]):
            _, best = alpha_grid(w, pows_z, lambda c, w=w: gain(c, w))
            if sb is None or best[0] > sb[0]:
                sb = (best[0], best[2], w)
        sbest = sb[0]
        print(f"  contrasts (nearest write): id {g_id:+.1%}, clean "
              f"{g_clean:+.1%}, best spectral {sbest:+.1%}")
        # P7 coverage: every battery patch (pre-run review fix — id,
        # spectral, D2 included alongside clean and converged reads)
        p7_pairs.append((f"{label} clean", g_clean,
                         lambda u=u_n: cl(np.outer(u, u))))
        P_id = rg.pull(oblique_patch(c_id[:, None], w_n[:, None]))
        p7_pairs.append((f"{label} id", g_id, lambda P=P_id: cl(P)))
        P_sp = rg.pull(oblique_patch(sb[1][:, None], sb[2][:, None]))
        p7_pairs.append((f"{label} spectral", sbest,
                         lambda P=P_sp: cl(P)))

        phen, conv = [], []
        for a_, s_, w_ in near:
            lab = f"{label} grad@{a_:.1f}"
            c_fin = optimize_affine(tobj, disc.n, d, args.lr, args.steps,
                                    args.batch, args.seed, w_, w_.copy(),
                                    True, lab, print_every=200)
            g = gain(c_fin, w_)
            ph = phenotype(g)
            phen.append((a_, g, ph))
            print(f"    {lab}: final train {g:+.1%} ({ph})")
            if ph == "converged":
                gv = c_obs_val(disc_val.run(model, rg.pull(
                    oblique_patch(c_fin[:, None], w_[:, None]))))
                P_x = rg.pull(oblique_patch(c_fin[:, None], w_[:, None]))
                qX = ev.run(model, P_x)
                u_ = back_u(w_)
                qC = ev.run(model, np.outer(u_, u_))
                rho = float(jeffreys_rows(qC, qX).mean()
                            / jeffreys_rows(qC, q0_ev).mean())
                fp, fj, fn = decompose(T @ c_fin, Qc, Qj)
                conv.append({"angle": a_, "g": g, "val": gv, "rho": rho})
                print(f"      converged read: val {gv:+.1%}, rho "
                      f"{rho:.3f}, plane/junk/neutral "
                      f"{fp:.0%}/{fj:.0%}/{fn:.0%}")
                p7_pairs.append((f"{lab}", g, lambda P=P_x: cl(P)))

        d2 = None
        if len(near) >= 2:
            U = orthonormal(np.column_stack([back_u(near[0][2]),
                                             back_u(near[1][2])]))
            P_d2 = U @ U.T
            d2 = cl(P_d2)
            obs_d2 = c_obs(disc.run(model, P_d2))
            print(f"  clean D2: exact closure {d2:.1%}, observable "
                  f"{obs_d2:+.1%}")
            p7_pairs.append((f"{label} D2", obs_d2, lambda v=d2: v))
        rows.append({"j": j, "kap": kap, "a": a_near, "accept": accept,
                     "id": g_id, "clean": g_clean, "spec": sbest,
                     "phen": phen, "conv": conv, "d2": d2})

        if j == 0 and kap == 100.0:
            ok = (abs(a_near - RECORDED["a1"]) <= 0.2
                  and abs(near[1][0] - RECORDED["a2"]) <= 0.2
                  and abs(g_id - RECORDED["id_w1"]) <= 0.02
                  and abs(g_clean - RECORDED["clean_w1"]) <= 0.02
                  and abs(phen[0][1] - RECORDED["grad_w1"]) <= 0.02
                  and abs(phen[1][1] - RECORDED["grad_w2"]) <= 0.02
                  and d2 is not None and abs(d2 - RECORDED["d2"]) <= 0.02)
            print(f"  draw-0 reproduction vs recorded: "
                  f"{'OK' if ok else 'FAILED'}")
            if not ok and not args.force_invalid:
                print("\nregistered halt: draw-0 battery does not "
                      "reproduce recorded exps 14-16 values "
                      "(determinism breach). P2-P8 NOT TESTED; "
                      "--force-invalid continues exploratorily.")
                print(f"  P9 validity gate: {'HOLDS' if p9 else 'FAILS'}")
                return
            p1_rep = ok
        print()

    # ----- summary table -----------------------------------------------------------
    print("unit summary (j, kappa | nearest deg | accept@eps "
          + "/".join(f"{e:.2f}" for e in EPS_GRID)
          + " | id / clean / spec | phenotypes | conv val | D2):")
    for r in rows:
        ph = ",".join(p[2][0] for p in r["phen"]) or "-"
        cv = ",".join(f"{c['val']:+.0%}/rho{c['rho']:.2f}"
                      for c in r["conv"]) or "-"
        d2s = f"{r['d2']:.1%}" if r["d2"] is not None else "-"
        print(f"  j={r['j']} k={r['kap']:g} | {r['a']:.1f} | "
              + "/".join(str(r["accept"][e]) for e in EPS_GRID)
              + f" | {r['id']:+.1%} / {r['clean']:+.1%} / "
              f"{r['spec']:+.1%} | {ph} | {cv} | {d2s}")
    print()

    prim = [r for r in rows if r["kap"] == 100.0]
    sec = [r for r in rows if r["kap"] != 100.0]

    # ----- verdicts ------------------------------------------------------------------
    print("verdicts:")
    print(f"  P1 draw-0 reproduction + anchors: "
          f"{'HOLDS' if p1_rep else 'FAILS'}")
    p2 = all(r["accept"][0.05] == 0 for r in prim)
    print(f"  P2 proposal death T-generic (accept@0.05 == 0, all 5 "
          f"draws): {'HOLDS' if p2 else 'FAILS'}")
    p3 = all(r["a"] <= 5.0 for r in prim)
    print(f"  P3 write reachability (nearest <= 5 deg, all draws): "
          + ", ".join(f"{r['a']:.1f}" for r in prim)
          + f" — {'HOLDS' if p3 else 'FAILS'}")
    p4 = all(r["clean"] >= 0.40 and r["id"] <= 0.10 and r["spec"] <= 0.10
             and r["d2"] is not None and r["d2"] >= 0.90 * cl_full
             for r in prim)
    print(f"  P4 core contrasts T-generic (clean >= 40%, id <= 10%, "
          f"spectral <= 10%, D2 >= 90% of full, all draws): "
          f"{'HOLDS' if p4 else 'FAILS'}")
    div_draws = sum(any(p[2] == "diverged" for p in r["phen"])
                    for r in prim)
    p5a = div_draws >= 4
    conv_all = [c for r in prim for c in r["conv"]]
    p5b = len(conv_all) >= 1
    print(f"  P5a divergent phenotype in >= 4/5 draws ({div_draws}/5): "
          f"{'HOLDS' if p5a else 'FAILS'}")
    print(f"  P5b converging read exists across draws "
          f"({len(conv_all)} found): {'HOLDS' if p5b else 'FAILS'}")
    if conv_all:
        p6 = all(c["val"] < 0.20 for c in conv_all)
        verdict6 = ("HOLDS" if p6 else
                    "FAILS (a transportable read appeared under a new T)")
        print(f"  P6 entanglement T-generic (every converging read val "
              f"< 20%): " + ", ".join(f"{c['val']:+.1%}" for c in conv_all)
              + f" — {verdict6}")
    else:
        print("  P6: NOT TESTED — no converging read")
    big = [(n, o, f) for n, o, f in p7_pairs if o >= 0.20]
    if big:
        checked = [(n, o, f()) for n, o, f in big]
        p7 = all(abs(o - e) <= 0.10 for _, o, e in checked)
        print(f"  P7 observable/exact on {len(checked)} patch(es) >= 20%: "
              + "; ".join(f"{n}: {o:+.1%} vs {e:+.1%}"
                          for n, o, e in checked)
              + f" — {'HOLDS' if p7 else 'FAILS (objective hacking)'}")
    else:
        print("  P7: NOT TESTED — nothing at 20%")
    p8a = all(benign_k[e] == 2 for e in EPS_GRID)
    p8b = all(r["accept"][e] == 0 for r in prim for e in EPS_GRID
              if e >= 0.02)
    print(f"  P8a benign k*(eps) == 2 across grid: "
          + ", ".join(f"{e:.2f}:{benign_k[e]}" for e in EPS_GRID)
          + f" — {'HOLDS' if p8a else 'FAILS'}")
    print(f"  P8b adversarial accept == 0 for eps >= 0.02, all draws: "
          f"{'HOLDS' if p8b else 'FAILS'}")
    sec_ok = all(r["clean"] >= 0.40 and r["id"] <= 0.10
                 and r["spec"] <= 0.10 and r["d2"] is not None
                 and r["d2"] >= 0.90 * cl_full for r in sec)
    print(f"  secondary (kappa arm, descriptive): full P4 contrasts "
          f"(incl. D2) hold at kappa 30/300 on draw 1 — {sec_ok}")
    print(f"  P9 validity gate: {'HOLDS' if p9 else 'FAILS'}")


if __name__ == "__main__":
    main()
