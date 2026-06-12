"""
mstair.py — Experiment 18: the m-staircase — do the diagnostics and
conclusions survive changing the completion horizon?

CONTEXT (see experiments/18-horizon.md). Every claim is indexed by the
standing horizon m = 3; the horizon IS the semantic target gamma_m. The
mm <= 3 staircase points are exact marginalizations of the m = 3 chain
(midstream.marginal / kl_by_horizon); mm = 4 is the only new compute
(81-continuation PairSets with ts PINNED to the m=3 positions — the
default ts formula moves with m). Arm A: fixed patches (clean / id /
spectral / reproduced learned reads / D2) evaluated across mm in
{1,2,3,4} per regime (benign; kappa=100 draw 0; kappa=30 draw 1;
kappa=300 draw 1 reduced) — obs closure, exact closure, obs/exact gap,
rho_mm, val gain. Arm B: CEGAR loops per regime per mm at eps=0.05;
re-learning with the objective at m in {1, 4} on the kappa in {30, 100}
converging writes (4 runs).

Run: python3 mstair.py --outdir out/mess3-L4   (~4-5 h)
`--selftest` adds: marginalization-identity, pinned-ts, and synthetic-
marginal checks.

RESULTS (see experiments/18-horizon.md): P1-P8 all hold — the battery is
horizon-stable (everything flat over mm in {1..4}; P2 worst gap 0.017
over 52 cells; kappa=30 transport +40%/+37% at every mm; benign k*=2
even at mm=1). Run-1 rho-reference defect fixed + re-run (ffb849b).
Decision: Mess3 calibration closes.
"""

import os

import json
import numpy as np
import torch

from adversarial import ZView
from discover import mined_direction, self_checks
from expcommon import (LAYER, PairSet, adversarial_regime, alpha_grid,
                       alpha_powers, basis_covariance, build_transform,
                       jeffreys_rows, kl_rows, load_model,
                       make_torch_objective, oblique_patch, optimize_affine,
                       orthonormal, principal_angles_deg, regression_link,
                       reproduce_anchor, standard_guard, standard_parser,
                       validity_gate, write_pool)
from midstream import kl_by_horizon, marginal
from processes import PROCESSES

REGISTERED = {"kappa": 100.0, "lr": 0.05, "steps": 200, "batch": 64,
              "pairs_disc": 400, "pairs_eval": 600, "basis_seqs": 800,
              "m": 3}
TS_STD, TS_VAL = (8, 16, 24), (12, 20)
MM = (1, 2, 3, 4)
EPS = 0.05
ADV_REGIMES = (("k100", 0, 100.0), ("k30", 1, 30.0), ("k300", 1, 300.0))
RECORDED = {"k100_w2_train": 0.425, "k30_t1": 0.525, "k30_t2": 0.515,
            "k30_v1": 0.400, "k30_v2": 0.374, "k30_r1": 0.046,
            "k30_r2": 0.075, "k100_d2": 0.978, "k30_d2": 0.852,
            "k300_d2": 0.983}


def mnorm(q, V, mm, m_full):
    qm = marginal(q, V, mm, m_full)
    return qm / np.clip(qm.sum(axis=1, keepdims=True), 1e-30, None)


def exp18_self_checks(model, proc, cfg):
    """Registered new checks: marginalization identity, pinned ts,
    synthetic marginal."""
    V = proc.V
    p3 = PairSet(model, proc, cfg, 12, 3, 12345, 30, layer=LAYER, ts=TS_STD)
    p1 = PairSet(model, proc, cfg, 12, 1, 12345, 30, layer=LAYER, ts=TS_STD)
    q3, q1 = p3.run(model, None), p1.run(model, None)
    assert np.abs(mnorm(q3, V, 1, 3) -
                  q1 / q1.sum(axis=1, keepdims=True)).max() < 1e-6, \
        "chain marginalization identity"  # float32 chain: measured 2.9e-8
    assert np.abs(marginal(p3.p_src3, V, 1, 3) - p1.p_src3).max() < 1e-12, \
        "m-gram table marginalization identity"
    p4 = PairSet(model, proc, cfg, 12, 4, 12345, 30, layer=LAYER, ts=TS_STD)
    assert [t for t, _ in p4.groups] == list(TS_STD), "m=4 ts not pinned"
    # pre-run review fix: same-seed pinned-ts PairSets at different m must
    # share pairs and prefix arrays bitwise (the mining input is
    # m-independent; only weights and scoring change with mm)
    assert (p3.a == p4.a).all() and (p3.b == p4.b).all(), "pair identity"
    assert all(torch.equal(p3.pref_src[t], p4.pref_src[t])
               and torch.equal(p3.pref_tgt[t], p4.pref_tgt[t])
               for t, _ in p3.groups), "prefix-array identity across m"
    syn = np.arange(1.0, 1.0 + V ** 2)[None, :]
    assert np.allclose(marginal(syn, V, 1, 2),
                       syn.reshape(1, V, V).sum(axis=2)), "marginal helper"
    print("marginalization-identity, pinned-ts, and synthetic checks "
          "passed")


def main(argv=None):
    args = standard_parser(REGISTERED).parse_args(argv)
    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    if not standard_guard(args, cfg, proc, "Experiment 18", REGISTERED):
        return
    V, m, d = proc.V, args.m, cfg["d_model"]
    model = load_model(args.outdir, cfg, proc)

    gap_opt, p8 = validity_gate(model, proc, cfg, args.seed)
    if not p8 and not args.selftest and not args.force_invalid:
        print("exiting: validity gate failed.")
        return

    disc3 = PairSet(model, proc, cfg, args.pairs_disc, m, args.seed + 111,
                    800, layer=LAYER)
    ev3 = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777,
                  800, layer=LAYER)
    self_checks(model, ev3, LAYER, m, V)
    exp18_self_checks(model, proc, cfg)
    if args.selftest:
        return

    print(f"=== Experiment 18: the m-staircase | {proc.name} | patch "
          f"L{LAYER} | mm in {list(MM)} | regimes benign + "
          f"{[r[0] for r in ADV_REGIMES]} ===\n")

    print("[sets] building staircase pair sets (m=4 is the only new "
          "chain; ts pinned):")
    val3 = PairSet(model, proc, cfg, args.pairs_disc, m, args.seed + 333,
                   800, layer=LAYER, ts=TS_VAL)
    disc4 = PairSet(model, proc, cfg, args.pairs_disc, 4, args.seed + 111,
                    800, layer=LAYER, ts=TS_STD)
    ev4 = PairSet(model, proc, cfg, args.pairs_eval, 4, args.seed + 777,
                  800, layer=LAYER, ts=TS_STD)
    val4 = PairSet(model, proc, cfg, args.pairs_disc, 4, args.seed + 333,
                   800, layer=LAYER, ts=TS_VAL)
    disc1 = PairSet(model, proc, cfg, args.pairs_disc, 1, args.seed + 111,
                    800, layer=LAYER, ts=TS_STD)
    print("  done\n")

    class Refs:
        """Observable references for one pair set at one chain m."""

        def __init__(self, ps, mf):
            self.ps, self.mf = ps, mf
            self.q_src = ps.run(model, None, src_side=True)
            self.q_un = ps.run(model, None)
            self.q_full = ps.run(model, np.eye(d))

        def obs(self, q, mm):
            D0 = float(kl_by_horizon(self.q_un, self.q_src, V,
                                     self.mf)[mm].mean())
            Df = float(kl_by_horizon(self.q_full, self.q_src, V,
                                     self.mf)[mm].mean())
            Dq = float(kl_by_horizon(q, self.q_src, V, self.mf)[mm].mean())
            return (D0 - Dq) / (D0 - Df)

    rd3, rd4 = Refs(disc3, 3), Refs(disc4, 4)
    rv3, rv4 = Refs(val3, 3), Refs(val4, 4)

    class Exact:
        """Exact closures for one eval set at one chain m."""

        def __init__(self, ps, mf):
            self.ps, self.mf = ps, mf
            self.q0 = ps.run(model, None)
            self.floor = {mm: float(kl_by_horizon(self.q0, ps.p_tgt3, V,
                                                  mf)[mm].mean())
                          for mm in range(1, mf + 1)}
            self.gap = {mm: float(kl_by_horizon(self.q0, ps.p_src3, V,
                                                mf)[mm].mean())
                        for mm in range(1, mf + 1)}

        def closure(self, q, mm):
            t = float(kl_by_horizon(q, self.ps.p_src3, V, self.mf)
                      [mm].mean())
            return (self.gap[mm] - t) / (self.gap[mm] - self.floor[mm])

        def rho(self, qC, qX, mm):
            a = mnorm(qC, V, mm, self.mf)
            b = mnorm(qX, V, mm, self.mf)
            u = mnorm(self.q0, V, mm, self.mf)
            return float(jeffreys_rows(a, b).mean()
                         / jeffreys_rows(a, u).mean())

    ex3, ex4 = Exact(ev3, 3), Exact(ev4, 4)

    q_src_d, q_un_d = rd3.q_src, rd3.q_un
    c_obs3 = lambda q: rd3.obs(q, 3)
    Qc = reproduce_anchor(model, disc3, q_src_d, q_un_d, c_obs3, d)
    Sig_x = basis_covariance(model, proc, cfg, args.seed, args.basis_seqs)
    w0w = kl_rows(q_src_d, q_un_d)
    cl_full3 = ex3.closure(ev3.run(model, np.eye(d)), 3)
    print(f"full-patch exact closure (mm=3): {cl_full3:.1%}\n")

    # ----- regimes, writes, patch sets, reproductions ---------------------------
    regimes = {}
    for name, j, kap in ADV_REGIMES:
        T, Tinv, Qj = build_transform(Qc, d, kap, junk_seed=j)
        rg, Sig_z = adversarial_regime(disc3, T, Tinv, Sig_x)
        pool = write_pool(rg, np.zeros((d, 0)), w0w, 1, d, args.seed)
        back_u = lambda w, rg=rg: (lambda u: u / np.linalg.norm(u))(
            rg.back(w))
        angled = sorted((principal_angles_deg(back_u(w)[:, None], Qc)[0],
                         src, w) for src, w in pool)
        near = [(a, s, w) for a, s, w in angled if a <= 15.0][:2]
        tobj3 = make_torch_objective(model, disc3, T, Tinv, q_src_d)
        regression_link(tobj3, model, disc3, rg, q_src_d, angled[0][2])
        regimes[name] = {"T": T, "Tinv": Tinv, "rg": rg, "pool": angled,
                         "near": near, "tobj3": tobj3, "kap": kap, "j": j,
                         "pows": alpha_powers(Sig_z), "back_u": back_u}
        print(f"[{name}] nearest {angled[0][1]} at {angled[0][0]:.1f} deg; "
              f"{len(near)} write(s) <= 15 deg")
    print()

    pb = lambda rg, c, w: rg.pull(oblique_patch(c[:, None], w[:, None]))

    def learn(w, tobj, n_pairs, label):
        return optimize_affine(tobj, n_pairs, d, args.lr, args.steps,
                               args.batch, args.seed, w, w.copy(), True,
                               label, print_every=200)

    print("[reproduce] m=3 learned reads (asserts = halt tripwires):")
    rep = {}
    rk = regimes["k100"]
    w2_100 = rk["near"][1][2]
    rep["k100_w2"] = learn(w2_100, rk["tobj3"], disc3.n, "k100/w2/m3")
    g_k100 = rd3.obs(disc3.run(model, pb(rk["rg"], rep["k100_w2"], w2_100)),
                     3)
    r3 = regimes["k30"]
    w1_30, w2_30 = r3["near"][0][2], r3["near"][1][2]
    rep["k30_a"] = learn(w1_30, r3["tobj3"], disc3.n, "k30/n1/m3")
    rep["k30_b"] = learn(w2_30, r3["tobj3"], disc3.n, "k30/n2/m3")
    g30a = rd3.obs(disc3.run(model, pb(r3["rg"], rep["k30_a"], w1_30)), 3)
    g30b = rd3.obs(disc3.run(model, pb(r3["rg"], rep["k30_b"], w2_30)), 3)
    v30a = rv3.obs(val3.run(model, pb(r3["rg"], rep["k30_a"], w1_30)), 3)
    v30b = rv3.obs(val3.run(model, pb(r3["rg"], rep["k30_b"], w2_30)), 3)
    print(f"  k100/w2 train {g_k100:+.1%} (rec +42.5%); k30 trains "
          f"{g30a:+.1%}/{g30b:+.1%} (rec +52.5%/+51.5%); k30 vals "
          f"{v30a:+.1%}/{v30b:+.1%} (rec +40.0%/+37.4%)")
    p1_rep = (abs(g_k100 - RECORDED["k100_w2_train"]) <= 0.02
              and abs(g30a - RECORDED["k30_t1"]) <= 0.02
              and abs(g30b - RECORDED["k30_t2"]) <= 0.02
              and abs(v30a - RECORDED["k30_v1"]) <= 0.02
              and abs(v30b - RECORDED["k30_v2"]) <= 0.02)
    if not p1_rep and not args.force_invalid:
        print("\nreproduction FAILED — registered halt: P2-P7 NOT TESTED "
              "(--force-invalid continues exploratorily).")
        print(f"P8 validity gate: {'HOLDS' if p8 else 'FAILS'}")
        return
    np.savez(os.path.join(args.outdir, "exp18_reads.npz"), **rep)
    print()

    # ----- patch sets per regime -------------------------------------------------
    def spectral_best(rgm, w):
        g = lambda c, w=w: rd3.obs(disc3.run(model, pb(rgm["rg"], c, w)), 3)
        return alpha_grid(w, rgm["pows"], g)[1][2]

    patches = {"benign": {"Pc": Qc @ Qc.T,
                          "rank1": np.outer(Qc[:, 0], Qc[:, 0])}}
    for name in ("k100", "k30", "k300"):
        rgm = regimes[name]
        rg = rgm["rg"]
        (a1, s1, wA), (a2, s2, wB) = rgm["near"]
        uA, uB = rgm["back_u"](wA), rgm["back_u"](wB)
        U = orthonormal(np.column_stack([uA, uB]))
        ps = {"clean": np.outer(uA, uA), "D2": U @ U.T,
              "id": pb(rg, wA / float(wA @ wA), wA),
              "spectral": pb(rg, spectral_best(rgm, wA), wA)}
        if name == "k100":
            ps["clean2"] = np.outer(uB, uB)
            ps["learned"] = pb(rg, rep["k100_w2"], w2_100)
            ps["id-z(w2)"] = pb(rg, wB / float(wB @ wB), wB)
        if name == "k30":
            ps["clean2"] = np.outer(uB, uB)
            ps["learned"] = pb(rg, rep["k30_a"], w1_30)
            ps["learned2"] = pb(rg, rep["k30_b"], w2_30)
        patches[name] = ps
    # clean reference per (regime, patch) for rho: same-write clean.
    # (Post-run review-of-run fix: the first run mapped k30/learned2 to
    # the FIRST write's clean — violating the registered "same-write
    # clean" and exp-17's recorded reference — which alone produced the
    # run-1 P1-rho and P3 failures. k30/clean2 added; mapping corrected.)
    clean_ref = {"k100": {"learned": "clean2", "id-z(w2)": "clean2",
                          "id": "clean", "spectral": "clean"},
                 "k30": {"learned": "clean", "learned2": "clean2",
                         "id": "clean", "spectral": "clean"},
                 "k300": {"id": "clean", "spectral": "clean"}}

    # ----- arm A: the staircase matrix -------------------------------------------
    learned_val = {("k100", "learned"), ("k30", "learned"),
                   ("k30", "learned2")}
    res = {}
    for name, ps in patches.items():
        qs3d = {k: disc3.run(model, P) for k, P in ps.items()}
        qs3e = {k: ev3.run(model, P) for k, P in ps.items()}
        qs4d = {k: disc4.run(model, P) for k, P in ps.items()}
        qs4e = {k: ev4.run(model, P) for k, P in ps.items()}
        for k in ps:
            for mm in MM:
                o = (rd3.obs(qs3d[k], mm) if mm <= 3
                     else rd4.obs(qs4d[k], 4))
                e = (ex3.closure(qs3e[k], mm) if mm <= 3
                     else ex4.closure(qs4e[k], 4))
                cell = {"obs": o, "exact": e}
                ref = clean_ref.get(name, {}).get(k)
                if ref:
                    cell["rho"] = (ex3.rho(qs3e[ref], qs3e[k], mm)
                                   if mm <= 3
                                   else ex4.rho(qs4e[ref], qs4e[k], 4))
                res[(name, k, mm)] = cell
        for k in ps:
            if (name, k) in learned_val:
                qv3 = val3.run(model, ps[k])
                qv4 = val4.run(model, ps[k])
                for mm in MM:
                    res[(name, k, mm)]["val"] = (rv3.obs(qv3, mm)
                                                 if mm <= 3
                                                 else rv4.obs(qv4, 4))
        print(f"[{name}] staircase (patch: mm=1..4 obs/exact"
              "(/rho)(/val)):")
        for k in ps:
            cells = []
            for mm in MM:
                c = res[(name, k, mm)]
                s = f"{c['obs']:+.0%}/{c['exact']:+.0%}"
                if "rho" in c:
                    s += f"/r{c['rho']:.2f}"
                if "val" in c:
                    s += f"/v{c['val']:+.0%}"
                cells.append(s)
            print(f"  {k}: " + " | ".join(cells))
        print()

    # ----- arm B: CEGAR per regime per mm; re-learning at m in {1, 4} ----------
    def cegar_mm(view, pull, mm, k_max):
        Q = np.zeros((d, 0))
        c_cur = 0.0
        q_cur3, q_cur4 = q_un_d, rd4.q_un
        while Q.shape[1] < k_max:
            w_rows = kl_by_horizon(q_cur3 if mm <= 3 else q_cur4,
                                   rd3.q_src if mm <= 3 else rd4.q_src,
                                   V, 3 if mm <= 3 else 4)[mm]
            v = mined_direction(view, Q, w_rows)
            Q_try = np.hstack([Q, v[:, None]])
            P = pull(Q_try @ Q_try.T)
            if mm <= 3:
                q_try = disc3.run(model, P)
                c_try = rd3.obs(q_try, mm)
            else:
                q_try = disc4.run(model, P)
                c_try = rd4.obs(q_try, 4)
            if c_try - c_cur < EPS:
                break
            Q, c_cur = Q_try, c_try
            if mm <= 3:
                q_cur3 = q_try
            else:
                q_cur4 = q_try
        return Q.shape[1]

    print("[CEGAR] k*(mm) / accept-count(mm) at eps 0.05:")
    # pre-run review fix: mm=4 loops mine from disc4-built views (the
    # registered m=4 discovery loop); disc3/disc4 prefix identity is
    # asserted in the selftest, so this is alignment, not a change of
    # mining input
    benign_k = {mm: cegar_mm(disc3 if mm <= 3 else disc4,
                             lambda P: P, mm, 8) for mm in MM}
    print("  benign k*: " + ", ".join(f"mm={mm}:{benign_k[mm]}"
                                      for mm in MM))
    adv_acc = {}
    for name in ("k100", "k30", "k300"):
        rg = regimes[name]["rg"]
        view4 = ZView(disc4, regimes[name]["T"])
        adv_acc[name] = {mm: cegar_mm(rg.view_raw if mm <= 3 else view4,
                                      rg.pull, mm, 4)
                         for mm in MM}
        print(f"  {name} accept: " + ", ".join(
            f"mm={mm}:{adv_acc[name][mm]}" for mm in MM))
    print()

    print("[re-learn] objectives at m=1 and m=4 (kappa 100 and 30):")
    q_src1 = disc1.run(model, None, src_side=True)
    q_src4 = rd4.q_src
    relearn = {}
    for name, w in (("k100", w2_100), ("k30", w1_30)):
        rgm = regimes[name]
        T, Tinv = rgm["T"], rgm["Tinv"]
        for mf, dset, qs in ((1, disc1, q_src1), (4, disc4, q_src4)):
            tobj = make_torch_objective(model, dset, T, Tinv, qs)
            regression_link(tobj, model, dset, rgm["rg"], qs, w)
            c = learn(w, tobj, dset.n, f"{name}/m{mf}")
            P = pb(rgm["rg"], c, w)
            if mf == 1:
                gt = rd3.obs(disc3.run(model, P), 1)
                gv = rv3.obs(val3.run(model, P), 1)
            else:
                gt = rd4.obs(disc4.run(model, P), 4)
                gv = rv4.obs(val4.run(model, P), 4)
            cross = [rd3.obs(disc3.run(model, P), mm) for mm in (1, 2, 3)]
            cross.append(rd4.obs(disc4.run(model, P), 4))
            relearn[(name, mf)] = {"train": gt, "val": gv}
            print(f"  {name}/m{mf}: own-m train {gt:+.1%}, val {gv:+.1%}; "
                  f"cross-mm obs " + ", ".join(f"{g:+.0%}" for g in cross))
    print()

    # ----- verdicts ---------------------------------------------------------------
    print("verdicts:")
    d2ok = all(abs(res[(n, "D2", 3)]["exact"] - RECORDED[f"{n}_d2"]) <= 0.02
               for n in ("k100", "k30", "k300"))
    rho_ok = (abs(res[("k30", "learned", 3)]["rho"]
                  - RECORDED["k30_r1"]) <= 0.02
              and abs(res[("k30", "learned2", 3)]["rho"]
                      - RECORDED["k30_r2"]) <= 0.02)
    p1 = p1_rep and d2ok and rho_ok
    print(f"  P1 m=3 reproduction + anchors: reads OK, D2s "
          f"{'OK' if d2ok else 'FAIL'}, rhos {'OK' if rho_ok else 'FAIL'} "
          f"— {'HOLDS' if p1 else 'FAILS'}")
    big = [(k, c["obs"], c["exact"]) for k, c in res.items()
           if c["obs"] >= 0.20]
    p2 = all(abs(o - e) <= 0.10 for _, o, e in big)
    worst = max(big, key=lambda x: abs(x[1] - x[2]))
    print(f"  P2 calibration across horizons ({len(big)} cells >= 20%): "
          f"worst gap {abs(worst[1] - worst[2]):.3f} at {worst[0]} — "
          f"{'HOLDS' if p2 else 'FAILS'}")
    p3 = True
    for mm in MM:
        mx = max(res[("k30", k, mm)]["rho"]
                 for k in ("learned", "learned2"))
        mn = min(res[("k100", "id-z(w2)", mm)]["rho"],
                 res[("k30", "id", mm)]["rho"])
        ok = mx <= 0.25 and mn >= 10 * mx
        p3 = p3 and ok
        print(f"  P3 rho separation mm={mm}: transported max {mx:.3f}, "
              f"destructive min {mn:.3f} — {'ok' if ok else 'VIOLATED'}")
    print(f"  P3 overall: {'HOLDS' if p3 else 'FAILS'}")
    p4 = all(res[("k30", k, mm)]["val"] >= 0.20
             for k in ("learned", "learned2") for mm in MM)
    print(f"  P4 transport horizon-stable (k30 val >= 20% at every mm): "
          + "; ".join(f"{k} " + ",".join(
              f"{res[('k30', k, mm)]['val']:+.0%}" for mm in MM)
              for k in ("learned", "learned2"))
          + f" — {'HOLDS' if p4 else 'FAILS'}")
    p5 = all(adv_acc[n][mm] == 0 for n in adv_acc for mm in MM)
    print(f"  P5 proposal death at every mm: "
          f"{'HOLDS' if p5 else 'FAILS'}")
    p6 = all(benign_k[mm] == 2 for mm in (2, 3, 4))
    print(f"  P6 benign k* = 2 for mm in {{2,3,4}} (mm=1 descriptive: "
          f"{benign_k[1]}): {'HOLDS' if p6 else 'FAILS'}")
    p7 = all(relearn[("k30", mf)]["train"] >= 0.20
             and relearn[("k30", mf)]["val"] >= 0.20 for mf in (1, 4))
    print(f"  P7 k30 re-learned reads transport at their own m: "
          + "; ".join(f"m={mf}: train "
                      f"{relearn[('k30', mf)]['train']:+.1%}, val "
                      f"{relearn[('k30', mf)]['val']:+.1%}"
                      for mf in (1, 4))
          + f" — {'HOLDS' if p7 else 'FAILS'}")
    print(f"  P8 validity gate: {'HOLDS' if p8 else 'FAILS'}")


if __name__ == "__main__":
    main()
