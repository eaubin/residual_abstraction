"""
transport.py — Experiment 15: statistical control vs transported state.

CONTEXT (see experiments/15-transport.md and FORMALISM.md §8). Exp 14 left
w2's learned reads transferring (+32.2%/+42.5%, observable = exact) with
zero plane mass and pooled EPR ~ 0.008. Two live readings: statistical
control (distribution-specific correlation exploitation) vs transported
state (behavioral equivalence to the clean patch; the geometry
diagnostics are the wrong instruments). This experiment is the registered
adjudication, and the registered test of FORMALISM 8's equivalence-class
claim.

Three tests. 1: per-pair equivalence — rho(X) = mean Jeffreys(clean, X) /
mean Jeffreys(clean, unpatched); bands equivalent <= 0.25 / distinct
>= 0.5. 2: EPR localization — per-(t-group, position) cells; position-t
cells adjudicate whether exp-14's pooled refutation was an aggregation
artifact. 3: distribution shift — Shift-A pair positions {12,20}; Shift-B
fixed initial hidden state 0; relative retention R = learned retention /
clean retention, with registered competence and clean-gain guards.
Targets are ALWAYS stationary-frame: shifts move the distribution over
prefixes, never the per-prefix target. Reads reproduced in-run (asserted
against exp-14 recorded values) and persisted to
out/<process>/exp15_reads.npz.

First script on the shared scaffolding (expcommon.py — extracted verbatim
from the readaffine/readopt lineage; concluded scripts stay frozen).

Run: python3 transport.py --outdir out/mess3-L4   (~80-110 min)
`--selftest` runs the standard four plus the new ts/init_state/Jeffreys/
EPR checks and exits.

RESULTS (see experiments/15-transport.md): P1/P4/P6/P7 hold; P2 fails
(split 0.20/0.44); P3 -> aggregation artifact (position-t EPR 0.85-0.93);
P5 -> statistical-control branch: learned reads are POSITION-ENTANGLED —
they compute the clean functional at trained positions and invert at
unseen ones (R -0.77/-0.41) while clean improves; rho separator validated.
"""

import os

import json
import numpy as np
import torch
import torch.nn.functional as F

from discover import self_checks
from expcommon import (LAYER, PairSet, Regime, adversarial_regime,
                       alpha_grid, alpha_powers, basis_covariance,
                       build_transform, epr, jeffreys_rows, kl_rows,
                       load_model, make_torch_objective, oblique_patch,
                       observable_refs, optimize_affine, orthonormal,
                       regression_link, reproduce_anchor, reproduce_writes,
                       standard_guard, standard_parser, validity_gate)
from midstream import kl_by_horizon
from processes import PROCESSES

REGISTERED = {"kappa": 100.0, "lr": 0.05, "steps": 200, "batch": 64,
              "pairs_disc": 400, "pairs_eval": 600, "basis_seqs": 800,
              "m": 3}
TS_SHIFT = (12, 20)        # Shift-A pair positions (registered)
INIT_SHIFT = 0             # Shift-B fixed initial hidden state (registered)
RHO_EQUIV, RHO_DISTINCT = 0.25, 0.50
R_ROBUST, R_FRAGILE = 0.70, 0.30


def exp15_self_checks(model, proc, cfg):
    """Registered new checks: ts/init_state kwargs, J plumbing, EPR cell."""
    L, burn = cfg["seq_len"], cfg["burn_in"]
    expect = list(np.unique(np.linspace(burn + 4, L - 1 - 3 - 4, 3)
                            .astype(int)))
    small = PairSet(model, proc, cfg, 12, 3, 12345, 30, layer=LAYER)
    assert [t for t, _ in small.groups] == expect, "default ts changed"
    small2 = PairSet(model, proc, cfg, 12, 3, 12345, 30, layer=LAYER,
                     ts=TS_SHIFT)
    assert [t for t, _ in small2.groups] == sorted(TS_SHIFT), "ts kwarg"
    rng = np.random.default_rng(7)
    Xs = proc.sample(2000, 2, rng, init_state=INIT_SHIFT)
    e0 = np.zeros(proc.S)
    e0[INIT_SHIFT] = 1.0
    p_exact = np.array([(e0 @ proc.T[s]).sum() for s in range(proc.V)])
    p_emp = np.bincount(Xs[:, 0], minlength=proc.V) / len(Xs)
    assert np.abs(p_emp - p_exact).max() < 0.05, "init_state emission law"
    q = rng.dirichlet(np.ones(27), size=50)
    assert float(np.abs(jeffreys_rows(q, q)).max()) <= 1e-12, "J(q,q) != 0"
    q0 = rng.dirichlet(np.ones(27), size=50)
    rho_cc = float(jeffreys_rows(q, q).mean() / jeffreys_rows(q, q0).mean())
    assert rho_cc == 0.0, "rho(C) != 0"
    D = rng.standard_normal((200, 8))
    u = rng.standard_normal(8)
    assert abs(epr(D, u, u) - 1.0) <= 1e-12, "EPR cell plumbing"
    print("ts/init_state, Jeffreys, and EPR-cell checks passed")


def main(argv=None):
    args = standard_parser(REGISTERED).parse_args(argv)
    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    if not standard_guard(args, cfg, proc, "Experiment 15", REGISTERED):
        return
    L, V, m, d = cfg["seq_len"], proc.V, args.m, cfg["d_model"]
    model = load_model(args.outdir, cfg, proc)

    gap_opt, p7 = validity_gate(model, proc, cfg, args.seed)
    if not p7 and not args.selftest and not args.force_invalid:
        print("exiting: validity gate failed.")
        return

    disc = PairSet(model, proc, cfg, args.pairs_disc, m, args.seed + 111, 800,
                   layer=LAYER)
    ev = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777, 800,
                 layer=LAYER)
    self_checks(model, ev, LAYER, m, V)
    exp15_self_checks(model, proc, cfg)
    if args.selftest:
        return

    print(f"=== Experiment 15: statistical control vs transported state | "
          f"{proc.name} | patch L{LAYER} | kappa = {args.kappa:g} ===\n")

    # ----- standard scaffolding (expcommon; asserts are the tripwires) ---------
    q_src_d, q_un_d, c_obs = observable_refs(model, disc, d)
    Qc = reproduce_anchor(model, disc, q_src_d, q_un_d, c_obs, d)
    T, Tinv, Qj = build_transform(Qc, d, args.kappa)
    Sig_x = basis_covariance(model, proc, cfg, args.seed, args.basis_seqs)
    rg_adv, Sig_z = adversarial_regime(disc, T, Tinv, Sig_x)
    pows_z = alpha_powers(Sig_z)
    (a1, s1, w1), (a2, s2, w2) = reproduce_writes(rg_adv, q_src_d, q_un_d,
                                                  Qc, d, args.seed)
    torch_objective = make_torch_objective(model, disc, T, Tinv, q_src_d)
    regression_link(torch_objective, model, disc, rg_adv, q_src_d, w1)

    def gain_obs(c, w):
        return c_obs(disc.run(model, rg_adv.pull(
            oblique_patch(c[:, None], w[:, None]))))

    def learn(w, init_c, label):
        return optimize_affine(torch_objective, disc.n, d, args.lr,
                               args.steps, args.batch, args.seed,
                               w, init_c, True, label)

    # ----- reproduce the reads (asserted against exp-14 recorded values) -------
    print("[reads] reproducing the exp-14 reads (asserts = tripwires):")
    grids = {}
    for tag, w in (("w1", w1), ("w2", w2)):
        _, grids[tag] = alpha_grid(w, pows_z, lambda c, w=w: gain_obs(c, w))
    c_w2_ba = learn(w2, grids["w2"][2], "aff/w2/best-a")
    g_w2_ba = gain_obs(c_w2_ba, w2)
    c_w2_id = learn(w2, w2.copy(), "aff/w2/id")
    g_w2_id = gain_obs(c_w2_id, w2)
    c_w1_id = learn(w1, w1.copy(), "aff/w1/id")
    g_w1_id = gain_obs(c_w1_id, w1)
    print(f"  reproduced gains: aff/w2/best-a {g_w2_ba:+.1%} (recorded "
          f"+32.2%); aff/w2/id {g_w2_id:+.1%} (recorded +42.5%); aff/w1/id "
          f"{g_w1_id:+.1%} (recorded -548.2%)")
    p1_rep = (abs(g_w2_ba - 0.322) <= 0.02 and abs(g_w2_id - 0.425) <= 0.02
              and g_w1_id <= -1.00)
    np.savez(os.path.join(args.outdir, "exp15_reads.npz"),
             w1=w1, w2=w2, c_w2_besta=c_w2_ba, c_w2_id=c_w2_id,
             c_w1_id=c_w1_id, Qc=Qc, Qj=Qj)
    print(f"  reads persisted to {args.outdir}/exp15_reads.npz\n")
    if not p1_rep and not args.force_invalid:
        # Registered halt (P1, pre-run review fix): a reproduction failure
        # is a determinism breach — downstream verdicts would describe
        # different objects than registered.
        print("read reproduction FAILED — registered halt.\n\nverdicts:")
        print(f"  P1 anchors + reproduction: FAILS (determinism breach: "
              f"reads {g_w2_ba:+.1%}/{g_w2_id:+.1%}/{g_w1_id:+.1%}; "
              "anchors not evaluated)")
        print("  P2-P6: NOT TESTED — reproduction halt "
              "(--force-invalid continues exploratorily)")
        print(f"  P7 validity gate: {'HOLDS' if p7 else 'FAILS'}")
        return

    # ----- patches under test ---------------------------------------------------
    u1 = rg_adv.back(w1); u1 = u1 / np.linalg.norm(u1)
    u2 = rg_adv.back(w2); u2 = u2 / np.linalg.norm(u2)
    U = orthonormal(np.column_stack([u1, u2]))
    P_d2 = U @ U.T
    pb = lambda c, w: rg_adv.pull(oblique_patch(c[:, None], w[:, None]))
    patches = {
        "w2/clean": ("w2", np.outer(u2, u2)),
        "w2/aff/best-a": ("w2", pb(c_w2_ba, w2)),
        "w2/aff/id": ("w2", pb(c_w2_id, w2)),
        "w2/spectral": ("w2", pb(grids["w2"][2], w2)),
        "w2/id-z": ("w2", pb(w2 / float(w2 @ w2), w2)),
        "w1/clean": ("w1", np.outer(u1, u1)),
        "w1/aff/id": ("w1", pb(c_w1_id, w1)),
    }
    accepted = ["w2/aff/best-a", "w2/aff/id"]
    destructive = ["w2/id-z", "w1/aff/id"]
    clean_of = {"w1": "w1/clean", "w2": "w2/clean"}

    # ----- pair sets: base + the two registered shifts --------------------------
    print("[sets] building shifted pair sets (stationary-frame targets):")
    ev_A = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777, 800,
                   layer=LAYER, ts=TS_SHIFT)
    ev_B = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 779, 800,
                   layer=LAYER, init_state=INIT_SHIFT)
    sets = {"base": ev, "shift-A": ev_A, "shift-B": ev_B}
    print(f"  base ts {[t for t, _ in ev.groups]}; shift-A ts "
          f"{[t for t, _ in ev_A.groups]}; shift-B init_state {INIT_SHIFT}\n")

    # shift-B competence guard: model NLL vs the exact predictor's NLL on the
    # SAME shifted sample, both in the stationary frame.
    XB = ev_B.Xe
    with torch.no_grad():
        tot, cnt = 0.0, 0
        for i in range(0, len(XB), 256):
            logits = model(torch.from_numpy(XB[i:i + 256]))
            tgt = torch.from_numpy(XB[i:i + 256, 1:]).reshape(-1)
            tot += F.cross_entropy(logits[:, :-1].reshape(-1, V), tgt,
                                   reduction="sum").item()
            cnt += tgt.numel()
    nll_model_B = tot / cnt
    pe = np.array([[(ev_B.B[i, t] @ proc.T[XB[i, t + 1]]).sum()
                    for t in range(L - 1)] for i in range(len(XB))])
    nll_exact_B = float(-np.log(pe).mean())
    gap_B = nll_model_B - nll_exact_B
    guard_B_comp = gap_B <= 0.01
    print(f"shift-B competence guard: model {nll_model_B:.4f} vs exact "
          f"{nll_exact_B:.4f} nats (gap {gap_B:+.4f}) — "
          f"{'PASS' if guard_B_comp else 'FAIL'}\n")

    # ----- closures + equivalence per set ---------------------------------------
    def closures_and_equiv(pset, label):
        q0 = pset.run(model, None)
        rows_f = kl_by_horizon(q0, pset.p_tgt3, V, m)
        rows_g = kl_by_horizon(q0, pset.p_src3, V, m)
        floor = float(rows_f[m].mean())
        gap = float(rows_g[m].mean())
        qs = {k: pset.run(model, P) for k, (_, P) in patches.items()}
        gains = {k: (gap - float(kl_by_horizon(q, pset.p_src3, V, m)[m]
                                 .mean())) / (gap - floor)
                 for k, q in qs.items()}
        rho, fworse = {}, {}
        for k, q in qs.items():
            ref = clean_of[patches[k][0]]
            if k == ref:
                continue
            Jx = jeffreys_rows(qs[ref], q)
            Jun = jeffreys_rows(qs[ref], q0)
            rho[k] = float(Jx.mean() / Jun.mean())
            fworse[k] = float((Jx > Jun).mean())
        print(f"[{label}] exact closure gains: "
              + ", ".join(f"{k} {gains[k]:+.1%}" for k in patches))
        print(f"[{label}] equivalence rho (vs same-write clean; frac_worse): "
              + ", ".join(f"{k} {rho[k]:.3f} ({fworse[k]:.0%})"
                          for k in rho))
        return gains, rho, fworse

    res = {lbl: closures_and_equiv(ps, lbl) for lbl, ps in sets.items()}
    g_base, rho_base, _ = res["base"]

    # anchors on base (P1)
    q0 = ev.run(model, None)
    rows_f = kl_by_horizon(q0, ev.p_tgt3, V, m)
    rows_g = kl_by_horizon(q0, ev.p_src3, V, m)
    floor, gapv = float(rows_f[m].mean()), float(rows_g[m].mean())
    cl = lambda P: (gapv - float(kl_by_horizon(ev.run(model, P), ev.p_src3,
                                               V, m)[m].mean())) \
        / (gapv - floor)
    cl_full, cl_d2 = cl(np.eye(d)), cl(P_d2)
    g_d1 = g_base["w1/clean"]
    print(f"\nanchors: full {cl_full:.1%}; D2 {cl_d2:.1%}; D1 {g_d1:+.1%}\n")

    # ----- EPR localization (base eval set) -------------------------------------
    epr_reads = {
        "w2/id": (T @ w2, u2), "w2/init": (T @ grids["w2"][2], u2),
        "w2/aff/best-a": (T @ c_w2_ba, u2), "w2/aff/id": (T @ c_w2_id, u2),
        "w2/clean": (u2, u2), "w1/aff/id": (T @ c_w1_id, u1),
    }
    print("[EPR] per-cell localization (t-group x position; base eval):")
    cells = {k: {} for k in epr_reads}
    for t, idx in ev.groups:
        D_t = (ev.pref_src[t] - ev.pref_tgt[t]).numpy().astype(np.float64)
        for k, (r, ucl) in epr_reads.items():
            for p in range(t + 1):
                cells[k][(t, p)] = epr(D_t[:, p, :], r, ucl)
    for k in epr_reads:
        if k == "w2/clean":
            assert all(abs(v - 1.0) <= 1e-9 for v in cells[k].values()), \
                "clean EPR cell != 1"
            print(f"  {k}: all cells = 1.000 (plumbing)")
            continue
        post = {t: cells[k][(t, t)] for t, _ in ev.groups}
        mx = max(cells[k].items(), key=lambda kv: kv[1])
        print(f"  {k}: position-t cells "
              + ", ".join(f"t={t}:{v:.3f}" for t, v in post.items())
              + f"; max cell {mx[1]:.3f} at (t={mx[0][0]}, p={mx[0][1]})")
        for t, _ in ev.groups:
            curve = ", ".join(f"{cells[k][(t, p)]:.2f}"
                              for p in range(t + 1))
            print(f"    curve t={t}: [{curve}]")

    # ----- shift retention -------------------------------------------------------
    print("\n[shift] relative retention R (learned vs clean):")
    guards = {}
    for s in ("shift-A", "shift-B"):
        gC = res[s][0]["w2/clean"]
        ok = gC >= 0.20 and (s != "shift-B" or guard_B_comp)
        guards[s] = ok
        print(f"  {s}: clean gain {gC:+.1%} — guard "
              f"{'PASS' if ok else 'FAIL (arm NOT TESTED)'}")
    R = {}
    for s in ("shift-A", "shift-B"):
        if not guards[s]:
            continue
        retC = res[s][0]["w2/clean"] / g_base["w2/clean"]
        for k in accepted:
            R[(k, s)] = (res[s][0][k] / g_base[k]) / retC
            print(f"  R({k}, {s}) = {R[(k, s)]:.2f} "
                  f"(gain {g_base[k]:+.1%} -> {res[s][0][k]:+.1%}; "
                  f"clean retention {retC:.2f})")

    # ----- verdicts ---------------------------------------------------------------
    print("\nverdicts:")
    p1 = (g_d1 >= 0.40 and cl_d2 >= 0.90 * cl_full and p1_rep)
    print(f"  P1 anchors + reproduction (D1 >= 40%, D2 >= 90% of full, "
          f"reads within 2 pts / <= -100%): D1 {g_d1:+.1%}, D2 {cl_d2:.1%}, "
          f"reads {g_w2_ba:+.1%}/{g_w2_id:+.1%}/{g_w1_id:+.1%} — "
          f"{'HOLDS' if p1 else 'FAILS (halt interpretation)'}")
    bands = {k: ("equivalent" if rho_base[k] <= RHO_EQUIV else
                 "distinct" if rho_base[k] >= RHO_DISTINCT else "partial")
             for k in accepted}
    p2 = all(b == "equivalent" for b in bands.values())
    print(f"  P2 per-pair equivalence (rho <= {RHO_EQUIV} both learned "
          f"reads): "
          + ", ".join(f"{k} rho {rho_base[k]:.3f} ({bands[k]})"
                      for k in accepted)
          + f" — {'HOLDS' if p2 else 'FAILS'}"
          + ("" if p2 or not all(b == 'distinct' for b in bands.values())
             else " — equivalence-class claim REFUTED for this instance"))
    post_max = max(cells[k][(t, t)] for k in accepted for t, _ in ev.groups)
    all_cells_low = all(v < 0.2 for k in accepted
                        for v in cells[k].values())
    if post_max >= 0.5:
        p3v = ("(a) AGGREGATION ARTIFACT — position-t EPR "
               f"{post_max:.3f} >= 0.5; exp-14's pooled refutation "
               "was the wrong aggregation")
    elif all_cells_low:
        p3v = ("(b) REFUTED AT EVERY GRANULARITY — all cells < 0.2 for "
               "both learned reads")
    else:
        p3v = (f"(c) PARTIAL localization — max position-t {post_max:.3f}, "
               "cells recorded above")
    print(f"  P3 EPR localization: {p3v}")
    big = {k: g_base[k] for k in accepted if g_base[k] >= 0.20}
    if big:
        obs = {"w2/aff/best-a": g_w2_ba, "w2/aff/id": g_w2_id}
        p4 = all(abs(obs[k] - g_base[k]) <= 0.10 for k in big)
        print(f"  P4 observable/exact on {len(big)} patch(es) >= 20%: "
              + "; ".join(f"{k}: {obs[k]:+.1%} vs {g_base[k]:+.1%}"
                          for k in big)
              + f" — {'HOLDS' if p4 else 'FAILS (objective hacking)'}")
    else:
        print("  P4: NOT TESTED — no reproduced patch reached 20%")
    if not R:
        print("  P5 shift robustness: NOT TESTED — both shift guards "
              "failed")
    else:
        fragile = [k for k, v in R.items() if v <= R_FRAGILE]
        robust_all = all(v >= R_ROBUST for v in R.values())
        complete = len(R) == 2 * len(accepted)
        if fragile:
            print(f"  P5 shift robustness: FAILS — STATISTICAL-CONTROL "
                  f"branch (cells <= {R_FRAGILE}: "
                  + ", ".join(f"{k} R {R[k]:.2f}" for k in fragile) + ")")
        elif robust_all and complete:
            print(f"  P5 shift robustness: HOLDS — all cells >= "
                  f"{R_ROBUST} (transported-state reading)")
        elif robust_all:
            print("  P5 shift robustness: PARTIAL — all tested cells "
                  f">= {R_ROBUST}, but a shift arm was NOT TESTED")
        else:
            mid = {k: v for k, v in R.items() if v < R_ROBUST}
            print("  P5 shift robustness: FAILS (partial retention: "
                  + ", ".join(f"{k} R {v:.2f}" for k, v in mid.items())
                  + ")")
    rho_dest = min(rho_base[k] for k in destructive)
    rho_acc = max(rho_base[k] for k in accepted)
    p6 = rho_dest >= 10 * rho_acc
    print(f"  P6 rho separator (min destructive {rho_dest:.3f} >= 10 x "
          f"max accepted {rho_acc:.3f}): {'HOLDS' if p6 else 'FAILS'}")
    print(f"  P7 validity gate: {'HOLDS' if p7 else 'FAILS'}")


if __name__ == "__main__":
    main()
