"""
dyck.py — Experiment 7: the Dyck/stack port — do the diagnostics transfer?

CONTEXT (see experiments/7-dyck.md, the pre-registration, committed before
training or running). Experiments 1-6 built and validated a battery on
Mess3/Z1R: decode-sufficiency curves with supervised proposal families
(exp 1-2), interchange interventions and the echo taxonomy (exp 3), the
per-step incremental-closure persistence statistic (exp 4), the depth
profile of state-vs-summary (exp 5), and oracle-free interventional CEGAR
discovery (exp 6). This experiment ports the battery to a structurally
different process — depth-bounded Dyck-2 (processes.dyck2: stack states,
bracket matching, longer-range constraints; still exactly an HMM, so every
oracle stays closed-form) — and asks whether the diagnostics, the typed
findings, and the oracle-free soundness (exp 6's P7) survive the move.

The registered tension worth the trip: bracket matching is what attention
does well from RAW tokens, so the re-derivation bypass that limited
persistence on Mess3 could be STRONGER here (shrinking the state region),
even though the naive intuition says longer-range structure should mean
more state. Either outcome is informative; the directional bets are in the
pre-registration.

STAGES (one runner, registered rules fix every data-dependent choice):
  A  Calibration on the train.py cache: held-out affine residual->belief
     R^2; the belief intrinsic dimension k_B (registered rule: minimal
     principal components of the train-split exact beliefs reaching 99%
     variance); decode k* for pls/pca under the Experiment-2 stopping rule.
  B  Depth profile (Experiment-5 form, full/pre only): per-step incremental
     closure and unemb-basis coherence at every interior layer. The state
     layer l_dagger := argmax over interior layers of step-2 incremental
     closure (ties -> smaller layer) — a registered rule, since Dyck's
     state layer is unknown a priori.
  C  Interventional CEGAR discovery (Experiment-6 form) at l_dagger, with
     full/pca/pls/rand/emb controls at matched k*, nested closure(k) curve,
     per-position stability, and principal angles.

HONESTY: stage-C discovery is supervised on the observable model-vs-model
objective only; exact belief-conditioned closures are evaluation-only, on
disjoint pairs. Anti-triviality guards carried from Experiment 6 (k_max
raised to 12 because k_B may exceed Mess3's 2; hitting it is NON-CONVERGENT).

Run:  python3 train.py --process dyck2 --layers 4 --outdir out/dyck2-L4
      python3 dyck.py --outdir out/dyck2-L4
`--selftest` runs the known-answer machinery checks against any 4-layer
model dir (no dyck model or cache needed) and exits.

RESULTS (see experiments/7-dyck.md): P3-P9 HOLD, P1-P2 FAIL informatively —
the new typed outcome REPRESENTATION-ORACLE MISMATCH. The linear-belief
calibration breaks on Dyck (affine R^2 0.66, decode k* > 12 for BOTH
families, k_B = 13) although the model is behaviorally near-exact; the
interventional battery transfers untouched: state at L1 (88.1%/85.2%
incremental), CEGAR k* = 4 at 92.6% vs full 93.6%, oracle-free scoring
sound to 5.9 points. Post-hoc check that killed the easy explanation: the
m=3 completion distributions are ALSO 13-dim (belief->mgram map full-rank),
so k*=4 is KL-weighting + the model's own routing, not horizon truncation.
No state interference at L3 (vs Mess3's negative closures); the pls echo is
0.2% (6-for-6 on the scale lesson); variance mimicry recurs, so Experiment
8 remains necessary.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from abstraction import (CompletionPLS, PCAAbstraction, affine_lstsq,
                         center_by_position, completeness_kl_rows, kl_rows,
                         mean_kl, r2_score)
from discover import PairSet, mined_direction, principal_angles_deg, \
    self_checks
from midstream import kl_by_horizon, marginal, orthonormal, stream_to
from model import GPT, GPTConfig
from processes import PROCESSES

REGISTERED = {"k_max": 12, "eps_gain": 0.05, "eps_drop": 0.01,
              "pairs_disc": 400, "pairs_eval": 600, "basis_seqs": 800,
              "m": 3}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
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
    # The registration fixes the whole model config, not just the process:
    # seq_len/burn_in determine the pooled positions (8/16/24) and d_model
    # the stream the bases live in (review fix, pre-run).
    registered_cfg = (proc.name == "dyck2" and cfg["layers"] == 4
                      and cfg["seq_len"] == 32 and cfg["d_model"] == 64
                      and cfg["burn_in"] == 4)
    if not registered_cfg and not args.selftest and not args.force_invalid:
        print("Experiment 7 is registered for dyck2 / 4 layers / d_model 64 /"
              f" seq_len 32 / burn_in 4; this config is {cfg['process']} "
              f"L{cfg['layers']} d{cfg['d_model']} T{cfg['seq_len']} "
              f"b{cfg['burn_in']}. Use --selftest or --force-invalid.")
        return
    overridden = [k for k, v in REGISTERED.items() if getattr(args, k) != v]
    if overridden and not args.selftest and not args.force_invalid:
        print(f"Experiment 7 parameters are registered; overridden: "
              f"{overridden}. Use --force-invalid for an exploratory run.")
        return
    if overridden:
        print(f"NOTE: EXPLORATORY RUN — non-registered parameters "
              f"{overridden}; verdicts below are NOT Experiment 7.\n")
    if args.seed != 0:
        print(f"NOTE: seed {args.seed} != 0 — a seed-robustness rerun.\n")

    L, burn, V, m = cfg["seq_len"], cfg["burn_in"], proc.V, args.m
    d = cfg["d_model"]
    interior = list(range(1, cfg["layers"]))

    model = GPT(GPTConfig(vocab=V, seq_len=L, d_model=d,
                          n_layers=cfg["layers"]))
    model.load_state_dict(torch.load(os.path.join(args.outdir, "model.pt"),
                                     map_location="cpu"))
    model.eval()

    # ----- P9 validity gate (Experiment-5/6 estimator, enforced) -------------
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
    p9 = gap_opt <= 0.005
    print(f"validity gate: gap-to-optimal {gap_opt:+.4f} nats — "
          f"{'PASS' if p9 else 'FAIL'}\n")
    if not p9 and not args.selftest and not args.force_invalid:
        print("exiting: validity gate failed. Retrain longer and rerun.")
        return

    # ----- self-checks ---------------------------------------------------------
    ev1 = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777, 800,
                  layer=interior[0])
    self_checks(model, ev1, interior[0], m, V)
    if args.selftest:
        return

    print(f"=== Experiment 7: Dyck port | {proc.name} | {cfg['layers']} "
          f"layers | S = {proc.S} states, V = {V} | {args.pairs_eval} eval / "
          f"{args.pairs_disc} discovery pairs | m = 1..{m} ===\n")

    # ===== Stage A: calibration on the cache ==================================
    dc = np.load(os.path.join(args.outdir, "cache.npz"))
    R, B, G = dc["resid"], dc["belief"], dc["mgram"]
    perm = np.random.default_rng(args.seed).permutation(len(R))
    R, B, G = R[perm], B[perm], G[perm]
    n_tr = int(0.7 * len(R))
    tr, te = slice(None, n_tr), slice(n_tr, None)
    mask = np.zeros(len(R), dtype=bool); mask[:n_tr] = True
    R = center_by_position(R, dc["pos"][perm], mask)

    Wb, b0, _ = affine_lstsq(R[tr], B[tr])
    r2_full = r2_score(B[te], R[te] @ Wb + b0)
    # k_B: registered rule — minimal PCs of train-split exact beliefs
    # reaching 99% variance.
    evals = np.linalg.eigvalsh(np.cov((B[tr] - B[tr].mean(0)).T))[::-1]
    k_B = int(np.searchsorted(np.cumsum(evals) / evals.sum(), 0.99) + 1)
    print(f"[A] affine full residual -> belief, held-out R^2 = {r2_full:.4f}"
          f"   |   belief intrinsic dim k_B (99% var rule) = {k_B}")
    # Horizon non-collapse diagnostic (the check that killed the
    # horizon-truncation hypothesis during the first run's analysis; in the
    # tracked output since review): intrinsic dim of the exact m-gram
    # distributions, and exactness/rank of the linear belief->mgram map —
    # if k_G ~ k_B and the map is full-rank, a small causal k* is
    # KL-weighting + model routing, NOT the m-horizon needing less state.
    evG = np.linalg.eigvalsh(np.cov((G[tr] - G[tr].mean(0)).T))[::-1]
    k_G = int(np.searchsorted(np.cumsum(evG) / evG.sum(), 0.99) + 1)
    Mlin, b_lin, _ = affine_lstsq(B[tr], G[tr])
    r2_lin = r2_score(G[te], B[te] @ Mlin + b_lin)
    print(f"[A] m-gram intrinsic dim k_G (99% var) = {k_G}; belief->mgram "
          f"affine fit held-out R^2 = {r2_lin:.6f}, "
          f"rank {np.linalg.matrix_rank(Mlin, tol=1e-8)}")

    # decode k* under the Experiment-2 stopping rule, pls & pca
    rows_or = completeness_kl_rows(B[tr], G[tr], B[te], G[te], seed=args.seed)
    kl_or = float(rows_or.mean())
    KL0 = mean_kl(G[te], np.tile(G[tr].mean(axis=0), (len(G[te]), 1)))
    margin = 0.02 * max(KL0 - kl_or, 0.0)
    fams_A = {"pls": CompletionPLS(R[tr], G[tr]),
              "pca": PCAAbstraction(R[tr])}
    # The decode search must extend past P2's threshold (k_B + 1) — the
    # CEGAR k_max is a different registered bound and stopping at it cannot
    # establish P2 failure (review fix).
    k_hi = max(args.k_max, k_B + 1)
    k_dec = {}
    for fname, fam in fams_A.items():
        k_dec[fname] = None
        for k in range(1, k_hi + 1):
            rows = completeness_kl_rows(fam(R[tr], k), G[tr],
                                        fam(R[te], k), G[te], seed=args.seed)
            diff = rows - rows_or
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            if diff.mean() <= max(2 * se, margin):
                k_dec[fname] = k
                break
        ks = k_dec[fname] if k_dec[fname] else f">{k_hi}"
        print(f"[A] decode k* ({fname}, exp-2 stopping rule): {ks}")
    print()

    # ===== Stage B: depth profile (full/pre), l_dagger rule ===================
    evs = {interior[0]: ev1}
    for l in interior[1:]:
        evs[l] = PairSet(model, proc, cfg, args.pairs_eval, m,
                         args.seed + 777, 800, layer=l)
    q0, r_un = ev1.run(model, None, with_resid=True)
    rows_f = kl_by_horizon(q0, ev1.p_tgt3, V, m)
    rows_g = kl_by_horizon(q0, ev1.p_src3, V, m)
    floor = {mm: float(rows_f[mm].mean()) for mm in rows_f}
    gapm = {mm: float(rows_g[mm].mean()) for mm in rows_g}
    ms = list(range(1, m + 1))
    print(f"[B] unpatched reference: "
          + " | ".join(f"m={mm}: floor {floor[mm]:.5f}, gap {gapm[mm]:.5f}"
                       for mm in ms))

    with torch.no_grad():
        Wu = model.head.weight.double().numpy()
        g_ln = model.ln_f.weight.double().numpy()
    U_coh = orthonormal((np.eye(d) - np.ones((d, d)) / d)
                        @ (g_ln[:, None] * Wu.T))
    w_star = np.argmax(marginal(ev1.p_src3, V, 1, m), axis=1)
    cont_idx = w_star * V ** (m - 1)
    rows_n = np.arange(args.pairs_eval)
    _, r_src = ev1.run(model, None, src_side=True, with_resid=True)
    z_src = r_src[rows_n, cont_idx] @ U_coh
    z_un = r_un[rows_n, cont_idx] @ U_coh

    inc, coh, full_cl = {}, {}, {}
    print(f"[B] full/pre profile:  layer   closure m=1..{m}   "
          "incr step2/step3   coherence")
    for l in interior:
        qp, rp = evs[l].run(model, np.eye(d), with_resid=True)
        rows_t = kl_by_horizon(qp, ev1.p_src3, V, m)
        cl = {mm: (gapm[mm] - float(rows_t[mm].mean()))
              / (gapm[mm] - floor[mm]) for mm in ms}
        full_cl[l] = cl
        trf = {mm: gapm[mm] - cl[mm] * (gapm[mm] - floor[mm]) for mm in ms}
        inc[l] = {mm: ((gapm[mm] - gapm[mm - 1]) - (trf[mm] - trf[mm - 1]))
                  / ((gapm[mm] - gapm[mm - 1])
                     - (floor[mm] - floor[mm - 1])) for mm in ms[1:]}
        z_p = rp[rows_n, cont_idx] @ U_coh
        coh[l] = float((np.linalg.norm(z_p - z_src, axis=1)
                        < np.linalg.norm(z_un - z_src, axis=1)).mean())
        print(f"       L{l}: "
              + "  ".join(f"{cl[mm]:>6.1%}" for mm in ms)
              + "   " + "/".join(f"{inc[l][mm]:.1%}" for mm in ms[1:])
              + f"   {coh[l]:.1%}")
    l_dag = min(interior, key=lambda l: (-inc[l][2], l))
    print(f"[B] state layer l_dagger (registered rule: argmax step-2 "
          f"incremental, ties -> smaller): L{l_dag}\n")

    # ===== Stage C: interventional CEGAR at l_dagger ===========================
    disc = PairSet(model, proc, cfg, args.pairs_disc, m, args.seed + 111, 800,
                   layer=l_dag)
    ev = evs[l_dag]
    q_src_d = disc.run(model, None, src_side=True)
    q_un_d = disc.run(model, None)
    q_full_d = disc.run(model, np.eye(d))
    D0 = float(kl_rows(q_src_d, q_un_d).mean())
    Dfull = float(kl_rows(q_src_d, q_full_d).mean())
    assert D0 > Dfull, "observable scale degenerate: D0 <= D_full"
    c_obs = lambda q: (D0 - float(kl_rows(q_src_d, q).mean())) / (D0 - Dfull)
    assert abs(c_obs(q_full_d) - 1.0) < 1e-12, "c_obs(full) != 1"
    print(f"[C] CEGAR at L{l_dag} (observable refs: D0 {D0:.5f}, "
          f"D_full {Dfull:.5f}):")

    Q = np.zeros((d, 0))
    q_cur, c_cur = q_un_d, 0.0
    converged = False
    while Q.shape[1] < args.k_max:
        v = mined_direction(disc, Q, kl_rows(q_src_d, q_cur))
        assert abs(np.linalg.norm(v) - 1) < 1e-9 and (
            Q.shape[1] == 0 or np.abs(Q.T @ v).max() < 1e-9)
        Q_try = np.hstack([Q, v[:, None]])
        q_try = disc.run(model, Q_try @ Q_try.T)
        c_try = c_obs(q_try)
        gain = c_try - c_cur
        if gain < args.eps_gain:
            print(f"    k={Q_try.shape[1]}: c_obs {c_try:.1%} "
                  f"(gain {gain:+.1%} < eps_gain) -> STOP, revert")
            converged = True
            break
        Q, q_cur, c_cur = Q_try, q_try, c_try
        print(f"    k={Q.shape[1]}: c_obs {c_cur:.1%} (gain {gain:+.1%}) "
              "-> accept")
    if not converged:
        print(f"    k_max = {args.k_max} reached: NON-CONVERGENT")
    changed = True
    while changed and Q.shape[1] > 1:
        changed = False
        for j in range(Q.shape[1]):
            Qj = np.delete(Q, j, axis=1)
            cj = c_obs(disc.run(model, Qj @ Qj.T))
            if c_cur - cj < args.eps_drop:
                print(f"    coarsen: dropped direction {j + 1} "
                      f"(cost {c_cur - cj:+.2%})")
                Q, c_cur, changed = Qj, cj, True
                break
    k_star = Q.shape[1]
    print(f"[C] fixed point: k* = {k_star}, c_obs = {c_cur:.1%}\n")

    # control bases at l_dagger (Experiment-5 discovery protocol)
    rng_d = np.random.default_rng(args.seed + 555)
    Xd = proc.sample(args.basis_seqs, L, rng_d)
    Sd = stream_to(model, torch.from_numpy(Xd), l_dag).double().numpy()
    keep = np.arange(burn, L - 1)
    Gd = np.concatenate([proc.mgram_table(proc.beliefs_along(row)[keep], m)
                         for row in Xd])
    Rd = center_by_position(Sd[:, keep].reshape(-1, d), np.tile(keep, len(Xd)),
                            np.ones(len(Xd) * len(keep), dtype=bool))
    pls_c = CompletionPLS(Rd, Gd)
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        W_tok = model.tok.weight.double().numpy()
    bases = {
        "disc": Q,
        "pca": PCAAbstraction(Rd).Vt[:k_star].T,
        "pls": orthonormal(pls_c.whiten @ pls_c.U[:, :k_star]),
        "rand": orthonormal(rng.standard_normal((d, k_star))),
        "emb": orthonormal(W_tok.T)[:, :min(k_star, V)],
    }

    q0_e = ev.run(model, None)
    rows_fe = kl_by_horizon(q0_e, ev.p_tgt3, V, m)
    rows_ge = kl_by_horizon(q0_e, ev.p_src3, V, m)
    floor_e = {mm: float(rows_fe[mm].mean()) for mm in rows_fe}
    gap_e = {mm: float(rows_ge[mm].mean()) for mm in rows_ge}

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

    print("[C] evaluation (disjoint pairs, exact targets; pooled m=1..3 | "
          "per-position at m=3):")
    closures, stab = {}, {}
    for name, P in [("full", np.eye(d))] + [(f, Bm @ Bm.T)
                                            for f, Bm in bases.items()]:
        cl, bp = true_closure(P)
        closures[name], stab[name] = cl, bp
        kk = d if name == "full" else bases[name].shape[1]
        print(f"    {name:>5} (k={kk})  "
              + "  ".join(f"m={mm}: {cl[mm]:>6.1%}" for mm in ms)
              + "   | " + "  ".join(f"t={t}: {bp[t]:.1%}"
                                    for t, _ in ev.groups))
    print("\n[C] nested closure(k) of the discovered basis (pooled m=3):")
    for k in range(1, k_star + 1):
        Qk = Q[:, :k]
        cl, _ = true_closure(Qk @ Qk.T)
        print(f"    k={k}: {cl[m]:.1%}")
    print("\n[C] principal angles (deg), discovered vs controls:")
    for f in ("pca", "pls", "emb"):
        ang = principal_angles_deg(Q, bases[f])
        print(f"    disc vs {f:>4}: " + ", ".join(f"{a:.1f}" for a in ang))

    # ===== verdicts (experiments/7-dyck.md) ====================================
    print("\nverdicts:")
    p1 = r2_full >= 0.95
    print(f"  P1 calibration R^2 >= 0.95: {r2_full:.4f} — "
          f"{'HOLDS' if p1 else 'FAILS'}")
    p2 = k_dec["pls"] is not None and k_dec["pls"] <= k_B + 1
    print(f"  P2 decode k*(pls) <= k_B + 1: "
          f"{k_dec['pls']} vs k_B {k_B} — {'HOLDS' if p2 else 'FAILS'}")
    p3 = inc[interior[0]][2] >= 0.50
    print(f"  P3 step-2 incremental at L{interior[0]} >= 50%: "
          f"{inc[interior[0]][2]:.1%} — {'HOLDS' if p3 else 'FAILS'}")
    weakly_dec = lambda vals: all(vals[i + 1] <= vals[i] + 0.02
                                  for i in range(len(vals) - 1))
    p4 = all(weakly_dec([inc[l][mm] for l in interior]) for mm in ms[1:])
    print(f"  P4 incremental closures weakly decreasing in depth: "
          f"{'HOLDS' if p4 else 'FAILS'}")
    p5 = closures["pls"][m] <= 0.50 * closures["full"][m]
    print(f"  P5 echo persists (pls <= 50% of full at L{l_dag}): pls "
          f"{closures['pls'][m]:.1%} vs full {closures['full'][m]:.1%} — "
          f"{'HOLDS' if p5 else 'FAILS'}")
    p6 = converged and k_star <= k_B + 2 and \
        closures["disc"][m] >= 0.90 * closures["full"][m]
    print(f"  P6 CEGAR converges at k* <= k_B + 2 with >= 90% of full: "
          f"k*={k_star} (k_B {k_B}), disc {closures['disc'][m]:.1%} vs full "
          f"{closures['full'][m]:.1%} — {'HOLDS' if p6 else 'FAILS'}")
    p7 = abs(c_cur - closures["disc"][m]) <= 0.10
    print(f"  P7 observable/exact agreement <= 10 points: c_obs {c_cur:.1%} "
          f"vs exact {closures['disc'][m]:.1%} — {'HOLDS' if p7 else 'FAILS'}")
    p8 = all(closures["rand"][mm] <= 0.25 for mm in ms)
    print(f"  P8 rand <= 25%: {'HOLDS' if p8 else 'FAILS'}")
    print(f"  P9 validity gate: {'HOLDS' if p9 else 'FAILS'}")
    print(f"\ncharacterization (no thresholds): Dyck profile at "
          f"L2/L3 step-2 incremental {inc.get(2, {}).get(2, float('nan')):.1%}"
          f"/{inc.get(3, {}).get(2, float('nan')):.1%} vs Mess3's "
          "52.5%/-29.7%; k* vs k_B above; angles above.")

    # ----- plot ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for mm in ms[1:]:
            axes[0].plot(interior, [inc[l][mm] for l in interior], "o-",
                         label=f"step {mm}")
        axes[0].plot(interior, [coh[l] for l in interior], "s--", c="gray",
                     label="coherence frac")
        axes[0].set_xlabel("patch layer"); axes[0].set_xticks(interior)
        axes[0].set_title("persistence profile (full/pre)")
        axes[0].axhline(0, ls=":", c="gray"); axes[0].legend(fontsize=8)
        names = list(closures)
        axes[1].bar(range(len(names)), [closures[f][m] for f in names])
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(
            [f"{f}\nk={d if f == 'full' else bases[f].shape[1]}"
             for f in names], fontsize=8)
        axes[1].axhline(0.9 * closures["full"][m], ls="--", c="r",
                        label="P6 bar")
        axes[1].set_title(f"discovery at L{l_dag}, k*={k_star}")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{proc.name}: Experiment 7")
        p = os.path.join(args.outdir, "experiment7.png")
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plotting skipped: {e})")


if __name__ == "__main__":
    main()
