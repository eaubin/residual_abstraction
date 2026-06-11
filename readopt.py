"""
readopt.py — Experiment 13: fixed-write read optimization (read-only
gradient).

CONTEXT (see experiments/13-read-gradient.md — the registration carries
the design rationale, the honesty status of gradients, and the P4 dual
reading). Experiments 10-12 cornered the adversarial failure to one
object: the read covector. Here it is LEARNED from behavioral signal:
write held fixed (the two exp-12 round-1 near-plane writes, reproduced
in-run), read c optimized by Adam on the observable model-vs-model KL
over discovery-pair minibatches, differentiably through the exact m=3
chain; <c, w> = 1 enforced by renormalization every step. Two registered
inits per adversarial write (best-alpha, id); benign sanity arm; stage B
composes the two optimized pairs against the recorded D2 ceiling.

The read-decomposition diagnostic (squared-norm fractions of the
stream-space read covector on causal plane / registered junk plane /
neutral remainder) turns exp-12's neutral-contamination hypothesis into a
measurement (P6), applied to inits, optimized reads, and the clean
diagnostic read.

Registered hyperparameters: Adam lr 0.05, 200 steps, minibatch 64, torch
seed 0, minibatch rng seed 0. Final scoring is always the full-pair
non-differentiable evaluator, so optimizer numerics cannot contaminate
verdicts; the two code paths are asserted to agree (rel 1e-4) before any
optimization step.

Run: python3 readopt.py --outdir out/mess3-L4   (~45-60 min)
`--selftest` runs the standard four machinery checks and exits.

RESULTS (see experiments/13-read-gradient.md): P1/P2/P4/P6a/P7 HOLD, P3
FAILS, P5/P6b NOT TESTED. P4 holds AT LAST (6th experiment): observable
tracks exact to 0.3/1.6 points on gradient-optimized adversarial patches —
oracle-free scoring survives maximal selection pressure. Discovery: w2's
learned read transfers +43.7% with ZERO causal-plane mass (49% junk / 51%
neutral) — reads are statistical predictors exploiting echo correlations,
not geometric aligners; the clean-read picture is too narrow. P3's
failure is a NEW typed defect: constraint-renormalization instability
(both w1 runs ASCENDED their own objective — the post-step c /= <c,w>
renormalization feeds back into a junk runaway, -498%); a
parameterization bug, not unlearnability (w2 converged cleanly through
the same machinery). Repair for the follow-up: affine-slice
parameterization c = c0 + v, v orthogonal to w. P6a at 100%: spectral
inits are pure-neutral — exp-12's hypothesis confirmed as measurement.
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
from patches import oblique_patch, write_pool
from reads import ALPHAS, mat_power
from miners import sqrt_and_inv
from model import GPT, GPTConfig
from processes import PROCESSES

LAYER = 1
REGISTERED = {"kappa": 100.0, "lr": 0.05, "steps": 200, "batch": 64,
              "pairs_disc": 400, "pairs_eval": 600, "basis_seqs": 800,
              "m": 3}


def decompose(s, Qc, Qj):
    """Squared-norm fractions of a stream covector on (plane, junk,
    neutral)."""
    s = s / np.linalg.norm(s)
    fp = float(np.sum((Qc.T @ s) ** 2))
    fj = float(np.sum((Qj.T @ s) ** 2))
    return fp, fj, 1.0 - fp - fj


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/mess3-L4")
    ap.add_argument("--kappa", type=float, default=REGISTERED["kappa"])
    ap.add_argument("--lr", type=float, default=REGISTERED["lr"])
    ap.add_argument("--steps", type=int, default=REGISTERED["steps"])
    ap.add_argument("--batch", type=int, default=REGISTERED["batch"])
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
        print("Experiment 13 is registered for the Experiment-6/8-12 "
              "setting; this config is "
              f"{cfg['process']} L{cfg['layers']} d{cfg['d_model']} "
              f"T{cfg['seq_len']} b{cfg['burn_in']}. Use --selftest or "
              "--force-invalid.")
        return
    overridden = [k for k, v in REGISTERED.items() if getattr(args, k) != v]
    if overridden and not args.selftest and not args.force_invalid:
        print(f"Experiment 13 parameters are registered; overridden: "
              f"{overridden}. Use --force-invalid for an exploratory run.")
        return
    if overridden:
        print(f"NOTE: EXPLORATORY RUN — non-registered parameters "
              f"{overridden}; verdicts below are NOT Experiment 13.\n")
    if args.seed != 0 and not args.selftest and not args.force_invalid:
        print(f"Experiment 13 registers seed 0; got {args.seed}. "
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
    for p_ in model.parameters():
        p_.requires_grad_(False)

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

    print(f"=== Experiment 13: fixed-write read optimization | {proc.name} "
          f"| patch L{LAYER} | kappa = {args.kappa:g} | Adam lr {args.lr}, "
          f"{args.steps} steps, batch {args.batch} ===\n")

    # ----- observable refs -----------------------------------------------------
    q_src_d = disc.run(model, None, src_side=True)
    q_un_d = disc.run(model, None)
    q_full_d = disc.run(model, np.eye(d))
    D0 = float(kl_rows(q_src_d, q_un_d).mean())
    Dfull = float(kl_rows(q_src_d, q_full_d).mean())
    assert D0 > Dfull
    c_obs = lambda q: (D0 - float(kl_rows(q_src_d, q).mean())) / (D0 - Dfull)
    assert abs(c_obs(q_full_d) - 1.0) < 1e-12

    # ----- anchor + T ----------------------------------------------------------
    print("[anchor] frozen Experiment-6 loop:")
    Q = np.zeros((d, 0))
    q_cur, c_cur = q_un_d, 0.0
    while Q.shape[1] < 8:
        v = mined_direction(disc, Q, kl_rows(q_src_d, q_cur))
        Q_try = np.hstack([Q, v[:, None]])
        q_try = disc.run(model, Q_try @ Q_try.T)
        if c_obs(q_try) - c_cur < 0.05:
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
    print("transform checks passed\n")

    rng_b = np.random.default_rng(args.seed + 555)
    Xb = proc.sample(args.basis_seqs, L, rng_b)
    Sb = stream_to(model, torch.from_numpy(Xb), LAYER).double().numpy()
    keep = np.arange(burn, L - 1)
    Rb = center_by_position(Sb[:, keep].reshape(-1, d), np.tile(keep, len(Xb)),
                            np.ones(len(Xb) * len(keep), dtype=bool))
    Sig_x = np.cov(Rb.T)
    Sig_z = T @ Sig_x @ T
    Sx, Sx_inv = sqrt_and_inv(Sig_x)
    Sz, Sz_inv = sqrt_and_inv(Sig_z)
    pows_z = {a: mat_power(Sig_z, -a) for a in ALPHAS if a > 0}

    class Regime:
        def __init__(self, view_raw, view_wht, S, Sinv, pull, back):
            self.view_raw, self.view_wht = view_raw, view_wht
            self.S, self.Sinv, self.pull, self.back = S, Sinv, pull, back

    rg_adv = Regime(ZView(disc, T), ZView(disc, T @ Sz_inv), Sz, Sz_inv,
                    lambda P: T @ P @ Tinv, lambda v: Tinv @ v)
    rg_ben = Regime(disc, ZView(disc, Sx_inv), Sx, Sx_inv,
                    lambda P: P, lambda v: v)

    # ----- the two fixed writes (registered exp-12 rule, reproduced) ----------
    w0w = kl_rows(q_src_d, q_un_d)
    pool = write_pool(rg_adv, np.zeros((d, 0)), w0w, 1, d, args.seed)
    angled = sorted(
        ((principal_angles_deg((rg_adv.back(w) /
                                np.linalg.norm(rg_adv.back(w)))[:, None],
                               Qc)[0], src, w) for src, w in pool))
    near = [(a, s, w) for a, s, w in angled if a <= 15.0]
    assert len(near) >= 2, "fewer than two near-plane writes reproduced"
    (a1, s1, w1), (a2, s2, w2) = near[0], near[1]
    print(f"fixed writes (exp-12 rule): {s1} at {a1:.1f} deg; {s2} at "
          f"{a2:.1f} deg\n")

    # ----- differentiable chain objective --------------------------------------
    pair_t = np.empty(disc.n, dtype=int)
    pair_loc = np.empty(disc.n, dtype=int)
    for t, idx in disc.groups:
        pair_t[idx] = t
        pair_loc[idx] = np.arange(len(idx))
    Tt = torch.from_numpy(T).float()
    Tinvt = torch.from_numpy(Tinv).float()
    qsrc_t = torch.from_numpy(q_src_d).float()
    pos_all = torch.arange(L)

    def torch_objective(c_t, w_t, batch, adversarial=True):
        """Mean cross-entropy -sum q_src log q_patched over `batch` pair ids
        (== KL + const). Differentiable in c_t."""
        Pz = torch.outer(c_t, w_t)
        P = (Tt @ Pz @ Tinvt) if adversarial else Pz
        total = c_t.new_zeros(())
        for t in np.unique(pair_t[batch]):
            sel = batch[pair_t[batch] == t]
            loc = pair_loc[sel]
            Xc = torch.from_numpy(disc.Xc_tgt[t][loc])          # (b, C, L)
            b, C, _ = Xc.shape
            pt = disc.pref_tgt[t][loc]
            ps = pt + (disc.pref_src[t][loc] - pt) @ P
            flat = Xc.reshape(b * C, L)
            ps_r = ps.repeat_interleave(C, dim=0)
            x = model.tok(flat) + model.pos(pos_all)
            for li, blk in enumerate(model.blocks):
                if li == LAYER:
                    x = torch.cat([ps_r, x[:, t + 1:]], dim=1)
                x = blk(x)
            logp = torch.log_softmax(model.head(model.ln_f(x)), dim=-1)
            rows = torch.arange(b * C)
            lq = sum(logp[rows, t + j, flat[:, t + 1 + j]] for j in range(m))
            total = total - (qsrc_t[sel] * lq.reshape(b, C)).sum()
        return total / len(batch)

    # regression link: torch path == numpy evaluator for a fixed read
    w1_t = torch.from_numpy(w1).float()
    ce_torch = float(torch_objective(w1_t, w1_t, np.arange(disc.n)))
    qp = disc.run(model, rg_adv.pull(np.outer(w1, w1)))
    ce_np = float(-(q_src_d * np.log(np.clip(qp, 1e-12, None))).sum(axis=1)
                  .mean())
    rel = abs(ce_torch - ce_np) / abs(ce_np)
    assert rel < 1e-4, f"torch/numpy objective mismatch: rel {rel:.2e}"
    print(f"objective regression link passed (rel {rel:.1e})\n")

    def optimize(w, init_c, adversarial, label):
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        w_t = torch.from_numpy(w).float()
        c_t = torch.from_numpy(
            (init_c / float(init_c @ w)).copy()).float().requires_grad_()
        opt = torch.optim.Adam([c_t], lr=args.lr)
        for step in range(1, args.steps + 1):
            batch = rng.choice(disc.n, args.batch, replace=False)
            loss = torch_objective(c_t, w_t, batch, adversarial)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                ip = float(c_t @ w_t)
                assert abs(ip) > 1e-6, "interchange constraint collapsed"
                c_t /= ip
            if step % 50 == 0 or step == 1:
                print(f"    [{label}] step {step:3d}: batch CE "
                      f"{loss.item():.4f}")    # .item(): silences the
                # harmless requires_grad cast warning seen in the recorded
                # exp-13 output (cosmetic; touches no computation)
        return c_t.detach().numpy().astype(np.float64)

    # ----- stage A: per-write optimization -------------------------------------
    results = {}
    print("[stage A] alpha-grid controls and optimization:")
    for tag, (ang, src, w) in (("w1", (a1, s1, w1)), ("w2", (a2, s2, w2))):
        grid = []
        for a in ALPHAS:
            c = w.copy() if a == 0 else pows_z[a] @ w
            ip = float(c @ w)
            if abs(ip) < 1e-12:
                continue
            g = c_obs(disc.run(model, rg_adv.pull(
                oblique_patch((c / ip)[:, None], w[:, None]))))
            grid.append((g, a, c / ip))
        grid_best = max(grid)
        print(f"  {tag} ({src}, {ang:.1f} deg): alpha-grid gains "
              + ", ".join(f"a{a:.2f}:{g:+.1%}" for g, a, _ in grid)
              + f"; best a{grid_best[1]:.2f}")
        inits = [("best-a", grid_best[2]), ("id", w.copy())]
        for iname, ic in inits:
            label = f"adv/{tag}/{iname}"
            print(f"  optimizing {label}:")
            c_opt = optimize(w, ic, True, label)
            g = c_obs(disc.run(model, rg_adv.pull(
                oblique_patch(c_opt[:, None], w[:, None]))))
            sread = T @ c_opt
            fp, fj, fn = decompose(sread, Qc, Qj)
            print(f"    [{label}] full-discovery gain {g:+.1%}; stream-read "
                  f"decomposition plane {fp:.0%} / junk {fj:.0%} / neutral "
                  f"{fn:.0%}")
            results[label] = (w, c_opt, g)
        # decompositions of init and clean read for P6
        s_init = T @ grid_best[2]
        fp_i, fj_i, fn_i = decompose(s_init, Qc, Qj)
        u = rg_adv.back(w)
        fp_c, fj_c, fn_c = decompose(u, Qc, Qj)
        print(f"  {tag} init (best-a) decomposition: plane {fp_i:.0%} / "
              f"junk {fj_i:.0%} / neutral {fn_i:.0%}; clean read: plane "
              f"{fp_c:.0%} / junk {fj_c:.0%} / neutral {fn_c:.0%}")
        if tag == "w1":
            neutral_init_w1 = fn_i

    # benign sanity arm
    w1b = rg_adv.back(w1)
    w1b = w1b / np.linalg.norm(w1b)
    g_id_ben = c_obs(disc.run(model, np.outer(w1b, w1b)))
    print(f"\n[stage A] benign arm: id-read gain {g_id_ben:+.1%}; "
          "optimizing ben/w1/id:")
    c_ben = optimize(w1b, w1b.copy(), False, "ben/w1/id")
    g_ben = c_obs(disc.run(model, oblique_patch(c_ben[:, None],
                                                w1b[:, None])))
    print(f"    [ben/w1/id] full-discovery gain {g_ben:+.1%}\n")

    # D1/D2 anchors
    u1 = rg_adv.back(w1); u1 = u1 / np.linalg.norm(u1)
    u2 = rg_adv.back(w2); u2 = u2 / np.linalg.norm(u2)
    g_d1 = c_obs(disc.run(model, np.outer(u1, u1)))
    U = orthonormal(np.column_stack([u1, u2]))
    P_d2 = U @ U.T
    print(f"D1 (clean rank-1, {s1}): {g_d1:+.1%}")

    # ----- evaluation -----------------------------------------------------------
    q0_e = ev.run(model, None)
    rows_fe = kl_by_horizon(q0_e, ev.p_tgt3, V, m)
    rows_ge = kl_by_horizon(q0_e, ev.p_src3, V, m)
    floor_e = {mm: float(rows_fe[mm].mean()) for mm in rows_fe}
    gap_e = {mm: float(rows_ge[mm].mean()) for mm in rows_ge}
    ms = list(range(1, m + 1))

    def true_closure(P):
        rows_t = kl_by_horizon(ev.run(model, P), ev.p_src3, V, m)
        return {mm: (gap_e[mm] - float(rows_t[mm].mean()))
                / (gap_e[mm] - floor_e[mm]) for mm in ms}

    print("\nevaluation (exact targets; pooled m=3):")
    cl_full = true_closure(np.eye(d))
    cl_d2 = true_closure(P_d2)
    print(f"  full {cl_full[m]:.1%}; D2 anchor {cl_d2[m]:.1%}")
    exacts = {}
    for label, (w, c_opt, g) in results.items():
        P = rg_adv.pull(oblique_patch(c_opt[:, None], w[:, None]))
        exacts[label] = true_closure(P)[m]
        print(f"  {label}: observable {g:+.1%} vs exact "
              f"{exacts[label]:+.1%}")
    # stage B: registered selection rule — the best final-gain read per
    # write across its inits (consistent with P3/P6b's selection).
    def best_for(tag):
        return max((results[k] for k in results if f"/{tag}/" in k),
                   key=lambda r: r[2])
    b1, b2 = best_for("w1"), best_for("w2")
    wB = np.column_stack([b1[0], b2[0]])
    cB = np.column_stack([b1[1], b2[1]])
    g_B = c_obs(disc.run(model, rg_adv.pull(oblique_patch(cB, wB))))
    cl_B = true_closure(rg_adv.pull(oblique_patch(cB, wB)))[m]
    print(f"  stage B composition: observable {g_B:+.1%}, exact {cl_B:.1%}")

    # ----- verdicts -------------------------------------------------------------
    print("\nverdicts:")
    p1 = g_d1 >= 0.40 and cl_d2[m] >= 0.90 * cl_full[m]
    print(f"  P1 anchors (D1 >= 40%, D2 exact >= 90% of full): D1 "
          f"{g_d1:+.1%}, D2 {cl_d2[m]:.1%} — {'HOLDS' if p1 else 'FAILS'}")
    p2 = g_ben >= g_id_ben - 0.05
    print(f"  P2 benign sanity (optimized >= id - 5pts): {g_ben:+.1%} vs id "
          f"{g_id_ben:+.1%} — {'HOLDS' if p2 else 'FAILS'}")
    g1 = b1[2]                    # best final gain for w1 across its inits
    p3 = g1 >= 0.40
    print(f"  P3 headline (optimized read for {s1} >= 40%): best "
          f"{g1:+.1%} — {'HOLDS' if p3 else 'FAILS'}")
    big = {k: v for k, v in results.items() if v[2] >= 0.20}
    if not big:
        p4 = False
        print("  P4 observable/exact (gain >= 20% patches): NOT TESTED — "
              "no optimized patch reached 20%")
    else:
        p4 = all(abs(v[2] - exacts[k]) <= 0.10 for k, v in big.items())
        print(f"  P4 observable/exact on {len(big)} patch(es) >= 20%: "
              + "; ".join(f"{k}: {v[2]:+.1%} vs {exacts[k]:+.1%}"
                          for k, v in big.items())
              + f" — {'HOLDS' if p4 else 'FAILS (objective hacking)'}")
    if b1[2] >= 0.20 and b2[2] >= 0.20:
        p5 = cl_B >= 0.90 * cl_full[m]
        print(f"  P5 composition >= 90% of full: {cl_B:.1%} — "
              f"{'HOLDS' if p5 else 'FAILS'}")
    else:
        p5 = False
        print("  P5 composition: NOT TESTED — both single-write gains must "
              "reach 20%")
    p6a = neutral_init_w1 >= 0.50
    print(f"  P6a init neutral fraction >= 50%: {neutral_init_w1:.0%} — "
          f"{'HOLDS' if p6a else 'FAILS'}")
    if p3:
        w_, c_, g_ = b1
        fp_o, _, _ = decompose(T @ c_, Qc, Qj)
        p6b = fp_o >= 0.50
        verdict6b = ("HOLDS" if p6b
                     else "FAILS (neutral hypothesis falsified interestingly)")
        print(f"  P6b optimized-read plane fraction >= 50%: {fp_o:.0%} — "
              f"{verdict6b}")
    else:
        print("  P6b: NOT TESTED — P3 failed")
    print(f"  P7 validity gate: {'HOLDS' if p7 else 'FAILS'}")


if __name__ == "__main__":
    main()
