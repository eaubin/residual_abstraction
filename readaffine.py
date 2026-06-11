"""
readaffine.py — Experiment 14: affine-slice read learning — mechanism
settle, repair, learned composition, and the statistical-read diagnostic.

CONTEXT (see experiments/14-affine-reads.md — the registration carries the
design rationale and the evaluation of the external proposal). Exp 13 left
four open objects: the w1 divergence mechanism (consistent-with
renormalization feedback, settling diagnostics named but not run), the
affine-slice repair (registered, untested), the zero-plane working read
(statistical-predictor hypothesis, needs the correlational diagnostic),
and the gated P5/P6b verdicts.

Four arms. A: instrumented rerun of exp-13's w1 divergence under the
ORIGINAL renormalized parameterization — registered three-part feedback
signature (full-objective ascent >= 0.02 nats; median pre-renorm <c,w> < 1;
norm growth >= 10x with junk fraction rising) upgrades the mechanism to
measured or demotes it. B: the repair — c(u) = c0 + (I - ww^T/|w|^2) u,
<c,w> = 1 by construction in-graph, no post-step operations; both writes x
both inits + benign arm. C: learned composition vs the D2 ceiling (gated:
both gains >= 20%). D: effective plane reading EPR(r) = corr^2(D.r, D.u)
over pooled held-out eval prefix deltas — "what does the read compute",
complementing the mass decomposition's "where does the norm live".

Registered hyperparameters: Adam lr 0.05, 200 steps, minibatch 64, torch
seed 0, minibatch rng seed 0, log_every 20. Final scoring is always the
full-pair non-differentiable evaluator; the torch/numpy regression link is
asserted before any optimization step.

Run: python3 readaffine.py --outdir out/mess3-L4   (~75-100 min)
`--selftest` runs the standard four machinery checks plus the new
affine-construction and EPR plumbing checks, then exits.

RESULTS (see experiments/14-affine-reads.md): P1/P2a/P4/P8 HOLD, P2b
FAILS, P3/P3a FAIL, P5/P6 NOT TESTED, P7 FAILS via its registered
refutation branch. Both exp-13 mechanism hypotheses died. (1) The
renormalization-feedback mechanism is REFUTED, measured twice: pre-renorm
<c,w> median 1.0008/1.0012 (renormalization nearly inactive during the
divergence) and the no-renorm affine parameterization diverges
identically for w1 (-500.5%/-548.2%, 100%-junk, both inits). The w1
failure is a per-write LANDSCAPE asymmetry: w1 escapes its good basin in
<= 20 steps under both parameterizations, then descends within the junk
plateau; w2 descends from a -187% init to +42.5% through identical
machinery. New hypothesis (not measured): Adam per-coordinate steps vs
kappa-sharpened junk curvature; settling: lr/optimizer sweep. (2) The
statistical-predictor account is refuted in pooled-linear form: w2's
working reads (+32.2%/+42.5%; obs-vs-exact 0.8/1.5 pts — P4's 2nd
consecutive hold, faithful across a ~550-point range) score EPR
0.008/0.007 against the clean functional (benign anchor 0.976). Prime
suspect: the pooled-rows operationalization (per-position EPR = exp-15's
first diagnostic). New open object: NO read-side diagnostic separates
working from catastrophic reads (both 0%-plane, majority-junk, EPR~0) —
behavior is currently the only separator. The affine construction itself
is sound (benign +52.2% >= id +51.3%) but repairs nothing.
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
from readopt import decompose
from reads import ALPHAS, mat_power
from miners import sqrt_and_inv
from model import GPT, GPTConfig
from processes import PROCESSES

LAYER = 1
REGISTERED = {"kappa": 100.0, "lr": 0.05, "steps": 200, "batch": 64,
              "pairs_disc": 400, "pairs_eval": 600, "basis_seqs": 800,
              "m": 3, "log_every": 20}


def epr(Drows, r, target):
    """Effective plane reading: corr^2 of the read functional against the
    clean functional over pooled delta rows. Scale-invariant."""
    a = Drows @ r
    b = Drows @ target
    aa, bb = a - a.mean(), b - b.mean()
    return float((aa @ bb) ** 2 / ((aa @ aa) * (bb @ bb)))


def plane_r2(Drows, r, Qc):
    """Descriptive companion: R^2 of regressing the read functional on the
    two plane coordinates (with intercept)."""
    a = Drows @ r
    X = np.column_stack([Drows @ Qc, np.ones(len(a))])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    resid = a - X @ coef
    var = (a - a.mean()) @ (a - a.mean())
    return float(1.0 - (resid @ resid) / var)


def affine_self_checks(d=64):
    """Registered new checks: affine construction (float32 in-graph) and
    EPR plumbing."""
    rng = np.random.default_rng(123)
    w = rng.standard_normal(d)
    c0 = rng.standard_normal(d)
    c0 = c0 / (c0 @ w)
    w_t = torch.from_numpy(w).float()
    c0_t = torch.from_numpy(c0).float()
    w_hat = w_t / w_t.norm()
    u_t = torch.from_numpy(rng.standard_normal(d)).float()
    c_t = c0_t + u_t - w_hat * (u_t @ w_hat)
    assert abs(float(c_t @ w_t) - 1.0) <= 1e-4, "affine constraint violated"
    u0 = torch.zeros(d)
    c_at_0 = c0_t + u0 - w_hat * (u0 @ w_hat)
    assert torch.equal(c_at_0, c0_t), "u=0 read != renormalized init"
    Drows = rng.standard_normal((200, d))
    u = rng.standard_normal(d)
    assert abs(epr(Drows, u, u) - 1.0) <= 1e-12, "EPR plumbing failed"
    print("affine-construction and EPR plumbing checks passed")


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
    ap.add_argument("--log-every", type=int, default=REGISTERED["log_every"])
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
        print("Experiment 14 is registered for the Experiment-6/8-13 "
              "setting; this config is "
              f"{cfg['process']} L{cfg['layers']} d{cfg['d_model']} "
              f"T{cfg['seq_len']} b{cfg['burn_in']}. Use --selftest or "
              "--force-invalid.")
        return
    overridden = [k for k, v in REGISTERED.items() if getattr(args, k) != v]
    if overridden and not args.selftest and not args.force_invalid:
        print(f"Experiment 14 parameters are registered; overridden: "
              f"{overridden}. Use --force-invalid for an exploratory run.")
        return
    if overridden:
        print(f"NOTE: EXPLORATORY RUN — non-registered parameters "
              f"{overridden}; verdicts below are NOT Experiment 14.\n")
    if args.seed != 0 and not args.selftest and not args.force_invalid:
        print(f"Experiment 14 registers seed 0; got {args.seed}. "
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

    # ----- validity gate (P8, enforced) ---------------------------------------
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
    p8 = gap_opt <= 0.005
    print(f"validity gate: gap-to-optimal {gap_opt:+.4f} nats — "
          f"{'PASS' if p8 else 'FAIL'}\n")
    if not p8 and not args.selftest and not args.force_invalid:
        print("exiting: validity gate failed.")
        return

    disc = PairSet(model, proc, cfg, args.pairs_disc, m, args.seed + 111, 800,
                   layer=LAYER)
    ev = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777, 800,
                 layer=LAYER)
    self_checks(model, ev, LAYER, m, V)
    affine_self_checks(d)
    if args.selftest:
        return

    print(f"=== Experiment 14: affine-slice read learning | {proc.name} "
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

    all_pairs = np.arange(disc.n)

    # ----- arm A: instrumented rerun, ORIGINAL renormalized param --------------
    def optimize_renorm(w, init_c, label):
        """The exact exp-13 optimizer (post-step c /= <c,w>), instrumented
        per the registered settling diagnostics."""
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        w_t = torch.from_numpy(w).float()
        c_t = torch.from_numpy(
            (init_c / float(init_c @ w)).copy()).float().requires_grad_()
        opt = torch.optim.Adam([c_t], lr=args.lr)
        hist = {"ip": [], "norm": [float(c_t.detach().norm())],
                "full_ce": [], "frac": []}

        def log_state(step):
            with torch.no_grad():
                ce = float(torch_objective(c_t, w_t, all_pairs, True))
                fp, fj, fn = decompose(
                    T @ c_t.detach().numpy().astype(np.float64), Qc, Qj)
            hist["full_ce"].append(ce)
            hist["frac"].append((fp, fj, fn))
            print(f"    [{label}] step {step:3d}: full-discovery CE "
                  f"{ce:.4f}; |c| {hist['norm'][-1]:.2e}; "
                  f"plane/junk/neutral {fp:.0%}/{fj:.0%}/{fn:.0%}")

        log_state(0)
        for step in range(1, args.steps + 1):
            batch = rng.choice(disc.n, args.batch, replace=False)
            loss = torch_objective(c_t, w_t, batch, True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                ip = float(c_t @ w_t)
                assert abs(ip) > 1e-6, "interchange constraint collapsed"
                hist["ip"].append(ip)
                c_t /= ip
                hist["norm"].append(float(c_t.norm()))
            if step % args.log_every == 0:
                log_state(step)
        return c_t.detach().numpy().astype(np.float64), hist

    def signature(hist):
        """The registered three-part feedback signature."""
        ascent = hist["full_ce"][-1] >= hist["full_ce"][0] + 0.02
        shrink = float(np.median(hist["ip"])) < 1.0
        growth = (hist["norm"][-1] >= 10 * hist["norm"][0]
                  and hist["frac"][-1][1] > hist["frac"][0][1])
        return ascent, shrink, growth

    # ----- arm B: the affine-slice repair --------------------------------------
    def optimize_affine(w, init_c, adversarial, label):
        """c(u) = c0 + (I - w_hat w_hat^T) u — <c,w> = 1 by construction;
        no post-step operations."""
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        w_t = torch.from_numpy(w).float()
        ip0 = float(init_c @ w)
        assert abs(ip0) > 1e-12, "init not renormalizable"
        c0_t = torch.from_numpy((init_c / ip0).copy()).float()
        w_hat = w_t / w_t.norm()
        u_t = torch.zeros(d, requires_grad=True)
        opt = torch.optim.Adam([u_t], lr=args.lr)
        for step in range(1, args.steps + 1):
            batch = rng.choice(disc.n, args.batch, replace=False)
            c_t = c0_t + u_t - w_hat * (u_t @ w_hat)
            loss = torch_objective(c_t, w_t, batch, adversarial)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0 or step == 1:
                print(f"    [{label}] step {step:3d}: batch CE "
                      f"{loss.item():.4f}")
        with torch.no_grad():
            c_t = c0_t + u_t - w_hat * (u_t @ w_hat)
        return c_t.numpy().astype(np.float64)

    def gain_of(c, w, adversarial=True):
        P = oblique_patch(c[:, None], w[:, None])
        return c_obs(disc.run(model, rg_adv.pull(P) if adversarial else P))

    # alpha grids (best-alpha inits, exp-13 rule) and clean/init decompositions
    grids = {}
    print("[inits] alpha-grid controls:")
    for tag, (ang, src, w) in (("w1", (a1, s1, w1)), ("w2", (a2, s2, w2))):
        grid = []
        for a in ALPHAS:
            c = w.copy() if a == 0 else pows_z[a] @ w
            ip = float(c @ w)
            if abs(ip) < 1e-12:
                continue
            grid.append((gain_of(c / ip, w), a, c / ip))
        grids[tag] = max(grid)
        print(f"  {tag} ({src}, {ang:.1f} deg): alpha-grid gains "
              + ", ".join(f"a{a:.2f}:{g:+.1%}" for g, a, _ in grid)
              + f"; best a{grids[tag][1]:.2f}")
    print()

    print("[arm A] instrumented rerun of the w1 divergence (renormalized "
          "parameterization):")
    res_ren = {}
    for iname, ic in (("best-a", grids["w1"][2]), ("id", w1.copy())):
        label = f"ren/w1/{iname}"
        print(f"  optimizing {label}:")
        c_fin, hist = optimize_renorm(w1, ic, label)
        g = gain_of(c_fin, w1)
        asc, shr, gro = signature(hist)
        print(f"    [{label}] final full-discovery gain {g:+.1%}; signature: "
              f"ascent {asc} (CE {hist['full_ce'][0]:.4f} -> "
              f"{hist['full_ce'][-1]:.4f}), shrinkage {shr} (median <c,w> "
              f"{float(np.median(hist['ip'])):.4f}), runaway {gro} (|c| "
              f"{hist['norm'][0]:.2e} -> {hist['norm'][-1]:.2e}, junk "
              f"{hist['frac'][0][1]:.0%} -> {hist['frac'][-1][1]:.0%})")
        res_ren[label] = (w1, c_fin, g, (asc, shr, gro))
    print()

    print("[arm B] affine-slice optimization:")
    res_aff = {}
    init_gains = {}
    for tag, (ang, src, w) in (("w1", (a1, s1, w1)), ("w2", (a2, s2, w2))):
        for iname, ic in (("best-a", grids[tag][2]), ("id", w.copy())):
            label = f"adv/{tag}/{iname}"
            init_gains[label] = gain_of(ic / float(ic @ w), w)
            print(f"  optimizing {label} (init gain "
                  f"{init_gains[label]:+.1%}):")
            c_opt = optimize_affine(w, ic, True, label)
            g = gain_of(c_opt, w)
            fp, fj, fn = decompose(T @ c_opt, Qc, Qj)
            print(f"    [{label}] full-discovery gain {g:+.1%}; stream-read "
                  f"decomposition plane {fp:.0%} / junk {fj:.0%} / neutral "
                  f"{fn:.0%}")
            res_aff[label] = (w, c_opt, g)

    # benign sanity arm
    w1b = rg_adv.back(w1)
    w1b = w1b / np.linalg.norm(w1b)
    g_id_ben = c_obs(disc.run(model, np.outer(w1b, w1b)))
    print(f"\n[arm B] benign arm: id-read gain {g_id_ben:+.1%}; "
          "optimizing ben/w1/id:")
    c_ben = optimize_affine(w1b, w1b.copy(), False, "ben/w1/id")
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
    for label, rec in {**res_aff,
                       **{k: v[:3] for k, v in res_ren.items()}}.items():
        w, c_opt, g = rec
        P = rg_adv.pull(oblique_patch(c_opt[:, None], w[:, None]))
        exacts[label] = true_closure(P)[m]
        print(f"  {label}: observable {g:+.1%} vs exact "
              f"{exacts[label]:+.1%}")

    # arm C: composition of the best affine read per write (exp-13 rule)
    def best_for(tag):
        return max((res_aff[k] for k in res_aff
                    if k.startswith(f"adv/{tag}/")), key=lambda r: r[2])
    b1, b2 = best_for("w1"), best_for("w2")
    wB = np.column_stack([b1[0], b2[0]])
    cB = np.column_stack([b1[1], b2[1]])
    g_B = c_obs(disc.run(model, rg_adv.pull(oblique_patch(cB, wB))))
    cl_B = true_closure(rg_adv.pull(oblique_patch(cB, wB)))[m]
    print(f"  arm C composition: observable {g_B:+.1%}, exact {cl_B:.1%}")

    # ----- arm D: effective plane reading (held-out eval deltas) ---------------
    Drows = np.vstack([
        (ev.pref_src[t] - ev.pref_tgt[t]).numpy().reshape(-1, d)
        for t, _ in ev.groups]).astype(np.float64)
    print(f"\n[arm D] effective plane reading over {len(Drows)} pooled "
          "eval prefix deltas:")
    eprs = {}
    for tag, w, ucl in (("w1", w1, u1), ("w2", w2, u2)):
        assert abs(epr(Drows, ucl, ucl) - 1.0) <= 1e-12
        rows = [("id", T @ w), ("init/best-a", T @ grids[tag][2])]
        rows += [("aff/" + k.split("/")[2], T @ res_aff[k][1])
                 for k in res_aff if k.startswith(f"adv/{tag}/")]
        if tag == "w1":
            rows += [("ren/" + k.split("/")[2] + " (diverged)",
                      T @ res_ren[k][1]) for k in res_ren]
        for name, r in rows:
            key = f"{tag}/{name}"
            eprs[key] = epr(Drows, r, ucl)
            print(f"  {key}: EPR {eprs[key]:.3f} (plane-2D R2 "
                  f"{plane_r2(Drows, r, Qc):.3f})")
    e_ben = epr(Drows, c_ben, w1b)
    print(f"  ben/w1/id learned: EPR {e_ben:.3f}")

    # ----- verdicts -------------------------------------------------------------
    print("\nverdicts:")
    gw2 = b2[2]
    p1 = (g_d1 >= 0.40 and cl_d2[m] >= 0.90 * cl_full[m]
          and g_ben >= g_id_ben - 0.05 and gw2 >= 0.20)
    print(f"  P1 anchors + no-regression (D1 >= 40%, D2 >= 90% of full, "
          f"benign affine >= id - 5pts, w2 affine >= 20%): D1 {g_d1:+.1%}, "
          f"D2 {cl_d2[m]:.1%}, benign {g_ben:+.1%} vs {g_id_ben:+.1%}, w2 "
          f"{gw2:+.1%} — {'HOLDS' if p1 else 'FAILS'}")
    p2a = all(v[2] <= -1.00 for v in res_ren.values())
    print(f"  P2a divergence reproduces (both renorm gains <= -100%): "
          + ", ".join(f"{k} {v[2]:+.1%}" for k, v in res_ren.items())
          + f" — {'HOLDS' if p2a else 'FAILS (determinism breach)'}")
    if p2a:
        p2b = all(all(v[3]) for v in res_ren.values())
        verdict2b = ("HOLDS — mechanism upgraded to MEASURED" if p2b else
                     "FAILS — renormalization feedback demoted")
        print(f"  P2b feedback signature (ascent, shrinkage, runaway in "
              f"both runs): "
              + ", ".join(f"{k} {v[3]}" for k, v in res_ren.items())
              + f" — {verdict2b}")
    else:
        print("  P2b: NOT TESTED — P2a failed")
    p3a = all(res_aff[k][2] >= init_gains[k] - 0.05 for k in res_aff)
    print(f"  P3a stability (every affine run >= init gain - 5pts): "
          + ", ".join(f"{k} {res_aff[k][2]:+.1%} (init "
                      f"{init_gains[k]:+.1%})" for k in res_aff)
          + f" — {'HOLDS' if p3a else 'FAILS'}")
    g1 = b1[2]
    p3 = g1 >= 0.40
    print(f"  P3 headline (best affine read for {s1} >= 40%): {g1:+.1%} — "
          f"{'HOLDS' if p3 else 'FAILS'}")
    big = {k: v for k, v in {**res_aff,
                             **{k_: v_[:3] for k_, v_ in res_ren.items()}
                             }.items() if v[2] >= 0.20}
    if not big:
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
        print(f"  P5 learned composition >= 90% of full: {cl_B:.1%} — "
              f"{'HOLDS' if p5 else 'FAILS'}")
    else:
        print("  P5 learned composition: NOT TESTED — both single-write "
              "gains must reach 20%")
    if b1[2] >= 0.20:
        fp1, _, _ = decompose(T @ b1[1], Qc, Qj)
        p6 = fp1 <= 0.20
        verdict6 = ("HOLDS — zero-plane reads generalize" if p6 else
                    "FAILS — proxy-read phenomenon is write-specific")
        print(f"  P6 w1 learned-read plane fraction <= 20% (direction "
              f"FLIPPED from exp-13 P6b): {fp1:.0%} — {verdict6}")
    else:
        print("  P6: NOT TESTED — w1 best affine gain under 20%")
    work_keys = [k for k in res_aff if res_aff[k][2] >= 0.20]
    if work_keys:
        vals = {k: eprs[f"{k.split('/')[1]}/aff/{k.split('/')[2]}"]
                for k in work_keys}
        p7 = all(v >= 0.5 for v in vals.values())
        refuted = [k for k, v in vals.items() if v < 0.2]
        verdict7 = ("HOLDS" if p7 else
                    ("FAILS (statistical-predictor account REFUTED: "
                     f"{refuted})" if refuted else "FAILS (partial)"))
        print(f"  P7 EPR >= 0.5 on working learned reads: "
              + ", ".join(f"{k} {v:.3f}" for k, v in vals.items())
              + f" — {verdict7}")
    else:
        print("  P7 EPR: NOT TESTED — no learned adversarial read "
              "reached 20%")
    print(f"  P8 validity gate: {'HOLDS' if p8 else 'FAILS'}")


if __name__ == "__main__":
    main()
