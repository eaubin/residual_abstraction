"""
expcommon.py — shared experiment scaffolding (adopted at exp 15).

Policy: concluded experiment scripts (exps 1-14) are FROZEN RECORDS and
keep their inline copies of this machinery; from exp 15 onward scripts
import this module, so the living edge has exactly one copy. Everything
here is extracted verbatim from the readaffine.py / readopt.py lineage —
behavior-preserving by construction, verified by the standard selftests
and by the in-run anchor / write / read reproduction asserts (the
registered tripwires). This module imports only from the stable shared
layer (discover/patches/reads/miners/midstream/abstraction/adversarial/
model/processes), never from frozen experiment scripts — small helpers
that originated in a frozen script (decompose, epr) live here as
canonical copies, with the originals untouched as historical record.

Contents: the standard argparse/guard pair, model loading, the validity
gate, observable references (c_obs), the frozen exp-6 anchor loop, the
adversarial transform T and regime, write-pool reproduction (exp-12
rule), the differentiable chain objective + its torch/numpy regression
link, the affine-slice optimizer (exp 14), spectral-read helpers
(exp 12), and small shared metrics (jeffreys_rows, decompose, epr).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from abstraction import center_by_position, kl_rows
from adversarial import ZView
from discover import PairSet, mined_direction, principal_angles_deg
from midstream import orthonormal, stream_to
from patches import oblique_patch, write_pool
from reads import ALPHAS, mat_power
from miners import sqrt_and_inv
from model import GPT, GPTConfig

LAYER = 1

__all__ = [
    "LAYER", "PairSet", "decompose", "epr", "jeffreys_rows",
    "standard_parser", "standard_guard", "load_model", "validity_gate",
    "observable_refs", "reproduce_anchor", "build_transform",
    "basis_covariance", "Regime", "adversarial_regime", "reproduce_writes",
    "make_torch_objective", "regression_link", "optimize_affine",
    "alpha_powers", "alpha_grid", "oblique_patch", "orthonormal", "kl_rows",
]


def jeffreys_rows(qa, qb):
    """Per-row Jeffreys divergence between two (n, C) distributions."""
    return 0.5 * (kl_rows(qa, qb) + kl_rows(qb, qa))


def decompose(s, Qc, Qj):
    """Squared-norm fractions of a stream covector on (plane, junk,
    neutral). Canonical copy (verbatim from the frozen readopt.py, exp 13
    — review fix: expcommon must not depend on frozen scripts)."""
    s = s / np.linalg.norm(s)
    fp = float(np.sum((Qc.T @ s) ** 2))
    fj = float(np.sum((Qj.T @ s) ** 2))
    return fp, fj, 1.0 - fp - fj


def epr(Drows, r, target):
    """Effective plane reading: corr^2 of the read functional against the
    clean functional over delta rows. Scale-invariant. Canonical copy
    (verbatim from the frozen readaffine.py, exp 14 — same review fix)."""
    a = Drows @ r
    b = Drows @ target
    aa, bb = a - a.mean(), b - b.mean()
    return float((aa @ bb) ** 2 / ((aa @ aa) * (bb @ bb)))


# ----- standard CLI + guards ---------------------------------------------------

def standard_parser(registered):
    """The standard argument set: one flag per REGISTERED entry plus
    --outdir/--seed/--selftest/--force-invalid."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/mess3-L4")
    for k, v in registered.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force-invalid", action="store_true")
    return ap


def standard_guard(args, cfg, proc, expname, registered):
    """Config / parameter / seed guards (exps 8-14 semantics). Returns
    False if the run must exit."""
    registered_cfg = (proc.name == "mess3" and cfg["layers"] == 4
                      and cfg["seq_len"] == 32 and cfg["d_model"] == 64
                      and cfg["burn_in"] == 4)
    if not registered_cfg and not args.selftest and not args.force_invalid:
        print(f"{expname} is registered for the Experiment-6/8-14 setting; "
              f"this config is {cfg['process']} L{cfg['layers']} "
              f"d{cfg['d_model']} T{cfg['seq_len']} b{cfg['burn_in']}. "
              "Use --selftest or --force-invalid.")
        return False
    overridden = [k for k, v in registered.items() if getattr(args, k) != v]
    if overridden and not args.selftest and not args.force_invalid:
        print(f"{expname} parameters are registered; overridden: "
              f"{overridden}. Use --force-invalid for an exploratory run.")
        return False
    if overridden:
        print(f"NOTE: EXPLORATORY RUN — non-registered parameters "
              f"{overridden}; verdicts below are NOT {expname}.\n")
    if args.seed != 0 and not args.selftest and not args.force_invalid:
        print(f"{expname} registers seed 0; got {args.seed}. "
              "Use --force-invalid.")
        return False
    if args.seed != 0:
        print(f"NOTE: EXPLORATORY RUN — seed {args.seed} != 0.\n")
    return True


def load_model(outdir, cfg, proc):
    model = GPT(GPTConfig(vocab=proc.V, seq_len=cfg["seq_len"],
                          d_model=cfg["d_model"], n_layers=cfg["layers"]))
    model.load_state_dict(torch.load(os.path.join(outdir, "model.pt"),
                                     map_location="cpu"))
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model


def validity_gate(model, proc, cfg, seed):
    """Token-weighted NLL gate on 2000 sequences. Returns (gap, passed);
    the caller decides whether to exit (selftest/force-invalid may not)."""
    L, V = cfg["seq_len"], proc.V
    Xg = proc.sample(2000, L, np.random.default_rng(seed + 999))
    with torch.no_grad():
        tot, cnt = 0.0, 0
        for i in range(0, len(Xg), 256):
            logits = model(torch.from_numpy(Xg[i:i + 256]))
            tgt = torch.from_numpy(Xg[i:i + 256, 1:]).reshape(-1)
            tot += F.cross_entropy(logits[:, :-1].reshape(-1, V), tgt,
                                   reduction="sum").item()
            cnt += tgt.numel()
    gap = tot / cnt - cfg["optimal_nll"]
    passed = gap <= 0.005
    print(f"validity gate: gap-to-optimal {gap:+.4f} nats — "
          f"{'PASS' if passed else 'FAIL'}\n")
    return gap, passed


# ----- observable refs, anchor, transform --------------------------------------

def observable_refs(model, disc, d):
    """Returns (q_src_d, q_un_d, c_obs) with the standard sanity asserts."""
    q_src_d = disc.run(model, None, src_side=True)
    q_un_d = disc.run(model, None)
    q_full_d = disc.run(model, np.eye(d))
    D0 = float(kl_rows(q_src_d, q_un_d).mean())
    Dfull = float(kl_rows(q_src_d, q_full_d).mean())
    assert D0 > Dfull
    c_obs = lambda q: (D0 - float(kl_rows(q_src_d, q).mean())) / (D0 - Dfull)
    assert abs(c_obs(q_full_d) - 1.0) < 1e-12
    return q_src_d, q_un_d, c_obs


def reproduce_anchor(model, disc, q_src_d, q_un_d, c_obs, d, eps_gain=0.05):
    """The frozen exp-6 CEGAR loop; asserts k* = 2, c_obs within 0.005 of
    0.998 (the registered tripwire). Returns Qc."""
    print("[anchor] frozen Experiment-6 loop:")
    Q = np.zeros((d, 0))
    q_cur, c_cur = q_un_d, 0.0
    while Q.shape[1] < 8:
        v = mined_direction(disc, Q, kl_rows(q_src_d, q_cur))
        Q_try = np.hstack([Q, v[:, None]])
        q_try = disc.run(model, Q_try @ Q_try.T)
        if c_obs(q_try) - c_cur < eps_gain:
            break
        Q, q_cur, c_cur = Q_try, q_try, c_obs(q_try)
        print(f"  k={Q.shape[1]}: c_obs {c_cur:.1%}")
    assert Q.shape[1] == 2 and abs(c_cur - 0.998) < 0.005
    print(f"[anchor] reproduced: k* = 2, c_obs = {c_cur:.1%}\n")
    return Q


def build_transform(Qc, d, kappa, junk_seed=0):
    """The registered adversarial T; asserts both transform checks.
    Returns (T, Tinv, Qj). `junk_seed` (exp 17, backward-compatible:
    default 0 is the historical draw of exps 8-16) selects the junk-plane
    draw."""
    Pc = Qc @ Qc.T
    rng0 = np.random.default_rng(junk_seed)
    Gj = rng0.standard_normal((d, 2))
    Gj -= Qc @ (Qc.T @ Gj)
    Qj = orthonormal(Gj)
    T = np.eye(d) - (1 - 1 / kappa) * Pc + (kappa - 1) * (Qj @ Qj.T)
    Tinv = np.eye(d) + (kappa - 1) * Pc + (1 / kappa - 1) * (Qj @ Qj.T)
    # float64 roundoff in the pullback product scales ~ kappa^4, measured
    # empirically (max |err|: 8.7e-12 / 1.2e-9 / 8.2e-8 at kappa
    # 30/100/300 on synthetic draws — the exp-17 crash twice over; a
    # kappa^2 model was wrong). Tolerance below: bit-identical 1e-9 at
    # kappa <= 100 (eight runs of evidence at that exact setting), and
    # the measured kappa^4 law with a 10x margin above.
    atol = 1e-9 if kappa <= 100.0 else 1e-8 * (kappa / 100.0) ** 4
    assert np.allclose(T @ Tinv, np.eye(d), atol=atol)
    Qzc = orthonormal(T @ Qc)
    assert np.allclose(T @ (Qzc @ Qzc.T) @ Tinv, Pc, atol=atol)
    print("transform checks passed\n")
    return T, Tinv, Qj


def basis_covariance(model, proc, cfg, seed, n_seqs, layer=LAYER):
    """Position-centered stream covariance on the standard basis sample."""
    L, burn, d = cfg["seq_len"], cfg["burn_in"], cfg["d_model"]
    rng_b = np.random.default_rng(seed + 555)
    Xb = proc.sample(n_seqs, L, rng_b)
    Sb = stream_to(model, torch.from_numpy(Xb), layer).double().numpy()
    keep = np.arange(burn, L - 1)
    Rb = center_by_position(Sb[:, keep].reshape(-1, d),
                            np.tile(keep, len(Xb)),
                            np.ones(len(Xb) * len(keep), dtype=bool))
    return np.cov(Rb.T)


class Regime:
    def __init__(self, view_raw, view_wht, S, Sinv, pull, back):
        self.view_raw, self.view_wht = view_raw, view_wht
        self.S, self.Sinv, self.pull, self.back = S, Sinv, pull, back


def adversarial_regime(disc, T, Tinv, Sig_x):
    """Returns (rg_adv, Sig_z)."""
    Sig_z = T @ Sig_x @ T
    Sz, Sz_inv = sqrt_and_inv(Sig_z)
    rg = Regime(ZView(disc, T), ZView(disc, T @ Sz_inv), Sz, Sz_inv,
                lambda P: T @ P @ Tinv, lambda v: Tinv @ v)
    return rg, Sig_z


def reproduce_writes(rg_adv, q_src_d, q_un_d, Qc, d, seed):
    """The exp-12 rule: the two nearest-to-plane round-1 writes from the
    registered pool. Returns ((angle, src, w), (angle, src, w))."""
    w0w = kl_rows(q_src_d, q_un_d)
    pool = write_pool(rg_adv, np.zeros((d, 0)), w0w, 1, d, seed)
    angled = sorted(
        ((principal_angles_deg((rg_adv.back(w) /
                                np.linalg.norm(rg_adv.back(w)))[:, None],
                               Qc)[0], src, w) for src, w in pool))
    near = [(a, s, w) for a, s, w in angled if a <= 15.0]
    assert len(near) >= 2, "fewer than two near-plane writes reproduced"
    (a1, s1, w1), (a2, s2, w2) = near[0], near[1]
    print(f"fixed writes (exp-12 rule): {s1} at {a1:.1f} deg; {s2} at "
          f"{a2:.1f} deg\n")
    return (a1, s1, w1), (a2, s2, w2)


# ----- differentiable chain (exp-13/14 machinery) -------------------------------

def make_torch_objective(model, disc, T, Tinv, q_src_d, layer=LAYER):
    """Returns torch_objective(c_t, w_t, batch, adversarial=True): mean
    cross-entropy -sum q_src log q_patched over `batch` pair ids
    (== KL + const), differentiable in c_t."""
    L, m = disc.Xe.shape[1], disc.m
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
                if li == layer:
                    x = torch.cat([ps_r, x[:, t + 1:]], dim=1)
                x = blk(x)
            logp = torch.log_softmax(model.head(model.ln_f(x)), dim=-1)
            rows = torch.arange(b * C)
            lq = sum(logp[rows, t + j, flat[:, t + 1 + j]] for j in range(m))
            total = total - (qsrc_t[sel] * lq.reshape(b, C)).sum()
        return total / len(batch)

    return torch_objective


def regression_link(torch_objective, model, disc, rg_adv, q_src_d, w):
    """Asserts the torch path matches the numpy evaluator (rel 1e-4)."""
    w_t = torch.from_numpy(w).float()
    ce_torch = float(torch_objective(w_t, w_t, np.arange(disc.n)))
    qp = disc.run(model, rg_adv.pull(np.outer(w, w)))
    ce_np = float(-(q_src_d * np.log(np.clip(qp, 1e-12, None))).sum(axis=1)
                  .mean())
    rel = abs(ce_torch - ce_np) / abs(ce_np)
    assert rel < 1e-4, f"torch/numpy objective mismatch: rel {rel:.2e}"
    print(f"objective regression link passed (rel {rel:.1e})\n")


def optimize_affine(torch_objective, n_pairs, d, lr, steps, batch, seed,
                    w, init_c, adversarial, label, print_every=100,
                    checkpoint_every=None):
    """The exp-14 affine-slice optimizer: c(u) = c0 + (I - w_hat w_hat^T) u
    — <c,w> = 1 by construction in-graph; no post-step operations.

    `checkpoint_every` (exp 16, backward-compatible: default None keeps
    the original return) additionally collects the read at step 0 (the
    renormalized init) and every `checkpoint_every` steps; returns
    (c_final, [(step, c), ...]). The trajectory is identical either way —
    checkpointing only reads c, it never touches the optimizer state."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    w_t = torch.from_numpy(w).float()
    ip0 = float(init_c @ w)
    assert abs(ip0) > 1e-12, "init not renormalizable"
    c0_t = torch.from_numpy((init_c / ip0).copy()).float()
    w_hat = w_t / w_t.norm()
    u_t = torch.zeros(d, requires_grad=True)
    opt = torch.optim.Adam([u_t], lr=lr)

    def read_now():
        with torch.no_grad():
            c_t = c0_t + u_t - w_hat * (u_t @ w_hat)
        return c_t.numpy().astype(np.float64)

    ckpts = [(0, read_now())] if checkpoint_every else None
    for step in range(1, steps + 1):
        b = rng.choice(n_pairs, batch, replace=False)
        c_t = c0_t + u_t - w_hat * (u_t @ w_hat)
        loss = torch_objective(c_t, w_t, b, adversarial)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % print_every == 0 or step == 1:
            print(f"    [{label}] step {step:3d}: batch CE "
                  f"{loss.item():.4f}")
        if checkpoint_every and step % checkpoint_every == 0:
            ckpts.append((step, read_now()))
    c_fin = read_now()
    return (c_fin, ckpts) if checkpoint_every else c_fin


# ----- spectral reads (exp-12 machinery) ----------------------------------------

def alpha_powers(Sig_z):
    return {a: mat_power(Sig_z, -a) for a in ALPHAS if a > 0}


def alpha_grid(w, pows_z, gain_fn):
    """The exp-12 grid c ∝ Σ̂_z^{-α}w (normalized to <c,w> = 1), scored by
    gain_fn. Returns (grid, best) with entries (gain, alpha, read)."""
    grid = []
    for a in ALPHAS:
        c = w.copy() if a == 0 else pows_z[a] @ w
        ip = float(c @ w)
        if abs(ip) < 1e-12:
            continue
        grid.append((gain_fn(c / ip), a, c / ip))
    return grid, max(grid)
