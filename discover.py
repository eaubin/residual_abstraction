"""
discover.py — Experiment 6: interventional discovery — CEGAR on interchange
closure at L1.

CONTEXT (see experiments/6-interventional-discovery.md, the pre-registration,
committed before the first run). Experiments 3-5 showed every
completions-supervised proposal family finds ECHOES — decode-sufficient,
causally weak subspaces (X-whitened PLS at L1 closes 3.3% where the full
patch closes 98.7%). This experiment makes the scoring itself
interventional: a CEGAR loop proposes directions mined from the prefix
differences of behaviorally-failed pairs, tests each by prefix-wide
interchange at L1 of the Experiment-5 model, and keeps a direction only if
it earns >= eps_gain of OBSERVABLE closure.

HONESTY (sharpened for discovery): the loop's objective is model-vs-model —
KL(q_source_run || q_patched) over the exact m=3 chain — the score that
exists on a real LLM with no oracle. Exact belief-conditioned closures are
EVALUATION-ONLY, computed on a disjoint pair sample (the Experiment-5
evaluation seed, for comparability).

ANTI-TRIVIALITY GUARDS (the registered answer to "grow k until full-space"):
(1) marginal-gain rule: dimension k+1 must add >= eps_gain = 0.05 of
observable closure, so the loop MUST stop once the remaining gap is small;
(2) k_max = 8 << d = 64, and reaching it is typed NON-CONVERGENT, a failure;
(3) the success criterion P1 jointly requires k* <= 4 AND >= 90% of the
full-patch closure; (4) a coarsen pass drops any direction worth
< eps_drop = 0.01 before k* is reported. The nested closure(k) curve is
reported in full.

Controls at matched k*: pca / pls (Experiment-5 discovery protocol at L1),
rand, full (the ceiling), and `emb` — the fixed architecture basis available
at L1 (orthonormalized token-embedding rows, <= V dims, no fitting).
Principal angles between discovered and pca/pls/emb subspaces are reported
as characterization. P1-P7 thresholds live in the pre-registration; the
verdict logic below implements them.

Run: python3 discover.py --outdir out/mess3-L4
`--selftest` runs the known-answer checks and exits (no cache.npz needed).

RESULTS (see experiments/6-interventional-discovery.md): P1-P7 ALL HOLD.
The loop converged in two rounds at k* = 2 (the belief-simplex dimension):
exact-target closure 98.3% vs the full 64-dim patch's 98.7%, vs the PLS
echo's 2.7% (principal angles 86-87 deg — the echo is near-orthogonal to
the causal plane). The oracle-free objective tracked exact closure to 1.5
points (P7) — the soundness datum the LLM phase needs. Declared limitation
realized: the discovered plane is 3.3-3.6 deg from PCA's top-2 (variance
mimicry), so this model cannot distinguish "interventional discovery works"
from "variance was right anyway"; the buried-causal-content regime is the
natural Experiment 7.
"""

import argparse
import json
import os
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

from abstraction import (CompletionPLS, PCAAbstraction, center_by_position,
                         kl_rows)
from midstream import (chain_probs, kl_by_horizon, marginal, orthonormal,
                       stream_to)
from model import GPT, GPTConfig
from processes import PROCESSES

LAYER = 1                      # registered patch point: input to block 2


def principal_angles_deg(Qa, Qb):
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


class PairSet:
    """Pairs at three pooled positions with per-group token/prefix arrays —
    the Experiment-4/5 protocol, packaged so discovery and evaluation sets
    stay disjoint and identically constructed."""

    def __init__(self, model, proc, cfg, n, m, seed_seqs, n_seqs):
        L, burn, V = cfg["seq_len"], cfg["burn_in"], proc.V
        rng = np.random.default_rng(seed_seqs)
        self.Xe = proc.sample(n_seqs, L, rng)
        self.S = stream_to(model, torch.from_numpy(self.Xe), LAYER)
        self.B = np.stack([proc.beliefs_along(row) for row in self.Xe])
        self.n, self.m, self.V, self.d = n, m, V, cfg["d_model"]
        a = rng.integers(0, n_seqs, n)
        b = rng.integers(0, n_seqs, n)
        self.a, self.b = a, np.where(b == a, (b + 1) % n_seqs, b)
        ts = np.unique(np.linspace(burn + 4, L - 1 - m - 4, 3).astype(int))
        t_of = ts[np.arange(n) % len(ts)]
        self.ts, self.groups = ts, [(int(t), np.where(t_of == t)[0])
                                    for t in ts]
        self.p_src3 = proc.mgram_table(self.B[self.b, t_of], m)
        self.p_tgt3 = proc.mgram_table(self.B[self.a, t_of], m)
        conts = np.array(list(product(range(V), repeat=m)))
        self.C = len(conts)
        self.Xc_tgt, self.Xc_src = {}, {}
        self.pref_tgt, self.pref_src = {}, {}
        for t, idx in self.groups:
            for store, seqs in ((self.Xc_tgt, self.a), (self.Xc_src, self.b)):
                xc = np.repeat(self.Xe[seqs[idx]][:, None, :], self.C, axis=1)
                xc = xc.copy()
                xc[:, :, t + 1:t + 1 + m] = conts[None, :, :]
                store[t] = xc
            self.pref_tgt[t] = self.S[torch.from_numpy(self.a[idx])][:, :t + 1]
            self.pref_src[t] = self.S[torch.from_numpy(self.b[idx])][:, :t + 1]

    def run(self, model, P, src_side=False):
        """Prefix-wide patch with projector P (None = unpatched). Returns
        the (n, C) m-step joint."""
        q = np.empty((self.n, self.C))
        for t, idx in self.groups:
            X = self.Xc_src[t] if src_side else self.Xc_tgt[t]
            ps = None
            if P is not None:
                pt = self.pref_tgt[t].double().numpy()
                d_ = self.pref_src[t].double().numpy() - pt
                ps = torch.from_numpy(pt + d_ @ P).float()
            qg, _ = chain_probs(model, X, LAYER, ps, t, self.m, self.V)
            q[idx] = qg
        return q


# The registered CEGAR/protocol parameters (experiments/
# 6-interventional-discovery.md). Overriding any of them demotes the run to
# exploratory — enforced below, like the model-config guard.
REGISTERED = {"k_max": 8, "eps_gain": 0.05, "eps_drop": 0.01,
              "pairs_disc": 400, "pairs_eval": 600, "basis_seqs": 800,
              "m": 3}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/mess3-L4")
    ap.add_argument("--k-max", type=int, default=REGISTERED["k_max"])
    ap.add_argument("--eps-gain", type=float, default=REGISTERED["eps_gain"])
    ap.add_argument("--eps-drop", type=float, default=REGISTERED["eps_drop"])
    ap.add_argument("--pairs-disc", type=int,
                    default=REGISTERED["pairs_disc"])
    ap.add_argument("--pairs-eval", type=int,
                    default=REGISTERED["pairs_eval"])
    ap.add_argument("--basis-seqs", type=int, default=REGISTERED["basis_seqs"],
                    help="sequences for fitting the pca/pls CONTROL bases "
                    "(the discovery/evaluation pair sets are part of the "
                    "registered protocol and are not controlled by this)")
    ap.add_argument("--m", type=int, default=REGISTERED["m"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force-invalid", action="store_true",
                    help="proceed on a non-registered config, failed gate, "
                    "or overridden registered parameters (explicitly "
                    "exploratory, not Experiment 6)")
    args = ap.parse_args(argv)

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    registered = proc.name == "mess3" and cfg["layers"] == 4
    if not registered and not args.selftest and not args.force_invalid:
        print(f"Experiment 6 is registered for mess3 with 4 layers (L1 "
              f"patch); this is {proc.name} with {cfg['layers']}. Use "
              "--selftest or --force-invalid.")
        return
    overridden = [k for k, v in REGISTERED.items()
                  if getattr(args, k) != v]
    if overridden and not args.selftest and not args.force_invalid:
        print("Experiment 6's loop/protocol parameters are registered; "
              f"overridden here: {overridden}. Use --force-invalid for an "
              "explicitly exploratory run.")
        return
    if overridden:
        print(f"NOTE: EXPLORATORY RUN — non-registered parameters "
              f"{overridden}; verdicts below are NOT Experiment 6.\n")
    if args.seed != 0:
        print(f"NOTE: seed {args.seed} != 0 — a seed-robustness rerun of the "
              "registered design.\n")
    L, burn, V, m = cfg["seq_len"], cfg["burn_in"], proc.V, args.m
    d = cfg["d_model"]

    model = GPT(GPTConfig(vocab=V, seq_len=L, d_model=d,
                          n_layers=cfg["layers"]))
    model.load_state_dict(torch.load(os.path.join(args.outdir, "model.pt"),
                                     map_location="cpu"))
    model.eval()

    # ----- validity gate (Experiment-5 estimator, enforced) ------------------
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
    p5 = gap_opt <= 0.005
    print(f"validity gate: gap-to-optimal {gap_opt:+.4f} nats — "
          f"{'PASS' if p5 else 'FAIL'}\n")
    if not p5 and not args.selftest and not args.force_invalid:
        print("exiting: validity gate failed (P5).")
        return

    # disjoint pair sets: discovery (own seed) and evaluation (Experiment-5
    # seed, for comparability of the full/pca/pls/rand numbers).
    disc = PairSet(model, proc, cfg, args.pairs_disc, m, args.seed + 111, 800)
    ev = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777, 800)

    # ----- self-checks (Experiment-4/5 set, at L1, on the eval pairs) --------
    t0, idx0 = ev.groups[0]
    s = idx0[:min(64, len(idx0))]
    gl = slice(0, len(s))
    q_un, _ = chain_probs(model, ev.Xc_tgt[t0][gl], LAYER, None, t0, m, V)
    q_noop, _ = chain_probs(model, ev.Xc_tgt[t0][gl], LAYER,
                            ev.pref_tgt[t0][gl], t0, m, V)
    assert np.array_equal(q_noop, q_un), "no-op patch changed chain probs"
    emb_src = stream_to(model, torch.from_numpy(ev.Xe[ev.b[s]]), 0)[:, :t0 + 1]
    q_l0, _ = chain_probs(model, ev.Xc_tgt[t0][gl], 0, emb_src, t0, m, V)
    q_srcrun, _ = chain_probs(model, ev.Xc_src[t0][gl], LAYER, None, t0, m, V)
    assert np.allclose(q_l0, q_srcrun, atol=1e-9), \
        "layer-0 prefix swap != source run"
    q_full, _ = chain_probs(model, ev.Xc_tgt[t0][gl], LAYER,
                            ev.pref_src[t0][gl], t0, m, V)
    assert np.allclose(marginal(q_full, V, 1, m), marginal(q_srcrun, V, 1, m),
                       atol=1e-9), "pre-scope full patch m=1 != source m=1"
    s_chk = stream_to(model, torch.from_numpy(ev.Xc_tgt[t0][0, :2]), LAYER)
    assert torch.allclose(s_chk[0, :t0 + 1], s_chk[1, :t0 + 1], atol=1e-6), \
        "prefix stream depends on continuation tokens"
    print("self-checks passed: no-op, layer-0 known answer, pre-full m=1 "
          "identity, causality\n")
    if args.selftest:
        return

    # ----- the CEGAR loop (observable objective only) -------------------------
    print(f"=== Experiment 6: interventional discovery | {proc.name} | "
          f"patch L{LAYER} (input to block {LAYER + 1}) | "
          f"{args.pairs_disc} discovery / {args.pairs_eval} evaluation pairs "
          f"| eps_gain {args.eps_gain}, eps_drop {args.eps_drop}, "
          f"k_max {args.k_max} ===\n")

    q_src_d = disc.run(model, None, src_side=True)
    q_un_d = disc.run(model, None)
    q_full_d = disc.run(model, np.eye(d))
    D0 = float(kl_rows(q_src_d, q_un_d).mean())
    Dfull = float(kl_rows(q_src_d, q_full_d).mean())
    c_obs = lambda q: (D0 - float(kl_rows(q_src_d, q).mean())) / (D0 - Dfull)
    # registered invariants of the observable scale (doc'd as self-checks):
    # the full patch must strictly improve on no patch, and must sit at
    # closure 1 — guards the denominator against future code drift.
    assert D0 > Dfull, "observable scale degenerate: D0 <= D_full"
    assert abs(c_obs(q_full_d) - 1.0) < 1e-12, "c_obs(full) != 1"
    print(f"observable refs (model-vs-model, m={m} joint): D0 {D0:.5f}, "
          f"D_full {Dfull:.5f} (invariants: D0 > D_full, c_obs(full) = 1)\n")

    def mined_direction(Q, weights):
        M = np.zeros((d, d))
        for t, idx in disc.groups:
            delta = (disc.pref_src[t] - disc.pref_tgt[t]).double().numpy()
            if Q.shape[1]:
                delta = delta - (delta @ Q) @ Q.T
            M += np.einsum("ipa,ipb,i->ab", delta, delta, weights[idx])
        w, vecs = np.linalg.eigh(M)
        v = vecs[:, -1]
        if Q.shape[1]:                       # exact re-orthogonalization
            v = v - Q @ (Q.T @ v)
        v /= np.linalg.norm(v)
        return v

    Q = np.zeros((d, 0))
    q_cur, c_cur = q_un_d, 0.0
    print("CEGAR trajectory (observable closure on discovery pairs):")
    print(f"  k=0: c_obs {c_cur:.1%}")
    converged = False
    while Q.shape[1] < args.k_max:
        v = mined_direction(Q, kl_rows(q_src_d, q_cur))
        assert abs(np.linalg.norm(v) - 1) < 1e-9 and (
            Q.shape[1] == 0 or np.abs(Q.T @ v).max() < 1e-9), \
            "loop invariant: proposal must be unit-norm and orthogonal"
        Q_try = np.hstack([Q, v[:, None]])
        q_try = disc.run(model, Q_try @ Q_try.T)
        c_try = c_obs(q_try)
        gain = c_try - c_cur
        if gain < args.eps_gain:
            print(f"  k={Q_try.shape[1]}: c_obs {c_try:.1%} "
                  f"(gain {gain:+.1%} < eps_gain) -> STOP, revert")
            converged = True
            break
        Q, q_cur, c_cur = Q_try, q_try, c_try
        print(f"  k={Q.shape[1]}: c_obs {c_cur:.1%} (gain {gain:+.1%}) "
              "-> accept")
    if not converged:
        print(f"  k_max = {args.k_max} reached: NON-CONVERGENT (typed "
              "failure; P6 fails)")

    # coarsen: drop directions worth < eps_drop of observable closure
    changed = True
    while changed and Q.shape[1] > 1:
        changed = False
        for j in range(Q.shape[1]):
            Qj = np.delete(Q, j, axis=1)
            cj = c_obs(disc.run(model, Qj @ Qj.T))
            if c_cur - cj < args.eps_drop:
                print(f"  coarsen: dropped direction {j + 1} "
                      f"(cost {c_cur - cj:+.2%})")
                Q, c_cur, changed = Qj, cj, True
                break
    k_star = Q.shape[1]
    print(f"\nfixed point: k* = {k_star}, observable closure "
          f"c_obs = {c_cur:.1%}\n")

    # ----- evaluation (exact targets; beliefs used here ONLY) -----------------
    # control bases at L1, Experiment-5 discovery protocol
    rng_d = np.random.default_rng(args.seed + 555)
    Xd = proc.sample(args.basis_seqs, L, rng_d)
    Sd = stream_to(model, torch.from_numpy(Xd), LAYER).double().numpy()
    keep = np.arange(burn, L - 1)
    Gd = np.concatenate([proc.mgram_table(proc.beliefs_along(row)[keep], m)
                         for row in Xd])
    Rd = center_by_position(Sd[:, keep].reshape(-1, d), np.tile(keep, len(Xd)),
                            np.ones(len(Xd) * len(keep), dtype=bool))
    pls = CompletionPLS(Rd, Gd)
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        W_tok = model.tok.weight.double().numpy()          # (V, d)
    bases = {
        "disc": Q,
        "pca": PCAAbstraction(Rd).Vt[:k_star].T,
        "pls": orthonormal(pls.whiten @ pls.U[:, :k_star]),
        "rand": orthonormal(rng.standard_normal((d, k_star))),
        "emb": orthonormal(W_tok.T)[:, :min(k_star, V)],
    }

    q0_e = ev.run(model, None)
    rows_f = kl_by_horizon(q0_e, ev.p_tgt3, V, m)
    rows_g = kl_by_horizon(q0_e, ev.p_src3, V, m)
    floor = {mm: float(rows_f[mm].mean()) for mm in rows_f}
    gapm = {mm: float(rows_g[mm].mean()) for mm in rows_g}
    ms = list(range(1, m + 1))

    def true_closure(P):
        rows_t = kl_by_horizon(ev.run(model, P), ev.p_src3, V, m)
        cl = {mm: (gapm[mm] - float(rows_t[mm].mean()))
              / (gapm[mm] - floor[mm]) for mm in ms}
        by_pos = {t: (float(rows_g[m][idx].mean())
                      - float(rows_t[m][idx].mean()))
                  / (float(rows_g[m][idx].mean())
                     - float(rows_f[m][idx].mean()))
                  for t, idx in ev.groups}
        return cl, by_pos

    print("evaluation on disjoint pairs, exact-target closures "
          "(pooled m=1..3 | per-position at m=3):")
    closures, stab = {}, {}
    conds = [("full", np.eye(d))] + [(f, B @ B.T) for f, B in bases.items()]
    for name, P in conds:
        cl, bp = true_closure(P)
        closures[name], stab[name] = cl, bp
        print(f"  {name:>5} (k={d if name == 'full' else bases[name].shape[1]})"
              + "  " + "  ".join(f"m={mm}: {cl[mm]:>6.1%}" for mm in ms)
              + "   | " + "  ".join(f"t={t}: {bp[t]:.1%}"
                                    for t, _ in ev.groups))

    print("\nnested closure(k) of the discovered basis (pooled m=3):")
    for k in range(1, k_star + 1):
        Qk = Q[:, :k]
        cl, _ = true_closure(Qk @ Qk.T)
        print(f"  k={k}: {cl[m]:.1%}")

    print("\nprincipal angles (deg) between discovered subspace and controls:")
    for f in ("pca", "pls", "emb"):
        ang = principal_angles_deg(Q, bases[f])
        print(f"  disc vs {f:>4}: " + ", ".join(f"{a:.1f}" for a in ang))

    # ----- verdicts (experiments/6-interventional-discovery.md) ---------------
    print("\nverdicts:")
    p1 = k_star <= 4 and closures["disc"][m] >= 0.90 * closures["full"][m]
    print(f"  P1 k* <= 4 and disc >= 90% of full (pooled m={m}): k*={k_star},"
          f" {closures['disc'][m]:.1%} vs full {closures['full'][m]:.1%} — "
          f"{'HOLDS' if p1 else 'FAILS'}")
    p2 = closures["disc"][m] - closures["pls"][m] >= 0.50
    print(f"  P2 disc - pls >= 50 points: "
          f"{closures['disc'][m] - closures['pls'][m]:+.1%} — "
          f"{'HOLDS' if p2 else 'FAILS'}")
    p3 = closures["disc"][m] >= closures["pca"][m] - 0.02
    print(f"  P3 disc within 2 points of pca: disc {closures['disc'][m]:.1%} "
          f"vs pca {closures['pca'][m]:.1%} — {'HOLDS' if p3 else 'FAILS'}")
    p4 = all(closures["rand"][mm] <= 0.25 for mm in ms)
    print(f"  P4 rand <= 25% everywhere: {'HOLDS' if p4 else 'FAILS'}")
    print(f"  P5 validity gate: {'HOLDS' if p5 else 'FAILS'}")
    p6 = converged and k_star <= 4
    print(f"  P6 loop converges at k* <= 4: {'HOLDS' if p6 else 'FAILS'}")
    p7 = abs(c_cur - closures["disc"][m]) <= 0.10
    print(f"  P7 observable/exact agreement: c_obs {c_cur:.1%} vs exact "
          f"{closures['disc'][m]:.1%} — {'HOLDS' if p7 else 'FAILS'}")

    # ----- plot ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(closures)
        ax.bar(range(len(names)), [closures[f][m] for f in names])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{f}\nk={d if f == 'full' else bases[f].shape[1]}"
                            for f in names], fontsize=8)
        ax.axhline(0.9 * closures["full"][m], ls="--", c="r",
                   label="P1 bar (90% of full)")
        ax.set_ylabel(f"exact-target closure, pooled m={m}")
        ax.set_title(f"{proc.name}: interventional discovery at L{LAYER}, "
                     f"k*={k_star}")
        ax.legend(fontsize=8)
        p = os.path.join(args.outdir, "experiment6.png")
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plotting skipped: {e})")


if __name__ == "__main__":
    main()
