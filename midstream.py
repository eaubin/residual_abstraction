"""
midstream.py — Experiment 4: mid-stream persistent interventions and
coherence under extension.

CONTEXT (see experiments/4-midstream-interventions.md, the pre-registration,
committed before the first run). Experiment 3 intervened at the readout,
where the remaining computation is ln_f + unembedding and the architecture
hands you a (first-order) causal basis. Mid-stream is the case that matters
for real models: a patch at the input to the final block changes that
block's keys/values, so every later position reads it through attention —
the intervention PERSISTS — and no closed-form reading basis exists.

Two questions: (a) is a discovered subspace causally load-bearing for the
downstream COMPUTATION, not just the readout? (b) does the intervention
persist over autoregressive extension — the coherence/bisimulation condition
(roadmap item #2)?

DESIGN (Mess3 only — the Z1R model has 1 layer, hence no interior stream
point; declared in the pre-registration):

* Patch point: input to the final block, position-aligned pairs.
* Scopes: `pos` (patch position t only) and `pre` (patch all p <= t).
* Subspaces are RE-DISCOVERED at the patch point (the Experiment-2/3 bases
  live in the final-layer stream and cannot be assumed to transfer):
  pls / pca / rand at k=2 plus `full` (Q=I) and the pls complement (pre
  scope). Same honesty constraint: supervised on completions only.
* Horizons m = 1, 2, 3: the model's exact m-step completion distribution
  under the patch is computed by the chain rule over all V^m = 27
  teacher-forced continuations; m=1,2 are marginals of the m=3 joint.
  Targets are the source's exact belief-conditioned m-gram distributions.
  closure_m = (gap_m - transfer_m) / (gap_m - floor_m), as in Experiment 3.
* By construction, the pre-scope FULL patch at m=1 equals the source's
  next-token behavior exactly (everything the final block sees at position t
  is swapped) — used as self-check #3, not a result. Its m>=2 shortfall from
  100% measures how much completion-relevant information BYPASSES the patch
  layer (continuation positions re-reading unpatched token embeddings
  through block 1).
* Coherence (state-level): teacher-force the source's most likely next
  token; compare the final-layer state at position t+1 across patched /
  source / unpatched runs in the Experiment-2 final-layer PLS coordinates.

PRE-REGISTERED PREDICTIONS (P1-P6) and the new failure modes (attention
bypass, lower-path bypass, incoherence) are in the pre-registration file;
the verdict logic below implements those thresholds.

SELF-CHECKS (every invocation; --selftest exits after them): (1) no-op patch
reproduces unpatched chain probabilities bit-for-bit; (2) pre-scope full
swap at layer 0 (the embedding stream) reproduces the source run's chain
probabilities — a known-answer validation of the whole chain machinery;
(3) pre-scope full patch at the real patch point matches the source's
next-token distribution at m=1; (4) prefix states are independent of
continuation tokens (causal-mask sanity).
"""

import argparse
import json
import os
from itertools import product

import numpy as np
import torch

from abstraction import (CompletionPLS, PCAAbstraction, center_by_position,
                         kl_rows)
from model import GPT, GPTConfig
from processes import PROCESSES


def orthonormal(A):
    Q, _ = np.linalg.qr(A)
    return Q


def stream_to(model, idx, layer):
    """Residual stream entering blocks[layer] (layer=0: embeddings)."""
    with torch.no_grad():
        L = idx.shape[1]
        x = model.tok(idx) + model.pos(torch.arange(L))
        for blk in model.blocks[:layer]:
            x = blk(x)
    return x


def chain_run(model, idx, layer, prefix_state, t):
    """Forward pass with x[:, :t+1] at blocks[layer]'s input replaced by
    prefix_state. Returns (softmax probs (B, L, V), final residual (B, L, d)).
    prefix_state None = unpatched."""
    with torch.no_grad():
        L = idx.shape[1]
        x = model.tok(idx) + model.pos(torch.arange(L))
        for li, blk in enumerate(model.blocks):
            if li == layer and prefix_state is not None:
                x = x.clone()
                x[:, :t + 1] = prefix_state
            x = blk(x)
        probs = torch.softmax(model.head(model.ln_f(x)), dim=-1)
    return probs.double().numpy(), x.double().numpy()


def chain_probs(model, X_cont, layer, prefix_state, t, m, V):
    """Exact m-step completion distribution at position t under the patch.

    X_cont: (n_pairs, V**m, L) token arrays — target prefix + each of the
    V**m continuations spliced in at t+1..t+m. Returns (n_pairs, V**m) joint
    q(w_1..w_m) by the chain rule, plus the final residual at t+1 of row 0
    (used by the coherence metric). Batched over pairs x continuations.
    """
    n, C, L = X_cont.shape
    flat = X_cont.reshape(n * C, L)
    ps = None
    if prefix_state is not None:
        ps = prefix_state.repeat_interleave(C, dim=0)
    out = np.empty((n * C,))
    resid_t1 = np.empty((n, C, model.cfg.d_model))
    for i in range(0, n * C, 1024):
        sl = slice(i, min(i + 1024, n * C))
        probs, resid = chain_run(model, torch.from_numpy(flat[sl]), layer,
                                 None if ps is None else ps[sl], t)
        rows = np.arange(sl.stop - sl.start)
        q = np.ones(sl.stop - sl.start)
        for j in range(m):
            q *= probs[rows, t + j, flat[sl][:, t + 1 + j]]
        out[sl] = q
        resid_t1.reshape(n * C, -1)[sl] = resid[:, t + 1]
    return out.reshape(n, C), resid_t1


def closure_table(joint, p_src3, p_tgt3, V, m_max=3):
    """Marginalize the m=3 joint to m=1,2,3 and return per-m mean KLs
    (to source truth, to target truth)."""
    res = {}
    for m in range(1, m_max + 1):
        # marginal over the first m tokens: sum out trailing dims
        shp = (-1,) + (V,) * m_max
        qm = joint.reshape(shp).sum(axis=tuple(range(1 + m, 1 + m_max)))
        qm = qm.reshape(len(joint), -1)
        qm = qm / np.clip(qm.sum(axis=1, keepdims=True), 1e-30, None)
        pm_s = p_src3.reshape(shp).sum(axis=tuple(range(1 + m, 1 + m_max)))
        pm_t = p_tgt3.reshape(shp).sum(axis=tuple(range(1 + m, 1 + m_max)))
        res[m] = (float(kl_rows(pm_s.reshape(len(joint), -1), qm).mean()),
                  float(kl_rows(pm_t.reshape(len(joint), -1), qm).mean()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/mess3",
                    help="dir with cache.npz, model.pt, config.json")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--disc-seqs", type=int, default=800,
                    help="fresh sequences for mid-stream subspace discovery")
    ap.add_argument("--eval-seqs", type=int, default=800)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="run the known-answer self-checks and exit")
    args = ap.parse_args(argv)

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    if cfg["layers"] < 2:
        print(f"{proc.name}: 1-layer model has no interior stream point; "
              "Experiment 4 is declared for Mess3 only (see pre-registration).")
        return
    L, burn, V, m = cfg["seq_len"], cfg["burn_in"], proc.V, args.m
    layer = cfg["layers"] - 1                    # input to the final block

    model = GPT(GPTConfig(vocab=V, seq_len=L, d_model=cfg["d_model"],
                          n_layers=cfg["layers"]))
    model.load_state_dict(torch.load(os.path.join(args.outdir, "model.pt"),
                                     map_location="cpu"))
    model.eval()

    # ----- subspace discovery AT THE PATCH POINT (completions only) ----------
    rng_d = np.random.default_rng(args.seed + 555)
    Xd = proc.sample(args.disc_seqs, L, rng_d)
    Sd = stream_to(model, torch.from_numpy(Xd), layer).double().numpy()
    keep = np.arange(burn, L - 1)
    Rd = Sd[:, keep].reshape(-1, cfg["d_model"])
    pos_d = np.tile(keep, len(Xd))
    Gd = np.concatenate([proc.mgram_table(proc.beliefs_along(row)[keep], m)
                         for row in Xd])
    mask = np.ones(len(Rd), dtype=bool)
    Rdc = center_by_position(Rd, pos_d, mask)
    pls = CompletionPLS(Rdc, Gd)
    pca = PCAAbstraction(Rdc)
    rng = np.random.default_rng(args.seed)
    d = cfg["d_model"]
    Q = {
        "full": np.eye(d),
        "pls": orthonormal(pls.whiten @ pls.U[:, :args.k]),
        "pca": pca.Vt[:args.k].T,
        "rand": orthonormal(rng.standard_normal((d, args.k))),
    }

    # final-layer pls basis (Experiment-2 cache protocol) for the coherence
    # metric — evaluation machinery, not a patch condition.
    dc = np.load(os.path.join(args.outdir, "cache.npz"))
    Rc, Gc = dc["resid"], dc["mgram"]
    permc = np.random.default_rng(args.seed).permutation(len(Rc))
    n_tr = int(0.7 * len(Rc))
    maskc = np.zeros(len(Rc), dtype=bool); maskc[:n_tr] = True
    Rcc = center_by_position(Rc[permc], dc["pos"][permc], maskc)
    pls_f = CompletionPLS(Rcc[:n_tr], Gc[permc][:n_tr])
    A_final = pls_f.whiten @ pls_f.U[:, :args.k]

    # ----- evaluation pairs ---------------------------------------------------
    rng_e = np.random.default_rng(args.seed + 777)
    Xe = proc.sample(args.eval_seqs, L, rng_e)
    S_mid = stream_to(model, torch.from_numpy(Xe), layer)   # (N, L, d) fp32
    B = np.stack([proc.beliefs_along(row) for row in Xe])

    n = args.pairs
    a = rng_e.integers(0, len(Xe), n)
    b = rng_e.integers(0, len(Xe), n)
    b = np.where(b == a, (b + 1) % len(Xe), b)
    # one shared position for all pairs keeps the chain forwards batchable;
    # mid-sequence, past burn-in, with room for the m-token horizon.
    t = (burn + (L - 1 - m - burn) // 2)
    p_src3 = proc.mgram_table(B[b, t], m)
    p_tgt3 = proc.mgram_table(B[a, t], m)

    conts = np.array(list(product(range(V), repeat=m)))      # (V**m, m)
    C = len(conts)
    X_cont = np.repeat(Xe[a][:, None, :], C, axis=1).copy()  # (n, C, L)
    X_cont[:, :, t + 1:t + 1 + m] = conts[None, :, :]
    X_cont_src = np.repeat(Xe[b][:, None, :], C, axis=1).copy()
    X_cont_src[:, :, t + 1:t + 1 + m] = conts[None, :, :]

    prefix_tgt = S_mid[torch.from_numpy(a)][:, :t + 1]       # (n, t+1, d)
    prefix_src = S_mid[torch.from_numpy(b)][:, :t + 1]

    # symmetric projectors onto each condition's subspace; the complement is
    # I - P_pls directly (a QR basis of the singular matrix I - QQ^T would
    # silently return a FULL orthonormal basis — found by the no-op check).
    projs = {f: Q[f] @ Q[f].T for f in ("full", "pls", "pca", "rand")}
    projs["comp"] = np.eye(d) - projs["pls"]

    def patched_prefix(P, scope):
        delta = (prefix_src - prefix_tgt).double().numpy()
        out = prefix_tgt.double().numpy().copy()
        if scope == "pre":
            out += delta @ P
        else:                                                # position t only
            out[:, t] += delta[:, t] @ P
        return torch.from_numpy(out).float()

    # ----- self-checks --------------------------------------------------------
    sub = slice(0, min(64, n))
    q_un, _ = chain_probs(model, X_cont[sub], layer, None, t, m, V)
    # (1) no-op patch == unpatched, bitwise
    q_noop, _ = chain_probs(model, X_cont[sub], layer, prefix_tgt[sub], t, m, V)
    assert np.array_equal(q_noop, q_un), "no-op patch changed chain probs"
    # (2) layer-0 full prefix swap == source run (known answer)
    emb_src = stream_to(model, torch.from_numpy(Xe[b[sub]]), 0)[:, :t + 1]
    q_l0, _ = chain_probs(model, X_cont[sub], 0, emb_src, t, m, V)
    q_srcrun, _ = chain_probs(model, X_cont_src[sub], layer, None, t, m, V)
    assert np.allclose(q_l0, q_srcrun, atol=1e-9), \
        "layer-0 prefix swap != source run"
    # (3) pre-scope full patch at the patch point: m=1 == source's m=1
    q_full, _ = chain_probs(model, X_cont[sub], layer, prefix_src[sub], t, m, V)
    m1 = lambda q: q.reshape(-1, V, V ** (m - 1)).sum(axis=2)
    assert np.allclose(m1(q_full), m1(q_srcrun), atol=1e-9), \
        "pre-scope full patch m=1 != source next-token distribution"
    # (4) prefix states independent of continuation (causality)
    s_chk = stream_to(model, torch.from_numpy(X_cont[sub.start, :2]), layer)
    assert torch.allclose(s_chk[0, :t + 1], s_chk[1, :t + 1], atol=1e-6), \
        "prefix stream depends on continuation tokens"
    print("self-checks passed: no-op, layer-0 known answer, pre-full m=1 "
          "identity, causality\n")
    if args.selftest:
        return

    # ----- the experiment -----------------------------------------------------
    print(f"=== Experiment 4: mid-stream interventions | {proc.name} | "
          f"k = {args.k} | patch at input to block {layer + 1}/{cfg['layers']}"
          f" | {n} pairs at t = {t} | horizons m = 1..{m} ===\n")

    q0, r_un_t1 = chain_probs(model, X_cont, layer, None, t, m, V)
    base = closure_table(q0, p_src3, p_tgt3, V, m)
    floor = {mm: base[mm][1] for mm in base}
    gap = {mm: base[mm][0] for mm in base}
    print("unpatched reference, per horizon m: "
          + " | ".join(f"m={mm}: floor {floor[mm]:.5f}, gap {gap[mm]:.5f}"
                       for mm in base) + "\n")

    conditions = [(f, s) for f in ("full", "pls", "pca", "rand")
                  for s in ("pos", "pre")] + [("comp", "pre")]
    closures, resid_t1_store = {}, {}
    print(f"{'condition':>10}  " + "  ".join(f"closure m={mm}" for mm in base)
          + "   KL(tgt) m=3")
    for fam, scope in conditions:
        ps = patched_prefix(projs[fam], scope)
        qp, r_t1 = chain_probs(model, X_cont, layer, ps, t, m, V)
        tab = closure_table(qp, p_src3, p_tgt3, V, m)
        cl = {mm: (gap[mm] - tab[mm][0]) / (gap[mm] - floor[mm])
              for mm in tab}
        closures[(fam, scope)] = cl
        resid_t1_store[(fam, scope)] = r_t1
        print(f"{fam + '/' + scope:>10}  "
              + "  ".join(f"{cl[mm]:>11.1%}" for mm in cl)
              + f"   {tab[m][1]:.5f}")

    # ----- coherence (state-level, final-layer pls coordinates) --------------
    # teacher-forced continuation = source's most likely next token; that is
    # continuation index argmax over the m=1 marginal of p_src3.
    w_star = np.argmax(p_src3.reshape(n, V, -1).sum(axis=2), axis=1)
    cont_idx = w_star * V ** (m - 1)   # first continuation starting with w*
    rows = np.arange(n)
    _, r_src_t1 = chain_probs(model, X_cont_src, layer, None, t, m, V)
    alpha = lambda R: R[rows, cont_idx] @ A_final
    z_src = alpha(r_src_t1)
    z_un = alpha(r_un_t1)
    print("\ncoherence at t+1 (final-layer pls coords, teacher-forced w*):")
    coh = {}
    for cond in (("full", "pre"), ("pls", "pre")):
        z_p = alpha(resid_t1_store[cond])
        d_p = np.linalg.norm(z_p - z_src, axis=1)
        d_u = np.linalg.norm(z_un - z_src, axis=1)
        frac = float((d_p < d_u).mean())
        coh[cond] = frac
        print(f"  {cond[0]}/{cond[1]}: patched state closer to source-run "
              f"state than unpatched in {frac:.1%} of pairs "
              f"(median dist ratio {np.median(d_p / np.clip(d_u, 1e-12, None)):.3f})")

    # ----- verdicts against the pre-registration ------------------------------
    print("\nverdicts (thresholds from experiments/4-midstream-interventions.md):")
    p1 = all(closures[(f, "pre")][mm] >= closures[(f, "pos")][mm] - 0.02
             for f in ("full", "pls", "pca", "rand") for mm in range(1, m + 1))
    print(f"  P1 scope monotonicity: {'HOLDS' if p1 else 'FAILS'}")
    cf = closures[("full", "pos")]
    p2 = cf[1] > cf[2] > cf[3]
    print(f"  P2 pos-scope full decays with m ({cf[1]:.1%} -> {cf[2]:.1%} -> "
          f"{cf[3]:.1%}): {'HOLDS' if p2 else 'FAILS'}")
    p3 = all(closures[("pca", s)][mm] >= closures[("pls", s)][mm]
             for s in ("pos", "pre") for mm in range(1, m + 1))
    print(f"  P3 pca >= pls everywhere: {'HOLDS' if p3 else 'FAILS'}")
    p4 = all(closures[("rand", s)][mm] <= 0.25
             for s in ("pos", "pre") for mm in range(1, m + 1))
    print(f"  P4 rand <= 25% everywhere: {'HOLDS' if p4 else 'FAILS'}")
    c3 = closures[("full", "pre")][3]
    p5 = 0.80 <= c3 <= 0.99
    print(f"  P5 pre-scope full at m=3 in [80%, 99%] (bypass through block 1):"
          f" {c3:.1%} — {'HOLDS' if p5 else 'FAILS'}")
    p6 = coh[("full", "pre")] >= 0.90
    print(f"  P6 coherence (full/pre >= 90% of pairs): "
          f"{coh[('full', 'pre')]:.1%} — {'HOLDS' if p6 else 'FAILS'}")

    # ----- plot ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        ms = list(range(1, m + 1))
        for ax, scope in zip(axes, ("pos", "pre")):
            for fam in ("full", "pls", "pca", "rand"):
                ax.plot(ms, [closures[(fam, scope)][mm] for mm in ms],
                        "o-", label=fam)
            if scope == "pre":
                ax.plot(ms, [closures[("comp", "pre")][mm] for mm in ms],
                        "s--", label="comp(pls)")
            ax.set_xlabel("horizon m"); ax.set_xticks(ms)
            ax.set_title(f"scope: {scope}")
            ax.axhline(1.0, ls=":", c="gray"); ax.axhline(0.0, ls=":", c="gray")
        axes[0].set_ylabel("closure")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{proc.name}: mid-stream interchange, patch at block "
                     f"{layer + 1} input, k={args.k}")
        p = os.path.join(args.outdir, "experiment4.png")
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plotting skipped: {e})")


if __name__ == "__main__":
    main()
