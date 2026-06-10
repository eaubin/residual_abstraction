"""
midstream.py — Experiment 4: mid-stream persistent interventions and
coherence under extension.

CONTEXT (see experiments/4-midstream-interventions.md, the pre-registration,
committed — with one pre-run amendment, see below — before the first run).
Experiment 3 intervened at the readout, where the remaining computation is
ln_f + unembedding and the architecture hands you a (first-order) causal
basis. Mid-stream is the case that matters for real models: a patch at the
input to the final block changes that block's keys/values, so every later
position reads it through attention — the intervention PERSISTS — and no
closed-form reading basis exists.

Two questions: (a) is a discovered subspace causally load-bearing for the
downstream COMPUTATION, not just the readout? (b) does the intervention
persist over autoregressive extension — the coherence/bisimulation condition
(roadmap item #2)?

DESIGN (Mess3 only — the Z1R model has 1 layer, hence no interior stream
point; declared in the pre-registration):

* Patch point: input to the final block. Pairs are position-aligned and
  evaluated at THREE fixed positions spanning the usable range (pre-run
  amendment: the first draft used a single mid-sequence t, which is narrower
  than the registered "position-aligned intervention"; metrics are pooled
  across positions and per-position closures are reported as a stability
  check, since learned positional embeddings could make one position
  unrepresentative).
* Scopes: `pos` (patch position t only) and `pre` (patch all p <= t).
* Subspaces are RE-DISCOVERED at the patch point (the Experiment-2/3 bases
  live in the final-layer stream and cannot be assumed to transfer):
  pls / pca / rand at k=2 plus `full` (identity) and the pls complement
  (pre scope). Same honesty constraint: supervised on completions only.
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
the verdict logic below implements those thresholds against POOLED closures.

RESULTS (see experiments/4-midstream-interventions.md): P1-P5 HOLD, P6
FAILS. Pooled m>=2 closures inherit the (exact) m=1 component; the per-step
incremental closure of the full prefix-wide patch is 12.5% (step 2) and
0.0% (step 3), and the t+1 state barely tracks the source in either the
registered pls coordinates or the post-hoc unemb coordinates. Future
positions re-derive their predictive state from raw token embeddings below
the patch layer: the mid-stream residual is read as a per-position SUMMARY
and is not propagated as STATE. Sufficiency-style completeness and
coherence-as-state are different certificates.

SELF-CHECKS (every invocation; --selftest exits after them and touches
neither cache.npz nor anything else the experiment alone needs): (1) no-op
patch reproduces unpatched chain probabilities bit-for-bit; (2) pre-scope
full swap at layer 0 (the embedding stream) reproduces the source run's
chain probabilities — a known-answer validation of the whole chain
machinery; (3) pre-scope full patch at the real patch point matches the
source's next-token distribution at m=1; (4) prefix states are independent
of continuation tokens (causal-mask sanity).
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
    prefix_state (None = unpatched). Returns (softmax probs, final residual).
    """
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

    X_cont: (n_g, V**m, L) token arrays — prefix + each continuation spliced
    at t+1..t+m. Returns the (n_g, V**m) joint q(w_1..w_m) by the chain rule
    plus the final residual at position t+1 for every row.
    """
    n, C, L = X_cont.shape
    flat = X_cont.reshape(n * C, L)
    ps = None
    if prefix_state is not None:
        ps = prefix_state.repeat_interleave(C, dim=0)
    out = np.empty((n * C,))
    resid_t1 = np.empty((n * C, model.cfg.d_model))
    for i in range(0, n * C, 1024):
        sl = slice(i, min(i + 1024, n * C))
        probs, resid = chain_run(model, torch.from_numpy(flat[sl]), layer,
                                 None if ps is None else ps[sl], t)
        rows = np.arange(sl.stop - sl.start)
        q = np.ones(sl.stop - sl.start)
        for j in range(m):
            q *= probs[rows, t + j, flat[sl][:, t + 1 + j]]
        out[sl] = q
        resid_t1[sl] = resid[:, t + 1]
    return out.reshape(n, C), resid_t1.reshape(n, C, -1)


def marginal(arr, V, m, m_max):
    """(n, V**m_max) joint -> (n, V**m) marginal over the first m tokens."""
    shp = (-1,) + (V,) * m_max
    out = arr.reshape(shp).sum(axis=tuple(range(1 + m, 1 + m_max)))
    return out.reshape(len(arr), -1)


def kl_by_horizon(joint, p3, V, m_max):
    """Per-pair KL(p_m || q_m) for each horizon m. Returns {m: (n,)}."""
    out = {}
    for m in range(1, m_max + 1):
        qm = marginal(joint, V, m, m_max)
        qm = qm / np.clip(qm.sum(axis=1, keepdims=True), 1e-30, None)
        out[m] = kl_rows(marginal(p3, V, m, m_max), qm)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/mess3",
                    help="dir with cache.npz, model.pt, config.json")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--disc-seqs", type=int, default=800,
                    help="fresh sequences for mid-stream subspace discovery")
    ap.add_argument("--eval-seqs", type=int, default=800)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="run the known-answer self-checks and exit "
                    "(needs only model.pt + config.json, not cache.npz)")
    args = ap.parse_args(argv)

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    if cfg["layers"] < 2:
        print(f"{proc.name}: 1-layer model has no interior stream point; "
              "Experiment 4 is declared for Mess3 only (see pre-registration).")
        return
    L, burn, V, m = cfg["seq_len"], cfg["burn_in"], proc.V, args.m
    d = cfg["d_model"]
    layer = cfg["layers"] - 1                    # input to the final block

    model = GPT(GPTConfig(vocab=V, seq_len=L, d_model=d,
                          n_layers=cfg["layers"]))
    model.load_state_dict(torch.load(os.path.join(args.outdir, "model.pt"),
                                     map_location="cpu"))
    model.eval()

    # ----- subspace discovery AT THE PATCH POINT (completions only) ----------
    rng_d = np.random.default_rng(args.seed + 555)
    Xd = proc.sample(args.disc_seqs, L, rng_d)
    Sd = stream_to(model, torch.from_numpy(Xd), layer).double().numpy()
    keep = np.arange(burn, L - 1)
    Rd = Sd[:, keep].reshape(-1, d)
    pos_d = np.tile(keep, len(Xd))
    Gd = np.concatenate([proc.mgram_table(proc.beliefs_along(row)[keep], m)
                         for row in Xd])
    Rdc = center_by_position(Rd, pos_d, np.ones(len(Rd), dtype=bool))
    pls = CompletionPLS(Rdc, Gd)
    pca = PCAAbstraction(Rdc)
    rng = np.random.default_rng(args.seed)
    Q = {
        "full": np.eye(d),
        "pls": orthonormal(pls.whiten @ pls.U[:, :args.k]),
        "pca": pca.Vt[:args.k].T,
        "rand": orthonormal(rng.standard_normal((d, args.k))),
    }
    # symmetric projectors; the complement is I - P_pls directly (a QR basis
    # of the singular matrix I - QQ^T silently spans the FULL space — found
    # by the no-op self-check).
    projs = {f: Qf @ Qf.T for f, Qf in Q.items()}
    projs["comp"] = np.eye(d) - projs["pls"]

    # ----- evaluation pairs at three positions --------------------------------
    rng_e = np.random.default_rng(args.seed + 777)
    Xe = proc.sample(args.eval_seqs, L, rng_e)
    S_mid = stream_to(model, torch.from_numpy(Xe), layer)   # (N, L, d) fp32
    B = np.stack([proc.beliefs_along(row) for row in Xe])

    n = args.pairs
    a = rng_e.integers(0, len(Xe), n)
    b = rng_e.integers(0, len(Xe), n)
    b = np.where(b == a, (b + 1) % len(Xe), b)
    # three fixed positions spanning the usable band [burn, L-1-m] (pre-run
    # amendment: a single t is unrepresentative under learned positional
    # embeddings); pairs are assigned round-robin, metrics pooled, and
    # per-position closures reported as a stability check.
    ts = np.unique(np.linspace(burn + 4, L - 1 - m - 4, 3).astype(int))
    t_of = ts[np.arange(n) % len(ts)]
    groups = [(int(t), np.where(t_of == t)[0]) for t in ts]

    p_src3 = proc.mgram_table(B[b, t_of], m)
    p_tgt3 = proc.mgram_table(B[a, t_of], m)

    conts = np.array(list(product(range(V), repeat=m)))      # (V**m, m)
    C = len(conts)
    Xc_tgt, Xc_src, pref_tgt, pref_src = {}, {}, {}, {}
    for t, idx in groups:
        for store, seqs in ((Xc_tgt, a), (Xc_src, b)):
            xc = np.repeat(Xe[seqs[idx]][:, None, :], C, axis=1).copy()
            xc[:, :, t + 1:t + 1 + m] = conts[None, :, :]
            store[t] = xc
        pref_tgt[t] = S_mid[torch.from_numpy(a[idx])][:, :t + 1]
        pref_src[t] = S_mid[torch.from_numpy(b[idx])][:, :t + 1]

    def run_all(prefix_of, src_side=False):
        """prefix_of(t) -> patched prefix tensor or None. Pooled over groups,
        results in original pair order."""
        q = np.empty((n, C))
        r = np.empty((n, C, d))
        for t, idx in groups:
            X = Xc_src[t] if src_side else Xc_tgt[t]
            qg, rg = chain_probs(model, X, layer,
                                 prefix_of(t), t, m, V)
            q[idx], r[idx] = qg, rg
        return q, r

    def patched(P, scope):
        def f(t):
            delta = (pref_src[t] - pref_tgt[t]).double().numpy()
            out = pref_tgt[t].double().numpy().copy()
            if scope == "pre":
                out += delta @ P
            else:                                            # position t only
                out[:, t] += delta[:, t] @ P
            return torch.from_numpy(out).float()
        return f

    # ----- self-checks (cache-independent) ------------------------------------
    t0, idx0 = groups[0]
    s = idx0[:min(64, len(idx0))]
    gl = slice(0, len(s))                          # group-local rows
    q_un, _ = chain_probs(model, Xc_tgt[t0][gl], layer, None, t0, m, V)
    # (1) no-op patch == unpatched, bitwise
    q_noop, _ = chain_probs(model, Xc_tgt[t0][gl], layer, pref_tgt[t0][gl],
                            t0, m, V)
    assert np.array_equal(q_noop, q_un), "no-op patch changed chain probs"
    # (2) layer-0 full prefix swap == source run (known answer)
    emb_src = stream_to(model, torch.from_numpy(Xe[b[s]]), 0)[:, :t0 + 1]
    q_l0, _ = chain_probs(model, Xc_tgt[t0][gl], 0, emb_src, t0, m, V)
    q_srcrun, _ = chain_probs(model, Xc_src[t0][gl], layer, None, t0, m, V)
    assert np.allclose(q_l0, q_srcrun, atol=1e-9), \
        "layer-0 prefix swap != source run"
    # (3) pre-scope full patch at the patch point: m=1 == source's m=1
    q_full, _ = chain_probs(model, Xc_tgt[t0][gl], layer, pref_src[t0][gl],
                            t0, m, V)
    assert np.allclose(marginal(q_full, V, 1, m), marginal(q_srcrun, V, 1, m),
                       atol=1e-9), \
        "pre-scope full patch m=1 != source next-token distribution"
    # (4) prefix states independent of continuation (causality)
    s_chk = stream_to(model, torch.from_numpy(Xc_tgt[t0][0, :2]), layer)
    assert torch.allclose(s_chk[0, :t0 + 1], s_chk[1, :t0 + 1], atol=1e-6), \
        "prefix stream depends on continuation tokens"
    print("self-checks passed: no-op, layer-0 known answer, pre-full m=1 "
          "identity, causality\n")
    if args.selftest:
        return

    # ----- the experiment -----------------------------------------------------
    print(f"=== Experiment 4: mid-stream interventions | {proc.name} | "
          f"k = {args.k} | patch at input to block {layer + 1}/{cfg['layers']}"
          f" | {n} pairs at t in {[int(t) for t in ts]} | "
          f"horizons m = 1..{m} ===\n")

    q0, r_un_t1 = run_all(lambda t: None)
    rows_f = kl_by_horizon(q0, p_tgt3, V, m)               # floor rows
    rows_g = kl_by_horizon(q0, p_src3, V, m)               # gap rows
    floor = {mm: float(rows_f[mm].mean()) for mm in rows_f}
    gap = {mm: float(rows_g[mm].mean()) for mm in rows_g}
    print("unpatched reference, per horizon m: "
          + " | ".join(f"m={mm}: floor {floor[mm]:.5f}, gap {gap[mm]:.5f}"
                       for mm in floor) + "\n")

    conditions = [(f, sc) for f in ("full", "pls", "pca", "rand")
                  for sc in ("pos", "pre")] + [("comp", "pre")]
    closures, by_pos, resid_t1_store = {}, {}, {}
    ms = list(range(1, m + 1))
    print(f"{'condition':>10}  " + "  ".join(f"closure m={mm}" for mm in ms)
          + "   KL(tgt) m=3")
    for fam, scope in conditions:
        qp, r_t1 = run_all(patched(projs[fam], scope))
        rows_t = kl_by_horizon(qp, p_src3, V, m)
        rows_tgt = kl_by_horizon(qp, p_tgt3, V, m)
        cl = {mm: (gap[mm] - float(rows_t[mm].mean()))
              / (gap[mm] - floor[mm]) for mm in ms}
        closures[(fam, scope)] = cl
        resid_t1_store[(fam, scope)] = r_t1
        # per-position closure at the longest horizon (stability check)
        by_pos[(fam, scope)] = {
            t: (float(rows_g[m][idx].mean()) - float(rows_t[m][idx].mean()))
            / (float(rows_g[m][idx].mean()) - float(rows_f[m][idx].mean()))
            for t, idx in groups}
        print(f"{fam + '/' + scope:>10}  "
              + "  ".join(f"{cl[mm]:>11.1%}" for mm in ms)
              + f"   {float(rows_tgt[m].mean()):.5f}")

    print(f"\nper-position closure at m={m} (stability across patch "
          "positions):")
    print(f"{'condition':>10}  " + "  ".join(f"t={t:<4}" for t, _ in groups))
    for cond in conditions:
        print(f"{cond[0] + '/' + cond[1]:>10}  "
              + "  ".join(f"{by_pos[cond][t]:>6.1%}" for t, _ in groups))

    # ----- coherence (state-level, final-layer pls coordinates) --------------
    # The final-layer basis comes from the Experiment-2 cache, loaded lazily
    # here so --selftest (above) never needs the gitignored cache.npz
    # (regenerate it with train.py --cache-only if missing).
    dc = np.load(os.path.join(args.outdir, "cache.npz"))
    Rc, Gc = dc["resid"], dc["mgram"]
    permc = np.random.default_rng(args.seed).permutation(len(Rc))
    n_tr = int(0.7 * len(Rc))
    maskc = np.zeros(len(Rc), dtype=bool); maskc[:n_tr] = True
    Rcc = center_by_position(Rc[permc], dc["pos"][permc], maskc)
    pls_f = CompletionPLS(Rcc[:n_tr], Gc[permc][:n_tr])
    A_final = pls_f.whiten @ pls_f.U[:, :args.k]

    # teacher-forced continuation = source's most likely next token, i.e. the
    # first continuation whose w_1 = argmax of the m=1 marginal of p_src3.
    w_star = np.argmax(marginal(p_src3, V, 1, m), axis=1)
    cont_idx = w_star * V ** (m - 1)
    rows = np.arange(n)
    _, r_src_t1 = run_all(lambda t: None, src_side=True)
    # POST-HOC bases (added after the first run, which they do not alter):
    # Experiment 3 showed the final-layer pls coordinates are echo
    # coordinates; if coherence looks poor in them but good in the
    # causally-validated unemb-pullback coordinates, the incoherence is a
    # property of the echo, not of the model's behavioral state. Two
    # variants, because they answer with different metrics: the raw pullback
    # weights directions by how strongly they move LOGITS (sensitivity
    # metric), the orthonormalized one measures plain subspace distance.
    # Only if BOTH agree with the registered basis is the artifact ruled out.
    with torch.no_grad():
        Wu = model.head.weight.double().numpy()
        g_ln = model.ln_f.weight.double().numpy()
    M_unemb = (np.eye(d) - np.ones((d, d)) / d) @ (g_ln[:, None] * Wu.T)
    coh = {}
    for basis_name, A in (("pls (registered)", A_final),
                          ("unemb logit-weighted (post-hoc)", M_unemb),
                          ("unemb orthonormal (post-hoc)",
                           orthonormal(M_unemb))):
        alpha = lambda R: R[rows, cont_idx] @ A
        z_src, z_un = alpha(r_src_t1), alpha(r_un_t1)
        print(f"\ncoherence at t+1, final-layer {basis_name} coords "
              "(teacher-forced w*):")
        for cond in (("full", "pre"), ("pls", "pre")):
            z_p = alpha(resid_t1_store[cond])
            d_p = np.linalg.norm(z_p - z_src, axis=1)
            d_u = np.linalg.norm(z_un - z_src, axis=1)
            frac = float((d_p < d_u).mean())
            if basis_name.startswith("pls"):
                coh[cond] = frac           # P6 is judged on the registered basis
            print(f"  {cond[0]}/{cond[1]}: patched state closer to source-run "
                  f"state than unpatched in {frac:.1%} of pairs "
                  f"(median dist ratio "
                  f"{np.median(d_p / np.clip(d_u, 1e-12, None)):.3f})")

    # ----- verdicts against the pre-registration (pooled closures) -----------
    print("\nverdicts (thresholds from experiments/4-midstream-interventions.md):")
    p1 = all(closures[(f, "pre")][mm] >= closures[(f, "pos")][mm] - 0.02
             for f in ("full", "pls", "pca", "rand") for mm in ms)
    print(f"  P1 scope monotonicity: {'HOLDS' if p1 else 'FAILS'}")
    cf = closures[("full", "pos")]
    p2 = cf[1] > cf[2] > cf[3]
    print(f"  P2 pos-scope full decays with m ({cf[1]:.1%} -> {cf[2]:.1%} -> "
          f"{cf[3]:.1%}): {'HOLDS' if p2 else 'FAILS'}")
    p3 = all(closures[("pca", sc)][mm] >= closures[("pls", sc)][mm]
             for sc in ("pos", "pre") for mm in ms)
    print(f"  P3 pca >= pls everywhere: {'HOLDS' if p3 else 'FAILS'}")
    p4 = all(closures[("rand", sc)][mm] <= 0.25
             for sc in ("pos", "pre") for mm in ms)
    print(f"  P4 rand <= 25% everywhere: {'HOLDS' if p4 else 'FAILS'}")
    c3 = closures[("full", "pre")][m]
    p5 = 0.80 <= c3 <= 0.99
    print(f"  P5 pre-scope full at m={m} in [80%, 99%] (bypass through "
          f"block 1): {c3:.1%} — {'HOLDS' if p5 else 'FAILS'}")
    p6 = coh[("full", "pre")] >= 0.90
    print(f"  P6 coherence (full/pre >= 90% of pairs): "
          f"{coh[('full', 'pre')]:.1%} — {'HOLDS' if p6 else 'FAILS'}")
    # Diagnostic learned from the first run: pooled closure at m >= 2 mostly
    # INHERITS the (perfect) m=1 component. The per-step view isolates how
    # much of each step's NEW information the patch carries; if these are
    # small while m=1 is ~1 and P6 fails, future positions are re-deriving
    # their predictive state from raw tokens below the patch layer — the
    # stream is read as a per-position summary, not propagated as state.
    trf = {mm: gap[mm] - closures[("full", "pre")][mm]
           * (gap[mm] - floor[mm]) for mm in ms}
    inc = {mm: ((gap[mm] - gap[mm - 1]) - (trf[mm] - trf[mm - 1]))
           / ((gap[mm] - gap[mm - 1]) - (floor[mm] - floor[mm - 1]))
           for mm in ms[1:]}
    print("  per-step incremental closure, full/pre (step 1 reproduces the "
          "source MODEL RUN exactly, hence ~100% vs the source's true "
          "distribution): "
          + ", ".join(f"step {mm}: {inc[mm]:.1%}" for mm in ms[1:]))

    # ----- plot ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
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
