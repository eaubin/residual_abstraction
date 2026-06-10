"""
depth.py — Experiment 5: a depth profile of patch persistence.

CONTEXT (see experiments/5-depth-profile.md, the pre-registration, committed
before training or running). Experiment 4 found that at the single interior
point of a 2-layer model the stream is a per-position SUMMARY, not
propagated STATE (per-step incremental closure 12.5% / 0.0% at steps 2/3):
future positions re-derived their predictive state from raw tokens below
the patch. With one block below the patch, "below" was nearly the whole
belief-synthesis pathway — so the honest follow-up is a PROFILE on a deeper
model: patch the prefix-wide full/pls/pca/rand interchange at the input to
every interior block of a 4-layer Mess3 model and ask at which depth, if
any, patched content is carried forward because the re-derivation pathway
below it is too shallow to bypass it.

Headline statistic (the Experiment-4 lesson): the PER-STEP INCREMENTAL
closure of the full prefix-wide patch, per patch layer — pooled multi-token
closures inherit the ~100% first step and flatter the patch. Coherence at
t+1 is measured per layer in the ORTHONORMALIZED unemb-pullback coordinates
(registered basis; lesson from the Experiment-4 review), with the
logit-weighted pullback and final-layer pls coordinates secondary.

The evaluation protocol (three positions pooled, exact m=1..3 chains over
all 27 continuations, honesty constraints, self-checks) is carried forward
from midstream.py unchanged; the low-level machinery is imported from
there. P1-P6 thresholds, including the validity gate on training adequacy
(P5) and the declared 2-point slack in "weakly decreasing", live in the
pre-registration; the verdict logic below implements them.

Run: python3 train.py --process mess3 --layers 4 --outdir out/mess3-L4
     python3 depth.py --outdir out/mess3-L4
`--selftest` exits after the known-answer checks and runs fine against the
existing 2-layer model (machinery validation; the profile needs >= 3
interior layers to be informative).

RESULTS (see experiments/5-depth-profile.md): P1-P6 ALL HOLD. The stream IS
state at L1 (step-2/3 incremental closure 93.7%/91.0%, coherence 94.8% —
never-state refuted), transitional at L2 (52.5%/12.4%), and at L3 the
incremental closures go NEGATIVE (-29.7%/-83.7%): a late patch creates
mixed-provenance state that predicts the source's continuation WORSE than
the unpatched target run — a new typed failure (state interference) that
single-step interchange scores cannot see. The scale lesson held at every
depth: pls k=2 is weak relative to pca/full everywhere — near-empty at
L1/L2 (3-9%), 35-45% at L3 — while pca k=2 ~ full; the rise of pls closure
with depth is the echo subspace gradually aligning with the readout channel
as the stream approaches the unembedding.
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/mess3-L4",
                    help="dir with cache.npz, model.pt, config.json")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--disc-seqs", type=int, default=800)
    ap.add_argument("--eval-seqs", type=int, default=800)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force-invalid", action="store_true",
                    help="proceed past a failed validity gate or a "
                    "non-registered model config (results are then "
                    "explicitly exploratory, not Experiment 5)")
    args = ap.parse_args(argv)

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    if cfg["layers"] < 2:
        print(f"{proc.name}: no interior stream point in a "
              f"{cfg['layers']}-layer model.")
        return
    # The registration is specific: Mess3, 4 layers. Anything else may only
    # run the machinery self-checks (or be explicitly forced as exploratory).
    registered = proc.name == "mess3" and cfg["layers"] == 4
    if not registered and not args.selftest and not args.force_invalid:
        print(f"Experiment 5 is registered for mess3 with 4 layers; this is "
              f"{proc.name} with {cfg['layers']}. Use --selftest for "
              "machinery validation or --force-invalid for an exploratory "
              "(non-Experiment-5) run.")
        return
    L, burn, V, m = cfg["seq_len"], cfg["burn_in"], proc.V, args.m
    d = cfg["d_model"]
    interior = list(range(1, cfg["layers"]))

    model = GPT(GPTConfig(vocab=V, seq_len=L, d_model=d,
                          n_layers=cfg["layers"]))
    model.load_state_dict(torch.load(os.path.join(args.outdir, "model.pt"),
                                     map_location="cpu"))
    model.eval()

    # ----- P5 validity gate: is the model trained to (near) optimality? ------
    # 2000 sequences: with 400 the estimator's noise (~0.004 nats) is
    # comparable to the 0.005 threshold — observed when the well-trained
    # 2-layer model (train-time gap +0.0016) read +0.0050 during selftest.
    # Token-weighted accumulation (an unweighted mean of batch means would
    # overweight the smaller final batch — flagged in review).
    Xg = proc.sample(2000, L, np.random.default_rng(args.seed + 999))
    with torch.no_grad():
        tot, cnt = 0.0, 0
        for i in range(0, len(Xg), 256):
            logits = model(torch.from_numpy(Xg[i:i + 256]))
            tgt = torch.from_numpy(Xg[i:i + 256, 1:]).reshape(-1)
            tot += F.cross_entropy(logits[:, :-1].reshape(-1, V), tgt,
                                   reduction="sum").item()
            cnt += tgt.numel()
        nll = tot / cnt
    gap_opt = nll - cfg["optimal_nll"]
    p5 = gap_opt <= 0.005
    print(f"validity gate: NLL/token {nll:.4f}, gap-to-optimal "
          f"{gap_opt:+.4f} nats — {'PASS' if p5 else 'FAIL (retrain longer; '
          'results below are NOT interpretable)'}\n")
    # Enforce the pre-registration rather than trusting the reader to: a
    # failed gate means retrain, not interpret.
    if not p5 and not args.selftest and not args.force_invalid:
        print("exiting: validity gate failed (P5). Retrain the model "
              "(longer / more steps) and rerun; --force-invalid overrides "
              "for explicitly exploratory runs only.")
        return

    # ----- evaluation pairs (Experiment-4 protocol, unchanged) ---------------
    rng_e = np.random.default_rng(args.seed + 777)
    Xe = proc.sample(args.eval_seqs, L, rng_e)
    B = np.stack([proc.beliefs_along(row) for row in Xe])
    n = args.pairs
    a = rng_e.integers(0, len(Xe), n)
    b = rng_e.integers(0, len(Xe), n)
    b = np.where(b == a, (b + 1) % len(Xe), b)
    ts = np.unique(np.linspace(burn + 4, L - 1 - m - 4, 3).astype(int))
    t_of = ts[np.arange(n) % len(ts)]
    groups = [(int(t), np.where(t_of == t)[0]) for t in ts]
    p_src3 = proc.mgram_table(B[b, t_of], m)
    p_tgt3 = proc.mgram_table(B[a, t_of], m)
    conts = np.array(list(product(range(V), repeat=m)))
    C = len(conts)
    Xc_tgt, Xc_src = {}, {}
    for t, idx in groups:
        for store, seqs in ((Xc_tgt, a), (Xc_src, b)):
            xc = np.repeat(Xe[seqs[idx]][:, None, :], C, axis=1).copy()
            xc[:, :, t + 1:t + 1 + m] = conts[None, :, :]
            store[t] = xc

    # per-layer streams and per-layer discovered bases
    S = {l: stream_to(model, torch.from_numpy(Xe), l) for l in interior}
    rng_d = np.random.default_rng(args.seed + 555)
    Xd = proc.sample(args.disc_seqs, L, rng_d)
    keep = np.arange(burn, L - 1)
    Gd = np.concatenate([proc.mgram_table(proc.beliefs_along(row)[keep], m)
                         for row in Xd])
    rng = np.random.default_rng(args.seed)
    projs = {}                                   # (layer, family) -> projector
    for l in interior:
        Sd = stream_to(model, torch.from_numpy(Xd), l).double().numpy()
        Rd = center_by_position(Sd[:, keep].reshape(-1, d),
                                np.tile(keep, len(Xd)),
                                np.ones(len(Xd) * len(keep), dtype=bool))
        pls = CompletionPLS(Rd, Gd)
        Qs = {"full": np.eye(d),
              "pls": orthonormal(pls.whiten @ pls.U[:, :args.k]),
              "pca": PCAAbstraction(Rd).Vt[:args.k].T,
              "rand": orthonormal(rng.standard_normal((d, args.k)))}
        for f, Qf in Qs.items():
            projs[(l, f)] = Qf @ Qf.T

    def run_all(layer, prefix_of, src_side=False):
        q = np.empty((n, C))
        r = np.empty((n, C, d))
        for t, idx in groups:
            X = Xc_src[t] if src_side else Xc_tgt[t]
            qg, rg = chain_probs(model, X, layer, prefix_of(layer, t, idx),
                                 t, m, V)
            q[idx], r[idx] = qg, rg
        return q, r

    unpatched = lambda layer, t, idx: None

    def patched(P):
        def f(layer, t, idx):
            pt = S[layer][torch.from_numpy(a[idx])][:, :t + 1].double().numpy()
            psrc = S[layer][torch.from_numpy(b[idx])][:, :t + 1].double().numpy()
            return torch.from_numpy(pt + (psrc - pt) @ P).float()
        return f

    # ----- self-checks at the earliest interior layer (Experiment-4 set) -----
    l0, (t0, idx0) = interior[0], groups[0]
    s = idx0[:min(64, len(idx0))]
    gl = slice(0, len(s))
    pref_t = S[l0][torch.from_numpy(a[s])][:, :t0 + 1]
    pref_s = S[l0][torch.from_numpy(b[s])][:, :t0 + 1]
    q_un, _ = chain_probs(model, Xc_tgt[t0][gl], l0, None, t0, m, V)
    q_noop, _ = chain_probs(model, Xc_tgt[t0][gl], l0, pref_t, t0, m, V)
    assert np.array_equal(q_noop, q_un), "no-op patch changed chain probs"
    emb_src = stream_to(model, torch.from_numpy(Xe[b[s]]), 0)[:, :t0 + 1]
    q_l0, _ = chain_probs(model, Xc_tgt[t0][gl], 0, emb_src, t0, m, V)
    q_srcrun, _ = chain_probs(model, Xc_src[t0][gl], l0, None, t0, m, V)
    assert np.allclose(q_l0, q_srcrun, atol=1e-9), \
        "layer-0 prefix swap != source run"
    q_full, _ = chain_probs(model, Xc_tgt[t0][gl], l0, pref_s, t0, m, V)
    assert np.allclose(marginal(q_full, V, 1, m), marginal(q_srcrun, V, 1, m),
                       atol=1e-9), "pre-scope full patch m=1 != source m=1"
    s_chk = stream_to(model, torch.from_numpy(Xc_tgt[t0][0, :2]), l0)
    assert torch.allclose(s_chk[0, :t0 + 1], s_chk[1, :t0 + 1], atol=1e-6), \
        "prefix stream depends on continuation tokens"
    print("self-checks passed: no-op, layer-0 known answer, pre-full m=1 "
          "identity, causality\n")
    if args.selftest:
        return

    # ----- the experiment ------------------------------------------------------
    print(f"=== Experiment 5: depth profile | {proc.name} | "
          f"{cfg['layers']} layers | k = {args.k} | {n} pairs at "
          f"t in {[int(t) for t in ts]} | horizons m = 1..{m} ===\n")

    q0, r_un_t1 = run_all(interior[0], unpatched)
    rows_f = kl_by_horizon(q0, p_tgt3, V, m)
    rows_g = kl_by_horizon(q0, p_src3, V, m)
    floor = {mm: float(rows_f[mm].mean()) for mm in rows_f}
    gapm = {mm: float(rows_g[mm].mean()) for mm in rows_g}
    print("unpatched reference, per horizon m: "
          + " | ".join(f"m={mm}: floor {floor[mm]:.5f}, gap {gapm[mm]:.5f}"
                       for mm in floor) + "\n")

    ms = list(range(1, m + 1))
    closures, inc, resid_full, by_pos = {}, {}, {}, {}
    print(f"{'patch':>9}  " + "  ".join(f"closure m={mm}" for mm in ms)
          + "   (pre scope; full/pls/pca/rand per layer)")
    for l in interior:
        for fam in ("full", "pls", "pca", "rand"):
            qp, r_t1 = run_all(l, patched(projs[(l, fam)]))
            rows_t = kl_by_horizon(qp, p_src3, V, m)
            cl = {mm: (gapm[mm] - float(rows_t[mm].mean()))
                  / (gapm[mm] - floor[mm]) for mm in ms}
            closures[(l, fam)] = cl
            by_pos[(l, fam)] = {
                t: (float(rows_g[m][idx].mean())
                    - float(rows_t[m][idx].mean()))
                / (float(rows_g[m][idx].mean())
                   - float(rows_f[m][idx].mean()))
                for t, idx in groups}
            if fam == "full":
                resid_full[l] = r_t1
                trf = {mm: gapm[mm] - cl[mm] * (gapm[mm] - floor[mm])
                       for mm in ms}
                inc[l] = {mm: ((gapm[mm] - gapm[mm - 1])
                               - (trf[mm] - trf[mm - 1]))
                          / ((gapm[mm] - gapm[mm - 1])
                             - (floor[mm] - floor[mm - 1]))
                          for mm in ms[1:]}
            print(f"L{l}/{fam:<5}  "
                  + "  ".join(f"{cl[mm]:>11.1%}" for mm in ms))

    # Registered stability diagnostic, carried forward from Experiment 4
    # (restored in review: the first run omitted it from the output).
    print(f"\nper-position closure at m={m} (stability across patch "
          "positions):")
    print(f"{'patch':>9}  " + "  ".join(f"t={t:<4}" for t, _ in groups))
    for key in closures:
        print(f"L{key[0]}/{key[1]:<5}  "
              + "  ".join(f"{by_pos[key][t]:>6.1%}" for t, _ in groups))

    print("\nper-step incremental closure of full/pre by patch layer "
          "(the headline; step 1 reproduces the source model run, ~100%):")
    print(f"{'layer':>6}  " + "  ".join(f"step {mm}" for mm in ms[1:]))
    for l in interior:
        print(f"{'L' + str(l):>6}  "
              + "  ".join(f"{inc[l][mm]:>6.1%}" for mm in ms[1:]))

    # ----- coherence per layer (registered basis: orthonormalized unemb) -----
    with torch.no_grad():
        Wu = model.head.weight.double().numpy()
        g_ln = model.ln_f.weight.double().numpy()
    M_unemb = (np.eye(d) - np.ones((d, d)) / d) @ (g_ln[:, None] * Wu.T)
    bases = [("unemb orthonormal (registered)", orthonormal(M_unemb)),
             ("unemb logit-weighted (secondary)", M_unemb)]
    cache_p = os.path.join(args.outdir, "cache.npz")
    if os.path.exists(cache_p):
        dc = np.load(cache_p)
        permc = np.random.default_rng(args.seed).permutation(len(dc["resid"]))
        n_tr = int(0.7 * len(permc))
        maskc = np.zeros(len(permc), dtype=bool); maskc[:n_tr] = True
        Rcc = center_by_position(dc["resid"][permc], dc["pos"][permc], maskc)
        pls_f = CompletionPLS(Rcc[:n_tr], dc["mgram"][permc][:n_tr])
        bases.append(("pls final-layer (secondary)",
                      pls_f.whiten @ pls_f.U[:, :args.k]))
    else:
        print("\n(final-layer pls coherence basis skipped: no cache.npz; "
              "regenerate with train.py --cache-only)")

    w_star = np.argmax(marginal(p_src3, V, 1, m), axis=1)
    cont_idx = w_star * V ** (m - 1)
    rows = np.arange(n)
    _, r_src_t1 = run_all(interior[0], unpatched, src_side=True)
    coh = {}
    print("\ncoherence at t+1 of full/pre by patch layer (fraction of pairs "
          "with patched state closer to the source run):")
    print(f"{'basis':>34}  " + "  ".join(f"L{l}" for l in interior))
    for bname, A in bases:
        fr = {}
        for l in interior:
            z_p = resid_full[l][rows, cont_idx] @ A
            z_s = r_src_t1[rows, cont_idx] @ A
            z_u = r_un_t1[rows, cont_idx] @ A
            fr[l] = float((np.linalg.norm(z_p - z_s, axis=1)
                           < np.linalg.norm(z_u - z_s, axis=1)).mean())
        if bname.endswith("(registered)"):
            coh = fr
        print(f"{bname:>34}  "
              + "  ".join(f"{fr[l]:.1%}" for l in interior))

    # ----- verdicts (thresholds: experiments/5-depth-profile.md) -------------
    print("\nverdicts:")
    weakly_dec = lambda vals: all(vals[i + 1] <= vals[i] + 0.02
                                  for i in range(len(vals) - 1))
    p1 = all(weakly_dec([inc[l][mm] for l in interior]) for mm in ms[1:])
    print(f"  P1 incremental closure weakly decreasing in depth: "
          f"{'HOLDS' if p1 else 'FAILS'}")
    p2 = inc[interior[0]][2] >= 0.50
    print(f"  P2 step-2 incremental closure at L{interior[0]} >= 50%: "
          f"{inc[interior[0]][2]:.1%} — {'HOLDS' if p2 else 'FAILS'}"
          + ("" if p2 else " (typed: NEVER-STATE if low at every layer)"))
    p3 = all(closures[(l, 'pca')][m] >= closures[(l, 'pls')][m]
             for l in interior)
    print(f"  P3 pca >= pls (pooled m={m}) at every layer: "
          f"{'HOLDS' if p3 else 'FAILS'}")
    p4 = all(closures[(l, 'rand')][mm] <= 0.25 for l in interior for mm in ms)
    print(f"  P4 rand <= 25% everywhere: {'HOLDS' if p4 else 'FAILS'}")
    print(f"  P5 validity gate (gap-to-optimal <= 0.005): "
          f"{'HOLDS' if p5 else 'FAILS — run not interpretable'}")
    p6 = weakly_dec([coh[l] for l in interior])
    print(f"  P6 coherence weakly decreasing in depth: "
          f"{'HOLDS' if p6 else 'FAILS'}")

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
        axes[0].set_ylabel("incremental closure / fraction")
        axes[0].set_title("persistence profile (full/pre)")
        axes[0].axhline(0, ls=":", c="gray"); axes[0].legend(fontsize=8)
        for fam in ("full", "pls", "pca", "rand"):
            axes[1].plot(interior, [closures[(l, fam)][m] for l in interior],
                         "o-", label=fam)
        axes[1].set_xlabel("patch layer"); axes[1].set_xticks(interior)
        axes[1].set_ylabel(f"pooled closure, m={m}")
        axes[1].set_title("pooled closure by family")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{proc.name}: depth profile, {cfg['layers']} layers, "
                     f"k={args.k}")
        p = os.path.join(args.outdir, "experiment5.png")
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plotting skipped: {e})")


if __name__ == "__main__":
    main()
