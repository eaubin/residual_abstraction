"""
intervene.py — Experiment 3: the interventional upgrade (interchange
interventions on discovered subspaces, scored behaviorally).

CONTEXT (see README.md and EXPERIMENTS.md). Experiments 1–2 established a
CORRELATIONAL claim: a 2-D completion-supervised subspace (whitened PLS) of
the residual stream is sufficient for completions and affinely identifies
with the true belief simplex (Mess3 affine R^2 0.9916 vs full-residual
reference 0.9917). Roadmap item #1 (AGENTS.md) is to upgrade this to a
CAUSAL claim in the sense of causal abstraction (Geiger, Icard, Potts et
al.): if the subspace IS the model's belief state, then transplanting a
source prefix's subspace component into a target prefix's residual must make
the model BEHAVE as if it had seen the source prefix — and transplanting the
orthogonal complement must change nothing.

DESIGN (stage 1, declared scope):

* Patch point: the final-layer residual (pre-ln_f) at position t — the same
  readout point every probe in Experiments 1–2 used. At this patch point the
  "rest of the network" is exactly ln_f + unembedding, so running the model
  forward from the patch is exact and cheap, and the behavioral readout is
  the model's OWN decoder, not a probe we fit. The trade-off, stated
  honestly: a final-layer patch at position t does not propagate to later
  positions (they attend to earlier-layer keys/values), so the behavioral
  horizon here is the NEXT-TOKEN distribution (m=1), computed exactly from
  beliefs. Completeness was always indexed by horizon; this is the m=1 line.
  The mid-stream persistent patch over multi-token horizons is its own
  pre-registered experiment: Experiment 4 (midstream.py).

* Interchange: for prefix pairs (target, source) at the SAME position t
  (so position-embedding content cancels in the difference),
      r' = r_tgt + Q Q^T (r_src - r_tgt)
  where Q is an orthonormal basis of the discovered k-D subspace. This is
  the minimal-norm edit making the subspace readout equal the source's
  (alpha(r') = alpha(r_src) exactly — asserted at runtime) while leaving the
  orthogonal complement untouched.

* Score: KL(p_src_true || q(r')) where p_src_true is the SOURCE's exact
  next-token distribution from its belief state, and q is the model's
  decoder on the patched residual. Reference points: the model's intrinsic
  floor KL(p_tgt || q(r_tgt)) and the unpatched gap KL(p_src || q(r_tgt)).
  closure = (gap - transfer) / (gap - floor): 1 = full interchange, 0 = the
  patch did nothing behaviorally.

* Subspaces compared (all discovered on the Experiment-2 cache, supervised
  on completions only — beliefs stay evaluation-only):
    pls   k-D whitened-PLS subspace (the Experiment-2 winner; the claim)
    pca   top-k principal subspace (the Experiment-2 loser; on Mess3 it is
          mostly current-token identity — a discriminating control)
    rand  random k-D subspace (no-information control)
    comp  orthogonal complement of pls (the "junk dimensions": patching ALL
          64-k of them should change nothing if pls is causally sufficient)

PRE-REGISTERED PREDICTIONS (also in experiments/3-readout-interventions.md,
committed before the first real run; thresholds chosen before seeing any
intervention numbers):
  P1  pls k=2 closure >= 0.90 on both processes.
  P2  (Mess3) pca k=2 closure <= pls closure - 0.05: the variance-dominant
      token plane transfers behavior measurably worse than the belief plane.
  P3  complement leak = (KL(p_tgt||q_comp) - floor)/(gap - floor) <= 0.05:
      the complement is causally inert.
  P4  rand k=2 closure <= 0.25.

TYPED FAILURE MODES this run can newly exhibit (extending the taxonomy):
  CORRELATIONAL-BUT-NOT-CAUSAL — alpha(r') equals the source readout by
      construction, yet behavior stays at the target (closure ~ 0): the
      decoder reads the information from somewhere else (redundant coding).
  OFF-MANIFOLD BREAKAGE — the patched residual leaves the reachable
      manifold and behavior lands far from BOTH source and target
      (KL to both above the unpatched gap).
  COMPLEMENT LEAK — the "junk" complement moves behavior: sufficiency of
      the subspace for READOUT did not imply causal localization.

Self-checks run before the experiment every time (the constructed-answer
discipline of AGENTS.md): a no-op patch (source = target) must reproduce the
unpatched distribution bit-for-bit, and a full-space patch (Q = I) must
reproduce the source's unpatched distribution exactly.

RESULTS (see experiments/3-readout-interventions.md): P1–P3 FAILED, P4
held. The pre-registered prediction was wrong in the most instructive way:
the PLS subspace is CORRELATIONAL-BUT-NOT-CAUSAL at this patch point
(closure 63% Mess3, ~0% Z1R, flat in k on Mess3), the high-variance PCA
plane carries MORE causal weight, and the post-hoc `unemb` family — the
unembedding row space pulled back through a LINEARIZED ln_f, a first-order
approximation to what the decoder reads — closes 100.0% at k = V on both
processes (the empirical validation that the linearization captures the
channel on these residuals; it is not exact, see the comment at its
construction). Decode-sufficiency under standardized probes is scale-blind;
causal load-bearing is not. Interventional scoring must enter the discovery
loop, not just the evaluation.
"""

import argparse
import json
import os

import numpy as np
import torch

from abstraction import (CompletionPLS, PCAAbstraction, center_by_position,
                         kl_rows)
from model import GPT, GPTConfig
from processes import PROCESSES


def orthonormal(A):
    """Orthonormal basis (d, k) for the column space of A."""
    Q, _ = np.linalg.qr(A)
    return Q


def decoder(model):
    """The rest of the network downstream of the patch point: ln_f + head."""
    def q_of(resid_np):
        with torch.no_grad():
            x = torch.from_numpy(resid_np.astype(np.float32))
            logits = model.head(model.ln_f(x))
            return torch.softmax(logits, dim=-1).double().numpy()
    return q_of


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/z1r",
                    help="dir with cache.npz, model.pt, config.json")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--eval-seqs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    name, L, burn = proc.name, cfg["seq_len"], cfg["burn_in"]

    # ----- discovery on the Experiment-2 cache (same protocol as compare.py:
    # same seed, same split, per-position centering; completions-only) -------
    d = np.load(os.path.join(args.outdir, "cache.npz"))
    R, G = d["resid"], d["mgram"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(R))
    R, G = R[perm], G[perm]
    n_tr = int(0.7 * len(R))
    mask = np.zeros(len(R), dtype=bool); mask[:n_tr] = True
    Rc = center_by_position(R, d["pos"][perm], mask)

    pls = CompletionPLS(Rc[:n_tr], G[:n_tr])
    pca = PCAAbstraction(Rc[:n_tr])
    A_pls = pls.whiten @ pls.U[:, :args.k]          # (d, k) in residual space
    dmodel = R.shape[1]
    Q = {
        "pls": orthonormal(A_pls),
        "pca": pca.Vt[:args.k].T,
        "rand": orthonormal(rng.standard_normal((dmodel, args.k))),
    }
    # POST-HOC family, added after the pre-registered k=2 runs (which it does
    # not alter — same seed reproduces them) as a diagnostic for the observed
    # CORRELATIONAL-BUT-NOT-CAUSAL verdict: at this patch point the decoder
    # is softmax(W_U · ln_f(r)), so to FIRST ORDER the raw-space directions
    # it reads are the unembedding rows pulled back through a linearized
    # LayerNorm — span((I - 11^T/d) diag(gain) W_U^T), at most V dims. This
    # is an APPROXIMATION: the true LN Jacobian is input-dependent (the
    # per-sample 1/sigma scale and a -x̂x̂ᵀ/d term are dropped), so "the
    # decoder reads exactly this subspace" is not guaranteed by the algebra
    # alone — the claim stands on the empirical closure it achieves (100.0%
    # at k = V on both processes in the recorded runs). (Uses model weights,
    # not observables: legitimate here because on a real LLM the unembedding
    # is equally available; it is a reading of the network, not of hidden
    # ground truth.)

    # ----- model + fresh evaluation prefixes --------------------------------
    model = GPT(GPTConfig(vocab=proc.V, seq_len=L, d_model=cfg["d_model"],
                          n_layers=cfg["layers"]))
    model.load_state_dict(torch.load(os.path.join(args.outdir, "model.pt"),
                                     map_location="cpu"))
    model.eval()
    q_of = decoder(model)

    with torch.no_grad():
        Wu = model.head.weight.double().numpy()          # (V, d)
        g = model.ln_f.weight.double().numpy()           # (d,)
    M = (np.eye(dmodel) - np.ones((dmodel, dmodel)) / dmodel) @ (g[:, None] * Wu.T)
    Q["unemb"] = orthonormal(M)[:, :min(args.k, proc.V)]

    rng_e = np.random.default_rng(args.seed + 777)
    X = proc.sample(args.eval_seqs, L, rng_e)
    with torch.no_grad():
        resid = []
        for i in range(0, len(X), 256):
            _, r = model(torch.from_numpy(X[i:i + 256]), return_resid=True)
            resid.append(r.double().numpy())
        resid = np.concatenate(resid)                # (N, L, d), pre-ln_f

    B = np.stack([proc.beliefs_along(row) for row in X])      # (N, L, S)
    E = proc.T.sum(axis=2).T                                  # (S, V)
    P1 = B @ E                              # exact next-token dists (N, L, V)

    # pairs: (target a, source b) at a shared position t past burn-in
    n = args.pairs
    a = rng_e.integers(0, len(X), n)
    b = rng_e.integers(0, len(X), n)
    b = np.where(b == a, (b + 1) % len(X), b)
    t = rng_e.integers(burn, L - 1, n)
    r_t, r_s = resid[a, t], resid[b, t]              # (n, d) float64
    p_tgt, p_src = P1[a, t], P1[b, t]                # (n, V) exact

    # ----- self-checks (known-answer code-path validation) ------------------
    q0 = q_of(r_t)
    # no-op patch: source := target
    noop = r_t + (r_t - r_t) @ Q["pls"] @ Q["pls"].T
    assert np.array_equal(q_of(noop), q0), "no-op patch changed the decode"
    # full-space patch: Q = I  =>  r' = r_s exactly
    full = r_t + (r_s - r_t) @ np.eye(dmodel)
    assert np.allclose(q_of(full), q_of(r_s)), "full patch != source decode"
    # interchange really swaps the pls readout: alpha(r') == alpha(r_src)
    swap = r_t + (r_s - r_t) @ Q["pls"] @ Q["pls"].T
    assert np.allclose(swap @ A_pls, r_s @ A_pls, atol=1e-8), \
        "projector does not realize the interchange on alpha_pls"
    print("self-checks passed: no-op, full-swap, alpha-interchange exact\n")

    # ----- the experiment ----------------------------------------------------
    floor_rows = kl_rows(p_tgt, q0)
    gap_rows = kl_rows(p_src, q0)
    floor, gap = float(floor_rows.mean()), float(gap_rows.mean())
    closable = gap - floor

    print(f"=== Experiment 3: interchange interventions | {name} | "
          f"k = {args.k} | {n} position-matched pairs ===\n")
    print("patch point: final-layer residual (pre-ln_f); readout: the model's")
    print("own ln_f + unembedding (the entire remaining network for this patch")
    print("point); horizon: next-token distribution (m=1), exact from beliefs.\n")
    print(f"floor KL(p_tgt || unpatched) = {floor:.5f} (model's intrinsic error)")
    print(f"gap   KL(p_src || unpatched) = {gap:.5f} (what a perfect interchange"
          " must close)\n")

    results = {}
    print("family   KL(p_src||patched)   closure   KL(p_tgt||patched)")
    for fam, Qf in Q.items():
        rp = r_t + (r_s - r_t) @ Qf @ Qf.T
        qp = q_of(rp)
        tr_rows, tg_rows = kl_rows(p_src, qp), kl_rows(p_tgt, qp)
        tr, tg = float(tr_rows.mean()), float(tg_rows.mean())
        se = tr_rows.std(ddof=1) / np.sqrt(n)
        closure = (gap - tr) / closable
        results[fam] = (tr, closure, tg)
        print(f"{fam:>5}    {tr:.5f} ± {se:.5f}    {closure:6.1%}   {tg:.5f}")

    # complement of pls: patch ALL other 64-k directions
    rp = r_t + (r_s - r_t) @ (np.eye(dmodel) - Q["pls"] @ Q["pls"].T)
    qp = q_of(rp)
    comp_tgt = float(kl_rows(p_tgt, qp).mean())
    comp_src = float(kl_rows(p_src, qp).mean())
    leak = (comp_tgt - floor) / closable
    print(f"\ncomplement (I - P_pls, {dmodel - args.k} dims): "
          f"KL(p_tgt||patched) = {comp_tgt:.5f}, leak = {leak:.1%} "
          f"(KL(p_src||patched) = {comp_src:.5f})")

    # -- typed verdicts (pre-registered: experiments/3-readout-interventions.md)
    print("\nverdicts:")
    tr, closure, tg = results["pls"]
    if closure >= 0.90:
        print(f"  pls k={args.k}: CAUSAL TRANSFER — closure {closure:.1%} "
              ">= 0.90 (P1). The discovered subspace is causally load-bearing"
              " for the readout path: writing the source's coordinates makes"
              " the model behave as if it had seen the source prefix.")
    elif tr > gap and tg > gap:
        print(f"  pls k={args.k}: OFF-MANIFOLD BREAKAGE — patched behavior is"
              f" farther than the unpatched gap from BOTH source ({tr:.5f})"
              f" and target ({tg:.5f}); the edit left the reachable manifold.")
    else:
        print(f"  pls k={args.k}: CORRELATIONAL-BUT-NOT-CAUSAL — the subspace"
              f" readout was swapped exactly, yet closure is only {closure:.1%};"
              " the decoder reads completion information from outside the"
              " discovered subspace (redundant coding).")
    if name == "mess3":
        diff = results["pls"][1] - results["pca"][1]
        print(f"  pca vs pls: closure difference {diff:+.1%} "
              f"({'consistent with' if diff >= 0.05 else 'CONTRARY to'} P2: "
              "the token-identity plane should transfer worse than the belief"
              " plane).")
    print(f"  complement: leak {leak:.1%} — "
          + ("causally inert (P3): the junk-precision claim is upgraded from"
             " correlational to causal." if abs(leak) <= 0.05 else
             "COMPLEMENT LEAK: completion-relevant causal pathway outside the"
             " discovered subspace."))
    print(f"  rand: closure {results['rand'][1]:.1%} — "
          + ("no-information control behaves (P4)." if results["rand"][1] <= 0.25
             else "UNEXPECTEDLY HIGH for a random subspace; suspect the"
             " harness before the science."))
    # Diagnostic learned from the first runs: when closure + leak ~ 1 the
    # readout is responding (locally) linearly and the causal effect simply
    # SPLITS between the subspace and its complement — redundant distributed
    # coding, not off-manifold breakage. When it is far from 1 the patch
    # interacts nonlinearly with the decoder (e.g. LayerNorm rescaling).
    total = closure + leak
    print(f"  decomposition: closure(pls) + leak(complement) = {total:.1%} — "
          + ("additive: locally linear readout; the effect splits between"
             " subspace and complement (redundant coding, no breakage)."
             if abs(total - 1.0) <= 0.05 else
             "NON-ADDITIVE: the patch interacts nonlinearly with the decoder."))

    # ----- plot ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = list(results) + ["comp"]
        srcs = [results[f][0] for f in results] + [comp_src]
        tgts = [results[f][2] for f in results] + [comp_tgt]
        x = np.arange(len(fams))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - 0.17, srcs, 0.34, label="KL to SOURCE truth (want low)")
        ax.bar(x + 0.17, tgts, 0.34, label="KL to TARGET truth")
        ax.axhline(floor, ls="--", c="g", label="floor (unpatched, target)")
        ax.axhline(gap, ls=":", c="r", label="gap (unpatched, source)")
        ax.set_xticks(x); ax.set_xticklabels(fams)
        ax.set_yscale("log"); ax.set_ylabel("held-out mean KL [nats]")
        ax.set_title(f"{name}: interchange interventions at k={args.k}")
        ax.legend(fontsize=8)
        p = os.path.join(args.outdir, "experiment3.png")
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"\n(plotting skipped: {e})")


if __name__ == "__main__":
    main()
