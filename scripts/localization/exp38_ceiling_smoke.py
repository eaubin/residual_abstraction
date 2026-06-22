"""exp38_ceiling_smoke.py — feasibility precondition for exp 38 (NOT the rung).

Question (exp 38 design doc, "Feasibility precondition"): does the FULL-PREFIX
residual patch move the m>=2 forced-close conditional from clean toward source?
If even the ceiling cannot, the graded (k>=1) signal is uninterpretable and the
rung is HARNESS_FAIL. This is a thin read-out on L0's existing machinery: L0
already builds the m=3 joint under a full-prefix patch (model_guards' q_full); it
only ever read the m=1 marginal. Here we read the k=1 conditional of that joint.

Observables (close-readiness after k forced closes; closers = tokens {2,3}):
  k=0  cr0 = P(w1 closer)                      -- L0's m=1 signal (must transport
                                                   exactly: reproduces the guard)
  k=1  cr1 = P(w2 closer | w1 closer)          -- the graded signal m=1 cannot see
       (source depth 2 -> after one close depth 1 -> still closeable -> high;
        clean depth 1 -> after one close empty -> must open -> low)

Pairs: clean = depth 1, source = depth 2, matched on top_type (so the contrast is
graded depth, not which closer). For each pair we measure, per observable:
  C = cr(clean tokens, no patch)              -- clean belief
  S = cr(source tokens, no patch)             -- source oracle target
  P = cr(clean tokens, source residual :t+1)  -- full-prefix patch (the ceiling)
  floor: P_rand using an unrelated random seq's residual (mismatched patch)
transport fraction f = (P - C) / (S - C), pooled over pairs with |S - C| >= GAP_MIN.
f ~ 1: patch reaches source; f ~ 0: contamination wins (clean block-0 dominates).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l0_substrate_gate as l0  # noqa: E402  (frozen L0; reuse wholesale)
from midstream import marginal, stream_to  # noqa: E402

GAP_MIN = 0.10        # only score pairs with a real oracle k1 gap |S - C|
N_SEQS = 6000
SEED = 700


def cr_k0(q, V, m):
    q1 = marginal(q, V, 1, m)
    q1 = q1 / np.clip(q1.sum(1, keepdims=True), 1e-30, None)
    return q1[:, 2] + q1[:, 3]


def cr_k1(q, V, m):
    """P(w2 closer | w1 closer) from the m=2 marginal q2[w1, w2]."""
    q2 = marginal(q, V, 2, m).reshape(-1, V, V)
    q2 = q2 / np.clip(q2.sum((1, 2), keepdims=True), 1e-30, None)
    num = q2[:, 2:4, 2:4].sum((1, 2))     # P(w1 closer, w2 closer)
    den = q2[:, 2:4, :].sum((1, 2))       # P(w1 closer)
    return num / np.clip(den, 1e-12, None)


def triples(Xe, t, m, rng):
    """Per top_type at position t, aligned indices:
      clean  = depth 1 (the tokens we patch onto),
      src_hi = depth 2 (the graded-depth patch -> ceiling),
      src_lo = depth 1, distinct instance (same-depth patch -> floor).
    A same-depth (src_lo) patch that still moves cr1 is a generic full-prefix
    artifact, not graded-depth transport; the real signal is src_hi vs src_lo."""
    labels = {i: l0.stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
    by_tt = {}
    for i, (d, tt) in labels.items():
        if tt < 0:
            continue
        by_tt.setdefault(tt, {}).setdefault(d, []).append(i)
    clean, src_hi, src_lo = [], [], []
    for tt, by_d in by_tt.items():
        if 2 not in by_d or len(by_d.get(1, [])) < 2:
            continue
        d1 = list(rng.permutation(by_d[1]))
        d2 = list(rng.permutation(by_d[2]))
        n = min(len(d1) // 2, len(d2))
        clean += d1[:n]
        src_lo += d1[n:2 * n]
        src_hi += d2[:n]
    return np.array(clean, int), np.array(src_hi, int), np.array(src_lo, int)


def frac(P, C, S):
    keep = np.abs(S - C) >= GAP_MIN
    if keep.sum() == 0:
        return float("nan"), 0
    return float(np.mean((P[keep] - C[keep]) / (S[keep] - C[keep]))), int(keep.sum())


def main():
    outdir = "out/dyck2-L4"
    import json
    with open(os.path.join(outdir, "config.json")) as f:
        cfg = json.load(f)
    l0.require_expected_config(cfg)
    proc = l0.PROCESSES[cfg["process"]]()
    model = l0.load_model(outdir, cfg, proc)
    m, V = cfg["m"], proc.V
    rng = np.random.default_rng(SEED)
    Xe = proc.sample(N_SEQS, cfg["seq_len"], rng)
    resid = stream_to(model, torch.from_numpy(Xe), l0.LAYER)
    print(f"=== exp38 ceiling smoke | L{cfg['layers']} d{cfg['d_model']} "
          f"m={m} | seed {SEED} | GAP_MIN={GAP_MIN} ===\n")

    for t in l0.POSITIONS:
        a, hi, lo = triples(Xe, t, m, rng)         # clean d1 / src d2 / src d1
        if len(a) < 50:
            print(f"t={t:2d}: only {len(a)} triples, skip")
            continue
        qC = l0.q_at(model, Xe[a], t, m, V)                 # clean, no patch
        qS = l0.q_at(model, Xe[hi], t, m, V)                # source oracle (d2)
        qHi = l0.q_at(model, Xe[a], t, m, V, prefix_state=resid[hi][:, :t + 1])
        qLo = l0.q_at(model, Xe[a], t, m, V, prefix_state=resid[lo][:, :t + 1])
        for k, cr in ((0, cr_k0), (1, cr_k1)):
            C, S = cr(qC, V, m), cr(qS, V, m)
            Phi, Plo = cr(qHi, V, m), cr(qLo, V, m)
            f_ceil, n = frac(Phi, C, S)            # d2 patch vs clean, /oracle
            f_floor, _ = frac(Plo, C, S)           # d1 (same-depth) patch
            net = f_ceil - f_floor                 # graded transport above floor
            print(f"t={t:2d} k={k} n={len(a):4d} (gap>= {GAP_MIN}: {n:4d}) | "
                  f"C={C.mean():.3f} S={S.mean():.3f} P_hi={Phi.mean():.3f} "
                  f"P_lo={Plo.mean():.3f} | f_ceil={f_ceil:+.3f} "
                  f"f_floor={f_floor:+.3f} net={net:+.3f}")
        print()


if __name__ == "__main__":
    main()
