"""exp38_ceiling_smoke.py — feasibility precondition for exp 38 (NOT the rung).

Question (exp 38 design doc, "Feasibility precondition"): does the FULL-PREFIX
residual patch move the m>=2 forced-close conditional from clean toward source?
If even the ceiling cannot, the graded (k>=1) signal is uninterpretable and the
rung is HARNESS_FAIL. This is a thin read-out on L0's existing machinery: L0
already builds the m=3 joint under a full-prefix patch (model_guards' q_full); it
only ever read the m=1 marginal. Here we read the k-step conditionals.

Observable: close-readiness after k forced closes (closers = tokens {2,3}):
  cr_cond(k) = P(w_{k+1} closer | w_1..w_k all closers).
The k-th conditional separates depth k vs depth k+1 (depth d absorbs d closes
before the stack empties), so we smoke each k on a matched depth pair:
  k=0: depth-? (L0's m=1 signal; transports exactly -> wiring sanity)
  k=1: depth 1 (clean) vs depth 2 (source); after one close -> empty vs depth-1
  k=2: depth 2 (clean) vs depth 3 (source); after two closes -> empty vs depth-1

Pairs matched on top_type so the contrast is graded depth, not which closer. For
each k we measure, per pair:
  C    = cr(clean tokens, no patch)               -- clean (lo-depth) belief
  S    = cr(source tokens, no patch)              -- source (hi-depth) oracle
  P_hi = cr(clean tokens, hi-depth residual :t+1) -- full-prefix patch (ceiling)
  P_lo = cr(clean tokens, lo-depth residual :t+1) -- SAME-DEPTH floor
transport f = (P - C)/(S - C) pooled over pairs with |S - C| >= GAP_MIN.
net = f_ceil - f_floor is the graded-depth transport above the same-depth floor.
A same-depth (P_lo) patch that still moves cr is a generic full-prefix artifact,
not graded-depth transport -> the real signal is net, not f_ceil alone.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize import (LAYER, q_at, require_expected_config,  # noqa: E402
                      stack_labels)
from midstream import marginal, stream_to  # noqa: E402
from processes import PROCESSES  # noqa: E402
from expcommon import load_model  # noqa: E402

GAP_MIN = 0.10
N_SEQS = 6000
SEED = 700
POSITIONS = (8, 12, 16, 20)                    # L0's registered interior positions
HORIZONS = ((0, 1, 2), (1, 1, 2), (2, 2, 3))   # (k, lo_depth, hi_depth)


def cr_cond(q, V, m, k):
    """P(w_{k+1} closer | w_1..w_k all closers) from the (k+1)-step marginal."""
    mm = k + 1
    arr = marginal(q, V, mm, m).reshape(len(q), *([V] * mm))
    arr = arr / np.clip(arr.sum(tuple(range(1, mm + 1)), keepdims=True), 1e-30, None)
    cond = (slice(None),) + (slice(2, 4),) * k
    den = arr[cond + (slice(None),)].sum(tuple(range(1, mm + 1)))
    num = arr[cond + (slice(2, 4),)].sum(tuple(range(1, mm + 1)))
    return num / np.clip(den, 1e-12, None)


def triples(Xe, t, m, lo, hi, rng):
    """Per top_type at position t, aligned indices matched on top_type:
      clean  = depth `lo` (the tokens we patch onto),
      src_hi = depth `hi` (the graded-depth patch -> ceiling),
      src_lo = depth `lo`, distinct instance (same-depth patch -> floor)."""
    labels = {i: stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
    by_tt = {}
    for i, (d, tt) in labels.items():
        if tt < 0:
            continue
        by_tt.setdefault(tt, {}).setdefault(d, []).append(i)
    clean, src_hi, src_lo = [], [], []
    for tt, by_d in by_tt.items():
        if hi not in by_d or len(by_d.get(lo, [])) < 2:
            continue
        dl = list(rng.permutation(by_d[lo]))
        dh = list(rng.permutation(by_d[hi]))
        n = min(len(dl) // 2, len(dh))
        clean += dl[:n]
        src_lo += dl[n:2 * n]
        src_hi += dh[:n]
    return np.array(clean, int), np.array(src_hi, int), np.array(src_lo, int)


def frac(P, C, S):
    keep = np.abs(S - C) >= GAP_MIN
    if keep.sum() == 0:
        return float("nan"), 0
    return float(np.mean((P[keep] - C[keep]) / (S[keep] - C[keep]))), int(keep.sum())


def main():
    outdir = "out/dyck2-L4"
    with open(os.path.join(outdir, "config.json")) as f:
        cfg = json.load(f)
    require_expected_config(cfg)
    proc = PROCESSES[cfg["process"]]()
    model = load_model(outdir, cfg, proc)
    m, V = cfg["m"], proc.V
    rng = np.random.default_rng(SEED)
    Xe = proc.sample(N_SEQS, cfg["seq_len"], rng)
    resid = stream_to(model, torch.from_numpy(Xe), LAYER)
    print(f"=== exp38 ceiling smoke | L{cfg['layers']} d{cfg['d_model']} "
          f"m={m} | seed {SEED} | N={N_SEQS} | GAP_MIN={GAP_MIN} ===")
    print("net = f_ceil - f_floor = graded-depth transport above the same-depth "
          "floor (full-prefix patch).\n")

    for k, lo, hi in HORIZONS:
        print(f"--- k={k}: depth {lo} (clean) vs depth {hi} (source), "
              f"conditional close-readiness after {k} forced closes ---")
        for t in POSITIONS:
            a, ihi, ilo = triples(Xe, t, m, lo, hi, rng)
            if len(a) < 50:
                print(f"t={t:2d}: only {len(a)} depth-{lo}/{hi} triples "
                      f"(abundance check) -> skip")
                continue
            qC = q_at(model, Xe[a], t, m, V)
            qS = q_at(model, Xe[ihi], t, m, V)
            qHi = q_at(model, Xe[a], t, m, V, prefix_state=resid[ihi][:, :t + 1])
            qLo = q_at(model, Xe[a], t, m, V, prefix_state=resid[ilo][:, :t + 1])
            C, S = cr_cond(qC, V, m, k), cr_cond(qS, V, m, k)
            Phi, Plo = cr_cond(qHi, V, m, k), cr_cond(qLo, V, m, k)
            f_ceil, n = frac(Phi, C, S)
            f_floor, _ = frac(Plo, C, S)
            print(f"t={t:2d} n={len(a):4d} (gap>= {GAP_MIN}: {n:4d}) | "
                  f"C={C.mean():.3f} S={S.mean():.3f} P_hi={Phi.mean():.3f} "
                  f"P_lo={Plo.mean():.3f} | f_ceil={f_ceil:+.3f} "
                  f"f_floor={f_floor:+.3f} net={f_ceil - f_floor:+.3f}")
        print()


if __name__ == "__main__":
    main()
