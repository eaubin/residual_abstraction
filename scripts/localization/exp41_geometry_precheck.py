"""exp41_geometry_precheck.py — NON-CLAIM design pre-check for exp 41 (guarded steer).

Not a registered experiment: it makes no verdict and routes nothing. It measures the
GEOMETRY of the exp-40 steering directions to decide how the exp-41 guard is built. It
rebuilds v_depth (per horizon) and v_type the way exp 40 does (localize.facet_diff_vector
over observable-matched pairs) and reports, per (position, horizon):

  - cos(v_depth, v_type) per steered prefix position [0..t]   — overlap with the MEAN type
    direction (the cheap single-direction guard target);
  - effective rank of the type-difference subspace (SVD participation ratio + top-1
    variance fraction)                                         — is "type" rank-1 or a subspace?
  - fraction of v_depth@t captured by the rank-r type-difference subspace (r=1,3) — how much
    of the depth direction a subspace erasure would remove.

Reading (informs the exp-41 guard, see experiments/41-*.md):
  - low cos vs the MEAN v_type  -> single-direction orthogonalization is the wrong/weak guard;
  - type PR >> 1                -> guard must be a SUBSPACE erasure, not one direction;
  - high v_depth-in-type capture at k=2, low at k=1 -> predicts a k-SPLIT in exp 41 and
    explains exp 40's k-graded drag geometrically.
Caveat (why exp 41 still runs): this uses the type-DIFFERENCE-variance subspace, which
includes nuisance; direction ENERGY inside it is not readout EFFECT. The exp-41 guard
targets the type-READOUT subspace and controls over-erasure; geometry only sizes the design.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize import (LAYER, depth_triples, facet_pairs,  # noqa: E402
                      require_expected_config, stack_labels)
from midstream import stream_to  # noqa: E402
from processes import PROCESSES  # noqa: E402
from expcommon import load_model  # noqa: E402

POSITIONS = (8, 12, 16, 20)
HORIZONS = {1: (1, 2), 2: (2, 3)}
N_SEQS = 6000


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


def subspace_capture(v, U_r):
    """Fraction of v's squared norm captured by orthonormal columns of U_r (d, r)."""
    proj = U_r @ (U_r.T @ v)
    return float(np.dot(proj, proj) / max(np.dot(v, v), 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
    ap.add_argument("--seed", type=int, default=700)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    require_expected_config(cfg)
    proc = PROCESSES[cfg["process"]]()
    model = load_model(args.outdir, cfg, proc)
    m = cfg["m"]
    rng = np.random.default_rng(args.seed)
    Xe = proc.sample(N_SEQS, cfg["seq_len"], rng)
    dev = next(model.parameters()).device
    print(f"=== exp41 geometry pre-check (NON-CLAIM) | device={dev} | "
          f"seed={args.seed} n_seqs={N_SEQS} ===\n")

    def cache(idx):
        return stream_to(model, torch.from_numpy(Xe[idx]), LAYER).float().cpu().numpy()

    for t in POSITIONS:
        labels = {i: stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
        ct, cs = facet_pairs(labels, "top_type", rng, len(Xe), oriented=True)
        rc_ct, rc_cs = cache(ct), cache(cs)
        v_type = (rc_cs[:, :t + 1] - rc_ct[:, :t + 1]).mean(0)        # (t+1, d) mean type dir

        Dt = rc_cs[:, t] - rc_ct[:, t]                               # per-pair type diffs (n, d)
        Dt = Dt - Dt.mean(0, keepdims=True)
        _, s, Vt = np.linalg.svd(Dt, full_matrices=False)
        lam = s ** 2
        PR = float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-30))
        top1 = float(lam[0] / max(lam.sum(), 1e-30))
        U = Vt.T                                                     # (d, k) type-diff dirs

        for k, (lo, hi) in HORIZONS.items():
            cd, chi, _ = depth_triples(labels, lo, hi, rng)
            rc_cd, rc_chi = cache(cd), cache(chi)
            v_depth = (rc_chi[:, :t + 1] - rc_cd[:, :t + 1]).mean(0)
            coss = [abs(cos(v_depth[p], v_type[p])) for p in range(t + 1)]
            print(f"t={t:2d} k={k} | cos@t={cos(v_depth[t], v_type[t]):+.3f}  "
                  f"mean|cos|[0..t]={np.mean(coss):.3f}  max|cos|={np.max(coss):.3f}  | "
                  f"type-subspace PR={PR:4.1f} top1var={top1:.2f}  "
                  f"v_depth@t in type: r1={subspace_capture(v_depth[t], U[:, :1]):.2f} "
                  f"r3={subspace_capture(v_depth[t], U[:, :3]):.2f}")
        print()


def _selftest():
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    assert abs(cos(e1, e1) - 1.0) < 1e-9 and abs(cos(e1, e2)) < 1e-9
    assert abs(cos(e1, -e1) + 1.0) < 1e-9
    assert abs(np.linalg.norm(unit(np.array([3.0, 4.0]))) - 1.0) < 1e-9
    U = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])               # span(e1,e2)
    assert abs(subspace_capture(e1, U[:, :1]) - 1.0) < 1e-9          # fully in
    assert abs(subspace_capture(np.array([0.0, 0.0, 1.0]), U)) < 1e-9  # orthogonal
    v = np.array([1.0, 0.0, 1.0])                                    # half in span(e1,e2)
    assert abs(subspace_capture(v, U[:, :2]) - 0.5) < 1e-9
    print("exp41 geometry pre-check selftest OK")


if __name__ == "__main__":
    main()
