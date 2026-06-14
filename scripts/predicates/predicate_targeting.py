"""
Experiment 29 — predicate-targeting diagnostic (ORIGINAL_SIN.md, reduction #3).

Do within-horizon completion predicates pick out non-PCA, causal residual
structure on pstack — or does predicate-targeting collapse onto the
variance/KL subspace, the way exp 26 found cegar ~ pca? A predicate phi is a
Boolean mask over V^m (predicates.py); p_phi = mask . q is one behavioral
coordinate. For each phi:

  decode      affine probe p_phi ~ w.r (observable); rank-1 direction w_phi.
              linear R2 low but kNN R2 high = present-not-affinely-decodable
              (the exp-1 Z1R interpreter gap).
  intervene   patch r along w_phi (source->target); p_phi-closure c_phi(w)
              vs the full-patch ceiling and a random-direction floor. decode
              without causal control = ECHO (the program's oldest lesson).
  diverge     principal angles w_phi vs PCA top-k and vs the cegar core. Off
              both, while causal = predicate picks out non-variance causal
              structure (DIVERGENT). Inside both = ALIGNED.

This is a DIAGNOSTIC, multi-seed (the angles are seed-fragile geometry, per
exps 24/25). It does NOT run a CEGAR-against-phi loop — it decides whether
that loop is worth building. ALIGNED is the likely, non-failure outcome on
near-mimicry pstack: it redirects to a designed-backward richer toy with
evidence, not vibes. Built from library primitives (cegar_loop,
PCAAbstraction, predicates) — no import from frozen experiment scripts.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import predicates as P
from abstraction import PCAAbstraction, center_by_position
from battery import Refs, cegar_loop
from discover import PairSet, principal_angles_deg, self_checks
from expcommon import LAYER, load_model
from midstream import orthonormal
from processes import PROCESSES

REGISTERED_CFG = {"process": "pstack", "seq_len": 40, "burn_in": 4,
                  "d_model": 64, "layers": 4, "m": 3, "seed": 0}
M, MM = 3, 3
TS = (10, 18, 26, 34)
SEEDS = (300, 301, 302, 303)              # 4 fresh seeds, disjoint from 0-7/100-207
PAIRS_DISC, PAIRS_EVAL, PAIR_POOL = 320, 1024, 900
EPS, EPS_DROP, K_MAX, MAX_DIM = 0.05, 0.01, 10, 8
LAM = 1e-2                                # ridge on centered residuals
KNN_K = 10
R2_MIN = 0.50                             # rank-1 decode sufficiency
VAR_MIN = 0.05                            # p_phi std floor (predicate non-vacuity)
ANGLE_MIN = 10.0                          # deg, off-subspace (arc threshold)
C_MIN = 0.50                              # p_phi-closure: w-patch causal floor
C_MARGIN = 0.20                           # c_w must beat the random floor by this
OE_BAND = 0.10                            # obs/exact soundness gate on p_phi
SEED_MAJORITY = 3                         # of 4 seeds


def r2(y, yhat):
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0


def ridge_fit(X, y, lam):
    mu, ym = X.mean(0), y.mean()
    Xc = X - mu
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, float(ym - mu @ w)


def knn_r2(Xtr, ytr, Xte, yte, k):
    d2 = ((Xte ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None, :]
          - 2 * Xte @ Xtr.T)
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    return r2(yte, ytr[nn].mean(1))


def pphi_closure(p_un, p_src, p_P, p_full):
    """Fraction of the p_phi gap to source closed by patch P (1 = full)."""
    d0 = float(((p_un - p_src) ** 2).mean())
    dP = float(((p_P - p_src) ** 2).mean())
    dfull = float(((p_full - p_src) ** 2).mean())
    denom = d0 - dfull
    return (d0 - dP) / denom if denom > 1e-9 else float("nan")


def gather_residuals(ps, d):
    R = np.empty((ps.n, d))
    pos = np.empty(ps.n, dtype=np.int64)
    beliefs = np.empty((ps.n, ps.B.shape[2]))
    S = ps.S.double().numpy()
    for t, idx in ps.groups:
        R[idx] = S[ps.a[idx], t]
        beliefs[idx] = ps.B[ps.a[idx], t]
        pos[idx] = t
    return R, pos, beliefs


def run_seed(model, proc, cfg, seed, masks):
    d = cfg["d_model"]
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + 111, PAIR_POOL,
                   layer=LAYER, ts=TS)
    eval_ps = PairSet(model, proc, cfg, PAIRS_EVAL, M, seed + 222, PAIR_POOL,
                      layer=LAYER, ts=TS)
    self_checks(model, eval_ps, LAYER, M, proc.V)

    refs_d = Refs(disc, model, d, M)
    k_raw, Qc_raw, _ = cegar_loop(model, disc, refs_d, d, EPS, K_MAX, MM,
                                  eps_drop=EPS_DROP)
    k_core = int(np.clip(k_raw, 1, MAX_DIM))
    Qcore = Qc_raw[:, :k_core]

    R, pos, beliefs = gather_residuals(eval_ps, d)
    Rc = center_by_position(R, pos, np.ones(eval_ps.n, dtype=bool))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(eval_ps.n)
    tr, te = perm[:eval_ps.n // 2], perm[eval_ps.n // 2:]
    pca = PCAAbstraction(Rc[tr])

    q_un = eval_ps.run(model, None)
    q_src = eval_ps.run(model, None, src_side=True)
    q_full = eval_ps.run(model, np.eye(d))
    rdir = orthonormal(rng.standard_normal((d, 1)))
    q_rand = eval_ps.run(model, rdir @ rdir.T)

    out = {}
    for name, mask in masks.items():
        y = P.obs_pphi(q_un, mask)
        w, b = ridge_fit(Rc[tr], y[tr], LAM)
        lin = r2(y[te], Rc[te] @ w + b)
        knn = knn_r2(Rc[tr], y[tr], Rc[te], y[te], KNN_K)
        wdir = (w / np.linalg.norm(w))[:, None]
        ang_pca = float(principal_angles_deg(wdir, pca.Vt[:k_core].T)[0])
        ang_core = float(principal_angles_deg(wdir, Qcore)[0])

        q_w = eval_ps.run(model, wdir @ wdir.T)
        p_un, p_src = P.obs_pphi(q_un, mask), P.obs_pphi(q_src, mask)
        p_full = P.obs_pphi(q_full, mask)
        c_w = pphi_closure(p_un, p_src, P.obs_pphi(q_w, mask), p_full)
        c_rand = pphi_closure(p_un, p_src, P.obs_pphi(q_rand, mask), p_full)
        oe = float(np.abs(p_un - P.exact_pphi(beliefs, mask, proc, M)).mean())

        out[name] = {"pphi_std": float(y.std()), "lin_r2": lin, "knn_r2": knn,
                     "ang_pca": ang_pca, "ang_core": ang_core, "c_w": c_w,
                     "c_rand": c_rand, "oe": oe}
    return out, k_core


def classify_phi(m):
    """Clean partition over one (phi, seed) measurement (FORMALISM 6.1)."""
    if m["pphi_std"] < VAR_MIN:
        return "VACUOUS"
    if m["lin_r2"] < R2_MIN:
        return "INTERPRETER_GAP" if m["knn_r2"] >= R2_MIN else "NOT_DECODABLE"
    if m["c_w"] < C_MIN or (m["c_w"] - m["c_rand"]) < C_MARGIN:
        return "ECHO"                          # decodes but not causal
    off_pca = m["ang_pca"] > ANGLE_MIN
    off_core = m["ang_core"] > ANGLE_MIN
    if off_pca and off_core:
        return "DIVERGENT"
    if not off_pca and not off_core:
        return "ALIGNED"
    return "MIXED"                             # off one subspace but not both


def aggregate_phi(verdicts, sounds):
    """A phi's multi-seed verdict reproduces iff it holds in >= SEED_MAJORITY
    seeds; DIVERGENT additionally requires obs/exact soundness in the
    majority."""
    counts = {v: verdicts.count(v) for v in set(verdicts)}
    top = max(counts, key=counts.get)
    if counts[top] < SEED_MAJORITY:
        return "UNSTABLE"
    if top == "DIVERGENT" and sum(sounds) < SEED_MAJORITY:
        return "DIVERGENT_UNSOUND"
    return top


def decide(phi_aggregates):
    div = [n for n, v in phi_aggregates.items() if v == "DIVERGENT"]
    if div:
        return "DIVERGENT(" + ",".join(div) + ")"
    causal = {n: v for n, v in phi_aggregates.items()
              if v in ("ALIGNED", "MIXED")}
    if causal and all(v == "ALIGNED" for v in causal.values()):
        return "ALIGNED"
    echo = [n for n, v in phi_aggregates.items() if v == "ECHO"]
    if echo and not causal:
        return "ECHO(" + ",".join(echo) + ")"
    return "INCONCLUSIVE"


def selftest():
    P._selftest()
    assert abs(r2(np.array([1., 2, 3]), np.array([1., 2, 3])) - 1.0) < 1e-12
    base = {"pphi_std": 0.2, "lin_r2": 0.8, "knn_r2": 0.85, "ang_pca": 3.0,
            "ang_core": 3.0, "c_w": 0.8, "c_rand": 0.1, "oe": 0.02}

    def w(**kw):
        d = dict(base); d.update(kw); return d
    assert classify_phi(w()) == "ALIGNED"
    assert classify_phi(w(ang_pca=40, ang_core=40)) == "DIVERGENT"
    assert classify_phi(w(ang_pca=40)) == "MIXED"          # off pca not core
    assert classify_phi(w(c_w=0.1)) == "ECHO"              # not causal
    assert classify_phi(w(c_w=0.8, c_rand=0.7)) == "ECHO"  # no margin over floor
    assert classify_phi(w(lin_r2=0.2, knn_r2=0.8)) == "INTERPRETER_GAP"
    assert classify_phi(w(lin_r2=0.2, knn_r2=0.2)) == "NOT_DECODABLE"
    assert classify_phi(w(pphi_std=0.01)) == "VACUOUS"

    assert aggregate_phi(["ALIGNED"] * 3 + ["MIXED"], [True] * 4) == "ALIGNED"
    assert aggregate_phi(["DIVERGENT"] * 3 + ["ECHO"], [True] * 4) == "DIVERGENT"
    assert aggregate_phi(["DIVERGENT"] * 3 + ["ECHO"],
                         [False] * 4) == "DIVERGENT_UNSOUND"
    assert aggregate_phi(["ALIGNED", "DIVERGENT", "ECHO", "MIXED"],
                         [True] * 4) == "UNSTABLE"

    assert decide({"a": "DIVERGENT", "b": "ALIGNED"}).startswith("DIVERGENT")
    assert decide({"a": "ALIGNED", "b": "ALIGNED"}) == "ALIGNED"
    assert decide({"a": "ECHO", "b": "ECHO"}).startswith("ECHO")
    assert decide({"a": "ALIGNED", "b": "MIXED"}) == "INCONCLUSIVE"
    print("selftest passed: decode/intervene/diverge partition, aggregate, decide")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/pstack-L4")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    mism = [(k, cfg.get(k), v) for k, v in REGISTERED_CFG.items()
            if cfg.get(k) != v]
    if mism:
        print("HALT: wrong checkpoint config:", mism)
        return 1
    proc = PROCESSES[cfg["process"]]()
    model = load_model(args.outdir, cfg, proc)
    masks = P.registered_masks(proc.V, M)

    print("=== Experiment 29: predicate-targeting diagnostic ===")
    print(f"target={proc.name} m={M} LAYER={LAYER} seeds={SEEDS}")
    print("predicates:", ", ".join(masks))
    print("Built from library primitives; exact p_phi is eval-only.\n")

    per_seed = {}
    for s in SEEDS:
        out, k_core = run_seed(model, proc, cfg, s, masks)
        per_seed[s] = out
        print(f"[seed {s}] k_core={k_core}")
        print("  phi                 std  linR2 knnR2  a(PCA) a(core)  c_w  "
              "c_rand   oe   -> verdict")
        for name, m in out.items():
            print(f"  {name:<18} {m['pphi_std']:.3f} {m['lin_r2']:>5.2f} "
                  f"{m['knn_r2']:>5.2f}  {m['ang_pca']:>5.1f}  {m['ang_core']:>5.1f}"
                  f"  {m['c_w']:>5.2f} {m['c_rand']:>5.2f}  {m['oe']:.3f}  "
                  f"-> {classify_phi(m)}")

    print("\n[multi-seed aggregate]")
    aggregates = {}
    for name in masks:
        verdicts = [classify_phi(per_seed[s][name]) for s in SEEDS]
        sounds = [per_seed[s][name]["oe"] <= OE_BAND for s in SEEDS]
        aggregates[name] = aggregate_phi(verdicts, sounds)
        print(f"  {name:<18} {verdicts} sound={sounds} -> {aggregates[name]}")

    decision = decide(aggregates)
    print(f"\nDECISION: {decision}")
    if decision.startswith("DIVERGENT"):
        print("  Predicate-targeting picks out non-variance, causal residual "
              "structure on pstack — CEGAR-against-phi would not collapse to "
              "PCA. Pre-register the predicate-CEGAR loop.")
    elif decision == "ALIGNED":
        print("  Every causal predicate direction sits inside PCA~core — the "
              "lattice adds no new residual geometry on near-mimicry pstack "
              "(consistent with exp 26). EVIDENCE to design a richer toy "
              "backward from predicates; do not keep mining pstack. NOT a "
              "failure.")
    elif decision.startswith("ECHO"):
        print("  Predicates decode but patching does not move them: "
              "decode-sufficiency != causal sufficiency at predicate "
              "granularity. The interventional criterion is the only honest "
              "one before any predicate-CEGAR.")
    else:
        print("  Inconclusive / unstable across seeds — report per-phi; the "
              "geometry did not reproduce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
