"""
Experiment 29 — predicate-targeting measurement pilot.

Do within-horizon completion predicates yield a non-vacuous, interpretable
measurement stack on pstack, and do the resulting affine readout directions
provide routing evidence about PCA/core geometry? A predicate phi is a Boolean
mask over V^m (predicates.py); p_phi = mask . q is one behavioral coordinate.
For each phi:

  decode      affine probe p_phi ~ w.r (observable); rank-1 direction w_phi.
              linear R2 low but kNN R2 high = present-not-affinely-decodable
              (the exp-1 Z1R interpreter gap).
  intervene   patch r along w_phi (source->target); p_phi-closure c_phi(w)
              vs the full-patch ceiling and a random-direction floor. decode
              without causal control = ECHO (a local decode/control split).
  geometry    principal angles w_phi vs PCA top-k and vs the cegar core. Off
              both, while causal and exact-sound = DIVERGENT. Inside both =
              ALIGNED. Off one = MIXED_GEOMETRY.

This is an instrumentation / construct-validity pilot, multi-seed (the angles
are seed-fragile geometry, per exps 24/25). It does NOT run a
CEGAR-against-phi loop and does not claim minimal predicate abstractions.
ALIGNED is the likely, non-failure outcome on near-mimicry pstack only if
coverage holds across the non-excluded predicates; partial alignment is limited
evidence. Built from library primitives (cegar_loop, PCAAbstraction,
predicates) — no import from frozen experiment scripts.
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
SOUNDNESS_GATED = ("DIVERGENT", "ALIGNED", "MIXED_GEOMETRY", "ECHO")
ALIGN_COVERAGE_MIN = 3                    # ALIGNED routes only with coverage


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
    """Fraction of the p_phi gap to source closed by patch P.

    Returns NaN when the full patch creates no meaningful predicate-level
    recovery room; the verdict partition reports that as NO_PATCH_ROOM."""
    d0 = float(((p_un - p_src) ** 2).mean())
    dP = float(((p_P - p_src) ** 2).mean())
    dfull = float(((p_full - p_src) ** 2).mean())
    denom = d0 - dfull
    return (d0 - dP) / denom if denom > 1e-9 else float("nan")


def gather_residuals(ps, d):
    R = np.empty((ps.n, d))
    pos = np.empty(ps.n, dtype=np.int64)
    beliefs_tgt = np.empty((ps.n, ps.B.shape[2]))
    beliefs_src = np.empty((ps.n, ps.B.shape[2]))
    S = ps.S.double().numpy()
    for t, idx in ps.groups:
        R[idx] = S[ps.a[idx], t]
        beliefs_tgt[idx] = ps.B[ps.a[idx], t]
        beliefs_src[idx] = ps.B[ps.b[idx], t]
        pos[idx] = t
    return R, pos, beliefs_tgt, beliefs_src


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

    R, pos, beliefs_tgt, beliefs_src = gather_residuals(eval_ps, d)
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
        p_un, p_src = P.obs_pphi(q_un, mask), P.obs_pphi(q_src, mask)
        p_full = P.obs_pphi(q_full, mask)
        c_rand = pphi_closure(p_un, p_src, P.obs_pphi(q_rand, mask), p_full)
        w_norm = float(np.linalg.norm(w))
        if w_norm > 1e-12:
            wdir = (w / w_norm)[:, None]
            ang_pca = float(principal_angles_deg(wdir, pca.Vt[:k_core].T)[0])
            ang_core = float(principal_angles_deg(wdir, Qcore)[0])
            q_w = eval_ps.run(model, wdir @ wdir.T)
            c_w = pphi_closure(p_un, p_src, P.obs_pphi(q_w, mask), p_full)
        else:
            ang_pca = float("nan")
            ang_core = float("nan")
            c_w = float("nan")
        p_tgt_exact = P.exact_pphi(beliefs_tgt, mask, proc, M)
        p_src_exact = P.exact_pphi(beliefs_src, mask, proc, M)
        oe_tgt = float(np.abs(p_un - p_tgt_exact).mean())
        oe_src = float(np.abs(p_src - p_src_exact).mean())
        oe = max(oe_tgt, oe_src)

        out[name] = {"pphi_std": float(y.std()), "lin_r2": lin, "knn_r2": knn,
                     "ang_pca": ang_pca, "ang_core": ang_core, "c_w": c_w,
                     "c_rand": c_rand, "oe": oe, "oe_tgt": oe_tgt,
                     "oe_src": oe_src}
    return out, k_core


def classify_phi(m):
    """Clean partition over one (phi, seed) measurement (FORMALISM 6.1)."""
    if m["pphi_std"] < VAR_MIN:
        return "PREDICATE_VACUOUS"
    if m["lin_r2"] < R2_MIN:
        return "INTERPRETER_GAP" if m["knn_r2"] >= R2_MIN else "NOT_DECODABLE"
    if not (np.isfinite(m["c_w"]) and np.isfinite(m["c_rand"])):
        return "NO_PATCH_ROOM"                 # full patch has no predicate room
    if m["c_w"] < C_MIN or (m["c_w"] - m["c_rand"]) < C_MARGIN:
        return "ECHO"                          # decodes but not causal
    off_pca = m["ang_pca"] > ANGLE_MIN
    off_core = m["ang_core"] > ANGLE_MIN
    if off_pca and off_core:
        return "DIVERGENT"
    if not off_pca and not off_core:
        return "ALIGNED"
    return "MIXED_GEOMETRY"                    # off one subspace but not both


def aggregate_phi(verdicts, sounds):
    """A phi's multi-seed verdict reproduces iff it holds in >= SEED_MAJORITY
    seeds; headline states additionally require obs/exact soundness on the
    reproducing seeds."""
    counts = {v: verdicts.count(v) for v in set(verdicts)}
    top = max(counts, key=counts.get)
    if counts[top] < SEED_MAJORITY:
        return "SEED_UNSTABLE"
    if top in SOUNDNESS_GATED:
        top_sounds = [sounds[i] for i, v in enumerate(verdicts) if v == top]
        if sum(top_sounds) < SEED_MAJORITY:
            return "OBS_EXACT_DRIFT"
    return top


def decide(phi_aggregates):
    drift = [n for n, v in phi_aggregates.items() if v == "OBS_EXACT_DRIFT"]
    if drift:
        return "OBS_EXACT_DRIFT(" + ",".join(drift) + ")"
    div = [n for n, v in phi_aggregates.items() if v == "DIVERGENT"]
    if div:
        return "DIVERGENT(" + ",".join(div) + ")"
    mixed = [n for n, v in phi_aggregates.items() if v == "MIXED_GEOMETRY"]
    if mixed:
        return "MIXED_GEOMETRY(" + ",".join(mixed) + ")"
    aligned = [n for n, v in phi_aggregates.items() if v == "ALIGNED"]
    blockers = [n for n, v in phi_aggregates.items()
                if v in ("ECHO", "INTERPRETER_GAP", "NOT_DECODABLE",
                         "SEED_UNSTABLE")]
    if len(aligned) >= ALIGN_COVERAGE_MIN and not blockers:
        return "ALIGNED(" + ",".join(aligned) + ")"
    if aligned and blockers:
        return ("PARTIAL_ALIGNED(aligned=" + ",".join(aligned)
                + "; blocked=" + ",".join(blockers) + ")")
    if aligned:
        return ("PARTIAL_ALIGNED(aligned=" + ",".join(aligned)
                + "; coverage=" + str(len(aligned)) + "/"
                + str(ALIGN_COVERAGE_MIN) + ")")
    echo = [n for n, v in phi_aggregates.items() if v == "ECHO"]
    if echo:
        return "ECHO(" + ",".join(echo) + ")"
    interp = [n for n, v in phi_aggregates.items()
              if v in ("INTERPRETER_GAP", "NOT_DECODABLE")]
    if interp:
        return "INTERPRETER_LIMIT(" + ",".join(interp) + ")"
    no_room = [n for n, v in phi_aggregates.items() if v == "NO_PATCH_ROOM"]
    if no_room:
        return "NO_PATCH_ROOM(" + ",".join(no_room) + ")"
    vac = [n for n, v in phi_aggregates.items() if v == "PREDICATE_VACUOUS"]
    if vac:
        return "PREDICATE_VACUOUS(" + ",".join(vac) + ")"
    unstable = [n for n, v in phi_aggregates.items() if v == "SEED_UNSTABLE"]
    if unstable:
        return "SEED_UNSTABLE(" + ",".join(unstable) + ")"
    return "SEED_UNSTABLE"


def selftest():
    P._selftest()
    assert abs(r2(np.array([1., 2, 3]), np.array([1., 2, 3])) - 1.0) < 1e-12
    base = {"pphi_std": 0.2, "lin_r2": 0.8, "knn_r2": 0.85, "ang_pca": 3.0,
            "ang_core": 3.0, "c_w": 0.8, "c_rand": 0.1, "oe": 0.02}

    def w(**kw):
        d = dict(base); d.update(kw); return d
    assert classify_phi(w()) == "ALIGNED"
    assert classify_phi(w(ang_pca=40, ang_core=40)) == "DIVERGENT"
    assert classify_phi(w(ang_pca=40)) == "MIXED_GEOMETRY" # off pca not core
    assert classify_phi(w(c_w=0.1)) == "ECHO"              # not causal
    assert classify_phi(w(c_w=0.8, c_rand=0.7)) == "ECHO"  # no margin over floor
    assert classify_phi(w(c_w=float("nan"))) == "NO_PATCH_ROOM"
    assert classify_phi(w(c_rand=float("inf"))) == "NO_PATCH_ROOM"
    assert classify_phi(w(lin_r2=0.2, knn_r2=0.8)) == "INTERPRETER_GAP"
    assert classify_phi(w(lin_r2=0.2, knn_r2=0.2)) == "NOT_DECODABLE"
    assert classify_phi(w(pphi_std=0.01)) == "PREDICATE_VACUOUS"

    assert aggregate_phi(["ALIGNED"] * 3 + ["MIXED_GEOMETRY"],
                         [True] * 4) == "ALIGNED"
    assert aggregate_phi(["DIVERGENT"] * 3 + ["ECHO"], [True] * 4) == "DIVERGENT"
    assert aggregate_phi(["DIVERGENT"] * 3 + ["ECHO"],
                         [False] * 4) == "OBS_EXACT_DRIFT"
    # Soundness must hold on the reproducing seeds, not just anywhere.
    assert aggregate_phi(["DIVERGENT"] * 3 + ["ECHO"],
                         [True, True, False, True]) == "OBS_EXACT_DRIFT"
    assert aggregate_phi(["ALIGNED"] * 3 + ["ECHO"],
                         [False] * 4) == "OBS_EXACT_DRIFT"
    assert aggregate_phi(["ECHO"] * 3 + ["ALIGNED"],
                         [False] * 4) == "OBS_EXACT_DRIFT"
    assert aggregate_phi(["NO_PATCH_ROOM"] * 3 + ["ALIGNED"],
                         [False] * 4) == "NO_PATCH_ROOM"
    assert aggregate_phi(["ALIGNED", "DIVERGENT", "ECHO", "MIXED_GEOMETRY"],
                         [True] * 4) == "SEED_UNSTABLE"

    assert decide({"a": "DIVERGENT", "b": "ALIGNED"}).startswith("DIVERGENT")
    assert decide({"a": "OBS_EXACT_DRIFT", "b": "ALIGNED"}) == \
        "OBS_EXACT_DRIFT(a)"
    assert decide({"a": "MIXED_GEOMETRY", "b": "ALIGNED"}) == \
        "MIXED_GEOMETRY(a)"
    assert decide({"a": "NO_PATCH_ROOM", "b": "ALIGNED"}).startswith(
        "PARTIAL_ALIGNED")
    assert decide({"a": "ALIGNED", "b": "ALIGNED", "c": "ALIGNED"}) == \
        "ALIGNED(a,b,c)"
    assert decide({"a": "ALIGNED", "b": "ALIGNED"}).startswith(
        "PARTIAL_ALIGNED")
    assert decide({"a": "ALIGNED", "b": "ECHO"}).startswith("PARTIAL_ALIGNED")
    assert decide({"a": "ALIGNED", "b": "INTERPRETER_GAP"}).startswith(
        "PARTIAL_ALIGNED")
    assert decide({"a": "ECHO", "b": "ECHO"}).startswith("ECHO")
    assert decide({"a": "INTERPRETER_GAP", "b": "NOT_DECODABLE"}).startswith(
        "INTERPRETER_LIMIT")
    assert decide({"a": "PREDICATE_VACUOUS"}) == "PREDICATE_VACUOUS(a)"
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

    print("=== Experiment 29: predicate-targeting measurement pilot ===")
    print(f"target={proc.name} m={M} LAYER={LAYER} seeds={SEEDS}")
    print("predicates:", ", ".join(masks))
    print("Built from library primitives; exact p_phi is eval-only.\n")

    per_seed = {}
    for s in SEEDS:
        out, k_core = run_seed(model, proc, cfg, s, masks)
        per_seed[s] = out
        print(f"[seed {s}] k_core={k_core}")
        print("  phi                 std  linR2 knnR2  a(PCA) a(core)  c_w  "
              "c_rand  oeT  oeS   oe   -> verdict")
        for name, m in out.items():
            print(f"  {name:<18} {m['pphi_std']:.3f} {m['lin_r2']:>5.2f} "
                  f"{m['knn_r2']:>5.2f}  {m['ang_pca']:>5.1f}  {m['ang_core']:>5.1f}"
                  f"  {m['c_w']:>5.2f} {m['c_rand']:>5.2f}  "
                  f"{m['oe_tgt']:.3f} {m['oe_src']:.3f} {m['oe']:.3f}  "
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
              "structure for these registered predicate directions under the "
              "affine/kNN interpreter and Euclidean rank-1 patch class. "
              "Pre-register the predicate-CEGAR loop.")
    elif decision.startswith("MIXED_GEOMETRY"):
        print("  A causal predicate direction separates from one comparator but "
              "not both. Report which comparator moved; do not claim full "
              "PCA/core divergence.")
    elif decision.startswith("NO_PATCH_ROOM"):
        print("  The full patch does not create enough predicate-level recovery "
              "room for one or more predicates. Change predicate, horizon, "
              "PairSet, patch point, or process before interpreting geometry.")
    elif decision.startswith("ALIGNED"):
        print("  Every causal predicate direction sits inside PCA~core — the "
              "registered predicate directions add no new rank-1 residual "
              "geometry under this interpreter/intervention class on "
              "near-mimicry pstack (consistent with exp 26). EVIDENCE to "
              "design a richer toy backward from predicates; do not keep "
              "mining pstack with this pilot. NOT a failure.")
    elif decision.startswith("PARTIAL_ALIGNED"):
        print("  Some interpretable predicate directions align with PCA~core, "
              "but coverage is too low or incomplete because other predicates "
              "were excluded, were not causally interpretable, or did not "
              "reproduce. Treat this as limited coverage, not as a routing "
              "result away from pstack.")
    elif decision.startswith("ECHO"):
        print("  Predicates decode but patching does not move them: "
              "decode-sufficiency != causal sufficiency at predicate "
              "granularity under this interpreter/intervention stack. Do not "
              "interpret predicate-readout geometry without a causal-control "
              "check.")
    elif decision.startswith("OBS_EXACT_DRIFT"):
        print("  Observable predicate endpoints do not match exact predicate "
              "truth for a reproduced headline state. Repair predicate "
              "scoring/calibration before geometry claims.")
    elif decision.startswith("INTERPRETER_LIMIT"):
        print("  The tested affine/kNN interpreters do not recover one or more "
              "predicates well enough for causal geometry claims.")
    elif decision.startswith("PREDICATE_VACUOUS"):
        print("  One or more predicates barely vary on this distribution. "
              "Change predicate, horizon, or distribution.")
    else:
        print("  Seed-unstable across the registered four seeds. Report per-phi; "
              "the geometry did not reproduce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
