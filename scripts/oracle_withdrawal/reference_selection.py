"""
Oracle-withdrawal experiment 2: hidden-oracle reference selection.

This script selects a compact reference candidate using only observable
model behavior, then reveals exact closure for audit. The full patch is a
ceiling/control and cannot be selected as a successful abstraction reference.
"""

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abstraction import CompletionPLS, PCAAbstraction, center_by_position
from battery import Exact, Refs, cegar_loop
from discover import (PairSet, mined_direction, principal_angles_deg,
                      self_checks)
from expcommon import LAYER, load_model
from midstream import chain_probs, kl_by_horizon, orthonormal, stream_to
from processes import PROCESSES
from scripts.oracle_withdrawal.substrate import observable_depths

REGISTERED_CFG = {
    "process": "pstack",
    "seq_len": 40,
    "burn_in": 4,
    "d_model": 64,
    "layers": 4,
    "m": 3,
    "seed": 0,
}

M = 3
MM = 3
TS_DISC = (10, 18, 26, 34)
TS_HELD = (12, 20, 28, 36)
PAIRS_DISC = 320
PAIRS_EVAL = 1024
PAIRS_HELD = 512
PAIR_POOL = 900
BASIS_SEQS = 240
EPS = 0.05
EPS_DROP = 0.01
K_MAX = 10
MIN_DIM = 1
MAX_DIM = 8
OBS_EVAL_MIN = 0.70
OBS_HELD_MIN = 0.60
OBS_STRATUM_MIN = 0.50
TIE_MARGIN = 0.03
EXACT_MIN = 0.70
OBS_EXACT_GAP_MAX = 0.15
EXACT_BEST_MARGIN = 0.10
MIN_STRATUM_COUNT = 20
# Per-stratum non-vacuity guard: a covered depth stratum's full-patch
# recovery room (d0 - df) must clear this fraction of the global room,
# else its closure denominator is degenerate (STRATA_VACUOUS). This is the
# per-stratum analog of discover.py's `assert D0 > D_full`.
STRATUM_DENOM_MIN_FRAC = 0.05
# Diagnostic-only: a REFERENCE_AMBIGUITY is CONFIRMED when tied candidates
# differ in exact closure OR span subspaces farther apart than this max
# principal angle (deg). Both ambiguity sub-branches are NO-GO, so this
# threshold refines the writeup label, not the transfer decision.
AMBIG_ANGLE_MAX = 10.0


def require_registered_config(cfg, force_invalid=False):
    mismatches = [(k, cfg.get(k), v) for k, v in REGISTERED_CFG.items()
                  if cfg.get(k) != v]
    if mismatches and not force_invalid:
        print("HALT: wrong checkpoint config for Exp24.")
        for key, got, want in mismatches:
            print(f"  {key}: got {got!r}, expected {want!r}")
        return False
    if mismatches:
        print("NOTE: exploratory run with config mismatches:")
        for key, got, want in mismatches:
            print(f"  {key}: got {got!r}, registered {want!r}")
    return True


def model_completion_basis(model, proc, cfg, n_seqs, seed):
    """Residual rows and observable model completion distributions.

    The labels are model chain probabilities under the unpatched run, not
    exact process m-grams. This keeps proposal fitting oracle-free.
    """
    rng = np.random.default_rng(seed)
    X = proc.sample(n_seqs, cfg["seq_len"], rng)
    S = stream_to(model, torch.from_numpy(X), LAYER).double().numpy()
    conts = np.asarray(list(product(range(proc.V), repeat=M)), dtype=np.int64)
    rows, labels, pos = [], [], []
    for t in TS_DISC:
        Xc = np.repeat(X[:, None, :], len(conts), axis=1).copy()
        Xc[:, :, t + 1:t + 1 + M] = conts[None, :, :]
        q, _ = chain_probs(model, Xc, LAYER, None, t, M, proc.V)
        rows.append(S[:, t])
        labels.append(q)
        pos.append(np.full(len(X), t, dtype=np.int64))
    R = np.concatenate(rows)
    Q = np.concatenate(labels)
    P = np.concatenate(pos)
    Rc = center_by_position(R, P, np.ones(len(R), dtype=bool))
    return Rc, Q


def fixed_mined_basis(ps, refs, dim):
    """Observable weighted prefix-delta basis without accept/coarsen logic."""
    Q = np.zeros((ps.d, 0))
    weights = kl_by_horizon(refs.q_un, refs.q_src, refs.V, refs.m_full)[MM]
    for _ in range(dim):
        v = mined_direction(ps, Q, weights)
        Q = np.hstack([Q, v[:, None]])
    return Q


def build_candidates(model, proc, cfg, disc, refs_d, seed):
    d = cfg["d_model"]
    k_cegar, Qc, _ = cegar_loop(model, disc, refs_d, d, EPS, K_MAX, MM,
                                eps_drop=EPS_DROP)
    k_ref = int(np.clip(k_cegar, MIN_DIM, MAX_DIM))

    Rb, Qobs = model_completion_basis(model, proc, cfg, BASIS_SEQS,
                                      seed + 555)
    pca = PCAAbstraction(Rb)
    pls = CompletionPLS(Rb, Qobs)
    with torch.no_grad():
        Wtok = model.tok.weight.double().numpy()

    bases = {
        "cegar": Qc[:, :k_ref],
        "pca": pca.Vt[:k_ref].T,
        "obs_pls": orthonormal(pls.whiten @ pls.U[:, :k_ref]),
        "delta": fixed_mined_basis(disc, refs_d, k_ref),
        "emb": orthonormal(Wtok.T)[:, :min(k_ref, proc.V)],
        "rand": orthonormal(np.random.default_rng(seed).standard_normal(
            (d, k_ref))),
    }
    candidates = {
        "full": {"P": np.eye(d), "Q": None, "dim": d, "selectable": False,
                 "kind": "ceiling"},
    }
    for name, Q in bases.items():
        if Q.shape[1] == 0:
            P = np.zeros((d, d))
        else:
            Q = orthonormal(Q)
            P = Q @ Q.T
        candidates[name] = {
            "P": P,
            "Q": Q,
            "dim": int(Q.shape[1]),
            "selectable": name != "rand" and Q.shape[1] >= MIN_DIM,
            "kind": "candidate" if name != "rand" else "destructive",
        }
    return candidates, k_cegar, k_ref


def depth_index(ps):
    depths = np.empty(ps.n, dtype=np.int64)
    for t, idx in ps.groups:
        for i in idx:
            depths[i] = observable_depths(ps.Xe[ps.a[i]])[t]
    return depths


def build_strata_layout(refs, ps):
    """Candidate-independent per-stratum setup on the eval pair set.

    Computes, once, the q-independent unpatched/full KL rows and, per
    observable depth stratum, its row indices, mean unpatched KL (d0),
    mean full-patch KL (df), and recovery room (denom = d0 - df). Returns
    the strata dict and the global recovery room.
    """
    depths = depth_index(ps)
    rows_un = kl_by_horizon(refs.q_un, refs.q_src, refs.V, refs.m_full)[MM]
    rows_full = kl_by_horizon(refs.q_full, refs.q_src, refs.V,
                              refs.m_full)[MM]
    global_denom = float(rows_un.mean() - rows_full.mean())
    strata = {}
    for depth in range(4):
        idx = np.where(depths == depth)[0]
        d0 = float(rows_un[idx].mean()) if len(idx) else float("nan")
        df = float(rows_full[idx].mean()) if len(idx) else float("nan")
        strata[depth] = {"idx": idx, "n": int(len(idx)),
                         "d0": d0, "df": df, "denom": d0 - df}
    return strata, global_denom


def strata_guard(strata, global_denom):
    """Pre-flight substrate guards on the stratified observable scale.

    Returns (ok, branch, detail). Every registered depth stratum 0..3 must
    be covered (>= MIN_STRATUM_COUNT rows) so the stratified score does not
    silently omit it, and non-vacuous (recovery room >= STRATUM_DENOM_MIN_FRAC
    of the global room) so its closure denominator is not degenerate.
    """
    for d in range(4):
        if strata[d]["n"] < MIN_STRATUM_COUNT:
            return (False, "STRATA_TOO_SPARSE",
                    f"depth {d} has {strata[d]['n']} < {MIN_STRATUM_COUNT} "
                    "rows; the deepest strata are uncovered")
    floor = STRATUM_DENOM_MIN_FRAC * global_denom
    for d in range(4):
        if strata[d]["denom"] <= floor:
            return (False, "STRATA_VACUOUS",
                    f"depth {d} recovery room {strata[d]['denom']:.4f} "
                    f"<= floor {floor:.4f} ({STRATUM_DENOM_MIN_FRAC:.0%} of "
                    f"global {global_denom:.4f})")
    return True, "OK", "all depth strata covered and non-vacuous"


def candidate_strata_min(refs, q, strata):
    """Minimum observable closure over the (guarded) depth strata 0..3.

    The guard guarantees all four strata are covered and non-vacuous, so
    every stratum contributes; only the q-dependent patched KL is computed
    here (d0/df/denom come precomputed from the layout).
    """
    rows_q = kl_by_horizon(q, refs.q_src, refs.V, refs.m_full)[MM]
    vals = []
    for d in range(4):
        idx = strata[d]["idx"]
        dq = float(rows_q[idx].mean())
        vals.append((strata[d]["d0"] - dq) / strata[d]["denom"])
    return min(vals)


def max_principal_angle(qa, qb):
    """Largest principal angle (deg) between two orthonormal column bases."""
    return float(principal_angles_deg(qa, qb).max())


def evaluate_observable(model, candidates, refs_e, refs_h, eval_ps, held_ps,
                        strata):
    rows = []
    q_eval = {}
    for name, cand in candidates.items():
        q_e = eval_ps.run(model, cand["P"])
        q_h = held_ps.run(model, cand["P"])
        q_eval[name] = q_e
        obs_eval = refs_e.obs(q_e, MM)
        obs_held = refs_h.obs(q_h, MM)
        strata_min = candidate_strata_min(refs_e, q_e, strata)
        score = min(obs_eval, obs_held, strata_min)
        rows.append({
            "name": name,
            "dim": cand["dim"],
            "kind": cand["kind"],
            "selectable": cand["selectable"],
            "obs_eval": obs_eval,
            "obs_held": obs_held,
            "strata_min": strata_min,
            "score": score,
        })
    return rows, q_eval


def select_observable(rows):
    eligible = [
        r for r in rows
        if r["selectable"]
        and r["obs_eval"] >= OBS_EVAL_MIN
        and r["obs_held"] >= OBS_HELD_MIN
        and r["strata_min"] >= OBS_STRATUM_MIN
    ]
    if not eligible:
        full = next(r for r in rows if r["name"] == "full")
        if full["obs_eval"] >= 0.90:
            return {"branch": "FULL_PATCH_FALLBACK", "selected": None,
                    "tied": []}
        return {"branch": "NO_TRUSTWORTHY_REFERENCE", "selected": None,
                "tied": []}
    best = max(r["score"] for r in eligible)
    tied = [r for r in eligible if best - r["score"] <= TIE_MARGIN]
    if len(tied) > 1:
        return {"branch": "REFERENCE_AMBIGUITY", "selected": None,
                "tied": [r["name"] for r in tied]}
    return {"branch": "SELECTED", "selected": tied[0]["name"], "tied": []}


def classify_exact_selection(row, exact_sel, exact_best):
    gap = abs(row["obs_eval"] - exact_sel)
    if exact_sel < EXACT_MIN:
        branch = "SELECTED_REFERENCE_TOO_WEAK"
    elif gap > OBS_EXACT_GAP_MAX:
        branch = "OBS_EXACT_DISAGREEMENT"
    elif exact_best - exact_sel > EXACT_BEST_MARGIN:
        branch = "ORACLE_REVEAL_INVERSION"
    else:
        branch = "REFERENCE_SELECTED_CORRECTLY"
    return branch, gap


def classify_exact_ambiguity(tied_exact, max_angle):
    """Diagnostic split of an observable REFERENCE_AMBIGUITY.

    CONFIRMED iff the tied candidates differ in exact closure (spread >
    EXACT_BEST_MARGIN) OR span genuinely distinct subspaces (max principal
    angle > AMBIG_ANGLE_MAX) — the latter being the ORACLE_WITHDRAWAL
    "disagree on downstream rho" case. BENIGN only when they agree on both.
    Both sub-branches are NO-GO; this labels the writeup, not the decision.
    """
    spread = max(tied_exact) - min(tied_exact)
    if spread > EXACT_BEST_MARGIN or max_angle > AMBIG_ANGLE_MAX:
        return "REFERENCE_AMBIGUITY_CONFIRMED"
    return "REFERENCE_AMBIGUITY_BENIGN"


def exact_audit(model, eval_ps, rows, q_eval, selection, candidates):
    exact = Exact(eval_ps, model, M)
    exact_rows = {}
    for r in rows:
        if r["name"] == "full":
            continue
        exact_cl = exact.closure(q_eval[r["name"]], MM)
        exact_rows[r["name"]] = exact_cl

    selectable = [r for r in rows if r["selectable"]]
    exact_best = max(exact_rows[r["name"]] for r in selectable)

    if selection["branch"] == "SELECTED":
        name = selection["selected"]
        row = next(r for r in rows if r["name"] == name)
        exact_sel = exact_rows[name]
        branch, gap = classify_exact_selection(row, exact_sel, exact_best)
        return {
            "branch": branch,
            "selected": name,
            "exact_selected": exact_sel,
            "obs_exact_gap": gap,
            "exact_best": exact_best,
            "exact_rows": exact_rows,
        }

    if selection["branch"] == "REFERENCE_AMBIGUITY":
        tied = selection["tied"]
        tied_angles = {}
        max_angle = 0.0
        for i in range(len(tied)):
            for j in range(i + 1, len(tied)):
                ang = max_principal_angle(candidates[tied[i]]["Q"],
                                          candidates[tied[j]]["Q"])
                tied_angles[(tied[i], tied[j])] = ang
                max_angle = max(max_angle, ang)
        return {
            "branch": classify_exact_ambiguity([exact_rows[n] for n in tied],
                                               max_angle),
            "selected": None,
            "exact_selected": None,
            "obs_exact_gap": None,
            "exact_best": exact_best,
            "exact_rows": exact_rows,
            "tied_angles": tied_angles,
            "max_tied_angle": max_angle,
        }

    return {
        "branch": selection["branch"],
        "selected": None,
        "exact_selected": None,
        "obs_exact_gap": None,
        "exact_best": exact_best,
        "exact_rows": exact_rows,
    }


def selftest():
    rows = [
        {"name": "full", "selectable": False, "obs_eval": 1.0,
         "obs_held": 1.0, "strata_min": 1.0, "score": 1.0},
        {"name": "a", "selectable": True, "obs_eval": 0.82,
         "obs_held": 0.78, "strata_min": 0.71, "score": 0.71},
        {"name": "b", "selectable": True, "obs_eval": 0.40,
         "obs_held": 0.30, "strata_min": 0.20, "score": 0.20},
    ]
    assert select_observable(rows)["selected"] == "a"
    rows[1]["score"] = 0.70
    rows[2].update({"obs_eval": 0.83, "obs_held": 0.76,
                    "strata_min": 0.69, "score": 0.69})
    assert select_observable(rows)["branch"] == "REFERENCE_AMBIGUITY"
    rows[1].update({"obs_eval": 0.1, "obs_held": 0.1, "strata_min": 0.1,
                    "score": 0.1})
    rows[2].update({"obs_eval": 0.1, "obs_held": 0.1, "strata_min": 0.1,
                    "score": 0.1})
    assert select_observable(rows)["branch"] == "FULL_PATCH_FALLBACK"
    row = {"obs_eval": 0.80}
    assert classify_exact_selection(row, 0.79, 0.84)[0] == \
        "REFERENCE_SELECTED_CORRECTLY"
    assert classify_exact_selection(row, 0.69, 0.84)[0] == \
        "SELECTED_REFERENCE_TOO_WEAK"
    assert classify_exact_selection(row, 0.61, 0.66)[0] == \
        "SELECTED_REFERENCE_TOO_WEAK"
    assert classify_exact_selection(row, 0.73, 0.86)[0] == \
        "ORACLE_REVEAL_INVERSION"
    assert classify_exact_selection(row, 0.64, 0.66)[0] == \
        "SELECTED_REFERENCE_TOO_WEAK"
    assert classify_exact_selection({"obs_eval": 0.90}, 0.74, 0.78)[0] == \
        "OBS_EXACT_DISAGREEMENT"
    # ambiguity is BENIGN only when tied candidates agree on BOTH exact
    # closure and subspace; either disagreement -> CONFIRMED.
    assert classify_exact_ambiguity([0.72, 0.81], 5.0) == \
        "REFERENCE_AMBIGUITY_BENIGN"
    assert classify_exact_ambiguity([0.72, 0.83], 5.0) == \
        "REFERENCE_AMBIGUITY_CONFIRMED"          # exact-closure spread
    assert classify_exact_ambiguity([0.72, 0.75], 30.0) == \
        "REFERENCE_AMBIGUITY_CONFIRMED"          # distinct subspaces

    # principal-angle helper: identical basis ~ 0 deg, orthogonal ~ 90 deg
    e = np.eye(4)
    assert max_principal_angle(e[:, :2], e[:, :2]) < 1e-6
    assert abs(max_principal_angle(e[:, :2], e[:, 2:]) - 90.0) < 1e-6

    # strata guards: coverage then non-vacuity
    def _strata(ns, denoms):
        return {d: {"idx": np.arange(ns[d]), "n": ns[d], "d0": 1.0,
                    "df": 1.0 - denoms[d], "denom": denoms[d]}
                for d in range(4)}
    ok, branch, _ = strata_guard(_strata([50, 50, 50, 50],
                                         [1.0, 1.0, 1.0, 1.0]), 2.0)
    assert ok and branch == "OK"
    ok, branch, _ = strata_guard(_strata([50, 50, 50, 5],
                                         [1.0, 1.0, 1.0, 1.0]), 2.0)
    assert not ok and branch == "STRATA_TOO_SPARSE"
    ok, branch, _ = strata_guard(_strata([50, 50, 50, 50],
                                         [1.0, 1.0, 1.0, 0.05]), 2.0)
    assert not ok and branch == "STRATA_VACUOUS"   # 0.05 <= 0.05*2.0
    print("selftest passed: selection, audit, strata, and angle branches")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/pstack-L4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force-invalid", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    if not require_registered_config(cfg, args.force_invalid):
        return 1
    proc = PROCESSES[cfg["process"]]()
    model = load_model(args.outdir, cfg, proc)

    # PairSet carries exact fields by library design; selection below uses
    # only residuals/tokens/model completion behavior until exact_audit().
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, args.seed + 111,
                   PAIR_POOL, layer=LAYER, ts=TS_DISC)
    eval_ps = PairSet(model, proc, cfg, PAIRS_EVAL, M, args.seed + 222,
                      PAIR_POOL, layer=LAYER, ts=TS_DISC)
    held_ps = PairSet(model, proc, cfg, PAIRS_HELD, M, args.seed + 333,
                      PAIR_POOL, layer=LAYER, ts=TS_HELD)
    self_checks(model, eval_ps, LAYER, M, proc.V)

    print("=== Experiment 24: hidden-oracle reference selection ===")
    print(f"target={proc.name} outdir={args.outdir} m={M} LAYER={LAYER}")
    print("Exact oracle has not been used for selection.\n")

    refs_d = Refs(disc, model, disc.d, M)
    refs_e = Refs(eval_ps, model, eval_ps.d, M)
    refs_h = Refs(held_ps, model, held_ps.d, M)

    # Stratified observable scale: verify coverage and non-vacuity before
    # any candidate is scored, so a substrate miss surfaces as itself
    # rather than corrupting strata_min or hiding in NO_TRUSTWORTHY_REFERENCE.
    strata, global_denom = build_strata_layout(refs_e, eval_ps)
    assert global_denom > 0, "observable scale degenerate: D0 <= D_full"
    print("[observable depth strata on eval set]")
    print("depth  n     d0      df      recovery_room")
    for d in range(4):
        s = strata[d]
        print(f"  {d}   {s['n']:>4d}  {s['d0']:>6.3f}  {s['df']:>6.3f}  "
              f"{s['denom']:>9.4f}")
    ok, guard_branch, guard_detail = strata_guard(strata, global_denom)
    if not ok:
        print(f"\nHALT: {guard_branch} — {guard_detail}")
        print("  NO-GO: stratified observable substrate is not usable; "
              "register a substrate repair before reference selection.")
        return 1

    candidates, k_cegar, k_ref = build_candidates(model, proc, cfg, disc,
                                                  refs_d, args.seed)
    print(f"\nCEGAR k*={k_cegar}; matched candidate dimension={k_ref}")
    # Variance-mimicry lens (exps 6/8): report cegar-vs-pca subspace angles
    # unconditionally — on a richer substrate they may coincide or diverge.
    qc, qp = candidates["cegar"]["Q"], candidates["pca"]["Q"]
    if qc is not None and qp is not None and qc.shape[1] and qp.shape[1]:
        ang = principal_angles_deg(qc, qp)
        print(f"cegar-vs-pca principal angles (deg): "
              f"{np.array2string(np.round(ang, 1))}")

    rows, q_eval = evaluate_observable(model, candidates, refs_e, refs_h,
                                       eval_ps, held_ps, strata)
    selection = select_observable(rows)

    print("\n[observable candidate table]")
    print("name        dim kind        obs_eval obs_held strata_min score")
    for r in sorted(rows, key=lambda x: x["score"], reverse=True):
        print(f"{r['name']:<10} {r['dim']:>3d} {r['kind']:<10} "
              f"{r['obs_eval']:>8.3f} {r['obs_held']:>8.3f} "
              f"{r['strata_min']:>10.3f} {r['score']:>5.3f}")
    print(f"\nOBSERVABLE_SELECTION: {selection['branch']}")
    if selection["selected"] is not None:
        print(f"selected={selection['selected']}")
    if selection["tied"]:
        print("tied=" + ",".join(selection["tied"]))

    audit = exact_audit(model, eval_ps, rows, q_eval, selection, candidates)
    print("\n[exact audit — revealed after observable selection]")
    for name, val in sorted(audit["exact_rows"].items(),
                            key=lambda kv: kv[1], reverse=True):
        print(f"{name:<10} exact_closure={val:.3f}")
    print(f"\nAUDIT_BRANCH: {audit['branch']}")
    if audit["selected"] is not None:
        print(f"selected={audit['selected']} "
              f"exact={audit['exact_selected']:.3f} "
              f"obs_exact_gap={audit['obs_exact_gap']:.3f} "
              f"exact_best={audit['exact_best']:.3f}")
    if audit.get("tied_angles"):
        print(f"max tied principal angle={audit['max_tied_angle']:.1f} deg "
              f"(CONFIRMED if > {AMBIG_ANGLE_MAX:.0f})")
        for (a, b), val in audit["tied_angles"].items():
            print(f"  angle({a},{b})={val:.1f} deg")

    print("\nDECISION:")
    if audit["branch"] == "REFERENCE_SELECTED_CORRECTLY":
        print("  GO: compact reference selected under hidden-oracle "
              "discipline; preregister battery transfer with earned "
              "reference.")
    elif audit["branch"] in {
        "FULL_PATCH_FALLBACK",
        "NO_TRUSTWORTHY_REFERENCE",
        "REFERENCE_AMBIGUITY_CONFIRMED",
        "REFERENCE_AMBIGUITY_BENIGN",
        "SELECTED_REFERENCE_TOO_WEAK",
        "OBS_EXACT_DISAGREEMENT",
        "ORACLE_REVEAL_INVERSION",
    }:
        print("  NO-GO: do not run battery transfer yet; register the typed "
              f"follow-up for {audit['branch']}.")
    else:
        raise AssertionError(f"unregistered audit branch: {audit['branch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
