"""
Oracle-withdrawal experiment 5: battery transfer under the earned reference
(Block 3).

Runs the six-member diagnostic battery on pstack using the earned cegar core
(selected/audited observably in exps 24-26) as the trusted reference, over 4
fresh seeds. Honest framing (exp-26 handoff): pstack is near variance-mimicry
and the anchor is declared, so the claim on offer is SYSTEM-LEVEL — does the
end-to-end hidden-oracle workflow yield a usable battery? — not new battery
physics.

Observable members (M1 closure, M2 rho, M3 held-out gain, M4 shift-retention,
M6 CEGAR staircase) are computed and printed before the exact reveal. The
exact oracle is read for M5 (accepted-cell obs/exact agreement) and the
per-process threshold recalibration (exp-19 precedent, frozen before the
verdict). The reference is not re-selected with the oracle here. Reuses
build_candidates by import; rho via a local observable helper.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from battery import Exact, Refs, cegar_staircase, jeffreys_rows, shift_retention
from discover import PairSet, self_checks
from expcommon import LAYER, load_model
from midstream import kl_by_horizon, marginal
from processes import PROCESSES
from scripts.oracle_withdrawal.reference_selection import (
    M, MM, PAIR_POOL, PAIRS_DISC, PAIRS_EVAL, TS_DISC, TS_HELD,
    build_candidates, require_registered_config,
)

SEEDS = (200, 201, 202, 203)             # 4 fresh seeds, disjoint from exps 24-26
ANCHOR = "cegar"
PROBES = ("pca", "delta", "emb", "rand", "full")
EQUIV_NAMES = ("pca", "delta")           # near-coincident estimates of the reference
INTERMEDIATE = "emb"                      # directionally-distinct intermediate probe (M2/M5)
RHO_EQUIV = 0.25                         # exp-26 transferred bands
RHO_DIST = 0.50
CLOSURE_MIN = 0.70                       # strong, non-vacuous reference
HELDOUT_GAP_MAX = 0.10                   # M3 no discovery-position overfitting
OBS_EXACT_BAND = 0.10                    # M5 transferred (Mess3 0.10 / Dyck 0.073)
LENIENT_GAP = 0.10                       # M2 emb over-acceptance: exact gap > this
R_MIN = 0.80                             # M4 shift retention floor
COMPETENCE_BAND = 0.030                  # exp-23 competence gate
EPS_GRID = (0.01, 0.02, 0.05, 0.10)
K_MAX = 10
SHIFT_INIT_STATE = 22                    # pstack (mode 1, full depth-3 stack)


def _mnorm(q, V, mm, m_full):
    qm = marginal(q, V, mm, m_full)
    return qm / np.clip(qm.sum(axis=1, keepdims=True), 1e-30, None)


def rho_obs(q0, qC, qX, V, mm, m_full):
    a = _mnorm(qC, V, mm, m_full)
    b = _mnorm(qX, V, mm, m_full)
    u = _mnorm(q0, V, mm, m_full)
    return float(jeffreys_rows(a, b).mean() / jeffreys_rows(a, u).mean())


def raw_gain(refs, q_patched):
    """Raw KL reduction D0 - Dq at horizon MM (not the normalized closure):
    the member-4 'gain' whose retention ratio is R."""
    V, mf = refs.V, refs.m_full
    d0 = float(kl_by_horizon(refs.q_un, refs.q_src, V, mf)[MM].mean())
    dq = float(kl_by_horizon(q_patched, refs.q_src, V, mf)[MM].mean())
    return d0 - dq


def competence_shift(model, proc, cfg, seed, init_state):
    """Model token NLL minus exact optimal NLL on sequences sampled from the
    shifted init_state (stationary belief frame = the model's frame)."""
    L, V = cfg["seq_len"], proc.V
    X = proc.sample(500, L, np.random.default_rng(seed), init_state=init_state)
    with torch.no_grad():
        total, count = 0.0, 0
        for i in range(0, len(X), 256):
            logits = model(torch.from_numpy(X[i:i + 256]))
            tgt = torch.from_numpy(X[i:i + 256, 1:]).reshape(-1)
            total += F.cross_entropy(logits[:, :-1].reshape(-1, V), tgt,
                                     reduction="sum").item()
            count += tgt.numel()
    opt, opt_count = 0.0, 0
    for row in X:
        b = proc.pi.copy()
        for t, s in enumerate(row[:-1]):
            b, _ = proc.belief_update(b, s)
            opt -= np.log((b @ proc.T[int(row[t + 1])]).sum())
            opt_count += 1
    return total / count - opt / opt_count


def run_seed(model, proc, cfg, seed):
    d, V = cfg["d_model"], proc.V
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + 111, PAIR_POOL,
                   layer=LAYER, ts=TS_DISC)
    eval_ps = PairSet(model, proc, cfg, PAIRS_EVAL, M, seed + 222, PAIR_POOL,
                      layer=LAYER, ts=TS_DISC)
    held_ps = PairSet(model, proc, cfg, PAIRS_EVAL, M, seed + 333, PAIR_POOL,
                      layer=LAYER, ts=TS_HELD)
    shift_ps = PairSet(model, proc, cfg, PAIRS_EVAL, M, seed + 444, PAIR_POOL,
                       layer=LAYER, ts=TS_DISC, init_state=SHIFT_INIT_STATE)
    self_checks(model, eval_ps, LAYER, M, V)

    refs_d = Refs(disc, model, d, M)
    refs_e = Refs(eval_ps, model, d, M)
    refs_h = Refs(held_ps, model, d, M)
    refs_s = Refs(shift_ps, model, d, M)
    candidates, _, k_ref = build_candidates(model, proc, cfg, disc, refs_d, seed)
    Pc = candidates[ANCHOR]["P"]
    Pfull = np.eye(d)
    P = {X: candidates[X]["P"] for X in PROBES}

    # ---- observable members ------------------------------------------------
    q_e = eval_ps.run(model, Pc)
    m1 = refs_e.obs(q_e, MM)                         # M1 reference closure
    q_un = refs_e.q_un
    q_probe = {X: eval_ps.run(model, P[X]) for X in PROBES}
    rho = {X: rho_obs(q_un, q_e, q_probe[X], V, MM, M) for X in PROBES}  # M2
    obs_cl = {ANCHOR: m1}
    obs_cl.update({X: refs_e.obs(q_probe[X], MM) for X in PROBES})  # for M5

    q_h = held_ps.run(model, Pc)
    m3_obs = refs_h.obs(q_h, MM)                     # M3 held-out observable

    # M4 shift-retention (raw gains; C = full ceiling)
    gC_base = raw_gain(refs_e, q_e)
    gC_shift = raw_gain(refs_s, shift_ps.run(model, Pc))
    gF_base = raw_gain(refs_e, eval_ps.run(model, Pfull))
    gF_shift = raw_gain(refs_s, shift_ps.run(model, Pfull))
    R = shift_retention(gC_shift, gC_base, gF_shift, gF_base)
    comp_shift = competence_shift(model, proc, cfg, seed + 777, SHIFT_INIT_STATE)

    stair = cegar_staircase(model, disc, refs_d, d, EPS_GRID, K_MAX, MM)  # M6

    # ---- exact (M5 + recalibration; revealed after the observable block) ---
    exact_e = Exact(eval_ps, model, M)
    exact_h = Exact(held_ps, model, M)
    exact_cl = {ANCHOR: exact_e.closure(q_e, MM)}
    exact_cl.update({X: exact_e.closure(q_probe[X], MM) for X in PROBES})
    m3_exact = exact_h.closure(q_h, MM)

    return {
        "seed": seed, "k_ref": k_ref,
        "m1": m1, "rho": rho, "obs_cl": obs_cl,
        "m3_obs": m3_obs, "m3_exact": m3_exact,
        "R": R, "gF_base": gF_base, "gF_shift": gF_shift,
        "comp_shift": comp_shift, "stair": stair, "exact_cl": exact_cl,
    }


# --------- per-member verdicts (status in {PASS, RECALIBRATE, FAIL}) --------
# Aggregated worst-case over seeds. RECALIBRATE is a non-fail pass-with-note
# (the exp-19 per-process recalibration), distinct from FAIL.

def verdict_m1(S):
    worst = min(s["m1"] for s in S)
    return ("PASS" if worst >= CLOSURE_MIN else "FAIL",
            f"min closure {worst:.3f} >= {CLOSURE_MIN}")


def verdict_m2(S):
    equiv_max = max(s["rho"][n] for s in S for n in EQUIV_NAMES)
    rand_min = min(s["rho"]["rand"] for s in S)
    if not (equiv_max <= RHO_EQUIV and rand_min >= RHO_DIST):
        return "FAIL", (f"RHO_BANDS_FAIL: equiv_max {equiv_max:.3f}, "
                        f"rand_min {rand_min:.3f}")
    # inherited lenient check on the directionally-distinct intermediate
    # probe emb: exact says meaningfully weaker than the core, yet rho reads
    # equivalent. Whether rho SHOULD be magnitude-sensitive is the open
    # exp-26 question, so this is a flagged recalibrate-with-note, not a fail.
    lenient = any((s["exact_cl"][ANCHOR] - s["exact_cl"][INTERMEDIATE])
                  > LENIENT_GAP and s["rho"][INTERMEDIATE] <= RHO_EQUIV
                  for s in S)
    if lenient:
        return "RECALIBRATE", (
            f"RHO_BAND_LENIENT (hint, not validated failure): {INTERMEDIATE} "
            f"reads equivalent (rho<={RHO_EQUIV}) despite exact closure "
            f">{LENIENT_GAP} below the core — the equivalent band may be "
            f"lenient on pstack; rho's magnitude-sensitivity is the open "
            f"exp-26 question")
    return "PASS", (f"equiv_max {equiv_max:.3f}<=0.25, rand_min "
                    f"{rand_min:.3f}>=0.5; {INTERMEDIATE} not over-accepted")


def verdict_m3(S):
    worst = min(s["m3_obs"] for s in S)
    gap = max(abs(s["m3_obs"] - s["m1"]) for s in S)
    ok = worst >= CLOSURE_MIN and gap <= HELDOUT_GAP_MAX
    return ("PASS" if ok else "FAIL",
            f"min held-out closure {worst:.3f}, max |held-base| {gap:.3f}")


def verdict_m4(S):
    comp_ok = all(s["comp_shift"] <= COMPETENCE_BAND for s in S)
    clean_ok = all(s["gF_shift"] > 0 for s in S)
    R_min = min(s["R"] for s in S)
    ok = comp_ok and clean_ok and R_min >= R_MIN
    # single registered shift: a fail flags fragility under THIS shift, not
    # proven general shift-fragility (Dyck used multiple shifts; out of scope).
    return ("PASS" if ok else "FAIL",
            f"R_min {R_min:.3f}>={R_MIN}, competence_ok={comp_ok}, "
            f"clean_gain_ok={clean_ok} (single shift; fail = fragile under "
            "THIS shift, not general)")


def verdict_m5(S):
    """Three-way directional split. Signed gap = obs - exact; obs >> exact is
    the dangerous OBS_EXACT_INVERSION (observable accepts what exact rejects).
    The conservative direction (exact > obs) and benign imprecision are
    recalibrate, not fail."""
    cells = []
    for s in S:
        cells.append(s["obs_cl"][ANCHOR] - s["exact_cl"][ANCHOR])
        cells.append(s["obs_cl"][INTERMEDIATE] - s["exact_cl"][INTERMEDIATE])
        cells.append(s["m3_obs"] - s["m3_exact"])
    max_over = max(cells)                 # observable overstates exact
    max_abs = max(abs(g) for g in cells)
    if max_abs <= OBS_EXACT_BAND:
        return "PASS", (f"max |obs-exact| {max_abs:.3f} <= {OBS_EXACT_BAND} "
                        "(transfers)")
    if max_over > OBS_EXACT_BAND:
        return "FAIL", (f"OBS_EXACT_INVERSION: observable overstates exact by "
                        f"{max_over:.3f} > {OBS_EXACT_BAND}")
    return "RECALIBRATE", (
        f"max |obs-exact| {max_abs:.3f} > {OBS_EXACT_BAND} but no inversion "
        f"(max obs-over-exact {max_over:.3f}); recalibrated pstack obs/exact "
        f"band = {max_abs:.3f}")


def verdict_m6(S):
    ks_at_005 = [s["stair"][0.05] for s in S]
    monotone = all(all(s["stair"][EPS_GRID[i + 1]] <= s["stair"][EPS_GRID[i]]
                       for i in range(len(EPS_GRID) - 1)) for s in S)
    ok = monotone and all(3 <= k <= 5 for k in ks_at_005)
    # accept-only staircase != the coarsen-based discovery loop that set
    # k_ref, so k*(0.05) need not equal k_ref; the 3<=k<=5 band absorbs it.
    return ("PASS" if ok else "FAIL",
            f"k*(0.05)={ks_at_005}, weakly-decreasing={monotone} "
            "(accept-only loop; distinct from the coarsen discovery)")


VERDICTS = [("M1_closure", verdict_m1), ("M2_rho", verdict_m2),
            ("M3_heldout", verdict_m3), ("M4_shift", verdict_m4),
            ("M5_obs_exact", verdict_m5), ("M6_staircase", verdict_m6)]


def decide(results):
    fails = [name for name, st, _ in results if st == "FAIL"]
    recals = [name for name, st, _ in results if st == "RECALIBRATE"]
    if fails:
        return "TYPED_BATTERY_FAILURE(" + ",".join(fails) + ")"
    if recals:
        return "BATTERY_TRANSFERS_WITH_RECALIBRATION(" + ",".join(recals) + ")"
    return "BATTERY_TRANSFERS"


def selftest():
    # probe == anchor -> numerator 0; anchor != baseline -> denominator > 0
    assert rho_obs(np.array([[.7, .3]]), np.array([[.4, .6]]),
                   np.array([[.4, .6]]), 2, 1, 1) < 1e-9

    def _S(m1, rho_pca, rho_rand, rho_emb, ex_core, ex_emb, m3, m3e, R,
           gFs, comp, k005, ks, obs_emb=None):
        return {"m1": m1, "rho": {"pca": rho_pca, "delta": rho_pca - 0.01,
                "rand": rho_rand, "emb": rho_emb, "full": 0.1},
                "obs_cl": {ANCHOR: m1,
                           "emb": ex_emb if obs_emb is None else obs_emb},
                "exact_cl": {ANCHOR: ex_core, "emb": ex_emb},
                "m3_obs": m3, "m3_exact": m3e, "R": R, "gF_shift": gFs,
                "comp_shift": comp,
                "stair": {0.01: ks[0], 0.02: ks[1], 0.05: k005, 0.10: ks[3]}}
    # all PASS: emb reads distinct (0.55), obs~exact everywhere
    good = [_S(0.92, 0.03, 0.95, 0.55, 0.92, 0.70, 0.91, 0.90, 0.95, 1.0,
               0.01, 4, (6, 5, 4, 3))]
    res = [(n, *f(good)) for n, f in VERDICTS]
    assert decide(res) == "BATTERY_TRANSFERS", res

    # M2 lenient -> RECALIBRATE (not fail): emb exact 0.70 (gap 0.22) rho 0.18
    lenient = [_S(0.92, 0.03, 0.95, 0.18, 0.92, 0.70, 0.91, 0.90, 0.95, 1.0,
                  0.01, 4, (6, 5, 4, 3))]
    assert verdict_m2(lenient)[0] == "RECALIBRATE"
    assert decide([(n, *f(lenient)) for n, f in VERDICTS]) == \
        "BATTERY_TRANSFERS_WITH_RECALIBRATION(M2_rho)"

    # M5 conservative excess -> RECALIBRATE: emb obs 0.50 under exact 0.70
    cons = [_S(0.92, 0.03, 0.95, 0.55, 0.92, 0.70, 0.91, 0.90, 0.95, 1.0,
               0.01, 4, (6, 5, 4, 3), obs_emb=0.50)]
    assert verdict_m5(cons)[0] == "RECALIBRATE"

    # M5 inversion -> FAIL: emb observable 0.90 but exact 0.60 (obs over 0.30)
    inv = [_S(0.92, 0.03, 0.95, 0.55, 0.92, 0.60, 0.91, 0.90, 0.95, 1.0,
              0.01, 4, (6, 5, 4, 3), obs_emb=0.90)]
    assert verdict_m5(inv)[0] == "FAIL"

    # M3 overfitting and M4 fragility -> FAIL
    of = [_S(0.92, 0.03, 0.95, 0.55, 0.92, 0.70, 0.50, 0.50, 0.95, 1.0,
             0.01, 4, (6, 5, 4, 3))]
    assert verdict_m3(of)[0] == "FAIL"
    frag = [_S(0.92, 0.03, 0.95, 0.55, 0.92, 0.70, 0.91, 0.90, 0.50, 1.0,
               0.01, 4, (6, 5, 4, 3))]
    assert verdict_m4(frag)[0] == "FAIL"
    assert "TYPED_BATTERY_FAILURE" in decide([(n, *f(frag)) for n, f in VERDICTS])
    print("selftest passed: rho, six member verdicts (tri-state), decide")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/pstack-L4")
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

    print("=== Experiment 27: battery transfer under the earned reference ===")
    print(f"target={proc.name} outdir={args.outdir} m={M} LAYER={LAYER}")
    print(f"seeds={SEEDS} reference={ANCHOR} (earned, declared)")
    print("Exact oracle is read only after the observable members are "
          "printed (M5 + recalibration).\n")

    S = [run_seed(model, proc, cfg, s) for s in SEEDS]

    print("[observable members — before exact reveal]")
    print("seed  M1    rho:pca  delta  emb   rand  full | M3obs  R     "
          "compShift  k*(0.05)")
    for s in S:
        r = s["rho"]
        print(f" {s['seed']}  {s['m1']:.3f}  {r['pca']:.3f} {r['delta']:.3f} "
              f"{r['emb']:.3f} {r['rand']:.3f} {r['full']:.3f} | "
              f"{s['m3_obs']:.3f}  {s['R']:.3f}  {s['comp_shift']:+.4f}   "
              f"{s['stair'][0.05]}")

    print("\n[exact reveal — M5 obs/exact + recalibration]")
    print("seed  exact: cegar  emb    m3_exact | obs: cegar  emb")
    for s in S:
        print(f" {s['seed']}         {s['exact_cl'][ANCHOR]:.3f}  "
              f"{s['exact_cl']['emb']:.3f}  {s['m3_exact']:.3f}  | "
              f"{s['obs_cl'][ANCHOR]:.3f}  {s['obs_cl']['emb']:.3f}")

    print("\n[member verdicts]")
    results = []
    for name, fn in VERDICTS:
        status, detail = fn(S)
        results.append((name, status, detail))
        print(f"  {name:<14} {status:<11} — {detail}")

    decision = decide(results)
    print(f"\nDECISION: {decision}")
    if decision.startswith("BATTERY_TRANSFERS"):
        print("  GO: the six-member battery transfers to pstack under the "
              "earned cegar reference — the hidden-oracle workflow yields a "
              "usable battery (system-level claim). Any RECALIBRATE member "
              "carries its registered pstack threshold. Consolidate the "
              "oracle-withdrawal reference program.")
    else:
        print("  TYPED FAILURE: register the named member/type repair before "
              "claiming battery transfer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
