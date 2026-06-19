"""
I0 — Implementation And Parity Preflight (INTERVENTION_CLASS_BENCHMARK.md).

Supporting / non-claim scaffolding, not a claim-producing experiment. It shows
that the new library helpers in ``interventions.py`` reproduce exp-29's
same-read/same-write result on the existing pstack-L4 artifact and reports
full-patch predicate room for every registered predicate, so the Phase-3
intervention work starts from a measurement stack with known behavior.

Non-claim outputs (per the I0 design step):
  - same-read/same-write helper reproduces c_w ~= 0 for the two decoded
    predicates (phi1_next_closes, phi2_net_return);
  - full-patch predicate room is reported for all registered predicates;
  - synthetic oblique-patch self-tests pass (interventions._selftest);
  - the intervention-family table format is stable.

This routes I1: GO if the same-write patch reproduces exp 29 (c_w ~= 0) while
the full patch has room for the decoded predicates; BLOCK otherwise (fix the
measurement stack or pick a different predicate target before any oblique run).

Built from library primitives (predicates, interventions, discover/abstraction);
no import from the frozen exp-29 script.
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

import interventions as IV
import predicates as P
from abstraction import center_by_position
from discover import PairSet, self_checks
from expcommon import LAYER, load_model
from midstream import orthonormal
from processes import PROCESSES

# Parity config — the exp-29 registered vehicle (pstack-L4, L1, m=3).
REGISTERED_CFG = {"process": "pstack", "seq_len": 40, "burn_in": 4,
                  "d_model": 64, "layers": 4, "m": 3, "seed": 0}
M = 3
TS = (10, 18, 26, 34)
SEEDS = (300, 301, 302, 303)
PAIRS_DISC, PAIRS_EVAL, PAIR_POOL = 320, 1024, 900
LAM = 1e-2                          # ridge on centered residuals (exp 29)
DECODED = ("phi1_next_closes", "phi2_net_return")   # exp-29 decoded targets

# Preflight parity thresholds.
CW_ZERO_TOL = 0.05                  # |c_w| this small reproduces exp-29's ~0.00
ROOM_MIN = IV.ROOM_TOL             # full-patch room must be finite/positive


def gather_residuals(ps, d):
    """L1 residual rows + per-side beliefs for an eval PairSet (exp-4 layout)."""
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


def r2(y, yhat):
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss if ss > 0 else 0.0


def run_seed(model, proc, seed, masks, d):
    eval_ps = PairSet(model, proc, REGISTERED_CFG, PAIRS_EVAL, M, seed + 222,
                      PAIR_POOL, layer=LAYER, ts=TS)
    self_checks(model, eval_ps, LAYER, M, proc.V)

    R, pos, beliefs_tgt, beliefs_src = gather_residuals(eval_ps, d)
    Rc = center_by_position(R, pos, np.ones(eval_ps.n, dtype=bool))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(eval_ps.n)
    tr, te = perm[:eval_ps.n // 2], perm[eval_ps.n // 2:]

    # Endpoints and reference/floor runs (observable model only).
    q_un = eval_ps.run(model, None)
    q_src = eval_ps.run(model, None, src_side=True)
    q_full = eval_ps.run(model, np.eye(d))
    rdir = orthonormal(rng.standard_normal((d, 1)))[:, 0]
    q_rand = eval_ps.run(model, IV.same_write_patch(rdir))

    out = {}
    for name, mask in masks.items():
        y = P.obs_pphi(q_un, mask)
        # affine-read write source: ridge readout direction (exp 29).
        w, b = IV.affine_readout(Rc[tr], y[tr], LAM)
        lin = r2(y[te], Rc[te] @ w + b)

        p_un = P.obs_pphi(q_un, mask)
        p_src = P.obs_pphi(q_src, mask)
        p_full = P.obs_pphi(q_full, mask)
        room = IV.predicate_room(p_un, p_src, p_full)
        c_rand = IV.predicate_control(p_un, p_src, P.obs_pphi(q_rand, mask),
                                      p_full)
        if float(np.linalg.norm(w)) > IV.RW_TOL:
            P_sw = IV.same_write_patch(w)         # same-read/same-write helper
            c_w = IV.predicate_control(p_un, p_src,
                                       P.obs_pphi(eval_ps.run(model, P_sw),
                                                  mask), p_full)
        else:
            c_w = float("nan")
        oe = IV.endpoint_audit(
            p_un, P.exact_pphi(beliefs_tgt, mask, proc, M),
            p_src, P.exact_pphi(beliefs_src, mask, proc, M))
        out[name] = {"std": float(y.std()), "lin_r2": lin, "room": room,
                     "c_w": c_w, "c_rand": c_rand, "oe": oe}
    return out


def decide(per_seed, masks):
    """GO/BLOCK routing for I1, plus per-predicate parity status."""
    status = {}
    for name in masks:
        cws = [per_seed[s][name]["c_w"] for s in SEEDS]
        rooms = [per_seed[s][name]["room"] for s in SEEDS]
        reproduced = all(np.isfinite(c) and abs(c) <= CW_ZERO_TOL for c in cws)
        has_room = all(r > ROOM_MIN for r in rooms)
        status[name] = {"cw_zero": reproduced, "has_room": has_room,
                        "cw_mean": float(np.nanmean(cws)),
                        "room_mean": float(np.mean(rooms))}
    # I1 GO needs: exp-29 reproduced (c_w ~= 0) AND room present, on both
    # decoded predicates. Otherwise BLOCK.
    go = all(status[n]["cw_zero"] and status[n]["has_room"] for n in DECODED)
    return status, go


def selftest():
    IV._selftest()
    P._selftest()
    # decide() parity logic on synthetic per-seed measurements.
    masks = {"phi1_next_closes": None, "phi2_net_return": None}
    good = {s: {n: {"c_w": 0.0, "room": 0.1, "c_rand": 0.0, "oe": 0.0,
                    "std": 0.2, "lin_r2": 0.6} for n in masks} for s in SEEDS}
    _, go = decide(good, masks)
    assert go, "reproduced-with-room should GO"
    noroom = {s: {n: {**good[s][n], "room": 0.0} for n in masks} for s in SEEDS}
    _, go = decide(noroom, masks)
    assert not go, "no room should BLOCK"
    moved = {s: {n: {**good[s][n], "c_w": 0.9} for n in masks} for s in SEEDS}
    _, go = decide(moved, masks)
    assert not go, "non-reproduced (c_w not ~0) should BLOCK"
    print("i0_preflight selftest passed: helpers, predicate scorer, "
          "GO/BLOCK parity logic")


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
    d = cfg["d_model"]

    print("=== I0 — implementation and parity preflight ===")
    print(f"target={proc.name} m={M} LAYER={LAYER} seeds={SEEDS}")
    print("predicates:", ", ".join(masks))
    print("Built from library primitives; exact p_phi is eval-only.\n")

    # Synthetic known-answer self-tests first (non-claim gate).
    IV._selftest()
    print()

    per_seed = {}
    for s in SEEDS:
        out = run_seed(model, proc, s, masks, d)
        per_seed[s] = out
        rows = []
        for name, m in out.items():
            rows.append({
                "target": name, "read": "affine", "write": "same",
                "patch_point": f"L{LAYER}", "room": m["room"],
                "control": m["c_w"], "specificity": m["c_rand"],
                "exact_audit": m["oe"], "transfer": "-",
                "failure_branch": "-" if abs(m["c_w"]) <= CW_ZERO_TOL
                else "moved"})
        print(f"[seed {s}] same-read/same-write reproduction "
              f"(control col is c_w; specificity col is random-write floor)")
        print(IV.format_intervention_table(rows))
        print()

    status, go = decide(per_seed, masks)
    print("[parity summary]")
    for name in masks:
        st = status[name]
        decoded = " (decoded)" if name in DECODED else ""
        print(f"  {name:<18} c_w~={st['cw_mean']:+.3f} "
              f"cw_zero={st['cw_zero']!s:<5} room~={st['room_mean']:.4f} "
              f"has_room={st['has_room']!s:<5}{decoded}")

    decision = "GO" if go else "BLOCK"
    print(f"\nI1 ROUTING: {decision}")
    if go:
        print("  Same-read/same-write reproduces exp 29 (c_w ~= 0) while the "
              "full patch has predicate room for the decoded predicates. The "
              "measurement stack is sound for a fixed-read oblique write "
              "search (I1).")
    else:
        bad = [n for n in DECODED
               if not (status[n]["cw_zero"] and status[n]["has_room"])]
        print("  Do NOT start an oblique experiment. For "
              f"{','.join(bad)} the helper did not reproduce exp 29 or the "
              "full patch has no predicate room. Fix the measurement stack or "
              "choose a different predicate target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
