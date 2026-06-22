"""
L0 substrate gate + harness — experiment 37 (see experiments/37-localization-substrate-gate.md).

State-localization phase, rung L0. Two parts:

  A. harness self-tests (--selftest): pure-function known-answer checks
     (parser, observables, facet pairing, verdict precedence,
     dissociability counting), plus model-level invariants run as guards in the
     main path (no-op patch is bit-exact; full residual patch reproduces source).

  B. substrate gate (GO/NO-GO): on the registered Dyck-2 checkpoint, for each
     facet {depth, top_type}, check non-vacuous / estimable / oracle-audited /
     source-separated / floor-clean (observable purity) / dissociable, over fresh
     seeds, and route a typed verdict. Movability by the full prefix patch is
     exact m=1 transport (proved by model_guards), not a gate.

Scope and thresholds are registered in the writeup. This is L0: residual-level
only (patch point L1 residual stream). The block/head component enumerator is
NOT built here — it is L1's, where it is first used (build the seam when earned).
The L0 floor is label->observable determinism (within-class observable spread over
facet-matched pairs / between-class gap), computed directly with no patch (under
exact transport a matched-source patch is a no-op transform); the random-unit
floor proper, with a real baseline, arrives with L1's units.

Honesty: selection labels are computed from the observed token prefix by the Dyck
parser; estimators are read from the model's completion distribution; the exact
oracle audits endpoints only.
"""
import argparse
import json
import os
import sys
from itertools import product

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from processes import PROCESSES                       # noqa: E402
from midstream import stream_to, chain_probs, marginal  # noqa: E402
from expcommon import LAYER, load_model, validity_gate  # noqa: E402
from battery import majority_vote, first_precedence    # noqa: E402

# ---- registered thresholds (experiments/37) -------------------------------
VAR_MIN = 0.05
OE_BAND = 0.10
SRC_DELTA_MIN = 0.05
NULL_TOL = 0.05             # clear-impurity floor ceiling
FLOOR_MARGIN_LO = 0.04     # conservative cut: floor > this -> FLOOR_FAIL until L1
                           # supplies a random-unit baseline (floor baseline is
                           # uncharacterized at L0); (FLOOR_MARGIN_LO, NULL_TOL]
                           # is flagged marginal but still fails.
MIN_PAIRS_PER_CELL = 256    # a (position x held-value) cell must reach this
MIN_CELLS = 2               # >= this many qualifying cells, else NOT_DISSOCIABLE
CLOSE_MASS_MIN = 0.05

# ---- registered scope -----------------------------------------------------
POSITIONS = (8, 12, 16, 20)
PAIR_SEEDS = (700, 701, 702, 703)
SEED_MAJORITY = 3
TARGET_PAIRS = 512          # sample up to this many pairs per cell
N_SEQS = 6000               # sequences per seed (enough to fill the cells)
COMPUTE_MIN = 16            # below this a cell is uncomputable -> NOT_DISSOCIABLE

EXPECTED_CFG = {"process": "dyck2", "seq_len": 32, "burn_in": 4,
                "d_model": 64, "layers": 4, "m": 3, "seed": 0}

PRECEDENCE = ["HARNESS_FAIL", "OBS_EXACT_DRIFT", "TARGET_VACUOUS",
              "SMALL_SOURCE_DELTA", "FLOOR_FAIL", "NOT_DISSOCIABLE",
              "SEED_UNSTABLE"]
FACETS = ("depth", "top_type")


def require_expected_config(cfg):
    bad = [(k, cfg.get(k), v) for k, v in EXPECTED_CFG.items() if cfg.get(k) != v]
    if bad:
        print("HALT: wrong checkpoint config for exp 37 (Dyck-2).")
        for k, got, want in bad:
            print(f"  {k}: got {got!r}, expected {want!r}")
        sys.exit(1)


# ---- Dyck parser: labels from the observed token prefix -------------------
# tokens: 0,1 = open type 0,1 ; 2,3 = close type 0,1 (closer of top).
def stack_labels(seq, positions, m):
    """For one token sequence, return {t: (depth_rel, top_type)} at each
    position t in `positions`. depth_rel = min(depth, m); top_type in {0,1} or
    -1 (empty stack). The stack after consuming seq[:t+1]."""
    stack = []
    out = {}
    want = set(positions)
    for t, tok in enumerate(seq):
        if tok < 2:
            stack.append(int(tok))
        else:
            if stack:
                stack.pop()
        if t in want:
            depth = len(stack)
            out[t] = (min(depth, m), stack[-1] if stack else -1)
    return out


# ---- m=1 observables on a completion joint q (n, C=V**m) ------------------
# The L1 position-t patch transports only the next-token (m=1) prediction
# (positions > t leak the clean prefix; this is the exp-4/5 summary patch). So
# both facets are m=1 functions of the next-token distribution, which factor the
# close behavior as close-readiness (depth proxy) x type-fraction (which closer):
#   depth  -> close-readiness  = q(next token is a closer) = q(2)+q(3)
#   top_type -> type-fraction  = q(2)/(q(2)+q(3)), depth-invariant, guarded
def facet_from_q1(q1, facet):
    """(value, defined_mask) per row from a next-token distribution q1 (n, V)."""
    q1 = q1 / np.clip(q1.sum(axis=1, keepdims=True), 1e-30, None)
    cm = q1[:, 2] + q1[:, 3]
    if facet == "depth":
        return cm, np.ones(len(q1), dtype=bool)
    val = q1[:, 2] / np.clip(cm, 1e-30, None)
    return val, cm >= CLOSE_MASS_MIN


def facet_observable(q, facet, V, m):
    """Facet value from the m=1 marginal of a completion joint q (n, V**m)."""
    return facet_from_q1(marginal(q, V, 1, m), facet)


# ---- completion distributions (reuse chain_probs) -------------------------
def make_Xc(seqs, t, m, V):
    conts = np.array(list(product(range(V), repeat=m)))
    n, L = seqs.shape
    C = len(conts)
    Xc = np.repeat(seqs[:, None, :], C, axis=1)
    Xc[:, :, t + 1:t + 1 + m] = conts[None, :, :]
    return Xc


def q_at(model, seqs, t, m, V, prefix_state=None):
    Xc = make_Xc(seqs, t, m, V)
    q, _ = chain_probs(model, Xc, LAYER, prefix_state, t, m, V)
    return q                                              # (n, C)


def exact_joint(proc, seqs, t, m):
    bel = np.stack([proc.beliefs_along(s)[t] for s in seqs])
    return proc.mgram_table(bel, m)                       # (n, V**m)


def unconditioned_std(model, Xe, facet, positions):
    """Std of the facet observable over an UNCONDITIONED eval sample at the
    registered positions (not the contrastive pairs) — the registered vacuity
    check, independent of SRC_DELTA. Uses the model's next-token distribution
    directly (one forward, no continuations)."""
    dev = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(Xe).to(dev))
        probs = torch.softmax(logits, dim=-1).cpu().double().numpy()
    vals = []
    for t in positions:
        v, mask = facet_from_q1(probs[:, t, :], facet)
        vals.append(v[mask])
    allv = np.concatenate(vals) if vals else np.zeros(0)
    return float(allv.std()) if len(allv) else 0.0


# ---- facet-conditioned pairing (the reusable core) ------------------------
def facet_pairs(labels, facet, rng, n_target):
    """labels: dict seq_idx -> (depth_rel, top_type) at one position.
    Returns (clean, source) index arrays that differ in `facet` only and match
    on the other facet. For top_type, both must be non-empty (top in {0,1})."""
    items = [(i, dr, tt) for i, (dr, tt) in labels.items()]
    pairs = []
    if facet == "depth":
        # match top_type, differ in depth_rel
        buckets = {}
        for i, dr, tt in items:
            if tt < 0:
                continue
            buckets.setdefault(tt, []).append((i, dr))
        for tt, lst in buckets.items():
            by_d = {}
            for i, dr in lst:
                by_d.setdefault(dr, []).append(i)
            ds = list(by_d)
            for _ in range(n_target):
                if len(ds) < 2:
                    break
                d1, d2 = rng.choice(ds, size=2, replace=False)
                a = rng.choice(by_d[d1]); b = rng.choice(by_d[d2])
                pairs.append((a, b))
    else:  # top_type: match depth_rel (>=1), differ in top_type
        buckets = {}
        for i, dr, tt in items:
            if tt < 0 or dr < 1:
                continue
            buckets.setdefault(dr, []).append((i, tt))
        for dr, lst in buckets.items():
            by_t = {}
            for i, tt in lst:
                by_t.setdefault(tt, []).append(i)
            if len(by_t) < 2:
                continue
            for _ in range(n_target):
                x, y = rng.choice(by_t[0]), rng.choice(by_t[1])
                pairs.append((x, y) if rng.random() < 0.5 else (y, x))
    pairs = list(dict.fromkeys(pairs))     # dedupe (sampling is with replacement)
    rng.shuffle(pairs)
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int)
    a, b = zip(*pairs)
    return np.array(a), np.array(b)


def floor_pairs(labels, facet, rng, n_target):
    """Facet-matched pairs: the two MATCH on `facet` (same label), so their
    observable spread is label->observable determinism. Differ on the nuisance
    where possible."""
    items = [(i, dr, tt) for i, (dr, tt) in labels.items()]
    pairs = []
    if facet == "depth":
        by_d = {}
        for i, dr, tt in items:
            if tt < 0:
                continue
            by_d.setdefault(dr, []).append((i, tt))
        for dr, lst in by_d.items():
            if len(lst) < 2:
                continue
            for _ in range(n_target):
                ia = lst[rng.choice(len(lst))]
                ib = lst[rng.choice(len(lst))]
                if ia[0] == ib[0]:
                    continue
                pairs.append((ia[0], ib[0]))
    else:
        by_t = {}
        for i, dr, tt in items:
            if tt < 0 or dr < 1:
                continue
            by_t.setdefault(tt, []).append((i, dr))
        for tt, lst in by_t.items():
            if len(lst) < 2:
                continue
            for _ in range(n_target):
                ia = lst[rng.choice(len(lst))]
                ib = lst[rng.choice(len(lst))]
                if ia[0] == ib[0]:
                    continue
                pairs.append((ia[0], ib[0]))
    pairs = list(dict.fromkeys(pairs))     # dedupe (sampling is with replacement)
    rng.shuffle(pairs)
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int)
    a, b = zip(*pairs)
    return np.array(a), np.array(b)


# ---- per-facet metrics at one seed ----------------------------------------
# Movability of a facet by the full prefix patch is NOT tested here: it is exact
# by construction (the patch transports the m=1 marginal; model_guards proves it
# bit-for-bit at every registered position), so a "room"/closure gate would be
# tautological. The live substrate checks are calibration (OBS_EXACT_DRIFT),
# UNCONDITIONED variation (TARGET_VACUOUS), a real clean-source gap
# (SMALL_SOURCE_DELTA), dissociable pairs per (position x held-value) cell
# (NOT_DISSOCIABLE), and FLOOR_FAIL.
#
# FLOOR is label->observable determinism, NOT a patch test: under exact transport
# a facet-matched-source patch's m=1 marginal == source's, so its observable is
# obs(fb) exactly. So floor_score = mean |obs(fb) - obs(fa)| / mean_gap over
# facet-MATCHED (same-facet, same-position) pairs = within-class observable spread
# / between-class gap (a purity/SNR ratio). We compute it directly (no patch),
# which also removes any dependence on transport holding away from the guard.
def facet_metrics(model, proc, Xe, facet, rng, m, V):
    """Aggregate metrics for one facet over all (position x held-value) cells."""
    deltas, oe, floor_mov, cells_ok, n_pairs, cell_sizes = [], [], [], 0, 0, []
    contrast = {"boundary": 0, "interior": 0}    # depth: surviving-gap counts
    for t in POSITIONS:
        labels = {i: stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
        a, b = facet_pairs(labels, facet, rng, TARGET_PAIRS)
        n_pairs += len(a)
        if len(a) == 0:
            continue
        # held-fixed value defining the cell: top_type for depth, depth for type
        held = np.array([labels[i][1 if facet == "depth" else 0] for i in a])
        for hv in np.unique(held):
            sel = np.where(held == hv)[0]
            if len(sel) < MIN_PAIRS_PER_CELL:    # thin cell: excluded
                continue
            aa, bb = a[sel], b[sel]
            ou, mu = facet_observable(q_at(model, Xe[aa], t, m, V), facet, V, m)
            os_, ms = facet_observable(q_at(model, Xe[bb], t, m, V), facet, V, m)
            ex, me = facet_observable(exact_joint(proc, Xe[aa], t, m),
                                      facet, V, m)
            keep = mu & ms & me
            ou, os_, ex = ou[keep], os_[keep], ex[keep]
            if len(ou) < COMPUTE_MIN:
                continue
            cells_ok += 1
            cell_sizes.append(int(len(ou)))
            deltas.append(np.abs(ou - os_))
            oe.append(np.abs(ou - ex))
            if facet == "depth":                 # which contrasts carry the gap
                good = np.abs(ou - os_) >= SRC_DELTA_MIN
                dra = np.array([labels[i][0] for i in aa])[keep][good]
                drb = np.array([labels[i][0] for i in bb])[keep][good]
                bnd = (np.minimum(dra, drb) == 0) | (np.maximum(dra, drb) == m)
                contrast["boundary"] += int(bnd.sum())
                contrast["interior"] += int((~bnd).sum())

        # floor: within-class observable spread (no patch; see header)
        fa, fb = floor_pairs(labels, facet, rng, TARGET_PAIRS)
        if len(fa) >= COMPUTE_MIN:
            fu, mfu = facet_observable(q_at(model, Xe[fa], t, m, V), facet, V, m)
            fv, mfv = facet_observable(q_at(model, Xe[fb], t, m, V), facet, V, m)
            kk = mfu & mfv
            floor_mov.append(np.abs(fv[kk] - fu[kk]))

    if cells_ok < MIN_CELLS or not deltas:
        return {"n_pairs": n_pairs, "cells_ok": cells_ok,
                "branch": "NOT_DISSOCIABLE"}
    delta = float(np.concatenate(deltas).mean())
    floor = (float(np.concatenate(floor_mov).mean()) / max(delta, 1e-6)
             if floor_mov else 0.0)
    oe_gap = float(np.concatenate(oe).mean())
    std = unconditioned_std(model, Xe, facet, POSITIONS)

    branch = "OK"
    if oe_gap > OE_BAND:
        branch = "OBS_EXACT_DRIFT"
    elif std < VAR_MIN:
        branch = "TARGET_VACUOUS"
    elif delta < SRC_DELTA_MIN:
        branch = "SMALL_SOURCE_DELTA"
    elif floor > FLOOR_MARGIN_LO:                # conservative cut (see constant)
        branch = "FLOOR_FAIL"
    out = {"n_pairs": n_pairs, "cells_ok": cells_ok, "delta": delta,
           "floor": floor, "floor_marginal": FLOOR_MARGIN_LO < floor <= NULL_TOL,
           "oe_gap": oe_gap, "std": std, "branch": branch,
           "cell_min": min(cell_sizes), "cell_med": int(np.median(cell_sizes))}
    if facet == "depth":
        out["contrast"] = contrast               # boundary vs interior gap counts
    return out


def model_guards(model, proc, cfg, m, V):
    """AGENTS bit-for-bit discipline: no-op patch reproduces unpatched; full
    residual patch reproduces source. Returns True if both hold."""
    rng = np.random.default_rng(0)
    Xe = proc.sample(64, cfg["seq_len"], rng)
    resid = stream_to(model, torch.from_numpy(Xe), LAYER)
    perm = rng.permutation(len(Xe))
    for t in POSITIONS:                         # assert at every registered cell
        q0 = q_at(model, Xe, t, m, V)
        q_noop = q_at(model, Xe, t, m, V, prefix_state=resid[:, :t + 1])
        if not np.allclose(q0, q_noop, atol=1e-6):
            print(f"  GUARD FAIL: no-op patch not bit-exact at t={t}")
            return False
        # full source patch: the m=1 next-token prediction equals source's
        # (only m=1 transports through this position-t patch).
        q_src = q_at(model, Xe[perm], t, m, V)
        q_full = q_at(model, Xe, t, m, V, prefix_state=resid[perm][:, :t + 1])
        if not np.allclose(marginal(q_full, V, 1, m), marginal(q_src, V, 1, m),
                           atol=1e-6):
            print(f"  GUARD FAIL: full patch m=1 != source m=1 at t={t}")
            return False
    return True


def run_gate(model, proc, cfg):
    m, V = cfg["m"], proc.V
    print("[guards] model-level patch invariants (no-op bit-exact; "
          "full m=1 = source m=1)")
    harness_ok = model_guards(model, proc, cfg, m, V)
    print(f"  -> {'OK' if harness_ok else 'HARNESS_FAIL'}\n")

    per_seed = {f: [] for f in FACETS}
    for seed in PAIR_SEEDS:
        rng = np.random.default_rng(seed)
        Xe = proc.sample(N_SEQS, cfg["seq_len"], rng)
        print(f"[seed {seed}]")
        for f in FACETS:
            mt = facet_metrics(model, proc, Xe, f, rng, m, V)
            per_seed[f].append(mt["branch"])
            extra = (f" delta={mt.get('delta', float('nan')):.3f}"
                     f" floor={mt.get('floor', float('nan')):.3f}"
                     f"{'*MARGINAL' if mt.get('floor_marginal') else ''}"
                     f" oe={mt.get('oe_gap', float('nan')):.3f}"
                     f" std={mt.get('std', float('nan')):.3f}"
                     f" cell[min={mt.get('cell_min', 0)},"
                     f"med={mt.get('cell_med', 0)}]"
                     if "delta" in mt else "")
            print(f"  {f:9s} n_pairs={mt['n_pairs']:5d} "
                  f"cells_ok={mt.get('cells_ok', 0)} -> {mt['branch']}{extra}")
            if f == "depth" and "contrast" in mt:
                c = mt["contrast"]
                print(f"      close-readiness gap carried by: "
                      f"boundary(0/full) n={c['boundary']} vs "
                      f"interior n={c['interior']}  "
                      "[interior-vs-interior mostly filtered by SRC_DELTA]")

    agg = {f: majority_vote(per_seed[f], threshold=SEED_MAJORITY,
                            unstable="SEED_UNSTABLE") for f in FACETS}
    if not harness_ok:
        agg = {"_harness": "HARNESS_FAIL", **agg}
    label, keys = first_precedence(agg, PRECEDENCE)
    print(f"\nper-facet aggregate: {agg}")
    if label is None:
        decision = "GO"
    else:
        decision = f"{label}({','.join(k for k in keys if k != '_harness')})" \
            if keys != ["_harness"] else label
    print(f"\nDECISION: {decision}")
    return decision


# ---- self-tests (pure functions, no checkpoint) ---------------------------
def _selftest():
    V, m = 4, 3
    conts = np.array(list(product(range(V), repeat=m)))
    # m=1 observables: depth = close-readiness, top_type = type-0 fraction.
    # build a joint with next-token mass: close0 0.3, close1 0.1, opens 0.6
    q1 = np.zeros((1, V ** m))
    for i, c in enumerate(conts):
        if c[0] == 2:
            q1[0, i] = 0.3 / (V ** (m - 1))
        elif c[0] == 3:
            q1[0, i] = 0.1 / (V ** (m - 1))
        else:
            q1[0, i] = 0.6 / (2 * V ** (m - 1))
    d_val, d_mask = facet_observable(q1, "depth", V, m)
    assert abs(d_val[0] - 0.4) < 1e-9 and d_mask[0]            # close-readiness
    t_val, t_mask = facet_observable(q1, "top_type", V, m)
    assert abs(t_val[0] - 0.75) < 1e-9 and t_mask[0]          # 0.3/0.4 type-0
    # type-fraction is depth-invariant: scale total close mass, ratio fixed
    q2 = q1.copy()
    for i, c in enumerate(conts):
        if c[0] >= 2:
            q2[0, i] *= 0.05                                   # far less closing
    q2 /= q2.sum()
    assert abs(facet_observable(q2, "top_type", V, m)[0][0] - 0.75) < 1e-6
    # low close mass -> type undefined (guard)
    assert not facet_observable(q2, "top_type", V, m)[1][0]

    # parser: ( [ ) -> after each: depth 1 (top0), 2 (top1), 1 (top0)
    seq = np.array([0, 1, 2, 1, 3, 2, 0])
    lab = stack_labels(seq, [0, 1, 2], m)
    assert lab[0] == (1, 0) and lab[1] == (2, 1) and lab[2] == (1, 0), lab

    # facet pairing: depth pairs share top_type, differ depth_rel
    labels = {0: (1, 0), 1: (2, 0), 2: (1, 1), 3: (2, 1), 4: (3, 0)}
    rng = np.random.default_rng(0)
    a, b = facet_pairs(labels, "depth", rng, 50)
    for i, j in zip(a, b):
        assert labels[i][1] == labels[j][1] and labels[i][0] != labels[j][0]
    a, b = facet_pairs(labels, "top_type", rng, 50)
    for i, j in zip(a, b):
        assert labels[i][0] == labels[j][0] and labels[i][1] != labels[j][1]

    # verdict precedence + majority
    agg = {"depth": "OK", "top_type": "FLOOR_FAIL"}
    assert first_precedence(agg, PRECEDENCE)[0] == "FLOOR_FAIL"
    agg2 = {"depth": "OBS_EXACT_DRIFT", "top_type": "FLOOR_FAIL"}
    assert first_precedence(agg2, PRECEDENCE)[0] == "OBS_EXACT_DRIFT"
    agg3 = {"_harness": "HARNESS_FAIL", "depth": "OK", "top_type": "OK"}
    assert first_precedence(agg3, PRECEDENCE)[0] == "HARNESS_FAIL"
    assert first_precedence({"depth": "OK", "top_type": "OK"},
                            PRECEDENCE)[0] is None
    assert majority_vote(["OK", "OK", "OK", "FLOOR_FAIL"], threshold=3,
                         unstable="SEED_UNSTABLE") == "OK"
    print("selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
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
    gap, passed = validity_gate(model, proc, cfg, EXPECTED_CFG["seed"])
    if not passed:
        print("HALT: validity gate failed.")
        sys.exit(1)
    print(f"=== Experiment 37: Dyck-2 localization substrate gate | "
          f"L{cfg['layers']} d{cfg['d_model']} | m={cfg['m']} ===\n")
    run_gate(model, proc, cfg)


if __name__ == "__main__":
    main()
