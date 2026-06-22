"""
L0 substrate gate + harness — experiment 37 (see experiments/37-localization-substrate-gate.md).

State-localization phase, rung L0. Two parts:

  A. harness self-tests (--selftest): pure-function known-answer checks
     (parser, observables, closure, floor normalization, verdict precedence,
     dissociability counting), plus model-level invariants run as guards in the
     main path (no-op patch is bit-exact; full residual patch reproduces source).

  B. substrate gate (GO/NO-GO): on the registered Dyck-2 checkpoint, for each
     facet {depth, top_type}, check non-vacuous / estimable / oracle-audited /
     source-separated / room-bearing (residual-full reference) / floor-clean /
     dissociable, over fresh seeds, and route a typed verdict.

Scope and thresholds are registered in the writeup. This is L0: residual-level
only (patch point L1 residual stream). The block/head component enumerator is
NOT built here — it is L1's, where it is first used (build the seam when earned).
The L0 floor is the facet-matched-source full patch (a residual patch carrying no
target-facet information); the random-unit floor proper arrives with L1's units.

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
ROOM_MIN = 0.50
NULL_TOL = 0.05
CEIL_MIN = 0.90
MIN_PAIRS_PER_CELL = 256
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
              "SMALL_SOURCE_DELTA", "FLOOR_FAIL", "NO_ROOM",
              "NOT_DISSOCIABLE", "SEED_UNSTABLE"]
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
def facet_observable(q, facet, V, m):
    """Returns (value, defined_mask) per row for one facet, from the m=1
    marginal of the completion joint q."""
    q1 = marginal(q, V, 1, m)
    q1 = q1 / np.clip(q1.sum(axis=1, keepdims=True), 1e-30, None)
    cm = q1[:, 2] + q1[:, 3]
    if facet == "depth":
        return cm, np.ones(len(q1), dtype=bool)
    val = q1[:, 2] / np.clip(cm, 1e-30, None)
    return val, cm >= CLOSE_MASS_MIN


def closure(obs_un, obs_src, obs_patch):
    """Per-pair facet closure: 1 reaches source, 0 no move. Caller filters on
    the denominator (SRC_DELTA)."""
    denom = np.abs(obs_un - obs_src)
    return (denom - np.abs(obs_patch - obs_src)) / np.clip(denom, 1e-9, None)


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
    rng.shuffle(pairs)
    pairs = pairs[:n_target]
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int)
    a, b = zip(*pairs)
    return np.array(a), np.array(b)


def floor_pairs(labels, facet, rng, n_target):
    """Facet-matched source pairs: clean and source MATCH on `facet` (so a full
    residual patch carries no target-facet information). Differ on the nuisance
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
    rng.shuffle(pairs)
    pairs = pairs[:n_target]
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int)
    a, b = zip(*pairs)
    return np.array(a), np.array(b)


# ---- per-facet metrics at one seed ----------------------------------------
def facet_metrics(model, proc, Xe, resid, facet, rng, m, V):
    """Aggregate metrics for one facet over all positions at one seed."""
    cl_full, deltas, oe, floor_mov, n_pairs, obs_un_all = [], [], [], [], 0, []
    for t in POSITIONS:
        labels = {i: stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
        a, b = facet_pairs(labels, facet, rng, TARGET_PAIRS)
        n_pairs += len(a)
        if len(a) < COMPUTE_MIN:
            continue
        ca, sb = Xe[a], Xe[b]
        q_un = q_at(model, ca, t, m, V)
        q_src = q_at(model, sb, t, m, V)
        q_pat = q_at(model, ca, t, m, V, prefix_state=resid[b][:, :t + 1])
        ou, mu = facet_observable(q_un, facet, V, m)
        os_, ms = facet_observable(q_src, facet, V, m)
        op, mp = facet_observable(q_pat, facet, V, m)
        ex, me = facet_observable(exact_joint(proc, ca, t, m), facet, V, m)
        keep = mu & ms & mp & me
        ou, os_, op, ex = ou[keep], os_[keep], op[keep], ex[keep]
        if len(ou) < COMPUTE_MIN:
            continue
        deltas.append(np.abs(ou - os_))
        good = np.abs(ou - os_) >= SRC_DELTA_MIN
        if good.sum() >= COMPUTE_MIN:
            cl_full.append(closure(ou[good], os_[good], op[good]))
        oe.append(np.abs(ou - ex))
        obs_un_all.append(np.concatenate([ou, os_]))   # both facet values

        # floor: facet-matched-source full patch (carries no facet info)
        fa, fb = floor_pairs(labels, facet, rng, TARGET_PAIRS)
        if len(fa) >= COMPUTE_MIN:
            fca = Xe[fa]
            q_fu = q_at(model, fca, t, m, V)
            q_fp = q_at(model, fca, t, m, V, prefix_state=resid[fb][:, :t + 1])
            fu, mfu = facet_observable(q_fu, facet, V, m)
            fp, mfp = facet_observable(q_fp, facet, V, m)
            kk = mfu & mfp
            floor_mov.append(np.abs(fp[kk] - fu[kk]))

    if n_pairs < MIN_PAIRS_PER_CELL or not deltas:
        return {"n_pairs": n_pairs, "branch": "NOT_DISSOCIABLE"}
    delta = float(np.concatenate(deltas).mean())
    G = max(delta, 1e-6)
    floor = (float(np.concatenate(floor_mov).mean()) / G) if floor_mov else 0.0
    room = float(np.concatenate(cl_full).mean()) if cl_full else 0.0
    oe_gap = float(np.concatenate(oe).mean())
    std = float(np.concatenate(obs_un_all).std())

    branch = "OK"
    if oe_gap > OE_BAND:
        branch = "OBS_EXACT_DRIFT"
    elif std < VAR_MIN:
        branch = "TARGET_VACUOUS"
    elif delta < SRC_DELTA_MIN:
        branch = "SMALL_SOURCE_DELTA"
    elif floor > NULL_TOL:
        branch = "FLOOR_FAIL"
    elif room < ROOM_MIN:
        branch = "NO_ROOM"
    return {"n_pairs": n_pairs, "delta": delta, "floor": floor, "room": room,
            "oe_gap": oe_gap, "std": std, "branch": branch}


def model_guards(model, proc, cfg, m, V):
    """AGENTS bit-for-bit discipline: no-op patch reproduces unpatched; full
    residual patch reproduces source. Returns True if both hold."""
    rng = np.random.default_rng(0)
    Xe = proc.sample(64, cfg["seq_len"], rng)
    resid = stream_to(model, torch.from_numpy(Xe), LAYER)
    t = 12
    q0 = q_at(model, Xe, t, m, V)
    q_noop = q_at(model, Xe, t, m, V, prefix_state=resid[:, :t + 1])
    if not np.allclose(q0, q_noop, atol=1e-6):
        print("  GUARD FAIL: no-op (own residual) patch not bit-exact")
        return False
    # full patch from a permuted source: the m=1 next-token prediction at t
    # equals source's (only m=1 transports through this position-t patch).
    perm = rng.permutation(len(Xe))
    q_src = q_at(model, Xe[perm], t, m, V)
    q_full = q_at(model, Xe, t, m, V, prefix_state=resid[perm][:, :t + 1])
    if not np.allclose(marginal(q_full, V, 1, m), marginal(q_src, V, 1, m),
                       atol=1e-6):
        print("  GUARD FAIL: full residual patch m=1 != source m=1")
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
        resid = stream_to(model, torch.from_numpy(Xe), LAYER)
        print(f"[seed {seed}]")
        for f in FACETS:
            mt = facet_metrics(model, proc, Xe, resid, f, rng, m, V)
            per_seed[f].append(mt["branch"])
            extra = (f" delta={mt.get('delta', float('nan')):.3f}"
                     f" floor={mt.get('floor', float('nan')):.3f}"
                     f" room={mt.get('room', float('nan')):.3f}"
                     f" oe={mt.get('oe_gap', float('nan')):.3f}"
                     if "delta" in mt else "")
            print(f"  {f:9s} n_pairs={mt['n_pairs']:5d} -> {mt['branch']}{extra}")

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

    # closure: known values
    assert abs(closure(np.array([0.0]), np.array([1.0]),
                       np.array([1.0]))[0] - 1.0) < 1e-9
    assert abs(closure(np.array([0.0]), np.array([1.0]),
                       np.array([0.0]))[0] - 0.0) < 1e-9
    assert abs(closure(np.array([0.0]), np.array([1.0]),
                       np.array([0.5]))[0] - 0.5) < 1e-9

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
    agg = {"depth": "OK", "top_type": "NO_ROOM"}
    assert first_precedence(agg, PRECEDENCE)[0] == "NO_ROOM"
    agg2 = {"depth": "OBS_EXACT_DRIFT", "top_type": "NO_ROOM"}
    assert first_precedence(agg2, PRECEDENCE)[0] == "OBS_EXACT_DRIFT"
    agg3 = {"_harness": "HARNESS_FAIL", "depth": "OK", "top_type": "OK"}
    assert first_precedence(agg3, PRECEDENCE)[0] == "HARNESS_FAIL"
    assert first_precedence({"depth": "OK", "top_type": "OK"},
                            PRECEDENCE)[0] is None
    assert majority_vote(["OK", "OK", "OK", "NO_ROOM"], threshold=3,
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
