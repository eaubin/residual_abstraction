"""localize.py — reusable core for the state-localization phase (L1+).

Promoted verbatim from the frozen L0 gate (`l0_substrate_gate.py`, exp 37): the
durable facet/Dyck pieces L1+ reuse, so the rung scripts import a library home
rather than a concluded experiment script. L0 keeps its own inline copy (the
accepted exp-15 duplication) and stays frozen; this module is the home from here.

Contents (the L0 core):
  - Dyck parser labels (`stack_labels`);
  - m=1 facet observables on a completion joint (`facet_from_q1`,
    `facet_observable`) — close-readiness (depth proxy) and type-fraction;
  - positioned completion-joint machinery (`make_Xc`, `q_at`, `exact_joint`),
    which wrap `midstream.chain_probs` and support the prefix-state patch;
  - facet-conditioned pairing (`facet_pairs`) and same-facet floor pairing
    (`floor_pairs`);
  - the phase checkpoint contract (`EXPECTED_CFG`, `require_expected_config`).

L1's NEW reusable primitives (added by exp 38, the propagation gate):
  - the m>=2 forced-close conditional scorer (`cr_cond`);
  - the windowed/single-position residual patch builder (`make_patched_prefix`);
  - graded-depth pair construction (`depth_triples`) and the in-model
    shared-prefix planted-locus generator (`planted_locus_pairs`).
The locality/necessity/horizon curve REDUCERS and the verdict live in the rung
script (experiment-specific orchestration), not here. The block/head unit
enumerator stays DEFERRED until the gate returns PROPAGATED/DISTRIBUTED (do not
build the seam before it is earned).

The directional-specificity rung (exp 40) adds the reusable steering core:
  - `transport_fraction` (promoted from exp 38's inline `_f`);
  - `facet_diff_vector` — the matched-pair diff-in-means steering vector per position;
  - `apply_additive_steer` — a rank-1 additive prefix splice (not replacement);
  - `random_matched_direction` — the matched-norm off-manifold floor.
The 2x2 dissociation scorer, alpha-sweep reducer, and verdict live in that rung
script. The block/head enumerator is still NOT built.

Threshold *values* here (e.g. CLOSE_MASS_MIN) are the L0-registered defaults;
an experiment may re-register its own — they are gate cutoffs, not library law.
"""
import os
import sys
from itertools import product

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from midstream import chain_probs, marginal  # noqa: E402
from expcommon import LAYER  # noqa: E402

CLOSE_MASS_MIN = 0.05          # denominator guard for type_obs (L0-registered)

# The localization-phase Dyck-2 checkpoint contract (exp-19 config; exps 37+).
EXPECTED_CFG = {"process": "dyck2", "seq_len": 32, "burn_in": 4,
                "d_model": 64, "layers": 4, "m": 3, "seed": 0}


def require_expected_config(cfg):
    """Halt unless the loaded checkpoint matches the phase's Dyck-2 contract."""
    bad = [(k, cfg.get(k), v) for k, v in EXPECTED_CFG.items() if cfg.get(k) != v]
    if bad:
        print("HALT: wrong checkpoint config for the localization phase (Dyck-2).")
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
def facet_from_q1(q1, facet, close_mass_min=CLOSE_MASS_MIN):
    """(value, defined_mask) per row from a next-token distribution q1 (n, V)."""
    q1 = q1 / np.clip(q1.sum(axis=1, keepdims=True), 1e-30, None)
    cm = q1[:, 2] + q1[:, 3]
    if facet == "depth":
        return cm, np.ones(len(q1), dtype=bool)
    val = q1[:, 2] / np.clip(cm, 1e-30, None)
    return val, cm >= close_mass_min


def facet_observable(q, facet, V, m, close_mass_min=CLOSE_MASS_MIN):
    """Facet value from the m=1 marginal of a completion joint q (n, V**m)."""
    return facet_from_q1(marginal(q, V, 1, m), facet, close_mass_min)


# ---- positioned completion distributions (wrap midstream.chain_probs) ------
def make_Xc(seqs, t, m, V):
    """Splice every length-m continuation onto each prefix at positions t+1..t+m."""
    conts = np.array(list(product(range(V), repeat=m)))
    n, L = seqs.shape
    C = len(conts)
    Xc = np.repeat(seqs[:, None, :], C, axis=1)
    Xc[:, :, t + 1:t + 1 + m] = conts[None, :, :]
    return Xc


def q_at(model, seqs, t, m, V, prefix_state=None, layer=LAYER):
    """Exact m-step completion joint at position t, optionally under a prefix
    residual patch (prefix_state replaces the residual at positions :t+1)."""
    Xc = make_Xc(seqs, t, m, V)
    q, _ = chain_probs(model, Xc, layer, prefix_state, t, m, V)
    return q                                              # (n, C)


def exact_joint(proc, seqs, t, m):
    """The Dyck oracle's exact completion joint at position t (audit only)."""
    bel = np.stack([proc.beliefs_along(s)[t] for s in seqs])
    return proc.mgram_table(bel, m)                       # (n, V**m)


# ---- facet-conditioned pairing (the reusable core) ------------------------
def facet_pairs(labels, facet, rng, n_target, oriented=False):
    """labels: dict seq_idx -> (depth_rel, top_type) at one position.
    Returns (clean, source) index arrays that differ in `facet` only and match
    on the other facet. For top_type, both must be non-empty (top in {0,1}).

    `oriented` controls pair direction. Default (False) randomizes orientation —
    correct for a sign-agnostic |Δ| transport SPREAD (l0_substrate_gate). For a
    SIGNED diff-in-means STEERING vector (exp 40's facet_diff_vector) the contrast
    must point one way, else +Δ and −Δ pairs cancel to a ~0 mean direction; set
    oriented=True for clean=low (depth_rel / top_type 0) -> source=high (1)."""
    items = [(i, dr, tt) for i, (dr, tt) in labels.items()]
    pairs = []
    if facet == "depth":
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
                if oriented and d1 > d2:           # clean=shallower -> source=deeper
                    d1, d2 = d2, d1
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
                x, y = rng.choice(by_t[0]), rng.choice(by_t[1])   # x=type0, y=type1
                if oriented:
                    pairs.append((x, y))                          # clean=0 -> source=1
                else:
                    pairs.append((x, y) if rng.random() < 0.5 else (y, x))
    pairs = list(dict.fromkeys(pairs))     # dedupe (sampling is with replacement)
    rng.shuffle(pairs)
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int)
    a, b = zip(*pairs)
    return np.array(a), np.array(b)


def floor_pairs(labels, facet, rng, n_target):
    """Facet-matched pairs: the two MATCH on `facet` (same label), so their
    observable spread is label->observable determinism. Self-pairs (same index)
    are skipped; the nuisance is left free (this is a spread estimator)."""
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


# ===========================================================================
# L1 primitives (exp 38, the propagation gate) — built on the L0 core above.
# ===========================================================================

# ---- the m>=2 forced-close conditional scorer -----------------------------
def cr_cond(q, V, m, k):
    """Close-readiness after k forced closes: P(w_{k+1} closer | w_1..w_k all
    closers), read from the (k+1)-step marginal of the completion joint q
    (n, V**m). closers = tokens {2,3}. k=0 is the m=1 close-readiness (L0's
    signal); k>=1 is the graded extension the m=1 marginal cannot see (the k-th
    conditional separates depth k from depth k+1)."""
    mm = k + 1
    arr = marginal(q, V, mm, m).reshape(len(q), *([V] * mm))
    arr = arr / np.clip(arr.sum(tuple(range(1, mm + 1)), keepdims=True), 1e-30, None)
    cond = (slice(None),) + (slice(2, 4),) * k          # first k steps are closers
    den = arr[cond + (slice(None),)].sum(tuple(range(1, mm + 1)))
    num = arr[cond + (slice(2, 4),)].sum(tuple(range(1, mm + 1)))
    return num / np.clip(den, 1e-12, None)


# ---- windowed / single-position residual patch ----------------------------
def make_patched_prefix(clean_resid, source_resid, t, patch_positions):
    """Residual prefix `[:, :t+1]` = the clean run's own residual, with `source`
    spliced in at `patch_positions` (positions in [0, t]). Empty positions ->
    a no-op (clean's own residual, reproduces unpatched); all of [0, t] -> the
    full-prefix source patch. This is the windowed interchange: only the named
    positions carry source; everything else stays clean."""
    ps = clean_resid[:, :t + 1].clone()
    if len(patch_positions):
        idx = torch.as_tensor(np.asarray(patch_positions), dtype=torch.long)
        ps[:, idx] = source_resid[:, idx]
    return ps


# ---- graded-depth pair construction ---------------------------------------
def depth_triples(labels, lo, hi, rng):
    """Per top_type at one position, aligned indices matched on top_type:
      clean  = depth `lo` (the tokens patched onto),
      src_hi = depth `hi` (the graded-depth source -> ceiling/transport target),
      src_lo = depth `lo`, a distinct instance (same-depth source -> floor).
    Matching on top_type makes the contrast graded depth, not which closer."""
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


# ---- planted-locus reference (in-model, shared-prefix) --------------------
def planted_locus_pairs(base_seqs, t, m):
    """Construct clean/source pairs that are TOKEN-IDENTICAL on [0, t-1] and
    differ only at position `t`, so the depth-determining divergence — hence the
    only differing residual at the patch layer on [0, t] — sits at `t`. A model
    that summarizes depth at a position must then transport at the smallest
    window (window=1), and dropping `t` must block it. This is the known-answer
    early-saturation reference, run through the real model on real continuations
    (same contamination regime as the model pairs).

    From base sequences with exact depth 2 at t-1 (label==2 is unambiguous since
    the cap is m=3): clean appends the legal CLOSE at t (-> depth 1), source
    appends an OPEN at t (-> depth 3). Positions > t are irrelevant (causal, and
    overwritten by spliced continuations at scoring). Returns (clean, source)."""
    L = base_seqs.shape[1]
    clean, src = [], []
    for s in base_seqs:
        d_prev, top = stack_labels(s, [t - 1], m)[t - 1]
        if d_prev != 2:                      # exact depth 2 -> clean 1 / source 3
            continue
        closer = 2 if top == 0 else 3
        c = s.copy(); c[t] = closer          # close: depth 2 -> 1
        u = s.copy(); u[t] = 0               # open type 0: depth 2 -> 3
        clean.append(c); src.append(u)
    if not clean:
        return np.zeros((0, L), int), np.zeros((0, L), int)
    return np.array(clean), np.array(src)


# ===========================================================================
# Directional-specificity primitives (exp 40) — built on the core above.
# A rank-1-per-position additive steering vector from observable-matched pairs,
# its application as a prefix patch, the matched-norm random-direction control,
# and the transport fraction promoted from exp 38's frozen inline `_f`.
# ===========================================================================

def transport_fraction(P, C, S, gap_min, valid=None):
    """Mean (P-C)/(S-C) over rows with a real gap |S-C| >= gap_min. Promoted from
    exp 38's inline `_f` (38 keeps its frozen copy, the accepted exp-15 duplication).
    `valid` is an optional row mask (e.g. top_type is defined only where the close
    mass clears CLOSE_MASS_MIN)."""
    P, C, S = np.asarray(P, float), np.asarray(C, float), np.asarray(S, float)
    g = S - C
    keep = (np.abs(g) >= gap_min) & np.isfinite(P) & np.isfinite(C) & np.isfinite(S)
    if valid is not None:
        keep &= np.asarray(valid, bool)
    if keep.sum() == 0:
        return float("nan")
    return float(np.mean((P[keep] - C[keep]) / g[keep]))


def read_facet(q, facet, V, m, k):
    """(value, defined_mask) for a facet on a completion joint q (n, V**m). `depth` is the
    GRADED forced-close conditional `cr_cond` at horizon k (exp 38's instrument, not the
    m=1 coarse close-readiness proxy); any other facet is its m=1 `facet_observable`
    (e.g. `top_type`, defined only where the close mass clears CLOSE_MASS_MIN)."""
    if facet == "depth":
        val = cr_cond(q, V, m, k)
        return val, np.isfinite(val)
    return facet_observable(q, facet, V, m)


def drag_fraction(P, C, gap, valid):
    """Off-target movement mean|P - C| under a steer, normalized by the off-target facet's
    OWN between-class gap (the move a genuine intervention on it would produce). There is
    no source value for the off-target (matched pairing), so ANY movement is drag. `valid`
    is a row mask; gap <= 0 or no valid rows -> nan."""
    keep = np.asarray(valid, bool) & np.isfinite(P) & np.isfinite(C)
    if keep.sum() == 0 or not np.isfinite(gap) or gap < 1e-9:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(P, float)[keep] - np.asarray(C, float)[keep])) / gap)


def facet_diff_vector(clean_resid, source_resid, t):
    """Per-position diff-in-means steering vector over the prefix [0, t].
    clean_resid, source_resid: (n, L, d) residual caches for INDEX-ALIGNED matched
    pairs (clean[i] vs source[i] differ in the target facet, matched on the other —
    `depth_triples` / `facet_pairs`). Returns v: (t+1, d) = mean_i (source - clean)
    per position. Adding alpha*v to a clean prefix steers along the facet contrast;
    the matched pairing cancels the off-target facet and (over many pairs) nuisance,
    leaving the facet-specific direction. Rank-1 per position, additive."""
    diff = source_resid[:, :t + 1] - clean_resid[:, :t + 1]
    return diff.mean(dim=0)


def apply_additive_steer(clean_resid, v, t, alpha, positions):
    """Clean prefix residual [:, :t+1] with `alpha*v` ADDED at `positions` (a subset
    of [0, t]); v is (t+1, d) from facet_diff_vector. alpha=0 or no positions -> a
    no-op (clean's own residual). Additive, NOT replacement: the off-target content
    at each position is left intact — which is exactly what makes cross-facet drag a
    non-trivial measurement (full replacement would transport the m=1 readout exactly
    and preserve any matched label by construction)."""
    ps = clean_resid[:, :t + 1].clone()
    if float(alpha) != 0.0 and len(positions):
        idx = torch.as_tensor(np.asarray(positions), dtype=torch.long)
        ps[:, idx] = ps[:, idx] + float(alpha) * v[idx].to(ps.dtype)
    return ps


def random_matched_direction(v, rng):
    """A random gaussian direction with the SAME per-position L2 norm as v (t+1, d):
    the matched-norm off-manifold floor for the additive steer (the direction-space
    analog of exp 38's random-placement control). A genuine facet direction must move
    its facet ABOVE this floor, and genuine cross-drag must exceed this floor's drag."""
    g = torch.from_numpy(rng.standard_normal(tuple(v.shape)).astype(np.float32)).to(v.device)
    gn = g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return g / gn * v.norm(dim=-1, keepdim=True)


# ---- self-tests (pure functions, no checkpoint) ---------------------------
def _selftest():
    V, m = 4, 3
    conts = np.array(list(product(range(V), repeat=m)))
    # m=1 observables: depth = close-readiness, top_type = type-0 fraction.
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
    assert not facet_observable(q2, "top_type", V, m)[1][0]    # low mass -> undef

    # parser: ( [ ) -> after each: depth 1 (top0), 2 (top1), 1 (top0)
    seq = np.array([0, 1, 2, 1, 3, 2, 0])
    lab = stack_labels(seq, [0, 1, 2], m)
    assert lab[0] == (1, 0) and lab[1] == (2, 1) and lab[2] == (1, 0), lab

    # make_Xc splices continuations at t+1..t+m, prefix untouched
    seqs = np.zeros((2, 8), dtype=int)
    Xc = make_Xc(seqs, 2, m, V)
    assert Xc.shape == (2, V ** m, 8)
    assert (Xc[:, :, :3] == 0).all()                          # prefix preserved
    assert (Xc[0, :, 3:6] == conts).all()                     # continuations

    # facet pairing: depth pairs share top_type, differ depth_rel; type vice versa
    labels = {0: (1, 0), 1: (2, 0), 2: (1, 1), 3: (2, 1), 4: (3, 0)}
    rng = np.random.default_rng(0)
    a, b = facet_pairs(labels, "depth", rng, 50)
    for i, j in zip(a, b):
        assert labels[i][1] == labels[j][1] and labels[i][0] != labels[j][0]
    a, b = facet_pairs(labels, "top_type", rng, 50)
    for i, j in zip(a, b):
        assert labels[i][0] == labels[j][0] and labels[i][1] != labels[j][1]
    # oriented=True fixes the contrast direction (clean=low -> source=high) so a
    # signed diff-in-means does not cancel (exp 40 F1):
    a, b = facet_pairs(labels, "top_type", rng, 50, oriented=True)
    for i, j in zip(a, b):
        assert labels[i][1] == 0 and labels[j][1] == 1
    a, b = facet_pairs(labels, "depth", rng, 50, oriented=True)
    for i, j in zip(a, b):
        assert labels[i][0] < labels[j][0]
    # floor pairs MATCH on facet
    fa, fb = floor_pairs(labels, "depth", rng, 50)
    for i, j in zip(fa, fb):
        assert labels[i][0] == labels[j][0] and i != j

    # --- L1 primitives ---
    # cr_cond: k=0 is close-readiness; k=1 is the conditional P(w2 cl | w1 cl).
    # build a joint where every step closes w.p. 0.4 (independent across steps):
    pc = np.array([0.3, 0.3, 0.25, 0.15])          # P(token): opens .6, closes .4
    joint = np.ones(V ** m)
    for i, c in enumerate(conts):
        joint[i] = np.prod([pc[w] for w in c])
    joint = joint[None, :]
    assert abs(cr_cond(joint, V, m, 0)[0] - 0.4) < 1e-9        # P(w1 closer)
    assert abs(cr_cond(joint, V, m, 1)[0] - 0.4) < 1e-9        # independent
    assert abs(cr_cond(joint, V, m, 2)[0] - 0.4) < 1e-9

    # make_patched_prefix: empty -> clean (no-op); full [0..t] -> source.
    cr = torch.arange(2 * 6 * 3, dtype=torch.float).reshape(2, 6, 3)
    sr = -cr.clone()
    t = 4
    assert torch.equal(make_patched_prefix(cr, sr, t, []), cr[:, :t + 1])
    full = make_patched_prefix(cr, sr, t, list(range(t + 1)))
    assert torch.equal(full, sr[:, :t + 1])
    win = make_patched_prefix(cr, sr, t, [t])                  # only position t
    assert torch.equal(win[:, :t], cr[:, :t]) and torch.equal(win[:, t], sr[:, t])

    # depth_triples: clean/src_lo at depth lo, src_hi at depth hi, top_type-matched
    dl = {0: (1, 0), 1: (1, 0), 2: (2, 0), 3: (1, 1), 4: (1, 1), 5: (2, 1)}
    c, hi, lo = depth_triples(dl, 1, 2, np.random.default_rng(0))
    for ic, ih, il in zip(c, hi, lo):
        assert dl[ic][0] == 1 and dl[il][0] == 1 and dl[ih][0] == 2
        assert dl[ic][1] == dl[ih][1] == dl[il][1] and ic != il

    # planted_locus_pairs: shared prefix [0,t-1], differ at t, depths 1 vs 3
    seq = np.array([0, 1, 0, 2, 2, 0, 1, 3])      # depth at t-1=3 is 2 (0,1,0 ->3? check)
    base = np.tile(seq, (1, 1))
    t = 4
    # build a base with exact depth 2 at t-1=3: tokens 0,1,0 -> depths 1,2,3; pop at 3 -> 2
    base = np.array([[0, 1, 0, 2, 0, 0, 0, 0]])   # after [0,1,0,2]: depth 2, top type0
    pc_, ps_ = planted_locus_pairs(base, t, m)
    assert len(pc_) == 1
    assert (pc_[0, :t] == ps_[0, :t]).all() and pc_[0, t] != ps_[0, t]
    assert stack_labels(pc_[0], [t], m)[t][0] == 1     # clean -> depth 1
    assert stack_labels(ps_[0], [t], m)[t][0] == 3     # source -> depth 3

    # --- exp 40 directional-specificity primitives ---
    # transport_fraction: clean->0, source->1, halfway->0.5, gap filter, valid mask
    C0, S1 = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    assert transport_fraction(C0, C0, S1, 0.1) == 0.0
    assert transport_fraction(S1, C0, S1, 0.1) == 1.0
    assert abs(transport_fraction(np.array([0.5, 0.5]), C0, S1, 0.1) - 0.5) < 1e-9
    assert np.isnan(transport_fraction(C0, C0, np.array([0.05, 0.05]), 0.1))  # gap<min
    assert transport_fraction(np.array([1.0, 0.0]), C0, S1, 0.1,
                              valid=np.array([True, False])) == 1.0   # masked row dropped

    # read_facet: dispatches depth -> graded cr_cond(k); any other facet -> m=1 observable
    rv, rm = read_facet(q1, "top_type", V, m, k=1)
    ov, om = facet_observable(q1, "top_type", V, m)
    assert np.array_equal(rv, ov, equal_nan=True) and np.array_equal(rm, om)
    dv, dm = read_facet(q1, "depth", V, m, k=0)
    assert np.allclose(dv, cr_cond(q1, V, m, 0), equal_nan=True) and dm[0]

    # drag_fraction: mean|P-C|/gap over valid rows; gap<=0 -> nan
    assert abs(drag_fraction(np.array([0.5, 0.7]), np.array([0.1, 0.1]), 1.0,
                             np.array([True, True])) - 0.5) < 1e-9
    assert np.isnan(drag_fraction(np.array([0.5]), np.array([0.1]), 0.0, np.array([True])))

    # facet_diff_vector: source = clean + per-position constant -> v = that constant
    cr = torch.zeros(5, 6, 3)
    sr = torch.stack([torch.full((5, 3), float(p)) for p in range(6)], dim=1)
    v = facet_diff_vector(cr, sr, t=4)                  # (5, 3)
    assert v.shape == (5, 3)
    for p in range(5):
        assert torch.allclose(v[p], torch.full((3,), float(p)))

    # apply_additive_steer: alpha=0 -> no-op; alpha at [t] adds v[t], off untouched
    base = torch.arange(5 * 6 * 3, dtype=torch.float).reshape(5, 6, 3)
    assert torch.equal(apply_additive_steer(base, v, 4, 0.0, [0, 1, 2, 3, 4]), base[:, :5])
    st = apply_additive_steer(base, v, 4, 2.0, [4])
    assert torch.equal(st[:, :4], base[:, :4])          # off positions intact
    assert torch.allclose(st[:, 4], base[:, 4] + 2.0 * v[4])

    # random_matched_direction: per-position L2 norm matches v
    rv = random_matched_direction(v, np.random.default_rng(0))
    assert rv.shape == v.shape and torch.allclose(rv.norm(dim=-1), v.norm(dim=-1), atol=1e-5)
    print("localize selftest OK")


if __name__ == "__main__":
    _selftest()
