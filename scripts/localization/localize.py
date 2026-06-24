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
import math
import os
import sys
from itertools import product

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from midstream import chain_probs, marginal  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
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


# ===========================================================================
# Component-write enumerator (exp 43, the dynamics/localization rung) — the
# phase's deferred "novel reusable core", earned here at the head level.
#
# Architecture-given component = the ADDITIVE write into the residual at one
# (layer, sublayer/head, position): the embedding, each attention head's
# contribution, and each MLP. Interchange semantics: a PATCHED component
# contributes the SOURCE run's RECORDED write at the prefix positions [0, t];
# UNPATCHED components recompute normally from the running (hybrid) residual.
# Two exact identities make this a clean substrate:
#   - empty splice == the per-head forward's own unpatched run (no-op);
#   - patching EVERY component (emb + all heads + all mlps) reproduces the
#     source residual trajectory exactly, so the conditional equals the source's
#     (the completeness ceiling). Subset patches are honest single-/multi-unit
#     interchange. Continuations are the same V**m grid for any prefix, so the
#     forced-close conditional `cr_cond` reads the patched prefix's belief.
# Heads are resolved by splitting `attn.proj` over head slices; the shared
# proj bias is added once (it cancels in any source-clean DIFFERENCE and is the
# always-clean baseline). Granularity knob (head here; direction is L3, not
# built). All forwards use this per-head path so the experiment is internally
# bit-consistent; it differs from the model's native single-matmul attention by
# ~1e-6 reattribution (asserted in the self-test), so the unpatched reference is
# taken from this path, not from `q_at`.
# Component ids: ("emb",) | (layer, "attn", head) | (layer, "mlp").
# ===========================================================================

def _attn_per_head(blk, x):
    """Per-head attention writes for one Block at residual `x` (B, L, d).
    Returns (per_head: (B, L, H, d), proj_bias: (d,)) with
    per_head.sum(dim=2) + proj_bias == blk.attn(blk.ln1(x)) (up to fp reassoc)."""
    B, L, D = x.shape
    h, dk = blk.attn.h, blk.attn.dk
    xn = blk.ln1(x)
    q, k, v = blk.attn.qkv(xn).split(D, dim=2)
    shp = (B, L, h, dk)
    q, k, v = (tt.view(shp).transpose(1, 2) for tt in (q, k, v))      # (B,h,L,dk)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
    att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
    y = (att @ v).transpose(1, 2).contiguous().view(B, L, D)          # head-concat
    W = blk.attn.proj.weight                                          # (D_out, D_in)
    writes = [y[:, :, hh * dk:(hh + 1) * dk] @ W[:, hh * dk:(hh + 1) * dk].T
              for hh in range(h)]
    return torch.stack(writes, dim=2), blk.attn.proj.bias             # (B,L,H,D),(D,)


def _run_components(model, idx, t, inject=None, record=False):
    """Per-head forward over `idx` (B, L). If `record`, return (x_final, rec) with
    `rec[comp_id]` = that component's write at prefix positions [0, t] (B, t+1, d).
    If `inject` (a dict comp_id -> (values (B, P, d), pos_idx (P,) LongTensor)),
    replace that component's write at the given prefix positions with `values`;
    unspecified positions/components recompute normally. Returns x_final (B, L, d)."""
    dev = next(model.parameters()).device
    B, L = idx.shape
    inj = inject or {}

    def _put(w, cid):
        if cid in inj:
            vals, pos = inj[cid]
            w = w.clone()
            w[:, pos.to(w.device)] = vals.to(w.dtype).to(w.device)
        return w

    rec = {}
    x = model.tok(idx.to(dev)) + model.pos(torch.arange(L, device=dev))
    if record:
        rec[("emb",)] = x[:, :t + 1].clone()
    if ("emb",) in inj:
        vals, pos = inj[("emb",)]
        x = x.clone(); x[:, pos.to(dev)] = vals.to(x.dtype).to(dev)
    for li, blk in enumerate(model.blocks):
        per_head, pbias = _attn_per_head(blk, x)                      # (B,L,H,D),(D,)
        if record:
            for hh in range(per_head.shape[2]):
                rec[(li, "attn", hh)] = per_head[:, :t + 1, hh].clone()
        if inj:
            per_head = per_head.clone()
            for hh in range(per_head.shape[2]):
                cid = (li, "attn", hh)
                if cid in inj:
                    vals, pos = inj[cid]
                    per_head[:, pos.to(dev), hh] = vals.to(per_head.dtype).to(dev)
        x = x + per_head.sum(dim=2) + pbias
        mlp_write = blk.mlp(blk.ln2(x))
        if record:
            rec[(li, "mlp")] = mlp_write[:, :t + 1].clone()
        mlp_write = _put(mlp_write, (li, "mlp"))
        x = x + mlp_write
    return (x, rec) if record else x


def component_ids(n_layers, n_heads, include_emb=True):
    """The enumerated component ids in a fixed order (emb, then per layer:
    each attention head, then mlp)."""
    ids = [("emb",)] if include_emb else []
    for li in range(n_layers):
        ids += [(li, "attn", hh) for hh in range(n_heads)]
        ids.append((li, "mlp"))
    return ids


def record_component_writes(model, idx, t):
    """Recorded per-component prefix writes for `idx` (n, L): dict
    comp_id -> (n, t+1, d). The SOURCE side of an interchange; cached once per pair
    set and re-used across single-/subset-component splices."""
    with torch.no_grad():
        _, rec = _run_components(model, torch.as_tensor(idx), t, record=True)
    return rec


def _splice_spec(splice, t):
    """Normalize a splice argument to {cid: positions LongTensor}. A list/iterable of
    cids means all prefix positions [0, t]; a dict cid -> positions keeps the subset."""
    allpos = torch.arange(t + 1)
    if isinstance(splice, dict):
        return {cid: torch.as_tensor(np.asarray(p), dtype=torch.long)
                for cid, p in splice.items()}
    return {cid: allpos for cid in splice}


def chain_probs_components(model, X_cont, t, m, V, source_rec, splice):
    """Exact m-step completion joint at `t` under a COMPONENT interchange: the
    components in `splice` take the source's recorded prefix writes (at the named
    positions); all others recompute. `splice` is a list of cids (all prefix
    positions) or a dict cid -> positions subset of [0, t]. `X_cont` (n, C, L) =
    clean prefixes with every length-m continuation at t+1..t+m (`make_Xc`).
    `source_rec` from `record_component_writes` on the index-aligned source rows.
    Empty -> the per-head unpatched reference; all ids over all positions -> the
    source ceiling. Returns q (n, C). Runs on the model's device (MPS/CUDA)."""
    n, C, L = X_cont.shape
    flat = X_cont.reshape(n * C, L)
    spec = _splice_spec(splice, t)
    out = np.empty(n * C)
    with torch.no_grad():
        for i in range(0, n * C, 1024):
            sl = slice(i, min(i + 1024, n * C))
            pair = torch.as_tensor(np.arange(sl.start, sl.stop) // C, dtype=torch.long)
            inj = {cid: (source_rec[cid][pair][:, pos], pos) for cid, pos in spec.items()}
            x = _run_components(model, torch.from_numpy(flat[sl]), t, inject=inj)
            probs = torch.softmax(model.head(model.ln_f(x)), dim=-1).cpu().double().numpy()
            r = np.arange(sl.stop - sl.start)
            q = np.ones(sl.stop - sl.start)
            for j in range(m):
                q *= probs[r, t + j, flat[sl][:, t + 1 + j]]
            out[sl] = q
    return out.reshape(n, C)


def splice_logits(model, idx, readout_end, source_rec, splice):
    """Softmax probs (n, L, V) under a component interchange splice over positions
    [0, readout_end]. For the DETERMINISTIC forced-close instrument (a single teacher-
    forced continuation, no V**m grid): the caller appends fixed tokens to `idx` and
    reads a position's distribution. `source_rec` from `record_component_writes` on the
    index-aligned source rows, recorded to >= readout_end. `splice` is a list of cids
    (all positions [0, readout_end]) or a dict cid -> positions subset. Runs on the
    model's device."""
    n = idx.shape[0]
    spec = _splice_spec(splice, readout_end)
    outs = []
    with torch.no_grad():
        for i in range(0, n, 1024):
            sl = slice(i, min(i + 1024, n))
            pair = torch.arange(sl.start, sl.stop)
            inj = {cid: (source_rec[cid][pair][:, pos], pos) for cid, pos in spec.items()}
            x = _run_components(model, torch.as_tensor(np.asarray(idx[sl])),
                                readout_end, inject=inj)
            outs.append(torch.softmax(model.head(model.ln_f(x)), dim=-1).cpu().double().numpy())
    return np.concatenate(outs, axis=0)


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

    # --- exp 43 component-write enumerator (tiny random model, no checkpoint) ---
    torch.manual_seed(0)
    cfg = GPTConfig(vocab=V, seq_len=10, d_model=16, n_heads=4, n_layers=3, d_mlp=32)
    gm = GPT(cfg).eval()
    rng = np.random.default_rng(1)
    Lc = cfg.seq_len
    clean = rng.integers(0, V, size=(6, Lc))
    src = rng.integers(0, V, size=(6, Lc))
    tt = 5
    ids = component_ids(cfg.n_layers, cfg.n_heads)
    assert ids[0] == ("emb",) and (0, "attn", 0) in ids and (2, "mlp") in ids
    assert len(ids) == 1 + cfg.n_layers * (cfg.n_heads + 1)

    # per-head writes sum (+bias) to the model's native attention output
    xt = gm.tok(torch.from_numpy(clean)) + gm.pos(torch.arange(Lc))
    ph, pb = _attn_per_head(gm.blocks[0], xt)
    assert torch.allclose(ph.sum(dim=2) + pb, gm.blocks[0].attn(gm.blocks[0].ln1(xt)),
                          atol=1e-5)

    Xc_cl = make_Xc(clean, tt, m, V)
    Xc_sr = make_Xc(src, tt, m, V)
    rec_src = record_component_writes(gm, src, tt)
    assert rec_src[("emb",)].shape == (6, tt + 1, cfg.d_model)
    # (1) empty splice reproduces the per-head unpatched run, bit-exact
    q_noop = chain_probs_components(gm, Xc_cl, tt, m, V, rec_src, [])
    q_ref = chain_probs_components(gm, Xc_cl, tt, m, V, rec_src, [])
    assert np.array_equal(q_noop, q_ref)
    # (2) completeness: splicing EVERY component reproduces the SOURCE conditional
    rec_self = record_component_writes(gm, src, tt)         # source recorded on itself
    q_all = chain_probs_components(gm, Xc_cl, tt, m, V, rec_src, ids)
    q_src = chain_probs_components(gm, Xc_sr, tt, m, V, rec_self, [])
    assert np.allclose(q_all, q_src, atol=1e-6), np.abs(q_all - q_src).max()
    # (3) a single-component splice is a strict subset effect (changes <= the all-splice)
    q_one = chain_probs_components(gm, Xc_cl, tt, m, V, rec_src, [(1, "attn", 0)])
    assert not np.array_equal(q_one, q_noop)               # it does something
    # (4) emb-only splice differs from internal-only-all (carried vs recomputed split)
    internal = [c for c in ids if c != ("emb",)]
    q_emb = chain_probs_components(gm, Xc_cl, tt, m, V, rec_src, [("emb",)])
    q_int = chain_probs_components(gm, Xc_cl, tt, m, V, rec_src, internal)
    assert not np.allclose(q_emb, q_int, atol=1e-6)        # the two ceilings differ
    # (5) position-resolved splice: a component at ALL prefix positions (dict form)
    #     equals the list form; at NO positions equals the no-op.
    q_dict_all = chain_probs_components(gm, Xc_cl, tt, m, V, rec_src,
                                        {(1, "attn", 0): np.arange(tt + 1)})
    assert np.array_equal(q_dict_all, q_one)
    q_dict_none = chain_probs_components(gm, Xc_cl, tt, m, V, rec_src,
                                         {(1, "attn", 0): np.array([], int)})
    assert np.array_equal(q_dict_none, q_noop)
    # (6) splice_logits (deterministic instrument): empty ~ native at readout_end;
    #     all components incl emb == source's own logits (completeness).
    re = tt + 2
    rec_src_re = record_component_writes(gm, src, re)
    pr_noop = splice_logits(gm, clean, re, rec_src_re, [])
    with torch.no_grad():
        pr_native = torch.softmax(gm.head(gm.ln_f(_run_components(
            gm, torch.from_numpy(clean), re))), dim=-1).cpu().double().numpy()
    assert np.allclose(pr_noop, pr_native, atol=1e-6)
    pr_all = splice_logits(gm, clean, re, rec_src_re,
                           component_ids(cfg.n_layers, cfg.n_heads, True))
    rec_self_re = record_component_writes(gm, src, re)
    pr_src = splice_logits(gm, src, re, rec_self_re, [])
    assert np.allclose(pr_all[:, :re + 1], pr_src[:, :re + 1], atol=1e-6)
    print("localize selftest OK")


if __name__ == "__main__":
    _selftest()
