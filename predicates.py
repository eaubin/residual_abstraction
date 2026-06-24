"""
predicates.py — the completion-predicate layer.

A completion predicate scores the model's next-`m`-token continuation. The
phase (`docs/COMPLETION_PREDICATES.md`) generalises the original Boolean
events to **graded** predicates:

    phi: V^m -> [0, 1]              (a span score, not a boolean event)

Its phase-relevant quantity is the *expected score* under a completion
distribution q in Delta(V^m):

    E_q[phi] = sum_c phi(c) * q(c) = mask . q          (one dot product)

A Boolean event is the special case phi(c) in {0, 1}, for which E_q[phi] is the
truth-probability p_phi. So the graded layer subsumes the older Boolean API
(REGISTERED / *_pphi) below, which frozen scripts (exp 29, intervention_eval)
still import unchanged.

Two estimators of E_q[phi], identical algebra, different q:
  - exact   (ground truth): q = process m-gram table  -> ``eq_exact``
  - observable (model):      q = model chain_probs     -> ``eq_obs`` / ``eq_obs_from_model``

phi is precomputed as a float mask by enumerating V^m once
(``graded_mask``), so a predicate is a (graded) rank-1 functional of the
completion distribution. Composition is exact and cheap by construction: for
within-horizon predicates phi_{A and B}(c) = phi_A(c) * phi_B(c), so the
product t-norm ``mask_and`` *is* the exact E_q[phi_{A and B}] mask — no
automata product is needed (registered scope decision: bounded, enumerable).

DESIGN DECISIONS inherited from the phase map (load-bearing):
  - Criterion is mean-predicate-error |E_q[phi] - E_q'[phi]|, not KL (dec. 1).
  - Predicate menu is a small fixed set of bounded templates, not a language
    (dec. 7). Two prefix-FREE templates are implemented here — threshold-count
    and net-return — per the leash "start with one or two trivial predicates
    computed directly from q; build a template only when a predicate needs
    it." The prefix-SEEDED templates (bounded reach to absolute depth 0,
    next-match binding, bounded order) are deferred to the first binding claim,
    which co-decides the claim toy (colored Dyck vs mode x stack); they are
    intentionally not stubbed here.
  - Horizon m is a per-experiment parameter (dec. 6), not fixed at 3.

Device note: this module is pure numpy (masks, dots) and needs no accelerator.
The only model-touching helper, ``eq_obs_from_model``, defers to
``midstream.chain_probs`` on a model already moved to MPS/CUDA by
``expcommon.load_model`` — never run that path on a CPU-resident model.

The four legacy ``REGISTERED`` predicates are pstack-specific (vocab 0..5:
0,1 open type 0/1; 2,3 close type 0/1; 4,5 neutral). The graded templates
below are vehicle-agnostic (you pass the symbol sets), so they run on dyck2
(vocab 0..3) — the machinery testbed — as well as pstack.
"""

from itertools import product

import numpy as np


def continuations(V, m):
    """All V^m continuations, in the mgram-table / chain_probs row order."""
    return np.asarray(list(product(range(V), repeat=m)), dtype=np.int64)


def predicate_mask(fn, V, m):
    """Boolean mask (length V^m) of a predicate fn(tuple-of-m-tokens).

    Back-compat: collapses any truthy return to 1.0. For graded predicates use
    ``graded_mask``."""
    return np.array([1.0 if fn(tuple(c)) else 0.0
                     for c in continuations(V, m)], dtype=np.float64)


def graded_mask(fn, V, m):
    """Float mask (length V^m) of a graded predicate fn(tuple) -> [0, 1].

    Validates the range so a mis-scaled template is caught at construction
    rather than silently distorting E_q[phi]."""
    msk = np.array([float(fn(tuple(c))) for c in continuations(V, m)],
                   dtype=np.float64)
    if msk.min() < -1e-12 or msk.max() > 1.0 + 1e-12:
        raise ValueError(f"graded predicate out of [0,1]: "
                         f"[{msk.min():.4g}, {msk.max():.4g}]")
    return np.clip(msk, 0.0, 1.0)


# ---- the registered template menu (bounded, saturating, prefix-free) --------
# Each constructor returns a graded predicate fn(continuation-tuple) -> [0,1].
# Vehicle-agnostic: the caller passes the symbol sets for the vocabulary.

def tmpl_threshold_count(symbols, c, graded=False):
    """'#(symbol in `symbols`) >= c within the next m' (dec. 7 threshold-count).

    Counting saturates at c (count up to a small constant, then stop) so the
    template is finite by construction. ``graded=True`` returns the saturating
    ramp min(count, c)/c in [0,1] (a genuine span score); the default is the
    Boolean event count >= c."""
    syms = frozenset(symbols)
    if c <= 0:
        raise ValueError("threshold c must be >= 1")

    def phi(cont):
        n = 0
        for t in cont:
            if t in syms:
                n += 1
                if n >= c:          # saturate
                    break
        return (min(n, c) / c) if graded else float(n >= c)
    return phi


def tmpl_net_return(opens, closes, c=1):
    """'net depth change over the window <= -c' (returns toward 0 by >= c).

    Prefix-free: opens count +1, closes -1, everything else 0; relative to the
    window start, so no prefix depth is needed. Generalises the legacy
    phi2_net_return (opens={0,1}, closes={2,3}, c=1)."""
    op, cl = frozenset(opens), frozenset(closes)

    def phi(cont):
        net = sum(1 if t in op else -1 if t in cl else 0 for t in cont)
        return float(net <= -c)
    return phi


# ---- prefix-seeded predicates (the ctx-reader seam) -------------------------
# A prefix-seeded predicate's DEFINITION references a feature of the prefix
# (here, the top stack frame). The ctx-reader maps a process hidden-state label
# to a small descriptor; the predicate then scores continuations GIVEN that
# descriptor, so its mask is built per distinct ctx. For (colored) Dyck the ctx
# is fully determined by the visible tokens, so the belief is one-hot
# (synchronized) and argmax recovers it exactly -- a deterministic ctx-reader.
# A latent-feature toy (e.g. hidden mode) would force a belief-integrated
# reader; that is deferred with its toy.

def top_frame_ctx(state_label):
    """ctx-reader for (colored) Dyck: the top stack frame, or None if empty.

    state_label is the process hidden-state label (a stack tuple). Returns a
    (type, color) pair for colored_dyck2, or a bare int type for dyck2."""
    return state_label[-1] if state_label else None


def _dyck_vocab(V):
    """(is_close, decode_close) for a (colored) Dyck vocabulary, by V.
    V=4 dyck2: opens 0,1 / closes 2,3 -> (type, None). V=8 colored: opens 0..3
    / closes 4..7 -> (type, color)."""
    if V == 8:
        return (lambda t: t >= 4,
                lambda t: ((t - 4) // 2, (t - 4) % 2))
    if V == 4:
        return (lambda t: t in (2, 3),
                lambda t: (t - 2, None))
    raise ValueError(f"next-close-matches expects a (colored) Dyck vocab, got V={V}")


def tmpl_next_close_matches(facet="both", V=8):
    """'the close that pops the prefix top matches it' on `facet` in
    {"type","color","both"} -- a prefix-seeded binding predicate (dec. 7
    next-match binding), the seeded generalisation of phi4_first_matched.

    Simulate a local stack seeded with the prefix top frame; the predicate
    fires iff the continuation token that finally pops that seeded frame
    matches it on the requested facet(s). On the TRUE model this is ~1 whenever
    the top is closed within the window (matching is grammar-forced); it
    discriminates only under an abstraction that has dropped the bound facet
    (see the color-scramble self-test). `V` fixes the vocabulary: facet="type"
    works on dyck2 (V=4); "color"/"both" need colored_dyck2 (V=8). Returns
    phi(cont, ctx)."""
    is_close, decode = _dyck_vocab(V)

    def phi(cont, ctx):
        if ctx is None:
            return 0.0
        ctop = ctx if isinstance(ctx, tuple) else (ctx, None)
        depth = 1                       # the seeded prefix top is the 1 frame
        for t in cont:
            if not is_close(t):
                depth += 1              # an open pushes a local frame
                continue
            depth -= 1                  # a close pops
            if depth == 0:              # popped the seeded prefix top
                ty, co = decode(t)
                if facet == "type":
                    return float(ty == ctop[0])
                if facet == "color":
                    return float(co == ctop[1])
                return float(ty == ctop[0] and co == ctop[1])
        return 0.0                      # prefix top not closed within window
    return phi


def ctx_along(proc, beliefs, reader=top_frame_ctx, tol=1e-6):
    """Per-prefix ctx from exact beliefs (argmax hidden state). Returns the
    ctx list and the count of prefixes whose belief is NOT one-hot (the
    not-synchronized / latent-ambiguity count; 0 for valid Dyck prefixes)."""
    ctxs, ambiguous = [], 0
    for b in beliefs:
        j = int(np.argmax(b))
        if b[j] < 1.0 - tol:
            ambiguous += 1
        ctxs.append(reader(proc.states[j]))
    return ctxs, ambiguous


def eq_exact_seeded(proc, beliefs, seeded_fn, m, reader=top_frame_ctx):
    """Exact E_q[phi] for a prefix-seeded predicate: group prefixes by ctx,
    build one mask per distinct ctx, dot with that prefix's exact m-gram.
    Returns (values (n,), ambiguous-count)."""
    ctxs, amb = ctx_along(proc, beliefs, reader)
    mgram = proc.mgram_table(beliefs, m)
    cache, out = {}, np.empty(len(beliefs))
    for i, ctx in enumerate(ctxs):
        if ctx not in cache:
            cache[ctx] = graded_mask(lambda c, x=ctx: seeded_fn(c, x), proc.V, m)
        out[i] = cache[ctx] @ mgram[i]
    return out, amb


def scramble_color(q, V, m):
    """Symmetrise a completion distribution over CLOSE-token color: average
    each continuation with its color-flipped twin (close 4<->5, 6<->7; opens
    untouched). Makes 'next close color' uninformative while preserving type
    structure -- the synthetic color-blind abstraction the binding instrument
    must detect. Pure relabelling on the V^m simplex; no model needed."""
    conts = continuations(V, m)
    index = {tuple(c): i for i, c in enumerate(conts)}
    perm = np.empty(len(conts), dtype=np.int64)
    for i, c in enumerate(conts):
        flipped = tuple((t ^ 1) if t >= 4 else t for t in c)   # flip color bit
        perm[i] = index[flipped]
    q = np.atleast_2d(q)
    return 0.5 * (q + q[:, perm])


# ---- the four legacy (Boolean, pstack) predicates ---------------------------
# Kept verbatim for frozen scripts (exp 29 predicate-targeting, intervention_eval).

def phi_next_closes(c):
    """phi1: the next token closes a bracket (common, high-variance)."""
    return c[0] in (2, 3)


def phi_net_return(c):
    """phi2: net depth change over the window is <= -1 (returns toward 0)."""
    net = sum((1 if t in (0, 1) else -1 if t in (2, 3) else 0) for t in c)
    return net <= -1


def phi_all_neutral(c):
    """phi3: all m tokens are neutral terminals (the hidden-mode emission)."""
    return all(t in (4, 5) for t in c)


def phi_first_matched(c):
    """phi4: the first opened bracket in the window is closed by a matching
    close inside the window (a within-window temporal/binding predicate).

    Simulate a local stack from empty; the first open is 'matched' iff a
    matching close (same type) pops it within c."""
    stack = []
    first_depth = None
    for t in c:
        if t in (0, 1):
            stack.append(t)
            if first_depth is None:
                first_depth = len(stack) - 1
        elif t in (2, 3):
            if stack and (t - 2) == stack[-1]:
                popped = len(stack) - 1
                stack.pop()
                if first_depth is not None and popped == first_depth:
                    return True
    return False


REGISTERED = {
    "phi1_next_closes": phi_next_closes,
    "phi2_net_return": phi_net_return,
    "phi3_all_neutral": phi_all_neutral,
    "phi4_first_matched": phi_first_matched,
}


def registered_masks(V, m):
    return {name: predicate_mask(fn, V, m) for name, fn in REGISTERED.items()}


# ---- minimal predicate algebra (graded; product t-norm) ---------------------

def mask_not(a):
    return 1.0 - a


def mask_and(a, b):
    """phi_A and phi_B. For within-horizon predicates this is the EXACT
    E_q[phi_{A and B}] mask (phi_{A and B}(c) = phi_A(c) * phi_B(c)); no
    automata product is needed."""
    return a * b


def mask_or(a, b):
    return np.clip(a + b - a * b, 0.0, 1.0)


# ---- E_q[phi] estimators ----------------------------------------------------
# E_q[phi] = mask . q. The exact estimator uses the process m-gram; the
# observable estimator uses the model's completion distribution. Same algebra.

def eq_obs(q, mask):
    """Observable E_q[phi] per row from a model completion distribution q
    (n, V^m). Returns (n,)."""
    return q @ mask


def eq_exact(beliefs, mask, proc, m):
    """Exact ground-truth E_q[phi] per belief: mask . mgram(belief). Returns
    (n,). O(V^m) per row, fully exact."""
    return proc.mgram_table(beliefs, m) @ mask


# Back-compat aliases (Boolean truth-probability names; frozen scripts import
# these). For Boolean phi, E_q[phi] == p_phi exactly.
obs_pphi = eq_obs
exact_pphi = eq_exact


def eq_obs_from_model(model, X_cont, layer, prefix_state, t, m, V, mask):
    """Observable E_q[phi] straight from a model (device-aware).

    Defers to ``midstream.chain_probs`` (model must already be on its
    accelerator via ``expcommon.load_model``; never CPU for a heavy pass).
    Imported lazily so the pure-numpy layer carries no torch dependency."""
    from midstream import chain_probs
    q, _ = chain_probs(model, X_cont, layer, prefix_state, t, m, V)
    return eq_obs(q, mask)


# ---- baselines (the in-layer, model-free ones) ------------------------------
# no-information and raw-m-gram are computable from masks/q alone. The patch
# baselines (full-patch, reference-patch) need the model + intervention harness
# and live in battery.py / the experiment scripts, not here (reuse, not copy).

def baseline_uniform(mask, V, m):
    """No-information baseline: E_q[phi] under a uniform q over V^m = mean(mask).
    The floor every sufficiency claim must clear."""
    return float(mask.mean())


# ---- verdict-branch detectors (the layer's falsifiers) ----------------------

def is_vacuous(mask, tol=1e-3):
    """PREDICATE_VACUOUS: phi is ~constant over V^m, so E_q[phi] cannot vary
    with q and the predicate carries no signal. (Per-q non-vacuity — that
    E_q[phi] actually varies across the *prefixes* in an experiment — is a
    stronger check the experiment makes against its own q distribution.)"""
    return float(mask.std()) < tol


def obs_exact_drift(eq_obs_vals, eq_exact_vals):
    """OBS_EXACT_DRIFT magnitude: max |E_q_obs[phi] - E_q_exact[phi]|. The
    observable estimator must track the exact one before any predicate claim;
    a large value routes to measurement repair, not semantic expansion."""
    return float(np.max(np.abs(np.asarray(eq_obs_vals)
                               - np.asarray(eq_exact_vals))))


def composition_interaction(mask_a, mask_b, q):
    """Interaction term of A,B under q: E_q[A and B] - E_q[A]*E_q[B], per row.

    Zero (within tolerance) means A,B are independent under q and E_q[A and B]
    factorises; non-zero is a real interaction (the COMPOSITION_FAIL branch
    when a registered direct-sum/independence relation was claimed). Note
    E_q[A and B] itself is always EXACT here via ``mask_and`` — what can fail is
    a *factorisation assumption*, which is exactly what this measures."""
    q = np.atleast_2d(q)
    e_ab = eq_obs(q, mask_and(mask_a, mask_b))
    e_a = eq_obs(q, mask_a)
    e_b = eq_obs(q, mask_b)
    return e_ab - e_a * e_b


# ---- self-tests -------------------------------------------------------------
# Hand-computable checks plus an adversarial case for each verdict branch.

def _selftest():
    V, m = 6, 3

    # --- legacy Boolean predicates (unchanged contract) ---------------------
    masks = registered_masks(V, m)
    for name, msk in masks.items():
        assert msk.shape == (V ** m,) and set(np.unique(msk)) <= {0.0, 1.0}, name
    assert phi_next_closes((2, 4, 4)) and not phi_next_closes((0, 2, 4))
    assert phi_net_return((2, 3, 4)) and not phi_net_return((0, 1, 4))
    assert phi_all_neutral((4, 5, 4)) and not phi_all_neutral((4, 0, 4))
    assert phi_first_matched((0, 2, 4))
    assert not phi_first_matched((0, 3, 4))
    assert not phi_first_matched((0, 1, 3))
    q = np.ones((1, V ** m)) / V ** m
    assert abs(eq_obs(q, masks["phi3_all_neutral"])[0]
               - (2 ** m) / V ** m) < 1e-12
    a, b = masks["phi1_next_closes"], masks["phi3_all_neutral"]
    assert np.array_equal(mask_and(a, b), np.zeros_like(a))   # disjoint
    assert np.allclose(mask_not(a), 1.0 - a)

    # --- graded templates (the phase layer) ---------------------------------
    # threshold-count, Boolean: ">=2 closers in next 3" on a dyck2 vocab.
    fn_b = tmpl_threshold_count(symbols={2, 3}, c=2, graded=False)
    assert fn_b((2, 3, 0)) == 1.0 and fn_b((2, 0, 0)) == 0.0
    # graded ramp saturates at c: 0,1,2,2 closers -> 0, .5, 1, 1.
    fn_g = tmpl_threshold_count(symbols={2, 3}, c=2, graded=True)
    assert fn_g((0, 0, 0)) == 0.0
    assert abs(fn_g((2, 0, 0)) - 0.5) < 1e-12
    assert fn_g((2, 3, 0)) == 1.0 and fn_g((2, 3, 2)) == 1.0   # saturated
    mg = graded_mask(fn_g, V=4, m=3)
    assert mg.shape == (4 ** 3,) and 0.0 <= mg.min() and mg.max() <= 1.0
    assert set(np.unique(mg)) == {0.0, 0.5, 1.0}               # genuinely graded
    # net-return generalises legacy phi2 exactly (opens 0,1 / closes 2,3).
    fn_nr = tmpl_net_return(opens={0, 1}, closes={2, 3}, c=1)
    assert np.array_equal(graded_mask(fn_nr, V, m),
                          predicate_mask(phi_net_return, V, m))

    # --- exact vs observable estimator on a real process --------------------
    from processes import dyck2
    proc = dyck2()
    rng = np.random.default_rng(0)
    X = proc.sample(8, 30, rng)
    beliefs = np.stack([proc.beliefs_along(row)[10] for row in X])
    mask = graded_mask(tmpl_threshold_count({2, 3}, 2, graded=True), proc.V, m)
    ex = eq_exact(beliefs, mask, proc, m)
    assert ex.shape == (8,) and (0.0 <= ex).all() and (ex <= 1.0).all()
    # exact mgram fed in as the "observable" q -> zero drift (estimators agree).
    q_exact = proc.mgram_table(beliefs, m)
    assert obs_exact_drift(eq_obs(q_exact, mask), ex) < 1e-12

    # --- verdict branch: PREDICATE_VACUOUS ----------------------------------
    # threshold c=1 over ALL symbols is always true -> constant mask -> vacuous.
    vac = graded_mask(tmpl_threshold_count(set(range(proc.V)), 1), proc.V, m)
    assert is_vacuous(vac) and not is_vacuous(mask)

    # --- verdict branch: OBS_EXACT_DRIFT ------------------------------------
    # a deliberately wrong q (mgram + renormalised noise) must trip the drift.
    noisy = q_exact + 0.3 * rng.random(q_exact.shape)
    noisy /= noisy.sum(axis=1, keepdims=True)
    assert obs_exact_drift(eq_obs(noisy, mask), ex) > 1e-3

    # --- verdict branch: COMPOSITION_FAIL (interaction term) ----------------
    # composition is always exact; what can be non-zero is the factorisation.
    mA = graded_mask(tmpl_threshold_count({2, 3}, 1), proc.V, m)   # >=1 close
    mB = graded_mask(tmpl_net_return({0, 1}, {2, 3}, 1), proc.V, m)  # net<=-1
    # exactness of mask_and: enumerated phi_{A and B} == product mask.
    fnA = tmpl_threshold_count({2, 3}, 1)
    fnB = tmpl_net_return({0, 1}, {2, 3}, 1)
    direct = graded_mask(lambda c: float(fnA(c) and fnB(c)), proc.V, m)
    assert np.array_equal(direct, mask_and(mA, mB))
    # these two are dependent under q (a close drives both) -> interaction != 0.
    inter = composition_interaction(mA, mB, q_exact)
    assert np.max(np.abs(inter)) > 1e-3
    # an independent construction -> interaction ~ 0: split the window so A
    # reads position 0 and B reads position 2 of an i.i.d.-token uniform q.
    qi = np.ones(proc.V ** m) / proc.V ** m
    mA0 = graded_mask(lambda c: float(c[0] in (2, 3)), proc.V, m)
    mB2 = graded_mask(lambda c: float(c[2] in (2, 3)), proc.V, m)
    assert abs(composition_interaction(mA0, mB2, qi)[0]) < 1e-12

    # --- prefix-seeded binding on colored Dyck (the ctx-reader seam) --------
    from processes import colored_dyck2
    cproc = colored_dyck2()
    assert cproc.V == 8
    Xc = cproc.sample(64, 30, rng)
    # Select SYNCHRONIZED prefixes (belief one-hot) with a non-empty stack, so
    # the top frame (the ctx) is observed and deterministic. Stationary-prior
    # prefixes that never returned to depth 0 carry an inherited, unobserved
    # top -- a separate memory-of-prior phenomenon, excluded here.
    Bc = []
    for row in Xc:
        b = cproc.beliefs_along(row)[12]
        j = int(np.argmax(b))
        if b[j] > 1.0 - 1e-6 and top_frame_ctx(cproc.states[j]) is not None:
            Bc.append(b)
    Bc = np.stack(Bc)
    assert len(Bc) >= 20, f"too few synchronized prefixes ({len(Bc)})"
    mm = 4                                  # horizon long enough to nest+close
    fn_both = tmpl_next_close_matches("both", V=8)
    fn_type = tmpl_next_close_matches("type", V=8)
    fn_color = tmpl_next_close_matches("color", V=8)
    e_both, amb = eq_exact_seeded(cproc, Bc, fn_both, mm)
    e_type, _ = eq_exact_seeded(cproc, Bc, fn_type, mm)
    e_color, _ = eq_exact_seeded(cproc, Bc, fn_color, mm)
    # selected prefixes are synchronized -> deterministic ctx, no ambiguity.
    assert amb == 0, f"selected prefixes must be synchronized, got {amb} ambiguous"
    # on the TRUE model, matching is forced, so all three facets agree and
    # equal P(top closed in window): in (0,1), not constant (non-vacuous).
    assert np.allclose(e_both, e_type) and np.allclose(e_both, e_color)
    assert 0.0 < e_both.mean() < 1.0 and e_both.std() > 1e-3

    # the discrimination: a color-blind abstraction (color-scrambled q) must
    # drop matches_color toward chance while leaving matches_type intact.
    mgram_c = cproc.mgram_table(Bc, mm)
    q_blind = scramble_color(mgram_c, cproc.V, mm)
    ctxs, _ = ctx_along(cproc, Bc)
    m_type = {x: graded_mask(lambda c, x=x: fn_type(c, x), cproc.V, mm)
              for x in set(ctxs)}
    m_color = {x: graded_mask(lambda c, x=x: fn_color(c, x), cproc.V, mm)
               for x in set(ctxs)}
    e_type_blind = np.array([m_type[x] @ q_blind[i] for i, x in enumerate(ctxs)])
    e_color_blind = np.array([m_color[x] @ q_blind[i] for i, x in enumerate(ctxs)])
    assert np.allclose(e_type_blind, e_type), "type must survive a color scramble"
    assert (e_color.mean() - e_color_blind.mean()) > 0.1, \
        "color-blind abstraction must drop matches_color"

    print("predicates selftest passed: legacy Boolean, graded templates, "
          "exact/observable estimators, the vacuous / drift / composition "
          "verdict branches, and prefix-seeded colored-Dyck binding "
          "(ctx-reader + color-scramble discrimination) all fire.")


if __name__ == "__main__":
    _selftest()
