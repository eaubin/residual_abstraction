"""
predicates.py — the completion-predicate layer (ORIGINAL_SIN.md scaffolding).

A within-horizon predicate is a Boolean function phi: V^m -> {0,1}, i.e. a
fixed mask over the V^m continuations. Its truth-probability under a
completion distribution q in Delta(V^m) is one dot product:

    p_phi = sum_c phi(c) * q(c)

- observable (model): obs_pphi(q_model, mask)
- exact (ground truth): exact_pphi(beliefs, mask, proc, m)

q_exact comes from the process m-gram tables (proc.mgram_table); phi is
precomputed by enumerating V^m once. A predicate is thus a rank-1 (linear-
functional) abstraction of the completion distribution. This is a reusable
library module, not a one-off — frozen scripts and the living edge both
import it.

The four registered pstack predicates span behavioral salience; pstack
tokens are 0,1 = open type 0/1, 2,3 = close type 0/1 (2 closes a type-0
open, 3 a type-1), 4,5 = neutral.
"""

from itertools import product

import numpy as np


def continuations(V, m):
    """All V^m continuations, in the mgram-table / chain_probs row order."""
    return np.asarray(list(product(range(V), repeat=m)), dtype=np.int64)


def predicate_mask(fn, V, m):
    """Boolean mask (length V^m) of a predicate fn(tuple-of-m-tokens)."""
    return np.array([1.0 if fn(tuple(c)) else 0.0
                     for c in continuations(V, m)], dtype=np.float64)


# ---- the four registered predicates (functions of a continuation tuple) ----

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


# ---- minimal predicate algebra (ORIGINAL_SIN.md: not / and / or) -----------

def mask_not(a):
    return 1.0 - a


def mask_and(a, b):
    return a * b


def mask_or(a, b):
    return np.clip(a + b - a * b, 0.0, 1.0)


# ---- truth probabilities ---------------------------------------------------

def obs_pphi(q, mask):
    """Observable p_phi per row from a model completion distribution q
    (n, V^m). Returns (n,)."""
    return q @ mask


def exact_pphi(beliefs, mask, proc, m):
    """Exact ground-truth p_phi per belief: mask . mgram(belief). Returns
    (n,). O(V^m) per row, fully exact."""
    return proc.mgram_table(beliefs, m) @ mask


def _selftest():
    V, m = 6, 3
    masks = registered_masks(V, m)
    for name, msk in masks.items():
        assert msk.shape == (V ** m,) and set(np.unique(msk)) <= {0.0, 1.0}, name
    # hand checks on specific continuations
    assert phi_next_closes((2, 4, 4)) and not phi_next_closes((0, 2, 4))
    assert phi_net_return((2, 3, 4)) and not phi_net_return((0, 1, 4))
    assert phi_all_neutral((4, 5, 4)) and not phi_all_neutral((4, 0, 4))
    # phi4: open type-0 then matching close 2 within window -> matched
    assert phi_first_matched((0, 2, 4))
    # open type-0 then close type-1 (3, non-matching) -> not matched
    assert not phi_first_matched((0, 3, 4))
    # nested: open0, open1, close1, close0 -> first (open0) matched at end...
    # window m=3 truncates: (0,1,3) -> open0,open1,close1 pops open1 not open0
    assert not phi_first_matched((0, 1, 3))
    # a uniform q gives p_phi = (#true)/V^m
    q = np.ones((1, V ** m)) / V ** m
    assert abs(obs_pphi(q, masks["phi3_all_neutral"])[0]
               - (2 ** m) / V ** m) < 1e-12          # 2^3 all-neutral / 6^3
    # algebra
    a, b = masks["phi1_next_closes"], masks["phi3_all_neutral"]
    assert np.array_equal(mask_and(a, b), np.zeros_like(a))  # disjoint
    assert np.allclose(mask_not(a), 1.0 - a)
    print("predicates selftest passed: masks, phi values, p_phi, algebra")


if __name__ == "__main__":
    _selftest()
