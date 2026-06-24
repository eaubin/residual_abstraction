"""
processes.py — exact data-generating processes with closed-form belief states.

CONTEXT (for readers without the originating conversation; see README.md):
This experiment treats a transformer's residual stream as a candidate
*abstract domain* whose concrete semantics is the probability measure over
completions. To measure abstraction quality exactly (not by sampling), the
data must come from a process where, for every prefix, we can compute in
closed form:

  (1) the *belief state* b_t = P(hidden state | tokens up to t) — the minimal
      sufficient statistic for the entire future (computational mechanics:
      the mixed-state presentation). This is the ideal abstraction; Shai et
      al. 2024 showed transformers linearly embed it in the residual stream.
  (2) the exact distribution over the next m tokens given b_t — our finite-
      horizon stand-in for the "space of completions". KL against it is the
      real-valued completeness measure that replaces the classical
      alpha∘F = F#∘alpha condition of abstract interpretation.

Representation: a hidden Markov process is given by token-labeled matrices
T[s] (s in vocabulary), where T[s][i, j] = P(emit s AND move i->j | state i).
Sum_s T[s] is the row-stochastic state transition matrix. Belief update on
observing s is the Bayes filter  b' ∝ b @ T[s]  with normalizer P(s | b).
"""

from itertools import product

import numpy as np


class HMMProcess:
    def __init__(self, name: str, T):
        T = np.asarray(T, dtype=np.float64)
        assert T.ndim == 3 and T.shape[1] == T.shape[2], "T must be (V, S, S)"
        M = T.sum(axis=0)
        assert np.allclose(M.sum(axis=1), 1.0), "sum_s T[s] must be row-stochastic"
        self.name, self.T = name, T
        self.V, self.S = T.shape[0], T.shape[1]
        # Stationary distribution = left fixed point of the marginal chain M.
        # Used as the prior belief b_0: prefixes are drawn from the stationary
        # process, mirroring how Shai et al. (and LM pretraining) sample data.
        w, vecs = np.linalg.eig(M.T)
        pi = np.real(vecs[:, np.argmin(np.abs(w - 1.0))])
        if pi.sum() < 0:          # eig may return the eigenvector negated
            pi = -pi
        pi = np.maximum(pi, 0.0)  # clip numerical dust
        self.pi = pi / pi.sum()

    # ----- sampling ---------------------------------------------------------
    def sample(self, n_seqs: int, length: int, rng: np.random.Generator,
               init_state=None):
        """Sample token sequences from the stationary process. `init_state`
        (exp 15) fixes the initial hidden state instead of drawing it from
        the stationary distribution — a registered distribution shift; the
        default path is unchanged (and draws from rng identically)."""
        X = np.zeros((n_seqs, length), dtype=np.int64)
        for i in range(n_seqs):
            state = (rng.choice(self.S, p=self.pi) if init_state is None
                     else init_state)
            for t in range(length):
                # joint over (symbol, next state) given current state
                probs = self.T[:, state, :].reshape(-1)
                k = rng.choice(self.V * self.S, p=probs)
                X[i, t], state = divmod(k, self.S)
        return X

    # ----- exact inference (the ground-truth abstraction) -------------------
    def belief_update(self, b, s):
        """Bayes filter step. Returns (posterior, P(s|b))."""
        u = b @ self.T[s]
        z = u.sum()
        return u / z, z

    def beliefs_along(self, seq):
        """beliefs[t] = P(hidden | seq[:t+1]) starting from the stationary prior.

        Pairing convention with the transformer: the residual at position t has
        attended tokens 0..t and predicts token t+1, so it is paired with
        beliefs[t] — the exact sufficient statistic for everything after t.
        """
        b, out = self.pi.copy(), np.zeros((len(seq), self.S))
        for t, s in enumerate(seq):
            b, _ = self.belief_update(b, s)
            out[t] = b
        return out

    def mgram_dist(self, b, m: int):
        """Exact distribution over the next m tokens given belief b.

        This is the truncated 'completion measure' — the concrete semantics
        that the residual-stream abstraction is judged against. V**m outcomes,
        so keep m small (m=3 -> 8 outcomes for Z1R, 27 for Mess3).
        P(s_1..s_m | b) = (b @ T[s_1] @ ... @ T[s_m]) . 1
        """
        out = np.empty(self.V ** m)
        for i, seq in enumerate(product(range(self.V), repeat=m)):
            v = b
            for s in seq:
                v = v @ self.T[s]
            out[i] = v.sum()
        return out

    def mgram_table(self, beliefs, m: int):
        """Vectorized-ish mgram_dist over an array of beliefs (N, S)."""
        return np.stack([self.mgram_dist(b, m) for b in beliefs])


# ----- the two processes -----------------------------------------------------

def z1r() -> HMMProcess:
    """Zero-One-Random: emit 0, then 1, then a fair coin, repeat.

    3 causal states (S0 -> S1 -> SR -> S0), vocabulary {0,1}. The belief set
    reachable after synchronization is small and discrete — ideal for
    debugging the pipeline before Mess3's fractal.
    """
    T0 = [[0.0, 1.0, 0.0],   # S0 emits 0, -> S1
          [0.0, 0.0, 0.0],
          [0.5, 0.0, 0.0]]   # SR emits 0 w.p. 1/2, -> S0
    T1 = [[0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0],   # S1 emits 1, -> SR
          [0.5, 0.0, 0.0]]   # SR emits 1 w.p. 1/2, -> S0
    return HMMProcess("z1r", [T0, T1])


def mess3(x: float = 0.15, a: float = 0.6) -> HMMProcess:
    """Mess3 (Marzen & Crutchfield): 3 states, 3 symbols.

    Its Bayes-filter dynamics map the 2-simplex into itself by three affine
    contractions — an iterated function system — so the reachable beliefs form
    a fractal. Shai et al. 2024 found this fractal linearly embedded in the
    residual stream of a trained transformer; reproducing that here is the
    calibration that the sufficiency machinery measures the right thing.
    Defaults follow the `simplexity` codebase.
    """
    b, y = (1 - a) / 2, 1 - 2 * x
    ay, ax, by, bx = a * y, a * x, b * y, b * x
    T = [
        [[ay, bx, bx], [ax, by, bx], [ax, bx, by]],
        [[by, ax, bx], [bx, ay, bx], [bx, ax, by]],
        [[by, bx, ax], [bx, by, ax], [bx, bx, ay]],
    ]
    return HMMProcess("mess3", T)


def dyck2(depth: int = 3, p_open: float = 0.4,
          open_split=(0.6, 0.4)) -> HMMProcess:
    """Depth-bounded Dyck-2 (Experiment 7): two bracket types, vocabulary
    ( [ ) ] = tokens 0 1 2 3, hidden state = the stack (a tuple of bracket
    types, length <= depth). Depth 3 gives 1+2+4+8 = 15 states.

    Why this is exactly an HMMProcess: bounding the depth makes the stack a
    finite-state machine; every transition emits exactly one symbol, so the
    token-labeled T[s] matrices are 0/1-sparse rows scaled by the policy
    below. This is the cheapest entry into the roadmap's "richer processes"
    step — stack structure and longer-range constraints with zero new
    inference machinery (the registered swap of Dyck ahead of PCFGs is
    pragmatic, not principled).

    Generation policy (registered in experiments/7-dyck.md): at depth 0,
    open (forced), type ( with prob open_split[0]; at interior depths, open
    with prob p_open (type by open_split), else emit the closer matching
    the stack top; at full depth, close (forced). The depth process is a
    birth-death chain; it changes parity every step, so the chain is
    periodic — harmless (the stationary distribution is still the unique
    left fixed point) but it enriches the belief set with parity structure.
    """
    stacks = [()]
    for d in range(1, depth + 1):
        stacks += list(product((0, 1), repeat=d))
    idx = {s: i for i, s in enumerate(stacks)}
    S, V = len(stacks), 4
    T = np.zeros((V, S, S))
    for s, i in idx.items():
        d = len(s)
        if d < depth:
            po = p_open if d > 0 else 1.0
            for b, w in enumerate(open_split):
                T[b, i, idx[s + (b,)]] = po * w        # emit open bracket b
        if d > 0:
            pc = (1.0 - p_open) if d < depth else 1.0
            T[2 + s[-1], i, idx[s[:-1]]] = pc          # emit matching closer
    proc = HMMProcess("dyck2", T)
    proc.states = stacks                 # hidden-state labels (stack tuples)
    proc.state_index = idx               # for ctx-readers (predicates.py)
    return proc


def colored_dyck2(depth: int = 3, p_open: float = 0.4,
                  type_split=(0.6, 0.4), color_split=(0.5, 0.5)) -> HMMProcess:
    """Colored Dyck-2: bracket TYPE and COLOR, both matched on close.

    Adds a color factor (palette size 2) on top of dyck2's depth/type, so the
    residual must maintain BOTH facets to predict closes. This is the second
    binding factor dyck2 lacks; it makes composition of binding abstractions
    (type-binding AND color-binding) studiable, and it is the first-claim toy
    chosen in docs/COMPLETION_PREDICATES.md.

    Vocab (V=8): opens 0..3, closes 4..7, each encoding (type, color) as
    2*type + color; a close token c has type=(c-4)//2, color=(c-4)%2. Hidden
    state: stack of (type, color) frames, length <= depth (1+4+16+64 = 85
    states at depth 3).

    Color matching is FORCED at the token level (only the matching closer is
    emitted), so on the true model "next close matches the top" is ~certain;
    its instrument value is that a color-/type-losing residual ABSTRACTION
    fails to reproduce it (see predicates.tmpl_next_close_matches). The ctx
    (top frame) is fully determined by the visible tokens -> deterministic
    ctx-reader, exact E_q[phi] (contrast a latent-mode toy, deferred).
    """
    frames = list(product((0, 1), (0, 1)))      # (type, color)
    stacks = [()]
    for d in range(1, depth + 1):
        stacks += list(product(frames, repeat=d))
    idx = {s: i for i, s in enumerate(stacks)}
    S, V = len(stacks), 8
    T = np.zeros((V, S, S))
    for s, i in idx.items():
        d = len(s)
        if d < depth:
            po = p_open if d > 0 else 1.0
            for ty, wt in enumerate(type_split):
                for co, wc in enumerate(color_split):
                    T[2 * ty + co, i, idx[s + ((ty, co),)]] = po * wt * wc
        if d > 0:
            pc = (1.0 - p_open) if d < depth else 1.0
            ty, co = s[-1]
            T[4 + 2 * ty + co, i, idx[s[:-1]]] = pc     # matching (type,color)
    proc = HMMProcess("colored_dyck2", T)
    proc.states = stacks
    proc.state_index = idx
    return proc


def pstack(depth: int = 3, p_open: float = 0.38, p_term: float = 0.22,
           p_switch: float = 0.08) -> HMMProcess:
    """Bounded probabilistic stack grammar for oracle-withdrawal rehearsal.

    This is deliberately between Dyck-2 and a full PCFG. Like Dyck, the
    bounded stack makes exact completion measures cheap through the existing
    HMMProcess machinery. Unlike Dyck, a hidden mode biases both bracket
    openings and neutral terminal emissions, so surface prefixes can leave
    ambiguity about the future even when the visible stack depth is known.

    Vocabulary: open0, open1, close0, close1, term0, term1 = 0..5.
    Hidden state: (mode in {0,1}, stack of open bracket types, depth <= 3).
    Neutral terminals leave the stack unchanged but can switch the hidden
    mode, introducing non-stack ambiguity while preserving exact inference.
    """
    stacks = [()]
    for d in range(1, depth + 1):
        stacks += list(product((0, 1), repeat=d))
    states = [(mode, stack) for mode in (0, 1) for stack in stacks]
    idx = {state: i for i, state in enumerate(states)}
    S, V = len(states), 6
    T = np.zeros((V, S, S))

    open_probs = {
        0: (0.78, 0.22),
        1: (0.22, 0.78),
    }
    term_probs = {
        0: (0.72, 0.28),
        1: (0.28, 0.72),
    }

    for (mode, stack), i in idx.items():
        d = len(stack)
        if d == 0:
            po, pt, pc = 0.70, 0.30, 0.0
        elif d == depth:
            po, pt, pc = 0.0, 0.38, 0.62
        else:
            po, pt = p_open, p_term
            pc = 1.0 - po - pt
        assert po >= 0.0 and pt >= 0.0 and pc >= 0.0

        if po:
            for b, w in enumerate(open_probs[mode]):
                T[b, i, idx[(mode, stack + (b,))]] += po * w
        if pc:
            T[2 + stack[-1], i, idx[(mode, stack[:-1])]] += pc
        if pt:
            for tok, w in enumerate(term_probs[mode], start=4):
                T[tok, i, idx[(mode, stack)]] += pt * w * (1.0 - p_switch)
                T[tok, i, idx[(1 - mode, stack)]] += pt * w * p_switch

    return HMMProcess("pstack", T)


PRODUCT_COUNTER_TOKENS = (
    "A_PLUS", "A_MINUS", "B_PLUS", "B_MINUS", "C0", "C1", "NOISE",
)

PRODUCT_COUNTER_CONSTANTS = {
    "W_a": 1.0,
    "W_b": 1.0,
    "W_c": 1.2,
    "W_n": 0.8,
    "u_a": (-0.6, -0.2, 0.2, 0.6),
    "u_b": (-0.6, 0.2, -0.2, 0.6),
    "u_c": (-0.75, 0.75),
}


def product_counter() -> HMMProcess:
    """Finite-state product counter for planted-carrier substrate gates.

    Hidden state is ``(a, b, c)`` with ``a,b in {0,1,2,3}`` and
    ``c in {0,1}``. Tokens update exactly one coordinate: A/B tokens move the
    corresponding counter modulo 4, C tokens set the binary coordinate, and
    NOISE leaves the state unchanged.

    The emission policy is a constant-group-mass softmax. For example,
    ``A_PLUS`` and ``A_MINUS`` share total mass ``W_a / Z`` and encode ``a`` as
    ``P(A_PLUS)-P(A_MINUS) = (W_a / Z) * u_a[a]``. Holding group masses
    constant makes one-step off-target observable leakage analytically zero,
    which is the point of this substrate gate.
    """
    states = list(product(range(4), range(4), range(2)))
    idx = {state: i for i, state in enumerate(states)}
    S, V = len(states), len(PRODUCT_COUNTER_TOKENS)
    T = np.zeros((V, S, S), dtype=np.float64)

    C = PRODUCT_COUNTER_CONSTANTS
    Wa, Wb, Wc, Wn = C["W_a"], C["W_b"], C["W_c"], C["W_n"]
    Z = Wa + Wb + Wc + Wn
    ua, ub, uc = C["u_a"], C["u_b"], C["u_c"]

    for (a, b, c), i in idx.items():
        probs = {
            "A_PLUS": Wa * (1.0 + ua[a]) / (2.0 * Z),
            "A_MINUS": Wa * (1.0 - ua[a]) / (2.0 * Z),
            "B_PLUS": Wb * (1.0 + ub[b]) / (2.0 * Z),
            "B_MINUS": Wb * (1.0 - ub[b]) / (2.0 * Z),
            "C0": Wc * (1.0 - uc[c]) / (2.0 * Z),
            "C1": Wc * (1.0 + uc[c]) / (2.0 * Z),
            "NOISE": Wn / Z,
        }
        next_state = {
            "A_PLUS": ((a + 1) % 4, b, c),
            "A_MINUS": ((a - 1) % 4, b, c),
            "B_PLUS": (a, (b + 1) % 4, c),
            "B_MINUS": (a, (b - 1) % 4, c),
            "C0": (a, b, 0),
            "C1": (a, b, 1),
            "NOISE": (a, b, c),
        }
        for tok, prob in probs.items():
            T[PRODUCT_COUNTER_TOKENS.index(tok), i, idx[next_state[tok]]] = prob

    proc = HMMProcess("product_counter", T)
    proc.states = states
    proc.state_index = idx
    proc.token_names = PRODUCT_COUNTER_TOKENS
    proc.constants = PRODUCT_COUNTER_CONSTANTS
    return proc


PROCESSES = {
    "z1r": z1r,
    "mess3": mess3,
    "dyck2": dyck2,
    "colored_dyck2": colored_dyck2,
    "pstack": pstack,
    "product_counter": product_counter,
}


# ----- self-test --------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name, ctor in PROCESSES.items():
        p = ctor()
        X = p.sample(200, 40, rng)
        # 1) m-gram distributions are probability vectors
        B = p.beliefs_along(X[0])
        G = p.mgram_table(B, 3)
        assert np.allclose(G.sum(axis=1), 1.0), name
        # 2) the belief state actually predicts next tokens: average exact
        #    next-token NLL under the filter should beat the unigram baseline.
        nll, base = 0.0, 0.0
        uni = np.array([p.mgram_dist(p.pi, 1)]).ravel()
        n = 0
        for row in X:
            b = p.pi.copy()
            for s in row:
                pred = np.array([(b @ p.T[k]).sum() for k in range(p.V)])
                nll -= np.log(pred[s]); base -= np.log(uni[s]); n += 1
                b, _ = p.belief_update(b, s)
        print(f"{name}: filter NLL/tok {nll/n:.4f}  vs unigram {base/n:.4f}")
        assert nll <= base + 1e-9
    print("processes.py self-test OK")
