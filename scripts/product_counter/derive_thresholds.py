"""
Analytic margins for the quarantined product-counter pilot.

This helper derives one-step observable identities and exhaustive contrast
margins from the product-counter constants. Exp 41 is a procedural failure; these
numbers are pilot inputs only.
"""

from itertools import product

import sympy as sp


TOKENS = ("A_PLUS", "A_MINUS", "B_PLUS", "B_MINUS", "C0", "C1", "NOISE")

Wa = sp.Rational(1, 1)
Wb = sp.Rational(1, 1)
Wc = sp.Rational(6, 5)
Wn = sp.Rational(4, 5)
Z = Wa + Wb + Wc + Wn

u_a = tuple(map(sp.Rational, (-sp.Rational(3, 5), -sp.Rational(1, 5),
                              sp.Rational(1, 5), sp.Rational(3, 5))))
u_b = tuple(map(sp.Rational, (-sp.Rational(3, 5), sp.Rational(1, 5),
                              -sp.Rational(1, 5), sp.Rational(3, 5))))
u_c = (-sp.Rational(3, 4), sp.Rational(3, 4))


def probs(a, b, c):
    return {
        "A_PLUS": Wa * (1 + u_a[a]) / (2 * Z),
        "A_MINUS": Wa * (1 - u_a[a]) / (2 * Z),
        "B_PLUS": Wb * (1 + u_b[b]) / (2 * Z),
        "B_MINUS": Wb * (1 - u_b[b]) / (2 * Z),
        "C0": Wc * (1 - u_c[c]) / (2 * Z),
        "C1": Wc * (1 + u_c[c]) / (2 * Z),
        "NOISE": Wn / Z,
    }


def obs(state):
    p = probs(*state)
    return {
        "a": sp.simplify(p["A_PLUS"] - p["A_MINUS"]),
        "b": sp.simplify(p["B_PLUS"] - p["B_MINUS"]),
        "c": sp.simplify(p["C1"] - p["C0"]),
    }


def quantile_nearest(xs, q):
    items = sorted(xs)
    idx = int(sp.floor(sp.Rational(q) * (len(items) - 1) + sp.Rational(1, 2)))
    return items[idx]


def contrast_rows(target):
    states = list(product(range(4), range(4), range(2)))
    rows = []
    for clean in states:
        for source in states:
            if clean == source:
                continue
            if target == "a" and clean[1:] == source[1:] and clean[0] != source[0]:
                rows.append((clean, source))
            if target == "b" and clean[0] == source[0] and clean[2] == source[2] and clean[1] != source[1]:
                rows.append((clean, source))
            if target == "c" and clean[:2] == source[:2] and clean[2] != source[2]:
                rows.append((clean, source))
    return rows


def summarize(target):
    rows = contrast_rows(target)
    deltas = {k: [] for k in ("a", "b", "c")}
    value_cells = {}
    for clean, source in rows:
        oc, os = obs(clean), obs(source)
        for key in deltas:
            deltas[key].append(abs(sp.simplify(os[key] - oc[key])))
        if target == "a":
            cell = (clean[0], source[0])
        elif target == "b":
            cell = (clean[1], source[1])
        else:
            cell = (clean[2], source[2])
        value_cells[cell] = value_cells.get(cell, 0) + 1
    own = deltas[target]
    return {
        "ordered_pairs": len(rows),
        "value_cells": len(value_cells),
        "min_cell_count": min(value_cells.values()),
        "mean_own": sp.simplify(sum(own) / len(own)),
        "min_own": min(own),
        "p10_own": quantile_nearest(own, sp.Rational(1, 10)),
        "p50_own": quantile_nearest(own, sp.Rational(1, 2)),
        "p90_own": quantile_nearest(own, sp.Rational(9, 10)),
        "mean_abs": {
            key: sp.simplify(sum(vals) / len(vals))
            for key, vals in deltas.items()
        },
    }


def fmt(x):
    return f"{x}:{float(x):.12g}"


def main():
    print("DERIVE artifact=quarantined_pilot")
    print(f"CONSTANTS W_a={Wa} W_b={Wb} W_c={Wc} W_n={Wn} Z={Z}")
    print(f"CODES variable=a values={u_a}")
    print(f"CODES variable=b values={u_b}")
    print(f"CODES variable=c values={u_c}")
    print("OBS_ID variable=a expr=(W_a/Z)*u_a[a]")
    print("OBS_ID variable=b expr=(W_b/Z)*u_b[b]")
    print("OBS_ID variable=c expr=(W_c/Z)*u_c[c]")

    for state in product(range(4), range(4), range(2)):
        if state[1:] == (0, 0):
            o = obs(state)
            print(f"STATE_OBS variable=a value={state[0]} obs={fmt(o['a'])}")
    for state in ((0, b0, 0) for b0 in range(4)):
        o = obs(state)
        print(f"STATE_OBS variable=b value={state[1]} obs={fmt(o['b'])}")
    for state in ((0, 0, c0) for c0 in range(2)):
        o = obs(state)
        print(f"STATE_OBS variable=c value={state[2]} obs={fmt(o['c'])}")

    summaries = {target: summarize(target) for target in ("a", "b", "c")}
    for target, row in summaries.items():
        print(
            f"CONTRAST target={target} ordered={row['ordered_pairs']} "
            f"cells={row['value_cells']} min_cell={row['min_cell_count']} "
            f"mean_own={fmt(row['mean_own'])} min_own={fmt(row['min_own'])} "
            f"p10_own={fmt(row['p10_own'])} p50_own={fmt(row['p50_own'])} "
            f"p90_own={fmt(row['p90_own'])}"
        )
        for obs_key, val in row["mean_abs"].items():
            print(f"MEAN_ABS target={target} obs={obs_key} value={fmt(val)}")

    print("GATE quantity=mean_own target=a threshold=>=0.10")
    print("GATE quantity=mean_own target=b threshold=>=0.10")
    print("GATE quantity=mean_own target=c threshold=>=0.30")
    print("GATE quantity=p10_own target=a threshold=>=0.05")
    print("GATE quantity=p10_own target=b threshold=>=0.05")
    print("GATE quantity=p10_own target=c threshold=>=0.30")
    print("GATE quantity=offtarget_leakage target=all threshold=<=1e-12")


if __name__ == "__main__":
    main()
