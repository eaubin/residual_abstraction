"""
Experiment 41 product-counter substrate gate.

This script is a finite-state substrate/calibration gate, not an abstraction or
transformer experiment. It checks that the registered product-counter process has
separable, behaviorally visible latent variables and that oracle / planted mixed
carriers preserve exact completion information under registered decoders.
"""

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processes import PRODUCT_COUNTER_CONSTANTS, PRODUCT_COUNTER_TOKENS, PROCESSES

M_DEFAULT = 3
D_HIDDEN_DEFAULT = 64
KAPPA_DEFAULT = 100.0

MEAN_OWN_MIN = {"a": 0.10, "b": 0.10, "c": 0.30}
P10_OWN_MIN = {"a": 0.05, "b": 0.05, "c": 0.30}
C_MEAN_OWN_MIN = 0.30
OFFTARGET_ZERO_MAX = 1e-12
MGRAM_RUNTIME_MAX_SEC = 10.0
ORACLE_JS_MAX = 1e-12
MIXED_JS_MAX = 1e-10
RECON_MAX = 1e-10
COND_REL_MAX = 1e-10

EXPECTED_ORDERED_PAIRS = {"a": 96, "b": 96, "c": 32}
EXPECTED_VALUE_CELLS = {"a": 12, "b": 12, "c": 2}

ROUTE_PRECEDENCE = (
    "HARNESS_FAIL",
    "NOT_DISSOCIABLE",
    "LOW_TARGET_ROOM",
    "CONTROL_LOW_ROOM",
    "LEAKAGE_FAIL",
    "TOO_EXPENSIVE",
    "CARRIER_FAITHFULNESS_FAIL",
)


def kl_rows(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, 1e-300, None)
    return np.where(p > 0.0, p * (np.log(p) - np.log(q)), 0.0).sum(axis=1)


def js_rows(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * (kl_rows(p, m) + kl_rows(q, m))


def next_state(state, token):
    a, b, c = state
    if token == "A_PLUS":
        return ((a + 1) % 4, b, c)
    if token == "A_MINUS":
        return ((a - 1) % 4, b, c)
    if token == "B_PLUS":
        return (a, (b + 1) % 4, c)
    if token == "B_MINUS":
        return (a, (b - 1) % 4, c)
    if token == "C0":
        return (a, b, 0)
    if token == "C1":
        return (a, b, 1)
    if token == "NOISE":
        return state
    raise ValueError(token)


def one_step_probs(proc, belief):
    return np.array([(belief @ proc.T[tok]).sum() for tok in range(proc.V)])


def observables(proc, belief):
    p = one_step_probs(proc, belief)
    tok = {name: i for i, name in enumerate(PRODUCT_COUNTER_TOKENS)}
    return {
        "a": float(p[tok["A_PLUS"]] - p[tok["A_MINUS"]]),
        "b": float(p[tok["B_PLUS"]] - p[tok["B_MINUS"]]),
        "c": float(p[tok["C1"]] - p[tok["C0"]]),
    }


def analytic_observables(state):
    a, b, c = state
    C = PRODUCT_COUNTER_CONSTANTS
    Z = C["W_a"] + C["W_b"] + C["W_c"] + C["W_n"]
    return {
        "a": (C["W_a"] / Z) * C["u_a"][a],
        "b": (C["W_b"] / Z) * C["u_b"][b],
        "c": (C["W_c"] / Z) * C["u_c"][c],
    }


def contrast_pairs(states, target):
    rows = []
    for clean in states:
        for source in states:
            if clean == source:
                continue
            if target == "a" and clean[1:] == source[1:] and clean[0] != source[0]:
                rows.append((clean, source))
            elif target == "b" and clean[0] == source[0] and clean[2] == source[2] and clean[1] != source[1]:
                rows.append((clean, source))
            elif target == "c" and clean[:2] == source[:2] and clean[2] != source[2]:
                rows.append((clean, source))
    return rows


def summarize_contrasts(proc):
    states = proc.states
    eye = np.eye(proc.S)
    state_obs = {
        state: observables(proc, eye[proc.state_index[state]])
        for state in states
    }
    summaries = {}
    leakage = {}
    for target in ("a", "b", "c"):
        rows = contrast_pairs(states, target)
        deltas = {key: [] for key in ("a", "b", "c")}
        cells = {}
        for clean, source in rows:
            for key in deltas:
                deltas[key].append(abs(state_obs[source][key] - state_obs[clean][key]))
            if target == "a":
                cell = (clean[0], source[0])
            elif target == "b":
                cell = (clean[1], source[1])
            else:
                cell = (clean[2], source[2])
            cells[cell] = cells.get(cell, 0) + 1

        own = np.asarray(deltas[target], dtype=np.float64)
        summaries[target] = {
            "ordered_pairs": len(rows),
            "unordered_pairs": len(rows) // 2,
            "value_cells": len(cells),
            "min_cell_count": min(cells.values()),
            "mean_own": float(own.mean()),
            "min_own": float(own.min()),
            "p10_own": float(np.quantile(own, 0.10, method="nearest")),
            "p50_own": float(np.quantile(own, 0.50, method="nearest")),
            "p90_own": float(np.quantile(own, 0.90, method="nearest")),
        }
        leakage[target] = {
            key: float(np.mean(vals))
            for key, vals in deltas.items()
        }
    return summaries, leakage


def planted_T(d_hidden, n_states, kappa, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d_hidden, n_states))
    u, _ = np.linalg.qr(a, mode="reduced")
    b = rng.standard_normal((n_states, n_states))
    v, _ = np.linalg.qr(b)
    sig = np.geomspace(1.0, 1.0 / float(kappa), n_states)
    return u @ np.diag(sig) @ v.T


def carrier_agreement(proc, exact, carrier, seed, kappa, d_hidden):
    if carrier == "oracle":
        decoded = list(range(proc.S))
        recon_error = 0.0
        cond = 1.0
        rank = proc.S
        cond_rel_error = 0.0
        min_column_sep = float(np.sqrt(2.0))
    elif carrier == "mixed":
        T = planted_T(d_hidden, proc.S, kappa, seed)
        pinv = np.linalg.pinv(T)
        H = T
        Y = pinv @ H
        decoded = list(np.argmax(Y, axis=0))
        recon_error = float(np.max(np.abs(Y - np.eye(proc.S))))
        cond = float(np.linalg.cond(T))
        rank = int(np.linalg.matrix_rank(T))
        cond_rel_error = abs(cond - float(kappa)) / float(kappa)
        cols = H.T
        dists = [
            np.linalg.norm(cols[i] - cols[j])
            for i in range(proc.S) for j in range(i + 1, proc.S)
        ]
        min_column_sep = float(np.min(dists))
    else:
        raise ValueError(carrier)

    implied = np.stack([exact[j] for j in decoded])
    js = js_rows(exact, implied)
    return {
        "decode_accuracy": int(sum(i == j for i, j in enumerate(decoded))),
        "mean_js": float(js.mean()),
        "max_js": float(js.max()),
        "recon_error": recon_error,
        "rank": rank,
        "condition_number": cond,
        "condition_rel_error": cond_rel_error,
        "min_column_sep": min_column_sep,
    }


def route_for(failures):
    if not failures:
        return "READY_FOR_PLANTED_INTERVENTIONS"
    labels = {label for label, _message in failures}
    for label in ROUTE_PRECEDENCE:
        if label in labels:
            return label
    return sorted(labels)[0]


def print_final(route, failures, confirm=False):
    print(f"ROUTE: {route}")
    if failures:
        print("gate failures:")
        for label, failure in failures:
            print(f"  - {label}: {failure}")
        print("NO-GO: product-counter instance is not ready for planted-carrier intervention experiments.")
        return 1
    if confirm:
        print("GO: product-counter regular-process oracle/mixed-carrier cell is ready for planted-carrier intervention experiments.")
    else:
        print("GO: development panel passed its registered gates.")
    return 0


def selftest(proc):
    assert proc.T.shape == (7, 32, 32)
    M = proc.T.sum(axis=0)
    assert np.allclose(M.sum(axis=1), 1.0)

    for state in proc.states:
        i = proc.state_index[state]
        for tok, name in enumerate(PRODUCT_COUNTER_TOKENS):
            nz = np.flatnonzero(proc.T[tok, i] > 0.0)
            assert len(nz) == 1, (state, name, nz)
            assert proc.states[int(nz[0])] == next_state(state, name)

    assert np.isclose(proc.pi.sum(), 1.0)
    assert np.min(proc.pi) > 1e-12

    eye = np.eye(proc.S)
    for m in (1, 2, 3):
        table = proc.mgram_table(eye, m)
        assert np.allclose(table.sum(axis=1), 1.0), m

    max_obs_err = 0.0
    for state in proc.states:
        b = eye[proc.state_index[state]]
        actual = observables(proc, b)
        expected = analytic_observables(state)
        for key in actual:
            max_obs_err = max(max_obs_err, abs(actual[key] - expected[key]))
    assert max_obs_err <= 1e-12

    print("selftest: PASS")
    print(f"  transition tensor shape: {proc.T.shape}")
    print(f"  stationary min mass: {np.min(proc.pi):.12g}")
    print(f"  max analytic observable error: {max_obs_err:.3e}")


def print_constants(args):
    C = PRODUCT_COUNTER_CONSTANTS
    print("registered constants:")
    print(f"  S=32 |V|=7 m={args.m} d_hidden={args.d_hidden}")
    print(f"  seed={args.seed} kappa={args.kappa:g}")
    print(f"  tokens={','.join(PRODUCT_COUNTER_TOKENS)}")
    print(f"  W_a={C['W_a']} W_b={C['W_b']} W_c={C['W_c']} W_n={C['W_n']}")
    print(f"  u_a={C['u_a']}")
    print(f"  u_b={C['u_b']}")
    print(f"  u_c={C['u_c']}")
    print()


def evaluate(args):
    proc = PROCESSES["product_counter"]()
    if args.selftest:
        failures = []
        try:
            selftest(proc)
        except AssertionError as exc:
            failures.append(("HARNESS_FAIL", f"selftest assertion failed: {exc}"))
        return print_final(route_for(failures), failures)
    if args.confirm:
        return confirm(args, proc)

    print_constants(args)
    failures = run_panel(proc, args, args.carrier)
    return print_final(route_for(failures), failures)


def run_panel(proc, args, carrier):
    print(f"panel: {carrier}")
    eye = np.eye(proc.S)

    summaries, leakage = summarize_contrasts(proc)
    print("pair counts and own-room:")
    for target, row in summaries.items():
        print(
            f"  {target}: ordered={row['ordered_pairs']} "
            f"unordered={row['unordered_pairs']} cells={row['value_cells']} "
            f"min_cell={row['min_cell_count']} mean={row['mean_own']:.12g} "
            f"min={row['min_own']:.12g} p10={row['p10_own']:.12g} "
            f"p50={row['p50_own']:.12g} p90={row['p90_own']:.12g}"
        )
    print()

    print("leakage / dominance matrix (mean absolute observable movement):")
    for target in ("a", "b", "c"):
        row = leakage[target]
        off = max(value for key, value in row.items() if key != target)
        dominance = "inf" if off == 0.0 else f"{row[target] / off:.12g}"
        print(
            f"  change {target}: obs_a={row['a']:.12g} "
            f"obs_b={row['b']:.12g} obs_c={row['c']:.12g} "
            f"own/off={dominance}"
        )
    print()
    print("synthetic coupled-reference baseline (descriptive):")
    print("  equal off-target coupling would have off/own=1 and own/off=1")
    print("  product-counter registered instance requires off-target <= 1e-12")
    print()

    t0 = time.time()
    exact = proc.mgram_table(eye, args.m)
    runtime = time.time() - t0
    norm_err = float(np.max(np.abs(exact.sum(axis=1) - 1.0)))
    print("exact m-gram:")
    print(f"  rows={proc.S} outcomes={proc.V ** args.m} runtime_sec={runtime:.6f}")
    print(f"  max_norm_error={norm_err:.3e}")
    print()

    carrier_stats = carrier_agreement(proc, exact, carrier, args.seed,
                                      args.kappa, args.d_hidden)
    print(f"carrier agreement ({carrier}):")
    for key, value in carrier_stats.items():
        if key == "decode_accuracy":
            print(f"  {key}={value}/{proc.S}")
        elif isinstance(value, int):
            print(f"  {key}={value}")
        else:
            print(f"  {key}={value:.12g}")
    print()

    failures = []
    for target, row in summaries.items():
        if row["mean_own"] < MEAN_OWN_MIN[target]:
            label = "CONTROL_LOW_ROOM" if target == "c" else "LOW_TARGET_ROOM"
            failures.append((label, f"{target}: mean own delta below gate"))
        if row["p10_own"] < P10_OWN_MIN[target]:
            label = "CONTROL_LOW_ROOM" if target == "c" else "LOW_TARGET_ROOM"
            failures.append((label, f"{target}: p10 own delta below gate"))
        if row["ordered_pairs"] != EXPECTED_ORDERED_PAIRS[target]:
            failures.append(("NOT_DISSOCIABLE", f"{target}: ordered pair count mismatch"))
        if row["value_cells"] != EXPECTED_VALUE_CELLS[target]:
            failures.append(("NOT_DISSOCIABLE", f"{target}: value cell count mismatch"))
        if row["min_cell_count"] < 1:
            failures.append(("NOT_DISSOCIABLE", f"{target}: missing or empty value cell"))

    if summaries["c"]["mean_own"] < C_MEAN_OWN_MIN:
        failures.append(("CONTROL_LOW_ROOM", "c: high-room control gate failed"))

    for target, row in leakage.items():
        off = max(value for key, value in row.items() if key != target)
        if off > OFFTARGET_ZERO_MAX:
            failures.append(("LEAKAGE_FAIL", f"{target}: analytic off-target zero gate failed"))

    if runtime > MGRAM_RUNTIME_MAX_SEC:
        failures.append(("TOO_EXPENSIVE", "m-gram runtime gate failed"))
    if norm_err > 1e-12:
        failures.append(("HARNESS_FAIL", "m-gram normalization gate failed"))
    if carrier_stats["decode_accuracy"] != proc.S:
        failures.append(("CARRIER_FAITHFULNESS_FAIL", f"{carrier}: state decode failed"))
    if carrier == "oracle" and carrier_stats["mean_js"] > ORACLE_JS_MAX:
        failures.append(("CARRIER_FAITHFULNESS_FAIL", "oracle: mean JS gate failed"))
    if carrier == "mixed" and carrier_stats["mean_js"] > MIXED_JS_MAX:
        failures.append(("CARRIER_FAITHFULNESS_FAIL", "mixed: mean JS gate failed"))
    if carrier == "mixed" and carrier_stats["recon_error"] > RECON_MAX:
        failures.append(("CARRIER_FAITHFULNESS_FAIL", "mixed: pseudoinverse reconstruction gate failed"))
    if carrier == "mixed" and carrier_stats["rank"] != proc.S:
        failures.append(("CARRIER_FAITHFULNESS_FAIL", "mixed: rank gate failed"))
    if carrier == "mixed" and carrier_stats["condition_rel_error"] > COND_REL_MAX:
        failures.append(("CARRIER_FAITHFULNESS_FAIL", "mixed: condition-number gate failed"))

    return failures


def confirm(args, proc):
    failures = []
    print("confirmatory aggregate: selftest + oracle + mixed")
    try:
        selftest(proc)
    except AssertionError as exc:
        failures.append(("HARNESS_FAIL", f"selftest assertion failed: {exc}"))
    print()

    oracle_args = argparse.Namespace(**vars(args))
    oracle_args.carrier = "oracle"
    failures.extend(run_panel(proc, oracle_args, "oracle"))

    mixed_args = argparse.Namespace(**vars(args))
    mixed_args.carrier = "mixed"
    failures.extend(run_panel(proc, mixed_args, "mixed"))

    return print_final(route_for(failures), failures, confirm=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--carrier", choices=("oracle", "mixed"), default="oracle")
    parser.add_argument("--m", type=int, default=M_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kappa", type=float, default=KAPPA_DEFAULT)
    parser.add_argument("--d-hidden", type=int, default=D_HIDDEN_DEFAULT)
    args = parser.parse_args()
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
