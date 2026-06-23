"""
Quarantined product-counter intervention pilot.

This runner checks the finite product-counter substrate, planted carriers,
contextual exact edits, two noncontextual handle classes, and matched random
floors. Exp 41 is a procedural failure, so this is a pilot harness only.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processes import PRODUCT_COUNTER_CONSTANTS, PRODUCT_COUNTER_TOKENS, PROCESSES

M_DEFAULT = 3
D_HIDDEN_DEFAULT = 64
KAPPA_DEFAULT = 100.0
SEED_DEFAULT = 0

MEAN_OWN_MIN = {"a": 0.10, "b": 0.10, "c": 0.30}
P10_OWN_MIN = {"a": 0.05, "b": 0.05, "c": 0.30}
C_MEAN_OWN_MIN = 0.30
OFFTARGET_ZERO_MAX = 1e-12
MGRAM_RUNTIME_MAX_SEC = 10.0
ORACLE_JS_MAX = 1e-12
MIXED_JS_MAX = 1e-10
RECON_MAX = 1e-10
COND_REL_MAX = 1e-10
CEILING_TRANSPORT_MIN = 0.99
LINEAR_TRANSPORT_MIN = 0.80
PROJECTED_TRANSPORT_MIN = 0.80
DRAG_MAX = 0.05
SIMPLEX_NEG_MAX = 1e-10
OFFSIMPLEX_NEG_MIN = 0.25
PROJECTED_SOURCE_JS_MAX = 0.02

EXPECTED_ORDERED_PAIRS = {"a": 96, "b": 96, "c": 32}
EXPECTED_VALUE_CELLS = {"a": 12, "b": 12, "c": 2}
CONFIRM_REGISTERED = {
    "m": M_DEFAULT,
    "seed": SEED_DEFAULT,
    "kappa": KAPPA_DEFAULT,
    "d_hidden": D_HIDDEN_DEFAULT,
}

ROUTE_PRECEDENCE = (
    "OUT_OF_SCOPE_CONFIG",
    "HARNESS_FAIL",
    "NOT_DISSOCIABLE",
    "LOW_TARGET_ROOM",
    "CONTROL_LOW_ROOM",
    "LEAKAGE_FAIL",
    "TOO_EXPENSIVE",
    "CARRIER_FAITHFULNESS_FAIL",
    "CEILING_FAIL",
    "FLOOR_FAIL",
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


def state_with_value(state, target, value):
    idx = {"a": 0, "b": 1, "c": 2}[target]
    out = list(state)
    out[idx] = value
    return tuple(out)


def target_values(target):
    return range(2) if target == "c" else range(4)


def project_simplex(v):
    """Euclidean projection onto the probability simplex."""
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    theta = cssv[cond][-1] / ind[cond][-1]
    return np.maximum(v - theta, 0.0)


def obs_vectors(proc):
    eye = np.eye(proc.S)
    rows = {key: [] for key in ("a", "b", "c")}
    for state in proc.states:
        obs = observables(proc, eye[proc.state_index[state]])
        for key in rows:
            rows[key].append(obs[key])
    return {key: np.asarray(vals, dtype=np.float64) for key, vals in rows.items()}


def readout_matrix(pinv, obs_vec):
    return np.stack([pinv.T @ obs_vec[key] for key in ("a", "b", "c")])


def score_edit(proc, exact, obs_vec, target, clean, source, y):
    clean_i = proc.state_index[clean]
    source_i = proc.state_index[source]
    denom = obs_vec[target][source_i] - obs_vec[target][clean_i]
    if abs(denom) < 1e-15:
        raise ValueError((target, clean, source, denom))
    own = (float(obs_vec[target] @ y) - obs_vec[target][clean_i]) / denom
    drag = max(
        abs(float(obs_vec[key] @ y) - obs_vec[key][clean_i]) / abs(denom)
        for key in ("a", "b", "c") if key != target
    )
    neg_l1 = float(np.clip(-y, 0.0, None).sum())
    min_y = float(np.min(y))
    sum_err = float(abs(np.sum(y) - 1.0))
    js_source = np.nan
    js_clean = np.nan
    if min_y >= -SIMPLEX_NEG_MAX and sum_err <= 1e-10:
        y_valid = np.clip(y, 0.0, None)
        y_valid = y_valid / y_valid.sum()
        dist = proc.mgram_table(y_valid[None, :], M_DEFAULT)[0]
        js_source = float(js_rows(exact[source_i][None, :], dist[None, :])[0])
        js_clean = float(js_rows(exact[clean_i][None, :], dist[None, :])[0])
    return {
        "own": float(own),
        "drag": float(drag),
        "min_y": min_y,
        "neg_l1": neg_l1,
        "sum_err": sum_err,
        "js_source": js_source,
        "js_clean": js_clean,
    }


def summarize_scores(rows):
    out = {}
    for key in rows[0]:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            out[f"{key}_min"] = np.nan
            out[f"{key}_mean"] = np.nan
            out[f"{key}_max"] = np.nan
        else:
            out[f"{key}_min"] = float(np.min(finite))
            out[f"{key}_mean"] = float(np.mean(finite))
            out[f"{key}_max"] = float(np.max(finite))
    return out


def intervention_panel(proc, args):
    eye = np.eye(proc.S)
    exact = proc.mgram_table(eye, M_DEFAULT)
    obs_vec = obs_vectors(proc)
    T = planted_T(args.d_hidden, proc.S, args.kappa, args.seed)
    pinv = np.linalg.pinv(T)
    H = T
    R = readout_matrix(pinv, obs_vec)
    RRi = np.linalg.pinv(R @ R.T)
    rows_by_arm = {
        "contextual_exact": {target: [] for target in ("a", "b", "c")},
        "mean_delta_linear": {target: [] for target in ("a", "b", "c")},
        "mean_delta_projected": {target: [] for target in ("a", "b", "c")},
        "obs_minnorm_linear": {target: [] for target in ("a", "b", "c")},
        "obs_minnorm_projected": {target: [] for target in ("a", "b", "c")},
        "random_linear": {target: [] for target in ("a", "b", "c")},
        "random_projected": {target: [] for target in ("a", "b", "c")},
    }

    for target in ("a", "b", "c"):
        target_idx = {"a": 0, "b": 1, "c": 2}[target]
        for clean_value in target_values(target):
            for source_value in target_values(target):
                if clean_value == source_value:
                    continue
                pair_rows = []
                hidden_deltas = []
                for clean in proc.states:
                    idx = {"a": 0, "b": 1, "c": 2}[target]
                    if clean[idx] != clean_value:
                        continue
                    source = state_with_value(clean, target, source_value)
                    pair_rows.append((clean, source))
                    hidden_deltas.append(
                        H[:, proc.state_index[source]] - H[:, proc.state_index[clean]]
                    )
                mean_delta = np.mean(hidden_deltas, axis=0)

                for clean, source in pair_rows:
                    clean_h = H[:, proc.state_index[clean]]
                    source_h = H[:, proc.state_index[source]]

                    full_y = pinv @ source_h
                    rows_by_arm["contextual_exact"][target].append(
                        score_edit(proc, exact, obs_vec, target, clean, source, full_y)
                    )

                    linear_y = pinv @ (clean_h + mean_delta)
                    rows_by_arm["mean_delta_linear"][target].append(
                        score_edit(proc, exact, obs_vec, target, clean, source, linear_y)
                    )

                    projected_y = project_simplex(linear_y)
                    rows_by_arm["mean_delta_projected"][target].append(
                        score_edit(proc, exact, obs_vec, target, clean, source, projected_y)
                    )

                    clean_i = proc.state_index[clean]
                    source_i = proc.state_index[source]
                    desired = np.zeros(3)
                    desired[target_idx] = obs_vec[target][source_i] - obs_vec[target][clean_i]
                    minnorm_delta = R.T @ RRi @ desired
                    minnorm_y = pinv @ (clean_h + minnorm_delta)
                    rows_by_arm["obs_minnorm_linear"][target].append(
                        score_edit(proc, exact, obs_vec, target, clean, source, minnorm_y)
                    )
                    minnorm_projected_y = project_simplex(minnorm_y)
                    rows_by_arm["obs_minnorm_projected"][target].append(
                        score_edit(proc, exact, obs_vec, target, clean, source, minnorm_projected_y)
                    )

                    rng = np.random.default_rng(
                        10000 + 1000 * target_idx + 100 * clean_i + source_i
                    )
                    random_delta = rng.standard_normal(H.shape[0])
                    random_delta *= np.linalg.norm(mean_delta) / np.linalg.norm(random_delta)
                    random_y = pinv @ (clean_h + random_delta)
                    rows_by_arm["random_linear"][target].append(
                        score_edit(proc, exact, obs_vec, target, clean, source, random_y)
                    )
                    random_projected_y = project_simplex(random_y)
                    rows_by_arm["random_projected"][target].append(
                        score_edit(proc, exact, obs_vec, target, clean, source, random_projected_y)
                    )

    summaries = {
        arm: {target: summarize_scores(rows) for target, rows in targets.items()}
        for arm, targets in rows_by_arm.items()
    }

    for arm in (
        "contextual_exact",
        "mean_delta_linear",
        "mean_delta_projected",
        "obs_minnorm_linear",
        "obs_minnorm_projected",
        "random_linear",
        "random_projected",
    ):
        for target in ("a", "b", "c"):
            row = summaries[arm][target]
            print(
                f"ARM arm={arm} target={target} own_min={row['own_min']:.12g} "
                f"own_mean={row['own_mean']:.12g} drag_max={row['drag_max']:.12g} "
                f"neg_l1_max={row['neg_l1_max']:.12g} min_y_min={row['min_y_min']:.12g} "
                f"js_source_mean={row['js_source_mean']:.12g} "
                f"js_clean_mean={row['js_clean_mean']:.12g}"
            )

    failures = []
    ceiling_ok = all(
        summaries["contextual_exact"][target]["own_min"] >= CEILING_TRANSPORT_MIN
        and summaries["contextual_exact"][target]["drag_max"] <= DRAG_MAX
        and summaries["contextual_exact"][target]["neg_l1_max"] <= SIMPLEX_NEG_MAX
        and summaries["contextual_exact"][target]["js_source_max"] <= ORACLE_JS_MAX
        for target in ("a", "b", "c")
    )
    if not ceiling_ok:
        failures.append(("CEILING_FAIL", "full replacement did not recover the exact source state"))
        return None, failures

    floor_bad = any(
        all(
            summaries[arm][target]["own_min"] >= LINEAR_TRANSPORT_MIN
            and summaries[arm][target]["drag_max"] <= DRAG_MAX
            for target in ("a", "b", "c")
        )
        for arm in ("random_linear", "random_projected")
    )
    if floor_bad:
        failures.append(("FLOOR_FAIL", "random matched-norm floor passed the intervention thresholds"))
        return None, failures

    linear_arms = ("mean_delta_linear", "obs_minnorm_linear")
    projected_arms = ("mean_delta_projected", "obs_minnorm_projected")
    linear_ok = {
        arm: all(
            summaries[arm][target]["own_min"] >= LINEAR_TRANSPORT_MIN
            and summaries[arm][target]["drag_max"] <= DRAG_MAX
            for target in ("a", "b", "c")
        )
        for arm in linear_arms
    }
    if not any(linear_ok.values()):
        return "CONTEXTUAL_ONLY", failures

    projected_ok = {
        arm: all(
            summaries[arm][target]["own_min"] >= PROJECTED_TRANSPORT_MIN
            and summaries[arm][target]["drag_max"] <= DRAG_MAX
            and summaries[arm][target]["js_source_mean"] <= PROJECTED_SOURCE_JS_MAX
            for target in ("a", "b", "c")
        )
        for arm in projected_arms
    }
    if any(projected_ok.values()):
        return "COHERENT_NONCONTEXTUAL_HANDLE", failures

    offsimplex = any(
        summaries[arm][target]["neg_l1_max"] >= OFFSIMPLEX_NEG_MIN
        for arm in linear_arms
        for target in ("a", "b", "c")
    )
    if offsimplex:
        return "READOUT_ONLY_NONCONTEXTUAL_HANDLES", failures
    return "PROJECTED_NONCONTEXTUAL_FAIL", failures


def route_for(failures):
    if not failures:
        return "PANEL_PASS"
    labels = {label for label, _message in failures}
    for label in ROUTE_PRECEDENCE:
        if label in labels:
            return label
    return sorted(labels)[0]


def confirm_scope_failures(args):
    failures = []
    for key, expected in CONFIRM_REGISTERED.items():
        got = getattr(args, key)
        if isinstance(expected, float):
            in_scope = np.isclose(got, expected, rtol=0.0, atol=1e-12)
        else:
            in_scope = got == expected
        if not in_scope:
            shown_key = key.replace("_", "-")
            failures.append((
                "OUT_OF_SCOPE_CONFIG",
                f"--{shown_key}={got!r} outside fixed pilot scope; expected {expected!r}",
            ))
    return failures


def print_final(route, failures, confirm=False):
    del confirm
    print(
        f"ROUTE artifact=quarantined_pilot "
        f"gate_status={'FAIL' if failures else 'PASS'} route={route}"
    )
    if failures:
        for label, failure in failures:
            print(f"FAIL label={label} detail={failure}")
        return 1
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
    projected = project_simplex(np.array([0.2, -0.1, 0.9]))
    assert np.all(projected >= 0.0)
    assert np.isclose(projected.sum(), 1.0)
    assert np.allclose(projected, np.array([0.15, 0.0, 0.85]))

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

    print(
        f"SELFTEST status=PASS tensor_shape={proc.T.shape} "
        f"stationary_min={np.min(proc.pi):.12g} max_obs_err={max_obs_err:.3e}"
    )


def print_constants(args):
    C = PRODUCT_COUNTER_CONSTANTS
    print(
        f"CONFIG S=32 V=7 m={args.m} d_hidden={args.d_hidden} "
        f"seed={args.seed} kappa={args.kappa:g} "
        f"tokens={','.join(PRODUCT_COUNTER_TOKENS)} "
        f"W_a={C['W_a']} W_b={C['W_b']} W_c={C['W_c']} W_n={C['W_n']}"
    )


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
    print(f"PANEL carrier={carrier}")
    eye = np.eye(proc.S)

    summaries, leakage = summarize_contrasts(proc)
    for target, row in summaries.items():
        leak_off = max(value for key, value in leakage[target].items() if key != target)
        print(
            f"SUBSTRATE carrier={carrier} target={target} "
            f"ordered={row['ordered_pairs']} unordered={row['unordered_pairs']} "
            f"cells={row['value_cells']} min_cell={row['min_cell_count']} "
            f"mean_own={row['mean_own']:.12g} min_own={row['min_own']:.12g} "
            f"p10_own={row['p10_own']:.12g} leak_off_mean={leak_off:.12g}"
        )
    for target in ("a", "b", "c"):
        row = leakage[target]
        print(
            f"LEAKAGE carrier={carrier} target={target} "
            f"obs_a={row['a']:.12g} obs_b={row['b']:.12g} obs_c={row['c']:.12g}"
        )

    t0 = time.time()
    exact = proc.mgram_table(eye, args.m)
    runtime = time.time() - t0
    norm_err = float(np.max(np.abs(exact.sum(axis=1) - 1.0)))
    print(
        f"MGRAM carrier={carrier} rows={proc.S} outcomes={proc.V ** args.m} "
        f"runtime_sec={runtime:.6f} max_norm_error={norm_err:.3e}"
    )

    carrier_stats = carrier_agreement(proc, exact, carrier, args.seed,
                                      args.kappa, args.d_hidden)
    print(
        f"CARRIER carrier={carrier} decode={carrier_stats['decode_accuracy']}/{proc.S} "
        f"mean_js={carrier_stats['mean_js']:.12g} max_js={carrier_stats['max_js']:.12g} "
        f"recon_error={carrier_stats['recon_error']:.12g} "
        f"rank={carrier_stats['rank']} condition_number={carrier_stats['condition_number']:.12g} "
        f"condition_rel_error={carrier_stats['condition_rel_error']:.12g} "
        f"min_column_sep={carrier_stats['min_column_sep']:.12g}"
    )

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
    print("RUN mode=confirm status=quarantined_pilot")
    print_constants(args)
    failures = confirm_scope_failures(args)
    if failures:
        return print_final(route_for(failures), failures, confirm=True)

    try:
        selftest(proc)
    except AssertionError as exc:
        failures.append(("HARNESS_FAIL", f"selftest assertion failed: {exc}"))

    oracle_args = argparse.Namespace(**vars(args))
    oracle_args.carrier = "oracle"
    failures.extend(run_panel(proc, oracle_args, "oracle"))

    mixed_args = argparse.Namespace(**vars(args))
    mixed_args.carrier = "mixed"
    failures.extend(run_panel(proc, mixed_args, "mixed"))

    substantive_route = None
    if not failures:
        substantive_route, intervention_failures = intervention_panel(proc, args)
        failures.extend(intervention_failures)

    return print_final(route_for(failures) if failures else substantive_route, failures, confirm=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--carrier", choices=("oracle", "mixed"), default="oracle")
    parser.add_argument("--m", type=int, default=M_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--kappa", type=float, default=KAPPA_DEFAULT)
    parser.add_argument("--d-hidden", type=int, default=D_HIDDEN_DEFAULT)
    args = parser.parse_args()
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
