"""
scripts/oracle_withdrawal/substrate.py — Oracle-withdrawal experiment 1:
measurement substrate.

This is the first runnable gate for ORACLE_WITHDRAWAL.md. It does not
select abstractions. It verifies whether the new bounded probabilistic stack
grammar target has a competent model, affordable exact completion measures,
usable sampled-completion estimates, observable strata, and a non-vacuous
full-patch ceiling. Exact oracle use here is limited to measurement
calibration; abstraction/verdict thresholds are not tuned by this script.

Registered training command, after pre-run review:

  python3 train.py --process pstack --outdir out/pstack-L4 --layers 4 \
    --seq-len 40 --steps 8000 --m 3 --eval-seqs 1200 --burn-in 4 --seed 0

Registered substrate run:

  python3 scripts/oracle_withdrawal/substrate.py --outdir out/pstack-L4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abstraction import kl_rows
from discover import PairSet, self_checks
from expcommon import LAYER, load_model
from processes import PROCESSES

REGISTERED_CFG = {
    "process": "pstack",
    "seq_len": 40,
    "burn_in": 4,
    "d_model": 64,
    "layers": 4,
    "m": 3,
    "seed": 0,
}

TS = (10, 18, 26, 34)
M = 3
SAMPLE_ROWS = 192
SAMPLE_BUDGETS = (64, 256, 1024)
SAMPLE_REPEATS = 4
BOOTSTRAP_REPS = 2000
CONF_LEVEL = 0.95
PAIR_N = 256
PAIR_POOL = 800
STRATA_SEQS = 2000
COMPETENCE_GAP_MAX = 0.030
EXACT_TIME_MAX_SEC = 30.0
SAMPLE_MEAN_JS_MAX = 0.050
SAMPLE_P90_JS_MAX = 0.100
MIN_STRATUM_COUNT = 80
UNPATCHED_KL_MIN = 0.010
FULL_REL_GAIN_MIN = 0.50


def js_rows(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_rows(p, m) + kl_rows(q, m))


def upper_ci(values, statistic, conf_level, seed):
    """Nonparametric bootstrap upper confidence bound for a row statistic."""
    vals = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    n = len(vals)
    for i in range(BOOTSTRAP_REPS):
        sample = vals[rng.integers(0, n, n)]
        boots[i] = statistic(sample)
    return float(np.quantile(boots, conf_level))


def completion_id(seq, vocab):
    out = 0
    for tok in seq:
        out = out * vocab + int(tok)
    return out


def sample_completion_dist(proc, belief, m, budget, rng):
    counts = np.zeros(proc.V ** m, dtype=np.float64)
    for _ in range(budget):
        state = rng.choice(proc.S, p=belief)
        toks = []
        for _step in range(m):
            probs = proc.T[:, state, :].reshape(-1)
            k = rng.choice(proc.V * proc.S, p=probs)
            tok, state = divmod(k, proc.S)
            toks.append(tok)
        counts[completion_id(toks, proc.V)] += 1.0
    return counts / budget


def observable_depths(row):
    depth = 0
    out = np.zeros(len(row), dtype=np.int64)
    for i, tok in enumerate(row):
        if tok in (0, 1):
            depth += 1
        elif tok in (2, 3):
            depth -= 1
        out[i] = depth
    return out


def require_registered_config(cfg, force_invalid=False):
    mismatches = [(k, cfg.get(k), v) for k, v in REGISTERED_CFG.items()
                  if cfg.get(k) != v]
    if mismatches and not force_invalid:
        print("HALT: wrong checkpoint config for registered substrate run.")
        for key, got, want in mismatches:
            print(f"  {key}: got {got!r}, expected {want!r}")
        return False
    if mismatches:
        print("NOTE: exploratory run with config mismatches:")
        for key, got, want in mismatches:
            print(f"  {key}: got {got!r}, registered {want!r}")
    return True


def competence_gap(model, proc, cfg, seed):
    L, V = cfg["seq_len"], proc.V
    X = proc.sample(1000, L, np.random.default_rng(seed + 999))
    with torch.no_grad():
        total, count = 0.0, 0
        for i in range(0, len(X), 256):
            logits = model(torch.from_numpy(X[i:i + 256]))
            tgt = torch.from_numpy(X[i:i + 256, 1:]).reshape(-1)
            total += F.cross_entropy(logits[:, :-1].reshape(-1, V), tgt,
                                     reduction="sum").item()
            count += tgt.numel()
    opt, opt_count = 0.0, 0
    for row in X:
        b = proc.pi.copy()
        for t, s in enumerate(row[:-1]):
            # Predict the next token after the current prefix. This matches
            # the model CE target X[:, 1:] used above.
            b, _ = proc.belief_update(b, s)
            nxt = int(row[t + 1])
            opt -= np.log((b @ proc.T[nxt]).sum())
            opt_count += 1
    assert opt_count == count
    return total / count - opt / opt_count


def substrate_rows(proc, cfg, n_rows, seed):
    rng = np.random.default_rng(seed)
    X = proc.sample(max(128, n_rows // len(TS) + 8), cfg["seq_len"], rng)
    beliefs = []
    for row in X:
        B = proc.beliefs_along(row)
        for t in TS:
            beliefs.append(B[t])
            if len(beliefs) >= n_rows:
                return np.stack(beliefs)
    return np.stack(beliefs)


def exact_sample_audit(proc, beliefs, m, seed):
    t0 = time.time()
    exact = proc.mgram_table(beliefs, m)
    exact_sec = time.time() - t0
    rows = []
    for budget in SAMPLE_BUDGETS:
        vals = []
        for rep in range(SAMPLE_REPEATS):
            rng = np.random.default_rng(seed + 10_000 * budget + rep)
            qhat = np.stack([
                sample_completion_dist(proc, b, m, budget, rng)
                for b in beliefs
            ])
            vals.extend(js_rows(exact, qhat))
        vals = np.asarray(vals)
        mean_ci = upper_ci(vals, np.mean, CONF_LEVEL,
                           seed + 20_000 * budget + 1)
        p90_ci = upper_ci(vals, lambda x: np.quantile(x, 0.90), CONF_LEVEL,
                          seed + 20_000 * budget + 2)
        rows.append({
            "budget": budget,
            "mean_js": float(vals.mean()),
            "p90_js": float(np.quantile(vals, 0.90)),
            "mean_js_u95": mean_ci,
            "p90_js_u95": p90_ci,
        })
    return exact_sec, rows


def stratum_counts(proc, cfg, seed):
    X = proc.sample(STRATA_SEQS, cfg["seq_len"],
                    np.random.default_rng(seed + 4242))
    counts = {depth: 0 for depth in range(4)}
    for row in X:
        D = observable_depths(row)
        for t in TS:
            d = int(D[t])
            if d in counts:
                counts[d] += 1
    return counts


def full_patch_nonvacuity(model, proc, cfg, seed):
    ps = PairSet(model, proc, cfg, PAIR_N, M, seed + 111, PAIR_POOL,
                 layer=LAYER, ts=TS)
    self_checks(model, ps, LAYER, M, proc.V)
    q_src = ps.run(model, None, src_side=True)
    q_un = ps.run(model, None)
    q_full = ps.run(model, np.eye(cfg["d_model"]))
    d0 = float(kl_rows(q_src, q_un).mean())
    dfull = float(kl_rows(q_src, q_full).mean())
    rel = (d0 - dfull) / d0 if d0 > 0 else -np.inf
    return d0, dfull, rel


def process_selftest():
    proc = PROCESSES["pstack"]()
    assert proc.name == "pstack"
    assert proc.V == 6 and proc.S == 30
    assert np.allclose(proc.T.sum(axis=0).sum(axis=1), 1.0)
    rng = np.random.default_rng(0)
    X = proc.sample(64, 32, rng)
    B = proc.beliefs_along(X[0])
    G = proc.mgram_table(B[:8], 3)
    assert np.allclose(G.sum(axis=1), 1.0)
    exact = proc.mgram_dist(B[5], 3)
    qhat = sample_completion_dist(proc, B[5], 3, 512, rng)
    assert np.isclose(qhat.sum(), 1.0)
    assert np.isfinite(js_rows(exact[None, :], qhat[None, :])).all()
    print("selftest passed: pstack process, exact mgrams, sampler, JS")


def print_verdict(name, passed, detail):
    print(f"{name}: {'PASS' if passed else 'FAIL'} — {detail}")
    return passed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/pstack-L4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force-invalid", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        process_selftest()
        return 0

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    if not require_registered_config(cfg, args.force_invalid):
        return 1
    proc = PROCESSES[cfg["process"]]()
    model = load_model(args.outdir, cfg, proc)

    print("=== Experiment 23: oracle-withdrawal substrate gate ===")
    print(f"target={proc.name} S={proc.S} V={proc.V} outdir={args.outdir}")
    print(f"registered ts={TS} m={M} sample_budgets={SAMPLE_BUDGETS}\n")

    results = []

    gap = competence_gap(model, proc, cfg, args.seed)
    results.append(print_verdict(
        "P1 model competence",
        gap <= COMPETENCE_GAP_MAX,
        f"gap-to-optimal {gap:+.4f} <= {COMPETENCE_GAP_MAX:.3f} nats",
    ))

    beliefs = substrate_rows(proc, cfg, SAMPLE_ROWS, args.seed + 222)
    exact_sec, sample_rows = exact_sample_audit(proc, beliefs, M,
                                                args.seed + 333)
    results.append(print_verdict(
        "P2 exact runtime",
        exact_sec <= EXACT_TIME_MAX_SEC,
        f"{len(beliefs)} rows x V^m={proc.V ** M} in {exact_sec:.2f}s "
        f"<= {EXACT_TIME_MAX_SEC:.1f}s",
    ))
    print(f"\n[sampled-vs-exact JS, {CONF_LEVEL:.0%} bootstrap upper bounds]")
    for row in sample_rows:
        print(f"  budget={row['budget']:4d}  mean_js={row['mean_js']:.4f}  "
              f"mean_u95={row['mean_js_u95']:.4f}  "
              f"p90_js={row['p90_js']:.4f}  p90_u95={row['p90_js_u95']:.4f}")
    last = sample_rows[-1]
    monotone = all(sample_rows[i + 1]["mean_js"] <=
                   sample_rows[i]["mean_js"] * 1.10
                   for i in range(len(sample_rows) - 1))
    sample_ok = (last["mean_js_u95"] <= SAMPLE_MEAN_JS_MAX and
                 last["p90_js_u95"] <= SAMPLE_P90_JS_MAX and monotone)
    results.append(print_verdict(
        "P3 sampled estimator",
        sample_ok,
        f"budget {last['budget']} mean_u95 {last['mean_js_u95']:.4f} <= "
        f"{SAMPLE_MEAN_JS_MAX:.3f}, p90_u95 {last['p90_js_u95']:.4f} <= "
        f"{SAMPLE_P90_JS_MAX:.3f}, monotone={monotone}",
    ))

    counts = stratum_counts(proc, cfg, args.seed)
    print("\n[observable depth strata]")
    for depth, count in counts.items():
        print(f"  depth={depth}: {count}")
    min_count = min(counts.values())
    results.append(print_verdict(
        "P4 PairSet strata",
        min_count >= MIN_STRATUM_COUNT,
        f"min depth count {min_count} >= {MIN_STRATUM_COUNT}",
    ))

    d0, dfull, rel = full_patch_nonvacuity(model, proc, cfg, args.seed)
    results.append(print_verdict(
        "P5 full-patch non-vacuity",
        d0 >= UNPATCHED_KL_MIN and rel >= FULL_REL_GAIN_MIN,
        f"D0={d0:.4f} >= {UNPATCHED_KL_MIN:.3f}, Dfull={dfull:.4f}, "
        f"relative_gain={rel:.1%} >= {FULL_REL_GAIN_MIN:.0%}",
    ))

    print("\n[oracle discipline]")
    print("  exact oracle used only for substrate calibration: competence "
          "yardstick, exact mgrams, sampled-vs-exact audit, and full-patch "
          "ceiling audit")
    print("  no proposal family, compact reference, acceptance threshold, or "
          "abstraction verdict is selected here")

    print("\nDECISION:")
    if all(results):
        print("  GO: pstack measurement substrate is registered-usable for "
              "hidden-oracle reference selection.")
    else:
        print("  NO-GO: do not start hidden-oracle reference selection. "
              "Register a substrate repair or the predeclared fallback "
              "target before abstraction experiments.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
