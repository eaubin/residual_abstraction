"""exp38_propagation_gate.py — L1 propagating-state instrument + propagation gate.

PRE-REGISTRATION script (writeup: experiments/38-l1-propagation-gate.md). Decides,
for graded stack depth on the Dyck-2 checkpoint, the typed fork:

  PROPAGATED  — graded depth is a LOCALIZABLE residual summary (locality curve
                saturates early vs random-placement, AND t is necessary vs
                random-drop) -> proceed to L2 coarse localization with this signal.
  DISTRIBUTED — full-prefix transport is real (clears the random floor) but no small
                window suffices and no single locus is necessary: carriable but not
                localized (a typed middle).
  RECOMPUTED  — graded transport at/below the random floor even at full window: not
                localizably summarized. CLAIM-BOUNDED: never licenses "the model
                does not carry/propagate graded depth" (redundant distributed
                carrying is not excludable by interchange).
  HARNESS_FAIL / OBS_DRIFT / SEED_UNSTABLE — see the verdict block below.

Instrument: the exact teacher-forced m>=2 forced-close conditional (localize.cr_cond),
read from the V**m completion joint under a windowed residual interchange patch
(localize.make_patched_prefix). No rollout. The graded signal m=1 cannot see is the
close-readiness after k forced closes; the k-th conditional separates depth k vs k+1.

Discriminator: two curves, read against MEASURED references, not absolute windows.
  - locality (sufficiency): contiguous window ending at t, growing 1..full, vs an
    equal-count RANDOM-PLACEMENT control (locus vs injected-signal mass);
  - necessity: full-prefix minus t, vs full-prefix minus a RANDOM position;
  - the planted-locus reference (in-model shared-prefix, localize.planted_locus_pairs)
    sets the early-saturation ceiling shape AND is the transfer-validity gate: if a
    known single-position summary cannot clear the random floor at window 1 in this
    model, the locality axis is uninterpretable -> HARNESS_FAIL.

Ground-truth discipline: pairs are built from the Dyck parser on observed tokens;
the oracle is used only for the endpoint calibration audit (OBS_DRIFT). Honesty,
scope, and the claim bound are in the writeup.

Feasibility (precondition, already GREEN; see out/exp38_ceiling_smoke.txt and the
writeup): full-prefix transports 0.84-0.98 of the oracle gap at k=1 and k=2 with a
same-depth floor ~0; planted-locus window-1 transports 0.70-0.84 above a 0 random
floor; pair yields abundant. The registered run below decides the localization fork.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize import (LAYER, cr_cond, depth_triples,  # noqa: E402
                      make_patched_prefix, planted_locus_pairs, q_at,
                      require_expected_config, stack_labels)
from midstream import marginal, stream_to  # noqa: E402
from processes import PROCESSES  # noqa: E402
from expcommon import load_model, validity_gate  # noqa: E402
from battery import majority_vote, first_precedence  # noqa: E402

# ---- registered scope -----------------------------------------------------
POSITIONS = (8, 12, 16, 20)            # L0's interior positions
SEEDS = (700, 701, 702, 703)           # 4 seeds, as L0
N_SEQS = 6000                          # sequences per seed
HORIZONS = {1: (1, 2), 2: (2, 3)}      # k -> (clean depth, source depth)
GAP_MIN = 0.10                         # only score pairs with a real oracle gap
SEED_MAJORITY = 3                      # >=3/4 seeds for a stable verdict
R_RAND = 3                             # random-control draws to average (denoise)

# ---- registered thresholds (gate cutoffs, printed/audited; see writeup) ---
# Calibrated against the MEASURED references: planted-locus (ceiling shape) and
# random-placement (floor). Values chosen well inside the feasibility-smoke margins.
W_SMALL = 2          # "small window": <= this many positions ending at t
SAT_FRAC = 0.50      # early-saturation: small-window f >= this * full-prefix f
LOCUS_MARGIN = 0.15  # small-window f must beat random-placement f by this (locus)
NEC_MARGIN = 0.15    # necessity: dropping t must drop f by at least this...
                     # ...AND by more than dropping a random position
FULL_MIN = 0.30      # carriability: full-prefix f must beat the random floor by this
PLANTED_MIN = 0.30   # transfer-validity: planted window-1 f must beat random by this
OE_BAND = 0.10       # max conditional-vs-oracle endpoint gap (else OBS_DRIFT)

PRECEDENCE = ["HARNESS_FAIL", "OBS_DRIFT", "RECOMPUTED", "DISTRIBUTED",
              "PROPAGATED", "SEED_UNSTABLE"]


def windows_ending_at(t):
    """The registered window ladder: contiguous windows ending at t, sizes
    1,2,4,8,..., plus the full prefix [0..t]. Returned as position arrays."""
    sizes, w = [], 1
    while w < t + 1:
        sizes.append(w)
        w *= 2
    sizes.append(t + 1)
    sizes = sorted(set(sizes))
    return [(w, np.arange(t - w + 1, t + 1)) for w in sizes]


def _f(P, C, S):
    """Transport fraction (P-C)/(S-C), pooled over pairs with a real oracle gap."""
    g = S - C
    keep = np.abs(g) >= GAP_MIN
    if keep.sum() == 0:
        return float("nan")
    return float(np.mean((P[keep] - C[keep]) / g[keep]))


def _patched_obs(model, clean, src_resid, clean_resid, t, k, m, V, positions):
    ps = make_patched_prefix(clean_resid, src_resid, t, positions)
    return cr_cond(q_at(model, clean, t, m, V, prefix_state=ps), V, m, k)


def curve_at(model, clean, hi, lo, t, k, m, V, rng):
    """Locality + necessity curves at one (position, horizon), with matched
    random controls. clean/hi/lo are aligned token arrays: clean (at depth `lo`)
    are the tokens patched onto, hi (depth `hi`) is the transport source, lo (a
    distinct depth-`lo` instance) is the SAME-DEPTH floor source. Returns the
    per-window f_contig/f_random, the same-depth floor, and necessity numbers."""
    rc = stream_to(model, torch.from_numpy(clean), LAYER)
    rh = stream_to(model, torch.from_numpy(hi), LAYER)
    rl = stream_to(model, torch.from_numpy(lo), LAYER)
    C = cr_cond(q_at(model, clean, t, m, V), V, m, k)             # clean belief
    S = cr_cond(q_at(model, hi, t, m, V), V, m, k)               # source oracle
    out = {"C": C, "S": S, "windows": [], "f_contig": [], "f_random": []}
    all_pos = np.arange(t + 1)
    for w, pos in windows_ending_at(t):
        f_c = _f(_patched_obs(model, clean, rh, rc, t, k, m, V, pos), C, S)
        # random-placement floor at this size: average over R_RAND random sets
        f_r = np.mean([_f(_patched_obs(model, clean, rh, rc, t, k, m, V,
                                       rng.choice(all_pos, size=w, replace=False)),
                          C, S) for _ in range(R_RAND)])
        out["windows"].append(w); out["f_contig"].append(f_c)
        out["f_random"].append(float(f_r))
    out["f_full"] = out["f_contig"][-1]
    # carriability floor: full-prefix patch with a SAME-DEPTH source must NOT move
    # the conditional (the smoke's ~0 floor). Carriability = f_full - this.
    out["f_samedepth_full"] = _f(
        _patched_obs(model, clean, rl, rc, t, k, m, V, all_pos), C, S)
    # necessity: full minus t, vs full minus a random position (averaged)
    drop_t = np.array([p for p in all_pos if p != t])
    out["nec_t"] = out["f_full"] - _f(_patched_obs(model, clean, rh, rc, t, k, m, V, drop_t), C, S)
    nec_r = []
    for _ in range(R_RAND):
        rnd_drop = int(rng.choice(drop_t))
        drop_r = np.array([p for p in all_pos if p != rnd_drop])
        nec_r.append(out["f_full"] - _f(_patched_obs(model, clean, rh, rc, t, k, m, V, drop_r), C, S))
    out["nec_rand"] = float(np.mean(nec_r))
    return out


def verdict_one(curve):
    """The registered verdict for one (position, horizon) curve."""
    # early saturation: best small-window transport vs its random-placement match
    small = [(fc, fr) for w, fc, fr in
             zip(curve["windows"], curve["f_contig"], curve["f_random"])
             if w <= W_SMALL]
    small_f = max((fc for fc, _ in small), default=float("nan"))
    small_fr = max((fr for _, fr in small), default=float("nan"))
    f_full = curve["f_full"]
    carriable = (f_full - curve["f_samedepth_full"]) >= FULL_MIN
    if not carriable:
        return "RECOMPUTED"
    saturates = (small_f >= SAT_FRAC * f_full) and (small_f - small_fr) >= LOCUS_MARGIN
    necessary = (curve["nec_t"] >= NEC_MARGIN) and (curve["nec_t"] > curve["nec_rand"])
    return "PROPAGATED" if (saturates and necessary) else "DISTRIBUTED"


# ---- harness gates (run before any verdict) -------------------------------
def model_guards(model, proc, cfg, m, V):
    """no-op patch bit-exact; full-prefix patch reproduces source m=1 (L0's guard,
    extended here as the wiring sanity for the windowed patch path)."""
    rng = np.random.default_rng(0)
    Xe = proc.sample(64, cfg["seq_len"], rng)
    resid = stream_to(model, torch.from_numpy(Xe), LAYER)
    perm = rng.permutation(len(Xe))
    for t in POSITIONS:
        q0 = q_at(model, Xe, t, m, V)
        noop = q_at(model, Xe, t, m, V,
                    prefix_state=make_patched_prefix(resid, resid, t, []))
        if not np.allclose(q0, noop, atol=1e-6):
            print(f"  GUARD FAIL: no-op patch not bit-exact at t={t}"); return False
        full = make_patched_prefix(resid, resid[perm], t, np.arange(t + 1))
        qf = q_at(model, Xe, t, m, V, prefix_state=full)
        qs = q_at(model, Xe[perm], t, m, V)
        if not np.allclose(marginal(qf, V, 1, m), marginal(qs, V, 1, m), atol=1e-6):
            print(f"  GUARD FAIL: full patch m=1 != source m=1 at t={t}"); return False
    return True


def planted_locus_gate(model, proc, cfg, m, V):
    """Transfer-validity: a KNOWN single-position summary (shared-prefix planted
    locus) must transport at window 1 above the random floor by PLANTED_MIN, at
    every registered position. Else the locality axis is uninterpretable."""
    rng = np.random.default_rng(0)
    base = proc.sample(20000, cfg["seq_len"], rng)
    ok = True
    for t in POSITIONS:
        clean, src = planted_locus_pairs(base, t, m)
        if len(clean) < 256:
            print(f"  PLANTED FAIL: thin yield {len(clean)} at t={t}"); ok = False; continue
        rc = stream_to(model, torch.from_numpy(clean), LAYER)
        rs = stream_to(model, torch.from_numpy(src), LAYER)
        C = cr_cond(q_at(model, clean, t, m, V), V, m, 1)
        S = cr_cond(q_at(model, src, t, m, V), V, m, 1)
        f_w1 = _f(_patched_obs(model, clean, rs, rc, t, 1, m, V, [t]), C, S)
        rp = int(rng.choice([p for p in range(t + 1) if p != t]))
        f_r = _f(_patched_obs(model, clean, rs, rc, t, 1, m, V, [rp]), C, S)
        margin = f_w1 - f_r
        flag = "OK" if margin >= PLANTED_MIN else "FAIL"
        if margin < PLANTED_MIN:
            ok = False
        print(f"  planted t={t:2d} n={len(clean):4d} f_win1={f_w1:+.3f} "
              f"f_rand={f_r:+.3f} margin={margin:+.3f} -> {flag}")
    return ok


# ---- main path ------------------------------------------------------------
def run_gate(model, proc, cfg, seeds=SEEDS, n_seqs=N_SEQS, min_pairs=256):
    m, V = cfg["m"], proc.V
    print("[guards] no-op bit-exact; full patch m=1 = source m=1")
    if not model_guards(model, proc, cfg, m, V):
        print("  -> HARNESS_FAIL\n"); return "HARNESS_FAIL"
    print("  -> OK")
    print("[planted-locus transfer-validity gate]")
    if not planted_locus_gate(model, proc, cfg, m, V):
        print("  -> HARNESS_FAIL (locality axis uninterpretable)\n"); return "HARNESS_FAIL"
    print("  -> OK\n")

    per_seed = {k: [] for k in HORIZONS}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        Xe = proc.sample(n_seqs, cfg["seq_len"], rng)
        print(f"[seed {seed}]")
        for k, (lo, hi) in HORIZONS.items():
            verdicts, oes, summary = [], [], []
            for t in POSITIONS:
                labels = {i: stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
                clean, ihi, ilo = depth_triples(labels, lo, hi, rng)
                if len(clean) < min_pairs:
                    continue
                cur = curve_at(model, Xe[clean], Xe[ihi], Xe[ilo], t, k, m, V, rng)
                # endpoint calibration: conditional vs exact oracle on source
                ex = cr_cond(exact_source_joint(proc, Xe[ihi], t, m), V, m, k)
                oe = float(np.nanmean(np.abs(cur["S"] - ex)))
                oes.append(oe)
                v = "OBS_DRIFT" if oe > OE_BAND else verdict_one(cur)
                verdicts.append(v)
                summary.append((t, cur, v, oe))
            kv = _reduce_positions(verdicts)
            per_seed[k].append(kv)
            _print_horizon(k, lo, hi, summary, kv)
        print()

    agg = {f"k{k}": majority_vote(per_seed[k], threshold=SEED_MAJORITY,
                                  unstable="SEED_UNSTABLE") for k in HORIZONS}
    label, keys = first_precedence(agg, PRECEDENCE)
    print(f"per-horizon aggregate: {agg}")
    decision = label if label is not None else "PROPAGATED"
    print(f"\nDECISION (highest-precedence across horizons): {decision}")
    return decision


def exact_source_joint(proc, seqs, t, m):
    """Exact oracle completion joint at t for the source tokens (audit only)."""
    bel = np.stack([proc.beliefs_along(s)[t] for s in seqs])
    return proc.mgram_table(bel, m)


def _reduce_positions(verdicts):
    """A horizon's verdict from its per-position verdicts: OBS_DRIFT if any drifts;
    else PROPAGATED only if a majority of positions propagate; RECOMPUTED if a
    majority recompute; else DISTRIBUTED (the typed middle)."""
    if not verdicts:
        return "SEED_UNSTABLE"          # no qualifying positions -> underpowered
    if "OBS_DRIFT" in verdicts:
        return "OBS_DRIFT"
    n = len(verdicts)
    if verdicts.count("PROPAGATED") > n / 2:
        return "PROPAGATED"
    if verdicts.count("RECOMPUTED") > n / 2:
        return "RECOMPUTED"
    return "DISTRIBUTED"


def _print_horizon(k, lo, hi, summary, kv):
    print(f"  k={k} (depth {lo} vs {hi}) -> {kv}")
    for t, cur, v, oe in summary:
        sm = [f"{w}:{fc:+.2f}/{fr:+.2f}" for w, fc, fr in
              zip(cur["windows"], cur["f_contig"], cur["f_random"])]
        print(f"    t={t:2d} {v:11s} full={cur['f_full']:+.2f} "
              f"(samedepth {cur['f_samedepth_full']:+.2f}) nec_t={cur['nec_t']:+.2f} "
              f"nec_rnd={cur['nec_rand']:+.2f} oe={oe:.3f} | win[c/r]: {' '.join(sm)}")


# ---- self-tests (pure / tiny, no checkpoint) ------------------------------
def _selftest():
    # window ladder ends at the full prefix and is sorted/deduped
    ws = [w for w, _ in windows_ending_at(20)]
    assert ws[-1] == 21 and ws == sorted(set(ws)) and ws[0] == 1, ws
    last_pos = windows_ending_at(20)[-1][1]
    assert last_pos[0] == 0 and last_pos[-1] == 20            # full = [0..t]
    assert (windows_ending_at(20)[0][1] == np.array([20])).all()  # w=1 -> [t]

    # _f: clean->0, source->1, halfway->0.5 (gap filter respected)
    C = np.array([0.0, 0.0]); S = np.array([1.0, 1.0])
    assert _f(C, C, S) == 0.0 and _f(S, C, S) == 1.0
    assert abs(_f(np.array([0.5, 0.5]), C, S) - 0.5) < 1e-9
    assert np.isnan(_f(C, C, np.array([0.05, 0.05])))         # gap < GAP_MIN -> none

    # verdict_one: a localized planted-style curve -> PROPAGATED
    prop = {"windows": [1, 2, 21], "f_contig": [0.9, 0.9, 0.9],
            "f_random": [0.0, 0.0, 0.9], "f_full": 0.9, "f_samedepth_full": 0.0,
            "nec_t": 0.9, "nec_rand": 0.0}
    assert verdict_one(prop) == "PROPAGATED"
    # carriable but late-ramping, t not necessary -> DISTRIBUTED
    dist = {"windows": [1, 2, 21], "f_contig": [0.05, 0.1, 0.8],
            "f_random": [0.0, 0.05, 0.8], "f_full": 0.8, "f_samedepth_full": 0.0,
            "nec_t": 0.05, "nec_rand": 0.04}
    assert verdict_one(dist) == "DISTRIBUTED"
    # full-prefix does not clear the same-depth floor -> RECOMPUTED
    rec = {"windows": [1, 21], "f_contig": [0.0, 0.25], "f_random": [0.0, 0.25],
           "f_full": 0.25, "f_samedepth_full": 0.10, "nec_t": 0.0, "nec_rand": 0.0}
    assert verdict_one(rec) == "RECOMPUTED"

    # position reducer + precedence
    assert _reduce_positions(["PROPAGATED", "PROPAGATED", "DISTRIBUTED"]) == "PROPAGATED"
    assert _reduce_positions(["DISTRIBUTED", "PROPAGATED", "RECOMPUTED"]) == "DISTRIBUTED"
    assert _reduce_positions(["OBS_DRIFT", "PROPAGATED"]) == "OBS_DRIFT"
    assert first_precedence({"k1": "PROPAGATED", "k2": "DISTRIBUTED"},
                            PRECEDENCE)[0] == "DISTRIBUTED"
    assert majority_vote(["PROPAGATED"] * 3 + ["DISTRIBUTED"], threshold=3,
                         unstable="SEED_UNSTABLE") == "PROPAGATED"
    print("exp38 selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true",
                    help="tiny runnability check (1 seed, few seqs); NOT the "
                    "registered run")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest(); return
    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    require_expected_config(cfg)
    proc = PROCESSES[cfg["process"]]()
    model = load_model(args.outdir, cfg, proc)
    gap, passed = validity_gate(model, proc, cfg, 0)
    if not passed:
        print("HALT: validity gate failed."); sys.exit(1)
    print(f"=== Experiment 38: Dyck-2 propagation gate | L{cfg['layers']} "
          f"d{cfg['d_model']} | m={cfg['m']} ===\n")
    if args.dry:
        print("** DRY runnability check — 1 seed, reduced seqs/pairs; "
              "NOT the registered run **\n")
        run_gate(model, proc, cfg, seeds=(700,), n_seqs=2500, min_pairs=64)
    else:
        run_gate(model, proc, cfg)


if __name__ == "__main__":
    main()
