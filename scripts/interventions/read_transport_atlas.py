"""
Experiment 31 — read-transport atlas diagnostic on pstack-L4 (pre-I2).

Diagnostic-only. NO interventions, NO patching, NO learned writes, NO CEGAR.
This script produces a typed *routing* decision about a read/representational-
geometry question, to run BEFORE committing to benchmark step I2. It emits no
writability, controllability, or intervention claim.

Question (see experiments/31-read-transport-atlas.md): exp 30 (I1) returned
FIXED_READ_LIMIT(phi1_next_closes,phi2_net_return) because the exp-29 affine
predicate reads decode on discovery positions {10,18} but fail the held-out
position R2 gate at {26,34}. Exp 30 only ever measured the disc->held
off-diagonal transfer cell; it never computed the in-place held-out diagonal.
This atlas asks the measurement exp 30 skipped:

    For each target predicate at L1 on pstack-L4, is the predicate linearly
    readable IN PLACE at the held-out positions, and if so, is the readout
    direction shared with the discovery positions or position-specific?

Claim-producing command, after preregistration review only:

    python3 scripts/interventions/read_transport_atlas.py --outdir out/pstack-L4

Review-only checks (allowed before approval):

    python3 scripts/interventions/read_transport_atlas.py --selftest

All quantities are observable/model-only (PairSet target run -> obs_pphi). No
exact predicate truth is consulted: the atlas has no patch endpoints to audit.
Built on the living helpers in interventions.py / predicates.py / abstraction.py
/ discover.py; frozen experiment scripts are never imported back here.

Read-prediction convention (registered difference from exp 30): every R2 here —
in-place and transfer — is scored with the FULL fitted affine map (w, b):
yhat = Rc @ w + b. Exp 30's decode_heldout applied the *unit* covector c=w/||w||
with the w-scale bias b, a scale mismatch that is harmless for the qualitative
"does it transport" verdict but changes the decimal transfer R2. The unit
covector is used here only for the direction cosine in measurement (3). The
atlas therefore reproduces exp 30's transfer failure QUALITATIVELY, not to the
decimal.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import interventions as IV
import predicates as P
from abstraction import center_by_position
from discover import PairSet, self_checks
from expcommon import LAYER, load_model
from processes import PROCESSES


REGISTERED_CFG = {"process": "pstack", "seq_len": 40, "burn_in": 4,
                  "d_model": 64, "layers": 4, "m": 3, "seed": 0}
M = 3
TARGETS = ("phi1_next_closes", "phi2_net_return")
CONTROL_PREDICATES = ("phi3_all_neutral", "phi4_first_matched")
SEEDS = (400, 401, 402, 403)

# Grouped bins reuse the exp-30 pair construction (counts + seed offsets) so the
# disc->held off-diagonal cell is a tripwire reproduction of exp 30.
TS_DISC = (10, 18)
TS_HELDOUT = (26, 34)
PAIRS_DISC = 512
PAIRS_HELDOUT = 1024
SEED_OFF_DISC = 111
SEED_OFF_HELD = 222
# The new dense atlas: single-position bins, fresh independent pools.
SINGLE_POSITIONS = (6, 10, 14, 18, 22, 26, 30, 34)
PAIRS_SINGLE = 512
SEED_OFF_SINGLE = 700  # single bin t uses seed_seqs = seed + SEED_OFF_SINGLE + t
PAIR_POOL = 900

# Registered thresholds (reusing exp 30 where applicable).
R2_MIN = 0.50         # "readable" / "transports" threshold (exp 30)
VAR_MIN = 0.05        # p_phi std vacuity floor (exp 30)
FLOOR_MARGIN = 0.10   # in-place R2 must beat its own shuffle floor by this
COS_SHARED = 0.70     # read covectors counted as a shared direction
PDIST_MAX = 0.15      # per-position p_phi mean spread above which positions are
                      # a poorly-chosen transfer axis
LAM = 1e-2            # ridge penalty (exp 30)
SEED_MAJORITY = 3     # >=3/4 seeds for an aggregate, else SEED_UNSTABLE


# ---------------------------------------------------------------------------
# observable bin frame (residuals + per-predicate p_phi, no patch)
# ---------------------------------------------------------------------------

def bin_frame(ps, model, masks, d, rng):
    """Residual frame + per-position centering + observable p_phi for one bin.

    Runs the living evaluator once on the UNPATCHED target run
    (``ps.run(model, None)`` -> ``midstream.chain_probs``) and reduces to
    observable ``p_phi`` per registered predicate via ``predicates.obs_pphi``.
    The train/test split is target-independent and fixed per bin so every
    predicate is fit/scored on the same halves. Centering uses the train mask
    (``abstraction.center_by_position``), the position deconfound.
    """
    R, _, pos, _, _ = IV.pairset_residual_frame(ps, d)
    n = ps.n
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    train_mask = np.zeros(n, dtype=bool)
    train_mask[tr] = True
    Rc = center_by_position(R, pos, train_mask)
    q = ps.run(model, None)                       # unpatched target run
    pphi = {name: P.obs_pphi(q, m) for name, m in masks.items()}
    return {"Rc": Rc, "tr": tr, "te": te, "pphi": pphi, "pos": pos, "n": n}


def fit_affine(frame, y):
    """Fit the affine ridge read on the train half; return (w, b)."""
    return IV.affine_readout(frame["Rc"][frame["tr"]], y[frame["tr"]], LAM)


def r2_test(frame, y, w, b):
    """Score R2 of the full fitted affine map on the test half of a bin."""
    te = frame["te"]
    return IV.r2_score(y[te], frame["Rc"][te] @ w + b)


def read_summary(frame, name):
    """std and in-place test R2 of one predicate's read on one bin (controls)."""
    y = frame["pphi"][name]
    w, b = fit_affine(frame, y)
    return float(y.std()), r2_test(frame, y, w, b)


# ---------------------------------------------------------------------------
# single-position transfer matrix
# ---------------------------------------------------------------------------

def single_matrix(single_frames, target):
    """Cross-position read transfer R2 matrix + per-position mean p_phi.

    Fit (w_i, b_i) on bin i's train half; evaluate on bin j's test half. The
    diagonal is honest in-place decodability; the off-diagonal is read carry
    (exp 30 measured exactly one off-diagonal cell). Returns (M, pos_means)."""
    fits, means = [], []
    for f in single_frames:
        y = f["pphi"][target]
        fits.append(fit_affine(f, y))
        means.append(float(y.mean()))
    nb = len(single_frames)
    Mx = np.full((nb, nb), np.nan)
    for i in range(nb):
        wi, bi = fits[i]
        for j in range(nb):
            f = single_frames[j]
            Mx[i, j] = r2_test(f, f["pphi"][target], wi, bi)
    return Mx, np.asarray(means)


# ---------------------------------------------------------------------------
# per-target atlas measurement
# ---------------------------------------------------------------------------

def target_measures(disc_f, held_f, single_frames, target, rng):
    yd = disc_f["pphi"][target]
    yh = held_f["pphi"][target]

    # (1) in-place decodability per grouped bin
    wd, bd = fit_affine(disc_f, yd)
    r2_inplace_disc = r2_test(disc_f, yd, wd, bd)
    wh, bh = fit_affine(held_f, yh)
    r2_inplace_held = r2_test(held_f, yh, wh, bh)

    # label-shuffle floor at the held bin (fit against permuted p_phi labels)
    yshuf = yh.copy()
    rng.shuffle(yshuf)
    ws, bs = fit_affine(held_f, yshuf)
    r2_shuffle_held = r2_test(held_f, yshuf, ws, bs)

    # (2) the single exp-30 off-diagonal cell: fit at disc, evaluate at held
    #     (full held rows; disc map never saw held). Full fitted map (w, b).
    r2_transfer = IV.r2_score(yh, held_f["Rc"] @ wd + bd)

    # (3) direction similarity (signed cosine of unit read covectors)
    cos = float(IV.unit(wd) @ IV.unit(wh))

    # (4) predicate distribution by position (exchangeability)
    Mx, pos_means = single_matrix(single_frames, target)
    spread = float(pos_means.max() - pos_means.min())

    return {
        "std_disc": float(yd.std()),
        "std_held": float(yh.std()),
        "r2_inplace_disc": float(r2_inplace_disc),
        "r2_inplace_held": float(r2_inplace_held),
        "r2_shuffle_held": float(r2_shuffle_held),
        "r2_transfer_disc_held": float(r2_transfer),
        "cos_held_disc": cos,
        "pphi_pos_spread": spread,
        "single_matrix": Mx,
        "pos_means": pos_means,
    }


# ---------------------------------------------------------------------------
# verdict partition (registered)
# ---------------------------------------------------------------------------

def classify_target(m):
    """Exactly one branch per (target, seed). Readability claim only."""
    if m["std_disc"] < VAR_MIN or m["std_held"] < VAR_MIN:
        return "TARGET_VACUOUS"
    if m["r2_inplace_disc"] < R2_MIN:
        return "DISC_READ_FAILED"
    if (m["r2_inplace_held"] < R2_MIN
            or m["r2_inplace_held"] - m["r2_shuffle_held"] < FLOOR_MARGIN):
        return "NOT_READABLE_LATE"
    if m["cos_held_disc"] >= COS_SHARED:
        return "SHARED_READ_SCALE_DRIFT"
    return "POSITION_SPECIFIC_READ"


def positions_exchangeable(m):
    return bool(m["pphi_pos_spread"] <= PDIST_MAX)


def aggregate(values):
    counts = {v: values.count(v) for v in set(values)}
    top = max(counts, key=counts.get)
    return top if counts[top] >= SEED_MAJORITY else "SEED_UNSTABLE"


def aggregate_bool(values):
    """Majority of a boolean overlay, kept in the SEED_UNSTABLE vocabulary."""
    return aggregate([str(bool(v)) for v in values])


def decide(target_aggs):
    parts = [f"{t}={target_aggs[t]}" for t in TARGETS]
    return "ATLAS(" + ", ".join(parts) + ")"


ROUTING = {
    "SHARED_READ_SCALE_DRIFT":
        "scale/bias drift of a shared read -> per-position calibration, "
        "then re-ask I1 cleanly (write question reopens).",
    "POSITION_SPECIFIC_READ":
        "readable late but read is position-specific -> I2 with "
        "position-conditioned reads; re-run I1 with a transport-valid read.",
    "NOT_READABLE_LATE":
        "predicate not linearly present at L1 at later positions -> I4/depth: "
        "patch point or layer is wrong there, not a read-freedom problem.",
    "DISC_READ_FAILED":
        "exp-30 premise not reproduced -> fix substrate/measurement before "
        "reinterpreting exp 30.",
    "TARGET_VACUOUS":
        "predicate too flat on a grouped bin to read.",
    "SEED_UNSTABLE":
        "branch not reproduced across seeds -> routing question not stably "
        "answerable on this substrate.",
}


# ---------------------------------------------------------------------------
# selftest (no model)
# ---------------------------------------------------------------------------

def selftest():
    IV._selftest()
    P._selftest()

    base = {
        "std_disc": 0.2, "std_held": 0.2,
        "r2_inplace_disc": 0.7, "r2_inplace_held": 0.7,
        "r2_shuffle_held": 0.0, "cos_held_disc": 0.9,
        "pphi_pos_spread": 0.05,
    }

    def m(**kw):
        x = dict(base)
        x.update(kw)
        return x

    # verdict partition: each branch is reachable and the order is enforced.
    assert classify_target(m()) == "SHARED_READ_SCALE_DRIFT"
    assert classify_target(m(cos_held_disc=0.3)) == "POSITION_SPECIFIC_READ"
    assert classify_target(m(std_held=0.01)) == "TARGET_VACUOUS"
    assert classify_target(m(std_disc=0.01)) == "TARGET_VACUOUS"
    assert classify_target(m(r2_inplace_disc=0.2)) == "DISC_READ_FAILED"
    assert classify_target(m(r2_inplace_held=0.2)) == "NOT_READABLE_LATE"
    # readable absolute, but does not beat its own shuffle floor by the margin
    assert classify_target(m(r2_inplace_held=0.55,
                             r2_shuffle_held=0.50)) == "NOT_READABLE_LATE"
    # vacuity takes precedence over a (spuriously) decoding read
    assert classify_target(m(std_disc=0.01,
                             r2_inplace_disc=0.2)) == "TARGET_VACUOUS"

    # exchangeability overlay
    assert positions_exchangeable(m(pphi_pos_spread=0.05))
    assert not positions_exchangeable(m(pphi_pos_spread=0.25))

    # aggregation: majority and instability
    assert aggregate(["A", "A", "A", "B"]) == "A"
    assert aggregate(["A", "A", "B", "B"]) == "SEED_UNSTABLE"
    assert aggregate_bool([True, True, True, False]) == "True"
    assert aggregate_bool([True, True, False, False]) == "SEED_UNSTABLE"

    # decision string is a routing string over both load-bearing targets
    dec = decide({"phi1_next_closes": "POSITION_SPECIFIC_READ",
                  "phi2_net_return": "NOT_READABLE_LATE"})
    assert dec == ("ATLAS(phi1_next_closes=POSITION_SPECIFIC_READ, "
                   "phi2_net_return=NOT_READABLE_LATE)"), dec

    # single-position transfer matrix: a shared direction transfers (diag and
    # same-direction off-diagonal high); an orthogonal-read bin does not carry.
    rng = np.random.default_rng(0)
    d = 6
    w1 = rng.standard_normal(d)
    w2 = rng.standard_normal(d)
    w2 = w2 - (w2 @ w1) / (w1 @ w1) * w1          # make w2 ⟂ w1

    def mkframe(wt):
        R = rng.standard_normal((80, d))
        y = R @ wt
        Rc = R - R.mean(0)
        idx = rng.permutation(80)
        return {"Rc": Rc, "tr": idx[:40], "te": idx[40:], "pphi": {"phiX": y}}

    frames = [mkframe(w1), mkframe(w1), mkframe(w2)]
    Mx, means = single_matrix(frames, "phiX")
    assert Mx[0, 0] > 0.9 and Mx[1, 1] > 0.9, Mx
    assert Mx[0, 1] > 0.9, Mx                      # same direction transfers
    assert Mx[0, 2] < 0.5, Mx                      # orthogonal read does not
    assert means.shape == (3,)

    print("read-transport-atlas selftest passed: verdict partition, "
          "exchangeability overlay, aggregation, transfer matrix")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def build_frames(model, proc, cfg, masks, seed, d):
    rng = np.random.default_rng(seed)
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + SEED_OFF_DISC,
                   PAIR_POOL, layer=LAYER, ts=TS_DISC)
    held = PairSet(model, proc, cfg, PAIRS_HELDOUT, M, seed + SEED_OFF_HELD,
                   PAIR_POOL, layer=LAYER, ts=TS_HELDOUT)
    self_checks(model, disc, LAYER, M, proc.V)
    self_checks(model, held, LAYER, M, proc.V)
    disc_f = bin_frame(disc, model, masks, d, rng)
    held_f = bin_frame(held, model, masks, d, rng)

    single_frames = []
    for t in SINGLE_POSITIONS:
        ps = PairSet(model, proc, cfg, PAIRS_SINGLE, M,
                     seed + SEED_OFF_SINGLE + t, PAIR_POOL, layer=LAYER,
                     ts=(t,))
        self_checks(model, ps, LAYER, M, proc.V)
        single_frames.append(bin_frame(ps, model, masks, d, rng))
    return disc_f, held_f, single_frames, rng


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/pstack-L4")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    mism = [(k, cfg.get(k), v) for k, v in REGISTERED_CFG.items()
            if cfg.get(k) != v]
    if mism:
        print("HALT: wrong checkpoint config:", mism)
        return 1
    proc = PROCESSES[cfg["process"]]()
    model = load_model(args.outdir, cfg, proc)
    masks = P.registered_masks(proc.V, M)
    d = cfg["d_model"]

    print("=== Exp 31: read-transport atlas (diagnostic, no patch) ===")
    print(f"target={proc.name} m={M} LAYER={LAYER} seeds={SEEDS}")
    print(f"grouped disc positions={TS_DISC}; grouped held positions={TS_HELDOUT}")
    print(f"single-position atlas={SINGLE_POSITIONS}")
    print("targets:", ", ".join(TARGETS))
    print("controls (read-only):", ", ".join(CONTROL_PREDICATES))
    print(f"thresholds: R2_MIN={R2_MIN} VAR_MIN={VAR_MIN} "
          f"FLOOR_MARGIN={FLOOR_MARGIN} COS_SHARED={COS_SHARED} "
          f"PDIST_MAX={PDIST_MAX} LAM={LAM}")
    print("Observable-only: no patch, no exact-truth audit, no intervention.\n")

    per_seed_verdict = {t: [] for t in TARGETS}
    per_seed_exch = {t: [] for t in TARGETS}
    for seed in SEEDS:
        print(f"[seed {seed}]")
        disc_f, held_f, single_frames, rng = build_frames(
            model, proc, cfg, masks, seed, d)

        for name in CONTROL_PREDICATES:
            sd, r2d = read_summary(disc_f, name)
            sh, r2h = read_summary(held_f, name)
            print(f"  control {name:<18} std disc/held {sd:.3f}/{sh:.3f} "
                  f"inplace R2 disc/held {r2d:.2f}/{r2h:.2f}")

        for target in TARGETS:
            mm = target_measures(disc_f, held_f, single_frames, target, rng)
            verdict = classify_target(mm)
            exch = positions_exchangeable(mm)
            per_seed_verdict[target].append(verdict)
            per_seed_exch[target].append(exch)
            print(f"\n  target {target}:")
            print(f"    std disc/held {mm['std_disc']:.3f}/{mm['std_held']:.3f}")
            print(f"    in-place R2 disc {mm['r2_inplace_disc']:.3f} | "
                  f"held {mm['r2_inplace_held']:.3f} "
                  f"(shuffle floor {mm['r2_shuffle_held']:.3f})")
            print(f"    disc->held transfer R2 {mm['r2_transfer_disc_held']:.3f} "
                  f"(exp-30 off-diagonal cell, qualitative)")
            print(f"    cos(held,disc) read covectors {mm['cos_held_disc']:.3f}")
            print(f"    per-position p_phi mean spread "
                  f"{mm['pphi_pos_spread']:.3f} "
                  f"(exchangeable={exch})")
            pm = ", ".join(f"t{t}:{v:.3f}"
                           for t, v in zip(SINGLE_POSITIONS, mm["pos_means"]))
            print(f"    per-position mean p_phi: {pm}")
            print("    single-position transfer R2 matrix "
                  "(rows=fit position, cols=eval position):")
            header = "      fit\\eval " + " ".join(
                f"t{t:<5}" for t in SINGLE_POSITIONS)
            print(header)
            for i, t in enumerate(SINGLE_POSITIONS):
                cells = " ".join(f"{mm['single_matrix'][i, j]:6.2f}"
                                 for j in range(len(SINGLE_POSITIONS)))
                print(f"      t{t:<7} {cells}")
            print(f"    -> {verdict}")
        print()

    print("[multi-seed aggregate]")
    target_aggs = {}
    for target in TARGETS:
        verdicts = per_seed_verdict[target]
        exch = per_seed_exch[target]
        target_aggs[target] = aggregate(verdicts)
        exch_agg = aggregate_bool(exch)
        print(f"  {target:<18} {verdicts} -> {target_aggs[target]}")
        print(f"  {'':<18} positions_exchangeable {exch} -> {exch_agg}")

    print(f"\nDECISION: {decide(target_aggs)}")
    for target in TARGETS:
        agg = target_aggs[target]
        route = ROUTING.get(agg, "see per-target aggregate.")
        print(f"  {target}: {agg} -> {route}")
    exch_overlay = {t: aggregate_bool(per_seed_exch[t]) for t in TARGETS}
    print(f"  exchangeability overlay: {exch_overlay} "
          f"(False -> positions are a poorly-chosen transfer axis; switch "
          f"transfer axis before further intervention work)")
    print("\nNOTE: this is a readability/representational-geometry routing "
          "decision only. No writability or controllability is claimed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
