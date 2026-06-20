"""
Experiment 32 — pre-I2 read-transport discriminator on pstack-L4.

Diagnostic-only. NO interventions, NO patching, NO learned writes, NO CEGAR.
This is INTERVENTION_CLASS_BENCHMARK.md's "Pre-I2 gate": before any read-freedom
work (I2), resolve the confound the exp-31 Result-Review Addendum flagged in its
shared-vs-position-specific call. It emits no writability, controllability, or
intervention claim — only a readability / representational-geometry routing
decision that gates whether I2 runs (and in which form).

Question (see experiments/32-read-transport-discriminator.md): exp 31 concluded
POSITION_SPECIFIC_READ for phi1_next_closes and phi2_net_return at L1/m=3 — both
readable in place at held-out positions {26,34} (in-place R2 ~0.6-0.75) but with
near-orthogonal disc/held read covectors. That call turned on one scalar cosine
sitting at the d=64 noise floor (~0.125) with no reliability baseline, and
cos~0 between two strong in-place reads is equally consistent with (a) a
genuinely position-specific read OR (b) a single shared direction two
underdetermined collinear fits failed to agree on. This experiment runs the two
checks that separate them:

    1. Gain/bias-refit discriminator. Freeze the discovery-fit direction wd and
       refit ONLY a scalar gain+bias at the held-out positions; measure the
       recovered in-place held R2. Plus a pooled-positions cross-check: fit one
       shared read across all single positions, measure per-position in-place R2.
    2. Cosine reliability ceiling. Within-position cosine between two independent
       fits of the SAME predicate at the SAME position (disjoint train halves) —
       the cosine a genuinely shared read achieves under fit noise — reported
       against the 1/sqrt(d) noise floor. This is the missing baseline that
       makes COS_SHARED interpretable.

Claim-producing command, after preregistration review only:

    python3 scripts/interventions/read_transport_discriminator.py --outdir out/pstack-L4

Review-only checks (allowed before approval):

    python3 scripts/interventions/read_transport_discriminator.py --selftest

All quantities are observable/model-only (PairSet target run -> obs_pphi). No
exact predicate truth is consulted: the discriminator has no patch endpoints to
audit. Built on the living helpers in interventions.py / predicates.py /
abstraction.py / discover.py / battery.py; frozen experiment scripts (exp 30,
exp 31) are never imported back here.

Frame convention (continuity with exp 31, on the record): the bin frame,
per-position centering, full-affine fit, and full-map R2 scoring mirror exp 31's
atlas helpers exactly, so the in-place R2 and cross cosine reproduce exp 31. The
helpers are re-declared locally rather than imported because exp 31 is a
concluded/frozen script (AGENTS.md library-home rule: frozen scripts are never
imported from); they are thin wrappers over the living-library primitives.
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
from battery import majority_vote
from discover import PairSet, self_checks
from expcommon import LAYER, load_model
from model import pick_device
from processes import PROCESSES


REGISTERED_CFG = {"process": "pstack", "seq_len": 40, "burn_in": 4,
                  "d_model": 64, "layers": 4, "m": 3, "seed": 0}
M = 3
TARGETS = ("phi1_next_closes", "phi2_net_return")
CONTROL_PREDICATES = ("phi3_all_neutral", "phi4_first_matched")
SEEDS = (400, 401, 402, 403)

# Grouped bins reuse the exp-30/31 pair construction (counts + seed offsets) so
# the disc/held in-place reads are continuous with exp 31.
TS_DISC = (10, 18)
TS_HELDOUT = (26, 34)
PAIRS_DISC = 512
PAIRS_HELDOUT = 1024
SEED_OFF_DISC = 111
SEED_OFF_HELD = 222
# Single-position bins as in exp 31 (used for the cosine ceiling and pooled read).
SINGLE_POSITIONS = (6, 10, 14, 18, 22, 26, 30, 34)
PAIRS_SINGLE = 512
SEED_OFF_SINGLE = 700  # single bin t uses seed_seqs = seed + SEED_OFF_SINGLE + t
PAIR_POOL = 900

# Registered thresholds (reuse exp 30/31 where applicable).
R2_MIN = 0.50         # "readable" / "recovers" threshold (exp 30/31)
VAR_MIN = 0.05        # p_phi std vacuity floor (exp 30/31)
FLOOR_MARGIN = 0.10   # an R2 must beat its own shuffle floor by this (exp 31)
COS_SHARED = 0.70     # read covectors counted as a shared direction (exp 31)
CEILING_MIN = 0.50    # cosine-reliability floor: same-position/same-predicate
                      # fits must agree at >= this cosine for the cosine to be a
                      # usable sharing instrument. Set well above the 1/sqrt(d)
                      # noise floor (~0.125) and below COS_SHARED.
LAM = 1e-2            # ridge penalty (exp 30/31)
SEED_MAJORITY = 3     # >=3/4 seeds for an aggregate, else SEED_UNSTABLE


# ---------------------------------------------------------------------------
# observable bin frame (residuals + per-predicate p_phi, no patch)
# Thin re-declarations of the exp-31 atlas helpers over the living library.
# ---------------------------------------------------------------------------

def bin_frame(ps, model, masks, d, rng):
    """Residual frame + per-position centering + observable p_phi for one bin.

    Mirrors exp 31's ``bin_frame``: runs the living evaluator once on the
    UNPATCHED target run (``ps.run(model, None)`` -> ``midstream.chain_probs``)
    and reduces to observable ``p_phi`` per registered predicate. Centering uses
    the train mask (``abstraction.center_by_position``)."""
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
    """Fit the full affine ridge read on the train half; return (w, b)."""
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
# discriminator measurements (new for exp 32)
# ---------------------------------------------------------------------------

def refit_gain_bias(frame, y, wd):
    """Freeze direction wd; refit ONLY scalar gain+bias at this bin.

    Projects the bin's centered residuals onto the frozen discovery direction
    ``wd`` to a scalar feature ``s = Rc @ wd``, fits ``y ~ g*s + b`` on the
    train half, and scores R2 with that 1-D map on the test half. Measures how
    much held-out in-place R2 a SHARED direction recovers once its
    per-position scale/bias is recalibrated — the direct shared-vs-specific
    test, independent of any cosine."""
    s = frame["Rc"] @ wd
    tr, te = frame["tr"], frame["te"]
    if np.std(s[tr]) < 1e-12:
        return 0.0
    g, b = np.polyfit(s[tr], y[tr], 1)            # slope, intercept
    return IV.r2_score(y[te], g * s[te] + b)


def pooled_read_per_position(single_frames, target):
    """Fit ONE shared affine read pooled over all single positions; score each.

    Cross-check on the gain/bias refit: if a single read pooled across all
    positions decodes every position in place, the read is shared. Returns the
    per-position in-place test R2 list (each frame is already per-position
    centered, so pooling centered rows is valid)."""
    Xtr = np.vstack([f["Rc"][f["tr"]] for f in single_frames])
    ytr = np.concatenate([f["pphi"][target][f["tr"]] for f in single_frames])
    w, b = IV.affine_readout(Xtr, ytr, LAM)
    return [r2_test(f, f["pphi"][target], w, b) for f in single_frames]


def cosine_ceiling(single_frames, target):
    """Within-position cosine between two independent same-predicate fits.

    For each single-position frame, split the train rows into two disjoint
    halves, fit a full affine read on each, and take the signed cosine of the
    two unit read covectors. This is the cosine a GENUINELY shared read
    achieves under fit noise at fixed position/predicate — the missing baseline
    for COS_SHARED. Returns the mean cosine across positions."""
    cosines = []
    for f in single_frames:
        tr = f["tr"]
        half = len(tr) // 2
        ta, tb = tr[:half], tr[half:]
        y = f["pphi"][target]
        wa, _ = IV.affine_readout(f["Rc"][ta], y[ta], LAM)
        wb, _ = IV.affine_readout(f["Rc"][tb], y[tb], LAM)
        cosines.append(float(IV.unit(wa) @ IV.unit(wb)))
    return float(np.mean(cosines))


def target_measures(disc_f, held_f, single_frames, target, rng, d):
    yd = disc_f["pphi"][target]
    yh = held_f["pphi"][target]

    # (0) exp-31 in-place premise: disc + held in-place readability and floor.
    wd, bd = fit_affine(disc_f, yd)
    r2_inplace_disc = r2_test(disc_f, yd, wd, bd)
    wh, bh = fit_affine(held_f, yh)
    r2_inplace_held = r2_test(held_f, yh, wh, bh)
    yshuf = yh.copy()
    rng.shuffle(yshuf)
    ws, bs = fit_affine(held_f, yshuf)
    r2_shuffle_held = r2_test(held_f, yshuf, ws, bs)

    # (1) gain/bias-refit discriminator: freeze disc direction, rescale at held.
    r2_refit_held = refit_gain_bias(held_f, yh, wd)
    pooled = pooled_read_per_position(single_frames, target)

    # (2) cosine: exp-31 cross cosine (continuity) + within-position ceiling.
    cos_cross = float(IV.unit(wd) @ IV.unit(wh))
    cos_ceiling = cosine_ceiling(single_frames, target)

    return {
        "std_disc": float(yd.std()),
        "std_held": float(yh.std()),
        "r2_inplace_disc": float(r2_inplace_disc),
        "r2_inplace_held": float(r2_inplace_held),
        "r2_shuffle_held": float(r2_shuffle_held),
        "r2_refit_held": float(r2_refit_held),
        "r2_pooled_min": float(min(pooled)),
        "r2_pooled_mean": float(np.mean(pooled)),
        "pooled_per_pos": [float(x) for x in pooled],
        "cos_cross": cos_cross,
        "cos_ceiling": cos_ceiling,
        "noise_floor": float(1.0 / np.sqrt(d)),
    }


# ---------------------------------------------------------------------------
# verdict partition (registered) — exactly one branch per (target, seed)
# ---------------------------------------------------------------------------

SHARED_WITH_DRIFT = "SHARED_WITH_DRIFT"
POSITION_SPECIFIC_CONFIRMED = "POSITION_SPECIFIC_CONFIRMED"
PREMISE_NOT_REPRODUCED = "PREMISE_NOT_REPRODUCED"

# Reported cosine-instrument overlay states (NOT verdict branches).
INSTRUMENT_UNRELIABLE = "UNRELIABLE"
INSTRUMENT_BELOW_BAR = "BELOW_SHARING_BAR"
INSTRUMENT_SHARP = "SHARP"


def recovers_held(m):
    """Did the frozen disc direction, recalibrated, recover held in-place R2?"""
    return (m["r2_refit_held"] >= R2_MIN
            and m["r2_refit_held"] - m["r2_shuffle_held"] >= FLOOR_MARGIN)


def classify_target(m):
    """Exactly one branch per (target, seed). Readability claim only.

    The verdict is decided by the gain/bias REFIT alone (the direct, well-
    controlled shared-vs-specific test); the cosine ceiling is a reported
    overlay (``cosine_instrument``), never a verdict input. Precedence:
    (1) exp-31 premise reproduction gate (vacuity + in-place disc/held
    readability vs the shuffle floor); then (2) ``recovers`` -> the frozen disc
    direction, just rescaled, decodes held -> one shared read with per-position
    drift; (3) ``not recovers`` -> rescaling does not decode held -> a genuinely
    different (position-specific) read direction."""
    # (1) exp-31 premise reproduction gate (vacuity + in-place readability).
    if m["std_disc"] < VAR_MIN or m["std_held"] < VAR_MIN:
        return PREMISE_NOT_REPRODUCED
    if m["r2_inplace_disc"] < R2_MIN:
        return PREMISE_NOT_REPRODUCED
    if (m["r2_inplace_held"] < R2_MIN
            or m["r2_inplace_held"] - m["r2_shuffle_held"] < FLOOR_MARGIN):
        return PREMISE_NOT_REPRODUCED

    # (2)/(3) refit-primary decision.
    return SHARED_WITH_DRIFT if recovers_held(m) else POSITION_SPECIFIC_CONFIRMED


def cosine_instrument(m):
    """Reported reliability overlay on the cosine sharing test (NOT a verdict
    input), modeled on exp 31's positions_exchangeable overlay. It answers
    "was exp-31's cosine test even a valid sharing instrument here?":

    - UNRELIABLE       (cos_ceiling < CEILING_MIN): same-position/same-predicate
      fits do not agree above noise -> the cosine carried no sharing signal.
    - BELOW_SHARING_BAR ([CEILING_MIN, COS_SHARED)): genuinely shared fits agree
      with themselves but cannot reach COS_SHARED -> exp-31's cosine test was
      rigged toward "specific" and could not have called "shared".
    - SHARP            (>= COS_SHARED): a shared read reaches COS_SHARED, so a
      near-zero cross cosine would be real direction specificity."""
    if m["cos_ceiling"] < CEILING_MIN:
        return INSTRUMENT_UNRELIABLE
    if m["cos_ceiling"] < COS_SHARED:
        return INSTRUMENT_BELOW_BAR
    return INSTRUMENT_SHARP


def aggregate(values):
    """Seed-majority of this experiment's local branch labels (>=3/4), else
    SEED_UNSTABLE. Thin wrapper over the shared ``battery.majority_vote``; the
    branch vocabulary stays local to this experiment."""
    return majority_vote(list(values), threshold=SEED_MAJORITY,
                         unstable="SEED_UNSTABLE")


def decide(target_aggs):
    parts = [f"{t}={target_aggs[t]}" for t in TARGETS]
    return "GATE(" + ", ".join(parts) + ")"


ROUTING = {
    SHARED_WITH_DRIFT:
        "frozen disc direction + recalibrated scale recovers held R2 -> one "
        "shared read with per-position drift, so exp-31's position-specific call "
        "was a fit-underdetermination artifact -> DROP I2 position-conditioned "
        "reads; re-ask I1 with a recalibrated transport-valid read.",
    POSITION_SPECIFIC_CONFIRMED:
        "recalibrating the frozen disc direction does NOT recover held R2 -> the "
        "held read is a genuinely different direction -> proceed to I2 with "
        "position-conditioned reads.",
    PREMISE_NOT_REPRODUCED:
        "exp-31 in-place readability does not reproduce (vacuous or in-place "
        "R2 below floor) -> fix substrate/measurement before interpreting.",
    "SEED_UNSTABLE":
        "branch not reproduced across seeds -> the gate is not stably "
        "answerable on this substrate.",
}


# ---------------------------------------------------------------------------
# selftest (no model) — adversarial known-answer fixtures
# ---------------------------------------------------------------------------

def selftest():
    IV._selftest()
    P._selftest()

    # numeric helpers: known-answer fixtures.
    rng = np.random.default_rng(0)
    d = 6
    wd = IV.unit(rng.standard_normal(d))
    R = rng.standard_normal((120, d))
    Rc = R - R.mean(0)
    idx = rng.permutation(120)
    frame = {"Rc": Rc, "tr": idx[:60], "te": idx[60:]}
    # y is an exact affine function of the frozen-direction projection ->
    # gain/bias refit must recover ~1.
    y = 3.0 * (Rc @ wd) + 5.0
    assert refit_gain_bias(frame, y, wd) > 0.999, refit_gain_bias(frame, y, wd)
    # y orthogonal to wd (depends on a different axis) -> refit cannot recover.
    w_other = IV.unit(rng.standard_normal(d))
    w_other = w_other - (w_other @ wd) * wd
    w_other = IV.unit(w_other)
    y_orth = 3.0 * (Rc @ w_other) + 1.0
    assert refit_gain_bias(frame, y_orth, wd) < 0.10, \
        refit_gain_bias(frame, y_orth, wd)

    # cosine ceiling: a strong shared read -> high ceiling; pure noise -> ~floor.
    def mkframe(wt, noise):
        Rr = rng.standard_normal((160, d))
        Rcc = Rr - Rr.mean(0)
        yy = (Rcc @ wt if wt is not None else 0.0) \
            + noise * rng.standard_normal(160)
        ix = rng.permutation(160)
        return {"Rc": Rcc, "tr": ix[:80], "te": ix[80:], "pphi": {"phiX": yy}}

    ws = IV.unit(rng.standard_normal(d))
    shared = [mkframe(ws, 0.05) for _ in range(4)]
    assert cosine_ceiling(shared, "phiX") > 0.9, cosine_ceiling(shared, "phiX")
    noisy = [mkframe(None, 1.0) for _ in range(4)]
    assert cosine_ceiling(noisy, "phiX") < CEILING_MIN, \
        cosine_ceiling(noisy, "phiX")
    pooled = pooled_read_per_position(shared, "phiX")
    assert min(pooled) > 0.9, pooled        # one read decodes every position

    # verdict partition: refit-driven 3-branch, independent of cos_ceiling.
    base = {
        "std_disc": 0.2, "std_held": 0.2,
        "r2_inplace_disc": 0.7, "r2_inplace_held": 0.7,
        "r2_shuffle_held": 0.0,
        "r2_refit_held": 0.65,          # recovers
        "cos_ceiling": 0.60,            # overlay only; must not affect verdict
        "cos_cross": 0.1,
    }

    def m(**kw):
        x = dict(base)
        x.update(kw)
        return x

    # SHARED_WITH_DRIFT iff the refit recovers held R2 — and the cosine ceiling
    # is irrelevant to the verdict (forced across all three instrument states).
    assert classify_target(m()) == SHARED_WITH_DRIFT
    for cc in (0.10, 0.60, 0.95):
        assert classify_target(m(cos_ceiling=cc)) == SHARED_WITH_DRIFT, cc
    # POSITION_SPECIFIC_CONFIRMED iff the refit does NOT recover — again
    # independent of the cosine ceiling.
    for cc in (0.10, 0.60, 0.95):
        assert classify_target(m(r2_refit_held=0.2, cos_ceiling=cc)) == \
            POSITION_SPECIFIC_CONFIRMED, cc
    # refit "recovers" must also beat the shuffle floor by FLOOR_MARGIN.
    assert classify_target(m(r2_refit_held=0.55, r2_shuffle_held=0.50)) == \
        POSITION_SPECIFIC_CONFIRMED
    # PREMISE_NOT_REPRODUCED: vacuous, disc premise gone, in-place held gone.
    assert classify_target(m(std_held=0.01)) == PREMISE_NOT_REPRODUCED
    assert classify_target(m(std_disc=0.01)) == PREMISE_NOT_REPRODUCED
    assert classify_target(m(r2_inplace_disc=0.2)) == PREMISE_NOT_REPRODUCED
    assert classify_target(m(r2_inplace_held=0.2)) == PREMISE_NOT_REPRODUCED
    assert classify_target(m(r2_inplace_held=0.55,
                             r2_shuffle_held=0.50)) == PREMISE_NOT_REPRODUCED
    # premise gate precedes everything, even a "clean" recovering refit.
    assert classify_target(m(std_held=0.01, r2_refit_held=0.9,
                             cos_ceiling=0.9)) == PREMISE_NOT_REPRODUCED

    # cosine-instrument overlay: each of the three states is forced.
    assert cosine_instrument(m(cos_ceiling=0.30)) == INSTRUMENT_UNRELIABLE
    assert cosine_instrument(m(cos_ceiling=0.60)) == INSTRUMENT_BELOW_BAR
    assert cosine_instrument(m(cos_ceiling=0.85)) == INSTRUMENT_SHARP
    # boundary: CEILING_MIN is reliable; COS_SHARED is sharp.
    assert cosine_instrument(m(cos_ceiling=CEILING_MIN)) == INSTRUMENT_BELOW_BAR
    assert cosine_instrument(m(cos_ceiling=COS_SHARED)) == INSTRUMENT_SHARP

    # aggregation: majority and instability.
    assert aggregate([SHARED_WITH_DRIFT] * 3 + [POSITION_SPECIFIC_CONFIRMED]) \
        == SHARED_WITH_DRIFT
    assert aggregate([SHARED_WITH_DRIFT, SHARED_WITH_DRIFT,
                      POSITION_SPECIFIC_CONFIRMED,
                      PREMISE_NOT_REPRODUCED]) == "SEED_UNSTABLE"

    # decision string lists both load-bearing targets, no precedence.
    dec = decide({"phi1_next_closes": SHARED_WITH_DRIFT,
                  "phi2_net_return": POSITION_SPECIFIC_CONFIRMED})
    assert dec == ("GATE(phi1_next_closes=SHARED_WITH_DRIFT, "
                   "phi2_net_return=POSITION_SPECIFIC_CONFIRMED)"), dec

    print("read-transport-discriminator selftest passed: refit recovery, "
          "cosine ceiling, pooled read, verdict partition, aggregation")


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
    # Accelerator: MPS on Apple silicon (else CUDA) when available, CPU fallback.
    # The living eval primitives (midstream.stream_to / chain_run) follow the
    # model's device and return CPU tensors, so this changes nothing for the
    # CPU-loaded frozen scripts; only this script moves the model.
    device = pick_device()
    model = model.to(device)
    masks = P.registered_masks(proc.V, M)
    d = cfg["d_model"]

    print("=== Exp 32: pre-I2 read-transport discriminator (diagnostic, no patch) ===")
    print(f"device={device}")
    print(f"target={proc.name} m={M} LAYER={LAYER} seeds={SEEDS}")
    print(f"grouped disc positions={TS_DISC}; grouped held positions={TS_HELDOUT}")
    print(f"single-position bins={SINGLE_POSITIONS}")
    print("targets:", ", ".join(TARGETS))
    print("controls (read-only):", ", ".join(CONTROL_PREDICATES))
    print(f"thresholds: R2_MIN={R2_MIN} VAR_MIN={VAR_MIN} "
          f"FLOOR_MARGIN={FLOOR_MARGIN} COS_SHARED={COS_SHARED} "
          f"CEILING_MIN={CEILING_MIN} LAM={LAM}")
    print(f"noise floor 1/sqrt(d) = {1.0/np.sqrt(d):.3f}")
    print("Observable-only: no patch, no exact-truth audit, no intervention.\n")

    per_seed_verdict = {t: [] for t in TARGETS}
    per_seed_instr = {t: [] for t in TARGETS}
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
            mm = target_measures(disc_f, held_f, single_frames, target, rng, d)
            verdict = classify_target(mm)
            instr = cosine_instrument(mm)
            per_seed_verdict[target].append(verdict)
            per_seed_instr[target].append(instr)
            print(f"\n  target {target}:")
            print(f"    std disc/held {mm['std_disc']:.3f}/{mm['std_held']:.3f}")
            print(f"    in-place R2 disc {mm['r2_inplace_disc']:.3f} | "
                  f"held {mm['r2_inplace_held']:.3f} "
                  f"(shuffle floor {mm['r2_shuffle_held']:.3f})")
            print(f"    [discriminator] frozen-disc gain/bias refit held R2 "
                  f"{mm['r2_refit_held']:.3f} "
                  f"(recovers={recovers_held(mm)})")
            print(f"    [pooled read] per-position in-place R2 min/mean "
                  f"{mm['r2_pooled_min']:.3f}/{mm['r2_pooled_mean']:.3f}")
            pp = ", ".join(f"t{t}:{v:.2f}"
                           for t, v in zip(SINGLE_POSITIONS, mm["pooled_per_pos"]))
            print(f"    [pooled read] per-position: {pp}")
            print(f"    -> {verdict}")
            print(f"    [overlay] cosine_instrument={instr} "
                  f"(ceiling {mm['cos_ceiling']:.3f}, cross {mm['cos_cross']:.3f}, "
                  f"noise floor {mm['noise_floor']:.3f}) "
                  f"-- reported, NOT a verdict input")
        print()

    print("[multi-seed aggregate]")
    target_aggs = {}
    for target in TARGETS:
        verdicts = per_seed_verdict[target]
        target_aggs[target] = aggregate(verdicts)
        print(f"  {target:<18} {verdicts} -> {target_aggs[target]}")
        instr = per_seed_instr[target]
        print(f"  {'':<18} cosine_instrument overlay {instr} -> "
              f"{aggregate(instr)} (reported; does not change the verdict)")

    print(f"\nDECISION: {decide(target_aggs)}")
    for target in TARGETS:
        agg = target_aggs[target]
        route = ROUTING.get(agg, "see per-target aggregate.")
        print(f"  {target}: {agg} -> {route}")
    print("\nNOTE: this is a readability/representational-geometry routing "
          "decision only. It gates whether I2 runs and in which form (drop vs "
          "keep position-conditioned reads). No writability or controllability "
          "is claimed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
