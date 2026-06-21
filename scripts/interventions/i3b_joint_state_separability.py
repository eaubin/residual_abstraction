"""
I2-prelude (design label 2b) — joint stack-state separability diagnostic.

Exp 34 (`i3_matched_delta_gate.py`) found `NONSPECIFIC_DELTA`: near-manifold
matched activation deltas move each pstack predicate but co-move the other
predicates too, so per-target specificity failed. That experiment scored the two
stack-state predicates (`phi1_next_closes`, `phi2_net_return`) as *mutually*
non-target, so phi1<->phi2 co-movement counted as a specificity failure.

This experiment tests the alternative reading: phi1 and phi2 are two facets of a
single stack-state variable, so their co-movement is *correct joint control*, and
the real specificity question is whether the same near-manifold move spares an
out-of-bundle predicate (`phi4_first_matched`, a within-window binding predicate)
and avoids broad full-m-gram replacement. It reuses the exp-34 near-manifold
delta machinery unchanged (no optimizer, no learned write) and only re-adjudicates
the same moves under a bundle vs out-of-bundle split.

Claim-producing command, after preregistration review only:

    python3 scripts/interventions/i3b_joint_state_separability.py \
        --outdir out/pstack-L4 | tee out/exp35_pstack-L4.txt

Review-only checks:

    python3 scripts/interventions/i3b_joint_state_separability.py --selftest
    python3 -m py_compile interventions.py intervention_eval.py \
        scripts/interventions/i3b_joint_state_separability.py

Exact predicate truth is endpoint-audit only. Matching, donor selection, dose
selection, predicate scores, the bundle/out-of-bundle split, and m-gram movement
use observable model completion probabilities. The exp-34 delta helpers are
copied inline (exp 34 is a frozen record and must not be imported from); a later
consolidation may promote the matched-delta harness into intervention_eval.py.
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

import intervention_eval as EV
import interventions as IV
import predicates as P
from abstraction import kl_rows
from battery import majority_vote
from discover import PairSet, self_checks
from expcommon import LAYER, load_model
from model import pick_device
from processes import PROCESSES


REGISTERED_CFG = {"process": "pstack", "seq_len": 40, "burn_in": 4,
                  "d_model": 64, "layers": 4, "m": 3, "seed": 0}
M = 3
BUNDLE = ("phi1_next_closes", "phi2_net_return")
OOB = "phi4_first_matched"
VACUOUS = "phi3_all_neutral"
SCORED = BUNDLE + (OOB,)
SEEDS = (600, 601, 602, 603)
TS_DISC = (10, 18)
TS_HELDOUT = (26, 34)
PAIRS_DISC, PAIRS_HELDOUT, PAIR_POOL = 512, 1024, 900
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
LAM = 1e-2

ELIGIBLE_FRAC = 0.35
PBIN_COUNT = 6
MIN_ELIGIBLE_PER_BIN = 24

VAR_MIN = 0.05
C_MIN = 0.50
C_MARGIN = 0.20
RETENTION_MIN = 0.50
COUPLE_MIN = 0.40
OOB_MAX = 0.35
SEP_MARGIN = 0.20
OOB_ROOM_MIN = 0.01
MGRAM_MAX = 0.85
OE_BAND = 0.10
SEED_MAJORITY = 3

POSITIVE = "JOINT_STACK_VARIABLE"
DECISION_PRECEDENCE = (
    "OBS_EXACT_DRIFT",
    "NO_OOB_ROOM",
    POSITIVE,
    "BROAD_STATE_REPLACEMENT",
    "SEPARABLE_PREDICATES",
    "NONJOINT_NONSPECIFIC",
    "NO_JOINT_CONTROL",
    "DELTA_GATE_INVALID",
    "NO_PATCH_ROOM",
    "LOW_MATCH_SUPPORT",
    "TARGET_VACUOUS",
)


# ---------------------------------------------------------------------------
# exp-34 near-manifold delta machinery (copied inline; exp 34 is frozen)
# ---------------------------------------------------------------------------

def pair_locations(ps):
    pair_t = np.empty(ps.n, dtype=np.int64)
    pair_loc = np.empty(ps.n, dtype=np.int64)
    for t, idx in ps.groups:
        pair_t[idx] = t
        pair_loc[idx] = np.arange(len(idx), dtype=np.int64)
    return pair_t, pair_loc


def pbin_labels(p_un, groups, count=PBIN_COUNT):
    labels = np.zeros(len(p_un), dtype=np.int64)
    for _, idx in groups:
        vals = p_un[idx]
        edges = np.quantile(vals, np.linspace(0.0, 1.0, count + 1)[1:-1])
        labels[idx] = np.searchsorted(edges, vals, side="right")
    return labels


def nearest_by_absdiff(i, candidates, absdiff):
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) == 0:
        return i
    return int(candidates[np.argmin(np.abs(absdiff[candidates] - absdiff[i]))])


def build_delta_matching(ps, p_un, p_src, seed):
    """Registered donor rule for matched/mismatched/shuffled controls.

    Eligibility and matching key off the predicate the delta is *built on*; the
    cross-predicate (bundle / out-of-bundle) scoring later reads the same move on
    the same eligible rows."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(p_src) - np.asarray(p_un)
    absdiff = np.abs(diff)
    sign = np.where(diff >= 0.0, 1, -1)
    pbin = pbin_labels(p_un, ps.groups)
    eligible = np.zeros(ps.n, dtype=bool)
    matched = np.arange(ps.n, dtype=np.int64)
    mismatched = np.arange(ps.n, dtype=np.int64)
    shuffled = np.arange(ps.n, dtype=np.int64)
    eligible_counts = []
    support_ok = True

    for _, idx in ps.groups:
        cut = np.quantile(absdiff[idx], 1.0 - ELIGIBLE_FRAC)
        elig = idx[absdiff[idx] >= cut]
        eligible_counts.append(int(len(elig)))
        eligible[elig] = True
        for i in elig:
            same_bin = elig[pbin[elig] == pbin[i]]
            same = same_bin[(sign[same_bin] == sign[i]) & (same_bin != i)]
            if len(same) == 0:
                same = elig[(sign[elig] == sign[i]) & (elig != i)]
            if len(same) == 0:
                support_ok = False
                same = np.array([i], dtype=np.int64)
            opp = same_bin[sign[same_bin] != sign[i]]
            if len(opp) == 0:
                opp = elig[sign[elig] != sign[i]]
            if len(opp) == 0:
                support_ok = False
                opp = np.array([i], dtype=np.int64)
            shuffle_pool = elig[elig != i]
            if len(shuffle_pool) == 0:
                support_ok = False
                shuffle_pool = np.array([i], dtype=np.int64)
            matched[i] = nearest_by_absdiff(i, same, absdiff)
            mismatched[i] = nearest_by_absdiff(i, opp, absdiff)
            shuffled[i] = int(rng.choice(shuffle_pool))

    return {
        "eligible": eligible,
        "eligible_counts": tuple(eligible_counts),
        "support_ok": bool(support_ok),
        "own_delta": np.arange(ps.n, dtype=np.int64),
        "matched_delta": matched,
        "mismatched_delta": mismatched,
        "shuffled_delta": shuffled,
    }


def group_locs(idx, sel):
    return np.searchsorted(idx, sel)


def run_q_endpoint(ps, model, eligible, mode):
    out = np.full((ps.n, ps.C), np.nan)
    for t, idx in ps.groups:
        sel = idx[eligible[idx]]
        if len(sel) == 0:
            continue
        loc = group_locs(idx, sel)
        if mode == "source":
            X, pref = ps.Xc_src[t][loc], None
        elif mode == "target":
            X, pref = ps.Xc_tgt[t][loc], None
        elif mode == "full":
            X = ps.Xc_tgt[t][loc]
            pref = ps.pref_src[t][loc].float()
        else:
            raise ValueError(mode)
        out[sel] = EV.chain_probs_only(model, X, ps.layer, pref, t, ps.m, ps.V)
    return out[eligible]


def run_q_delta(ps, model, donor_idx, alpha, eligible):
    out = np.full((ps.n, ps.C), np.nan)
    pair_t, pair_loc = pair_locations(ps)
    for t, idx in ps.groups:
        sel = idx[eligible[idx]]
        if len(sel) == 0:
            continue
        donors = donor_idx[sel]
        if not np.all(pair_t[donors] == t):
            raise ValueError("delta donor crossed position groups")
        loc_i, loc_j = pair_loc[sel], pair_loc[donors]
        X = ps.Xc_tgt[t][loc_i]
        pt = ps.pref_tgt[t][loc_i].float()
        delta = ps.pref_src[t][loc_j].float() - ps.pref_tgt[t][loc_j].float()
        pref = pt + float(alpha) * delta
        out[sel] = EV.chain_probs_only(model, X, ps.layer, pref, t, ps.m, ps.V)
    return out[eligible]


def endpoint_bundle(ps, model, proc, mask, eligible):
    q_un = run_q_endpoint(ps, model, eligible, "target")
    q_src = run_q_endpoint(ps, model, eligible, "source")
    q_full = run_q_endpoint(ps, model, eligible, "full")
    keep = np.asarray(mask, dtype=bool)
    p_un = q_un[:, keep].sum(axis=1)
    p_src = q_src[:, keep].sum(axis=1)
    p_full = q_full[:, keep].sum(axis=1)
    _, _, _, b_tgt, b_src = IV.pairset_residual_frame(ps, ps.d)
    exact_tgt = P.exact_pphi(b_tgt[eligible], mask, proc, ps.m)
    exact_src = P.exact_pphi(b_src[eligible], mask, proc, ps.m)
    d0 = float(kl_rows(q_src, q_un).mean())
    dfull = float(kl_rows(q_src, q_full).mean())
    return {
        "q_un": q_un, "q_src": q_src, "q_full": q_full,
        "p_un": p_un, "p_src": p_src, "p_full": p_full,
        "room": IV.predicate_room(p_un, p_src, p_full),
        "oe": IV.endpoint_audit(p_un, exact_tgt, p_src, exact_src),
        "mgram_room": d0 - dfull,
        "std": float(p_un.std()),
    }


def mgram_control(ep, q_patch):
    if ep["mgram_room"] <= IV.ROOM_TOL:
        return float("nan")
    d0 = float(kl_rows(ep["q_src"], ep["q_un"]).mean())
    dp = float(kl_rows(ep["q_src"], q_patch).mean())
    return (d0 - dp) / ep["mgram_room"]


def score_delta(ps, model, mask, ep, donor_idx, eligible, alphas):
    keep = np.asarray(mask, dtype=bool)

    def at_alpha(alpha):
        q = ep["q_un"] if alpha == 0.0 else run_q_delta(
            ps, model, donor_idx, alpha, eligible)
        p_patch = q[:, keep].sum(axis=1)
        ctl = IV.predicate_control(ep["p_un"], ep["p_src"], p_patch,
                                   ep["p_full"])
        return ctl, q

    curve = []
    for alpha in alphas:
        ctl, _ = at_alpha(alpha)
        curve.append((alpha, float(ctl)))
    finite = [(ctl, alpha) for alpha, ctl in curve if np.isfinite(ctl)]
    if not finite:
        return {"alpha": float("nan"), "control": float("nan"),
                "q": ep["q_un"], "curve": curve}
    _, best_alpha = max(finite, key=lambda x: x[0])
    ctl, q = at_alpha(best_alpha)
    return {"alpha": float(best_alpha), "control": float(ctl), "q": q,
            "curve": curve}


# ---------------------------------------------------------------------------
# bundle vs out-of-bundle cross scoring
# ---------------------------------------------------------------------------

def control_from_q(ep, mask, q_patch):
    """Signed closure-fraction control of an arbitrary predicate under q_patch."""
    keep = np.asarray(mask, dtype=bool)
    p_patch = q_patch[:, keep].sum(axis=1)
    return IV.predicate_control(ep["p_un"], ep["p_src"], p_patch, ep["p_full"])


def read_geometry(ps, model, masks, seed):
    """Cosines between in-place affine read directions (descriptive only)."""
    rng = np.random.default_rng(seed + 555)
    c = {n: EV.fit_inplace_read(ps, model, masks[n], M, ps.d, LAM, rng)["c"]
         for n in SCORED}
    cos = lambda a, b: float(abs(np.dot(c[a], c[b])))
    return {"cos_bundle": cos(BUNDLE[0], BUNDLE[1]),
            "cos_oob_1": cos(BUNDLE[0], OOB),
            "cos_oob_2": cos(BUNDLE[1], OOB)}


# ---------------------------------------------------------------------------
# per-(matched-on target) measurement on one seed
# ---------------------------------------------------------------------------

def measure_arm(disc, held, model, proc, masks, match_d, match_h, t):
    """Build the t-matched delta and read it on phi1, phi2, phi4 + m-gram."""
    elig_d, elig_h = match_d["eligible"], match_h["eligible"]
    eps_d = {n: endpoint_bundle(disc, model, proc, masks[n], elig_d)
             for n in SCORED}
    eps_h = {n: endpoint_bundle(held, model, proc, masks[n], elig_h)
             for n in SCORED}
    ep_d, ep_h = eps_d[t], eps_h[t]
    other = BUNDLE[1] if t == BUNDLE[0] else BUNDLE[0]

    scored = {}
    for fam in ("own_delta", "matched_delta", "mismatched_delta",
                "shuffled_delta"):
        scored[fam] = {
            "d": score_delta(disc, model, masks[t], ep_d, match_d[fam],
                             elig_d, ALPHAS),
            "h": score_delta(held, model, masks[t], ep_h, match_h[fam],
                             elig_h, ALPHAS),
        }
    qh = scored["matched_delta"]["h"]["q"]
    target_d = scored["matched_delta"]["d"]["control"]
    target_h = scored["matched_delta"]["h"]["control"]
    return {
        "min_eligible_disc": min(match_d["eligible_counts"]),
        "min_eligible_heldout": min(match_h["eligible_counts"]),
        "support_ok": match_d["support_ok"] and match_h["support_ok"],
        "std_disc": ep_d["std"], "std_heldout": ep_h["std"],
        "room_disc": ep_d["room"], "room_heldout": ep_h["room"],
        "oe_disc": ep_d["oe"], "oe_heldout": ep_h["oe"],
        "oob_room": eps_h[OOB]["room"],
        "own_heldout": scored["own_delta"]["h"]["control"],
        "target_disc": target_d, "target_heldout": target_h,
        "bundle_co_heldout": control_from_q(eps_h[other], masks[other], qh),
        "oob_heldout": control_from_q(eps_h[OOB], masks[OOB], qh),
        "mgram_heldout": mgram_control(ep_h, qh),
        "mismatched_heldout": scored["mismatched_delta"]["h"]["control"],
        "shuffled_heldout": scored["shuffled_delta"]["h"]["control"],
        "alpha_disc": scored["matched_delta"]["d"]["alpha"],
        "alpha_heldout": scored["matched_delta"]["h"]["alpha"],
        "retention": (target_h / target_d
                      if target_d and np.isfinite(target_d) else float("nan")),
    }


# ---------------------------------------------------------------------------
# joint per-seed verdict
# ---------------------------------------------------------------------------

def _all(arms, pred):
    return all(pred(a) for a in arms)


def classify_seed(arms):
    """One joint verdict from the two matched-on-bundle-predicate arms."""
    a = list(arms.values())
    if not _all(a, lambda x: x["std_disc"] >= VAR_MIN
                and x["std_heldout"] >= VAR_MIN):
        return "TARGET_VACUOUS"
    if not _all(a, lambda x: x["min_eligible_disc"] >= MIN_ELIGIBLE_PER_BIN
                and x["min_eligible_heldout"] >= MIN_ELIGIBLE_PER_BIN
                and x["support_ok"]):
        return "LOW_MATCH_SUPPORT"
    if not _all(a, lambda x: x["room_disc"] > IV.ROOM_TOL
                and x["room_heldout"] > IV.ROOM_TOL):
        return "NO_PATCH_ROOM"
    if not _all(a, lambda x: np.isfinite(x["oob_room"])
                and x["oob_room"] >= OOB_ROOM_MIN):
        return "NO_OOB_ROOM"
    if not _all(a, lambda x: x["oe_disc"] <= OE_BAND
                and x["oe_heldout"] <= OE_BAND):
        return "OBS_EXACT_DRIFT"
    if not _all(a, lambda x: np.isfinite(x["own_heldout"])
                and x["own_heldout"] >= C_MIN):
        return "DELTA_GATE_INVALID"

    target_ok = _all(a, lambda x: (
        np.isfinite(x["target_disc"]) and x["target_disc"] >= C_MIN
        and np.isfinite(x["target_heldout"]) and x["target_heldout"] >= C_MIN
        and np.isfinite(x["retention"]) and x["retention"] >= RETENTION_MIN
        and (x["target_heldout"]
             - max(x["mismatched_heldout"], x["shuffled_heldout"]))
        >= C_MARGIN))
    if not target_ok:
        return "NO_JOINT_CONTROL"

    coupled = _all(a, lambda x: np.isfinite(x["bundle_co_heldout"])
                   and x["bundle_co_heldout"] >= COUPLE_MIN)
    oob_spared = _all(a, lambda x: (
        np.isfinite(x["oob_heldout"]) and abs(x["oob_heldout"]) <= OOB_MAX
        and np.isfinite(x["mgram_heldout"]) and x["mgram_heldout"] <= MGRAM_MAX
        and (x["target_heldout"] - abs(x["oob_heldout"])) >= SEP_MARGIN))

    if coupled and oob_spared:
        return POSITIVE
    if coupled and not oob_spared:
        return "BROAD_STATE_REPLACEMENT"
    if not coupled and oob_spared:
        return "SEPARABLE_PREDICATES"
    return "NONJOINT_NONSPECIFIC"


def arm_rows(arms, verdict):
    rows = []
    for t, x in arms.items():
        rows.append({
            "matched_on": t,
            "alpha_held": x["alpha_heldout"],
            "target_ctl": x["target_heldout"],
            "bundle_co": x["bundle_co_heldout"],
            "oob_ctl": x["oob_heldout"],
            "mgram": x["mgram_heldout"],
            "own_ctl": x["own_heldout"],
            "mis": x["mismatched_heldout"],
            "shuf": x["shuffled_heldout"],
            "room": x["room_heldout"],
            "oob_room": x["oob_room"],
            "exact_audit": x["oe_heldout"],
            "seed_branch": verdict,
        })
    return rows


def run_seed(model, proc, cfg, seed, masks):
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + 111, PAIR_POOL,
                   layer=LAYER, ts=TS_DISC)
    held = PairSet(model, proc, cfg, PAIRS_HELDOUT, M, seed + 222, PAIR_POOL,
                   layer=LAYER, ts=TS_HELDOUT)
    self_checks(model, disc, LAYER, M, proc.V)
    self_checks(model, held, LAYER, M, proc.V)

    arms = {}
    for t in BUNDLE:
        ep_d = EV.endpoints(disc, model, proc, masks[t], cfg["d_model"])
        ep_h = EV.endpoints(held, model, proc, masks[t], cfg["d_model"])
        match_d = build_delta_matching(disc, ep_d["p_un"], ep_d["p_src"],
                                       seed + 333)
        match_h = build_delta_matching(held, ep_h["p_un"], ep_h["p_src"],
                                       seed + 444)
        arms[t] = measure_arm(disc, held, model, proc, masks, match_d,
                              match_h, t)
    geo = read_geometry(disc, model, masks, seed)
    verdict = classify_seed(arms)
    return arms, geo, verdict


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def selftest():
    IV._selftest()
    EV._selftest()
    P._selftest()

    def arm(**kw):
        base = {
            "std_disc": 0.2, "std_heldout": 0.2,
            "min_eligible_disc": 100, "min_eligible_heldout": 150,
            "support_ok": True, "room_disc": 0.1, "room_heldout": 0.1,
            "oob_room": 0.05, "oe_disc": 0.01, "oe_heldout": 0.01,
            "own_heldout": 0.9, "target_disc": 0.75, "target_heldout": 0.70,
            "bundle_co_heldout": 0.65, "oob_heldout": 0.05,
            "mgram_heldout": 0.4, "mismatched_heldout": 0.1,
            "shuffled_heldout": 0.1, "alpha_disc": 1.0, "alpha_heldout": 1.0,
            "retention": 0.93,
        }
        base.update(kw)
        return base

    def both(**kw):
        return {BUNDLE[0]: arm(**kw), BUNDLE[1]: arm(**kw)}

    assert classify_seed(both()) == POSITIVE
    assert classify_seed(both(std_disc=0.01)) == "TARGET_VACUOUS"
    assert classify_seed(both(min_eligible_disc=4)) == "LOW_MATCH_SUPPORT"
    assert classify_seed(both(support_ok=False)) == "LOW_MATCH_SUPPORT"
    assert classify_seed(both(room_heldout=0.0)) == "NO_PATCH_ROOM"
    assert classify_seed(both(oob_room=0.0)) == "NO_OOB_ROOM"
    assert classify_seed(both(oe_heldout=0.2)) == "OBS_EXACT_DRIFT"
    assert classify_seed(both(own_heldout=0.2)) == "DELTA_GATE_INVALID"
    assert classify_seed(both(target_heldout=0.2)) == "NO_JOINT_CONTROL"
    assert classify_seed(both(retention=0.2)) == "NO_JOINT_CONTROL"
    assert classify_seed(both(shuffled_heldout=0.6)) == "NO_JOINT_CONTROL"
    # coupled but phi4 also moves -> broad replacement
    assert classify_seed(both(oob_heldout=0.6)) == "BROAD_STATE_REPLACEMENT"
    assert classify_seed(both(mgram_heldout=0.95)) == "BROAD_STATE_REPLACEMENT"
    # phi4 spared but the other bundle predicate does not co-move -> separable
    assert classify_seed(both(bundle_co_heldout=0.1)) == "SEPARABLE_PREDICATES"
    # neither coupled nor spared
    assert classify_seed(both(bundle_co_heldout=0.1,
                              oob_heldout=0.6)) == "NONJOINT_NONSPECIFIC"
    # separation-margin failure (target barely beats phi4) -> not spared
    assert classify_seed(both(target_heldout=0.50,
                              oob_heldout=0.34)) == "BROAD_STATE_REPLACEMENT"
    # a per-seed split aggregates to unstable
    assert majority_vote([POSITIVE, POSITIVE, "BROAD_STATE_REPLACEMENT",
                          "SEPARABLE_PREDICATES"], threshold=SEED_MAJORITY,
                         unstable="SEED_UNSTABLE") == "SEED_UNSTABLE"
    assert majority_vote([POSITIVE] * 3 + ["BROAD_STATE_REPLACEMENT"],
                         threshold=SEED_MAJORITY,
                         unstable="SEED_UNSTABLE") == POSITIVE

    class Dummy:
        n = 8
        groups = [(10, np.arange(8))]
    p_un = np.array([0.1, 0.1, 0.2, 0.2, 0.7, 0.7, 0.8, 0.8])
    p_src = np.array([0.8, 0.75, 0.7, 0.65, 0.1, 0.15, 0.2, 0.25])
    mm = build_delta_matching(Dummy(), p_un, p_src, 0)
    diff = p_src - p_un
    for i in np.where(mm["eligible"])[0]:
        assert np.sign(diff[mm["matched_delta"][i]]) == np.sign(diff[i])
        assert mm["shuffled_delta"][i] != i
    print("i3b joint-separability selftest passed: helpers, joint verdict")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/pstack-L4")
    ap.add_argument("--i0-artifact", default="out/i0_preflight_pstack-L4.txt")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if not os.path.exists(args.i0_artifact):
        print(f"HALT: missing I0 preflight artifact {args.i0_artifact}")
        return 1
    with open(args.i0_artifact) as f:
        if "I1 ROUTING: GO" not in f.read():
            print(f"HALT: I0 preflight did not route to intervention work in "
                  f"{args.i0_artifact}")
            return 1

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    mism = [(k, cfg.get(k), v) for k, v in REGISTERED_CFG.items()
            if cfg.get(k) != v]
    if mism:
        print("HALT: wrong checkpoint config:", mism)
        return 1
    proc = PROCESSES[cfg["process"]]()
    device = pick_device()
    model = load_model(args.outdir, cfg, proc).to(device)
    masks = P.registered_masks(proc.V, M)

    print("=== I2-prelude: joint stack-state separability diagnostic ===")
    print(f"device={device}")
    print(f"target={proc.name} m={M} LAYER={LAYER} seeds={SEEDS}")
    print(f"discovery positions={TS_DISC}; heldout positions={TS_HELDOUT}")
    print(f"bundle={BUNDLE}; out-of-bundle control={OOB}; vacuous={VACUOUS}")
    print("intervention: exp-34 near-manifold matched deltas, re-adjudicated as "
          "joint bundle control vs out-of-bundle specificity")
    print("Exact p_phi is endpoint-audit only; matching/scores are observable.\n")

    cols = ("matched_on", "alpha_held", "target_ctl", "bundle_co", "oob_ctl",
            "mgram", "own_ctl", "mis", "shuf", "room", "oob_room",
            "exact_audit", "seed_branch")
    per_seed = []
    for seed in SEEDS:
        print(f"[seed {seed}]")
        arms, geo, verdict = run_seed(model, proc, cfg, seed, masks)
        per_seed.append(verdict)
        print(f"  read-cos bundle(phi1,phi2)={geo['cos_bundle']:.2f}; "
              f"oob(phi1,phi4)={geo['cos_oob_1']:.2f}; "
              f"oob(phi2,phi4)={geo['cos_oob_2']:.2f}")
        for t, x in arms.items():
            print(f"  [{t}] target d/h {x['target_disc']:.2f}/"
                  f"{x['target_heldout']:.2f}; bundle_co={x['bundle_co_heldout']:.2f}; "
                  f"oob={x['oob_heldout']:.2f}; mgram={x['mgram_heldout']:.2f}; "
                  f"own={x['own_heldout']:.2f}; mis/shuf "
                  f"{x['mismatched_heldout']:.2f}/{x['shuffled_heldout']:.2f}; "
                  f"alpha d/h {x['alpha_disc']:.2f}/{x['alpha_heldout']:.2f}; "
                  f"oob_room={x['oob_room']:.4f}; ret={x['retention']:.2f}")
        print(IV.format_intervention_table(arm_rows(arms, verdict), columns=cols))
        print(f"  -> seed verdict: {verdict}\n")

    print("[multi-seed aggregate]")
    agg = majority_vote(per_seed, threshold=SEED_MAJORITY,
                        unstable="SEED_UNSTABLE")
    print(f"  {per_seed} -> {agg}")
    print(f"\nDECISION: {agg}(stack_state_bundle)")
    routing = {
        POSITIVE: "  phi1 and phi2 are facets of one writable stack-state "
                  "variable: near-manifold deltas move both bundle predicates "
                  "together while sparing phi4 and avoiding broad replacement. "
                  "Exp-34's NONSPECIFIC was a target-decomposition artifact. "
                  "Carry rank-1 oblique writes forward as a JOINT write "
                  "(fit one write to the shared readout); do NOT spend on full "
                  "per-predicate I2.",
        "BROAD_STATE_REPLACEMENT":
            "  The near-manifold move also drags the out-of-bundle predicate / "
            "the full distribution: exp-34's non-specificity is genuine broad "
            "replacement, not a clean joint variable. Route to I4 patch-point "
            "(find where the bundle is separable) or consolidate.",
        "SEPARABLE_PREDICATES":
            "  phi1 and phi2 do NOT co-move under the matched delta while phi4 "
            "is spared: the entanglement hypothesis is rejected; the targets are "
            "separable. Exp-33's per-predicate write failure is a genuine "
            "write-class failure, not a decomposition artifact. Route to full I2 "
            "(read may be wrong) or I4.",
        "NONJOINT_NONSPECIFIC":
            "  The move neither couples the bundle nor spares out-of-bundle "
            "structure: broad, non-bundle-respecting. Consolidate the "
            "residual-level failure.",
        "NO_JOINT_CONTROL":
            "  Matched deltas do not reliably move both bundle predicates over "
            "the floors: the joint-control precondition fails. Do not spend on "
            "a learned joint write without a new diagnostic.",
        "NO_OOB_ROOM":
            "  The out-of-bundle predicate phi4 has no full-patch room, so the "
            "separability test is non-diagnostic. A richer toy with a separable "
            "out-of-bundle predicate is needed.",
    }
    print(routing.get(agg, "  See branch label: the joint-separability "
                            "diagnostic did not resolve cleanly."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
