"""
I3-lite — matched activation-delta feasibility gate on pstack.

Pre-registration vehicle for the next step after exp 33. Exp 33 showed that
position-conditioned fixed-read rank-1 oblique writes fail for the two pstack
predicate targets despite readable in-place predicates, full-patch room,
calibrated endpoints, and low same-read/random controls. Before spending a full
experiment on optimizer-heavy learned read/write pairs, this gate asks whether
observed, near-manifold residual deltas can move the predicates at
interpolation strength; extrapolated-only control is reported separately.

Claim-producing command, after preregistration review only:

    python3 scripts/interventions/i3_matched_delta_gate.py \
        --outdir out/pstack-L4 | tee out/exp34_pstack-L4.txt

Review-only checks:

    python3 scripts/interventions/i3_matched_delta_gate.py --selftest
    python3 -m py_compile interventions.py intervention_eval.py \
        scripts/interventions/i3_matched_delta_gate.py

Exact predicate truth is endpoint-audit only. Matching, donor selection, dose
selection, predicate scores, specificity, and m-gram movement use observable
model completion probabilities.
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
from battery import first_precedence, majority_vote
from discover import PairSet, self_checks
from expcommon import LAYER, load_model
from model import pick_device
from processes import PROCESSES


REGISTERED_CFG = {"process": "pstack", "seq_len": 40, "burn_in": 4,
                  "d_model": 64, "layers": 4, "m": 3, "seed": 0}
M = 3
TARGETS = ("phi1_next_closes", "phi2_net_return")
CONTROL_PREDICATES = ("phi3_all_neutral", "phi4_first_matched")
SEEDS = (500, 501, 502, 503)
TS_DISC = (10, 18)
TS_HELDOUT = (26, 34)
PAIRS_DISC, PAIRS_HELDOUT, PAIR_POOL = 512, 1024, 900
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)

ELIGIBLE_FRAC = 0.35
PBIN_COUNT = 6
MIN_ELIGIBLE_PER_BIN = 24

VAR_MIN = 0.05
C_MIN = 0.50
C_MARGIN = 0.20
RETENTION_MIN = 0.50
SPEC_MAX = 0.35
SPEC_ROOM_MIN = 0.01
MGRAM_MAX = 0.85
INTERP_ALPHA_MAX = 1.0
OE_BAND = 0.10
SEED_MAJORITY = 3

POSITIVE = "MATCHED_DELTA_CONTROL"
DECISION_PRECEDENCE = (
    "OBS_EXACT_DRIFT",
    POSITIVE,
    "BROAD_MGRAM_REPLACEMENT",
    "MGRAM_UNAUDITABLE",
    "EXTRAPOLATED_DELTA_CONTROL",
    "NONSPECIFIC_DELTA",
    "DISCOVERY_ONLY_DELTA",
    "SHUFFLED_MATCHED_CONTROL",
    "NO_MATCHED_DELTA_CONTROL",
    "DELTA_GATE_INVALID",
    "NO_PATCH_ROOM",
    "LOW_MATCH_SUPPORT",
    "TARGET_VACUOUS",
)


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
    """Registered donor rule for matched/mismatched/shuffled controls."""
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
        "diff": diff,
        "absdiff": absdiff,
    }


def group_locs(idx, sel):
    # PairSet groups store row ids sorted increasingly, so searchsorted gives
    # the local row inside ps.pref_* and ps.Xc_* for that position group.
    return np.searchsorted(idx, sel)


def run_q_endpoint(ps, model, eligible, mode):
    out = np.full((ps.n, ps.C), np.nan)
    for t, idx in ps.groups:
        sel = idx[eligible[idx]]
        if len(sel) == 0:
            continue
        loc = group_locs(idx, sel)
        if mode == "source":
            X = ps.Xc_src[t][loc]
            pref = None
        elif mode == "target":
            X = ps.Xc_tgt[t][loc]
            pref = None
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
        loc_i = pair_loc[sel]
        loc_j = pair_loc[donors]
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
        return ctl, mgram_control(ep, q), q

    curve = []
    for alpha in alphas:
        ctl, mg, _ = at_alpha(alpha)
        curve.append((alpha, float(ctl), float(mg)))
    finite = [(ctl, alpha) for alpha, ctl, _ in curve if np.isfinite(ctl)]
    if not finite:
        return {"alpha": float("nan"), "control": float("nan"),
                "mgram_control": float("nan"), "q": ep["q_un"],
                "curve": curve}
    _, best_alpha = max(finite, key=lambda x: x[0])
    ctl, mg, q = at_alpha(best_alpha)
    return {"alpha": float(best_alpha), "control": float(ctl),
            "mgram_control": float(mg), "q": q, "curve": curve}


def specificity_predicates(masks, target, eps, room_min):
    included, skipped = [], []
    for name in masks:
        if name == target:
            continue
        room = eps[name]["room"]
        if np.isfinite(room) and room >= room_min:
            included.append(name)
        else:
            skipped.append(name)
    return included, skipped


def specificity_from_q(masks, target, eps, q_patch, spec_names):
    vals = []
    for name in spec_names:
        keep = np.asarray(masks[name], dtype=bool)
        p_patch = q_patch[:, keep].sum(axis=1)
        ep = eps[name]
        ctl = IV.predicate_control(ep["p_un"], ep["p_src"], p_patch,
                                   ep["p_full"])
        if np.isfinite(ctl):
            vals.append(abs(float(ctl)))
    return max(vals) if vals else 0.0


def classify_target(m):
    if m["std_disc"] < VAR_MIN or m["std_heldout"] < VAR_MIN:
        return "TARGET_VACUOUS"
    if (m["min_eligible_disc"] < MIN_ELIGIBLE_PER_BIN or
            m["min_eligible_heldout"] < MIN_ELIGIBLE_PER_BIN or
            not m["support_ok_disc"] or not m["support_ok_heldout"]):
        return "LOW_MATCH_SUPPORT"
    if m["room_disc"] <= IV.ROOM_TOL or m["room_heldout"] <= IV.ROOM_TOL:
        return "NO_PATCH_ROOM"
    if m["oe_disc"] > OE_BAND or m["oe_heldout"] > OE_BAND:
        return "OBS_EXACT_DRIFT"
    if (not np.isfinite(m["own_disc"]) or m["own_disc"] < C_MIN or
            not np.isfinite(m["own_heldout"]) or m["own_heldout"] < C_MIN):
        return "DELTA_GATE_INVALID"
    if not np.isfinite(m["matched_disc"]) or m["matched_disc"] < C_MIN:
        return "NO_MATCHED_DELTA_CONTROL"
    if (not np.isfinite(m["matched_heldout"]) or
            m["matched_heldout"] < C_MIN or
            not np.isfinite(m["retention"]) or
            m["retention"] < RETENTION_MIN):
        return "DISCOVERY_ONLY_DELTA"
    if m["specificity"] > SPEC_MAX:
        return "NONSPECIFIC_DELTA"
    if not np.isfinite(m["mgram_heldout"]):
        return "MGRAM_UNAUDITABLE"
    if m["mgram_heldout"] > MGRAM_MAX:
        return "BROAD_MGRAM_REPLACEMENT"
    if (m["matched_alpha_disc"] > INTERP_ALPHA_MAX or
            m["matched_alpha_heldout"] > INTERP_ALPHA_MAX):
        return "EXTRAPOLATED_DELTA_CONTROL"
    if (m["matched_disc"] - max(m["mismatched_disc"], m["shuffled_disc"]) <
            C_MARGIN or
            m["matched_heldout"] - max(m["mismatched_heldout"],
                                       m["shuffled_heldout"]) < C_MARGIN):
        return "SHUFFLED_MATCHED_CONTROL"
    return POSITIVE


def family_rows(target, scored_disc, scored_held, specificity, verdict):
    rows = []
    for fam in ("own_delta", "matched_delta", "mismatched_delta",
                "shuffled_delta"):
        h = scored_held[fam]
        d = scored_disc[fam]
        rows.append({
            "target": target,
            "read": "none",
            "write": fam,
            "alpha_disc": d["alpha_disc"],
            "alpha_heldout": h["alpha_heldout"],
            "patch_point": f"L{LAYER}/disc{TS_DISC}->hold{TS_HELDOUT}",
            "room": h["room_heldout"],
            "control": h["control_heldout"],
            "mgram_control": h["mgram_heldout"],
            "specificity": specificity if fam == "matched_delta" else None,
            "exact_audit": h["oe_heldout"],
            "transfer": (h["control_heldout"] / d["control_disc"]
                         if d["control_disc"] and
                         np.isfinite(d["control_disc"]) else float("nan")),
            "failure_branch": verdict if fam == "matched_delta" else "-",
        })
    return rows


def run_target(model, proc, cfg, seed, masks, target):
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + 111, PAIR_POOL,
                   layer=LAYER, ts=TS_DISC)
    held = PairSet(model, proc, cfg, PAIRS_HELDOUT, M, seed + 222, PAIR_POOL,
                   layer=LAYER, ts=TS_HELDOUT)
    self_checks(model, disc, LAYER, M, proc.V)
    self_checks(model, held, LAYER, M, proc.V)

    ep_d_all = EV.endpoints(disc, model, proc, masks[target], cfg["d_model"])
    ep_h_all = EV.endpoints(held, model, proc, masks[target], cfg["d_model"])
    match_d = build_delta_matching(disc, ep_d_all["p_un"], ep_d_all["p_src"],
                                   seed + 333)
    match_h = build_delta_matching(held, ep_h_all["p_un"], ep_h_all["p_src"],
                                   seed + 444)
    elig_d, elig_h = match_d["eligible"], match_h["eligible"]

    eps_d = {name: endpoint_bundle(disc, model, proc, msk, elig_d)
             for name, msk in masks.items()}
    eps_h = {name: endpoint_bundle(held, model, proc, msk, elig_h)
             for name, msk in masks.items()}
    ep_d, ep_h = eps_d[target], eps_h[target]
    spec_names, spec_skipped = specificity_predicates(
        masks, target, eps_h, SPEC_ROOM_MIN)

    scored_d, scored_h = {}, {}
    for fam in ("own_delta", "matched_delta", "mismatched_delta",
                "shuffled_delta"):
        sd = score_delta(disc, model, masks[target], ep_d, match_d[fam],
                         elig_d, ALPHAS)
        sh = score_delta(held, model, masks[target], ep_h, match_h[fam],
                         elig_h, ALPHAS)
        scored_d[fam] = {
            "alpha_disc": sd["alpha"], "control_disc": sd["control"],
            "mgram_disc": sd["mgram_control"], "room_disc": ep_d["room"],
            "oe_disc": ep_d["oe"],
        }
        scored_h[fam] = {
            "alpha_heldout": sh["alpha"], "control_heldout": sh["control"],
            "mgram_heldout": sh["mgram_control"],
            "room_heldout": ep_h["room"], "oe_heldout": ep_h["oe"],
            "q": sh["q"],
        }

    matched_h = scored_h["matched_delta"]
    spec = specificity_from_q(masks, target, eps_h, matched_h["q"], spec_names)
    matched_disc = scored_d["matched_delta"]["control_disc"]
    matched_held = scored_h["matched_delta"]["control_heldout"]
    retention = (matched_held / matched_disc
                 if matched_disc and np.isfinite(matched_disc)
                 else float("nan"))
    summary = {
        "eligible_disc": int(elig_d.sum()),
        "eligible_heldout": int(elig_h.sum()),
        "min_eligible_disc": min(match_d["eligible_counts"]),
        "min_eligible_heldout": min(match_h["eligible_counts"]),
        "support_ok_disc": match_d["support_ok"],
        "support_ok_heldout": match_h["support_ok"],
        "std_disc": ep_d["std"], "std_heldout": ep_h["std"],
        "room_disc": ep_d["room"], "room_heldout": ep_h["room"],
        "oe_disc": ep_d["oe"], "oe_heldout": ep_h["oe"],
        "own_disc": scored_d["own_delta"]["control_disc"],
        "own_heldout": scored_h["own_delta"]["control_heldout"],
        "matched_disc": matched_disc, "matched_heldout": matched_held,
        "matched_alpha_disc": scored_d["matched_delta"]["alpha_disc"],
        "matched_alpha_heldout": scored_h["matched_delta"]["alpha_heldout"],
        "mismatched_disc": scored_d["mismatched_delta"]["control_disc"],
        "mismatched_heldout": scored_h["mismatched_delta"]["control_heldout"],
        "shuffled_disc": scored_d["shuffled_delta"]["control_disc"],
        "shuffled_heldout": scored_h["shuffled_delta"]["control_heldout"],
        "mgram_heldout": matched_h["mgram_heldout"],
        "specificity": spec, "retention": retention,
        "spec_included": tuple(spec_names),
        "spec_skipped_low_room": tuple(spec_skipped),
    }
    summary["verdict"] = classify_target(summary)
    rows = family_rows(target, scored_d, scored_h, spec, summary["verdict"])
    return summary, rows


def aggregate(verdicts):
    return majority_vote(list(verdicts), threshold=SEED_MAJORITY,
                         unstable="SEED_UNSTABLE")


def decide(aggregates):
    label, keys = first_precedence(aggregates, DECISION_PRECEDENCE)
    if label is None:
        return "SEED_UNSTABLE"
    return f"{label}(" + ",".join(keys) + ")"


def selftest():
    IV._selftest()
    EV._selftest()
    P._selftest()
    base = {
        "std_disc": 0.2, "std_heldout": 0.2,
        "eligible_disc": 200, "eligible_heldout": 300,
        "min_eligible_disc": 100, "min_eligible_heldout": 150,
        "support_ok_disc": True, "support_ok_heldout": True,
        "room_disc": 0.1, "room_heldout": 0.1,
        "oe_disc": 0.01, "oe_heldout": 0.01,
        "own_disc": 0.9, "own_heldout": 0.9,
        "matched_disc": 0.75, "matched_heldout": 0.70,
        "matched_alpha_disc": 1.0, "matched_alpha_heldout": 1.0,
        "mismatched_disc": 0.1, "mismatched_heldout": 0.1,
        "shuffled_disc": 0.1, "shuffled_heldout": 0.1,
        "mgram_heldout": 0.4, "specificity": 0.1, "retention": 0.93,
    }

    def m(**kw):
        x = dict(base); x.update(kw); return x

    assert classify_target(m()) == POSITIVE
    assert classify_target(m(std_disc=0.01)) == "TARGET_VACUOUS"
    assert classify_target(m(min_eligible_disc=4)) == "LOW_MATCH_SUPPORT"
    assert classify_target(m(support_ok_disc=False)) == "LOW_MATCH_SUPPORT"
    assert classify_target(m(room_heldout=0.0)) == "NO_PATCH_ROOM"
    assert classify_target(m(oe_heldout=0.2)) == "OBS_EXACT_DRIFT"
    assert classify_target(m(own_heldout=0.2)) == "DELTA_GATE_INVALID"
    assert classify_target(m(matched_disc=0.2)) == "NO_MATCHED_DELTA_CONTROL"
    assert classify_target(m(matched_heldout=0.2)) == "DISCOVERY_ONLY_DELTA"
    assert classify_target(m(retention=0.2)) == "DISCOVERY_ONLY_DELTA"
    assert classify_target(m(specificity=0.8)) == "NONSPECIFIC_DELTA"
    assert classify_target(m(mgram_heldout=float("nan"))) == "MGRAM_UNAUDITABLE"
    assert classify_target(m(mgram_heldout=0.95)) == "BROAD_MGRAM_REPLACEMENT"
    assert classify_target(m(matched_alpha_heldout=1.5)) == \
        "EXTRAPOLATED_DELTA_CONTROL"
    assert classify_target(m(shuffled_heldout=0.6)) == \
        "SHUFFLED_MATCHED_CONTROL"
    assert aggregate([POSITIVE] * 3 + ["NO_MATCHED_DELTA_CONTROL"]) == POSITIVE
    assert aggregate([POSITIVE, "A", "B", "C"]) == "SEED_UNSTABLE"
    assert decide({"phi1": POSITIVE}).startswith(POSITIVE)

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
    print("i3 matched-delta selftest passed: helpers, verdict partition")


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

    print("=== I3-lite: matched activation-delta feasibility gate ===")
    print(f"device={device}")
    print(f"target={proc.name} m={M} LAYER={LAYER} seeds={SEEDS}")
    print(f"discovery positions={TS_DISC}; heldout positions={TS_HELDOUT}")
    print(f"eligible rows: top {ELIGIBLE_FRAC:.0%} |p_src-p_un| per position")
    print("donor rule: same position, same predicate-diff sign, matched p_phi bin")
    print("targets:", ", ".join(TARGETS))
    print("controls:", ", ".join(CONTROL_PREDICATES))
    print("Exact p_phi is endpoint-audit only; matching/scores are observable.\n")

    per_seed = {name: [] for name in TARGETS}
    columns = ("target", "read", "write", "alpha_disc", "alpha_heldout",
               "patch_point", "room", "control", "mgram_control",
               "specificity", "exact_audit", "transfer", "failure_branch")
    for seed in SEEDS:
        print(f"[seed {seed}]")
        for target in TARGETS:
            summary, rows = run_target(model, proc, cfg, seed, masks, target)
            per_seed[target].append(summary["verdict"])
            print(f"\n  target {target}: eligible disc/held "
                  f"{summary['eligible_disc']}/{summary['eligible_heldout']} "
                  f"(min per pos {summary['min_eligible_disc']}/"
                  f"{summary['min_eligible_heldout']}); "
                  f"std disc/held {summary['std_disc']:.3f}/"
                  f"{summary['std_heldout']:.3f}; room disc/held "
                  f"{summary['room_disc']:.4f}/{summary['room_heldout']:.4f}; "
                  f"oe disc/held {summary['oe_disc']:.3f}/"
                  f"{summary['oe_heldout']:.3f}")
            print(f"  matched c disc/held "
                  f"{summary['matched_disc']:.2f}/"
                  f"{summary['matched_heldout']:.2f}; "
                  f"own held={summary['own_heldout']:.2f}; "
                  f"mismatch held={summary['mismatched_heldout']:.2f}; "
                  f"shuffle held={summary['shuffled_heldout']:.2f}; "
                  f"alpha disc/held={summary['matched_alpha_disc']:.2f}/"
                  f"{summary['matched_alpha_heldout']:.2f}; "
                  f"spec={summary['specificity']:.2f}; "
                  f"mgram={summary['mgram_heldout']:.2f}; "
                  f"ret={summary['retention']:.2f} -> {summary['verdict']}")
            print("  specificity predicates included="
                  f"{summary['spec_included']} skipped_low_room="
                  f"{summary['spec_skipped_low_room']} "
                  f"(room floor {SPEC_ROOM_MIN:.3f})")
            print(IV.format_intervention_table(rows, columns=columns))
            print()

    print("[multi-seed aggregate]")
    aggregates = {}
    for target, verdicts in per_seed.items():
        aggregates[target] = aggregate(verdicts)
        print(f"  {target:<18} {verdicts} -> {aggregates[target]}")
    decision = decide(aggregates)
    print(f"\nDECISION: {decision}")
    if decision.startswith(POSITIVE):
        print("  Matched observed activation deltas move the predicate on "
              "held-out positions with specificity, transfer, and control "
              "margins. Use this as evidence that near-manifold residual "
              "writes exist before trying to learn compact read/write pairs.")
    elif decision.startswith("NO_MATCHED_DELTA_CONTROL"):
        print("  Own source-target deltas have room, but same-sign matched "
              "observed deltas do not control the predicate. Do not spend a "
              "large learned read/write-pair run without a new diagnostic.")
    elif decision.startswith("DISCOVERY_ONLY_DELTA"):
        print("  Matched deltas work only on discovery positions or fail "
              "retention. Treat as position-specific/interchange overfit.")
    elif decision.startswith("BROAD_MGRAM_REPLACEMENT"):
        print("  Matched deltas move too much of the full m-gram distribution "
              "to count as predicate-specific control.")
    elif decision.startswith("MGRAM_UNAUDITABLE"):
        print("  Full m-gram movement could not be audited, so predicate "
              "movement cannot carry a positive matched-delta claim.")
    elif decision.startswith("EXTRAPOLATED_DELTA_CONTROL"):
        print("  Matched deltas control only at alpha > 1.0. Treat as a "
              "direction signal, not a near-manifold interpolation success.")
    elif decision.startswith("NONSPECIFIC_DELTA"):
        print("  Matched deltas move non-target predicates too strongly.")
    else:
        print("  See branch label: the feasibility gate did not establish a "
              "specific, transferable matched-delta predicate handle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
