"""
I1' — position-conditioned fixed-read oblique write search on pstack.

Pre-registration vehicle for INTERVENTION_CLASS_BENCHMARK.md (the fixed-read
baseline the §I2 "Pre-I2 read status" block mandates be run explicitly). This is
exp 30's I1 write search with the read repaired: instead of fitting one global
affine read on discovery positions and transporting it to held-out positions
(exp 30's `FIXED_READ_NOT_TRANSPORTED` wall, localized to a position-specific
read by exp 31 and confirmed by exp 32), I1' fits a position-conditioned read
IN PLACE per bin, so the read gate no longer blocks and the write question is
finally adjudicated.

Claim-producing command, after preregistration review only:

    python3 scripts/interventions/i1prime_poscond_read_write_search.py \
        --outdir out/pstack-L4 | tee out/exp33_pstack-L4.txt

RESULTS (see experiments/33-poscond-read-write-search.md):
NO_POSCOND_READ_WRITE_WORKS(phi1_next_closes,phi2_net_return). Repaired in-place
reads decode with room, but the registered fixed-read rank-1 oblique write menu
is a clean negative.

Review-only checks (allowed before approval):

    python3 scripts/interventions/i1prime_poscond_read_write_search.py --selftest
    python3 -m py_compile interventions.py intervention_eval.py \
        scripts/interventions/i1prime_poscond_read_write_search.py

Built from the living helpers in interventions.py (patch API, predicate-control
scorer, write-source constructors) and intervention_eval.py (the predicate-
control eval harness promoted from exp 30). Exact predicate truth is
evaluation-only endpoint audit; reads, writes, selection, and scores use
observable model completion probabilities.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import interventions as IV
import intervention_eval as EV
import predicates as P
from battery import Refs, cegar_loop, first_precedence, majority_vote
from discover import PairSet, self_checks
from expcommon import LAYER, load_model
from model import pick_device
from processes import PROCESSES


REGISTERED_CFG = {"process": "pstack", "seq_len": 40, "burn_in": 4,
                  "d_model": 64, "layers": 4, "m": 3, "seed": 0}
M, MM = 3, 3
TARGETS = ("phi1_next_closes", "phi2_net_return")
CONTROL_PREDICATES = ("phi3_all_neutral", "phi4_first_matched")
SEEDS = (400, 401, 402, 403)
TS_DISC = (10, 18)
TS_HELDOUT = (26, 34)
PAIRS_DISC, PAIRS_HELDOUT, PAIR_POOL = 512, 1024, 900
EPS, EPS_DROP, K_MAX, MAX_DIM = 0.05, 0.01, 10, 8
LAM = 1e-2
RANDOM_WRITES = 16
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
DELTA_FRAC = 0.35
LEARN_STEPS, LEARN_BATCH, LEARN_LR = 80, 64, 0.05

VAR_MIN = 0.05
R2_MIN = 0.50
C_MIN = 0.50
C_MARGIN = 0.20
RETENTION_MIN = 0.50
SPEC_MAX = 0.35
SPEC_ROOM_MIN = 0.01
OE_BAND = 0.10
SEED_MAJORITY = 3
LEARN_NORM_MAX = 1e4

POSITIVE = "POSCOND_READ_WRITE_CONTROL"

# Per-target branch precedence for the top-level routing string. Most decisive
# first; branch labels are mechanism-neutral unless directly measured.
DECISION_PRECEDENCE = (
    "OBS_EXACT_DRIFT",
    POSITIVE,
    "HELD_READ_NOT_WRITABLE",
    "DISCOVERY_ONLY_WRITE",
    "NONSPECIFIC_CONTROL",
    "RANDOM_MATCHED_CONTROL",
    "SAME_READ_BASELINE_CONTROL",
    "NO_POSCOND_READ_WRITE_WORKS",
    "NO_PATCH_ROOM",
    "READ_NOT_DECODABLE",
    "TARGET_VACUOUS",
)
DECISION_RELABEL = {}


# ---------------------------------------------------------------------------
# write-candidate construction (experiment-local; faithful to exp 30)
# ---------------------------------------------------------------------------

def add_candidate(out, name, w):
    try:
        out[name] = IV.unit(w)
    except ValueError:
        pass


def build_write_candidates(disc, model, cfg, proc, c, target_ep, seed, d):
    """Observable write-source constructors for the I1' arms, built on the
    discovery bin with the discovery in-place read ``c``: same-read/same-write,
    CEGAR-core directions + core-projected read, source-target predicate-delta
    writes, and norm-matched random writes. The learned write is added by the
    caller."""
    rng = np.random.default_rng(seed)
    out = {}
    add_candidate(out, "same_read", c)

    refs = Refs(disc, model, d, M)
    k_raw, Qcore_raw, _ = cegar_loop(model, disc, refs, d, EPS, K_MAX, MM,
                                     eps_drop=EPS_DROP)
    k_core = int(np.clip(k_raw, 1, MAX_DIM))
    Qcore = Qcore_raw[:, :k_core]
    for j, w in enumerate(IV.core_directions(Qcore)):
        add_candidate(out, f"core_b{j + 1}", w)
    add_candidate(out, "core_proj_read", Qcore @ (Qcore.T @ c))

    _, D, _, _, _ = IV.pairset_residual_frame(disc, d)
    diff = target_ep["p_src"] - target_ep["p_un"]
    qhi = np.quantile(diff, 1.0 - DELTA_FRAC)
    qlo = np.quantile(diff, DELTA_FRAC)
    top = np.abs(diff) >= np.quantile(np.abs(diff), 1.0 - DELTA_FRAC)
    if top.any():
        add_candidate(out, "delta_signed_top",
                      (np.sign(diff[top])[:, None] * D[top]).mean(0))
        add_candidate(out, "delta_top_axis", IV.delta_direction(D[top], "top"))
    if np.any(diff >= qhi):
        add_candidate(out, "delta_pos_mean", D[diff >= qhi].mean(0))
    if np.any(diff <= qlo):
        add_candidate(out, "delta_neg_mean", D[diff <= qlo].mean(0))

    for j in range(RANDOM_WRITES):
        out[f"random_{j + 1:02d}"] = IV.random_direction(rng, d)
    return out, k_core


def pair_locations(ps):
    pair_t = np.empty(ps.n, dtype=np.int64)
    pair_loc = np.empty(ps.n, dtype=np.int64)
    for t, idx in ps.groups:
        pair_t[idx] = t
        pair_loc[idx] = np.arange(len(idx), dtype=np.int64)
    return pair_t, pair_loc


def learn_write(model, disc, c_np, mask_np, p_src_np, init_w, seed, cfg):
    """Learn a write direction with the read covector fixed (faithful to exp
    30). Parameterization ``w = c + u_perp`` keeps ``c.w = 1``; the loss is
    observable predicate MSE against source-run p_phi on discovery pairs. Exact
    process labels are not used."""
    d, L = cfg["d_model"], cfg["seq_len"]
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    c = IV.unit(c_np)
    ip = float(c @ init_w)
    if abs(ip) < IV.RW_TOL:
        init_w = c.copy()
        ip = 1.0
    w0 = init_w / ip
    u0 = w0 - c
    u0 = u0 - c * float(c @ u0)
    dev = next(model.parameters()).device
    u = torch.from_numpy(u0.astype(np.float32)).to(dev).requires_grad_()
    c_t = torch.from_numpy(c.astype(np.float32)).to(dev)
    keep = np.asarray(mask_np, dtype=bool)
    psrc_t = torch.from_numpy(p_src_np.astype(np.float32)).to(dev)
    pos_all = torch.arange(L, device=dev)
    pair_t, pair_loc = pair_locations(disc)
    opt = torch.optim.Adam([u], lr=LEARN_LR)

    def current_w():
        return c_t + u - c_t * (u @ c_t)

    for _ in range(LEARN_STEPS):
        batch = rng.choice(disc.n, min(LEARN_BATCH, disc.n), replace=False)
        w_t = current_w()
        P_t = c_t[:, None] @ w_t[None, :]
        total = w_t.new_zeros(())
        for t in np.unique(pair_t[batch]):
            sel = batch[pair_t[batch] == t]
            loc = pair_loc[sel]
            Xc = torch.from_numpy(disc.Xc_tgt[t][loc][:, keep, :]).to(dev)
            bsz, C, _ = Xc.shape
            pt = disc.pref_tgt[t][loc].float().to(dev)
            delta = disc.pref_src[t][loc].float().to(dev) - pt
            ps = pt + delta @ P_t
            flat = Xc.reshape(bsz * C, L)
            ps_r = ps.repeat_interleave(C, dim=0)
            x = model.tok(flat) + model.pos(pos_all)
            for li, blk in enumerate(model.blocks):
                if li == LAYER:
                    x = torch.cat([ps_r, x[:, t + 1:]], dim=1)
                x = blk(x)
            logp = torch.log_softmax(model.head(model.ln_f(x)), dim=-1)
            rows = torch.arange(bsz * C, device=dev)
            lq = sum(logp[rows, t + j, flat[:, t + 1 + j]] for j in range(M))
            q = torch.exp(lq.reshape(bsz, C))
            pphi = q.sum(dim=1)
            total = total + ((pphi - psrc_t[sel]) ** 2).mean()
        loss = total / len(np.unique(pair_t[batch]))
        opt.zero_grad()
        loss.backward()
        opt.step()

    w = current_w().detach().cpu().numpy().astype(np.float64)
    n = float(np.linalg.norm(w))
    if not np.isfinite(n) or n > LEARN_NORM_MAX:
        raise FloatingPointError(f"learned write norm invalid: {n}")
    return IV.unit(w), n


# ---------------------------------------------------------------------------
# per-target evaluation
# ---------------------------------------------------------------------------


def finite_key(x):
    return float(x) if np.isfinite(x) else float("-inf")

def select_family(scored, prefix):
    items = [(name, m) for name, m in scored.items() if name.startswith(prefix)]
    finite = [(m["control"], name, m) for name, m in items
              if np.isfinite(m["control"])]
    return max(finite, default=(float("nan"), None, None), key=lambda x: x[0])


def family_rows(target, families):
    rows = []
    for fam, m in families.items():
        rows.append({
            "target": target,
            "read": "poscond_inplace",
            "write": fam,
            "patch_point": f"L{LAYER}/disc{TS_DISC}->hold{TS_HELDOUT}",
            "room": m["room_heldout"],
            "control": m["control_heldout"],
            "specificity": m["specificity"],
            "exact_audit": m["oe_heldout"],
            "transfer": m["retention"],
            "failure_branch": m["branch"],
        })
    return rows


def classify_target(m):
    """One branch per (target, seed). Partition mirrors exp 30 with the read
    gate repaired: in-place decodability at BOTH bins replaces the transport
    gate. Mutually exclusive and exhaustive by the precedence below."""
    if m["std_disc"] < VAR_MIN or m["std_heldout"] < VAR_MIN:
        return "TARGET_VACUOUS"
    if m["r2_disc"] < R2_MIN or m["r2_heldout"] < R2_MIN:
        return "READ_NOT_DECODABLE"
    if m["room_disc"] <= IV.ROOM_TOL or m["room_heldout"] <= IV.ROOM_TOL:
        return "NO_PATCH_ROOM"
    if m["oe_disc"] > OE_BAND or m["oe_heldout"] > OE_BAND:
        return "OBS_EXACT_DRIFT"
    if not np.isfinite(m["best_disc"]) or m["best_disc"] < C_MIN:
        return "NO_POSCOND_READ_WRITE_WORKS"
    transfer_failed = (
        not np.isfinite(m["best_heldout"]) or
        m["best_heldout"] < C_MIN or
        not np.isfinite(m["retention"]) or
        m["retention"] < RETENTION_MIN
    )
    if transfer_failed:
        if (not np.isfinite(m["best_held_inplace"]) or
                m["best_held_inplace"] < C_MIN):
            return "HELD_READ_NOT_WRITABLE"
        return "DISCOVERY_ONLY_WRITE"
    if m["specificity"] > SPEC_MAX:
        return "NONSPECIFIC_CONTROL"
    if (m["best_disc"] - m["random_disc"] < C_MARGIN or
            m["best_heldout"] - m["random_heldout"] < C_MARGIN):
        return "RANDOM_MATCHED_CONTROL"
    if m["best_heldout"] - m["same_heldout"] < C_MARGIN:
        return "SAME_READ_BASELINE_CONTROL"
    return POSITIVE


def run_target(model, proc, cfg, seed, masks, target):
    d = cfg["d_model"]
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + 111, PAIR_POOL,
                   layer=LAYER, ts=TS_DISC)
    held = PairSet(model, proc, cfg, PAIRS_HELDOUT, M, seed + 222, PAIR_POOL,
                   layer=LAYER, ts=TS_HELDOUT)
    self_checks(model, disc, LAYER, M, proc.V)
    self_checks(model, held, LAYER, M, proc.V)

    rng = np.random.default_rng(seed)
    mask = masks[target]
    # Position-conditioned reads: fit IN PLACE per bin, never transported.
    read_d = EV.fit_inplace_read(disc, model, mask, M, d, LAM, rng)
    read_h = EV.fit_inplace_read(held, model, mask, M, d, LAM, rng)
    c_disc, c_held = read_d["c"], read_h["c"]

    ep_d = EV.endpoints(disc, model, proc, mask, d)
    ep_h = EV.endpoints(held, model, proc, mask, d)
    held_eps = {target: ep_h}
    for name, msk in masks.items():
        if name != target:
            held_eps[name] = EV.endpoints(held, model, proc, msk, d)
    spec_names, spec_skipped = EV.specificity_predicates(
        masks, target, held_eps, SPEC_ROOM_MIN)

    candidates, k_core = build_write_candidates(disc, model, cfg, proc, c_disc,
                                                ep_d, seed, d)
    scored_d = {name: EV.score_write(disc, model, mask, c_disc, w, ep_d, ALPHAS)
                for name, w in candidates.items()}
    nonrandom = [n for n in scored_d if not n.startswith("random_")]
    init_name = max(nonrandom, key=lambda n: finite_key(scored_d[n]["control"]))
    learned_norm = float("nan")
    learned_failed = False
    try:
        learned_w, learned_norm = learn_write(model, disc, c_disc, mask,
                                              ep_d["p_src"],
                                              candidates[init_name], seed, cfg)
        candidates["learned_write"] = learned_w
        scored_d["learned_write"] = EV.score_write(disc, model, mask, c_disc,
                                                   learned_w, ep_d, ALPHAS)
    except FloatingPointError:
        learned_failed = True

    families = {}
    for fam, prefix in (("same_read", "same_read"),
                        ("core_best", "core_"),
                        ("delta_best", "delta_"),
                        ("learned_write", "learned_write"),
                        ("random_best", "random_")):
        _, cname, dm = select_family(scored_d, prefix)
        if cname is None:
            if fam == "learned_write" and learned_failed:
                families[fam] = {
                    "candidate": "learned_write", "alpha_disc": float("nan"),
                    "control_disc": float("nan"), "alpha_heldout": float("nan"),
                    "control_heldout": float("nan"), "room_disc": ep_d["room"],
                    "room_heldout": ep_h["room"], "specificity": float("nan"),
                    "oe_disc": ep_d["oe"], "oe_heldout": ep_h["oe"],
                    "retention": float("nan"), "branch": "LEARNED_DIVERGED",
                }
            continue
        w = candidates[cname]
        # Transfer arm: discovery-selected writes are scored on held with the
        # HELD in-place read. The same-read baseline is per-bin by construction:
        # oblique(c_disc, c_disc) on discovery and oblique(c_held, c_held) on held.
        w_held = c_held if fam == "same_read" else w
        hm = EV.score_write(held, model, mask, c_held, w_held, ep_h, ALPHAS)
        spec = EV.specificity(held, model, masks, target, c_held, w_held,
                              hm["alpha"], held_eps, spec_names)
        retention = (hm["control"] / dm["control"]
                     if dm["control"] and np.isfinite(dm["control"])
                     else float("nan"))
        branch = "LEARNED_DIVERGED" if (
            fam == "learned_write" and not np.isfinite(learned_norm)) else "-"
        families[fam] = {
            "candidate": cname, "alpha_disc": dm["alpha"],
            "control_disc": dm["control"], "alpha_heldout": hm["alpha"],
            "control_heldout": hm["control"], "room_disc": ep_d["room"],
            "room_heldout": ep_h["room"], "specificity": spec,
            "oe_disc": ep_d["oe"], "oe_heldout": ep_h["oe"],
            "retention": retention, "branch": branch,
        }

    best_pool = {k: v for k, v in families.items() if k != "random_best"}
    best_family = max(
        best_pool, key=lambda k: finite_key(best_pool[k]["control_disc"]))
    best = best_pool[best_family]
    rand = families["random_best"]
    same = families["same_read"]

    # Verdict input for transfer failures: best held-bin control by registered
    # non-random writes paired with the HELD in-place read. Random writes are
    # excluded so a no-information write cannot make the held read look writable.
    held_write_pool = [("same_read_held", c_held)] + [
        (name, w) for name, w in candidates.items()
        if not name.startswith("random_") and name != "same_read"
    ]
    held_inplace = max(
        (EV.score_write(held, model, mask, c_held, w, ep_h, ALPHAS)["control"]
         for _, w in held_write_pool),
        key=finite_key, default=float("nan"))

    summary = {
        "std_disc": read_d["std"], "std_heldout": read_h["std"],
        "r2_disc": read_d["r2"], "r2_heldout": read_h["r2"],
        "room_disc": ep_d["room"], "room_heldout": ep_h["room"],
        "oe_disc": ep_d["oe"], "oe_heldout": ep_h["oe"],
        "best_family": best_family, "best_disc": best["control_disc"],
        "best_heldout": best["control_heldout"],
        "random_disc": rand["control_disc"],
        "random_heldout": rand["control_heldout"],
        "same_heldout": same["control_heldout"],
        "specificity": best["specificity"], "retention": best["retention"],
        "best_held_inplace": float(held_inplace),
        "k_core": k_core, "learned_norm": learned_norm,
        "spec_included": tuple(spec_names),
        "spec_skipped_low_room": tuple(spec_skipped),
    }
    summary["verdict"] = classify_target(summary)
    for fam in families.values():
        if fam["branch"] == "-":
            fam["branch"] = summary["verdict"] if fam is best else "-"
    return summary, families


def control_predicate_report(model, proc, cfg, seed, masks):
    """Read controls only: in-place decodability of the two control predicates
    at both bins. Reported, never promoted to targets."""
    d = cfg["d_model"]
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + 333, PAIR_POOL,
                   layer=LAYER, ts=TS_DISC)
    held = PairSet(model, proc, cfg, PAIRS_HELDOUT, M, seed + 444, PAIR_POOL,
                   layer=LAYER, ts=TS_HELDOUT)
    rng = np.random.default_rng(seed + 999)
    out = {}
    for name in CONTROL_PREDICATES:
        rd = EV.fit_inplace_read(disc, model, masks[name], M, d, LAM, rng)
        rh = EV.fit_inplace_read(held, model, masks[name], M, d, LAM, rng)
        out[name] = {"std_disc": rd["std"], "std_heldout": rh["std"],
                     "r2_disc": rd["r2"], "r2_heldout": rh["r2"]}
    return out



# ---------------------------------------------------------------------------
# aggregation + routing (shared battery helpers)
# ---------------------------------------------------------------------------

def aggregate(verdicts):
    return majority_vote(list(verdicts), threshold=SEED_MAJORITY,
                         unstable="SEED_UNSTABLE")


def decide(aggregates):
    label, keys = first_precedence(aggregates, DECISION_PRECEDENCE)
    if label is None:
        return "SEED_UNSTABLE"
    return f"{DECISION_RELABEL.get(label, label)}(" + ",".join(keys) + ")"


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def selftest():
    IV._selftest()
    EV._selftest()
    P._selftest()
    base = {
        "std_disc": 0.2, "std_heldout": 0.2, "r2_disc": 0.7, "r2_heldout": 0.7,
        "room_disc": 0.1, "room_heldout": 0.1, "oe_disc": 0.01, "oe_heldout": 0.01,
        "best_disc": 0.8, "best_heldout": 0.7, "random_disc": 0.1,
        "random_heldout": 0.1, "same_heldout": 0.0, "specificity": 0.1,
        "retention": 0.875, "best_held_inplace": 0.7,
    }

    def m(**kw):
        x = dict(base); x.update(kw); return x
    assert classify_target(m()) == POSITIVE
    assert classify_target(m(std_disc=0.01)) == "TARGET_VACUOUS"
    assert classify_target(m(r2_disc=0.2)) == "READ_NOT_DECODABLE"
    assert classify_target(m(r2_heldout=0.2)) == "READ_NOT_DECODABLE"
    assert classify_target(m(room_heldout=0.0)) == "NO_PATCH_ROOM"
    assert classify_target(m(oe_heldout=0.2)) == "OBS_EXACT_DRIFT"
    assert classify_target(m(best_heldout=0.2)) == "DISCOVERY_ONLY_WRITE"
    assert classify_target(m(best_heldout=float("nan"))) == "DISCOVERY_ONLY_WRITE"
    assert classify_target(m(retention=0.2)) == "DISCOVERY_ONLY_WRITE"
    assert classify_target(m(retention=float("nan"))) == "DISCOVERY_ONLY_WRITE"
    assert classify_target(m(best_heldout=0.2, best_held_inplace=0.2)) == \
        "HELD_READ_NOT_WRITABLE"
    assert classify_target(
        m(best_heldout=0.2, best_held_inplace=float("nan"))) == \
        "HELD_READ_NOT_WRITABLE"
    assert classify_target(m(best_disc=0.2, best_heldout=0.2)) == \
        "NO_POSCOND_READ_WRITE_WORKS"
    assert classify_target(m(best_disc=float("nan"))) == \
        "NO_POSCOND_READ_WRITE_WORKS"
    assert classify_target(m(specificity=0.8)) == "NONSPECIFIC_CONTROL"
    assert classify_target(m(random_heldout=0.6)) == "RANDOM_MATCHED_CONTROL"
    assert classify_target(m(same_heldout=0.6)) == "SAME_READ_BASELINE_CONTROL"

    # aggregation + routing via the shared battery helpers.
    assert aggregate([POSITIVE] * 3 + ["NO_POSCOND_READ_WRITE_WORKS"]) == POSITIVE
    assert aggregate([POSITIVE, "A", "B", "C"]) == "SEED_UNSTABLE"
    assert decide({"phi1": POSITIVE}).startswith(POSITIVE)
    assert decide({"phi1": "DISCOVERY_ONLY_WRITE"}).startswith(
        "DISCOVERY_ONLY_WRITE")
    assert decide({"phi1": "HELD_READ_NOT_WRITABLE"}).startswith(
        "HELD_READ_NOT_WRITABLE")
    assert decide({"phi1": "NO_POSCOND_READ_WRITE_WORKS"}).startswith(
        "NO_POSCOND_READ_WRITE_WORKS")
    # precedence: a positive on one target outranks a negative on the other.
    assert decide({"phi1": POSITIVE,
                   "phi2": "NO_POSCOND_READ_WRITE_WORKS"}).startswith(POSITIVE)
    # oblique patch sets the read value exactly (sanity on the patch path).
    c = IV.unit(np.array([1.0, 0.0, 0.0]))
    u = IV.unit(np.array([0.0, 1.0, 0.0]))
    P1 = IV.oblique_patch(c, c + u)
    delta = np.array([[2.0, 5.0, 0.0]])
    assert np.allclose(delta @ P1 @ c, delta @ c)
    print("i1prime selftest passed: helpers, verdict partition, routing")


# ---------------------------------------------------------------------------
# main (guards / halt / run)
# ---------------------------------------------------------------------------

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
            print(f"HALT: I0 preflight did not route to I1 in {args.i0_artifact}")
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

    print("=== I1': position-conditioned fixed-read oblique write search ===")
    print(f"device={device}")
    print(f"target={proc.name} m={M} LAYER={LAYER} seeds={SEEDS}")
    print(f"discovery positions={TS_DISC}; heldout positions={TS_HELDOUT}")
    print("reads: position-conditioned, fit IN PLACE per bin (not transported)")
    print("targets:", ", ".join(TARGETS))
    print("controls:", ", ".join(CONTROL_PREDICATES))
    print(f"specificity non-target room floor={SPEC_ROOM_MIN:.3f}")
    print("Exact p_phi is endpoint-audit only; writes are observable-selected.\n")

    per_seed = {name: [] for name in TARGETS}
    for seed in SEEDS:
        print(f"[seed {seed}]")
        controls = control_predicate_report(model, proc, cfg, seed, masks)
        for name, mc in controls.items():
            print(f"  control {name:<18} std disc/held "
                  f"{mc['std_disc']:.3f}/{mc['std_heldout']:.3f} "
                  f"R2 disc/held {mc['r2_disc']:.2f}/{mc['r2_heldout']:.2f}")
        for target in TARGETS:
            summary, families = run_target(model, proc, cfg, seed, masks, target)
            per_seed[target].append(summary["verdict"])
            print(f"\n  target {target}: k_core={summary['k_core']} "
                  f"inplace read R2 disc/held {summary['r2_disc']:.2f}/"
                  f"{summary['r2_heldout']:.2f}; room disc/held "
                  f"{summary['room_disc']:.4f}/{summary['room_heldout']:.4f}; "
                  f"oe disc/held {summary['oe_disc']:.3f}/"
                  f"{summary['oe_heldout']:.3f}")
            print(f"  best={summary['best_family']} c disc/held "
                  f"{summary['best_disc']:.2f}/{summary['best_heldout']:.2f}; "
                  f"rand held={summary['random_heldout']:.2f}; "
                  f"same held={summary['same_heldout']:.2f}; "
                  f"spec={summary['specificity']:.2f}; "
                  f"ret={summary['retention']:.2f}; "
                  f"held-inplace(nonrandom)={summary['best_held_inplace']:.2f} "
                  f"-> {summary['verdict']}")
            print("  specificity predicates included="
                  f"{summary['spec_included']} skipped_low_room="
                  f"{summary['spec_skipped_low_room']} "
                  f"(room floor {SPEC_ROOM_MIN:.3f})")
            print(IV.format_intervention_table(family_rows(target, families)))
            print()

    print("[multi-seed aggregate]")
    aggregates = {}
    for target, verdicts in per_seed.items():
        aggregates[target] = aggregate(verdicts)
        print(f"  {target:<18} {verdicts} -> {aggregates[target]}")
    decision = decide(aggregates)
    print(f"\nDECISION: {decision}")
    if decision.startswith(POSITIVE):
        print("  A position-conditioned in-place predicate read admits an "
              "oblique write that controls the registered predicate with "
              "held-out-position transfer, specificity, and random/same-read "
              "margins. Carry the fixed-read oblique class into I2.")
    elif decision.startswith("DISCOVERY_ONLY_WRITE"):
        print("  A write controlled on discovery positions but did not "
              "transfer to held-out positions even with the held in-place "
              "read, while some non-random write still controlled through "
              "the held read. Treat as measured position-entangled write "
              "control under this registered menu.")
    elif decision.startswith("HELD_READ_NOT_WRITABLE"):
        print("  A write controlled on discovery positions, but no registered "
              "non-random candidate controlled through the held in-place read. "
              "Treat as readable-but-not-writable at held positions under this "
              "rank-1 oblique menu; route to I2 read/write-pair or I3 interchange.")
    elif decision.startswith("NO_POSCOND_READ_WRITE_WORKS"):
        print("  No registered write controls the predicate through the "
              "position-conditioned in-place read despite full-patch room and "
              "calibrated endpoints. The read is readable-but-not-writable at "
              "rank-1 oblique; route to I2 read/write-pair or I3 interchange "
              "before new toys.")
    elif decision.startswith("NO_PATCH_ROOM"):
        print("  The target lacks full-patch predicate room under this "
              "vehicle; change target, patch point, or process.")
    elif decision.startswith("READ_NOT_DECODABLE"):
        print("  The position-conditioned in-place read failed to decode at a "
              "bin: the exp 31/32 premise did not reproduce. Fix the "
              "substrate/measurement before interpreting the write result.")
    else:
        print("  See per-target aggregate: the registered branches did not "
              "support a stable positive position-conditioned write result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
