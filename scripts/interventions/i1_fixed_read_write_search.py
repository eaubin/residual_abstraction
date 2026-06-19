"""
I1 — fixed-read oblique write search on pstack.

Pre-registration vehicle for INTERVENTION_CLASS_BENCHMARK.md I1. Given the
affine predicate readout construction from exp 29 as a fixed read, search only
over the write side of a rank-1 oblique residual-stream patch on pstack-L4.

Claim-producing command, after preregistration review only:

    python3 scripts/interventions/i1_fixed_read_write_search.py --outdir out/pstack-L4

Review-only checks:

    python3 scripts/interventions/i1_fixed_read_write_search.py --selftest

Built from the I0 living helpers in interventions.py. Exact predicate truth is
evaluation-only endpoint audit; reads, writes, selection, and scores use
observable model completion probabilities.

RESULTS (see experiments/30-fixed-read-oblique-write-search.md):
FIXED_READ_LIMIT(phi1_next_closes,phi2_net_return). The fixed affine reads
decode on discovery positions but fail the registered held-out-position R2 gate.
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
import predicates as P
from abstraction import center_by_position
from battery import Refs, cegar_loop
from discover import PairSet, self_checks
from expcommon import LAYER, load_model
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
CHAIN_BATCH = 4096

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

POSITIVE = "FIXED_READ_WRITE_CONTROL"


def chain_probs_only(model, X_cont, layer, prefix_state, t, m, V):
    """Exact m-step probabilities without materializing final residuals.

    This is an execution-preserving fast path for the I1 scorer. The approved
    design only consumes completion probabilities; the standard
    ``midstream.chain_probs`` also allocates residual outputs for coherence
    diagnostics that I1 never reads. Keeping this local avoids changing the
    shared historical evaluator.
    """
    n, C, L = X_cont.shape
    flat_np = X_cont.reshape(n * C, L)
    ps = None
    if prefix_state is not None:
        ps = prefix_state.repeat_interleave(C, dim=0)
    out = np.empty((n * C,))
    pos_all = torch.arange(L)
    with torch.no_grad():
        for i in range(0, n * C, CHAIN_BATCH):
            sl = slice(i, min(i + CHAIN_BATCH, n * C))
            flat = torch.from_numpy(flat_np[sl])
            x = model.tok(flat) + model.pos(pos_all)
            for li, blk in enumerate(model.blocks):
                if li == layer and ps is not None:
                    x = x.clone()
                    x[:, :t + 1] = ps[sl]
                x = blk(x)
            probs = torch.softmax(model.head(model.ln_f(x)), dim=-1)
            rows = torch.arange(sl.stop - sl.start)
            q = torch.ones(sl.stop - sl.start, dtype=probs.dtype)
            for j in range(m):
                q *= probs[rows, t + j, flat[:, t + 1 + j]]
            out[sl] = q.double().numpy()
    return out.reshape(n, C)


def run_ps(ps, model, Ppatch, src_side=False):
    """PairSet.run equivalent for q-only scoring."""
    q = np.empty((ps.n, ps.C))
    for t, idx in ps.groups:
        X = ps.Xc_src[t] if src_side else ps.Xc_tgt[t]
        pref = None
        if Ppatch is not None:
            pt = ps.pref_tgt[t].double().numpy()
            delta = ps.pref_src[t].double().numpy() - pt
            pref = torch.from_numpy(pt + delta @ Ppatch).float()
        q[idx] = chain_probs_only(model, X, ps.layer, pref, t, ps.m, ps.V)
    return q


def run_pphi(ps, model, Ppatch, mask, src_side=False):
    """Exact observable p_phi without evaluating mask-false continuations."""
    keep = np.asarray(mask, dtype=bool)
    out = np.empty(ps.n)
    for t, idx in ps.groups:
        Xfull = ps.Xc_src[t] if src_side else ps.Xc_tgt[t]
        X = Xfull[:, keep, :]
        pref = None
        if Ppatch is not None:
            pt = ps.pref_tgt[t].double().numpy()
            delta = ps.pref_src[t].double().numpy() - pt
            pref = torch.from_numpy(pt + delta @ Ppatch).float()
        out[idx] = chain_probs_only(model, X, ps.layer, pref, t, ps.m,
                                    ps.V).sum(axis=1)
    return out


def split_fit_read(ps, model, mask, d, rng):
    """Fit the fixed affine read on observable p_phi at discovery positions."""
    R, _, pos, _, _ = IV.pairset_residual_frame(ps, d)
    Rc = center_by_position(R, pos, np.ones(ps.n, dtype=bool))
    y = run_pphi(ps, model, None, mask)
    perm = rng.permutation(ps.n)
    tr, te = perm[:ps.n // 2], perm[ps.n // 2:]
    w, b = IV.affine_readout(Rc[tr], y[tr], LAM)
    return {
        "c": IV.unit(w),
        "b": b,
        "r2_disc": IV.r2_score(y[te], Rc[te] @ w + b),
        "std_disc": float(y.std()),
    }


def decode_heldout(ps, model, mask, c, b, d):
    R, _, pos, _, _ = IV.pairset_residual_frame(ps, d)
    Rc = center_by_position(R, pos, np.ones(ps.n, dtype=bool))
    y = run_pphi(ps, model, None, mask)
    return IV.r2_score(y, Rc @ c + b), float(y.std())


def endpoints(ps, model, proc, mask, d):
    p_un = run_pphi(ps, model, None, mask)
    p_src = run_pphi(ps, model, None, mask, src_side=True)
    p_full = run_pphi(ps, model, np.eye(d), mask)
    _, _, _, b_tgt, b_src = IV.pairset_residual_frame(ps, d)
    exact_tgt = P.exact_pphi(b_tgt, mask, proc, M)
    exact_src = P.exact_pphi(b_src, mask, proc, M)
    return {
        "p_un": p_un, "p_src": p_src, "p_full": p_full,
        "room": IV.predicate_room(p_un, p_src, p_full),
        "oe": IV.endpoint_audit(p_un, exact_tgt, p_src, exact_src),
    }


def patch_control(ps, model, mask, c, w, alpha, ep):
    if alpha == 0.0:
        return 0.0 if ep["room"] > IV.ROOM_TOL else float("nan")
    try:
        base = IV.oblique_patch(c, w)
    except IV.SingularReadWrite:
        return float("nan")
    p_patch = run_pphi(ps, model, IV.scaled_patch(base, alpha), mask)
    return IV.predicate_control(ep["p_un"], ep["p_src"], p_patch, ep["p_full"])


def score_write(ps, model, mask, c, w, ep):
    curve = [(a, patch_control(ps, model, mask, c, w, a, ep)) for a in ALPHAS]
    finite = [(s, a) for a, s in curve if np.isfinite(s)]
    if not finite:
        return {"alpha": float("nan"), "control": float("nan"), "curve": curve}
    best_s, best_a = max(finite, key=lambda x: x[0])
    return {"alpha": best_a, "control": best_s, "curve": curve}


def specificity_predicates(masks, target, held_eps):
    included, skipped_low_room = [], []
    for name in masks:
        if name == target:
            continue
        room = held_eps[name]["room"]
        if np.isfinite(room) and room >= SPEC_ROOM_MIN:
            included.append(name)
        else:
            skipped_low_room.append(name)
    return included, skipped_low_room


def specificity(ps, model, masks, target, c, w, alpha, held_eps, spec_names):
    vals = []
    try:
        Pbase = IV.oblique_patch(c, w)
    except IV.SingularReadWrite:
        return float("nan")
    for name in spec_names:
        mask = masks[name]
        ep = held_eps[name]
        p_patch = run_pphi(ps, model, IV.scaled_patch(Pbase, alpha), mask)
        ctl = IV.predicate_control(ep["p_un"], ep["p_src"], p_patch,
                                   ep["p_full"])
        if np.isfinite(ctl):
            vals.append(abs(float(ctl)))
    return max(vals) if vals else 0.0


def add_candidate(out, name, w):
    try:
        out[name] = IV.unit(w)
    except ValueError:
        pass


def build_write_candidates(disc, model, cfg, proc, c, target_ep, seed, d):
    """Observable write-source constructors for I1 arms."""
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
    """Learn a write direction with the read covector fixed.

    The parameterization is w = c + u_perp for unit c, so c.w = 1 throughout.
    The loss is observable predicate MSE against source-run p_phi on discovery
    pairs. Exact process labels are not used.
    """
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
    u = torch.from_numpy(u0.astype(np.float32)).requires_grad_()
    c_t = torch.from_numpy(c.astype(np.float32))
    keep = np.asarray(mask_np, dtype=bool)
    psrc_t = torch.from_numpy(p_src_np.astype(np.float32))
    pos_all = torch.arange(L)
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
            Xc = torch.from_numpy(disc.Xc_tgt[t][loc][:, keep, :])
            bsz, C, _ = Xc.shape
            pt = disc.pref_tgt[t][loc].float()
            delta = disc.pref_src[t][loc].float() - pt
            ps = pt + delta @ P_t
            flat = Xc.reshape(bsz * C, L)
            ps_r = ps.repeat_interleave(C, dim=0)
            x = model.tok(flat) + model.pos(pos_all)
            for li, blk in enumerate(model.blocks):
                if li == LAYER:
                    x = torch.cat([ps_r, x[:, t + 1:]], dim=1)
                x = blk(x)
            logp = torch.log_softmax(model.head(model.ln_f(x)), dim=-1)
            rows = torch.arange(bsz * C)
            lq = sum(logp[rows, t + j, flat[:, t + 1 + j]] for j in range(M))
            q = torch.exp(lq.reshape(bsz, C))
            pphi = q.sum(dim=1)
            total = total + ((pphi - psrc_t[sel]) ** 2).mean()
        loss = total / len(np.unique(pair_t[batch]))
        opt.zero_grad()
        loss.backward()
        opt.step()

    w = current_w().detach().numpy().astype(np.float64)
    n = float(np.linalg.norm(w))
    if not np.isfinite(n) or n > LEARN_NORM_MAX:
        raise FloatingPointError(f"learned write norm invalid: {n}")
    return IV.unit(w), n


def select_family(scored, prefix):
    items = [(name, m) for name, m in scored.items() if name.startswith(prefix)]
    finite = [(m["control"], name, m) for name, m in items
              if np.isfinite(m["control"])]
    return max(finite, default=(float("nan"), None, None), key=lambda x: x[0])


def family_rows(target, family_results):
    rows = []
    for fam, m in family_results.items():
        rows.append({
            "target": target,
            "read": "affine_fixed",
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
    if m["std_disc"] < VAR_MIN or m["std_heldout"] < VAR_MIN:
        return "TARGET_VACUOUS"
    if m["r2_disc"] < R2_MIN:
        return "FIXED_READ_NOT_DECODABLE"
    if m["r2_heldout"] < R2_MIN:
        return "FIXED_READ_NOT_TRANSPORTED"
    if m["room_disc"] <= IV.ROOM_TOL or m["room_heldout"] <= IV.ROOM_TOL:
        return "NO_PATCH_ROOM"
    if m["oe_disc"] > OE_BAND or m["oe_heldout"] > OE_BAND:
        return "OBS_EXACT_DRIFT"
    if (m["best_disc"] >= C_MIN and
            (m["best_heldout"] < C_MIN or m["retention"] < RETENTION_MIN)):
        return "DISCOVERY_ONLY_WRITE"
    if m["best_heldout"] < C_MIN:
        return "NO_FIXED_READ_WRITE_WORKS"
    if m["specificity"] > SPEC_MAX:
        return "NONSPECIFIC_CONTROL"
    if (m["best_disc"] - m["random_disc"] < C_MARGIN or
            m["best_heldout"] - m["random_heldout"] < C_MARGIN):
        return "RANDOM_MATCHED_CONTROL"
    if m["best_heldout"] - m["same_heldout"] < C_MARGIN:
        return "SAME_READ_BASELINE_CONTROL"
    return POSITIVE


def aggregate(verdicts):
    counts = {v: verdicts.count(v) for v in set(verdicts)}
    top = max(counts, key=counts.get)
    return top if counts[top] >= SEED_MAJORITY else "SEED_UNSTABLE"


def decide(aggregates):
    drift = [n for n, v in aggregates.items() if v == "OBS_EXACT_DRIFT"]
    if drift:
        return "OBS_EXACT_DRIFT(" + ",".join(drift) + ")"
    pos = [n for n, v in aggregates.items() if v == POSITIVE]
    if pos:
        return "FIXED_READ_WRITE_CONTROL(" + ",".join(pos) + ")"
    disc = [n for n, v in aggregates.items() if v == "DISCOVERY_ONLY_WRITE"]
    if disc:
        return "POSITION_ENTANGLED_WRITE(" + ",".join(disc) + ")"
    nonspec = [n for n, v in aggregates.items() if v == "NONSPECIFIC_CONTROL"]
    if nonspec:
        return "NONSPECIFIC_CONTROL(" + ",".join(nonspec) + ")"
    rand = [n for n, v in aggregates.items() if v == "RANDOM_MATCHED_CONTROL"]
    if rand:
        return "RANDOM_MATCHED_CONTROL(" + ",".join(rand) + ")"
    same = [n for n, v in aggregates.items()
            if v == "SAME_READ_BASELINE_CONTROL"]
    if same:
        return "SAME_READ_BASELINE_CONTROL(" + ",".join(same) + ")"
    no_work = [n for n, v in aggregates.items()
               if v == "NO_FIXED_READ_WRITE_WORKS"]
    if no_work:
        return "NO_FIXED_READ_WRITE_WORKS(" + ",".join(no_work) + ")"
    no_room = [n for n, v in aggregates.items() if v == "NO_PATCH_ROOM"]
    if no_room:
        return "NO_PATCH_ROOM(" + ",".join(no_room) + ")"
    read = [n for n, v in aggregates.items()
            if v in ("FIXED_READ_NOT_DECODABLE",
                     "FIXED_READ_NOT_TRANSPORTED")]
    if read:
        return "FIXED_READ_LIMIT(" + ",".join(read) + ")"
    vac = [n for n, v in aggregates.items() if v == "TARGET_VACUOUS"]
    if vac:
        return "TARGET_VACUOUS(" + ",".join(vac) + ")"
    return "SEED_UNSTABLE"


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
    read = split_fit_read(disc, model, mask, d, rng)
    c = read["c"]
    r2_h, std_h = decode_heldout(held, model, mask, c, read["b"], d)
    ep_d = endpoints(disc, model, proc, mask, d)
    ep_h = endpoints(held, model, proc, mask, d)
    held_eps = {target: ep_h}
    for name, msk in masks.items():
        if name != target:
            held_eps[name] = endpoints(held, model, proc, msk, d)
    spec_names, spec_skipped = specificity_predicates(masks, target, held_eps)
    candidates, k_core = build_write_candidates(disc, model, cfg, proc, c,
                                                ep_d, seed, d)

    scored_d = {name: score_write(disc, model, mask, c, w, ep_d)
                for name, w in candidates.items()}
    nonrandom = [n for n in scored_d if not n.startswith("random_")]
    init_name = max(nonrandom, key=lambda n: scored_d[n]["control"])
    learned_norm = float("nan")
    learned_failed = False
    try:
        learned_w, learned_norm = learn_write(model, disc, c, mask,
                                              ep_d["p_src"],
                                              candidates[init_name], seed, cfg)
        candidates["learned_write"] = learned_w
        scored_d["learned_write"] = score_write(disc, model, mask, c,
                                                learned_w, ep_d)
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
                    "candidate": "learned_write",
                    "alpha_disc": float("nan"),
                    "control_disc": float("nan"),
                    "alpha_heldout": float("nan"),
                    "control_heldout": float("nan"),
                    "room_disc": ep_d["room"],
                    "room_heldout": ep_h["room"],
                    "specificity": float("nan"),
                    "oe_disc": ep_d["oe"],
                    "oe_heldout": ep_h["oe"],
                    "retention": float("nan"),
                    "branch": "LEARNED_DIVERGED",
                }
            continue
        w = candidates[cname]
        hm = score_write(held, model, mask, c, w, ep_h)
        spec = specificity(held, model, masks, target, c, w, hm["alpha"],
                           held_eps, spec_names)
        retention = (hm["control"] / dm["control"]
                     if dm["control"] and np.isfinite(dm["control"])
                     else float("nan"))
        branch = "-"
        if fam == "learned_write" and not np.isfinite(learned_norm):
            branch = "LEARNED_DIVERGED"
        families[fam] = {
            "candidate": cname,
            "alpha_disc": dm["alpha"],
            "control_disc": dm["control"],
            "alpha_heldout": hm["alpha"],
            "control_heldout": hm["control"],
            "room_disc": ep_d["room"],
            "room_heldout": ep_h["room"],
            "specificity": spec,
            "oe_disc": ep_d["oe"],
            "oe_heldout": ep_h["oe"],
            "retention": retention,
            "branch": branch,
        }

    best_pool = {k: v for k, v in families.items() if k != "random_best"}
    best_family = max(best_pool, key=lambda k: best_pool[k]["control_disc"])
    best = best_pool[best_family]
    rand = families["random_best"]
    same = families["same_read"]
    summary = {
        "std_disc": read["std_disc"],
        "std_heldout": std_h,
        "r2_disc": read["r2_disc"],
        "r2_heldout": r2_h,
        "room_disc": ep_d["room"],
        "room_heldout": ep_h["room"],
        "oe_disc": ep_d["oe"],
        "oe_heldout": ep_h["oe"],
        "best_family": best_family,
        "best_disc": best["control_disc"],
        "best_heldout": best["control_heldout"],
        "random_disc": rand["control_disc"],
        "random_heldout": rand["control_heldout"],
        "same_heldout": same["control_heldout"],
        "specificity": best["specificity"],
        "retention": best["retention"],
        "k_core": k_core,
        "learned_norm": learned_norm,
        "spec_included": tuple(spec_names),
        "spec_skipped_low_room": tuple(spec_skipped),
    }
    summary["verdict"] = classify_target(summary)
    for fam in families.values():
        if fam["branch"] == "-":
            fam["branch"] = summary["verdict"] if fam is best else "-"
    return summary, families


def control_predicate_report(model, proc, cfg, seed, masks):
    d = cfg["d_model"]
    disc = PairSet(model, proc, cfg, PAIRS_DISC, M, seed + 333, PAIR_POOL,
                   layer=LAYER, ts=TS_DISC)
    held = PairSet(model, proc, cfg, PAIRS_HELDOUT, M, seed + 444, PAIR_POOL,
                   layer=LAYER, ts=TS_HELDOUT)
    rng = np.random.default_rng(seed + 999)
    out = {}
    for name in CONTROL_PREDICATES:
        read = split_fit_read(disc, model, masks[name], d, rng)
        r2_h, std_h = decode_heldout(held, model, masks[name], read["c"],
                                     read["b"], d)
        out[name] = {
            "std_disc": read["std_disc"],
            "std_heldout": std_h,
            "r2_disc": read["r2_disc"],
            "r2_heldout": r2_h,
        }
    return out


def selftest():
    IV._selftest()
    P._selftest()
    base = {
        "std_disc": 0.2, "std_heldout": 0.2,
        "r2_disc": 0.7, "r2_heldout": 0.7,
        "room_disc": 0.1, "room_heldout": 0.1,
        "oe_disc": 0.01, "oe_heldout": 0.01,
        "best_disc": 0.8, "best_heldout": 0.7,
        "random_disc": 0.1, "random_heldout": 0.1,
        "same_heldout": 0.0, "specificity": 0.1, "retention": 0.875,
    }

    def m(**kw):
        x = dict(base); x.update(kw); return x
    assert classify_target(m()) == POSITIVE
    assert classify_target(m(std_disc=0.01)) == "TARGET_VACUOUS"
    assert classify_target(m(r2_disc=0.2)) == "FIXED_READ_NOT_DECODABLE"
    assert classify_target(m(r2_heldout=0.2)) == "FIXED_READ_NOT_TRANSPORTED"
    assert classify_target(m(room_heldout=0.0)) == "NO_PATCH_ROOM"
    assert classify_target(m(oe_heldout=0.2)) == "OBS_EXACT_DRIFT"
    assert classify_target(m(best_heldout=0.2)) == "DISCOVERY_ONLY_WRITE"
    assert classify_target(m(best_disc=0.2, best_heldout=0.2)) == \
        "NO_FIXED_READ_WRITE_WORKS"
    assert classify_target(m(specificity=0.8)) == "NONSPECIFIC_CONTROL"
    assert classify_target(m(random_heldout=0.6)) == "RANDOM_MATCHED_CONTROL"
    assert classify_target(m(same_heldout=0.6)) == "SAME_READ_BASELINE_CONTROL"
    fake_eps = {
        "target": {"room": 0.1},
        "near_flat": {"room": 0.0004},
        "practical": {"room": 0.02},
        "nan_room": {"room": float("nan")},
    }
    inc, skip = specificity_predicates(fake_eps, "target", fake_eps)
    assert inc == ["practical"], (inc, skip)
    assert skip == ["near_flat", "nan_room"], (inc, skip)
    assert aggregate([POSITIVE] * 3 + ["NO_FIXED_READ_WRITE_WORKS"]) == POSITIVE
    assert aggregate([POSITIVE, "A", "B", "C"]) == "SEED_UNSTABLE"
    assert decide({"phi1": POSITIVE}).startswith(POSITIVE)
    assert decide({"phi1": "DISCOVERY_ONLY_WRITE"}).startswith(
        "POSITION_ENTANGLED_WRITE")
    assert decide({"phi1": "NO_FIXED_READ_WRITE_WORKS"}).startswith(
        "NO_FIXED_READ_WRITE_WORKS")
    c = IV.unit(np.array([1.0, 0.0, 0.0]))
    u = IV.unit(np.array([0.0, 1.0, 0.0]))
    P1 = IV.oblique_patch(c, c + u)
    delta = np.array([[2.0, 5.0, 0.0]])
    assert np.allclose(delta @ P1 @ c, delta @ c)
    print("i1 selftest passed: helpers, verdict partition, aggregation")


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
        i0_text = f.read()
    if "I1 ROUTING: GO" not in i0_text:
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
    model = load_model(args.outdir, cfg, proc)
    masks = P.registered_masks(proc.V, M)

    print("=== I1: fixed-read oblique write search on pstack ===")
    print(f"target={proc.name} m={M} LAYER={LAYER} seeds={SEEDS}")
    print(f"discovery positions={TS_DISC}; heldout positions={TS_HELDOUT}")
    print("targets:", ", ".join(TARGETS))
    print("controls:", ", ".join(CONTROL_PREDICATES))
    print(f"specificity non-target room floor={SPEC_ROOM_MIN:.3f}")
    print("Exact p_phi is endpoint-audit only; writes are observable-selected.\n")

    per_seed = {name: [] for name in TARGETS}
    for seed in SEEDS:
        print(f"[seed {seed}]")
        controls = control_predicate_report(model, proc, cfg, seed, masks)
        for name, m in controls.items():
            print(f"  control {name:<18} std disc/held "
                  f"{m['std_disc']:.3f}/{m['std_heldout']:.3f} "
                  f"R2 disc/held {m['r2_disc']:.2f}/{m['r2_heldout']:.2f}")
        for target in TARGETS:
            summary, families = run_target(model, proc, cfg, seed, masks, target)
            per_seed[target].append(summary["verdict"])
            print(f"\n  target {target}: k_core={summary['k_core']} "
                  f"read R2 disc/held {summary['r2_disc']:.2f}/"
                  f"{summary['r2_heldout']:.2f}; room disc/held "
                  f"{summary['room_disc']:.4f}/{summary['room_heldout']:.4f}; "
                  f"oe disc/held {summary['oe_disc']:.3f}/"
                  f"{summary['oe_heldout']:.3f}")
            print(f"  best={summary['best_family']} c disc/held "
                  f"{summary['best_disc']:.2f}/{summary['best_heldout']:.2f}; "
                  f"rand held={summary['random_heldout']:.2f}; "
                  f"spec={summary['specificity']:.2f}; "
                  f"ret={summary['retention']:.2f} -> {summary['verdict']}")
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
        print("  A fixed affine predicate read admits an oblique write that "
              "controls the registered predicate with held-out-position "
              "transfer, specificity, and random/same-read margins.")
    elif decision.startswith("POSITION_ENTANGLED_WRITE"):
        print("  Some write search worked on discovery positions but did not "
              "transport to held-out positions. Treat as position-entangled "
              "control, not a stable intervention primitive.")
    elif decision.startswith("NO_FIXED_READ_WRITE_WORKS"):
        print("  No registered fixed-read write controls the predicate despite "
              "full-patch room and calibrated observable endpoints. Route to "
              "read/write-pair or path/interchange follow-up before new toys.")
    elif decision.startswith("NO_PATCH_ROOM"):
        print("  The target lacks full-patch predicate room under this vehicle; "
              "change target, patch point, or process before interpreting I1.")
    elif decision.startswith("FIXED_READ_LIMIT"):
        print("  The fixed affine read is not a transported interpreter for the "
              "target under this split; I1 cannot adjudicate write failure.")
    else:
        print("  See per-target aggregate: the registered branches did not "
              "support a stable positive fixed-read write result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
