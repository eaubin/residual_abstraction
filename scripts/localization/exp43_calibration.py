"""exp43_calibration.py — calibration reference run for the exp-43 depth-carrier
localization rung (NOT a model claim). Derives the registered thresholds from
MEASURED references, the exp-38 discipline (thresholds read against a ceiling and a
floor, not data-tuned), before the experiment's predictions are frozen.

It enumerates the architecture-given component writes (emb + per-layer attention
heads + MLP, position-resolved; localize.chain_probs_components) and, on
top_type-matched depth_triples pairs at horizon k=1, measures:

  - the internal ceiling f_internal (all internal writes spliced, emb CLEAN) and
    the full ceiling f_all (incl emb -> ~source);
  - per-unit sufficiency suff(cid,p) -> a ranking;
  - the ranked cumulative transport curve (add top-j units) vs the RANDOM-component
    cumulative floor (the no-locus shape) -> SAT_K / LOCUS_MARGIN;
  - necessity (leave-one-out) for the top units and the suff-nec gap (the redundancy
    signature) -> REDUND_GAP;
  - top_type drag of the top units vs random units -> SPEC_MARGIN;
  - a PLANTED single-carrier detectability check (a known one-unit carrier must
    saturate the ranked curve at j=1 with gap~0) — the transfer-validity anchor.

Runs on the accelerator (load_model -> MPS/CUDA). Reference only; the registered
4-seed verdict run is the separate experiment script.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize import (chain_probs_components, component_ids, cr_cond,  # noqa: E402
                      depth_triples, facet_observable, make_Xc, q_at,
                      record_component_writes, require_expected_config,
                      stack_labels)
from expcommon import load_model, validity_gate  # noqa: E402
from processes import PROCESSES  # noqa: E402

GAP_MIN = 0.10


def _f(P, C, S):
    g = S - C
    keep = np.abs(g) >= GAP_MIN
    return float(np.mean((P[keep] - C[keep]) / g[keep])) if keep.sum() else float("nan")


def _drag(P_top, C_top, valid):
    keep = valid & np.isfinite(P_top) & np.isfinite(C_top)
    return float(np.mean(np.abs(P_top[keep] - C_top[keep]))) if keep.sum() else float("nan")


def build_pairset(model, proc, cfg, t, k, lo, hi, rng, cap):
    """Return everything a transport call needs at (t, k): clean continuations Xc,
    the source recorded writes, the unpatched/source endpoints (depth + top_type)."""
    m, V = cfg["m"], proc.V
    Xe = proc.sample(cfg["eval_seqs"] * 3, cfg["seq_len"], rng)
    labels = {i: stack_labels(Xe[i], [t], m)[t] for i in range(len(Xe))}
    clean, ihi, _ = depth_triples(labels, lo, hi, rng)
    if len(clean) > cap:
        clean, ihi = clean[:cap], ihi[:cap]
    Xc = make_Xc(Xe[clean], t, m, V)
    rec_src = record_component_writes(model, Xe[ihi], t)
    q_un = chain_probs_components(model, Xc, t, m, V, rec_src, [])     # per-head no-op
    C = cr_cond(q_un, V, m, k)
    Ct, vt = facet_observable(q_un, "top_type", V, m)
    S = cr_cond(q_at(model, Xe[ihi], t, m, V), V, m, k)               # source oracle cond
    return dict(Xc=Xc, rec_src=rec_src, C=C, S=S, Ct=Ct, vt=vt, n=len(clean))


def transport(model, ps, t, k, m, V, splice, rec=None):
    """Depth transport fraction (and top_type drag) under a component splice. `rec`
    overrides the source recorded writes (for the planted reference)."""
    rec = ps["rec_src"] if rec is None else rec
    q = chain_probs_components(model, ps["Xc"], t, m, V, rec, splice)
    f = _f(cr_cond(q, V, m, k), ps["C"], ps["S"])
    Pt, _ = facet_observable(q, "top_type", V, m)
    return f, _drag(Pt, ps["Ct"], ps["vt"])


def calibrate_at(model, proc, cfg, t, k, lo, hi, rng, cap, topn):
    m, V = cfg["m"], proc.V
    ps = build_pairset(model, proc, cfg, t, k, lo, hi, rng, cap)
    internal = component_ids(cfg["layers"], model.cfg.n_heads, include_emb=False)
    units = [(cid, p) for cid in internal for p in range(t + 1)]
    print(f"\n=== t={t} k={k} (depth {lo}->{hi}) | n={ps['n']} pairs | "
          f"{len(units)} units ===")

    f_all, _ = transport(model, ps, t, k, m, V,
                         component_ids(cfg["layers"], model.cfg.n_heads, True))
    f_int, _ = transport(model, ps, t, k, m, V, internal)
    print(f"ceiling: f_all(+emb)={f_all:+.3f} (->~source)   "
          f"f_internal(emb clean)={f_int:+.3f}")

    suff = {}
    drag = {}
    for cid, p in units:
        f, dg = transport(model, ps, t, k, m, V, {cid: [p]})
        suff[(cid, p)] = f
        drag[(cid, p)] = dg
    ranked = sorted(units, key=lambda u: suff[u], reverse=True)
    print("top units by sufficiency:")
    for u in ranked[:topn]:
        print(f"  {str(u[0]):>16} @p={u[1]:<2}  suff={suff[u]:+.3f}  "
              f"top_drag={drag[u]:+.3f}")

    # ranked vs random cumulative curves
    js = [1, 2, 3, 5, 8, 12, 20, len(units)]
    js = sorted({j for j in js if j <= len(units)})
    print("cumulative transport (ranked top-j vs random-j, frac of f_internal):")
    rand_curve = {}
    for j in js:
        spec = {}
        for cid, p in ranked[:j]:
            spec.setdefault(cid, []).append(p)
        f_rank, _ = transport(model, ps, t, k, m, V, spec)
        rr = []
        for _ in range(3):
            pick = [units[i] for i in rng.choice(len(units), j, replace=False)]
            sp = {}
            for cid, p in pick:
                sp.setdefault(cid, []).append(p)
            rr.append(transport(model, ps, t, k, m, V, sp)[0])
        f_rand = float(np.mean(rr))
        rand_curve[j] = f_rand
        denom = f_int if abs(f_int) > 1e-6 else 1.0
        print(f"  j={j:>4}: ranked={f_rank:+.3f} ({f_rank/denom:5.0%})  "
              f"random={f_rand:+.3f} ({f_rand/denom:5.0%})  margin={f_rank-f_rand:+.3f}")

    # necessity (leave-one-out) for the top units -> suff-nec gap
    print("necessity / redundancy for top units (suff - nec gap):")
    allspec = {cid: list(range(t + 1)) for cid in internal}
    for u in ranked[:topn]:
        cid, p = u
        sp = {c: list(ps_) for c, ps_ in allspec.items()}
        sp[cid] = [pp for pp in sp[cid] if pp != p]
        f_drop, _ = transport(model, ps, t, k, m, V, sp)
        nec = f_int - f_drop
        print(f"  {str(cid):>16} @p={p:<2}  suff={suff[u]:+.3f}  nec={nec:+.3f}  "
              f"gap={suff[u]-nec:+.3f}")

    # planted single-carrier detectability: clean writes everywhere, source at one
    # chosen unit -> a known one-unit carrier. Splicing all units (frozen) gives its
    # contribution; any OTHER single unit is a no-op. (Frozen construction: it sets
    # the SAT_K=1 / gap~0 anchor; the model curves above are live.)
    rec_clean = record_component_writes(model, ps["Xc"][:, 0, :], t)  # clean prefixes
    chosen = ranked[0]
    rec_plant = {c: rec_clean[c].clone() for c in rec_clean}
    rec_plant[chosen[0]] = rec_plant[chosen[0]].clone()
    rec_plant[chosen[0]][:, chosen[1]] = ps["rec_src"][chosen[0]][:, chosen[1]]
    allids = component_ids(cfg["layers"], model.cfg.n_heads, True)
    f_plant_full, _ = transport(model, ps, t, k, m, V, allids, rec=rec_plant)
    f_plant_one, _ = transport(model, ps, t, k, m, V, {chosen[0]: [chosen[1]]}, rec=rec_plant)
    other = ranked[1]
    f_plant_other, _ = transport(model, ps, t, k, m, V, {other[0]: [other[1]]}, rec=rec_plant)
    print(f"planted single-carrier {chosen[0]}@p={chosen[1]}: "
          f"full(frozen)={f_plant_full:+.3f}  chosen-only={f_plant_one:+.3f}  "
          f"other-only={f_plant_other:+.3f} (expect chosen>>other~0)")
    return dict(f_int=f_int, suff=suff, drag=drag, ranked=ranked,
                rand_curve=rand_curve)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
    ap.add_argument("--positions", type=int, nargs="+", default=[12, 20])
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--seed", type=int, default=700)
    args = ap.parse_args(argv)

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    require_expected_config({**cfg, "burn_in": cfg.get("burn_in", 4)})
    model = load_model(args.outdir, cfg, proc)
    print(f"device: {next(model.parameters()).device}")
    _, ok = validity_gate(model, proc, cfg, args.seed)
    if not ok:
        print("HALT: validity gate failed"); return

    rng = np.random.default_rng(args.seed)
    for t in args.positions:
        calibrate_at(model, proc, cfg, t, k=1, lo=1, hi=2, rng=rng,
                     cap=args.cap, topn=args.topn)
    print("\n(reference only — NOT a model verdict. Use these ceiling/floor numbers "
          "to set SAT_K_MAX, LOCUS_MARGIN, REDUND_GAP, SPEC_MARGIN before freezing.)")


if __name__ == "__main__":
    main()
