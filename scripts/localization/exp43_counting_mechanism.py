"""exp43_counting_mechanism.py — Localizing the depth-counting mechanism.

Pre-registration: experiments/43-depth-counting-mechanism.md. Given the premise that
graded depth is recomputed from the prefix bracket-embeddings (not stored internally;
established by the premise gate, found in calibration), localize WHICH attention heads
at WHICH readout positions aggregate those embeddings into the forced-close graded-depth
conditional.

Instrument: the DETERMINISTIC forced-close conditional. Append the k oracle-legal closers
to the prefix and read P(close)=q2+q3 at position t+k (= cr_cond's quantity for the closer
continuation, but with fixed readout positions t..t+k so head outputs there are patchable).
Pairs are top-k-matched depth_triples so the appended closers are IDENTICAL within a pair
(the k>=2 closer-token control). Units = (layer, head/mlp, readout position); prefix units
are the calibration-established ~0 control. suff/nec cross-checked (the redundancy
measurement), a ranked-vs-random cumulative curve (concentration). Cross-facet top_type drag is
read at t (the matched, defined position) and is DESCRIPTIVE ONLY: a t+k-located unit is causally
insulated from t, so drag is a valid specificity probe only for p=t units and does not enter the
verdict (F1) — specificity at the readout locus is left untested by this instrument.

`--calibrate` runs the references (premise + ceilings + ranked/random floor) and recommends
the thresholds; the default run is the registered 4-seed verdict. Runs on the accelerator
(load_model -> MPS/CUDA).
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
from localize import (_selftest as _localize_selftest,  # noqa: E402
                      component_ids, exact_joint, facet_observable,
                      record_component_writes, require_expected_config,
                      splice_logits)
from expcommon import load_model, validity_gate  # noqa: E402
from processes import PROCESSES  # noqa: E402
from battery import majority_vote  # noqa: E402

# ---- registered scope -----------------------------------------------------
POSITIONS = (8, 12, 16, 20)
SEEDS = (701, 702, 703, 704)           # 700 is burned on calibration -> 4 fresh claim seeds
HORIZONS = {1: (1, 2), 2: (2, 3)}      # k -> (clean depth lo, source depth hi)
N_SEQS = 6000
GAP_MIN = 0.10
OE_BAND = 0.10
SEED_MAJORITY = 3
MIN_PAIRS = 256
PAIR_CAP = 1024        # bound the matched-pair count per (t,k) for the verdict run
N_RAND_DRAWS = 12      # random-floor MC draws per j (the load-bearing concentration reference)

# ---- premise-gate thresholds (calibration: f_emb=1.0, f_internal=0.06-0.24) ----
EMB_MIN = 0.80         # source embeddings alone must transport >= this
INT_MAX = 0.30         # all internal writes (emb clean) must transport <= this
INT_NEC_MAX = 0.20     # reverting internal (source emb) must drop transport < this

# ---- localization thresholds (FROZEN from --calibrate, seed 700, t in {8,12,20}) -----
# Refs: all-readout-window ceiling f=1.0; random-component floor <0.2 for j<=5; ranked
# cumulative >=0.83*ceiling by j=3 (margin >0.77); top units genuine (suff-nec <=0.12);
# top_type drag identically 0.0 (depth-specific). Cuts sit well inside these bands.
SAT_FRAC = 0.80        # "reaches the ceiling" = >= this fraction of the all-window f
SAT_K = 5              # localized iff ranked cumulative hits SAT_FRAC within this many units
LOCUS_MARGIN = 0.15    # ranked cumulative must beat random by this at saturation
REDUND_GAP = 0.25      # suff - nec >= this at the top units -> redundant
# top_type drag is read at t (the matched, defined position). A t+k-located unit is causally
# insulated from position t, so drag is DESCRIPTIVE ONLY (a valid specificity probe just for
# p=t units) and does NOT enter the verdict -> no SPEC_MARGIN / NONSPECIFIC branch (F1).


# ===========================================================================
# Dyck forced-close instrument
# ===========================================================================
def _stack(seq, t):
    st = []
    for tok in seq[:t + 1]:
        if tok < 2:
            st.append(int(tok))
        elif st:
            st.pop()
    return st


def legal_closers(seq, t, k):
    """The k oracle-legal closer tokens that pop the top k of the stack at t
    (token 2 closes type-0 top, 3 closes type-1). None if depth < k."""
    st = _stack(seq, t)
    if len(st) < k:
        return None
    return [2 if st[-1 - j] == 0 else 3 for j in range(k)]


def topk_state(seq, t, K):
    """(exact depth, top-K stack types top-first) at t; tuple None if depth < K."""
    st = _stack(seq, t)
    d = len(st)
    return d, (tuple(st[-1:-K - 1:-1]) if d >= K else None)


def make_ext(seqs, idx, t, k):
    """Append the k legal closers at t+1..t+k for rows `idx`; returns (ext, closers).
    Assumes depth >= k (guaranteed by the pairing)."""
    ext = seqs[idx].copy()
    closers = []
    for r, i in enumerate(idx):
        cl = legal_closers(seqs[i], t, k)
        closers.append(cl)
        ext[r, t + 1:t + 1 + k] = cl
    return ext, closers


def topk_pairs(Xe, t, k, rng):
    """clean (depth=k) / source (depth=k+1) index arrays, matched on the top-k stack
    types so their k legal closers are identical (the closer-token control)."""
    lo, hi = k, k + 1
    by_lo, by_hi = {}, {}
    for i in range(len(Xe)):
        d, tk = topk_state(Xe[i], t, k)
        if tk is None:
            continue
        if d == lo:
            by_lo.setdefault(tk, []).append(i)
        elif d == hi:
            by_hi.setdefault(tk, []).append(i)
    clean, src = [], []
    for tk in set(by_lo) & set(by_hi):
        a, b = by_lo[tk], by_hi[tk]
        n = min(len(a), len(b))
        clean += list(rng.permutation(a))[:n]
        src += list(rng.permutation(b))[:n]
    if not clean:
        return np.zeros(0, int), np.zeros(0, int)
    order = rng.permutation(len(clean))
    return np.array(clean)[order], np.array(src)[order]


def pclose(model, ext, t, k, source_rec, splice):
    """P(close)=q2+q3 at the readout position t+k under a component splice."""
    probs = splice_logits(model, ext, t + k, source_rec, splice)
    return probs[:, t + k, 2:4].sum(axis=1)


def oracle_pclose(proc, seqs, idx, t, k, closers, m):
    """Exact P(close at t+k | the k legal closers) from the Dyck oracle joint."""
    J = exact_joint(proc, seqs[idx], t, m).reshape(len(idx), *([proc.V] * m))
    out = np.empty(len(idx))
    for r in range(len(idx)):
        sub = J[r]
        for j in range(k):
            sub = sub[closers[r][j]]
        den = sub.sum()
        out[r] = sub[2:4].sum() / den if den > 0 else np.nan
    return out


def _f(P, C, S):
    g = S - C
    keep = np.abs(g) >= GAP_MIN
    return float(np.mean((P[keep] - C[keep]) / g[keep])) if keep.sum() else float("nan")


# ===========================================================================
# Pair set: everything an attribution call needs at (t, k)
# ===========================================================================
def build(model, proc, cfg, t, k, rng, cap=None):
    m, V = cfg["m"], proc.V
    Xe = proc.sample(N_SEQS, cfg["seq_len"], rng)
    lo, hi = topk_pairs(Xe, t, k, rng)
    if len(lo) < MIN_PAIRS:
        return None
    if cap:
        lo, hi = lo[:cap], hi[:cap]
    lo_ext, _ = make_ext(Xe, lo, t, k)
    hi_ext, hi_cl = make_ext(Xe, hi, t, k)
    re = t + k
    rec_src = record_component_writes(model, hi_ext, re)        # source readout writes
    C = pclose(model, lo_ext, t, k, {}, [])
    S = pclose(model, hi_ext, t, k, {}, [])
    Sora = oracle_pclose(proc, Xe, hi, t, k, hi_cl, m)
    oe = float(np.nanmean(np.abs(S - Sora)))
    Ct, vt = facet_observable(_m1joint(model, lo_ext, t, m, V), "top_type", V, m)
    return dict(t=t, k=k, n=len(lo), lo_ext=lo_ext, hi_ext=hi_ext, rec_src=rec_src,
                C=C, S=S, oe=oe, Ct=Ct, vt=vt, Xe=Xe, lo=lo, hi=hi)


def _m1joint(model, ext, t, m, V):
    """The V**m completion joint at t for the m=1 top_type read (unpatched). Built from
    the next-token dist at t only (top_type needs just the m=1 marginal)."""
    probs = splice_logits(model, ext, t, {}, [])[:, t, :]      # (n, V) next-token at t
    n = probs.shape[0]
    J = np.zeros((n, V ** m))
    J[:, ::V ** (m - 1)] = probs                               # put mass on w1, rest uniform-free
    return J


def transport(ps, model, splice):
    P = pclose(model, ps["lo_ext"], ps["t"], ps["k"], ps["rec_src"], splice)
    f = _f(P, ps["C"], ps["S"])
    Pt, _ = facet_observable(_m1joint_patched(model, ps, splice), "top_type",
                             model.cfg.vocab, 3)
    drag = _drag(Pt, ps["Ct"], ps["vt"])
    return f, drag


def _m1joint_patched(model, ps, splice):
    V = model.cfg.vocab
    probs = splice_logits(model, ps["lo_ext"], ps["t"] + ps["k"], ps["rec_src"], splice)
    nt = probs[:, ps["t"], :]
    J = np.zeros((nt.shape[0], V ** 3))
    J[:, ::V ** 2] = nt
    return J


def _drag(Pt, Ct, vt):
    keep = vt & np.isfinite(Pt) & np.isfinite(Ct)
    return float(np.mean(np.abs(Pt[keep] - Ct[keep]))) if keep.sum() else float("nan")


# ===========================================================================
# Premise gate + readout-window attribution
# ===========================================================================
def premise(ps, model, cfg):
    H = model.cfg.n_heads
    t = ps["t"]
    emb = {("emb",): np.arange(t + 1)}
    internal = {cid: np.arange(t + 1)
                for cid in component_ids(cfg["layers"], H, include_emb=False)}
    allpre = {**emb, **internal}
    f_emb, _ = transport(ps, model, emb)
    f_int, _ = transport(ps, model, internal)
    f_all, _ = transport(ps, model, allpre)
    int_nec = f_all - f_emb                       # drop from reverting internal (emb source)
    ok = (f_emb >= EMB_MIN) and (f_int <= INT_MAX) and (abs(int_nec) < INT_NEC_MAX)
    return ok, dict(f_emb=f_emb, f_int=f_int, f_all=f_all, int_nec=int_nec)


def readout_units(cfg, model, t, k):
    H = model.cfg.n_heads
    cids = component_ids(cfg["layers"], H, include_emb=False)
    return [(cid, p) for cid in cids for p in range(t, t + k + 1)]


def suff_all(ps, model, units):
    s, dg = {}, {}
    for cid, p in units:
        f, d = transport(ps, model, {cid: [p]})
        s[(cid, p)] = f
        dg[(cid, p)] = d
    return s, dg


def cumulative(ps, model, ranked, ceiling, rng, js):
    units = ranked
    rank_c, rand_c = {}, {}
    for j in js:
        spec = {}
        for cid, p in ranked[:j]:
            spec.setdefault(cid, []).append(p)
        rank_c[j], _ = transport(ps, model, spec)
        rr = []
        for _ in range(N_RAND_DRAWS):
            pick = [units[i] for i in rng.choice(len(units), j, replace=False)]
            sp = {}
            for cid, p in pick:
                sp.setdefault(cid, []).append(p)
            rr.append(transport(ps, model, sp)[0])
        rand_c[j] = float(np.mean(rr))
    return rank_c, rand_c


def necessity(ps, model, units, top, allf):
    """nec(u) = allf - transport(all units except u). Computed for `top` units only."""
    nec = {}
    base = {}
    for cid, p in units:
        base.setdefault(cid, []).append(p)
    for cid, p in top:
        sp = {c: list(ps_) for c, ps_ in base.items()}
        sp[cid] = [pp for pp in sp[cid] if pp != p]
        f, _ = transport(ps, model, sp)
        nec[(cid, p)] = allf - f
    return nec


# ===========================================================================
# Calibration / references (recommend the TBD thresholds)
# ===========================================================================
def calibrate(model, proc, cfg, positions, seed, cap):
    rng = np.random.default_rng(seed)
    for t in positions:
        for k in (1, 2):
            ps = build(model, proc, cfg, t, k, rng, cap=cap)
            if ps is None:
                print(f"t={t} k={k}: < {MIN_PAIRS} matched pairs -> skip")
                continue
            ok, pm = premise(ps, model, cfg)
            print(f"\n=== t={t} k={k} | n={ps['n']} | oe={ps['oe']:.3f} ===")
            print(f"premise: f_emb={pm['f_emb']:+.3f} f_int={pm['f_int']:+.3f} "
                  f"int_nec={pm['int_nec']:+.3f} -> {'PASS' if ok else 'FAIL'}")
            units = readout_units(cfg, model, t, k)
            ceiling, _ = transport(ps, model, {cid: list(range(t, t + k + 1))
                                               for cid in
                                               component_ids(cfg["layers"],
                                                             model.cfg.n_heads, False)})
            print(f"all-readout-window ceiling f={ceiling:+.3f}  ({len(units)} units)")
            suff, drag = suff_all(ps, model, units)
            ranked = sorted(units, key=lambda u: suff[u], reverse=True)
            print("top readout units by sufficiency:")
            for u in ranked[:8]:
                print(f"  {str(u[0]):>16} @p={u[1]:<2} suff={suff[u]:+.3f} "
                      f"drag={drag[u]:+.3f}")
            js = sorted({j for j in (1, 2, 3, 4, 6, 8, len(units)) if j <= len(units)})
            rank_c, rand_c = cumulative(ps, model, ranked, ceiling, rng, js)
            print("cumulative (ranked vs random, frac of ceiling):")
            den = ceiling if abs(ceiling) > 1e-6 else 1.0
            for j in js:
                print(f"  j={j:>3}: ranked={rank_c[j]:+.3f} ({rank_c[j]/den:5.0%}) "
                      f"random={rand_c[j]:+.3f} margin={rank_c[j]-rand_c[j]:+.3f}")
            nec = necessity(ps, model, units, ranked[:6], ceiling)
            print("necessity / redundancy (top units):")
            for u in ranked[:6]:
                print(f"  {str(u[0]):>16} @p={u[1]:<2} suff={suff[u]:+.3f} "
                      f"nec={nec[u]:+.3f} gap={suff[u]-nec[u]:+.3f}")
    print("\n(reference only — set SAT_K/LOCUS_MARGIN/REDUND_GAP from these:")
    print(" ceiling = all-readout-window f; floor = the random-component cumulative.")
    print(" detectability is shown empirically by ranked clearing the random floor.)")


# ===========================================================================
# Verdict (registered)
# ===========================================================================
def cell_verdict(ps, model, cfg):
    if ps["oe"] > OE_BAND:
        return "OBS_DRIFT", {}
    ok, pm = premise(ps, model, cfg)
    if not ok:
        return "PREMISE_FAIL", pm
    t, k = ps["t"], ps["k"]
    rng = np.random.default_rng(0)
    units = readout_units(cfg, model, t, k)
    ceiling, _ = transport(ps, model, {cid: list(range(t, t + k + 1))
                                       for cid in component_ids(cfg["layers"],
                                                                model.cfg.n_heads, False)})
    suff, drag = suff_all(ps, model, units)
    ranked = sorted(units, key=lambda u: suff[u], reverse=True)
    js = sorted({j for j in (1, 2, 3, SAT_K, len(units)) if j <= len(units)})
    rank_c, rand_c = cumulative(ps, model, ranked, ceiling, rng, js)
    target = SAT_FRAC * ceiling
    hit = [j for j in js if j <= SAT_K and rank_c[j] >= target
           and rank_c[j] - rand_c[j] >= LOCUS_MARGIN]
    top = ranked[:max(hit[0], 1)] if hit else ranked[:SAT_K]
    nec = necessity(ps, model, units, top, ceiling)
    # top_type drag is descriptive only (read at t; p=t units only) — see F1, not in verdict.
    drag_t = max((drag[u] for u in top if u[1] == ps["t"]), default=float("nan"))
    info = dict(ceiling=ceiling, suff=suff[ranked[0]], topk=hit[0] if hit else None,
                drag_t=drag_t, pm=pm)
    if not hit:
        return "DISTRIBUTED_COUNTER", info
    redundant = any(suff[u] - nec[u] >= REDUND_GAP for u in top)
    return ("REDUNDANT_COUNTER" if redundant else "LOCALIZED_COUNTER"), info


def run(model, proc, cfg, seeds):
    print(f"device: {next(model.parameters()).device}")
    try:                       # HARNESS_FAIL: the splice identities (no-op, completeness)
        _localize_selftest()   # are exact arithmetic, model-independent -> assert before claims
    except AssertionError as e:
        print(f"HALT: localize identity self-test failed: {e}"); return "HARNESS_FAIL"
    _, ok = validity_gate(model, proc, cfg, seeds[0])
    if not ok:
        print("HALT: validity gate failed"); return "HARNESS_FAIL"
    per_k = {k: [] for k in HORIZONS}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        print(f"[seed {seed}]")
        for k, _ in HORIZONS.items():
            cells = []
            for t in POSITIONS:
                ps = build(model, proc, cfg, t, k, rng, cap=PAIR_CAP)
                if ps is None:
                    continue
                v, info = cell_verdict(ps, model, cfg)
                cells.append(v)
                print(f"  t={t:2d} k={k} n={ps['n']:4d} -> {v}")
            kv = _reduce(cells)
            per_k[k].append(kv)
            print(f"  k={k} -> {kv}")
    agg = {f"k{k}": majority_vote(per_k[k], threshold=SEED_MAJORITY,
                                  unstable="SEED_UNSTABLE") for k in HORIZONS}
    print(f"\nper-horizon: {agg}")
    return agg


def _reduce(cells):
    if not cells:
        return "SEED_UNSTABLE"
    for guard in ("HARNESS_FAIL", "OBS_DRIFT", "PREMISE_FAIL"):
        if cells.count(guard) > len(cells) / 2:
            return guard
    sub = [c for c in cells if c not in ("HARNESS_FAIL", "OBS_DRIFT", "PREMISE_FAIL")]
    if not sub:
        return max(set(cells), key=cells.count)
    return max(set(sub), key=sub.count)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/dyck2-L4")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--positions", type=int, nargs="+", default=[12, 20])
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--seed", type=int, default=700)
    args = ap.parse_args(argv)

    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    require_expected_config({**cfg, "burn_in": cfg.get("burn_in", 4)})
    model = load_model(args.outdir, cfg, proc)
    print(f"device: {next(model.parameters()).device}")
    if args.calibrate:
        validity_gate(model, proc, cfg, args.seed)
        calibrate(model, proc, cfg, args.positions, args.seed, args.cap)
    else:
        run(model, proc, cfg, SEEDS)


if __name__ == "__main__":
    main()
