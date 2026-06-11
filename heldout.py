"""
heldout.py — Experiment 16: position transportability — can the
training/selection protocol repair position entanglement?

CONTEXT (see experiments/16-position-transport.md). Exp 15: learned reads
compute the clean functional at trained positions and invert at unseen
ones. One question: does held-out-position selection (arm A: checkpoint
selection by observable gain on P_val = {12,20}) or mixed-position
training (arm B: minibatches over {8,12,16,20,24}) produce reads that
transport to the unseen P_test = {10,14,22}? Selection is honest
(observable only); rho / EPR / retentions are evaluation-side. Small
write widening (two extra pool draws, <= 15 deg, deduped); T fixed.

Run: python3 heldout.py --outdir out/mess3-L4   (~2-2.5 h)
`--selftest` adds: checkpointing-inert, position-set, and write-dedup
checks.

RESULTS: not yet run.
"""

import os

import json
import numpy as np
import torch

from discover import self_checks
from expcommon import (LAYER, PairSet, adversarial_regime, basis_covariance,
                       build_transform, epr, jeffreys_rows, kl_rows,
                       load_model, make_torch_objective, oblique_patch,
                       observable_refs, optimize_affine, orthonormal,
                       principal_angles_deg, regression_link,
                       reproduce_anchor, reproduce_writes, standard_guard,
                       standard_parser, validity_gate, write_pool)
from midstream import kl_by_horizon
from processes import PROCESSES

REGISTERED = {"kappa": 100.0, "lr": 0.05, "steps": 200, "batch": 64,
              "pairs_disc": 400, "pairs_eval": 600, "basis_seqs": 800,
              "m": 3, "checkpoint_every": 20}
TS_VAL = (12, 20)            # selection positions (registered)
TS_TEST = (10, 14, 22)       # unseen final-evaluation positions
TS_MIX = (8, 12, 16, 20, 24)  # arm-B mixed training positions
SEED_VAL, SEED_MIX = 333, 211
GAIN_ACCEPT = 0.20
RHO_CONSIST = 0.50
DEDUP_COS = 0.999


def is_new_write(u_new, us):
    """Registered dedup rule: reject |cos| > DEDUP_COS with any selected
    back-mapped direction."""
    return all(abs(float(u_new @ u)) <= DEDUP_COS for u in us)


def exp16_self_checks(model, proc, cfg):
    """Registered new checks: checkpointing inert, position sets,
    write-dedup rule."""
    rng = np.random.default_rng(11)
    tgt = torch.from_numpy(rng.standard_normal(8)).float()

    def synth(c_t, w_t, batch, adversarial=True):
        return ((c_t - tgt) ** 2).sum()

    w = rng.standard_normal(8)
    ic = rng.standard_normal(8)
    a = optimize_affine(synth, 10, 8, 0.05, 2, 4, 0, w, ic, True,
                        "selftest", print_every=999)
    b, cks = optimize_affine(synth, 10, 8, 0.05, 2, 4, 0, w, ic, True,
                             "selftest", print_every=999,
                             checkpoint_every=1)
    assert np.array_equal(a, b), "checkpointing not inert"
    assert cks[0][0] == 0 and len(cks) == 3, "checkpoint schedule"
    for ts in (TS_VAL, TS_TEST, TS_MIX):
        ps = PairSet(model, proc, cfg, 12, 3, 12345, 30, layer=LAYER, ts=ts)
        assert [t for t, _ in ps.groups] == sorted(ts), f"ts {ts}"
    u1 = rng.standard_normal(16)
    u1 /= np.linalg.norm(u1)
    u2 = np.zeros(16)
    u2[np.argmin(np.abs(u1))] = 1.0
    u2 -= u1 * (u2 @ u1)
    u2 /= np.linalg.norm(u2)
    assert not is_new_write(u1, [u1]) and is_new_write(u2, [u1]), "dedup"
    print("checkpoint-inert, position-set, and write-dedup checks passed")


def main(argv=None):
    args = standard_parser(REGISTERED).parse_args(argv)
    with open(os.path.join(args.outdir, "config.json")) as f:
        cfg = json.load(f)
    proc = PROCESSES[cfg["process"]]()
    if not standard_guard(args, cfg, proc, "Experiment 16", REGISTERED):
        return
    V, m, d = proc.V, args.m, cfg["d_model"]
    model = load_model(args.outdir, cfg, proc)

    gap_opt, p8 = validity_gate(model, proc, cfg, args.seed)
    if not p8 and not args.selftest and not args.force_invalid:
        print("exiting: validity gate failed.")
        return

    disc = PairSet(model, proc, cfg, args.pairs_disc, m, args.seed + 111,
                   800, layer=LAYER)
    ev_train = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777,
                       800, layer=LAYER)
    self_checks(model, ev_train, LAYER, m, V)
    exp16_self_checks(model, proc, cfg)
    if args.selftest:
        return

    print(f"=== Experiment 16: position transportability | {proc.name} | "
          f"patch L{LAYER} | kappa = {args.kappa:g} | select on "
          f"{list(TS_VAL)}, test on {list(TS_TEST)} ===\n")

    # ----- standard scaffolding -------------------------------------------------
    q_src_d, q_un_d, c_obs = observable_refs(model, disc, d)
    Qc = reproduce_anchor(model, disc, q_src_d, q_un_d, c_obs, d)
    T, Tinv, Qj = build_transform(Qc, d, args.kappa)
    Sig_x = basis_covariance(model, proc, cfg, args.seed, args.basis_seqs)
    rg_adv, _ = adversarial_regime(disc, T, Tinv, Sig_x)
    (a1, s1, w1), (a2, s2, w2) = reproduce_writes(rg_adv, q_src_d, q_un_d,
                                                  Qc, d, args.seed)
    torch_objective = make_torch_objective(model, disc, T, Tinv, q_src_d)
    regression_link(torch_objective, model, disc, rg_adv, q_src_d, w1)

    # write widening (registered rule: two extra draws, <= 15 deg, deduped)
    back_unit = lambda w: (lambda u: u / np.linalg.norm(u))(rg_adv.back(w))
    writes = [("w1", a1, w1), ("w2", a2, w2)]
    us = [back_unit(w1), back_unit(w2)]
    w0w = kl_rows(q_src_d, q_un_d)
    for s_off in (1, 2):
        pool = write_pool(rg_adv, np.zeros((d, 0)), w0w, 1, d,
                          args.seed + s_off)
        cand = sorted((principal_angles_deg(back_unit(w)[:, None], Qc)[0],
                       src, w) for src, w in pool)
        slot = next(((a, src, w) for a, src, w in cand
                     if a <= 15.0 and is_new_write(back_unit(w), us)), None)
        if slot is None:
            print(f"write widening: draw seed+{s_off} yielded no new "
                  "near-plane write (recorded)")
            continue
        a_, src_, w_ = slot
        tag = f"w{len(writes) + 1}"
        writes.append((tag, a_, w_))
        us.append(back_unit(w_))
        print(f"write widening: {tag} = {src_} at {a_:.1f} deg "
              f"(draw seed+{s_off})")
    print()

    # ----- selection set (discovery-side, observable only) ---------------------
    disc_val = PairSet(model, proc, cfg, args.pairs_disc, m,
                       args.seed + SEED_VAL, 800, layer=LAYER, ts=TS_VAL)
    q_src_v, q_un_v, c_obs_val = observable_refs(model, disc_val, d)
    pb = lambda c, w: rg_adv.pull(oblique_patch(c[:, None], w[:, None]))
    gain_train = lambda c, w: c_obs(disc.run(model, pb(c, w)))
    gain_val = lambda c, w: c_obs_val(disc_val.run(model, pb(c, w)))

    # ----- arm A: held-out checkpoint selection ---------------------------------
    def arm_A(tag, w):
        label = f"A/{tag}/id"
        print(f"  optimizing {label}:")
        c_fin, cks = optimize_affine(torch_objective, disc.n, d, args.lr,
                                     args.steps, args.batch, args.seed,
                                     w, w.copy(), True, label,
                                     checkpoint_every=args.checkpoint_every)
        traj = [(step, gain_val(c, w)) for step, c in cks]
        s_best, g_best = max(traj, key=lambda sg: sg[1])
        c_sel = dict(cks)[s_best]
        print(f"    [{label}] val-gain trajectory: "
              + ", ".join(f"{s}:{g:+.0%}" for s, g in traj))
        gt_fin, gt_sel = gain_train(c_fin, w), gain_train(c_sel, w)
        print(f"    [{label}] selected step {s_best} (val {g_best:+.1%}, "
              f"train {gt_sel:+.1%}); final (train {gt_fin:+.1%}, val "
              f"{traj[-1][1]:+.1%})")
        return {"c_sel": c_sel, "c_fin": c_fin, "step": s_best,
                "g_val_sel": g_best, "g_train_sel": gt_sel,
                "g_train_fin": gt_fin}

    print("[arm A] held-out checkpoint selection:")
    A = {}
    for tag, ang, w in writes[:2]:
        A[tag] = arm_A(tag, w)
    p1_rep = (abs(A["w1"]["g_train_fin"] - (-5.482)) <= 0.02
              and abs(A["w2"]["g_train_fin"] - 0.425) <= 0.02)
    print(f"  reproduction (A-final train gains vs exp-15): w1 "
          f"{A['w1']['g_train_fin']:+.1%} (recorded -548.2%), w2 "
          f"{A['w2']['g_train_fin']:+.1%} (recorded +42.5%) — "
          f"{'OK' if p1_rep else 'FAILED'}")
    if not p1_rep and not args.force_invalid:
        print("read reproduction FAILED — registered halt.\n\nverdicts:")
        print("  P1: FAILS (determinism breach; anchors not evaluated)")
        print("  P2-P7: NOT TESTED — reproduction halt "
              "(--force-invalid continues exploratorily)")
        print(f"  P8 validity gate: {'HOLDS' if p8 else 'FAILS'}")
        return
    for tag, ang, w in writes[2:]:
        A[tag] = arm_A(tag, w)
    print()

    # ----- arm B: mixed-position training ---------------------------------------
    print("[arm B] mixed-position training (final read, judged on P_test):")
    disc_mix = PairSet(model, proc, cfg, args.pairs_disc, m,
                       args.seed + SEED_MIX, 800, layer=LAYER, ts=TS_MIX)
    q_src_m, _, _ = observable_refs(model, disc_mix, d)
    torch_objective_mix = make_torch_objective(model, disc_mix, T, Tinv,
                                               q_src_m)
    B = {}
    for tag, ang, w in writes[:2]:
        label = f"B/{tag}/id"
        print(f"  optimizing {label}:")
        B[tag] = optimize_affine(torch_objective_mix, disc_mix.n, d, args.lr,
                                 args.steps, args.batch, args.seed,
                                 w, w.copy(), True, label)
    print()
    np.savez(os.path.join(args.outdir, "exp16_reads.npz"),
             **{f"A_sel_{t}": A[t]["c_sel"] for t in A},
             **{f"A_fin_{t}": A[t]["c_fin"] for t in A},
             **{f"B_fin_{t}": B[t] for t in B},
             **{t: w for t, _, w in writes}, Qc=Qc, Qj=Qj)
    print(f"reads persisted to {args.outdir}/exp16_reads.npz\n")

    # ----- evaluation matrix -----------------------------------------------------
    ev_val = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777,
                     800, layer=LAYER, ts=TS_VAL)
    ev_test = PairSet(model, proc, cfg, args.pairs_eval, m, args.seed + 777,
                      800, layer=LAYER, ts=TS_TEST)
    sets = {"train": ev_train, "val": ev_val, "test": ev_test}
    u_of = {tag: back_unit(w) for tag, _, w in writes}
    patches = {}
    for tag, ang, w in writes:
        u = u_of[tag]
        patches[f"{tag}/clean"] = (tag, np.outer(u, u))
        patches[f"{tag}/A-sel"] = (tag, pb(A[tag]["c_sel"], w))
        patches[f"{tag}/A-fin"] = (tag, pb(A[tag]["c_fin"], w))
        if tag in B:
            patches[f"{tag}/B-fin"] = (tag, pb(B[tag], w))

    def eval_set(pset, label):
        q0 = pset.run(model, None)
        floor = float(kl_by_horizon(q0, pset.p_tgt3, V, m)[m].mean())
        gap = float(kl_by_horizon(q0, pset.p_src3, V, m)[m].mean())
        qs = {k: pset.run(model, P) for k, (_, P) in patches.items()}
        gains = {k: (gap - float(kl_by_horizon(q, pset.p_src3, V, m)[m]
                                 .mean())) / (gap - floor)
                 for k, q in qs.items()}
        rho = {}
        for k, q in qs.items():
            ref = f"{patches[k][0]}/clean"
            if k == ref:
                continue
            rho[k] = float(jeffreys_rows(qs[ref], q).mean()
                           / jeffreys_rows(qs[ref], q0).mean())
        print(f"[{label}] exact closure gains: "
              + ", ".join(f"{k} {gains[k]:+.1%}" for k in patches))
        print(f"[{label}] rho vs same-write clean: "
              + ", ".join(f"{k} {rho[k]:.3f}" for k in rho))
        return gains, rho

    res = {lbl: eval_set(ps, lbl) for lbl, ps in sets.items()}
    g_tr, _ = res["train"]

    # anchors via the train eval set
    q0 = ev_train.run(model, None)
    floor = float(kl_by_horizon(q0, ev_train.p_tgt3, V, m)[m].mean())
    gapv = float(kl_by_horizon(q0, ev_train.p_src3, V, m)[m].mean())
    cl_full = (gapv - float(kl_by_horizon(ev_train.run(model, np.eye(d)),
                                          ev_train.p_src3, V, m)[m].mean())) \
        / (gapv - floor)
    U2 = orthonormal(np.column_stack([u_of["w1"], u_of["w2"]]))
    cl_d2 = (gapv - float(kl_by_horizon(ev_train.run(model, U2 @ U2.T),
                                        ev_train.p_src3, V, m)[m].mean())) \
        / (gapv - floor)
    g_d1 = g_tr["w1/clean"]
    print(f"\nanchors: full {cl_full:.1%}; D2 {cl_d2:.1%}; D1 {g_d1:+.1%}\n")

    # position-t EPR for selected reads, per eval set (evaluation-side)
    print("[EPR] position-t cells for selected reads:")
    for tag in A:
        r, u = T @ A[tag]["c_sel"], u_of[tag]
        for lbl, ps in sets.items():
            cells = []
            for t, idx in ps.groups:
                D_t = (ps.pref_src[t] - ps.pref_tgt[t]).numpy() \
                    .astype(np.float64)
                cells.append((t, epr(D_t[:, t, :], r, u)))
            print(f"  {tag}/A-sel on {lbl}: "
                  + ", ".join(f"t={t}:{v:.3f}" for t, v in cells))

    # own-retention (observable-side analogue, exact-closure based)
    print("\nown-retention gain_test/gain_train (selected reads):")
    for tag in A:
        k = f"{tag}/A-sel"
        gt, gx = g_tr[k], res["test"][0][k]
        rr = gx / gt if abs(gt) > 1e-9 else float("nan")
        print(f"  {k}: train {gt:+.1%}, test {gx:+.1%} (retention {rr:.2f})")

    # ----- verdicts ----------------------------------------------------------------
    print("\nverdicts:")
    p1 = g_d1 >= 0.40 and cl_d2 >= 0.90 * cl_full and p1_rep
    print(f"  P1 anchors + reproduction: D1 {g_d1:+.1%}, D2 {cl_d2:.1%}, "
          f"repro OK — {'HOLDS' if p1 else 'FAILS'}")

    def transportable(tag):
        return (A[tag]["g_val_sel"] >= GAIN_ACCEPT
                and res["test"][0][f"{tag}/A-sel"] >= GAIN_ACCEPT)

    gv2, gx2 = A["w2"]["g_val_sel"], res["test"][0]["w2/A-sel"]
    if transportable("w2"):
        p2v = "TRANSPORTABLE — HOLDS"
    elif gv2 >= GAIN_ACCEPT:
        p2v = "VAL-ONLY (selection overfit to selection positions) — FAILS"
    else:
        p2v = "NOT RESCUED — FAILS"
    print(f"  P2 w2 A-selected (val obs {gv2:+.1%}, test exact "
          f"{gx2:+.1%}; both >= 20% required): {p2v}")
    gb2 = res["test"][0]["w2/B-fin"]
    p3 = gb2 >= GAIN_ACCEPT
    print(f"  P3 w2 B-final on test ({gb2:+.1%} >= 20%): "
          f"{'HOLDS' if p3 else 'FAILS'}")
    accepted_pairs = []
    for tag in A:
        if A[tag]["g_val_sel"] >= GAIN_ACCEPT:
            accepted_pairs.append((f"{tag}/A-sel val", A[tag]["g_val_sel"],
                                   res["val"][0][f"{tag}/A-sel"]))
        if A[tag]["g_train_fin"] >= GAIN_ACCEPT:
            accepted_pairs.append((f"{tag}/A-fin train",
                                   A[tag]["g_train_fin"],
                                   g_tr[f"{tag}/A-fin"]))
    if accepted_pairs:
        p4 = all(abs(o - e) <= 0.10 for _, o, e in accepted_pairs)
        print(f"  P4 observable/exact on {len(accepted_pairs)} accepted "
              "read-set pair(s): "
              + "; ".join(f"{n}: {o:+.1%} vs {e:+.1%}"
                          for n, o, e in accepted_pairs)
              + f" — {'HOLDS' if p4 else 'FAILS (objective hacking)'}")
    else:
        print("  P4: NOT TESTED — nothing accepted at 20%")
    trans = [tag for tag in A if transportable(tag)]
    if trans or p3:
        rhos = {f"{t}/A-sel": res["test"][1][f"{t}/A-sel"] for t in trans}
        if p3:
            rhos["w2/B-fin"] = res["test"][1]["w2/B-fin"]
        p5 = all(v <= RHO_CONSIST for v in rhos.values())
        verdict5 = ("HOLDS" if p5 else
                    "FAILS (position-generic control distinct from clean)")
        print(f"  P5 transportable reads rho(test) <= {RHO_CONSIST}: "
              + ", ".join(f"{k} {v:.3f}" for k, v in rhos.items())
              + f" — {verdict5}")
    else:
        print("  P5: NOT TESTED — no transportable read")
    if len(writes) > 2:
        n_ok = sum(transportable(tag) for tag in A)
        p6 = n_ok >= (len(A) + 1) // 2
        print(f"  P6 write generality ({n_ok}/{len(A)} writes "
              f"transportable; >= half required): "
              f"{'HOLDS' if p6 else 'FAILS'}")
    else:
        print("  P6: NOT TESTED — both extra draws empty")
    gv1 = A["w1"]["g_val_sel"]
    p7 = gv1 >= GAIN_ACCEPT
    print(f"  P7 w1 rescue (A-selected val {gv1:+.1%} >= 20%): "
          f"{'HOLDS' if p7 else 'FAILS'}")
    print(f"  P8 validity gate: {'HOLDS' if p8 else 'FAILS'}")


if __name__ == "__main__":
    main()
