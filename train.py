"""
train.py — train the small transformer and build the experiment's cache.

CONTEXT (see README.md): this script produces the *dataset of the actual
experiment*: a cache pairing, for every position t of held-out sequences,

  resid[t]   the model's residual-stream vector (the candidate abstraction),
  belief[t]  the exact Bayes posterior over hidden states (the known minimal
             sufficient statistic — the ideal abstraction),
  mgram[t]   the exact distribution over the next m tokens (the truncated
             completion measure — the concrete semantics).

Note that beliefs and m-gram distributions come from the *generating process*
in closed form, not from model samples: abstraction error downstream is
measured against ground truth, never estimated. analysis.py and refine.py are
numpy-only consumers of this cache, so the heavy dependency (torch) is
quarantined here.

Pairing convention: residual at position t has seen tokens 0..t and predicts
token t+1; it is paired with the belief AFTER observing tokens 0..t.
The first `burn_in` positions are dropped from the cache (the model has
little context there and the Bayes filter is still synchronizing; including
them only adds a known, uninteresting error floor).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, pick_device
from processes import PROCESSES


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", choices=PROCESSES, default="z1r")
    ap.add_argument("--steps", type=int, default=None,
                    help="default: 1500 for z1r, 6000 for mess3")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=None,
                    help="default: 1 for z1r, 2 for mess3")
    ap.add_argument("--m", type=int, default=3,
                    help="completion horizon (V**m outcomes; keep small)")
    ap.add_argument("--eval-seqs", type=int, default=1500)
    ap.add_argument("--burn-in", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache-only", action="store_true",
                    help="skip training: load <outdir>/model.pt and rebuild "
                    "cache.npz on CPU (deterministic). This is how the "
                    "gitignored cache is regenerated in a clean checkout "
                    "from the tracked checkpoint; the checkpoint and "
                    "config.json are left untouched.")
    args = ap.parse_args(argv)

    proc = PROCESSES[args.process]()
    steps = args.steps or {"z1r": 1500, "mess3": 6000}[args.process]
    layers = args.layers or {"z1r": 1, "mess3": 2}[args.process]
    outdir = args.outdir or os.path.join("out", args.process)
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cpu" if args.cache_only else pick_device()
    cfg = GPTConfig(vocab=proc.V, seq_len=args.seq_len,
                    d_model=args.d_model, n_layers=layers)
    model = GPT(cfg).to(device)

    # Optimality yardstick: the exact entropy rate achievable by the Bayes
    # filter on this data. A well-trained model's loss should approach it —
    # a precondition for its residuals approximating the belief geometry.
    probe = proc.sample(400, args.seq_len, np.random.default_rng(args.seed + 1))
    opt_nll = 0.0
    for row in probe:
        b = proc.pi.copy()
        for s in row:
            opt_nll -= np.log((b @ proc.T[s]).sum())
            b, _ = proc.belief_update(b, s)
    opt_nll /= probe.size
    print(f"[train] process={proc.name} device={device} "
          f"optimal NLL/token={opt_nll:.4f}")

    if args.cache_only:
        model.load_state_dict(torch.load(os.path.join(outdir, "model.pt"),
                                         map_location="cpu"))
        print(f"[train] --cache-only: loaded {outdir}/model.pt; rebuilding "
              "cache on CPU (eval sequences are fixed-seeded, so this is "
              "deterministic; caches recorded from MPS runs match to float "
              "tolerance)")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.01)
        t0 = time.time()
        model.train()
        for step in range(1, steps + 1):
            X = torch.from_numpy(proc.sample(args.batch, args.seq_len + 1, rng))
            X = X.to(device)
            logits = model(X[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, proc.V),
                                   X[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % max(1, steps // 10) == 0 or step == 1:
                gap = loss.item() - opt_nll
                print(f"[train] step {step:5d}/{steps}  loss {loss.item():.4f}"
                      f"  gap-to-optimal {gap:+.4f}  ({time.time()-t0:.0f}s)")

    # ----- build the cache ---------------------------------------------------
    model.eval()
    Xe = proc.sample(args.eval_seqs, args.seq_len, np.random.default_rng(123))
    with torch.no_grad():
        resid_chunks = []
        for i in range(0, len(Xe), 256):
            _, r = model(torch.from_numpy(Xe[i:i + 256]).to(device),
                         return_resid=True)
            resid_chunks.append(r.float().cpu().numpy())
        resid = np.concatenate(resid_chunks)            # (N, L, d)

    keep = slice(args.burn_in, args.seq_len - 1)        # need a next token too
    pos_row = np.arange(args.seq_len)[keep]
    R, B, G, P, Tk = [], [], [], [], []
    for i, row in enumerate(Xe):
        beliefs = proc.beliefs_along(row)
        R.append(resid[i, keep])
        B.append(beliefs[keep])
        G.append(proc.mgram_table(beliefs[keep], args.m))
        P.append(pos_row)
        Tk.append(row[keep])            # current token: for variance audits
    R, B, G, P, Tk = (np.concatenate(a) for a in (R, B, G, P, Tk))
    print(f"[train] cache: {len(R)} (residual, belief, mgram) triples; "
          f"resid dim {R.shape[1]}, completion outcomes {G.shape[1]}")

    np.savez_compressed(
        os.path.join(outdir, "cache.npz"),
        resid=R.astype(np.float32), belief=B.astype(np.float32),
        mgram=G.astype(np.float32), pos=P.astype(np.int64),
        tok=Tk.astype(np.int64),
        m=args.m, process=proc.name, optimal_nll=opt_nll,
    )
    if not args.cache_only:        # never clobber the tracked checkpoint
        torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump({**vars(args), "steps": steps, "layers": layers,
                       "device": device, "optimal_nll": opt_nll}, f, indent=2)
    print(f"[train] wrote {outdir}/cache.npz")


if __name__ == "__main__":
    main()
