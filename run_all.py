"""
run_all.py — orchestrate the full experiment for one process.

  python3 run_all.py --process z1r     # fast pipeline check (~minutes, CPU)
  python3 run_all.py --process mess3   # the fractal + the real curves

Stages (each runnable standalone; see module docstrings + README.md):
  1. train.py    — train transformer, cache (residual, exact belief,
                   exact completion distribution) triples
  2. analysis.py — Shai et al. calibration + sufficiency/completeness curve
  3. refine.py   — CEGAR-style outer loop discovering the complete shell
"""

import argparse
import os

import analysis
import compare
import refine
import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", choices=["z1r", "mess3"], default="z1r")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    outdir = os.path.join("out", args.process)
    train_args = ["--process", args.process, "--outdir", outdir]
    if args.steps:
        train_args += ["--steps", str(args.steps)]
    train.main(train_args)
    print()
    analysis.main(["--outdir", outdir])
    print()
    refine.main(["--outdir", outdir])
    print()
    compare.main(["--outdir", outdir])


if __name__ == "__main__":
    main()
