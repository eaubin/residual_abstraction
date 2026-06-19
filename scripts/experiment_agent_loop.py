"""Run the repository's worker/reviewer experiment loop.

This automates the repetitive parts of the existing practice:

1. worker orients in the repo, writes a preregistration + runnable code, commits;
2. reviewer performs the repository's experiment review and returns an approval marker;
3. worker addresses requested changes and commits; repeat until approved;
4. run the same loop for result/conclusion review.

The script does not replace scientific judgment. It provides consistent prompts,
transcripts, approval markers, and a way to swap Codex/Claude roles.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
import subprocess
import textwrap
from pathlib import Path

from agent_harness import make_harness

ROOT = Path(__file__).resolve().parents[1]
APPROVED = "APPROVED"
CHANGES = "CHANGES_REQUESTED"

ORIENTATION = """
Repository orientation, before doing task work:

Read or inspect the relevant parts of:
- AGENTS.md for standing commitments and library-home rules;
- EXPERIMENT_REVIEW_PROTOCOL.md for preregistration/result review rules;
- INTERVENTION_CLASS_BENCHMARK.md for current Phase 3 intervention-class scope;
- HANDOFF.md for current state and re-entry order;
- EXPERIMENTS.md, especially rows 10-17 and 23-29;
- experiments/29-predicate-targeting.md and out/exp29_pstack-L4.txt;
- experiments 10-17 when read/write intervention history is load-bearing;
- FORMALISM.md section 6.1 when verdict partitions or tolerance branches appear.

Terminology discipline:
- widely used terms are allowed when they are standard;
- repo-local terms must preserve their scope indices;
- experiment-local labels must stay local unless deliberately promoted;
- avoid broad labels when the measured construct is narrow.
""".strip()

WORKER_PERSONA = f"""
You are the WORKER agent for residual_abstraction experiments and their
supporting implementation work.

Your job is to create or revise experiment artifacts, not to merely propose them.
Follow the repository method: every claim should be checkable, every failure
branch typed, and privileged ground truth evaluation-only unless an explicit
control says otherwise.

{ORIENTATION}

Worker rules:
- For preregistration work, produce both artifacts before committing: the
  experiment writeup and runnable implementation; do not run the claim-producing
  experiment before preregistration review is approved; and include guards,
  self-checks, output tables, verdict predicates, and halt conditions in code
  before that review.
- For supporting implementation work (helpers, preflights, parity checks), there
  is no preregistration pause and no conclusion: build the helpers in their
  correct library home, wire up self-tests and stable outputs, and you may run
  non-claim checks freely.
- Keep concluded scripts frozen unless the task explicitly requires a reviewed
  reproducibility fix.
- Prefer living library helpers over importing from frozen experiment scripts.
- Commit your completed changes. Do not include unrelated user changes.
- End your final reply with exactly these marker lines:
  WORKER_DONE: yes
  COMMIT: <hash or none>
""".strip()

REVIEWER_PERSONA = f"""
You are the REVIEWER agent for residual_abstraction experiments and their
supporting implementation work.

For claim-producing work, use EXPERIMENT_REVIEW_PROTOCOL.md: this is not generic
code review, and you must evaluate whether the experiment, implementation,
verdict logic, and conclusion all refer to the same registered construct. For
supporting implementation work (helpers, preflights, parity checks), apply the
protocol's terminology, library-home, and construct-discipline rules together
with ordinary code-review judgment.

{ORIENTATION}

Reviewer rules:
- Do not edit files or commit.
- Lead with findings, ordered by severity, with file/line references.
- Check conceptual alignment before style.
- Explicitly check LLM-work creep and maintainability regressions.
- For preregistration review, decide whether the run may proceed.
- For result review, decide whether the conclusion can be accepted.
- For implementation review, decide whether the implementation can be accepted:
  correct library home, passing self-tests, stable outputs, terminology
  discipline, and no claim-shaped language in comments or docs.
- End your final reply with exactly one of:
  REVIEW_DECISION: APPROVED
  REVIEW_DECISION: CHANGES_REQUESTED
""".strip()


def git_status() -> str:
    proc = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                          text=True, capture_output=True, check=True)
    return proc.stdout.strip()


def git_head() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          text=True, capture_output=True, check=True)
    return proc.stdout.strip()


def git_range_patch(before: str, after: str) -> str:
    """Log + diff for everything the worker committed this turn (empty if none)."""
    if before == after:
        return ""
    log = subprocess.run(["git", "log", "--oneline", f"{before}..{after}"],
                         cwd=ROOT, text=True, capture_output=True, check=True).stdout
    diff = subprocess.run(["git", "diff", f"{before}..{after}"],
                          cwd=ROOT, text=True, capture_output=True, check=True).stdout
    return f"{log}\n{diff}"


def task_text(args: argparse.Namespace) -> str:
    chunks: list[str] = []
    if args.task:
        chunks.append(args.task.strip())
    if args.task_file:
        chunks.append(Path(args.task_file).read_text().strip())
    if not chunks:
        raise SystemExit("Provide --task or --task-file.")
    return "\n\n".join(chunks)


def transcript_dir(args: argparse.Namespace) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = args.slug.replace("/", "-")
    out = ROOT / args.transcript_dir / f"{stamp}-{args.mode}-{slug}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_turn(out: Path, idx: int, role: str, prompt: str, reply: str | None = None) -> None:
    (out / f"{idx:02d}-{role}-prompt.md").write_text(prompt)
    if reply is not None:
        (out / f"{idx:02d}-{role}-reply.md").write_text(reply)


def usage_line(usage: dict | None) -> str:
    """One-line per-turn usage summary for stdout."""
    if not usage:
        return "tokens: n/a"
    ti = usage["total_input_tokens"]
    ci = usage["cached_input_tokens"]
    ratio = ci / ti if ti else 0.0
    cost = usage.get("cost_usd")
    cost_s = f"  ${cost:.4f}" if cost is not None else ""
    return (f"tokens: {ti:,} in ({ci:,} cached, {ratio:.0%})  "
            f"{usage['output_tokens']:,} out{cost_s}")


def _sum_usage(rows: list[dict]) -> dict:
    ti = ci = ot = 0
    cost = 0.0
    any_cost = False
    for r in rows:
        if "total_input_tokens" not in r:
            continue
        ti += r["total_input_tokens"]
        ci += r["cached_input_tokens"]
        ot += r["output_tokens"]
        if r.get("cost_usd") is not None:
            cost += r["cost_usd"]
            any_cost = True
    return {"total_input_tokens": ti, "cached_input_tokens": ci,
            "output_tokens": ot, "cost_usd": cost if any_cost else None}


def write_usage(out: Path, rows: list[dict]) -> None:
    """Persist per-turn token accounting as usage.json and usage.md."""
    totals = _sum_usage(rows)
    (out / "usage.json").write_text(
        json.dumps({"turns": rows, "totals": totals}, indent=2) + "\n")

    def cells(u: dict) -> str:
        ti = u["total_input_tokens"]
        ci = u["cached_input_tokens"]
        ratio = f"{ci / ti:.0%}" if ti else "-"
        cost = u.get("cost_usd")
        cost_s = f"${cost:.4f}" if cost is not None else "-"
        return f"{ti:,} | {ci:,} | {ratio} | {u['output_tokens']:,} | {cost_s}"

    lines = ["# Token usage", "",
             "| turn | role | engine | input | cached | cache% | output | cost |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if "total_input_tokens" in r:
            lines.append(f"| {r['turn']} | {r['role']} | {r['engine']} | {cells(r)} |")
        else:
            lines.append(f"| {r['turn']} | {r['role']} | {r['engine']} | n/a | - | - | - | - |")
    lines.append(f"| **total** |  |  | {cells(totals)} |")
    lines += ["", "Cost is reported for Claude turns only; Codex reports tokens "
              "without a dollar figure. cache% is cache-read / total input — a "
              "high ratio on later turns means context rode cheaply.", ""]
    (out / "usage.md").write_text("\n".join(lines))


def review_decision(text: str) -> str:
    m = re.search(r"^REVIEW_DECISION:\s*(APPROVED|CHANGES_REQUESTED)\s*$",
                  text, flags=re.MULTILINE)
    return m.group(1) if m else CHANGES


def worker_initial_prompt(mode: str, task: str) -> str:
    if mode == "prereg":
        objective = """
Create the next experiment preregistration and runnable implementation for the
task below. Commit the preregistration state before the first run. Do not run
the claim-producing experiment.
""".strip()
    elif mode == "impl":
        objective = """
Implement the supporting (non-claim) task below: build the reusable helpers in
their correct library home, wire up self-tests and stable output tables, and
reproduce any stated parity/preflight targets. This is scaffolding, not a
claim-producing run, so there is no preregistration pause and no conclusion to
write. Run the non-claim checks, then commit the completed implementation.
""".strip()
    else:
        objective = """
Run the approved registered experiment for the task below, write the result and
conclusion, update required records, and commit. Preserve the registered verdict
logic; do not reinterpret skipped or failed predicates as successes.
""".strip()
    return f"""{objective}

Task:
{task}

Before finishing, run appropriate selftests or cheap checks. End with the
required WORKER_DONE marker and commit hash.
""".strip()


def reviewer_prompt(mode: str, task: str, worker_reply: str) -> str:
    review_kind = {
        "prereg": "pre-registration",
        "impl": "implementation",
        "result": "result/conclusion",
    }[mode]
    return f"""Review the latest committed {review_kind} work for this task.

Task:
{task}

Worker final reply:
{worker_reply}

Use the repository review protocol. Decide whether this {review_kind} is
approved or needs changes. End with REVIEW_DECISION.
""".strip()


def worker_revision_prompt(mode: str, task: str, review: str) -> str:
    return f"""Address this reviewer feedback for the {mode} workflow. Make the necessary
repo edits, run appropriate checks, and commit.

Task:
{task}

Reviewer feedback:
{review}

End with the required WORKER_DONE marker and commit hash.
""".strip()


async def run_loop(args: argparse.Namespace) -> int:
    task = task_text(args)
    if not args.dry_run and not args.allow_dirty:
        dirty = git_status()
        if dirty:
            raise SystemExit(
                "Working tree is dirty; commit or stash first, or pass "
                "--allow-dirty. The worker commits its own changes and could "
                "otherwise sweep up unrelated edits:\n" + dirty
            )
    out = transcript_dir(args)
    (out / "task.md").write_text(task)
    (out / "initial-git-status.txt").write_text(git_status() + "\n")

    worker = make_harness(args.worker, WORKER_PERSONA, model=args.worker_model,
                          cwd=ROOT, role="worker", codex_danger=args.codex_danger,
                          timeout=args.turn_timeout, claude_reply_format=args.reply_format)
    reviewer = make_harness(args.reviewer, REVIEWER_PERSONA, model=args.reviewer_model,
                            cwd=ROOT, role="reviewer", codex_danger=False,
                            timeout=args.turn_timeout, claude_reply_format=args.reply_format)

    print(f"Transcript: {out}")
    print(f"Worker: {worker.kind}  Reviewer: {reviewer.kind}  Mode: {args.mode}")
    if args.dry_run:
        p1 = worker_initial_prompt(args.mode, task)
        write_turn(out, 1, "worker", p1)
        p2 = reviewer_prompt(args.mode, task, "<worker reply will be inserted here>")
        write_turn(out, 2, "reviewer", p2)
        print("Dry run: wrote prompt templates only.")
        return 0

    usage_rows: list[dict] = []

    def record(turn: int, role: str, engine: str, usage: dict | None) -> None:
        row = {"turn": turn, "role": role, "engine": engine}
        if usage:
            row.update(usage)
        usage_rows.append(row)
        print("  " + usage_line(usage))
        write_usage(out, usage_rows)

    async with worker, reviewer:
        prompt = worker_initial_prompt(args.mode, task)
        for round_no in range(1, args.max_rounds + 1):
            turn = 2 * round_no - 1
            print(f"\n[worker round {round_no}]", flush=True)
            head_before = git_head()
            worker_reply = await worker.send(
                prompt, event_log=out / f"{turn:02d}-worker-events.jsonl")
            write_turn(out, turn, "worker", prompt, worker_reply)
            patch = git_range_patch(head_before, git_head())
            if patch:
                (out / f"{turn:02d}-worker-commits.patch").write_text(patch)
            record(turn, "worker", worker.kind, worker.last_usage)
            print(textwrap.shorten(worker_reply.replace("\n", " "), width=240))

            rprompt = reviewer_prompt(args.mode, task, worker_reply)
            print(f"\n[reviewer round {round_no}]", flush=True)
            review = await reviewer.send(
                rprompt, event_log=out / f"{turn + 1:02d}-reviewer-events.jsonl")
            write_turn(out, turn + 1, "reviewer", rprompt, review)
            record(turn + 1, "reviewer", reviewer.kind, reviewer.last_usage)
            print(textwrap.shorten(review.replace("\n", " "), width=240))

            decision = review_decision(review)
            (out / f"{turn + 1:02d}-decision.txt").write_text(decision + "\n")
            if decision == APPROVED:
                (out / "final-git-status.txt").write_text(git_status() + "\n")
                print("\nApproved.")
                return 0
            prompt = worker_revision_prompt(args.mode, task, review)

    (out / "final-git-status.txt").write_text(git_status() + "\n")
    print(f"\nStopped after {args.max_rounds} rounds without approval.")
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run worker/reviewer experiment loops.")
    ap.add_argument("mode", choices=["prereg", "impl", "result"],
                    help="which review pause to automate: prereg/result for "
                         "claim-producing experiments, impl for supporting work")
    ap.add_argument("--slug", required=True, help="short label for transcript directory")
    ap.add_argument("--task", help="task prompt text")
    ap.add_argument("--task-file", help="file containing task prompt text")
    ap.add_argument("--worker", default="codex", choices=["codex", "claude"])
    ap.add_argument("--reviewer", default="codex", choices=["codex", "claude"])
    ap.add_argument("--worker-model")
    ap.add_argument("--reviewer-model")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--transcript-dir", default=".agent_runs")
    ap.add_argument("--dry-run", action="store_true", help="write prompts without invoking agents")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="allow a real worker loop to start with uncommitted changes")
    ap.add_argument("--codex-danger", action="store_true",
                    help="Codex worker bypasses its own approvals+sandbox; REQUIRED when "
                         "the loop runs inside an external sandbox such as `nono run` "
                         "(codex cannot nest its seatbelt sandbox and will fail otherwise)")
    ap.add_argument("--turn-timeout", type=float, default=None,
                    help="per-agent-turn timeout in seconds (default: none; "
                         "claim-producing runs can take hours)")
    ap.add_argument("--reply-format", choices=["stream-json", "text"],
                    default="stream-json",
                    help="Claude output format: stream-json captures per-turn "
                         "event logs (default); text is a parser-free fallback")
    args = ap.parse_args(argv)
    return asyncio.run(run_loop(args))


if __name__ == "__main__":
    raise SystemExit(main())
