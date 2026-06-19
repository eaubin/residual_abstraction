"""CLI harnesses for long-running worker/reviewer agent loops.

A harness is one ongoing conversation with one CLI-backed agent. The driver in
``experiment_agent_loop.py`` only calls ``send()``; engine-specific details stay
here.

This module intentionally has no third-party dependencies. It uses local
``codex`` and ``claude`` CLIs when available.
"""

from __future__ import annotations

import abc
import asyncio
import json
import os
import tempfile
import textwrap
import uuid
from collections.abc import Callable
from pathlib import Path


def _default_progress(line: str) -> None:
    print(line, flush=True)


# A Claude reviewer (under dontAsk) may read, search, and run commands so it can
# verify the worker's claims by execution — but not mutate the record. Edit/Write
# are denied by dontAsk (not whitelisted); git-writes are denied explicitly so a
# reviewer cannot commit/push/reset/checkout over the work under review.
REVIEWER_ALLOWED_TOOLS = ["Read", "Grep", "Glob", "Bash"]
REVIEWER_DISALLOWED_TOOLS = [
    "Edit", "Write", "NotebookEdit", "MultiEdit",
    "Bash(git commit:*)", "Bash(git push:*)", "Bash(git reset:*)",
    "Bash(git checkout:*)", "Bash(git rebase:*)", "Bash(git merge:*)",
]


class Harness(abc.ABC):
    kind = "base"

    def __init__(self, persona: str, *, model: str | None = None,
                 cwd: str | Path = ".", timeout: float | None = None,
                 progress: Callable[[str], None] | None = _default_progress) -> None:
        self.persona = persona
        self.model = model
        self.cwd = str(Path(cwd).resolve())
        self.timeout = timeout
        self.progress = progress
        # Normalized usage from the most recent send(), or None if unavailable.
        # Schema: total_input_tokens, cached_input_tokens (cache-read/hit),
        # output_tokens, cost_usd (float or None).
        self.last_usage: dict | None = None

    # asyncio's default StreamReader line limit is 64 KiB; agent events embed
    # large tool outputs, so raise it well past any single JSONL line.
    _STREAM_LIMIT = 16 * 1024 * 1024

    async def _run_streaming(self, argv: list[str],
                             event_log: Path | None) -> tuple[str, str, int]:
        """Run a subprocess, streaming stdout JSONL line-by-line: append each
        line to ``event_log`` immediately and emit a compact progress line, so
        a long turn shows live activity instead of buffering until it returns.
        Returns (full_stdout, stderr, returncode). Enforces ``self.timeout``.
        """
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=os.environ.copy(),
            limit=self._STREAM_LIMIT,
        )
        out_chunks: list[str] = []
        err_chunks: list[str] = []
        sink = open(event_log, "w") if event_log is not None else None

        async def pump_out() -> None:
            assert proc.stdout is not None
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace")
                out_chunks.append(line)
                if sink is not None:
                    sink.write(line)
                    sink.flush()
                if self.progress is not None:
                    summary = self._summarize(line)
                    if summary:
                        self.progress(summary)

        async def pump_err() -> None:
            assert proc.stderr is not None
            err_chunks.append((await proc.stderr.read()).decode(errors="replace"))

        try:
            await asyncio.wait_for(asyncio.gather(pump_out(), pump_err()),
                                   timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            if sink is not None:
                sink.close()
            raise RuntimeError(f"{self.kind} turn timed out after {self.timeout}s")
        await proc.wait()
        if sink is not None:
            sink.close()
        return "".join(out_chunks), "".join(err_chunks), proc.returncode

    def _summarize(self, line: str) -> str | None:
        """Compact one-line progress description for a stdout event, or None to
        stay quiet. Overridden per engine."""
        return None

    async def start(self) -> None:
        """Open the underlying session if the implementation needs it."""

    @abc.abstractmethod
    async def send(self, prompt: str, *, event_log: Path | None = None) -> str:
        """Send one user turn and return the final text reply.

        If ``event_log`` is given and the engine exposes a per-turn event stream
        (tool calls, files read, reasoning), the raw stream is written there
        incrementally as the turn runs.
        """

    async def close(self) -> None:
        """Tear down the session if needed."""

    async def __aenter__(self) -> "Harness":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


class CodexHarness(Harness):
    kind = "codex"

    def __init__(self, persona: str, *, model: str | None = None,
                 cwd: str | Path = ".", sandbox: str = "workspace-write",
                 danger: bool = False, timeout: float | None = None) -> None:
        super().__init__(persona, model=model, cwd=cwd, timeout=timeout)
        self.sandbox = sandbox
        self.danger = danger
        self._session_id: str | None = None

    def _initial_argv(self, out_path: str, prompt: str) -> list[str]:
        argv = [
            "codex", "exec",
            "--json",
            "--cd", self.cwd,
            "--output-last-message", out_path,
        ]
        if self.danger:
            # No --sandbox: bypass means codex applies no sandbox of its own and
            # relies on an external one (e.g. nono). Passing both is contradictory
            # and, nested, codex's seatbelt profile fails to initialize.
            argv.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            argv += ["--sandbox", self.sandbox]
        if self.model:
            argv += ["--model", self.model]
        argv.append(prompt)
        return argv

    def _resume_argv(self, out_path: str, prompt: str) -> list[str]:
        assert self._session_id is not None
        argv = [
            "codex", "exec", "resume", self._session_id,
            "--json",
            "--output-last-message", out_path,
        ]
        if self.danger:
            argv.append("--dangerously-bypass-approvals-and-sandbox")
        if self.model:
            argv += ["--model", self.model]
        argv.append(prompt)
        return argv

    async def send(self, prompt: str, *, event_log: Path | None = None) -> str:
        full_prompt = prompt if self._session_id else (
            f"{self.persona}\n\n----\n\n{prompt}"
        )
        with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as tf:
            out_path = tf.name
        argv = (self._initial_argv(out_path, full_prompt)
                if self._session_id is None
                else self._resume_argv(out_path, full_prompt))
        # _run_streaming reads stdout incrementally, writing the event log and
        # emitting progress as the turn runs. (stdin is DEVNULL there, required:
        # codex exec reads a piped stdin as a <stdin> block and would hang.)
        out, stderr, returncode = await self._run_streaming(argv, event_log)
        self._capture_session_id(out)
        self.last_usage = self._parse_usage(out)
        try:
            reply = Path(out_path).read_text().strip()
        except FileNotFoundError:
            reply = ""
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass
        if returncode != 0 or not reply:
            tail = "\n".join(stderr.strip().splitlines()[-8:])
            raise RuntimeError(
                f"codex turn failed with code {returncode}"
                + (f"\n{tail}" if tail else "")
            )
        return reply

    def _summarize(self, line: str) -> str | None:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return None
        etype = ev.get("type")
        if etype not in ("item.started", "item.completed"):
            return None
        item = ev.get("item") or {}
        itype = item.get("type")
        # agent_message only carries text at completion; everything else is
        # shown once at start to avoid duplicate started/completed lines.
        if itype == "agent_message":
            if etype != "item.completed":
                return None
            txt = (item.get("text") or "").strip().replace("\n", " ")
            return "    · " + textwrap.shorten(txt, 80) if txt else None
        if etype != "item.started":
            return None
        if itype == "file_change":
            names = [Path(c.get("path", "?")).name for c in item.get("changes", [])]
            return "    · edit " + ", ".join(names)
        if itype in ("command_execution", "local_shell_call", "exec_command"):
            cmd = item.get("command") or item.get("cmd") or ""
            return "    · run " + textwrap.shorten(str(cmd), 80)
        return f"    · {itype}" if itype else None

    @staticmethod
    def _parse_usage(out: str) -> dict | None:
        """Sum usage across ``turn.completed`` events. codex reports
        ``input_tokens`` as the total prompt size with ``cached_input_tokens``
        a subset of it; codex does not report a dollar cost."""
        total_in = cached = out_tok = 0
        found = False
        for line in out.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "turn.completed":
                u = ev.get("usage") or {}
                total_in += int(u.get("input_tokens", 0))
                cached += int(u.get("cached_input_tokens", 0))
                out_tok += int(u.get("output_tokens", 0))
                found = True
        if not found:
            return None
        return {"total_input_tokens": total_in, "cached_input_tokens": cached,
                "output_tokens": out_tok, "cost_usd": None}

    def _capture_session_id(self, stdout: str) -> None:
        if self._session_id:
            return
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("session_id", "thread_id", "conversation_id"):
                val = event.get(key)
                if val:
                    self._session_id = str(val)
                    return
            nested = event.get("thread") or event.get("session") or {}
            if isinstance(nested, dict):
                val = nested.get("id")
                if val:
                    self._session_id = str(val)
                    return


class ClaudeHarness(Harness):
    kind = "claude"

    def __init__(self, persona: str, *, model: str | None = None,
                 cwd: str | Path = ".", permission_mode: str = "bypassPermissions",
                 timeout: float | None = None, reply_format: str = "stream-json",
                 allowed_tools: list[str] | None = None,
                 disallowed_tools: list[str] | None = None) -> None:
        super().__init__(persona, model=model, cwd=cwd, timeout=timeout)
        self.permission_mode = permission_mode
        self.reply_format = reply_format
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools
        self._session_id = str(uuid.uuid4())
        self._started = False

    async def send(self, prompt: str, *, event_log: Path | None = None) -> str:
        stream = self.reply_format == "stream-json"
        argv = ["claude", "--print", "--output-format", self.reply_format]
        # --allowed/--disallowedTools are variadic, so they must be followed by
        # another flag (below) — never the prompt positional, which they would
        # otherwise swallow. disallowed overrides allowed.
        if self.allowed_tools:
            argv += ["--allowedTools", ",".join(self.allowed_tools)]
        if self.disallowed_tools:
            argv += ["--disallowedTools", ",".join(self.disallowed_tools)]
        if stream:
            # stream-json requires --verbose with --print.
            argv.append("--verbose")
        if self._started:
            # Continue the same conversation; the persona was set at creation.
            argv += ["--resume", self._session_id]
        else:
            argv += ["--session-id", self._session_id,
                     "--system-prompt", self.persona]
        argv += ["--permission-mode", self.permission_mode]
        if self.model:
            argv += ["--model", self.model]
        argv.append(prompt)
        # In text mode stdout is the plain reply, not events: don't tee it to
        # the event log (and _summarize stays quiet on non-JSON lines).
        out, stderr, returncode = await self._run_streaming(
            argv, event_log if stream else None)
        if stream:
            reply, is_error = self._parse_stream(out)
            self.last_usage = self._parse_usage(out)
        else:
            reply, is_error = out.strip(), False
            self.last_usage = None  # text format carries no usage
        if returncode != 0 or is_error or not reply:
            tail = "\n".join(stderr.strip().splitlines()[-8:])
            raise RuntimeError(
                f"claude turn failed with code {returncode}"
                + (" (result is_error)" if is_error else "")
                + (f"\n{tail}" if tail else "")
            )
        self._started = True
        return reply

    def _summarize(self, line: str) -> str | None:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return None
        if ev.get("type") != "assistant":
            return None
        out: list[str] = []
        for block in ev.get("message", {}).get("content", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                txt = block.get("text", "").strip().replace("\n", " ")
                if txt:
                    out.append("    · " + textwrap.shorten(txt, 80))
            elif block.get("type") == "tool_use":
                inp = block.get("input") or {}
                hint = (inp.get("file_path") or inp.get("path")
                        or inp.get("command") or inp.get("pattern") or "")
                out.append(f"    · {block.get('name')} "
                           f"{textwrap.shorten(str(hint), 60)}".rstrip())
        return "\n".join(out) if out else None

    @staticmethod
    def _parse_stream(out: str) -> tuple[str, bool]:
        """Extract final text and error flag from a stream-json transcript.

        Prefer the terminal ``result`` event; fall back to concatenated
        assistant text blocks if no result line is present.
        """
        result_text: str | None = None
        result_is_error = False
        chunks: list[str] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "result":
                result_is_error = bool(ev.get("is_error"))
                if isinstance(ev.get("result"), str):
                    result_text = ev["result"]
            elif etype == "assistant":
                for block in ev.get("message", {}).get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        chunks.append(block.get("text", ""))
        reply = result_text if result_text is not None else "".join(chunks)
        return reply.strip(), result_is_error

    @staticmethod
    def _parse_usage(out: str) -> dict | None:
        """Usage from the terminal ``result`` event. Anthropic reports
        ``input_tokens`` exclusive of cache, so total input adds the cache
        creation/read counts; ``total_cost_usd`` gives a real dollar cost."""
        for line in reversed(out.splitlines()):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                u = ev.get("usage") or {}
                inp = int(u.get("input_tokens", 0))
                cc = int(u.get("cache_creation_input_tokens", 0))
                cr = int(u.get("cache_read_input_tokens", 0))
                return {"total_input_tokens": inp + cc + cr,
                        "cached_input_tokens": cr,
                        "output_tokens": int(u.get("output_tokens", 0)),
                        "cost_usd": ev.get("total_cost_usd")}
        return None


def make_harness(kind: str, persona: str, *, model: str | None = None,
                 cwd: str | Path = ".", role: str = "worker",
                 codex_danger: bool = False, timeout: float | None = None,
                 claude_reply_format: str = "stream-json") -> Harness:
    kind = kind.lower()
    if kind == "codex":
        sandbox = "workspace-write" if role == "worker" else "read-only"
        return CodexHarness(persona, model=model, cwd=cwd, sandbox=sandbox,
                            danger=codex_danger, timeout=timeout)
    if kind == "claude":
        # Worker may edit/commit. Reviewer is deny-by-default (dontAsk) but may
        # read, search, and run commands to verify the work — while edits and
        # git-writes stay denied so it cannot mutate the record under review.
        # Without an allow-list, dontAsk denies even Read and review can't start.
        if role == "worker":
            return ClaudeHarness(persona, model=model, cwd=cwd,
                                 permission_mode="bypassPermissions",
                                 timeout=timeout, reply_format=claude_reply_format)
        return ClaudeHarness(persona, model=model, cwd=cwd,
                             permission_mode="dontAsk", timeout=timeout,
                             reply_format=claude_reply_format,
                             allowed_tools=REVIEWER_ALLOWED_TOOLS,
                             disallowed_tools=REVIEWER_DISALLOWED_TOOLS)
    raise ValueError(f"unknown harness {kind!r}; expected codex or claude")
