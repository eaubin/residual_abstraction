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
import uuid
from pathlib import Path


class Harness(abc.ABC):
    kind = "base"

    def __init__(self, persona: str, *, model: str | None = None,
                 cwd: str | Path = ".", timeout: float | None = None) -> None:
        self.persona = persona
        self.model = model
        self.cwd = str(Path(cwd).resolve())
        self.timeout = timeout
        # Normalized usage from the most recent send(), or None if unavailable.
        # Schema: total_input_tokens, cached_input_tokens (cache-read/hit),
        # output_tokens, cost_usd (float or None).
        self.last_usage: dict | None = None

    async def _communicate(self, proc: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
        """Wait for the process, enforcing ``self.timeout`` if one is set.

        Default is no timeout: claim-producing runs can take hours.
        """
        try:
            return await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"{self.kind} turn timed out after {self.timeout}s")

    async def start(self) -> None:
        """Open the underlying session if the implementation needs it."""

    @abc.abstractmethod
    async def send(self, prompt: str, *, event_log: Path | None = None) -> str:
        """Send one user turn and return the final text reply.

        If ``event_log`` is given and the engine exposes a per-turn event stream
        (tool calls, files read, reasoning), the raw stream is written there.
        """

    @staticmethod
    def _write_events(event_log: Path | None, raw: str) -> None:
        if event_log is not None and raw.strip():
            Path(event_log).write_text(raw)

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
            "--sandbox", self.sandbox,
            "--output-last-message", out_path,
        ]
        if self.danger:
            argv.append("--dangerously-bypass-approvals-and-sandbox")
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
        proc = await asyncio.create_subprocess_exec(
            *argv,
            # DEVNULL is required: with an inherited/piped stdin, `codex exec`
            # reads it to append a <stdin> block and blocks forever on no EOF.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=os.environ.copy(),
        )
        stdout, stderr = await self._communicate(proc)
        out = stdout.decode()
        # codex exec --json already streams JSONL events to stdout.
        self._write_events(event_log, out)
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
        if proc.returncode != 0 or not reply:
            tail = "\n".join(stderr.decode().strip().splitlines()[-8:])
            raise RuntimeError(
                f"codex turn failed with code {proc.returncode}"
                + (f"\n{tail}" if tail else "")
            )
        return reply

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
                 timeout: float | None = None, reply_format: str = "stream-json") -> None:
        super().__init__(persona, model=model, cwd=cwd, timeout=timeout)
        self.permission_mode = permission_mode
        self.reply_format = reply_format
        self._session_id = str(uuid.uuid4())
        self._started = False

    async def send(self, prompt: str, *, event_log: Path | None = None) -> str:
        stream = self.reply_format == "stream-json"
        argv = ["claude", "--print", "--output-format", self.reply_format]
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
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=os.environ.copy(),
        )
        stdout, stderr = await self._communicate(proc)
        out = stdout.decode()
        if stream:
            self._write_events(event_log, out)
            reply, is_error = self._parse_stream(out)
            self.last_usage = self._parse_usage(out)
        else:
            reply, is_error = out.strip(), False
            self.last_usage = None  # text format carries no usage
        if proc.returncode != 0 or is_error or not reply:
            tail = "\n".join(stderr.decode().strip().splitlines()[-8:])
            raise RuntimeError(
                f"claude turn failed with code {proc.returncode}"
                + (" (result is_error)" if is_error else "")
                + (f"\n{tail}" if tail else "")
            )
        self._started = True
        return reply

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
        # Claude read-only enforcement is not as strong as Codex's sandbox here;
        # the reviewer persona also explicitly forbids edits.
        permission = "bypassPermissions" if role == "worker" else "dontAsk"
        return ClaudeHarness(persona, model=model, cwd=cwd,
                             permission_mode=permission, timeout=timeout,
                             reply_format=claude_reply_format)
    raise ValueError(f"unknown harness {kind!r}; expected codex or claude")
