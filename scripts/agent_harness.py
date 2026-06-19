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
    async def send(self, prompt: str) -> str:
        """Send one user turn and return the final text reply."""

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

    async def send(self, prompt: str) -> str:
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
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=os.environ.copy(),
        )
        stdout, stderr = await self._communicate(proc)
        self._capture_session_id(stdout.decode())
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
                 timeout: float | None = None) -> None:
        super().__init__(persona, model=model, cwd=cwd, timeout=timeout)
        self.permission_mode = permission_mode
        self._session_id = str(uuid.uuid4())
        self._started = False

    async def send(self, prompt: str) -> str:
        argv = ["claude", "--print", "--output-format", "text"]
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
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=os.environ.copy(),
        )
        stdout, stderr = await self._communicate(proc)
        reply = stdout.decode().strip()
        if proc.returncode != 0 or not reply:
            tail = "\n".join(stderr.decode().strip().splitlines()[-8:])
            raise RuntimeError(
                f"claude turn failed with code {proc.returncode}"
                + (f"\n{tail}" if tail else "")
            )
        self._started = True
        return reply


def make_harness(kind: str, persona: str, *, model: str | None = None,
                 cwd: str | Path = ".", role: str = "worker",
                 codex_danger: bool = False, timeout: float | None = None) -> Harness:
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
                             permission_mode=permission, timeout=timeout)
    raise ValueError(f"unknown harness {kind!r}; expected codex or claude")
