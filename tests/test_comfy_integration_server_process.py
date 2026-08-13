# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test exact Windows Comfy command and owned-process cleanup."""

from __future__ import annotations

import io
import signal
import subprocess
from pathlib import Path
from typing import cast

import pytest

from tools.comfy_integration import server_process
from tools.comfy_integration.server_process import (
    ComfyServerCommand,
    WindowsComfyProcess,
)


class _FakeProcess:
    """Model the Popen operations used by process cleanup."""

    pid = 4242

    def __init__(
        self,
        *,
        time_out_once: bool = False,
        exit_after_signal_error: bool = False,
    ) -> None:
        """Configure whether graceful waiting expires once."""

        self.running = True
        self.time_out_once = time_out_once
        self.exit_after_signal_error = exit_after_signal_error
        self.signals: list[int] = []
        self.wait_timeouts: list[float] = []

    def poll(self) -> int | None:
        """Return the current fake exit state."""

        return None if self.running else 0

    def send_signal(self, requested_signal: int) -> None:
        """Record delivery to the created process group."""

        self.signals.append(requested_signal)
        if self.exit_after_signal_error:
            self.running = False
            raise OSError("process exited during signal delivery")

    def wait(self, timeout: float) -> int:
        """Complete or raise one configured timeout."""

        self.wait_timeouts.append(timeout)
        if self.time_out_once:
            self.time_out_once = False
            raise subprocess.TimeoutExpired(cmd="comfy", timeout=timeout)
        self.running = False
        return 0


def _owned_process(
    fake: _FakeProcess,
) -> tuple[WindowsComfyProcess, io.BytesIO, io.BytesIO]:
    """Build a process owner around typed in-memory boundaries."""

    stdout = io.BytesIO()
    stderr = io.BytesIO()
    command = ComfyServerCommand(Path("<COMFY_ROOT>"), Path("E:/python.exe"), 8299)
    process = WindowsComfyProcess(
        command,
        cast(subprocess.Popen[bytes], fake),
        stdout,
        stderr,
    )
    return process, stdout, stderr


def test_command_uses_exact_loopback_normal_install_arguments(tmp_path: Path) -> None:
    """Build a shell-free command against the supplied normal install."""

    root = tmp_path / "ComfyUI"
    root.mkdir()
    (root / "main.py").write_text("", encoding="utf-8")
    python = root / "venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    command = ComfyServerCommand(root, python, 8299)

    command.validate()

    assert command.arguments() == (
        str(python),
        str(root / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        "8299",
        "--disable-auto-launch",
        "--database-url",
        "sqlite:///:memory:",
    )


def test_stop_signals_only_the_created_group_and_closes_logs() -> None:
    """Use a bounded graceful process-group stop before closing owned logs."""

    fake = _FakeProcess()
    process, stdout, stderr = _owned_process(fake)

    process.stop(graceful_timeout=3.0)

    assert fake.signals == [signal.CTRL_BREAK_EVENT]
    assert fake.wait_timeouts == [3.0]
    assert stdout.closed
    assert stderr.closed


def test_stop_force_kills_exact_tree_after_graceful_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Escalate one timed-out owned PID tree and wait for its exit."""

    fake = _FakeProcess(time_out_once=True)
    process, stdout, stderr = _owned_process(fake)
    calls: list[tuple[int, bool]] = []

    def record_taskkill(pid: int, *, force: bool) -> None:
        """Capture the exact requested escalation."""

        calls.append((pid, force))

    monkeypatch.setattr(server_process, "_taskkill", record_taskkill)

    process.stop(graceful_timeout=2.0)

    assert calls == [(4242, True)]
    assert fake.wait_timeouts == [2.0, 2.0]
    assert stdout.closed
    assert stderr.closed


def test_stop_accepts_owned_exit_during_forced_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept a taskkill race only after the retained process reports exit."""

    fake = _FakeProcess(time_out_once=True)
    process, stdout, stderr = _owned_process(fake)

    def exited_before_taskkill(pid: int, *, force: bool) -> None:
        """Model the exact owned process exiting before taskkill observes it."""

        assert (pid, force) == (4242, True)
        fake.running = False
        raise RuntimeError('ERROR: The process "4242" not found.')

    monkeypatch.setattr(server_process, "_taskkill", exited_before_taskkill)

    process.stop(graceful_timeout=2.0)

    assert fake.wait_timeouts == [2.0, 2.0]
    assert stdout.closed
    assert stderr.closed


def test_stop_accepts_owned_process_exit_during_signal_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not mask an earlier failure when the owned server exits during cleanup."""

    fake = _FakeProcess(exit_after_signal_error=True)
    process, stdout, stderr = _owned_process(fake)
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        server_process,
        "_taskkill",
        lambda pid, force: calls.append((pid, force)),
    )

    process.stop(graceful_timeout=2.0)

    assert calls == []
    assert stdout.closed
    assert stderr.closed


def test_taskkill_retries_the_same_exact_tree_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry only the original PID tree when Windows first returns no diagnostic."""

    results = [
        subprocess.CompletedProcess[bytes](
            args=[], returncode=1, stdout=b"", stderr=b""
        ),
        subprocess.CompletedProcess[bytes](
            args=[], returncode=0, stdout=b"SUCCESS", stderr=b""
        ),
    ]
    calls: list[tuple[list[str], float]] = []
    sleeps: list[float] = []

    def run(
        arguments: list[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        """Return the configured taskkill result and retain exact arguments."""

        assert not check
        assert capture_output
        calls.append((arguments, timeout))
        return results.pop(0)

    monkeypatch.setattr("tools.comfy_integration.server_process.subprocess.run", run)
    monkeypatch.setattr(
        "tools.comfy_integration.server_process.time.sleep", sleeps.append
    )

    server_process._taskkill(4242, force=True)

    assert calls == [
        (["taskkill", "/PID", "4242", "/T", "/F"], 10.0),
        (["taskkill", "/PID", "4242", "/T", "/F"], 10.0),
    ]
    assert sleeps == [server_process.TASKKILL_RETRY_DELAY_SECONDS]


def test_taskkill_reports_actionable_output_after_bounded_exact_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose Windows stdout when every exact-tree termination attempt fails."""

    calls = 0

    def fail(
        arguments: list[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        """Return one persistent taskkill failure."""

        nonlocal calls
        del arguments, check, capture_output, timeout
        calls += 1
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"access denied", stderr=b""
        )

    monkeypatch.setattr("tools.comfy_integration.server_process.subprocess.run", fail)
    monkeypatch.setattr(
        "tools.comfy_integration.server_process.time.sleep", lambda _: None
    )

    with pytest.raises(RuntimeError, match="after 3 attempts: access denied"):
        server_process._taskkill(4242, force=True)

    assert calls == server_process.TASKKILL_ATTEMPTS
