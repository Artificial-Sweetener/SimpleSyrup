# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own one exact Windows ComfyUI process tree and its redirected logs."""

from __future__ import annotations

import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .loopback_port import validate_loopback_port

TASKKILL_ATTEMPTS = 3
TASKKILL_RETRY_DELAY_SECONDS = 0.2


@dataclass(frozen=True)
class ComfyServerCommand:
    """Describe the exact normal-install Comfy launch command."""

    comfy_root: Path
    python_executable: Path
    port: int
    launch_arguments: tuple[str, ...] = ()

    def arguments(self) -> tuple[str, ...]:
        """Return the shell-free loopback launch argument list."""

        return (
            str(self.python_executable),
            str(self.comfy_root / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--disable-auto-launch",
            "--database-url",
            "sqlite:///:memory:",
            *self.launch_arguments,
        )

    def validate(self) -> None:
        """Fail before launch when the authoritative install is incomplete."""

        validate_loopback_port(self.port)
        if any(
            not isinstance(argument, str) or not argument.strip()
            for argument in self.launch_arguments
        ):
            raise ValueError("Comfy launch arguments must be non-empty strings.")
        if not self.comfy_root.is_dir() or not (self.comfy_root / "main.py").is_file():
            raise ValueError("Comfy root must contain main.py.")
        if not self.python_executable.is_file():
            raise ValueError("Comfy Python executable does not exist.")


class WindowsComfyProcess:
    """Launch, observe, and terminate only one created Windows process tree."""

    def __init__(
        self,
        command: ComfyServerCommand,
        process: subprocess.Popen[bytes],
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> None:
        """Retain exact process and log ownership until cleanup."""

        self.command = command
        self._process = process
        self._stdout = stdout
        self._stderr = stderr

    @classmethod
    def start(
        cls, command: ComfyServerCommand, *, stdout_path: Path, stderr_path: Path
    ) -> WindowsComfyProcess:
        """Start the validated process with dedicated binary log files."""

        command.validate()
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = stdout_path.open("wb")
        stderr = stderr_path.open("wb")
        try:
            process = subprocess.Popen(
                command.arguments(),
                cwd=command.comfy_root,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        except BaseException:
            stdout.close()
            stderr.close()
            raise
        return cls(command, process, stdout, stderr)

    @property
    def pid(self) -> int:
        """Return the exact created parent PID."""

        return self._process.pid

    @property
    def is_running(self) -> bool:
        """Report whether the created parent remains alive."""

        return self._process.poll() is None

    def stop(self, *, graceful_timeout: float = 15.0) -> None:
        """Terminate the exact parent tree, forcing only after a bounded wait."""

        if graceful_timeout <= 0:
            raise ValueError("Process cleanup timeout must be positive.")
        try:
            if self.is_running:
                try:
                    self._process.send_signal(signal.CTRL_BREAK_EVENT)
                except OSError:
                    if self.is_running:
                        self._force_stop_if_running()
                try:
                    self._process.wait(timeout=graceful_timeout)
                except subprocess.TimeoutExpired:
                    self._force_stop_if_running()
                    self._process.wait(timeout=graceful_timeout)
        finally:
            self._stdout.close()
            self._stderr.close()

    def _force_stop_if_running(self) -> None:
        """Escalate the exact tree while accepting a verified concurrent exit."""

        try:
            _taskkill(self.pid, force=True)
        except RuntimeError:
            if self.is_running:
                raise


def _taskkill(pid: int, *, force: bool) -> None:
    """Terminate one explicit PID tree despite transient Windows command races."""

    arguments = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        arguments.append("/F")
    failure = "no diagnostic output"
    for attempt in range(1, TASKKILL_ATTEMPTS + 1):
        try:
            completed = subprocess.run(
                arguments,
                check=False,
                capture_output=True,
                timeout=10.0,
            )
        except subprocess.TimeoutExpired as error:
            failure = f"attempt {attempt} timed out after {error.timeout} seconds"
        else:
            if completed.returncode == 0:
                return
            output = completed.stdout.decode(errors="replace").strip()
            error_output = completed.stderr.decode(errors="replace").strip()
            failure = error_output or output or "no diagnostic output"
        if attempt < TASKKILL_ATTEMPTS:
            time.sleep(TASKKILL_RETRY_DELAY_SECONDS)
    raise RuntimeError(
        "Unable to terminate owned Comfy process tree "
        f"{pid} after {TASKKILL_ATTEMPTS} attempts: {failure}"
    )
