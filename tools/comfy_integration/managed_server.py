# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coordinate one context-managed Comfy server from focused lifecycle owners."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from tools.comfy_api import JsonObject, LoopbackComfyClient

from .artifacts import IntegrationArtifacts
from .loopback_port import select_unused_loopback_port
from .readiness import wait_for_server
from .server_process import ComfyServerCommand, WindowsComfyProcess


@dataclass(frozen=True)
class RunningManagedComfy:
    """Expose ready server state without transferring lifecycle ownership."""

    client: LoopbackComfyClient
    process: WindowsComfyProcess
    system_stats: JsonObject
    port: int


class ManagedComfyServer:
    """Start, verify, expose, and always clean one exact Comfy process tree."""

    def __init__(
        self,
        *,
        comfy_root: Path,
        artifacts: IntegrationArtifacts,
        required_node_ids: frozenset[str],
        readiness_timeout: float = 180.0,
    ) -> None:
        """Retain immutable launch inputs without starting external state."""

        self._root = comfy_root.resolve()
        self._artifacts = artifacts
        self._required = required_node_ids
        self._readiness_timeout = readiness_timeout
        self._running: RunningManagedComfy | None = None

    def __enter__(self) -> RunningManagedComfy:
        """Start and verify a new loopback server."""

        port = select_unused_loopback_port()
        command = ComfyServerCommand(
            self._root, self._root / "venv" / "Scripts" / "python.exe", port
        )
        process = WindowsComfyProcess.start(
            command,
            stdout_path=self._artifacts.stdout_path,
            stderr_path=self._artifacts.stderr_path,
        )
        self._artifacts.record_started(
            command=command.arguments(),
            environment={
                "comfy_root": str(command.comfy_root),
                "python_executable": str(command.python_executable),
                "platform": platform.platform(),
            },
            required_node_ids=self._required,
            port=port,
            pid=process.pid,
        )
        client = LoopbackComfyClient(f"http://127.0.0.1:{port}")
        try:
            stats = wait_for_server(
                client,
                process,
                self._required,
                timeout=self._readiness_timeout,
            )
        except BaseException:
            process.stop()
            raise
        self._running = RunningManagedComfy(client, process, stats, port)
        return self._running

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Stop only the process tree created by this context."""

        del exc_type, exc, traceback
        if self._running is not None:
            self._running.process.stop()
