# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test context-managed startup, readiness, and exact cleanup coordination."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tools.comfy_api import JsonObject
from tools.comfy_integration import managed_server
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.comfy_integration.server_process import ComfyServerCommand


class _FakeProcess:
    """Expose the process boundary required by the lifecycle coordinator."""

    pid = 4242

    def __init__(self) -> None:
        """Initialize as live and un-stopped."""

        self.running = True
        self.stop_calls = 0

    @property
    def is_running(self) -> bool:
        """Return current fake process liveness."""

        return self.running

    def stop(self) -> None:
        """Record cleanup and transition to stopped."""

        self.stop_calls += 1
        self.running = False


class _FakeClient:
    """Retain the selected loopback endpoint."""

    def __init__(self, base_url: str) -> None:
        """Record the endpoint supplied by the lifecycle owner."""

        self.base_url = base_url


class _FakeReservation:
    """Expose one deterministic candidate port and release observation."""

    def __init__(self, port: int) -> None:
        """Retain the selected fake port."""

        self.port = port
        self.release_calls = 0

    def release(self) -> None:
        """Record the handoff immediately before process launch."""

        self.release_calls += 1


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    process: _FakeProcess,
    *,
    readiness_error: BaseException | None = None,
) -> None:
    """Replace external lifecycle boundaries with deterministic fakes."""

    reservation = _FakeReservation(8299)
    monkeypatch.setattr(managed_server, "reserve_loopback_port", lambda: reservation)
    monkeypatch.setattr(
        "tools.comfy_integration.managed_server.WindowsComfyProcess.start",
        lambda command, stdout_path, stderr_path: process,
    )
    monkeypatch.setattr(managed_server, "LoopbackComfyClient", _FakeClient)

    def ready(
        client: object, live: object, required: object, *, timeout: float
    ) -> JsonObject:
        """Return metadata or the configured readiness failure."""

        del client, live, required, timeout
        if readiness_error is not None:
            raise readiness_error
        return {"ready": True}

    monkeypatch.setattr(managed_server, "wait_for_server", ready)


def test_port_collision_retries_with_new_owned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry only after cleanup proves another process claimed the candidate port."""

    reservations = [_FakeReservation(8299), _FakeReservation(8300)]
    first_reservation, second_reservation = reservations
    processes = [_FakeProcess(), _FakeProcess()]
    monkeypatch.setattr(
        managed_server,
        "reserve_loopback_port",
        lambda: reservations.pop(0),
    )
    monkeypatch.setattr(
        "tools.comfy_integration.managed_server.WindowsComfyProcess.start",
        lambda command, stdout_path, stderr_path: processes.pop(0),
    )
    monkeypatch.setattr(managed_server, "LoopbackComfyClient", _FakeClient)
    readiness_calls = 0

    def ready(
        client: object, live: object, required: object, *, timeout: float
    ) -> JsonObject:
        """Fail the collided attempt and accept the replacement."""

        nonlocal readiness_calls
        del client, live, required, timeout
        readiness_calls += 1
        if readiness_calls == 1:
            raise RuntimeError("Managed Comfy process exited before readiness.")
        return {"ready": True}

    monkeypatch.setattr(managed_server, "wait_for_server", ready)
    availability = iter((False,))
    monkeypatch.setattr(
        managed_server,
        "is_loopback_port_available",
        lambda _port: next(availability),
    )
    first, second = processes

    with ManagedComfyServer(
        comfy_root=Path("<COMFY_ROOT>"),
        artifacts=IntegrationArtifacts(tmp_path),
        required_node_ids=frozenset(),
    ) as running:
        assert running.port == 8300

    assert first.stop_calls == 1
    assert second.stop_calls == 1
    assert first_reservation.release_calls == 1
    assert second_reservation.release_calls == 1
    assert readiness_calls == 2


def test_context_stops_exact_created_process_after_body_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guarantee owned cleanup when workflow execution raises."""

    process = _FakeProcess()
    _configure(monkeypatch, process)
    artifacts = IntegrationArtifacts(tmp_path)

    with pytest.raises(ValueError, match="workflow failed"):
        with ManagedComfyServer(
            comfy_root=Path("<COMFY_ROOT>"),
            artifacts=artifacts,
            required_node_ids=frozenset({"Required.Node"}),
        ) as running:
            assert running.port == 8299
            assert cast(_FakeClient, running.client).base_url == "http://127.0.0.1:8299"
            raise ValueError("workflow failed")

    assert process.stop_calls == 1
    record: object = json.loads((artifacts.root / "run.json").read_text("utf-8"))
    assert isinstance(record, dict)
    assert record["parent_pid"] == 4242


def test_readiness_failure_stops_process_before_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clean the created tree when startup never becomes usable."""

    process = _FakeProcess()
    _configure(monkeypatch, process, readiness_error=TimeoutError("not ready"))
    artifacts = IntegrationArtifacts(tmp_path)

    with pytest.raises(TimeoutError, match="not ready"):
        with ManagedComfyServer(
            comfy_root=Path("<COMFY_ROOT>"),
            artifacts=artifacts,
            required_node_ids=frozenset(),
        ):
            pytest.fail("unreachable")

    assert process.stop_calls == 1


def test_context_passes_optional_launch_arguments_to_command_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route CPU-only registration probes through the canonical lifecycle."""

    process = _FakeProcess()
    _configure(monkeypatch, process)
    artifacts = IntegrationArtifacts(tmp_path)
    captured: list[tuple[str, ...]] = []

    def start(command: ComfyServerCommand, **kwargs: object) -> _FakeProcess:
        """Capture the exact immutable launch arguments."""

        del kwargs
        captured.append(command.launch_arguments)
        return process

    monkeypatch.setattr(
        "tools.comfy_integration.managed_server.WindowsComfyProcess.start",
        start,
    )

    with ManagedComfyServer(
        comfy_root=Path("<COMFY_ROOT>"),
        artifacts=artifacts,
        required_node_ids=frozenset(),
        launch_arguments=("--cpu",),
    ):
        pass

    assert captured == [("--cpu",)]
