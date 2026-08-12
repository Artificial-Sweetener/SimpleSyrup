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


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    process: _FakeProcess,
    *,
    readiness_error: BaseException | None = None,
) -> None:
    """Replace external lifecycle boundaries with deterministic fakes."""

    monkeypatch.setattr(managed_server, "select_unused_loopback_port", lambda: 8299)
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
