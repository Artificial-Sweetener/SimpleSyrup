# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test bounded server readiness and process-liveness policy."""

from __future__ import annotations

import pytest

from tools.comfy_api import JsonObject
from tools.comfy_integration.readiness import wait_for_server


class _LiveProcess:
    """Expose configurable managed-process liveness."""

    def __init__(self, running: bool = True) -> None:
        """Set the initial liveness response."""

        self.running = running

    @property
    def is_running(self) -> bool:
        """Return configured liveness."""

        return self.running


class _RetryingClient:
    """Fail one connection before returning system metadata."""

    def __init__(self) -> None:
        """Initialize the call counter."""

        self.calls = 0

    def verify_server(self, required_node_ids: object) -> JsonObject:
        """Retry once and then report ready."""

        del required_node_ids
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("starting")
        return {"system": "ready"}


class _MissingNodeClient:
    """Report a permanent node-discovery failure."""

    def verify_server(self, required_node_ids: object) -> JsonObject:
        """Raise the non-retryable validation error."""

        del required_node_ids
        raise ValueError("missing node")


def test_readiness_retries_only_connection_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry startup transport failure and preserve the ready response."""

    client = _RetryingClient()
    monkeypatch.setattr("tools.comfy_integration.readiness.time.sleep", lambda _: None)

    result = wait_for_server(client, _LiveProcess(), {"Required.Node"})

    assert result == {"system": "ready"}
    assert client.calls == 2


def test_readiness_fails_immediately_when_process_exits() -> None:
    """Stop polling when the owned parent no longer runs."""

    with pytest.raises(RuntimeError, match="exited"):
        wait_for_server(_RetryingClient(), _LiveProcess(False), set())


def test_readiness_does_not_retry_missing_nodes() -> None:
    """Surface permanent discovery errors without waiting for timeout."""

    with pytest.raises(ValueError, match="missing node"):
        wait_for_server(_MissingNodeClient(), _LiveProcess(), {"Required.Node"})
