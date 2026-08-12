"""Own bounded readiness polling for a live managed Comfy process."""

from __future__ import annotations

import time
from collections.abc import Collection
from typing import Protocol

from tools.comfy_api import JsonObject


class ReadinessClient(Protocol):
    """Describe the existing HTTP operation required by readiness."""

    def verify_server(self, required_node_ids: Collection[str]) -> JsonObject:
        """Return system stats after required-node validation."""


class LiveProcess(Protocol):
    """Describe managed-process liveness without exposing mutation."""

    @property
    def is_running(self) -> bool:
        """Return whether the process remains alive."""


def wait_for_server(
    client: ReadinessClient,
    process: LiveProcess,
    required_node_ids: Collection[str],
    *,
    timeout: float = 120.0,
    poll_interval: float = 1.0,
) -> JsonObject:
    """Retry connection failures until ready, exit, or timeout."""

    if timeout <= 0 or poll_interval <= 0:
        raise ValueError("Readiness timeout and poll interval must be positive.")
    deadline = time.monotonic() + timeout
    last_error: ConnectionError | None = None
    while time.monotonic() < deadline:
        if not process.is_running:
            raise RuntimeError("Managed Comfy process exited before readiness.")
        try:
            return client.verify_server(required_node_ids)
        except ConnectionError as error:
            last_error = error
            time.sleep(poll_interval)
    raise TimeoutError(f"Managed Comfy did not become ready: {last_error}")
