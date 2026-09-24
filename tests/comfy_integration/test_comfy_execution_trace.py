# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify ordered loopback Comfy execution trace aggregation."""

from __future__ import annotations

import pytest

from tools.comfy_api import LoopbackComfyClient
from tools.comfy_integration.execution_trace import (
    ComfyExecutionTraceAccumulator,
)


def test_client_exposes_its_matching_loopback_websocket_session() -> None:
    """Keep HTTP submission and WebSocket events on one generated client id."""

    client = LoopbackComfyClient("http://127.0.0.1:8188")

    assert client.websocket_url.startswith("ws://127.0.0.1:8188/ws?clientId=")


def test_accumulator_filters_foreign_events_and_aggregates_ordered_nodes() -> None:
    """Measure each node until the next start or terminal success."""

    accumulator = _accumulator()
    assert not accumulator.observe(
        _message("executing", prompt_id="foreign", node="1"),
        observed_at_ns=1_010_000_000,
    )
    assert not accumulator.observe(
        _message("execution_cached", nodes=["0"]),
        observed_at_ns=1_020_000_000,
    )
    assert not accumulator.observe(
        _message("executing", node="1"),
        observed_at_ns=1_100_000_000,
    )
    assert not accumulator.observe(
        _message("executing", node="2"),
        observed_at_ns=1_160_000_000,
    )
    assert accumulator.observe(
        _message("execution_success"),
        observed_at_ns=1_220_000_000,
    )

    result = accumulator.finish({"outputs": {}})

    assert result.cached_node_ids == ("0",)
    assert [
        (node.node_id, node.class_type, node.elapsed_ms) for node in result.nodes
    ] == [
        ("1", "First", 60.0),
        ("2", "Second", 60.0),
    ]
    assert result.class_totals_ms() == {"First": 60.0, "Second": 60.0}
    assert result.elapsed_ms == 220.0


def test_accumulator_fails_closed_on_execution_error_and_missing_success() -> None:
    """Surface runtime errors and incomplete traces without partial evidence."""

    accumulator = _accumulator()
    with pytest.raises(RuntimeError, match="RuntimeError: failed"):
        accumulator.observe(
            _message(
                "execution_error",
                exception_type="RuntimeError",
                exception_message="failed",
            ),
            observed_at_ns=1_100_000_000,
        )
    with pytest.raises(TimeoutError, match="terminal success"):
        accumulator.finish({})


def _accumulator() -> ComfyExecutionTraceAccumulator:
    """Return one identity-neutral two-node trace fixture."""

    return ComfyExecutionTraceAccumulator(
        prompt_id="prompt-1",
        prompt={
            "1": {"class_type": "First", "inputs": {}},
            "2": {"class_type": "Second", "inputs": {}},
        },
        submission_started_at_ns=1_000_000_000,
    )


def _message(event_type: str, **data: object) -> dict[str, object]:
    """Build one prompt-scoped WebSocket lifecycle message."""

    return {
        "type": event_type,
        "data": {"prompt_id": data.pop("prompt_id", "prompt-1"), **data},
    }
