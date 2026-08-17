# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trace ordered Comfy node execution over one loopback WebSocket session."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from tools.comfy_api import JsonObject, LoopbackComfyClient


@dataclass(frozen=True, slots=True)
class ComfyNodeExecutionTiming:
    """Retain one node's observed start and duration until the next event."""

    node_id: str
    class_type: str
    started_at_ns: int
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class ComfyExecutionTraceResult:
    """Retain one completed prompt history and its ordered node timings."""

    prompt_id: str
    submission_started_at_ns: int
    completed_at_ns: int
    cached_node_ids: tuple[str, ...]
    nodes: tuple[ComfyNodeExecutionTiming, ...]
    history: JsonObject

    @property
    def elapsed_ms(self) -> float:
        """Return observed submission-to-terminal wall time."""

        return (self.completed_at_ns - self.submission_started_at_ns) / 1_000_000.0

    def class_totals_ms(self) -> dict[str, float]:
        """Aggregate ordered node durations by exact runtime class type."""

        totals: dict[str, float] = {}
        for node in self.nodes:
            totals[node.class_type] = totals.get(node.class_type, 0.0) + node.elapsed_ms
        return totals


class ComfyExecutionTraceAccumulator:
    """Convert WebSocket lifecycle messages into ordered node intervals."""

    def __init__(
        self,
        *,
        prompt_id: str,
        prompt: dict[str, JsonObject],
        submission_started_at_ns: int,
    ) -> None:
        """Retain immutable prompt identity and initialize empty trace state."""

        if not prompt_id:
            raise ValueError("Execution trace prompt id must be non-empty.")
        if submission_started_at_ns < 1:
            raise ValueError("Execution trace submission timestamp must be positive.")
        self._prompt_id = prompt_id
        self._prompt = prompt
        self._submission_started_at_ns = submission_started_at_ns
        self._cached_node_ids: tuple[str, ...] = ()
        self._current: tuple[str, int] | None = None
        self._nodes: list[ComfyNodeExecutionTiming] = []
        self._completed_at_ns: int | None = None

    def observe(self, message: object, *, observed_at_ns: int) -> bool:
        """Consume one JSON message and report terminal completion."""

        if observed_at_ns < self._submission_started_at_ns:
            raise ValueError("Execution trace observation precedes submission.")
        if not isinstance(message, dict):
            return False
        event_type = message.get("type")
        data = message.get("data")
        if not isinstance(event_type, str) or not isinstance(data, dict):
            return False
        message_prompt_id = data.get("prompt_id")
        if message_prompt_id != self._prompt_id:
            return False
        if event_type == "execution_cached":
            self._cached_node_ids = self._cached_nodes(data.get("nodes"))
            return False
        if event_type == "executing":
            node_id = data.get("node")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("Execution trace node id is invalid.")
            self._close_current(observed_at_ns)
            self._current = (node_id, observed_at_ns)
            return False
        if event_type == "execution_error":
            raise RuntimeError(
                "Comfy execution trace observed an error: "
                f"{data.get('exception_type')}: {data.get('exception_message')}"
            )
        if event_type == "execution_success":
            self._close_current(observed_at_ns)
            self._completed_at_ns = observed_at_ns
            return True
        return False

    def finish(self, history: JsonObject) -> ComfyExecutionTraceResult:
        """Return a completed trace or fail on missing lifecycle evidence."""

        if self._completed_at_ns is None:
            raise TimeoutError("Comfy execution trace did not reach terminal success.")
        if not self._nodes:
            raise ValueError("Comfy execution trace observed no node starts.")
        return ComfyExecutionTraceResult(
            prompt_id=self._prompt_id,
            submission_started_at_ns=self._submission_started_at_ns,
            completed_at_ns=self._completed_at_ns,
            cached_node_ids=self._cached_node_ids,
            nodes=tuple(self._nodes),
            history=history,
        )

    def _close_current(self, completed_at_ns: int) -> None:
        """Close the current node interval against one observed boundary."""

        if self._current is None:
            return
        node_id, started_at_ns = self._current
        if completed_at_ns < started_at_ns:
            raise ValueError("Execution trace node completion precedes its start.")
        node = self._prompt.get(node_id)
        class_type = node.get("class_type") if isinstance(node, dict) else None
        if not isinstance(class_type, str) or not class_type:
            raise ValueError(f"Execution trace node {node_id!r} has no class type.")
        self._nodes.append(
            ComfyNodeExecutionTiming(
                node_id=node_id,
                class_type=class_type,
                started_at_ns=started_at_ns,
                elapsed_ms=(completed_at_ns - started_at_ns) / 1_000_000.0,
            )
        )
        self._current = None

    @staticmethod
    def _cached_nodes(value: object) -> tuple[str, ...]:
        """Return canonical cached node ids from one lifecycle message."""

        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError("Execution trace cached node ids are invalid.")
        return tuple(value)


class LoopbackExecutionTrace:
    """Submit one prompt and capture its loopback WebSocket node lifecycle."""

    def execute(
        self,
        client: LoopbackComfyClient,
        *,
        prompt: dict[str, JsonObject],
        timeout: float,
    ) -> ComfyExecutionTraceResult:
        """Return one synchronized trace and completed history."""

        if timeout <= 0.0:
            raise ValueError("Execution trace timeout must be positive.")
        websocket: Any = import_module("websocket")
        connection: Any = websocket.create_connection(
            client.websocket_url,
            timeout=min(timeout, 5.0),
        )
        try:
            submission_started_at_ns = time.perf_counter_ns()
            prompt_id = client.submit(prompt)
            accumulator = ComfyExecutionTraceAccumulator(
                prompt_id=prompt_id,
                prompt=prompt,
                submission_started_at_ns=submission_started_at_ns,
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                connection.settimeout(min(1.0, max(deadline - time.monotonic(), 0.01)))
                try:
                    raw_message = connection.recv()
                except Exception as error:
                    if isinstance(error, websocket.WebSocketTimeoutException):
                        continue
                    raise
                if not isinstance(raw_message, str):
                    continue
                decoded: object = json.loads(raw_message)
                if accumulator.observe(
                    decoded,
                    observed_at_ns=time.perf_counter_ns(),
                ):
                    history = client.wait_for_history(prompt_id, timeout=timeout)
                    return accumulator.finish(history)
            raise TimeoutError(
                f"Comfy execution trace exceeded {timeout}s: {prompt_id}."
            )
        finally:
            connection.close()


LOOPBACK_EXECUTION_TRACE = LoopbackExecutionTrace()
