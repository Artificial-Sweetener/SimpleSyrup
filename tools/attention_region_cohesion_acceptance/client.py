# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Submit ComfyUI acceptance graphs and resolve their saved outputs."""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .workflow import Graph


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Describe one completed ComfyUI graph execution."""

    outputs: dict[str, Path]
    elapsed_seconds: float


def execute(server: str, graph: Graph, output_root: Path) -> dict[str, Path]:
    """Execute one graph and return saved image paths by output node id."""

    return execute_with_metrics(server, graph, output_root).outputs


def execute_with_metrics(
    server: str,
    graph: Graph,
    output_root: Path,
) -> ExecutionResult:
    """Execute one graph and return outputs with server-side elapsed time."""

    payload = json.dumps({"prompt": graph}).encode("utf-8")
    request = urllib.request.Request(
        f"{server}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        submission: dict[str, Any] = json.loads(response.read())
    node_errors = submission.get("node_errors")
    if node_errors:
        raise RuntimeError(
            "ComfyUI rejected acceptance output nodes:\n"
            f"{json.dumps(node_errors, indent=2)}"
        )
    prompt_id = str(submission["prompt_id"])
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        with urllib.request.urlopen(
            f"{server}/history/{prompt_id}", timeout=30
        ) as response:
            history: dict[str, Any] = json.loads(response.read())
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, indent=2))
            outputs = _saved_outputs(record, output_root)
            if outputs:
                return ExecutionResult(
                    outputs=outputs,
                    elapsed_seconds=_elapsed_seconds(record),
                )
        time.sleep(0.5)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish within 300 seconds.")


def _saved_outputs(record: dict[str, Any], output_root: Path) -> dict[str, Path]:
    """Resolve the first saved image for each completed output node."""

    outputs: dict[str, Path] = {}
    for node_id, output in record.get("outputs", {}).items():
        images = output.get("images", [])
        if images:
            image = images[0]
            outputs[str(node_id)] = (
                output_root / image.get("subfolder", "") / image["filename"]
            )
    return outputs


def _elapsed_seconds(record: dict[str, Any]) -> float:
    """Return ComfyUI's server-side duration for a completed prompt."""

    timestamps = {
        str(message[0]): int(message[1]["timestamp"])
        for message in record.get("status", {}).get("messages", [])
        if (
            isinstance(message, list)
            and len(message) == 2
            and isinstance(message[1], dict)
            and "timestamp" in message[1]
        )
    }
    try:
        return (timestamps["execution_success"] - timestamps["execution_start"]) / 1000
    except KeyError as error:
        raise RuntimeError(
            "ComfyUI history did not include execution timing."
        ) from error
