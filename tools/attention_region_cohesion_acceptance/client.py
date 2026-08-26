"""Submit ComfyUI acceptance graphs and resolve their saved outputs."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from .workflow import Graph


def execute(server: str, graph: Graph, output_root: Path) -> dict[str, Path]:
    """Execute one graph and return saved image paths by output node id."""

    payload = json.dumps({"prompt": graph}).encode("utf-8")
    request = urllib.request.Request(
        f"{server}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        prompt_id = str(json.loads(response.read())["prompt_id"])
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
                return outputs
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
