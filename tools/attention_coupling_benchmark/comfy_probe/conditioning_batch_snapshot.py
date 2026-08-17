# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose ordered conditioning-batch tensor identities as managed evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch

from .conditioning_batch_bridge import normalize_conditioning_batch
from .tensor_snapshot import snapshot_tensor

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for the benchmark-only node."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io
_mixed_conditioning_io: Any = (
    None if TYPE_CHECKING else _comfy_io.Custom("CONDITIONING,CONDITIONING_BATCH")
)


class SnapshotConditioningBatchV3(_ComfyNodeBase):
    """Observe paired conditioning tensors without changing graph values."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare paired mixed-conditioning inputs and JSON evidence output."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.SnapshotConditioningBatch",
            display_name="Benchmark Snapshot Conditioning Batch",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _mixed_conditioning_io.Input("positive"),
                _mixed_conditioning_io.Input("negative"),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[_comfy_io.String.Output("snapshot_json")],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, positive: object, negative: object, run_id: str) -> Any:
        """Publish ordered tensor identities for both conditioning branches."""

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Conditioning-batch snapshot run ID must not be empty.")
        snapshot = {
            "run_id": run_id,
            "positive": snapshot_conditioning_value(positive),
            "negative": snapshot_conditioning_value(negative),
        }
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            encoded,
            ui={"conditioning_batch_snapshot": [snapshot]},
        )


def snapshot_conditioning_value(value: object) -> dict[str, object]:
    """Record one conditioning or ordered ConditioningBatch tensor structure."""

    normalized = normalize_conditioning_batch(value)
    if isinstance(normalized, ConditioningBatch):
        kind = "conditioning_batch"
        batches = normalized.entries
    else:
        kind = "conditioning"
        batches = (normalized,)
    return {
        "kind": kind,
        "batch_entries": [
            {
                "batch_index": batch_index,
                "conditioning_entries": _snapshot_conditioning(
                    conditioning,
                    batch_index=batch_index,
                ),
            }
            for batch_index, conditioning in enumerate(batches)
        ],
    }


def _snapshot_conditioning(
    conditioning: object,
    *,
    batch_index: int,
) -> list[dict[str, object]]:
    """Record every tensor entry in one Comfy conditioning sequence."""

    if not isinstance(conditioning, Sequence) or isinstance(conditioning, str | bytes):
        raise TypeError(
            f"conditioning batch entry {batch_index} must be a conditioning sequence."
        )
    result: list[dict[str, object]] = []
    for entry_index, entry in enumerate(conditioning):
        label = f"conditioning[{batch_index}][{entry_index}]"
        if not isinstance(entry, Sequence) or isinstance(entry, str | bytes):
            raise TypeError(f"{label} must contain tensor and metadata.")
        if len(entry) != 2:
            raise TypeError(f"{label} must contain tensor and metadata.")
        tensor, metadata = entry
        if not isinstance(tensor, torch.Tensor) or not isinstance(metadata, dict):
            raise TypeError(f"{label} has invalid tensor or metadata.")
        result.append(
            {
                "entry_index": entry_index,
                "tensor": snapshot_tensor(tensor),
            }
        )
    return result
