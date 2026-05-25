# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build ComfyUI VAE option node expansions without owning VAE execution."""

from __future__ import annotations

from importlib import import_module
from typing import Any, TypedDict, cast


class ExpansionResult(TypedDict):
    """ComfyUI dynamic expansion result with one output link."""

    expand: dict[str, dict[str, Any]]
    result: tuple[list[object], ...]


class VAEOptionsGraphBuilder:
    """Expand VAE option choices to ComfyUI's native VAE nodes."""

    def build_encode(
        self,
        *,
        pixels: object,
        vae: object,
        use_tiling: bool,
        tile_size: int,
        overlap: int,
        temporal_size: int,
        temporal_overlap: int,
    ) -> ExpansionResult:
        """Build a native VAE encode or tiled VAE encode expansion."""

        inputs: dict[str, object] = {
            "pixels": _graph_value(pixels),
            "vae": _graph_value(vae),
        }
        class_type = "VAEEncode"
        if use_tiling:
            class_type = "VAEEncodeTiled"
            inputs.update(
                {
                    "tile_size": int(tile_size),
                    "overlap": int(overlap),
                    "temporal_size": int(temporal_size),
                    "temporal_overlap": int(temporal_overlap),
                }
            )

        return _single_node_expansion(class_type, inputs)

    def build_decode(
        self,
        *,
        samples: object,
        vae: object,
        use_tiling: bool,
        tile_size: int,
        overlap: int,
        temporal_size: int,
        temporal_overlap: int,
    ) -> ExpansionResult:
        """Build a native VAE decode or tiled VAE decode expansion."""

        inputs: dict[str, object] = {
            "samples": _graph_value(samples),
            "vae": _graph_value(vae),
        }
        class_type = "VAEDecode"
        if use_tiling:
            class_type = "VAEDecodeTiled"
            inputs.update(
                {
                    "tile_size": int(tile_size),
                    "overlap": int(overlap),
                    "temporal_size": int(temporal_size),
                    "temporal_overlap": int(temporal_overlap),
                }
            )

        return _single_node_expansion(class_type, inputs)


def _single_node_expansion(
    class_type: str,
    inputs: dict[str, object],
) -> ExpansionResult:
    """Return a one-node dynamic expansion for a native ComfyUI node."""

    builder = _graph_builder()
    node = builder.node(class_type, **inputs)
    return {
        "expand": cast(dict[str, dict[str, Any]], builder.finalize()),
        "result": (cast(list[object], node.out(0)),),
    }


def _graph_value(value: object) -> object:
    """Normalize tuple graph links to ComfyUI's serialized list shape."""

    if isinstance(value, tuple) and len(value) == 2:
        node_id, output_slot = value
        if isinstance(node_id, str) and isinstance(output_slot, int):
            return [node_id, output_slot]
    return value


def _graph_builder() -> Any:
    """Return ComfyUI's graph builder without import-time Comfy coupling."""

    graph_utils = import_module("comfy_execution.graph_utils")
    graph_builder = cast(Any, graph_utils.GraphBuilder)
    return graph_builder()
