# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for VAE Encode/Decode Options nodes."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from simple_syrup.nodes.vae_options import (
    VAE_DECODE_TILE_SIZE_STEP,
    VAE_ENCODE_TILE_SIZE_STEP,
    VAE_OVERLAP_DEFAULT,
    VAE_OVERLAP_MAX,
    VAE_OVERLAP_MIN,
    VAE_OVERLAP_STEP,
    VAE_TEMPORAL_OVERLAP_DEFAULT,
    VAE_TEMPORAL_OVERLAP_MAX,
    VAE_TEMPORAL_OVERLAP_MIN,
    VAE_TEMPORAL_OVERLAP_STEP,
    VAE_TEMPORAL_SIZE_DEFAULT,
    VAE_TEMPORAL_SIZE_MAX,
    VAE_TEMPORAL_SIZE_MIN,
    VAE_TEMPORAL_SIZE_STEP,
    VAE_TILE_SIZE_DEFAULT,
    VAE_TILE_SIZE_MAX,
    VAE_TILE_SIZE_MIN,
    VAEDecodeOptions,
    VAEEncodeOptions,
)


def test_vae_encode_options_declares_inputs() -> None:
    """VAE Encode (Options) exposes normal/tiled controls."""

    inputs = VAEEncodeOptions.INPUT_TYPES()["required"]

    assert VAEEncodeOptions.RETURN_TYPES == ("LATENT",)
    assert VAEEncodeOptions.RETURN_NAMES == ("latent",)
    assert list(inputs) == [
        "use_tiling",
        "pixels",
        "vae",
        "tile_size",
        "overlap",
        "temporal_size",
        "temporal_overlap",
    ]
    assert inputs["use_tiling"][0] == "BOOLEAN"
    assert inputs["use_tiling"][1]["default"] is False
    assert inputs["pixels"][0] == "IMAGE"
    assert inputs["pixels"][1]["rawLink"] is True
    assert inputs["vae"][0] == "VAE"
    assert inputs["vae"][1]["rawLink"] is True
    _assert_tile_controls(inputs, tile_size_step=VAE_ENCODE_TILE_SIZE_STEP)


def test_vae_decode_options_declares_inputs() -> None:
    """VAE Decode (Options) exposes normal/tiled controls."""

    inputs = VAEDecodeOptions.INPUT_TYPES()["required"]

    assert VAEDecodeOptions.RETURN_TYPES == ("IMAGE",)
    assert VAEDecodeOptions.RETURN_NAMES == ("image",)
    assert list(inputs) == [
        "use_tiling",
        "samples",
        "vae",
        "tile_size",
        "overlap",
        "temporal_size",
        "temporal_overlap",
    ]
    assert inputs["use_tiling"][0] == "BOOLEAN"
    assert inputs["use_tiling"][1]["default"] is False
    assert inputs["samples"][0] == "LATENT"
    assert inputs["samples"][1]["rawLink"] is True
    assert inputs["vae"][0] == "VAE"
    assert inputs["vae"][1]["rawLink"] is True
    _assert_tile_controls(inputs, tile_size_step=VAE_DECODE_TILE_SIZE_STEP)


def test_vae_encode_options_expands_to_native_encode() -> None:
    """Normal encode mode expands to ComfyUI's native VAEEncode."""

    _set_graph_prefix("ENCODE_NORMAL")
    result = VAEEncodeOptions().encode(
        use_tiling=False,
        pixels=("image", 0),
        vae=("loader", 2),
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
    )

    node = _single_node(result["expand"])
    assert node["class_type"] == "VAEEncode"
    assert node["inputs"] == {"pixels": ["image", 0], "vae": ["loader", 2]}
    assert result["result"][0][0] in result["expand"]
    assert result["result"][0][1] == 0


def test_vae_encode_options_expands_to_native_tiled_encode() -> None:
    """Tiled encode mode expands to ComfyUI's native VAEEncodeTiled."""

    _set_graph_prefix("ENCODE_TILED")
    result = VAEEncodeOptions().encode(
        use_tiling=True,
        pixels=["image", 0],
        vae=["loader", 2],
        tile_size=768,
        overlap=96,
        temporal_size=48,
        temporal_overlap=12,
    )

    node = _single_node(result["expand"])
    assert node["class_type"] == "VAEEncodeTiled"
    assert node["inputs"] == {
        "pixels": ["image", 0],
        "vae": ["loader", 2],
        "tile_size": 768,
        "overlap": 96,
        "temporal_size": 48,
        "temporal_overlap": 12,
    }


def test_vae_decode_options_expands_to_native_decode() -> None:
    """Normal decode mode expands to ComfyUI's native VAEDecode."""

    _set_graph_prefix("DECODE_NORMAL")
    result = VAEDecodeOptions().decode(
        use_tiling=False,
        samples=("latent", 0),
        vae=("loader", 2),
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
    )

    node = _single_node(result["expand"])
    assert node["class_type"] == "VAEDecode"
    assert node["inputs"] == {"samples": ["latent", 0], "vae": ["loader", 2]}
    assert result["result"][0][0] in result["expand"]
    assert result["result"][0][1] == 0


def test_vae_decode_options_expands_to_native_tiled_decode() -> None:
    """Tiled decode mode expands to ComfyUI's native VAEDecodeTiled."""

    _set_graph_prefix("DECODE_TILED")
    result = VAEDecodeOptions().decode(
        use_tiling=True,
        samples=["latent", 0],
        vae=["loader", 2],
        tile_size=1024,
        overlap=128,
        temporal_size=96,
        temporal_overlap=16,
    )

    node = _single_node(result["expand"])
    assert node["class_type"] == "VAEDecodeTiled"
    assert node["inputs"] == {
        "samples": ["latent", 0],
        "vae": ["loader", 2],
        "tile_size": 1024,
        "overlap": 128,
        "temporal_size": 96,
        "temporal_overlap": 16,
    }


def _assert_tile_controls(
    inputs: dict[str, tuple[Any, ...]],
    *,
    tile_size_step: int,
) -> None:
    """Assert shared VAE tile control metadata."""

    tile_size = inputs["tile_size"][1]
    assert tile_size["default"] == VAE_TILE_SIZE_DEFAULT
    assert tile_size["min"] == VAE_TILE_SIZE_MIN
    assert tile_size["max"] == VAE_TILE_SIZE_MAX
    assert tile_size["step"] == tile_size_step
    assert tile_size["advanced"] is True

    overlap = inputs["overlap"][1]
    assert overlap["default"] == VAE_OVERLAP_DEFAULT
    assert overlap["min"] == VAE_OVERLAP_MIN
    assert overlap["max"] == VAE_OVERLAP_MAX
    assert overlap["step"] == VAE_OVERLAP_STEP
    assert overlap["advanced"] is True

    temporal_size = inputs["temporal_size"][1]
    assert temporal_size["default"] == VAE_TEMPORAL_SIZE_DEFAULT
    assert temporal_size["min"] == VAE_TEMPORAL_SIZE_MIN
    assert temporal_size["max"] == VAE_TEMPORAL_SIZE_MAX
    assert temporal_size["step"] == VAE_TEMPORAL_SIZE_STEP
    assert temporal_size["advanced"] is True

    temporal_overlap = inputs["temporal_overlap"][1]
    assert temporal_overlap["default"] == VAE_TEMPORAL_OVERLAP_DEFAULT
    assert temporal_overlap["min"] == VAE_TEMPORAL_OVERLAP_MIN
    assert temporal_overlap["max"] == VAE_TEMPORAL_OVERLAP_MAX
    assert temporal_overlap["step"] == VAE_TEMPORAL_OVERLAP_STEP
    assert temporal_overlap["advanced"] is True


def _single_node(graph: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return the only node from a dynamic expansion graph."""

    assert len(graph) == 1
    return next(iter(graph.values()))


def _set_graph_prefix(prefix: str) -> None:
    """Set a deterministic Comfy graph-builder prefix for assertions."""

    graph_utils = import_module("comfy_execution.graph_utils")
    graph_utils.GraphBuilder.set_default_prefix(prefix, 0, 0)
