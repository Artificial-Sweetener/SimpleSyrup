# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the VAE Decode (Options) Comfy v3 wrapper."""

from __future__ import annotations

from importlib import import_module

from simple_syrup.nodes import tooltips
from simple_syrup.nodes.vae_options import (
    VAE_DECODE_TILE_SIZE_STEP,
    VAE_OVERLAP_DEFAULT,
    VAE_TEMPORAL_OVERLAP_DEFAULT,
    VAE_TEMPORAL_SIZE_DEFAULT,
    VAE_TILE_SIZE_DEFAULT,
    VAEDecodeOptions,
)
from simple_syrup.nodes_v3.vae_decode_options import VAEDecodeOptionsV3


def test_vae_decode_options_v3_schema() -> None:
    """VAE Decode (Options) v3 schema mirrors the legacy node contract."""

    schema = VAEDecodeOptionsV3.define_schema()

    assert schema.node_id == "SimpleSyrup.VAEDecodeOptions"
    assert schema.display_name == "VAE Decode (Options)"
    assert schema.category == "SimpleSyrup/Latent"
    assert schema.description == VAEDecodeOptions.DESCRIPTION
    assert schema.enable_expand is True
    assert [input_item.id for input_item in schema.inputs] == [
        "use_tiling",
        "samples",
        "vae",
        "tile_size",
        "overlap",
        "temporal_size",
        "temporal_overlap",
    ]
    assert schema.inputs[0].io_type == "BOOLEAN"
    assert schema.inputs[0].default is False
    assert schema.inputs[0].tooltip == tooltips.VAE_OPTIONS_USE_TILING
    assert schema.inputs[1].io_type == "LATENT"
    assert schema.inputs[1].rawLink is True
    assert schema.inputs[2].io_type == "VAE"
    assert schema.inputs[2].rawLink is True
    assert schema.inputs[3].default == VAE_TILE_SIZE_DEFAULT
    assert schema.inputs[3].step == VAE_DECODE_TILE_SIZE_STEP
    assert schema.inputs[4].default == VAE_OVERLAP_DEFAULT
    assert schema.inputs[5].default == VAE_TEMPORAL_SIZE_DEFAULT
    assert schema.inputs[6].default == VAE_TEMPORAL_OVERLAP_DEFAULT
    assert [output.id for output in schema.outputs] == ["image"]
    assert schema.outputs[0].io_type == "IMAGE"
    assert schema.outputs[0].tooltip == tooltips.VAE_OPTIONS_IMAGE_OUTPUT


def test_vae_decode_options_v3_execute_delegates_to_legacy_node() -> None:
    """VAE Decode (Options) v3 execution returns legacy expansion behavior."""

    graph_utils = import_module("comfy_execution.graph_utils")
    graph_utils.GraphBuilder.set_default_prefix("V3_DECODE", 0, 0)

    result = VAEDecodeOptionsV3.execute(
        True,
        ["latent", 0],
        ["loader", 2],
        1024,
        128,
        96,
        16,
    )
    node = next(iter(result["expand"].values()))

    assert node["class_type"] == "VAEDecodeTiled"
    assert node["inputs"]["tile_size"] == 1024
    assert result["result"][0][1] == 0
