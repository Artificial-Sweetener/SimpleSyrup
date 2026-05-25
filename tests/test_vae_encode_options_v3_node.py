# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the VAE Encode (Options) Comfy v3 wrapper."""

from __future__ import annotations

from importlib import import_module

from simple_syrup.nodes import tooltips
from simple_syrup.nodes.vae_options import (
    VAE_ENCODE_TILE_SIZE_STEP,
    VAE_OVERLAP_DEFAULT,
    VAE_TEMPORAL_OVERLAP_DEFAULT,
    VAE_TEMPORAL_SIZE_DEFAULT,
    VAE_TILE_SIZE_DEFAULT,
    VAEEncodeOptions,
)
from simple_syrup.nodes_v3.vae_encode_options import VAEEncodeOptionsV3


def test_vae_encode_options_v3_schema() -> None:
    """VAE Encode (Options) v3 schema mirrors the legacy node contract."""

    schema = VAEEncodeOptionsV3.define_schema()

    assert schema.node_id == "SimpleSyrup.VAEEncodeOptions"
    assert schema.display_name == "VAE Encode (Options)"
    assert schema.category == "SimpleSyrup/Latent"
    assert schema.description == VAEEncodeOptions.DESCRIPTION
    assert schema.enable_expand is True
    assert [input_item.id for input_item in schema.inputs] == [
        "use_tiling",
        "pixels",
        "vae",
        "tile_size",
        "overlap",
        "temporal_size",
        "temporal_overlap",
    ]
    assert schema.inputs[0].io_type == "BOOLEAN"
    assert schema.inputs[0].default is False
    assert schema.inputs[0].tooltip == tooltips.VAE_OPTIONS_USE_TILING
    assert schema.inputs[1].io_type == "IMAGE"
    assert schema.inputs[1].rawLink is True
    assert schema.inputs[2].io_type == "VAE"
    assert schema.inputs[2].rawLink is True
    assert schema.inputs[3].default == VAE_TILE_SIZE_DEFAULT
    assert schema.inputs[3].step == VAE_ENCODE_TILE_SIZE_STEP
    assert schema.inputs[4].default == VAE_OVERLAP_DEFAULT
    assert schema.inputs[5].default == VAE_TEMPORAL_SIZE_DEFAULT
    assert schema.inputs[6].default == VAE_TEMPORAL_OVERLAP_DEFAULT
    assert [output.id for output in schema.outputs] == ["latent"]
    assert schema.outputs[0].io_type == "LATENT"
    assert schema.outputs[0].tooltip == tooltips.VAE_OPTIONS_LATENT_OUTPUT


def test_vae_encode_options_v3_execute_delegates_to_legacy_node() -> None:
    """VAE Encode (Options) v3 execution returns legacy expansion behavior."""

    graph_utils = import_module("comfy_execution.graph_utils")
    graph_utils.GraphBuilder.set_default_prefix("V3_ENCODE", 0, 0)

    result = VAEEncodeOptionsV3.execute(
        True,
        ["image", 0],
        ["loader", 2],
        768,
        96,
        48,
        12,
    )
    node = next(iter(result["expand"].values()))

    assert node["class_type"] == "VAEEncodeTiled"
    assert node["inputs"]["tile_size"] == 768
    assert result["result"][0][1] == 0
