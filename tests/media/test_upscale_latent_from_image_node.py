# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Upscale Latent From Image node."""

from __future__ import annotations

from typing import Any

import pytest

from simple_syrup.nodes.provenance_latent import (
    LATENT_PROVENANCE_ERROR,
    UpscaleLatentFromImage,
)


def test_upscale_latent_from_image_declares_raw_link_image_input() -> None:
    """Upscale Latent From Image exposes image plus latent scale controls."""

    inputs = UpscaleLatentFromImage.INPUT_TYPES()

    assert UpscaleLatentFromImage.RETURN_TYPES == ("LATENT",)
    assert UpscaleLatentFromImage.RETURN_NAMES == ("latent",)
    assert inputs["required"]["image"][0] == "IMAGE"
    assert inputs["required"]["image"][1]["rawLink"] is True
    assert inputs["required"]["scale_factor"][0] == "FLOAT"
    assert inputs["hidden"]["prompt"] == "PROMPT"


def test_upscale_method_choices_match_comfy_latent_upscale_by() -> None:
    """The node mirrors Comfy's LatentUpscaleBy interpolation choices."""

    methods = UpscaleLatentFromImage.INPUT_TYPES()["required"]["upscale_method"][0]

    assert methods == ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]


def test_upscale_latent_from_image_expands_to_latent_upscale_by() -> None:
    """Valid decode provenance emits a LatentUpscaleBy dynamic graph."""

    result = UpscaleLatentFromImage().upscale(
        ["decode", 0],
        "bislerp",
        2.0,
        {
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["latent", 0], "vae": ["loader", 2]},
            }
        },
    )

    node = _single_node(result["expand"])
    assert node["class_type"] == "LatentUpscaleBy"
    assert node["inputs"] == {
        "samples": ["latent", 0],
        "upscale_method": "bislerp",
        "scale_by": 2.0,
    }
    assert result["result"][0][0] in result["expand"]
    assert result["result"][0][1] == 0


def test_upscale_latent_from_image_fails_when_provenance_breaks() -> None:
    """Latent upscale refuses to encode or upscale edited image pixels."""

    with pytest.raises(ValueError, match="Unable to find an unmodified VAE Decode"):
        UpscaleLatentFromImage().upscale(
            ["edited", 0],
            "bilinear",
            1.5,
            {
                "edited": {
                    "class_type": "ImageEdit",
                    "inputs": {"image": ["decode", 0]},
                }
            },
        )

    assert "VAE Decode" in LATENT_PROVENANCE_ERROR


def _single_node(graph: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return the only node from a dynamic expansion graph."""

    assert len(graph) == 1
    return next(iter(graph.values()))
