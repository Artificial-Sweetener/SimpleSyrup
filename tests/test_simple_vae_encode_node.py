# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Simple VAE Encode node."""

from __future__ import annotations

from typing import Any

from simple_syrup.nodes.provenance_latent import SimpleVAEEncode


def test_simple_vae_encode_declares_raw_link_inputs() -> None:
    """Simple VAE Encode exposes image and VAE inputs with raw graph links."""

    inputs = SimpleVAEEncode.INPUT_TYPES()

    assert SimpleVAEEncode.RETURN_TYPES == ("LATENT",)
    assert SimpleVAEEncode.RETURN_NAMES == ("latent",)
    assert inputs["required"]["image"][0] == "IMAGE"
    assert inputs["required"]["image"][1]["rawLink"] is True
    assert inputs["required"]["vae"][0] == "VAE"
    assert inputs["required"]["vae"][1]["rawLink"] is True
    assert inputs["hidden"]["prompt"] == "PROMPT"


def test_simple_vae_encode_reuses_matching_decode_latent() -> None:
    """Matching VAE decode provenance returns the original latent link."""

    prompt = {
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["latent", 0], "vae": ["loader", 2]},
        }
    }

    result = SimpleVAEEncode().encode(["decode", 0], ["loader", 2], prompt)

    assert result["expand"] == {}
    assert result["result"] == (["latent", 0],)


def test_simple_vae_encode_falls_back_when_provenance_breaks() -> None:
    """Broken provenance emits a fallback VAEEncode expansion."""

    result = SimpleVAEEncode().encode(
        ["edited", 0],
        ["loader", 2],
        {
            "edited": {
                "class_type": "ImageEdit",
                "inputs": {"image": ["decode", 0]},
            }
        },
    )

    graph = result["expand"]
    assert _single_node(graph)["class_type"] == "VAEEncode"
    assert _single_node(graph)["inputs"] == {
        "pixels": ["edited", 0],
        "vae": ["loader", 2],
    }
    assert result["result"][0][0] in graph
    assert result["result"][0][1] == 0


def test_simple_vae_encode_falls_back_when_vae_differs() -> None:
    """Mismatched VAE links use normal VAEEncode behavior."""

    result = SimpleVAEEncode().encode(
        ["decode", 0],
        ["other_loader", 2],
        {
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["latent", 0], "vae": ["loader", 2]},
            }
        },
    )

    node = _single_node(result["expand"])
    assert node["class_type"] == "VAEEncode"
    assert node["inputs"]["pixels"] == ["decode", 0]
    assert node["inputs"]["vae"] == ["other_loader", 2]


def _single_node(graph: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return the only node from a dynamic expansion graph."""

    assert len(graph) == 1
    return next(iter(graph.values()))
