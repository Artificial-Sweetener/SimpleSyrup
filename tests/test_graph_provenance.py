# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Comfy graph provenance tracing."""

from __future__ import annotations

from typing import Any

from simple_syrup.domain.graph_provenance import BrokenProvenance, VaeDecodeProvenance
from simple_syrup.runtime.comfy_graph_provenance import trace_vae_decode_provenance


class TransparentNode:
    """Fake node that declares exact value pass-through provenance."""

    GRAPH_PASSTHROUGH_OUTPUTS = {0: "value"}


class NonTransparentNode:
    """Fake node that intentionally has no pass-through contract."""


class MalformedTransparentNode:
    """Fake node with invalid pass-through metadata."""

    GRAPH_PASSTHROUGH_OUTPUTS = {0: 1}


def test_direct_vae_decode_resolves_samples_and_vae_links() -> None:
    """Direct VAEDecode image output resolves to the source latent and VAE."""

    prompt = {
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["latent", 0], "vae": ["loader", 2]},
        }
    }

    result = trace_vae_decode_provenance(prompt, ["decode", 0], {})

    assert isinstance(result, VaeDecodeProvenance)
    assert result.decode_node_id == "decode"
    assert result.image_output == ("decode", 0)
    assert result.samples_link == ("latent", 0)
    assert result.vae_link == ("loader", 2)


def test_transparent_node_resolves_to_upstream_vae_decode() -> None:
    """A declared pass-through node is traversed to its source link."""

    prompt = {
        "marker": {
            "class_type": "TransparentNode",
            "inputs": {"value": ["decode", 0]},
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["latent", 0], "vae": ["loader", 2]},
        },
    }

    result = trace_vae_decode_provenance(
        prompt,
        ["marker", 0],
        {"TransparentNode": TransparentNode},
    )

    assert isinstance(result, VaeDecodeProvenance)
    assert result.samples_link == ("latent", 0)


def test_multiple_transparent_nodes_resolve_to_upstream_vae_decode() -> None:
    """Transparent marker chains preserve decode provenance."""

    prompt = {
        "outer": {
            "class_type": "TransparentNode",
            "inputs": {"value": ["inner", 0]},
        },
        "inner": {
            "class_type": "TransparentNode",
            "inputs": {"value": ["decode", 0]},
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["latent", 0], "vae": ["loader", 2]},
        },
    }

    result = trace_vae_decode_provenance(
        prompt,
        ["outer", 0],
        {"TransparentNode": TransparentNode},
    )

    assert isinstance(result, VaeDecodeProvenance)
    assert result.decode_node_id == "decode"


def test_non_transparent_node_breaks_provenance() -> None:
    """Nodes without an exact pass-through contract stop tracing."""

    result = trace_vae_decode_provenance(
        {
            "edited": {
                "class_type": "NonTransparentNode",
                "inputs": {"image": ["decode", 0]},
            }
        },
        ["edited", 0],
        {"NonTransparentNode": NonTransparentNode},
    )

    assert isinstance(result, BrokenProvenance)
    assert result.reason == "source node does not declare exact pass-through provenance"
    assert result.node_id == "edited"
    assert result.class_type == "NonTransparentNode"


def test_vae_decode_non_image_output_breaks_provenance() -> None:
    """Only VAEDecode output slot 0 is trusted as decoded image provenance."""

    result = trace_vae_decode_provenance(
        {
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["latent", 0], "vae": ["loader", 2]},
            }
        },
        ["decode", 1],
        {},
    )

    assert isinstance(result, BrokenProvenance)
    assert result.reason == "VAEDecode output is not the image output"


def test_missing_samples_link_breaks_provenance() -> None:
    """VAEDecode without a graph-linked samples input cannot supply provenance."""

    result = trace_vae_decode_provenance(
        {"decode": {"class_type": "VAEDecode", "inputs": {"samples": "latent"}}},
        ["decode", 0],
        {},
    )

    assert isinstance(result, BrokenProvenance)
    assert result.reason == "VAEDecode samples input is not a graph link"


def test_missing_node_breaks_provenance() -> None:
    """Missing source nodes produce broken provenance."""

    result = trace_vae_decode_provenance({}, ["missing", 0], {})

    assert isinstance(result, BrokenProvenance)
    assert result.reason == "source node is missing"
    assert result.node_id == "missing"


def test_malformed_passthrough_metadata_breaks_provenance() -> None:
    """Invalid transparency metadata is treated as unsafe."""

    prompt: dict[str, dict[str, Any]] = {
        "marker": {
            "class_type": "MalformedTransparentNode",
            "inputs": {"value": ["decode", 0]},
        }
    }

    result = trace_vae_decode_provenance(
        prompt,
        ["marker", 0],
        {"MalformedTransparentNode": MalformedTransparentNode},
    )

    assert isinstance(result, BrokenProvenance)
    assert result.reason == "pass-through metadata is malformed"


def test_cycle_is_detected() -> None:
    """Tracing stops when transparent nodes form a cycle."""

    prompt = {
        "a": {"class_type": "TransparentNode", "inputs": {"value": ["b", 0]}},
        "b": {"class_type": "TransparentNode", "inputs": {"value": ["a", 0]}},
    }

    result = trace_vae_decode_provenance(
        prompt,
        ["a", 0],
        {"TransparentNode": TransparentNode},
    )

    assert isinstance(result, BrokenProvenance)
    assert result.reason == "provenance trace contains a cycle"


def test_hop_limit_is_enforced() -> None:
    """Tracing stops before walking unbounded transparent chains."""

    prompt = {
        "a": {"class_type": "TransparentNode", "inputs": {"value": ["b", 0]}},
        "b": {"class_type": "TransparentNode", "inputs": {"value": ["decode", 0]}},
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["latent", 0], "vae": ["loader", 2]},
        },
    }

    result = trace_vae_decode_provenance(
        prompt,
        ["a", 0],
        {"TransparentNode": TransparentNode},
        max_hops=1,
    )

    assert isinstance(result, BrokenProvenance)
    assert result.reason == "provenance trace exceeded the hop limit"
