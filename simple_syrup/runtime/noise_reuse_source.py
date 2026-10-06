# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt source sampling graphs through their shared model and latent contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .comfy_graph_provenance import parse_graph_link


@dataclass(frozen=True, slots=True)
class SourceSamplingGraph:
    """Describe the exact graph dependencies needed to observe one source sampler."""

    model_owner: str
    latent_input: object
    sampler_name: object
    sampler: object | None = None


def resolve_source_sampling_graph(
    prompt: Mapping[str, Any], node_id: str
) -> SourceSamplingGraph | None:
    """Recognize named and custom samplers by their host-facing sampling inputs."""
    node = prompt.get(node_id)
    if not isinstance(node, Mapping):
        return None
    inputs = node.get("inputs")
    if not isinstance(inputs, Mapping) or "latent_image" not in inputs:
        return None
    if "model" in inputs and "sampler_name" in inputs:
        return SourceSamplingGraph(
            node_id, inputs["latent_image"], inputs["sampler_name"]
        )
    if "sampler" not in inputs or "sigmas" not in inputs:
        return None
    owner = node_id
    if "model" not in inputs:
        guider = parse_graph_link(inputs.get("guider"))
        producer = prompt.get(guider[0]) if guider is not None else None
        if not isinstance(producer, Mapping):
            return None
        assert guider is not None
        guider_inputs = producer.get("inputs")
        if not isinstance(guider_inputs, Mapping) or "model" not in guider_inputs:
            raise ValueError(
                "The source custom sampler's guider must expose its connected model "
                "so Noise Reuse Options can observe that source run."
            )
        owner = guider[0]
    link = parse_graph_link(inputs["sampler"])
    selector = prompt.get(link[0]) if link is not None else None
    selector_inputs = (
        selector.get("inputs", {}) if isinstance(selector, Mapping) else {}
    )
    name = (
        selector_inputs.get("sampler_name", "")
        if isinstance(selector_inputs, Mapping)
        else ""
    )
    return SourceSamplingGraph(owner, inputs["latent_image"], name, inputs["sampler"])


def describe_graph_node(prompt: Mapping[str, Any], node_id: str) -> str:
    """Name user titles, node types and graph IDs together in actionable errors."""
    node = prompt.get(node_id, {})
    if not isinstance(node, Mapping):
        return f"missing node {node_id}"
    kind = str(node.get("class_type", "unknown node"))
    metadata = node.get("_meta", {})
    title = metadata.get("title") if isinstance(metadata, Mapping) else None
    return (
        f"'{title}' ({kind}, node {node_id})"
        if isinstance(title, str) and title
        else f"{kind} (node {node_id})"
    )


def capture_graph_context(
    prompt: Mapping[str, Any], source: str, receivers: list[str]
) -> str:
    """Explain which downstream option reaches back into the source graph."""
    if not receivers:
        return (
            "Latent format discovery is observing source "
            f"{describe_graph_node(prompt, source)}."
        )
    downstream = "; ".join(
        describe_graph_node(prompt, node_id) for node_id in receivers
    )
    return (
        f"Noise Reuse Options on receiving sampler {downstream} reaches back into "
        f"source sampler {describe_graph_node(prompt, source)} "
        "to save its original noise."
    )
