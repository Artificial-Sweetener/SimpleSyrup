# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trace ComfyUI prompt links back to unmodified VAE decode sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..domain.graph_provenance import (
    BrokenProvenance,
    GraphLink,
    PassthroughRule,
    VaeDecodeProvenance,
)

MAX_PROVENANCE_HOPS = 128
PASSTHROUGH_ATTRIBUTE = "GRAPH_PASSTHROUGH_OUTPUTS"

PromptNode = Mapping[str, Any]
PromptGraph = Mapping[str, Any]
NodeRegistry = Mapping[str, type[object]]
ProvenanceTrace = VaeDecodeProvenance | BrokenProvenance


def trace_vae_decode_provenance(
    prompt: PromptGraph,
    start_link: object,
    node_registry: NodeRegistry,
    *,
    max_hops: int = MAX_PROVENANCE_HOPS,
) -> ProvenanceTrace:
    """Trace an image link through exact pass-through nodes to `VAEDecode`."""

    current = parse_graph_link(start_link)
    if current is None:
        return BrokenProvenance("image input is not a graph link")

    visited: set[GraphLink] = set()
    for _ in range(max_hops):
        if current in visited:
            return BrokenProvenance(
                "provenance trace contains a cycle",
                node_id=current[0],
            )
        visited.add(current)

        node_id, output_slot = current
        node = _prompt_node(prompt, node_id)
        if node is None:
            return BrokenProvenance("source node is missing", node_id=node_id)

        class_type = _class_type(node)
        if class_type is None:
            return BrokenProvenance(
                "source node class_type is missing",
                node_id=node_id,
            )

        inputs = _node_inputs(node)
        if inputs is None:
            return BrokenProvenance(
                "source node inputs are missing",
                node_id=node_id,
                class_type=class_type,
            )

        if class_type == "VAEDecode":
            return _trace_vae_decode(node_id, output_slot, inputs)

        class_def = node_registry.get(class_type)
        if class_def is None:
            return BrokenProvenance(
                "source node class is not registered",
                node_id=node_id,
                class_type=class_type,
            )

        passthrough = _resolve_passthrough_rule(class_def, output_slot)
        if isinstance(passthrough, BrokenProvenance):
            return BrokenProvenance(
                passthrough.reason,
                node_id=node_id,
                class_type=class_type,
            )
        if passthrough is None:
            return BrokenProvenance(
                "source node does not declare exact pass-through provenance",
                node_id=node_id,
                class_type=class_type,
            )

        next_link = parse_graph_link(inputs.get(passthrough.input_name))
        if next_link is None:
            return BrokenProvenance(
                "pass-through input is not a graph link",
                node_id=node_id,
                class_type=class_type,
            )
        current = next_link

    node_id, _ = current
    return BrokenProvenance("provenance trace exceeded the hop limit", node_id=node_id)


def parse_graph_link(value: object) -> GraphLink | None:
    """Convert a Comfy graph link value to a typed link tuple."""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    node_id = value[0]
    output_slot = value[1]
    if not isinstance(node_id, str) or not node_id:
        return None
    if isinstance(output_slot, bool):
        return None
    if isinstance(output_slot, int):
        return (node_id, output_slot)
    if isinstance(output_slot, float) and output_slot.is_integer():
        return (node_id, int(output_slot))
    return None


def links_match(left: object, right: object) -> bool:
    """Return whether two raw Comfy values refer to the same graph output."""

    left_link = parse_graph_link(left)
    right_link = parse_graph_link(right)
    return left_link is not None and left_link == right_link


def _trace_vae_decode(
    node_id: str,
    output_slot: int,
    inputs: Mapping[str, Any],
) -> ProvenanceTrace:
    """Resolve the latent and VAE links from a `VAEDecode` prompt node."""

    image_output = (node_id, output_slot)
    if output_slot != 0:
        return BrokenProvenance(
            "VAEDecode output is not the image output",
            node_id=node_id,
            class_type="VAEDecode",
        )

    samples_link = parse_graph_link(inputs.get("samples"))
    if samples_link is None:
        return BrokenProvenance(
            "VAEDecode samples input is not a graph link",
            node_id=node_id,
            class_type="VAEDecode",
        )

    return VaeDecodeProvenance(
        decode_node_id=node_id,
        image_output=image_output,
        samples_link=samples_link,
        vae_link=parse_graph_link(inputs.get("vae")),
    )


def _resolve_passthrough_rule(
    class_def: type[object],
    output_slot: int,
) -> PassthroughRule | BrokenProvenance | None:
    """Return the exact pass-through rule declared by a node class."""

    raw_rules = getattr(class_def, PASSTHROUGH_ATTRIBUTE, None)
    if raw_rules is None:
        return None
    if not isinstance(raw_rules, Mapping):
        return BrokenProvenance("pass-through metadata is malformed")
    if output_slot not in raw_rules:
        return None

    input_name = raw_rules[output_slot]
    if not isinstance(input_name, str) or not input_name.strip():
        return BrokenProvenance("pass-through metadata is malformed")
    return PassthroughRule(input_name=input_name.strip())


def _prompt_node(prompt: PromptGraph, node_id: str) -> PromptNode | None:
    """Return a prompt node mapping when the prompt contains one."""

    node = prompt.get(node_id)
    if not isinstance(node, Mapping):
        return None
    return node


def _class_type(node: PromptNode) -> str | None:
    """Return a prompt node class type when it is valid."""

    class_type = node.get("class_type")
    if not isinstance(class_type, str) or not class_type:
        return None
    return class_type


def _node_inputs(node: PromptNode) -> Mapping[str, Any] | None:
    """Return a prompt node input mapping when it is valid."""

    inputs = node.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    return inputs
