# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve strict latent lineage and inject cache-keyed upstream capture
dependencies.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..domain.graph_provenance import BrokenProvenance, GraphLink
from .comfy_graph_provenance import (
    NodeRegistry,
    parse_graph_link,
    trace_vae_decode_provenance,
)
from .noise_reuse_source import (
    capture_graph_context,
    describe_graph_node,
    resolve_source_sampling_graph,
)


@dataclass(frozen=True, slots=True)
class CaptureConsumer:
    """Describe source capture requirements without storing runtime model objects."""

    source: str
    receiver: str | None
    geometry_json: str


def source_lineage(
    prompt: Mapping[str, Any], start: object, registry: NodeRegistry
) -> tuple[str, str]:
    """Trace only latent-preserving resizing and unmodified decoded-image bridges."""
    current = parse_graph_link(start)
    visited: set[GraphLink] = set()
    geometry: list[dict[str, object]] = []
    while current is not None and len(visited) < 128:
        if current in visited:
            raise ValueError("Noise reuse latent provenance contains a cycle.")
        visited.add(current)
        node = prompt.get(current[0])
        if not isinstance(node, Mapping) or current[1] != 0:
            raise ValueError(
                "Noise reuse source must be a supported sampler's latent output."
            )
        kind, inputs = node.get("class_type"), node.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("Noise reuse source node has missing inputs.")
        if resolve_source_sampling_graph(prompt, current[0]) is not None:
            return current[0], json.dumps(list(reversed(geometry)))
        if kind == "SimpleSyrup.LatentFromImage":
            decoded = trace_vae_decode_provenance(prompt, inputs.get("image"), registry)
            if isinstance(decoded, BrokenProvenance):
                raise ValueError(
                    f"Noise reuse image provenance is broken: {decoded.reason}."
                )
            current = decoded.samples_link
            continue
        if kind in {"SimpleSyrup.SimpleUpscaleLatent", "LatentUpscaleBy"}:
            key = (
                "scale_factor"
                if kind == "SimpleSyrup.SimpleUpscaleLatent"
                else "scale_by"
            )
            scale = inputs.get(key, 2.0 if key == "scale_factor" else 1.5)
            geometry.append({"scale": scale})
            current = parse_graph_link(
                inputs.get("latent" if key == "scale_factor" else "samples")
            )
            continue
        if kind == "LatentUpscale":
            if inputs.get("crop", "disabled") != "disabled":
                raise ValueError(
                    "Noise reuse requires latent upscaling with crop disabled."
                )
            geometry.append({"pixels": [inputs.get("height"), inputs.get("width")]})
            current = parse_graph_link(inputs.get("samples"))
            continue
        class_def = registry.get(str(kind))
        rules = getattr(class_def, "GRAPH_PASSTHROUGH_OUTPUTS", None)
        if isinstance(rules, Mapping) and isinstance(rules.get(0), str):
            current = parse_graph_link(inputs.get(rules[0]))
            continue
        raise ValueError(
            f"Noise reuse cannot resolve latent provenance through "
            f"{describe_graph_node(prompt, current[0])}. "
            "Use a direct sampler latent, Simple Upscale "
            f"Latent, or Latent From Image."
        )
    raise ValueError("Noise reuse requires a graph-connected source sampler latent.")


def has_noise_reuse(prompt: Mapping[str, Any], start: object) -> bool:
    """Detect potentially enabled reuse without evaluating options or loading models."""
    current = parse_graph_link(start)
    visited: set[GraphLink] = set()
    while current is not None and current not in visited and len(visited) < 128:
        visited.add(current)
        node = prompt.get(current[0])
        if not isinstance(node, Mapping):
            return False
        inputs = node.get("inputs", {})
        if not isinstance(inputs, Mapping):
            return False
        if (
            node.get("class_type") == "SimpleSyrup.NoiseReuseOptions"
            and inputs.get("enabled", True) is not False
        ):
            return True
        current = parse_graph_link(inputs.get("options"))
    return False


def rewrite_noise_reuse_graph(prompt: dict[str, Any], registry: NodeRegistry) -> None:
    """Group receiving requests and redirect every source latent consumer once."""
    if any(
        isinstance(node, Mapping)
        and node.get("class_type")
        in {"SimpleSyrup.TrajectoryCaptureModel", "SimpleSyrup.NoiseReuseError"}
        for node in prompt.values()
    ):
        return
    consumers: list[CaptureConsumer] = []
    failures: dict[str, str] = {}
    for node_id, node in tuple(prompt.items()):
        if not isinstance(node, Mapping):
            continue
        kind, inputs = node.get("class_type"), node.get("inputs", {})
        if not isinstance(inputs, Mapping):
            continue
        receiver = kind == "SimpleSyrup.KSampler" and has_noise_reuse(
            prompt, inputs.get("options")
        )
        auto = (
            kind == "SimpleSyrup.SimpleUpscaleLatent"
            and inputs.get("method", "Auto") == "Auto"
        )
        if not receiver and not auto:
            continue
        try:
            source, geometry = source_lineage(
                prompt, inputs.get("latent_image" if receiver else "latent"), registry
            )
            consumers.append(
                CaptureConsumer(source, node_id if receiver else None, geometry)
            )
        except ValueError as error:
            if receiver:
                failures[node_id] = (
                    f"Noise Reuse Options on receiving sampler "
                    f"{describe_graph_node(prompt, node_id)} cannot reach a usable "
                    f"source sampler. {error}"
                )
    for source in sorted({consumer.source for consumer in consumers}):
        _inject_source(
            prompt,
            source,
            [consumer for consumer in consumers if consumer.source == source],
        )
    for node_id, message in failures.items():
        error_id = _unique_id(prompt, f"{node_id}.noise_reuse_error")
        prompt[error_id] = {
            "class_type": "SimpleSyrup.NoiseReuseError",
            "inputs": {"message": message},
        }
        inputs = prompt[node_id]["inputs"]
        inputs[
            "latent_image"
            if prompt[node_id]["class_type"] == "SimpleSyrup.KSampler"
            else "latent"
        ] = [error_id, 0]


def _inject_source(
    prompt: dict[str, Any], source: str, consumers: list[CaptureConsumer]
) -> None:
    """Put receiver settings in the source MODEL dependency chain, then attach
    its result.
    """
    source_inputs = prompt[source]["inputs"]
    source_graph = resolve_source_sampling_graph(prompt, source)
    if source_graph is None:
        raise ValueError(
            f"Source {describe_graph_node(prompt, source)} is not a sampler."
        )
    request: list[object] | None = None
    for consumer in consumers:
        if consumer.receiver is None:
            continue
        receiver = prompt[consumer.receiver]["inputs"]
        request_id = _unique_id(prompt, f"{consumer.receiver}.noise_capture_request")
        values = {
            key: receiver[key]
            for key in ("model", "steps", "sampler_name", "scheduler")
        }
        values.update(
            options=receiver["options"],
            denoise=receiver.get("denoise", 1.0),
            source_latent=source_graph.latent_input,
            geometry_json=consumer.geometry_json,
        )
        if request is not None:
            values["previous"] = request
        prompt[request_id] = {
            "class_type": "SimpleSyrup.NoiseCaptureRequest",
            "inputs": values,
        }
        request = [request_id, 0]
    capture_id = _unique_id(prompt, f"{source}.trajectory_capture")
    model_inputs = prompt[source_graph.model_owner]["inputs"]
    if source_graph.model_owner != source:
        guider_id = _unique_id(prompt, f"{source}.trajectory_guider")
        guider_node = prompt[source_graph.model_owner].copy()
        model_inputs = model_inputs.copy()
        guider_node["inputs"] = model_inputs
        prompt[guider_id] = guider_node
        guider_link = parse_graph_link(source_inputs["guider"])
        if guider_link is None:
            raise ValueError("Source sampler has an invalid guider connection.")
        source_inputs["guider"] = [guider_id, guider_link[1]]
    values = {
        "model": model_inputs["model"],
        "sampler_name": source_graph.sampler_name,
        "graph_context": capture_graph_context(
            prompt,
            source,
            [
                consumer.receiver
                for consumer in consumers
                if consumer.receiver is not None
            ],
        ),
    }
    if source_graph.sampler is not None:
        values["sampler"] = source_graph.sampler
    if request is not None:
        values["requests"] = request
    prompt[capture_id] = {
        "class_type": "SimpleSyrup.TrajectoryCaptureModel",
        "inputs": values,
    }
    model_inputs["model"] = [capture_id, 0]
    attach_id = _unique_id(prompt, f"{source}.sampling_provenance")
    for node in prompt.values():
        inputs = node.get("inputs", {})
        for key, value in tuple(inputs.items()):
            if parse_graph_link(value) == (source, 0):
                inputs[key] = [attach_id, 0]
    prompt[attach_id] = {
        "class_type": "SimpleSyrup.AttachSamplingProvenance",
        "inputs": {"latent": [source, 0], "capture": [capture_id, 1]},
    }


def _unique_id(prompt: Mapping[str, Any], base: str) -> str:
    """Avoid collisions with user-authored or earlier injected graph identifiers."""
    result = base
    while result in prompt:
        result += "_"
    return result
