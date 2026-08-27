# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build matched baseline and attention-capture benchmark workflows."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from PIL import Image

from .workflow import Graph, build_workflow, configure_mask


def build_benchmark_pair(
    source_png: Path,
    *,
    concept: str,
    seed: int,
    output_prefix: str,
) -> tuple[Graph, Graph]:
    """Return same-seed baseline and minimal concept-capture graphs."""

    source = _source_graph(source_png)
    capture_id = next(
        node_id
        for node_id, node in source.items()
        if node["class_type"] == "SimpleSyrup.AttentionCaptureModel"
    )
    sampler_id = next(
        node_id for node_id, node in source.items() if node["class_type"] == "KSampler"
    )
    source[sampler_id]["inputs"]["model"] = source[capture_id]["inputs"]["model"]
    baseline = _retain(source, {"1", "2", "3", "4", "5", "6", "900"})
    baseline[sampler_id]["inputs"]["seed"] = seed
    baseline["900"]["inputs"]["filename_prefix"] = f"{output_prefix}/baseline"

    capture = _configured_capture(
        source_png,
        output_prefix=output_prefix,
        concept=concept,
        seed=seed,
    )
    capture = _retain(capture, {"1", "2", "3", "4", "5", "6", "900", "14"})
    capture["900"]["inputs"]["images"] = ["14", 0]
    capture["900"]["inputs"]["filename_prefix"] = f"{output_prefix}/capture"
    return baseline, capture


def build_benchmark_proof(
    source_png: Path,
    *,
    concept: str,
    seed: int,
    output_prefix: str,
) -> Graph:
    """Return an untimed representative graph that saves the resulting mask."""

    capture = _configured_capture(
        source_png,
        output_prefix=output_prefix,
        concept=concept,
        seed=seed,
    )
    return _retain(
        capture,
        {"1", "2", "3", "4", "5", "6", "900", "14", "1130", "1131"},
    )


def _configured_capture(
    source_png: Path,
    *,
    output_prefix: str,
    concept: str,
    seed: int,
) -> Graph:
    """Return the representative default concept-capture graph."""

    capture = build_workflow(
        source_png,
        output_prefix=output_prefix,
        concept=concept,
        strength=0.15,
        consensus=0.25,
        split=0.0,
    )
    configure_mask(
        capture,
        request_id="14",
        save_id="1131",
        concept=concept,
        strength=0.15,
        consensus=0.25,
        split=0.0,
        minimum_size=512,
        keep_only=1,
        solidity=0.75,
        evidence_mode="concept isolation",
        prefix=f"{output_prefix}/mask",
    )
    sampler_id = next(
        node_id for node_id, node in capture.items() if node["class_type"] == "KSampler"
    )
    capture[sampler_id]["inputs"]["seed"] = seed
    capture["900"]["inputs"]["filename_prefix"] = f"{output_prefix}/capture"
    return capture


def _source_graph(source_png: Path) -> Graph:
    """Read the embedded ComfyUI graph from one proof generation."""

    with Image.open(source_png) as image:
        return json.loads(image.info["prompt"])


def _retain(graph: Graph, node_ids: set[str]) -> Graph:
    """Copy only explicitly selected benchmark nodes."""

    return {
        node_id: deepcopy(node)
        for node_id, node in graph.items()
        if node_id in node_ids
    }
