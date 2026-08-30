# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build focused attention-cohesion acceptance workflows from saved prompts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

Graph = dict[str, dict[str, Any]]


def build_workflow(
    source_png: Path,
    *,
    output_prefix: str,
    concept: str,
    strength: float,
    consensus: float,
    split: float,
) -> Graph:
    """Return a same-seed graph comparing raw and isolated concept evidence."""

    prompt = json.loads(Image.open(source_png).info["prompt"])
    graph: Graph = deepcopy(prompt)
    capture_id = next(
        node_id
        for node_id, node in graph.items()
        if node["class_type"] == "SimpleSyrup.AttentionCaptureModel"
    )
    sampler_id = next(
        node_id for node_id, node in graph.items() if node["class_type"] == "KSampler"
    )
    graph[sampler_id]["inputs"]["model"] = graph[capture_id]["inputs"]["model"]

    retained = {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "900",
        "10",
        "100",
        "101",
        "11",
        "1100",
        "1101",
        "14",
        "1130",
        "1131",
    }
    graph = {node_id: node for node_id, node in graph.items() if node_id in retained}
    graph["15"] = deepcopy(graph["14"])
    graph["1140"] = deepcopy(graph["1130"])
    graph["1141"] = deepcopy(graph["1131"])
    graph["1140"]["inputs"]["mask"] = ["15", 2]
    graph["1141"]["inputs"]["images"] = ["1140", 0]
    graph["900"]["inputs"]["filename_prefix"] = f"{output_prefix}/image"
    configure_mask(
        graph,
        request_id="10",
        save_id="101",
        concept=concept,
        strength=strength,
        consensus=consensus,
        split=split,
        minimum_size=1,
        keep_only=0,
        solidity=0.0,
        evidence_mode="raw attention",
        prefix=f"{output_prefix}/raw_alpha",
    )
    configure_mask(
        graph,
        request_id="11",
        save_id="1101",
        concept=concept,
        strength=0.15,
        consensus=0.25,
        split=0.35,
        minimum_size=512,
        keep_only=1,
        solidity=0.75,
        evidence_mode="raw attention",
        prefix=f"{output_prefix}/previous_aggregate",
    )
    configure_mask(
        graph,
        request_id="14",
        save_id="1131",
        concept=concept,
        strength=0.15,
        consensus=0.25,
        split=0.35,
        minimum_size=512,
        keep_only=1,
        solidity=0.0,
        evidence_mode="concept isolation",
        prefix=f"{output_prefix}/concept_isolation",
    )
    configure_mask(
        graph,
        request_id="15",
        save_id="1141",
        concept=concept,
        strength=0.15,
        consensus=0.25,
        split=0.35,
        minimum_size=512,
        keep_only=1,
        solidity=1.0,
        evidence_mode="concept isolation",
        prefix=f"{output_prefix}/concept_isolation_solid",
    )
    return graph


def configure_mask(
    graph: Graph,
    *,
    request_id: str,
    save_id: str,
    concept: str,
    strength: float,
    consensus: float,
    split: float,
    minimum_size: int,
    keep_only: int,
    solidity: float,
    evidence_mode: str,
    prefix: str,
    edge_feather: int = 8,
) -> None:
    """Configure one focused concept-mask output."""

    inputs = graph[request_id]["inputs"]
    inputs.update(
        {
            "concepts": concept,
            "sampler_stage": 1,
            "capture_start": 0.0,
            "capture_end": 1.0,
            "minimum_strength": strength,
            "minimum_consensus": consensus,
            "geometry_recall": 0.85,
            "split_sensitivity": split,
            "instance_recall": 0.65,
            "minimum_region_size": minimum_size,
            "keep_only": keep_only,
            "keep_by": "largest size",
            "combine_segs": False,
            "matte_solidity": solidity,
            "edge_feather": edge_feather,
            "capture_profile": "fast",
            "evidence_mode": evidence_mode,
        }
    )
    graph[save_id]["inputs"]["filename_prefix"] = prefix
