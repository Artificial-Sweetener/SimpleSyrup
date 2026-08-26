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
    """Return a minimal same-seed graph with raw, default, and solid masks."""

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
    graph["900"]["inputs"]["filename_prefix"] = f"{output_prefix}/image"
    _configure_mask(
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
        prefix=f"{output_prefix}/raw_alpha",
    )
    _configure_mask(
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
        prefix=f"{output_prefix}/default_cohesive",
    )
    _configure_mask(
        graph,
        request_id="14",
        save_id="1131",
        concept=concept,
        strength=0.15,
        consensus=0.25,
        split=0.35,
        minimum_size=512,
        keep_only=1,
        solidity=1.0,
        prefix=f"{output_prefix}/full_solid",
    )
    return graph


def _configure_mask(
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
    prefix: str,
) -> None:
    """Configure one focused concept-mask output."""

    inputs = graph[request_id]["inputs"]
    inputs.update(
        {
            "concepts": concept,
            "minimum_strength": strength,
            "minimum_consensus": consensus,
            "split_sensitivity": split,
            "minimum_region_size": minimum_size,
            "keep_only": keep_only,
            "combine_segs": False,
            "matte_solidity": solidity,
            "edge_feather": 8,
            "capture_profile": "fast",
        }
    )
    graph[save_id]["inputs"]["filename_prefix"] = prefix
