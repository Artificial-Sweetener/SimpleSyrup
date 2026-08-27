# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build exact-seed Anima tail-recovery diagnostic workflows."""

from __future__ import annotations

from pathlib import Path

from .phrase_diagnostic import build_saved_phrase_diagnostic_workflow
from .workflow import Graph, configure_mask

TAIL_SEED = 992_702
TAIL_CONCEPT = "holding cat"


def build_tail_evidence_workflow(
    source_png: Path,
    *,
    output_prefix: str,
    profile: str,
    raw_only: bool = False,
) -> Graph:
    """Return raw and thresholded tail evidence from one exact generation."""

    strengths = (
        (0.0, 0.02, 0.05, 0.1, 0.15, 0.15)
        if raw_only
        else (0.0, 0.0, 0.02, 0.05, 0.1, 0.15)
    )
    graph = build_saved_phrase_diagnostic_workflow(
        source_png,
        output_prefix=output_prefix,
        concepts=(TAIL_CONCEPT,) * len(strengths),
        strengths=strengths,
    )
    sampler_id = next(
        node_id for node_id, node in graph.items() if node["class_type"] == "KSampler"
    )
    graph[sampler_id]["inputs"]["seed"] = TAIL_SEED
    for index in range(1, len(strengths) + 1):
        request = graph[str(1000 + index)]["inputs"]
        request["capture_profile"] = profile
        request["evidence_mode"] = (
            "raw attention" if raw_only or index == 1 else "concept isolation"
        )
        if raw_only and index == len(strengths):
            request["minimum_consensus"] = 0.25
    return graph


def tail_evidence_labels(*, raw_only: bool = False) -> tuple[str, ...]:
    """Return labels aligned with the exact-seed diagnostic outputs."""

    if raw_only:
        return (
            "RAW 0.000",
            "RAW 0.020",
            "RAW 0.050",
            "RAW 0.100",
            "RAW 0.150",
            "RAW 0.150 + CONSENSUS 0.25",
        )
    return (
        "RAW ATTENTION",
        "ISOLATION 0.000",
        "ISOLATION 0.020",
        "ISOLATION 0.050",
        "ISOLATION 0.100",
        "ISOLATION 0.150",
    )


def build_tail_processing_workflow(
    source_png: Path,
    *,
    output_prefix: str,
) -> Graph:
    """Return each post-isolation boundary for the exact tail generation."""

    stages = (
        ("STRENGTH 0.15", 1, 0, 0.0, 0),
        ("MINIMUM 512", 512, 0, 0.0, 0),
        ("KEEP LARGEST", 512, 1, 0.0, 0),
        ("SOLIDITY 0.75", 512, 1, 0.75, 0),
        ("SOLIDITY + FEATHER 8", 512, 1, 0.75, 8),
        ("FULL SOLID + FEATHER 8", 512, 1, 1.0, 8),
    )
    graph = build_saved_phrase_diagnostic_workflow(
        source_png,
        output_prefix=output_prefix,
        concepts=(TAIL_CONCEPT,) * len(stages),
        strengths=(0.15,) * len(stages),
    )
    sampler_id = next(
        node_id for node_id, node in graph.items() if node["class_type"] == "KSampler"
    )
    graph[sampler_id]["inputs"]["seed"] = TAIL_SEED
    for index, (label, minimum_size, keep_only, solidity, feather) in enumerate(
        stages,
        start=1,
    ):
        configure_mask(
            graph,
            request_id=str(1000 + index),
            save_id=str(3000 + index),
            concept=TAIL_CONCEPT,
            strength=0.15,
            consensus=0.25,
            split=0.0,
            minimum_size=minimum_size,
            keep_only=keep_only,
            solidity=solidity,
            evidence_mode="concept isolation",
            prefix=f"{output_prefix}/{index}_{_slug(label)}",
            edge_feather=feather,
        )
    return graph


def tail_processing_labels() -> tuple[str, ...]:
    """Return labels aligned with the processing-boundary outputs."""

    return (
        "STRENGTH 0.15",
        "MINIMUM 512",
        "KEEP LARGEST",
        "SOLIDITY 0.75",
        "SOLIDITY + FEATHER 8",
        "FULL SOLID + FEATHER 8",
    )


def _slug(value: str) -> str:
    """Return a stable filename fragment for one processing stage."""

    return "_".join(value.casefold().replace("+", " ").split())
