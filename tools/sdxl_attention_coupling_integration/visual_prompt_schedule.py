# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply optional layout-first scheduling to visual regional prompts."""

from __future__ import annotations

import math

from .graph import NodeReference, SdxlWorkflowGraph


def schedule_visual_regional_prompt(
    graph: SdxlWorkflowGraph,
    conditioning: NodeReference,
    *,
    start_percent: float,
) -> NodeReference:
    """Delay one regional positive while leaving base conditioning unchanged."""

    if not isinstance(graph, SdxlWorkflowGraph):
        raise TypeError("Visual regional prompt scheduling requires a graph.")
    if isinstance(start_percent, bool) or not isinstance(start_percent, int | float):
        raise TypeError("Visual regional prompt start must be a real number.")
    normalized = float(start_percent)
    if not math.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError("Visual regional prompt start must be finite in [0, 1).")
    if normalized == 0.0:
        return conditioning
    scheduled = graph.add(
        "ConditioningSetTimestepRange",
        conditioning=conditioning,
        start=normalized,
        end=1.0,
    )
    return [scheduled, 0]
