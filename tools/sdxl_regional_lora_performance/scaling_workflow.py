# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build prepared zero/one/four SDXL regional scaling workflows."""

from __future__ import annotations

from .sampling_workflow import add_sdxl_sampler, prepare_regional_sampling
from .scaling_cases import DeclaredSdxlRegionalScalingCase
from .steady_state_workflow import BuiltSdxlSteadyStateWorkflow


def build_regional_scaling_steady_state_workflow(
    *,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    declared: DeclaredSdxlRegionalScalingCase,
) -> BuiltSdxlSteadyStateWorkflow:
    """Build one stable image-free regional scaling graph."""

    if not isinstance(declared, DeclaredSdxlRegionalScalingCase):
        raise TypeError("Regional scaling workflow declaration is invalid.")
    prepared = prepare_regional_sampling(
        checkpoint_name=checkpoint_name,
        mask_names=mask_names,
        case=declared.case,
    )
    sampler = add_sdxl_sampler(prepared, model=prepared.model)
    completion = prepared.graph.add(
        "SimpleSyrupBenchmark.CompleteLatent",
        latent=[sampler, 0],
    )
    return BuiltSdxlSteadyStateWorkflow(
        prompt=prepared.graph.prompt,
        sampler_node_id=sampler,
        completion_node_id=completion,
    )
