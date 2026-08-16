# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build shared prepared SDXL regional sampling graphs."""

from __future__ import annotations

from dataclasses import dataclass

from tools.sdxl_attention_coupling_integration.graph import (
    NodeReference,
    SdxlWorkflowGraph,
)
from tools.sdxl_attention_coupling_integration.matrix import (
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
)
from tools.sdxl_attention_coupling_integration.sampling_controls import (
    SDXL_VISUAL_SAMPLING,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import SdxlVisualCase
from tools.sdxl_attention_coupling_integration.visual_conditioning import (
    SdxlVisualConditioningBuilder,
)


@dataclass(frozen=True, slots=True)
class PreparedSdxlSampling:
    """Retain one complete graph before selecting its timing terminal."""

    graph: SdxlWorkflowGraph
    model: NodeReference
    positive: NodeReference
    negative: NodeReference
    latent: NodeReference
    sampler_class: str
    sampler_extras: dict[str, object]


def prepare_regional_sampling(
    *,
    checkpoint_name: str,
    mask_names: tuple[str, str],
    case: SdxlVisualCase,
) -> PreparedSdxlSampling:
    """Build one native SDXL regional-conditioning sampling graph."""

    if not isinstance(case, SdxlVisualCase):
        raise TypeError("Prepared regional sampling requires an SDXL visual case.")
    graph = SdxlWorkflowGraph()
    loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
    conditioning = SdxlVisualConditioningBuilder().build(
        graph,
        case=case,
        model=[loader, 0],
        clip=[loader, 1],
    )
    latent = graph.add(
        "EmptyLatentImage",
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        batch_size=1,
    )
    masks = graph.add(
        "SimpleSyrup.LoadMaskBatch",
        image={"__value__": list(mask_names)},
        channel="red",
    )
    return PreparedSdxlSampling(
        graph=graph,
        model=conditioning.model,
        positive=conditioning.positive,
        negative=conditioning.negative,
        latent=[latent, 0],
        sampler_class="SimpleSyrup.KSamplerAttentionCoupling",
        sampler_extras={
            "region_masks": [masks, 0],
            "regional_prompt_weight": case.regional_prompt_weight,
            "region_mask_feather": case.region_mask_feather,
        },
    )


def add_sdxl_sampler(
    prepared: PreparedSdxlSampling,
    *,
    model: NodeReference,
) -> str:
    """Append the locked sampler using one selected model reference."""

    return prepared.graph.add(
        prepared.sampler_class,
        model=model,
        seed=SDXL_VISUAL_SAMPLING.seed,
        steps=SDXL_VISUAL_SAMPLING.steps,
        cfg=SDXL_VISUAL_SAMPLING.cfg,
        sampler_name=SDXL_VISUAL_SAMPLING.sampler,
        scheduler=SDXL_VISUAL_SAMPLING.scheduler,
        positive=prepared.positive,
        negative=prepared.negative,
        latent_image=prepared.latent,
        denoise=1.0,
        **prepared.sampler_extras,
    )
