# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node registration for SimpleSyrup."""

from __future__ import annotations

from ..runtime.prompt_control_availability import prompt_control_is_available


def get_nodes() -> list[type[object]]:
    """Return v3 nodes that can be advertised in this environment."""

    from .batch_region_conditioning import BatchRegionConditioningV3
    from .batch_segs import BatchSEGSV3
    from .external_llm_prompt import ExternalLLMPromptV3
    from .legacy_node_wrappers import (
        ConditioningBatchAppendV3,
        ConditioningBatchStartV3,
        DetailSEGSAsRegionsV3,
        DetailSEGSByScaleFactorTiledDiffusionV3,
        DetailSEGSByScaleFactorV3,
        DetectSEGSWithUltralyticsV3,
        EncodePromptBatchV3,
        GroundedSAMModelInfoV3,
        GroundingDINOModelLoaderV3,
        KSamplerExtrasV3,
        KSamplerTiledDiffusionV3,
        LatentDiagnosticsV3,
        LayerStyleSAMModelsAdapterV3,
        LoadUltralyticsModelV3,
        PromptEncodeStyleAndNormalizationV3,
        PromptEncodeStyleV3,
        PromptSEGSWithSAMV3,
        ResizeImageToTargetV3,
        SAMModelLoaderV3,
        SeedV3,
        SimpleLoadAnimaV3,
        SimpleVAEEncodeV3,
        UpscaleLatentFromImageV3,
        ViTMatteModelLoaderV3,
    )
    from .scale_factor import ScaleFactorV3
    from .simple_load_checkpoint import SimpleLoadCheckpointV3
    from .tag_segs_with_wd14 import TagSEGSWithWD14V3
    from .tile_and_tag_segs import TileAndTagSEGSV3
    from .vae_decode_options import VAEDecodeOptionsV3
    from .vae_encode_options import VAEEncodeOptionsV3
    from .wd14_tagger_loader import WD14TaggerLoaderV3

    nodes: list[type[object]] = [
        BatchRegionConditioningV3,
        BatchSEGSV3,
        ConditioningBatchAppendV3,
        ConditioningBatchStartV3,
        DetailSEGSAsRegionsV3,
        DetailSEGSByScaleFactorTiledDiffusionV3,
        DetailSEGSByScaleFactorV3,
        DetectSEGSWithUltralyticsV3,
        EncodePromptBatchV3,
        ExternalLLMPromptV3,
        GroundedSAMModelInfoV3,
        GroundingDINOModelLoaderV3,
        KSamplerExtrasV3,
        KSamplerTiledDiffusionV3,
        LatentDiagnosticsV3,
        LayerStyleSAMModelsAdapterV3,
        LoadUltralyticsModelV3,
        PromptEncodeStyleAndNormalizationV3,
        PromptEncodeStyleV3,
        PromptSEGSWithSAMV3,
        ResizeImageToTargetV3,
        SAMModelLoaderV3,
        ScaleFactorV3,
        SeedV3,
        SimpleLoadAnimaV3,
        SimpleLoadCheckpointV3,
        SimpleVAEEncodeV3,
        TagSEGSWithWD14V3,
        TileAndTagSEGSV3,
        UpscaleLatentFromImageV3,
        VAEDecodeOptionsV3,
        VAEEncodeOptionsV3,
        ViTMatteModelLoaderV3,
        WD14TaggerLoaderV3,
    ]

    if not prompt_control_is_available():
        return nodes

    from .encode_prompt_batch_with_prompt_control import (
        EncodePromptBatchWithPromptControl,
    )
    from .schedule_and_encode_prompts_with_prompt_control import (
        ScheduleAndEncodePromptsWithPromptControl,
    )

    return [
        *nodes,
        EncodePromptBatchWithPromptControl,
        ScheduleAndEncodePromptsWithPromptControl,
    ]


__all__ = ["get_nodes"]
