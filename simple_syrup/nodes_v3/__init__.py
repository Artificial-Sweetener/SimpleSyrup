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
    from .compose_regional_conditioning import ComposeRegionalConditioningV3
    from .external_llm_prompt import ExternalLLMPromptV3
    from .ksampler_prompt_by_region import KSamplerPromptByRegionV3
    from .ksampler_prompt_by_tiled_region import KSamplerPromptByTiledRegionV3
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
        KSamplerContextualDiffusionV3,
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
        SEGSFromSAMOutputV3,
        SimpleLoadAnimaV3,
        SimplePreviewSEGSV3,
        SimpleVAEEncodeV3,
        UpscaleLatentFromImageV3,
        ViTMatteModelLoaderV3,
    )
    from .load_image_list import LoadImageListV3
    from .load_mask_batch import LoadMaskBatchV3
    from .mask_to_segs import MaskToSEGSV3
    from .scale_factor import ScaleFactorV3
    from .simple_load_checkpoint import SimpleLoadCheckpointV3
    from .simple_load_flux import SimpleLoadFluxV3
    from .simple_load_flux2 import SimpleLoadFlux2V3
    from .tag_segs_with_external_llm import TagSEGSWithExternalLLMV3
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
        ComposeRegionalConditioningV3,
        DetailSEGSAsRegionsV3,
        DetailSEGSByScaleFactorTiledDiffusionV3,
        DetailSEGSByScaleFactorV3,
        DetectSEGSWithUltralyticsV3,
        EncodePromptBatchV3,
        ExternalLLMPromptV3,
        GroundedSAMModelInfoV3,
        GroundingDINOModelLoaderV3,
        KSamplerExtrasV3,
        KSamplerPromptByRegionV3,
        KSamplerPromptByTiledRegionV3,
        KSamplerContextualDiffusionV3,
        KSamplerTiledDiffusionV3,
        LatentDiagnosticsV3,
        LayerStyleSAMModelsAdapterV3,
        LoadUltralyticsModelV3,
        LoadImageListV3,
        LoadMaskBatchV3,
        MaskToSEGSV3,
        PromptEncodeStyleAndNormalizationV3,
        PromptEncodeStyleV3,
        PromptSEGSWithSAMV3,
        ResizeImageToTargetV3,
        SAMModelLoaderV3,
        SEGSFromSAMOutputV3,
        ScaleFactorV3,
        SeedV3,
        SimpleLoadAnimaV3,
        SimplePreviewSEGSV3,
        SimpleLoadCheckpointV3,
        SimpleLoadFluxV3,
        SimpleLoadFlux2V3,
        SimpleVAEEncodeV3,
        TagSEGSWithExternalLLMV3,
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

    from .attach_regional_global_conditioning import (
        AttachRegionalGlobalConditioningV3,
    )
    from .encode_prompt_batch_with_prompt_control import (
        EncodePromptBatchWithPromptControl,
    )
    from .prepare_regional_lora_hooks import PrepareRegionalLoraHooksV3
    from .schedule_and_encode_prompts_with_prompt_control import (
        ScheduleAndEncodePromptsWithPromptControl,
    )

    return [
        *nodes,
        AttachRegionalGlobalConditioningV3,
        EncodePromptBatchWithPromptControl,
        PrepareRegionalLoraHooksV3,
        ScheduleAndEncodePromptsWithPromptControl,
    ]


__all__ = ["get_nodes"]
