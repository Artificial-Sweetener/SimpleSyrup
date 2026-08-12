# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for selectable tiled diffusion latent sampling."""

from __future__ import annotations

from typing import Any, ClassVar

from ..domain.conditioning_batch import ConditioningBatch
from ..domain.regional_features import (
    EMPTY_REGIONAL_FEATURE_REQUEST,
    TILED_DIFFUSION_REGIONAL_SAMPLER_CAPABILITIES,
    RegionalFeature,
    RegionalFeatureRequest,
)
from ..domain.tiled_diffusion import validate_tiled_diffusion_mode
from ..runtime.detail_previews import DetailPreviewContext
from .regional_capability_admission_service import (
    RegionalCapabilityAdmissionService,
)
from .regional_sampling_preparation_service import (
    RegionalSamplingPreparationService,
)
from .regional_tiled_diffusion_sampling_service import (
    RegionalTiledDiffusionSamplingService,
)
from .segs_guided_tiled_diffusion_sampling_service import (
    SEGSGuidedTiledDiffusionSamplingService,
)
from .tiled_diffusion_conditioning_batch_service import (
    TiledDiffusionConditioningBatchService,
)
from .tiled_diffusion_item_sampling_service import TiledDiffusionItemSamplingService

Latent = dict[str, Any]


class TiledDiffusionSamplingService:
    """Route tiled diffusion sampling requests to the selected runtime."""

    regional_preparation_service_class: ClassVar[
        type[RegionalSamplingPreparationService]
    ] = RegionalSamplingPreparationService
    capability_admission_service_class: ClassVar[
        type[RegionalCapabilityAdmissionService]
    ] = RegionalCapabilityAdmissionService
    regional_sampling_service_class: ClassVar[
        type[RegionalTiledDiffusionSamplingService]
    ] = RegionalTiledDiffusionSamplingService
    item_sampling_service_class: ClassVar[type[TiledDiffusionItemSamplingService]] = (
        TiledDiffusionItemSamplingService
    )
    segs_sampling_service_class: ClassVar[
        type[SEGSGuidedTiledDiffusionSamplingService]
    ] = SEGSGuidedTiledDiffusionSamplingService
    conditioning_batch_service_class: ClassVar[
        type[TiledDiffusionConditioningBatchService]
    ] = TiledDiffusionConditioningBatchService

    def sample(
        self,
        *,
        diffusion_mode: str,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: Latent,
        denoise: float,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
        preview_context: DetailPreviewContext | None = None,
        differential_diffusion: bool = False,
        feature_request: RegionalFeatureRequest = EMPTY_REGIONAL_FEATURE_REQUEST,
        segs: object | None = None,
        region_masks: object | None = None,
        regional_prompt_weight: float = 0.5,
        region_mask_feather: int = 0,
    ) -> Latent:
        """Sample a latent with the selected tiled diffusion method."""

        validate_tiled_diffusion_mode(diffusion_mode)
        regional = self.regional_preparation_service_class().prepare(
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            region_masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
        )
        effective_request = feature_request
        if regional.active:
            effective_request = effective_request.with_feature(
                RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING
            )
        capability_admission = self.capability_admission_service_class().admit(
            request=effective_request,
            sampler_capabilities=TILED_DIFFUSION_REGIONAL_SAMPLER_CAPABILITIES,
            model=model,
        )
        if regional.mask_bank is not None:
            return self.regional_sampling_service_class().sample(
                item_sampler=self.item_sampling_service_class().sample,
                region_masks=regional.mask_bank.planning_masks,
                segs=segs,
                diffusion_mode=diffusion_mode,
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=regional.positive,
                negative=regional.negative,
                latent_image=latent_image,
                denoise=denoise,
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                capability_admission=capability_admission,
            )
        if segs is not None:
            return self.segs_sampling_service_class().sample(
                item_sampler=self.item_sampling_service_class().sample,
                diffusion_mode=diffusion_mode,
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise,
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                capability_admission=capability_admission,
                segs=segs,
            )
        if isinstance(positive, ConditioningBatch) or isinstance(
            negative,
            ConditioningBatch,
        ):
            return self.conditioning_batch_service_class().sample(
                item_sampler=self.item_sampling_service_class().sample,
                diffusion_mode=diffusion_mode,
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise,
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                capability_admission=capability_admission,
            )
        return self.item_sampling_service_class().sample(
            diffusion_mode=diffusion_mode,
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            latent_tile_width=latent_tile_width,
            latent_tile_height=latent_tile_height,
            latent_tile_overlap=latent_tile_overlap,
            latent_tile_batch_size=latent_tile_batch_size,
            preview_context=preview_context,
            differential_diffusion=differential_diffusion,
            capability_admission=capability_admission,
        )
