# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compose Attention Coupling through the tiled sampling authority."""

from __future__ import annotations

from typing import Any, ClassVar

from ..domain.attention_coupling_request import (
    AttentionCouplingRequestMode,
    classify_attention_coupling_request,
)
from ..domain.regional_attention_execution import RegionalAttentionExecutionMode
from ..domain.regional_features import (
    EMPTY_REGIONAL_FEATURE_REQUEST,
    RegionalFeature,
    RegionalFeatureRequest,
)
from ..runtime.detail_previews import DetailPreviewContext
from .attention_coupling_model_preparation_service import (
    AttentionCouplingModelPreparationService,
)
from .tiled_diffusion_sampling_service import TiledDiffusionSamplingService

_TILED_ATTENTION_REQUEST = RegionalFeatureRequest(
    frozenset({RegionalFeature.ATTENTION_COUPLING})
)


class TiledAttentionCouplingSamplingService:
    """Prepare regional attention once and delegate all tiled execution."""

    model_preparation_service_class: ClassVar[
        type[AttentionCouplingModelPreparationService]
    ] = AttentionCouplingModelPreparationService
    tiled_sampling_service_class: ClassVar[type[TiledDiffusionSamplingService]] = (
        TiledDiffusionSamplingService
    )

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
        positive: object,
        negative: object,
        region_masks: object | None,
        regional_prompt_weight: float,
        region_mask_feather: int,
        latent_image: dict[str, Any],
        denoise: float,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
        preview_context: DetailPreviewContext | None = None,
        differential_diffusion: bool = False,
    ) -> dict[str, Any]:
        """Bypass ordinary requests or prepare one complete regional request."""

        mode = classify_attention_coupling_request(
            positive=positive,
            negative=negative,
            region_masks=region_masks,
        )
        if mode is AttentionCouplingRequestMode.BYPASS:
            return self._sample_tiled(
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
                diffusion_mode=diffusion_mode,
                latent_tile_width=latent_tile_width,
                latent_tile_height=latent_tile_height,
                latent_tile_overlap=latent_tile_overlap,
                latent_tile_batch_size=latent_tile_batch_size,
                preview_context=preview_context,
                differential_diffusion=differential_diffusion,
                feature_request=EMPTY_REGIONAL_FEATURE_REQUEST,
            )

        prepared = self.model_preparation_service_class().prepare(
            model=model,
            positive=positive,
            negative=negative,
            region_masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
            latent_image=latent_image,
            execution_mode=RegionalAttentionExecutionMode.TILED,
        )
        return self._sample_tiled(
            diffusion_mode=diffusion_mode,
            model=prepared.model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=prepared.positive,
            negative=prepared.negative,
            latent_image=latent_image,
            denoise=denoise,
            latent_tile_width=latent_tile_width,
            latent_tile_height=latent_tile_height,
            latent_tile_overlap=latent_tile_overlap,
            latent_tile_batch_size=latent_tile_batch_size,
            preview_context=preview_context,
            differential_diffusion=differential_diffusion,
            feature_request=_TILED_ATTENTION_REQUEST,
        )

    def _sample_tiled(
        self,
        *,
        diffusion_mode: str,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: dict[str, Any],
        denoise: float,
        latent_tile_width: int,
        latent_tile_height: int,
        latent_tile_overlap: int,
        latent_tile_batch_size: int,
        preview_context: DetailPreviewContext | None,
        differential_diffusion: bool,
        feature_request: RegionalFeatureRequest,
    ) -> dict[str, Any]:
        """Delegate one ordinary or Attention Coupling tiled request."""

        return self.tiled_sampling_service_class().sample(
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
            feature_request=feature_request,
            segs=None,
            region_masks=None,
            regional_prompt_weight=0.5,
            region_mask_feather=0,
        )


TILED_ATTENTION_COUPLING_SAMPLING_SERVICE = TiledAttentionCouplingSamplingService()
