# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Orchestrate full-context Attention Coupling sampling."""

from __future__ import annotations

from typing import Any, ClassVar

from ..domain.attention_coupling_request import (
    AttentionCouplingRequestMode,
    classify_attention_coupling_request,
)
from ..domain.regional_attention_execution import RegionalAttentionExecutionMode
from .attention_coupling_model_preparation_service import (
    AttentionCouplingModelPreparationService,
)
from .ksampler_sampling_service import KSamplerSamplingService


class AttentionCouplingSamplingService:
    """Prepare and sample one admitted full-context Attention Coupling request."""

    model_preparation_service_class: ClassVar[
        type[AttentionCouplingModelPreparationService]
    ] = AttentionCouplingModelPreparationService
    sampling_service_class: ClassVar[type[KSamplerSamplingService]] = (
        KSamplerSamplingService
    )

    def sample(
        self,
        *,
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
    ) -> dict[str, Any]:
        """Bypass ordinary requests or prepare one complete regional request."""

        mode = classify_attention_coupling_request(
            positive=positive,
            negative=negative,
            region_masks=region_masks,
        )
        if mode is AttentionCouplingRequestMode.BYPASS:
            return self.sampling_service_class().sample(
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
            )

        prepared = self.model_preparation_service_class().prepare(
            model=model,
            positive=positive,
            negative=negative,
            region_masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
            latent_image=latent_image,
            execution_mode=RegionalAttentionExecutionMode.FULL,
        )
        return self.sampling_service_class().sample(
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
        )


ATTENTION_COUPLING_SAMPLING_SERVICE = AttentionCouplingSamplingService()
