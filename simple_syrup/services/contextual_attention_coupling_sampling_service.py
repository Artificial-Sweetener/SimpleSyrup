# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compose Attention Coupling through Contextual Diffusion authority."""

from __future__ import annotations

from typing import Any, ClassVar

from ..domain.regional_attention_execution import RegionalAttentionExecutionMode
from ..domain.regional_features import RegionalFeature, RegionalFeatureRequest
from .attention_coupling_model_preparation_service import (
    AttentionCouplingModelPreparationService,
)
from .contextual_diffusion_sampling_service import (
    ContextualDiffusionSamplingResult,
    ContextualDiffusionSamplingService,
)

_CONTEXTUAL_ATTENTION_REQUEST = RegionalFeatureRequest(
    frozenset({RegionalFeature.ATTENTION_COUPLING})
)


class ContextualAttentionCouplingSamplingService:
    """Prepare regional attention once and delegate Contextual execution."""

    model_preparation_service_class: ClassVar[
        type[AttentionCouplingModelPreparationService]
    ] = AttentionCouplingModelPreparationService
    contextual_sampling_service_class: ClassVar[
        type[ContextualDiffusionSamplingService]
    ] = ContextualDiffusionSamplingService

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
        region_masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
        latent_image: dict[str, Any],
        denoise: float,
        diffusion_mode: str,
        latent_context_size: int,
        latent_context_overlap: int,
        latent_context_batch_size: int,
        global_weight: float,
        global_steps: int,
        global_decay: float,
        segs: object | None = None,
    ) -> ContextualDiffusionSamplingResult:
        """Prepare once and invoke established Contextual local-view execution."""

        prepared = self.model_preparation_service_class().prepare(
            model=model,
            positive=positive,
            negative=negative,
            region_masks=region_masks,
            regional_prompt_weight=regional_prompt_weight,
            region_mask_feather=region_mask_feather,
            latent_image=latent_image,
            execution_mode=RegionalAttentionExecutionMode.CONTEXTUAL,
        )
        return self.contextual_sampling_service_class().sample(
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
            diffusion_mode=diffusion_mode,
            latent_context_size=latent_context_size,
            latent_context_overlap=latent_context_overlap,
            latent_context_batch_size=latent_context_batch_size,
            global_weight=global_weight,
            global_steps=global_steps,
            global_decay=global_decay,
            segs=segs,
            region_masks=None,
            regional_prompt_weight=0.5,
            region_mask_feather=0,
            feature_request=_CONTEXTUAL_ATTENTION_REQUEST,
            planning_region_masks=prepared.mask_bank.planning_masks,
        )


CONTEXTUAL_ATTENTION_COUPLING_SAMPLING_SERVICE = (
    ContextualAttentionCouplingSamplingService()
)
