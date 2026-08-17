# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Profile exact preparation and sampler-delegate boundaries without altering them."""

from __future__ import annotations

from typing import Any, ClassVar

from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.services.attention_coupling_model_preparation_service import (
    AttentionCouplingModelPreparationService,
)
from simple_syrup.services.attention_coupling_sampling_service import (
    AttentionCouplingSamplingService,
)
from simple_syrup.services.ksampler_sampling_service import KSamplerSamplingService

from .preparation_collaborator_profile import (
    ProfiledAttentionCouplingModelPreparationService,
)
from .synchronized_phase_timing import measure_synchronized_phase, model_device


class ProfiledAttentionCouplingSamplingService(AttentionCouplingSamplingService):
    """Delegate the production two-call sequence with synchronized phase timing."""

    model_preparation_service_class: ClassVar[
        type[AttentionCouplingModelPreparationService]
    ] = ProfiledAttentionCouplingModelPreparationService
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
        region_masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
        latent_image: dict[str, Any],
        denoise: float,
    ) -> dict[str, Any]:
        """Return the exact production result after timing its two owners."""

        device = model_device(model)
        with measure_synchronized_phase("model_preparation_total", device=device):
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
        with measure_synchronized_phase("ksampler_delegate_total", device=device):
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
