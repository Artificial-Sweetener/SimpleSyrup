# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify full-context Attention Coupling sampling orchestration."""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.services.attention_coupling_model_preparation_service import (
    PreparedAttentionCouplingModel,
)
from simple_syrup.services.attention_coupling_sampling_service import (
    AttentionCouplingSamplingService,
)


class _SamplingService:
    """Record the final ordinary sampler request."""

    calls: ClassVar[list[dict[str, object]]] = []
    output: ClassVar[dict[str, Any]] = {"samples": torch.ones((1, 16, 1, 2, 2))}

    def sample(self, **kwargs: object) -> dict[str, Any]:
        """Return one recognizable completed latent."""

        type(self).calls.append(kwargs)
        return self.output


def test_full_context_service_delegates_prepared_model_to_ordinary_sampler() -> None:
    """Keep full-context sampling separate from reusable model preparation."""

    calls: list[dict[str, object]] = []

    class PreparedService:
        """Return one fixed prepared model while recording regional inputs."""

        def prepare(self, **kwargs: object) -> PreparedAttentionCouplingModel:
            """Record and return the fixed preparation."""

            calls.append(kwargs)
            masks = torch.ones((1, 1, 1))
            return PreparedAttentionCouplingModel(
                "derived",
                "base+",
                "base-",
                RegionalMaskBank(masks, masks.clone(), 1, 1),
            )

    original_preparation = (
        AttentionCouplingSamplingService.model_preparation_service_class
    )
    original_sampling = AttentionCouplingSamplingService.sampling_service_class
    AttentionCouplingSamplingService.model_preparation_service_class = PreparedService  # type: ignore[assignment]
    AttentionCouplingSamplingService.sampling_service_class = _SamplingService  # type: ignore[assignment]
    _SamplingService.calls = []
    latent = {"samples": torch.zeros((1, 16, 1, 2, 2))}
    try:
        output = AttentionCouplingSamplingService().sample(
            model="model",
            seed=5,
            steps=12,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive="regional+",
            negative="regional-",
            region_masks="masks",
            regional_prompt_weight=0.75,
            region_mask_feather=8,
            latent_image=latent,
            denoise=0.8,
        )
    finally:
        AttentionCouplingSamplingService.model_preparation_service_class = (
            original_preparation
        )
        AttentionCouplingSamplingService.sampling_service_class = original_sampling

    assert output is _SamplingService.output
    assert calls[0]["region_masks"] == "masks"
    assert calls[0]["execution_mode"] is RegionalAttentionExecutionMode.FULL
    sampler_call = _SamplingService.calls[0]
    assert sampler_call["model"] == "derived"
    assert sampler_call["positive"] == "base+"
    assert sampler_call["negative"] == "base-"
    assert sampler_call["latent_image"] is latent
