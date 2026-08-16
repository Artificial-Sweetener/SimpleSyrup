# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify Attention Coupling composes through Contextual Diffusion authority."""

from __future__ import annotations

from typing import ClassVar, cast

import pytest
import torch

from simple_syrup.domain.context_segs import ContextSegmentSequence
from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_features import (
    RegionalFeature,
    RegionalFeatureRequest,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.services.contextual_attention_coupling_sampling_service import (
    ContextualAttentionCouplingSamplingService,
)
from simple_syrup.services.contextual_diffusion_sampling_service import (
    ContextualDiffusionSamplingResult,
)
from simple_syrup.services.prepared_attention_coupling_model import (
    PreparedAttentionCouplingModel,
)


class _PreparationService:
    """Return one recognizable prepared attention model and base pair."""

    calls: ClassVar[list[dict[str, object]]] = []

    def prepare(self, **kwargs: object) -> PreparedAttentionCouplingModel:
        """Record the request and return one fixed preparation."""

        type(self).calls.append(kwargs)
        masks = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        return PreparedAttentionCouplingModel(
            "attention-model",
            "base-positive",
            "base-negative",
            RegionalMaskBank(masks, masks.clone(), 2, 2),
        )


class _ContextualSamplingService:
    """Capture the sole Contextual sampling delegation."""

    calls: ClassVar[list[dict[str, object]]] = []
    output = ContextualDiffusionSamplingResult(
        {"samples": torch.ones((1, 4, 4, 4))},
        ((32, 32), ContextSegmentSequence(())),
    )

    def sample(self, **kwargs: object) -> ContextualDiffusionSamplingResult:
        """Record and return the fixed Contextual result."""

        type(self).calls.append(kwargs)
        return self.output


@pytest.mark.parametrize("diffusion_mode", ["multidiffusion", "mixture_of_diffusers"])
def test_contextual_attention_service_prepares_once_and_delegates_local_views(
    monkeypatch: pytest.MonkeyPatch,
    diffusion_mode: str,
) -> None:
    """Keep attention preparation separate from Contextual planning and fusion."""

    monkeypatch.setattr(
        ContextualAttentionCouplingSamplingService,
        "model_preparation_service_class",
        _PreparationService,
    )
    monkeypatch.setattr(
        ContextualAttentionCouplingSamplingService,
        "contextual_sampling_service_class",
        _ContextualSamplingService,
    )
    _PreparationService.calls = []
    _ContextualSamplingService.calls = []
    latent = {"samples": torch.zeros((1, 4, 4, 4))}
    masks = torch.ones((2, 4, 4))

    output = ContextualAttentionCouplingSamplingService().sample(
        model="base-model",
        seed=9,
        steps=12,
        cfg=1.0,
        sampler_name="er_sde",
        scheduler="simple",
        positive="regional-positive",
        negative="regional-negative",
        region_masks=masks,
        regional_prompt_weight=0.8,
        region_mask_feather=16,
        latent_image=latent,
        denoise=1.0,
        diffusion_mode=diffusion_mode,
        latent_context_size=64,
        latent_context_overlap=16,
        latent_context_batch_size=4,
        global_weight=1.0,
        global_steps=3,
        global_decay=0.5,
        segs="segs",
    )

    assert output is _ContextualSamplingService.output
    assert _PreparationService.calls == [
        {
            "model": "base-model",
            "positive": "regional-positive",
            "negative": "regional-negative",
            "region_masks": masks,
            "regional_prompt_weight": 0.8,
            "region_mask_feather": 16,
            "latent_image": latent,
            "execution_mode": RegionalAttentionExecutionMode.CONTEXTUAL,
        }
    ]
    call = _ContextualSamplingService.calls[0]
    assert call["model"] == "attention-model"
    assert call["positive"] == "base-positive"
    assert call["negative"] == "base-negative"
    assert call["region_masks"] is None
    torch.testing.assert_close(
        cast(torch.Tensor, call["planning_region_masks"]),
        torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
    )
    assert call["latent_image"] is latent
    assert call["segs"] == "segs"
    assert call["diffusion_mode"] == diffusion_mode
    request = cast(RegionalFeatureRequest, call["feature_request"])
    assert request.features == frozenset({RegionalFeature.ATTENTION_COUPLING})
