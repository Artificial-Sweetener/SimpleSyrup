# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify Attention Coupling composes through the tiled sampling authority."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import torch

from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_features import (
    RegionalFeature,
    RegionalFeatureRequest,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.services.prepared_attention_coupling_model import (
    PreparedAttentionCouplingModel,
)
from simple_syrup.services.tiled_attention_coupling_sampling_service import (
    TiledAttentionCouplingSamplingService,
)


class _PreparationService:
    """Return one recognizable derived attention model and base conditioning."""

    calls: ClassVar[list[dict[str, object]]] = []

    def prepare(self, **kwargs: object) -> PreparedAttentionCouplingModel:
        """Record the request and return a fixed prepared sample."""

        type(self).calls.append(kwargs)
        masks = torch.ones((1, 1, 1))
        return PreparedAttentionCouplingModel(
            model="attention-model",
            positive="base-positive",
            negative="base-negative",
            mask_bank=RegionalMaskBank(masks, masks.clone(), 1, 1),
        )


class _TiledSamplingService:
    """Capture the sole tiled sampling delegation."""

    calls: ClassVar[list[dict[str, object]]] = []
    output: ClassVar[dict[str, Any]] = {"samples": torch.ones((1, 4, 4, 4))}

    def sample(self, **kwargs: object) -> dict[str, Any]:
        """Record and return the fixed tiled output."""

        type(self).calls.append(kwargs)
        return self.output


@pytest.mark.parametrize(
    "diffusion_mode",
    ["multidiffusion", "mixture_of_diffusers"],
)
def test_tiled_attention_service_prepares_once_and_delegates_all_tiling(
    monkeypatch: pytest.MonkeyPatch,
    diffusion_mode: str,
) -> None:
    """Keep model derivation separate from existing tiled execution ownership."""

    service = TiledAttentionCouplingSamplingService()
    monkeypatch.setattr(
        TiledAttentionCouplingSamplingService,
        "model_preparation_service_class",
        _PreparationService,
    )
    monkeypatch.setattr(
        TiledAttentionCouplingSamplingService,
        "tiled_sampling_service_class",
        _TiledSamplingService,
    )
    latent = {"samples": torch.zeros((1, 4, 4, 4))}
    masks = torch.ones((2, 4, 4))
    _PreparationService.calls = []
    _TiledSamplingService.calls = []
    output = service.sample(
        diffusion_mode=diffusion_mode,
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
        latent_tile_width=64,
        latent_tile_height=80,
        latent_tile_overlap=16,
        latent_tile_batch_size=4,
    )

    assert output is _TiledSamplingService.output
    assert _PreparationService.calls == [
        {
            "model": "base-model",
            "positive": "regional-positive",
            "negative": "regional-negative",
            "region_masks": masks,
            "regional_prompt_weight": 0.8,
            "region_mask_feather": 16,
            "latent_image": latent,
            "execution_mode": RegionalAttentionExecutionMode.TILED,
        }
    ]
    call = _TiledSamplingService.calls[0]
    assert call["model"] == "attention-model"
    assert call["positive"] == "base-positive"
    assert call["negative"] == "base-negative"
    assert call["region_masks"] is None
    assert call["latent_image"] is latent
    request = cast(RegionalFeatureRequest, call["feature_request"])
    assert request.features == frozenset({RegionalFeature.ATTENTION_COUPLING})
    assert call["latent_tile_batch_size"] == 4
    assert call["diffusion_mode"] == diffusion_mode
