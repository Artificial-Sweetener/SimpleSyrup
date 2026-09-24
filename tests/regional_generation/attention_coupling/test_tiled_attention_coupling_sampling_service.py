# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify Attention Coupling composes through the tiled sampling authority."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_features import (
    EMPTY_REGIONAL_FEATURE_REQUEST,
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


class _PreparationMustNotRun:
    """Fail if an ordinary tiled request reaches regional preparation."""

    def prepare(self, **kwargs: object) -> PreparedAttentionCouplingModel:
        """Reject an unexpected preparation call."""

        raise AssertionError(f"unexpected Attention Coupling preparation: {kwargs}")


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
    positive = ConditioningBatch(("base-positive", "regional-positive"))
    negative = ConditioningBatch(("base-negative", "regional-negative"))
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
        positive=positive,
        negative=negative,
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
            "positive": positive,
            "negative": negative,
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


def test_ordinary_request_bypasses_preparation_and_preserves_tiled_img2img(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route ordinary conditioning through normal tiled diffusion unchanged."""

    monkeypatch.setattr(
        TiledAttentionCouplingSamplingService,
        "model_preparation_service_class",
        _PreparationMustNotRun,
    )
    monkeypatch.setattr(
        TiledAttentionCouplingSamplingService,
        "tiled_sampling_service_class",
        _TiledSamplingService,
    )
    _TiledSamplingService.calls = []
    latent = {"samples": torch.zeros((1, 16, 1, 8, 12))}

    output = TiledAttentionCouplingSamplingService().sample(
        diffusion_mode="mixture_of_diffusers",
        model="model",
        seed=23,
        steps=30,
        cfg=5.0,
        sampler_name="euler_ancestral",
        scheduler="normal",
        positive="positive",
        negative="negative",
        region_masks=None,
        regional_prompt_weight=0.9,
        region_mask_feather=24,
        latent_image=latent,
        denoise=0.35,
        latent_tile_width=96,
        latent_tile_height=80,
        latent_tile_overlap=20,
        latent_tile_batch_size=3,
    )

    assert output is _TiledSamplingService.output
    assert len(_TiledSamplingService.calls) == 1
    call = _TiledSamplingService.calls[0]
    assert call["model"] == "model"
    assert call["positive"] == "positive"
    assert call["negative"] == "negative"
    assert call["latent_image"] is latent
    assert call["denoise"] == 0.35
    assert call["diffusion_mode"] == "mixture_of_diffusers"
    assert call["latent_tile_width"] == 96
    assert call["latent_tile_height"] == 80
    assert call["latent_tile_overlap"] == 20
    assert call["latent_tile_batch_size"] == 3
    assert call["feature_request"] is EMPTY_REGIONAL_FEATURE_REQUEST
    assert call["region_masks"] is None


@pytest.mark.parametrize(
    ("positive", "negative", "region_masks", "message"),
    [
        (
            ConditioningBatch(("global", "region")),
            "negative",
            None,
            "batches require region_masks",
        ),
        (
            "positive",
            "negative",
            "masks",
            "require a CONDITIONING_BATCH",
        ),
    ],
)
def test_incomplete_tiled_request_fails_before_both_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
    positive: object,
    negative: object,
    region_masks: object | None,
    message: str,
) -> None:
    """Reject partial regional tiled inputs before preparation or tiling."""

    monkeypatch.setattr(
        TiledAttentionCouplingSamplingService,
        "model_preparation_service_class",
        _PreparationMustNotRun,
    )
    monkeypatch.setattr(
        TiledAttentionCouplingSamplingService,
        "tiled_sampling_service_class",
        _TiledSamplingService,
    )
    _TiledSamplingService.calls = []

    with pytest.raises(ValueError, match=message):
        TiledAttentionCouplingSamplingService().sample(
            diffusion_mode="multidiffusion",
            model="model",
            seed=1,
            steps=10,
            cfg=1.0,
            sampler_name="euler",
            scheduler="normal",
            positive=positive,
            negative=negative,
            region_masks=region_masks,
            regional_prompt_weight=0.5,
            region_mask_feather=0,
            latent_image={"samples": torch.zeros((1, 4, 2, 2))},
            denoise=1.0,
            latent_tile_width=64,
            latent_tile_height=64,
            latent_tile_overlap=16,
            latent_tile_batch_size=1,
        )

    assert _TiledSamplingService.calls == []
