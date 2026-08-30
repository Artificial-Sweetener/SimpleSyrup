# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify full-context Attention Coupling sampling orchestration."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.services.attention_coupling_sampling_service import (
    AttentionCouplingSamplingService,
)
from simple_syrup.services.prepared_attention_coupling_model import (
    PreparedAttentionCouplingModel,
)


class _SamplingService:
    """Record the final ordinary sampler request."""

    calls: ClassVar[list[dict[str, object]]] = []
    output: ClassVar[dict[str, Any]] = {"samples": torch.ones((1, 16, 1, 2, 2))}

    def sample(self, **kwargs: object) -> dict[str, Any]:
        """Return one recognizable completed latent."""

        type(self).calls.append(kwargs)
        return self.output


class _PreparationMustNotRun:
    """Fail if an ordinary request reaches Attention Coupling preparation."""

    def prepare(self, **kwargs: object) -> PreparedAttentionCouplingModel:
        """Reject an unexpected preparation call."""

        raise AssertionError(f"unexpected Attention Coupling preparation: {kwargs}")


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
    positive = ConditioningBatch(("base+", "regional+"))
    negative = ConditioningBatch(("base-", "regional-"))
    try:
        output = AttentionCouplingSamplingService().sample(
            model="model",
            seed=5,
            steps=12,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
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
    assert calls[0]["positive"] is positive
    assert calls[0]["negative"] is negative
    assert calls[0]["region_masks"] == "masks"
    assert calls[0]["execution_mode"] is RegionalAttentionExecutionMode.FULL
    sampler_call = _SamplingService.calls[0]
    assert sampler_call["model"] == "derived"
    assert sampler_call["positive"] == "base+"
    assert sampler_call["negative"] == "base-"
    assert sampler_call["latent_image"] is latent


def test_ordinary_request_bypasses_preparation_and_preserves_img2img_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send ordinary conditioning directly to the normal KSampler authority."""

    monkeypatch.setattr(
        AttentionCouplingSamplingService,
        "model_preparation_service_class",
        _PreparationMustNotRun,
    )
    monkeypatch.setattr(
        AttentionCouplingSamplingService,
        "sampling_service_class",
        _SamplingService,
    )
    _SamplingService.calls = []
    latent = {"samples": torch.zeros((1, 16, 1, 4, 6))}

    output = AttentionCouplingSamplingService().sample(
        model="model",
        seed=17,
        steps=24,
        cfg=4.5,
        sampler_name="euler",
        scheduler="normal",
        positive="positive",
        negative="negative",
        region_masks=None,
        regional_prompt_weight=0.8,
        region_mask_feather=12,
        latent_image=latent,
        denoise=0.42,
    )

    assert output is _SamplingService.output
    assert _SamplingService.calls == [
        {
            "model": "model",
            "seed": 17,
            "steps": 24,
            "cfg": 4.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "positive": "positive",
            "negative": "negative",
            "latent_image": latent,
            "denoise": 0.42,
        }
    ]


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
def test_incomplete_request_fails_before_preparation_or_sampling(
    monkeypatch: pytest.MonkeyPatch,
    positive: object,
    negative: object,
    region_masks: object | None,
    message: str,
) -> None:
    """Fail closed before either runtime authority observes partial inputs."""

    monkeypatch.setattr(
        AttentionCouplingSamplingService,
        "model_preparation_service_class",
        _PreparationMustNotRun,
    )
    monkeypatch.setattr(
        AttentionCouplingSamplingService,
        "sampling_service_class",
        _SamplingService,
    )
    _SamplingService.calls = []

    with pytest.raises(ValueError, match=message):
        AttentionCouplingSamplingService().sample(
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
        )

    assert _SamplingService.calls == []
