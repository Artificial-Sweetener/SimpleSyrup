# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify optional sampler masks admit only complete regional requests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.sampler_options import (
    AttentionCouplingOptions,
    ContextualDiffusionOptions,
    SamplerOptions,
    TilingOptions,
)
from simple_syrup.nodes_v3.ksampler import KSamplerV3
from simple_syrup.services import sampler_options_sampling_service as routing


@pytest.fixture(params=["full", "tiled", "context"])
def attention_options(request: pytest.FixtureRequest) -> SamplerOptions:
    """Configure attention independently from every spatial sampling authority."""
    return SamplerOptions(
        attention_coupling=AttentionCouplingOptions(),
        tiling=TilingOptions() if request.param == "tiled" else None,
        contextual_diffusion=(
            ContextualDiffusionOptions() if request.param == "context" else None
        ),
    )


def sampling_inputs() -> dict[str, Any]:
    """Describe a small host-facing latent without loading any neural model."""
    return {
        "model": object(),
        "seed": 17,
        "steps": 4,
        "cfg": 3.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "positive": [],
        "negative": None,
        "latent_image": {"samples": torch.zeros((1, 4, 64, 96))},
    }


@pytest.mark.parametrize("missing", ["masks", "batch"])
def test_partial_attention_request_fails_before_model_execution(
    attention_options: SamplerOptions,
    missing: str,
) -> None:
    """Surface the established actionable errors through the actual sampler node."""
    values = sampling_inputs()
    if missing == "masks":
        values["positive"] = ConditioningBatch(([], []))
        message = "conditioning batches require region_masks"
    else:
        values["region_masks"] = torch.ones((1, 64, 96))
        message = "region_masks require a CONDITIONING_BATCH"
    with pytest.raises(ValueError, match=message):
        KSamplerV3.execute(**values, options=attention_options)


def test_attention_options_without_regional_inputs_use_ordinary_sampling(
    attention_options: SamplerOptions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypass inactive attention while preserving the selected spatial strategy."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def boundary(label: str) -> type:
        """Replace neural sampling after the real node and compiler admit inputs."""

        class Recorder:
            """Capture the expensive sampling boundary without preparing a model."""

            def sample(self, **kwargs: Any) -> Any:
                """Retain sampling payloads and the host's exact output shape."""
                calls.append((label, kwargs))
                if label == "context":
                    return SimpleNamespace(latent=kwargs["latent_image"])
                return kwargs["latent_image"]

        return Recorder

    for name, label in (
        ("KSamplerSamplingService", "full"),
        ("TiledDiffusionSamplingService", "tiled"),
        ("ContextualDiffusionSamplingService", "context"),
    ):
        monkeypatch.setattr(routing, name, boundary(label))
    values = sampling_inputs()
    assert KSamplerV3.execute(**values, options=attention_options) == (
        values["latent_image"],
    )
    expected = (
        "context"
        if attention_options.contextual_diffusion is not None
        else "tiled"
        if attention_options.tiling is not None
        else "full"
    )
    assert len(calls) == 1 and calls[0][0] == expected
    assert "region_masks" not in calls[0][1]
