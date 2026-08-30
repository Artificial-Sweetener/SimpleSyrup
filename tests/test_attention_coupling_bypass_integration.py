# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify public Attention Coupling nodes reach ordinary sampling unchanged."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import torch

from simple_syrup.domain.regional_features import EMPTY_REGIONAL_FEATURE_REQUEST
from simple_syrup.nodes_v3.ksampler_attention_coupling import (
    KSamplerAttentionCouplingV3,
)
from simple_syrup.nodes_v3.ksampler_tiled_attention_coupling import (
    KSamplerTiledAttentionCouplingV3,
)
from simple_syrup.services.attention_coupling_sampling_service import (
    AttentionCouplingSamplingService,
)
from simple_syrup.services.prepared_attention_coupling_model import (
    PreparedAttentionCouplingModel,
)
from simple_syrup.services.tiled_attention_coupling_sampling_service import (
    TiledAttentionCouplingSamplingService,
)


class _PreparationMustNotRun:
    """Fail if a public ordinary request enters model preparation."""

    def prepare(self, **kwargs: object) -> PreparedAttentionCouplingModel:
        """Reject an unexpected Attention Coupling preparation call."""

        raise AssertionError(f"unexpected Attention Coupling preparation: {kwargs}")


class _OrdinarySamplingBoundary:
    """Capture full-latent requests at the ordinary KSampler boundary."""

    calls: ClassVar[list[dict[str, object]]] = []

    def sample(self, **kwargs: object) -> dict[str, Any]:
        """Return the unchanged latent after recording the request."""

        type(self).calls.append(kwargs)
        latent = kwargs["latent_image"]
        assert isinstance(latent, dict)
        return latent


class _TiledSamplingBoundary:
    """Capture requests at the ordinary tiled-diffusion boundary."""

    calls: ClassVar[list[dict[str, object]]] = []

    def sample(self, **kwargs: object) -> dict[str, Any]:
        """Return the unchanged latent after recording the request."""

        type(self).calls.append(kwargs)
        latent = kwargs["latent_image"]
        assert isinstance(latent, dict)
        return latent


def test_public_full_node_bypasses_to_normal_img2img(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove node-to-service wiring omits every regional runtime owner."""

    monkeypatch.setattr(
        KSamplerAttentionCouplingV3,
        "sampling_service_class",
        AttentionCouplingSamplingService,
    )
    monkeypatch.setattr(
        AttentionCouplingSamplingService,
        "model_preparation_service_class",
        _PreparationMustNotRun,
    )
    monkeypatch.setattr(
        AttentionCouplingSamplingService,
        "sampling_service_class",
        _OrdinarySamplingBoundary,
    )
    _OrdinarySamplingBoundary.calls = []
    latent = {"samples": torch.zeros((1, 16, 1, 5, 7))}

    (output,) = KSamplerAttentionCouplingV3.execute(
        model="model",
        seed=31,
        steps=18,
        cfg=4.0,
        sampler_name="euler",
        scheduler="normal",
        positive="positive",
        negative="negative",
        latent_image=latent,
        denoise=0.38,
    )

    assert output is latent
    assert len(_OrdinarySamplingBoundary.calls) == 1
    call = _OrdinarySamplingBoundary.calls[0]
    assert call["model"] == "model"
    assert call["positive"] == "positive"
    assert call["negative"] == "negative"
    assert call["latent_image"] is latent
    assert call["denoise"] == 0.38


def test_public_tiled_node_bypasses_to_normal_tiled_img2img(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove tiled node-to-service wiring preserves every image-to-image control."""

    monkeypatch.setattr(
        KSamplerTiledAttentionCouplingV3,
        "sampling_service_class",
        TiledAttentionCouplingSamplingService,
    )
    monkeypatch.setattr(
        TiledAttentionCouplingSamplingService,
        "model_preparation_service_class",
        _PreparationMustNotRun,
    )
    monkeypatch.setattr(
        TiledAttentionCouplingSamplingService,
        "tiled_sampling_service_class",
        _TiledSamplingBoundary,
    )
    _TiledSamplingBoundary.calls = []
    latent = {"samples": torch.zeros((1, 16, 1, 10, 14))}

    (output,) = KSamplerTiledAttentionCouplingV3.execute(
        model="model",
        seed=37,
        steps=26,
        cfg=4.5,
        sampler_name="euler_ancestral",
        scheduler="normal",
        positive="positive",
        negative="negative",
        latent_image=latent,
        denoise=0.44,
        diffusion_mode="mixture_of_diffusers",
        latent_tile_width=112,
        latent_tile_height=96,
        latent_tile_overlap=28,
        latent_tile_batch_size=3,
    )

    assert output is latent
    assert len(_TiledSamplingBoundary.calls) == 1
    call = _TiledSamplingBoundary.calls[0]
    assert call["model"] == "model"
    assert call["positive"] == "positive"
    assert call["negative"] == "negative"
    assert call["latent_image"] is latent
    assert call["denoise"] == 0.44
    assert call["diffusion_mode"] == "mixture_of_diffusers"
    assert call["latent_tile_width"] == 112
    assert call["latent_tile_height"] == 96
    assert call["latent_tile_overlap"] == 28
    assert call["latent_tile_batch_size"] == 3
    assert call["feature_request"] is EMPTY_REGIONAL_FEATURE_REQUEST
