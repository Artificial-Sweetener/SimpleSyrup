# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize SDXL Base and Refiner native conditioning contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import comfy.model_base
import torch


class _Embedder:
    """Expose recognizable deterministic scalar embeddings at SDXL width."""

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """Repeat one supplied scalar over the installed 256-wide field."""

        return value.reshape(1, 1).expand(1, 256)


def test_sdxl_base_preserves_pooled_output_and_all_six_geometry_fields() -> None:
    """Retain pooled CLIP-G plus original/crop/target microconditioning."""

    model = cast(
        Any,
        SimpleNamespace(embedder=_Embedder(), noise_augmentor=object()),
    )
    pooled = torch.arange(1280, dtype=torch.float32).reshape(1, 1280)

    vector = comfy.model_base.SDXL.encode_adm(
        model,
        pooled_output=pooled,
        width=1024,
        height=768,
        crop_w=16,
        crop_h=32,
        target_width=1536,
        target_height=1152,
    )

    assert vector.shape == (1, 2816)
    assert vector[:, :1280] is not pooled
    torch.testing.assert_close(vector[:, :1280], pooled)
    fields = vector[:, 1280:].reshape(1, 6, 256)
    torch.testing.assert_close(
        fields[:, :, 0],
        torch.tensor([[768.0, 1024.0, 32.0, 16.0, 1152.0, 1536.0]]),
    )


def test_sdxl_refiner_preserves_pooled_output_and_aesthetic_contract() -> None:
    """Retain Refiner pooled CLIP-G plus size/crop/aesthetic conditioning."""

    model = cast(
        Any,
        SimpleNamespace(embedder=_Embedder(), noise_augmentor=object()),
    )
    pooled = torch.ones((1, 1280), dtype=torch.float32)

    positive = comfy.model_base.SDXLRefiner.encode_adm(
        model,
        pooled_output=pooled,
        width=1536,
        height=1152,
        crop_w=8,
        crop_h=4,
        aesthetic_score=7.5,
        prompt_type="positive",
    )
    negative = comfy.model_base.SDXLRefiner.encode_adm(
        model,
        pooled_output=pooled,
        width=1536,
        height=1152,
        crop_w=8,
        crop_h=4,
        aesthetic_score=1.25,
        prompt_type="negative",
    )

    assert positive.shape == negative.shape == (1, 2560)
    torch.testing.assert_close(positive[:, :1280], pooled)
    positive_fields = positive[:, 1280:].reshape(1, 5, 256)
    negative_fields = negative[:, 1280:].reshape(1, 5, 256)
    torch.testing.assert_close(
        positive_fields[:, :, 0],
        torch.tensor([[1152.0, 1536.0, 4.0, 8.0, 7.5]]),
    )
    torch.testing.assert_close(
        negative_fields[:, :, 0],
        torch.tensor([[1152.0, 1536.0, 4.0, 8.0, 1.25]]),
    )
