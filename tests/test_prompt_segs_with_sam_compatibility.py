# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compatibility tests for Prompt SEGS w/ SAM model socket objects."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from simple_syrup.masking.prompt_segs_with_sam_service import PromptSEGSWithSAMService
from test_helpers import make_image_tensor


class PredictBoxesModel:
    """DINO object exposing predict_boxes."""

    def predict_boxes(
        self,
        image: torch.Tensor,
        prompt: str,
        threshold: float,
    ) -> torch.Tensor:
        """Return one full-image box."""

        return torch.tensor([[0.0, 0.0, float(image.shape[1]), float(image.shape[0])]])


class Wrapper:
    """Impact-style SAM wrapper exposing predict."""

    def predict(
        self,
        image: object,
        points: list[object],
        plabs: list[int],
        bbox: list[float],
        threshold: float,
    ) -> list[torch.Tensor]:
        """Return one all-white mask."""

        return [torch.ones((2, 2), dtype=torch.float32)]


def test_prompt_service_accepts_impact_style_sam_model() -> None:
    """Impact-style SAM_MODEL objects can produce prompt SEGS."""

    service = PromptSEGSWithSAMService()

    result = service.prompt(
        image=make_image_tensor(batch_size=1, height=2, width=2),
        sam_model=SimpleNamespace(sam_wrapper=Wrapper()),
        grounding_dino_model=PredictBoxesModel(),
        vitmatte_model=None,
        positive_prompt="face",
        negative_prompt="",
        confidence_threshold=0.3,
        size_threshold=1,
        bbox_dilation=0,
        mask_dilation=0,
        detail_method="GuidedFilter",
        detail_erode=0,
        detail_dilate=0,
        black_point=0.0,
        white_point=1.0,
        refine_mask=False,
        mask_refinement_max_size=4,
        execution_device="cpu",
        crop_factor=1.0,
    )

    assert len(result[1]) == 1
    assert torch.equal(
        cast(torch.Tensor, result[1][0].cropped_mask), torch.ones((2, 2))
    )


def test_prompt_service_rejects_invalid_dino_model() -> None:
    """Invalid DINO socket objects fail clearly."""

    service = PromptSEGSWithSAMService()

    with pytest.raises(TypeError, match="Prompt SEGS w/ SAM"):
        service.prompt(
            image=make_image_tensor(batch_size=1, height=2, width=2),
            sam_model=SimpleNamespace(sam_wrapper=Wrapper()),
            grounding_dino_model=object(),
            vitmatte_model=None,
            positive_prompt="face",
            negative_prompt="",
            confidence_threshold=0.3,
            size_threshold=1,
            bbox_dilation=0,
            mask_dilation=0,
            detail_method="GuidedFilter",
            detail_erode=0,
            detail_dilate=0,
            black_point=0.0,
            white_point=1.0,
            refine_mask=False,
            mask_refinement_max_size=4,
            execution_device="cpu",
            crop_factor=1.0,
        )
