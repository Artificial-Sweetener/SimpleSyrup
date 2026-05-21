# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Prompt SEGS w/ SAM ComfyUI node declaration."""

from __future__ import annotations

from typing import Any, cast

import torch

from simple_syrup.domain.segs import (
    SORT_ORDER_OPTIONS,
    BoundingBox,
    CropRegion,
    Segment,
)
from simple_syrup.nodes.prompt_segs_with_sam import PromptSEGSWithSAM
from simple_syrup.services.segs_output_service import CombinedSegsResult
from test_helpers import make_image_tensor


def test_prompt_segs_with_sam_node_contract_constants() -> None:
    """Node constants match the public ComfyUI contract."""

    assert PromptSEGSWithSAM.RETURN_TYPES == ("SEGS", "MASK")
    assert PromptSEGSWithSAM.RETURN_NAMES == ("segs", "mask")
    assert PromptSEGSWithSAM.OUTPUT_IS_LIST == (True, False)
    assert PromptSEGSWithSAM.OUTPUT_TOOLTIPS == (
        "Prompted regions as separate or combined SEGS based on combine_segs.",
        "Combined prompted area as a standard ComfyUI mask.",
    )
    assert PromptSEGSWithSAM.FUNCTION == "prompt"
    assert PromptSEGSWithSAM.CATEGORY == "SimpleSyrup/Detection"
    assert PromptSEGSWithSAM.DESCRIPTION == (
        "Finds prompt-matched regions with GroundingDINO, segments them with SAM, "
        "and returns SEGS plus a combined mask."
    )


def test_prompt_segs_with_sam_node_declares_expected_inputs() -> None:
    """Node input declaration includes aligned prompt SEGS controls."""

    input_types: dict[str, dict[str, Any]] = PromptSEGSWithSAM.INPUT_TYPES()
    required = input_types["required"]
    optional = input_types["optional"]

    assert list(required) == [
        "image",
        "sam_model",
        "grounding_dino_model",
        "positive_prompt",
        "negative_prompt",
        "confidence_threshold",
        "size_threshold",
        "bbox_dilation",
        "mask_dilation",
        "detail_method",
        "detail_erode",
        "detail_dilate",
        "black_point",
        "white_point",
        "refine_mask",
        "mask_refinement_max_size",
        "execution_device",
        "crop_factor",
        "sort_order",
        "combine_segs",
    ]
    assert set(optional) == {"vitmatte_model"}
    assert "hidden" not in input_types
    assert "box_threshold" not in required
    assert "process_detail" not in required
    assert "max_size_pixels" not in required
    assert required["sam_model"][0] == "SAM_MODEL"
    assert required["grounding_dino_model"][0] == "GROUNDING_DINO_MODEL,DINO_MODEL"
    assert optional["vitmatte_model"][0] == "VITMATTE_MODEL"
    assert required["positive_prompt"][1]["default"] == ""
    assert required["positive_prompt"][1]["tooltip"] == (
        "Text describing the regions to detect."
    )
    assert required["detail_method"][0] == ["GuidedFilter", "PyMatting", "VITMatte"]
    assert required["sort_order"][0] == SORT_ORDER_OPTIONS
    assert required["sort_order"][1]["default"] == SORT_ORDER_OPTIONS[0]
    assert required["combine_segs"][0] == "BOOLEAN"
    assert required["combine_segs"][1]["default"] is False
    for input_declaration in required.values():
        assert "tooltip" in input_declaration[1]
    assert "tooltip" in optional["vitmatte_model"][1]


def test_prompt_segs_with_sam_node_returns_individual_segs_when_combine_disabled() -> (
    None
):
    """Node execution keeps prompted SEGS separate when combine_segs is false."""

    image = make_image_tensor(batch_size=1, height=3, width=3)
    segs = _segs(_segment("face", CropRegion(0, 0, 2, 2), 0.9))
    combined_mask = torch.ones((1, 3, 3), dtype=torch.float32)
    combined = CombinedSegsResult(
        segs=_segs(_segment("combined", CropRegion(0, 0, 3, 3), 1.0)),
        mask=combined_mask,
    )

    class FakeService:
        """Service double for node delegation."""

        def prompt(
            self, **kwargs: object
        ) -> tuple[tuple[int, int], tuple[Segment, ...]]:
            """Return fixed SEGS while checking key forwarded values."""

            assert kwargs["confidence_threshold"] == 0.3
            assert kwargs["size_threshold"] == 10
            assert kwargs["refine_mask"] is True
            assert kwargs["mask_refinement_max_size"] == 2048
            assert kwargs["vitmatte_model"] == {"vitmatte": "model"}
            return segs

    builder = _FixedCombinedBuilder(image, segs, combined)
    result = _prompt_with_fakes(
        service=FakeService(),
        builder=builder,
        image=image,
        combine_segs=False,
        vitmatte_model={"vitmatte": "model"},
    )

    assert result[0] == [(segs[0], list(segs[1]))]
    assert torch.equal(cast(torch.Tensor, result[1]), combined_mask)
    assert builder.call_count == 1


def test_prompt_segs_with_sam_node_returns_unioned_segs_when_combine_enabled() -> None:
    """Node execution emits one unioned SEGS when combine_segs is true."""

    image = make_image_tensor(batch_size=1, height=3, width=3)
    segs = _segs(_segment("face", CropRegion(0, 0, 2, 2), 0.9))
    combined = CombinedSegsResult(
        segs=_segs(_segment("combined", CropRegion(0, 0, 3, 3), 1.0)),
        mask=torch.ones((1, 3, 3), dtype=torch.float32),
    )

    result = _prompt_with_fakes(
        service=_FakePromptService(segs),
        builder=_FixedCombinedBuilder(image, segs, combined),
        image=image,
        combine_segs=True,
    )

    segs_list = cast(list[tuple[object, list[Segment]]], result[0])
    _header, segments = segs_list[0]
    assert [segment.label for segment in segments] == ["combined"]
    assert torch.equal(cast(torch.Tensor, result[1]), combined.mask)


def test_prompt_segs_with_sam_node_processes_image_batches_with_individual_segs() -> (
    None
):
    """A batch image produces one prompted SEGS output per image."""

    image = make_image_tensor(batch_size=2, height=3, width=3)
    service = _FakeBatchPromptService()
    builder = _FakeBatchCombinedBuilder()

    result = _prompt_with_fakes(
        service=service,
        builder=builder,
        image=image,
        combine_segs=False,
    )

    segs_list = cast(list[tuple[object, list[Segment]]], result[0])
    assert service.call_count == 2
    assert builder.call_count == 2
    assert [segments[0].label for _header, segments in segs_list] == [
        "image-0",
        "image-1",
    ]
    assert cast(torch.Tensor, result[1]).shape == (2, 3, 3)


def test_prompt_segs_with_sam_node_processes_image_batches_with_unioned_segs() -> None:
    """A batch image produces one combined SEGS output per image."""

    image = make_image_tensor(batch_size=2, height=3, width=3)
    service = _FakeBatchPromptService()
    builder = _FakeBatchCombinedBuilder()

    result = _prompt_with_fakes(
        service=service,
        builder=builder,
        image=image,
        combine_segs=True,
    )

    segs_list = cast(list[tuple[object, list[Segment]]], result[0])
    assert [segments[0].label for _header, segments in segs_list] == [
        "combined-image-0",
        "combined-image-1",
    ]
    assert builder.call_count == 2
    assert cast(torch.Tensor, result[1]).shape == (2, 3, 3)


class _FakePromptService:
    """Service double that returns fixed SEGS."""

    def __init__(self, segs: tuple[tuple[int, int], tuple[Segment, ...]]) -> None:
        """Store the SEGS payload returned by prompt calls."""

        self._segs = segs

    def prompt(self, **kwargs: object) -> tuple[tuple[int, int], tuple[Segment, ...]]:
        """Return fixed SEGS."""

        del kwargs
        return self._segs


class _FakeBatchPromptService:
    """Service double that records per-image prompt calls."""

    def __init__(self) -> None:
        """Create an empty call counter."""

        self.call_count = 0

    def prompt(self, **kwargs: object) -> tuple[tuple[int, int], tuple[Segment, ...]]:
        """Return a labeled SEGS payload for one image slice."""

        assert cast(torch.Tensor, kwargs["image"]).shape == (1, 3, 3, 3)
        label = f"image-{self.call_count}"
        self.call_count += 1
        return _segs(_segment(label, CropRegion(0, 0, 2, 2), 0.9))


class _FixedCombinedBuilder:
    """Combined-result builder double with input assertions."""

    def __init__(
        self,
        expected_image: torch.Tensor,
        expected_segs: tuple[tuple[int, int], tuple[Segment, ...]],
        combined: CombinedSegsResult,
    ) -> None:
        """Store expected call arguments and the fixed return value."""

        self._expected_image = expected_image
        self._expected_segs = expected_segs
        self._combined = combined
        self.call_count = 0

    def __call__(
        self,
        source_image: object,
        source_segs: tuple[tuple[int, int], tuple[Segment, ...]],
    ) -> CombinedSegsResult:
        """Return the fixed combined result after checking call arguments."""

        self.call_count += 1
        assert torch.equal(cast(torch.Tensor, source_image), self._expected_image)
        assert source_segs is self._expected_segs
        return self._combined


class _FakeBatchCombinedBuilder:
    """Combined-result builder that derives labels from source SEGS."""

    def __init__(self) -> None:
        """Create an empty call counter."""

        self.call_count = 0

    def __call__(
        self,
        source_image: object,
        source_segs: tuple[tuple[int, int], tuple[Segment, ...]],
    ) -> CombinedSegsResult:
        """Return a combined result for one image slice."""

        self.call_count += 1
        assert cast(torch.Tensor, source_image).shape == (1, 3, 3, 3)
        segment_label = source_segs[1][0].label
        return CombinedSegsResult(
            segs=_segs(
                _segment(
                    f"combined-{segment_label}",
                    CropRegion(0, 0, 1, 1),
                    1.0,
                )
            ),
            mask=torch.ones((1, 3, 3), dtype=torch.float32),
        )


def _prompt_with_fakes(
    service: object,
    builder: object,
    image: torch.Tensor,
    combine_segs: bool,
    vitmatte_model: object | None = None,
) -> tuple[object, object]:
    """Run the prompt node with temporary service and builder doubles."""

    node = PromptSEGSWithSAM()
    original_service = PromptSEGSWithSAM._service
    original_builder = PromptSEGSWithSAM.combined_builder
    PromptSEGSWithSAM._service = service  # type: ignore[assignment]
    PromptSEGSWithSAM.combined_builder = builder  # type: ignore[assignment]
    try:
        return node.prompt(
            image=image,
            sam_model={"sam": "model"},
            grounding_dino_model={"dino": "model"},
            positive_prompt="face",
            negative_prompt="",
            confidence_threshold=0.3,
            size_threshold=10,
            bbox_dilation=0,
            mask_dilation=0,
            detail_method="GuidedFilter",
            detail_erode=6,
            detail_dilate=6,
            black_point=0.15,
            white_point=0.99,
            refine_mask=True,
            mask_refinement_max_size=2048,
            execution_device="cpu",
            crop_factor=3.0,
            sort_order="largest to smallest",
            combine_segs=combine_segs,
            vitmatte_model=vitmatte_model,
        )
    finally:
        PromptSEGSWithSAM._service = original_service
        PromptSEGSWithSAM.combined_builder = original_builder


def _segs(*segments: Segment) -> tuple[tuple[int, int], tuple[Segment, ...]]:
    """Create native SEGS for tests."""

    return (3, 3), tuple(segments)


def _segment(label: str, crop_region: CropRegion, confidence: float) -> Segment:
    """Create a test segment."""

    return Segment(
        cropped_image=None,
        cropped_mask=torch.ones((crop_region.height, crop_region.width)),
        confidence=confidence,
        crop_region=crop_region,
        bbox=BoundingBox(
            crop_region.left,
            crop_region.top,
            crop_region.right,
            crop_region.bottom,
        ),
        label=label,
    )
