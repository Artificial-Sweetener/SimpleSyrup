# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for external-LLM tagging of existing SEGS."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.services.tag_segs_with_external_llm_service import (
    LLMTagFormattingControls,
    TagSEGSWithExternalLLMService,
)

DEFAULT_CROP_REGION = CropRegion(1, 1, 3, 3)


def test_service_preserves_segs_llm_prompt_and_conditioning_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Existing SEGS, LLM calls, formatted prompts, and conditioning stay aligned."""

    progress = _ProgressRecorder()
    llm = _FakeLLM(("blue_hair, smile, bad_tag", "green_eyes"))
    image_encoder = _FakeImageEncoder()
    conditioning_encoder = _FakeConditioningEncoder()
    service = TagSEGSWithExternalLLMService(
        llm=llm,
        image_encoder=image_encoder,
        conditioning_encoder=conditioning_encoder,
        progress_factory=lambda _total: progress,
    )
    segs = _native_segs(("first", "second"))

    with caplog.at_level(logging.INFO, logger="simple_syrup"):
        result = service.tag(
            image=_image(),
            segs=segs,
            clip="clip",
            model="vision-model",
            system_prompt="system",
            user_prompt="user",
            universal_positive="masterpiece",
            seg_image_mode="transparent mask",
            formatting=LLMTagFormattingControls(
                replace_underscore=True,
                trailing_comma=True,
                exclude_tags="bad tag",
            ),
            max_tokens=64,
            reasoning_effort="off",
        )

    assert [segment.label for segment in result.segs[1]] == ["first", "second"]
    assert image_encoder.calls == (
        ("first", "transparent mask"),
        ("second", "transparent mask"),
    )
    assert [call["image_data_url"] for call in llm.calls] == [
        "data:image/png;base64,first",
        "data:image/png;base64,second",
    ]
    assert [call["model"] for call in llm.calls] == ["vision-model", "vision-model"]
    assert [call["max_tokens"] for call in llm.calls] == [64, 64]
    assert [call["reasoning_effort"] for call in llm.calls] == ["off", "off"]
    assert conditioning_encoder.chunks == (
        "masterpiece, blue hair, smile,",
        "masterpiece, green eyes,",
    )
    assert result.positive.entries == (
        "clip:masterpiece, blue hair, smile,",
        "clip:masterpiece, green eyes,",
    )
    assert progress.updates == [1, 1, 1, 1]
    record = caplog.records[-1]
    assert record.__dict__["operation"] == "tag_segs_with_external_llm"
    assert record.__dict__["segment_count"] == 2
    assert record.__dict__["external_llm_model"] == "vision-model"
    assert record.__dict__["seg_image_mode"] == "transparent mask"
    assert record.__dict__["universal_positive_present"] is True


def test_service_preserves_underscores_when_requested() -> None:
    """Formatting can keep booru-style underscores."""

    conditioning_encoder = _FakeConditioningEncoder()
    service = _service(
        llm=_FakeLLM(("blue_hair, smile",)),
        conditioning_encoder=conditioning_encoder,
    )

    service.tag(
        image=_image(),
        segs=_native_segs(("first",)),
        clip="clip",
        model="vision-model",
        system_prompt="system",
        user_prompt="user",
        universal_positive="",
        seg_image_mode="black mask",
        formatting=LLMTagFormattingControls(replace_underscore=False),
    )

    assert conditioning_encoder.chunks == ("blue_hair, smile",)


def test_service_rejects_empty_segs() -> None:
    """External LLM tagging needs at least one SEG to tag."""

    with pytest.raises(ValueError, match="No SEGS"):
        _service().tag(
            image=_image(),
            segs=((4, 4), ()),
            clip="clip",
            model="vision-model",
            system_prompt="system",
            user_prompt="user",
            universal_positive="",
            seg_image_mode="transparent mask",
            formatting=LLMTagFormattingControls(),
        )


def test_service_rejects_segs_image_header_mismatch() -> None:
    """SEGS must describe the source image dimensions."""

    with pytest.raises(ValueError, match="SEGS is 8x4, image is 4x4"):
        _service().tag(
            image=_image(),
            segs=((8, 4), _native_segs(("first",))[1]),
            clip="clip",
            model="vision-model",
            system_prompt="system",
            user_prompt="user",
            universal_positive="",
            seg_image_mode="transparent mask",
            formatting=LLMTagFormattingControls(),
        )


def test_service_rejects_crop_regions_outside_image() -> None:
    """SEG crop regions must fit inside the connected source image."""

    with pytest.raises(ValueError, match="crop_region must fit"):
        _service().tag(
            image=_image(),
            segs=_native_segs(("outside",), crop_region=CropRegion(3, 3, 5, 5)),
            clip="clip",
            model="vision-model",
            system_prompt="system",
            user_prompt="user",
            universal_positive="",
            seg_image_mode="transparent mask",
            formatting=LLMTagFormattingControls(),
        )


def test_service_rejects_empty_llm_responses() -> None:
    """Empty provider responses cannot produce usable regional prompts."""

    with pytest.raises(ValueError, match="empty response"):
        _service(llm=_FakeLLM((" ",))).tag(
            image=_image(),
            segs=_native_segs(("first",)),
            clip="clip",
            model="vision-model",
            system_prompt="system",
            user_prompt="user",
            universal_positive="",
            seg_image_mode="transparent mask",
            formatting=LLMTagFormattingControls(),
        )


def test_service_rejects_responses_removed_by_exclusions() -> None:
    """Exclusion filtering must not silently create blank prompts."""

    with pytest.raises(ValueError, match="no usable tags"):
        _service(llm=_FakeLLM(("bad_tag",))).tag(
            image=_image(),
            segs=_native_segs(("first",)),
            clip="clip",
            model="vision-model",
            system_prompt="system",
            user_prompt="user",
            universal_positive="",
            seg_image_mode="transparent mask",
            formatting=LLMTagFormattingControls(exclude_tags="bad tag"),
        )


def test_service_rejects_conditioning_count_mismatch() -> None:
    """Conditioning output count must stay aligned to SEGS count."""

    with pytest.raises(ValueError, match="returned 1 entries for 2 SEGS"):
        _service(
            llm=_FakeLLM(("first", "second")),
            conditioning_encoder=_ShortConditioningEncoder(),
        ).tag(
            image=_image(),
            segs=_native_segs(("first", "second")),
            clip="clip",
            model="vision-model",
            system_prompt="system",
            user_prompt="user",
            universal_positive="",
            seg_image_mode="transparent mask",
            formatting=LLMTagFormattingControls(),
        )


class _FakeLLM:
    """Return ordered external LLM responses."""

    def __init__(self, responses: tuple[str, ...] = ("tag",)) -> None:
        """Store fixed responses."""

        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def model_choices(self) -> list[str]:
        """Return deterministic model choices."""

        return ["vision-model"]

    def generate_with_image_data_url(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        reasoning_effort: str = "default",
        image_data_url: str | None = None,
    ) -> str:
        """Capture one LLM call and return its configured response."""

        self.calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "image_data_url": image_data_url,
            }
        )
        return self.responses[len(self.calls) - 1]


class _FakeImageEncoder:
    """Return visible data URLs for SEG crops."""

    def __init__(self) -> None:
        """Initialize captured calls."""

        self.calls: tuple[tuple[str, str], ...] = ()

    def encode_segment_as_data_url(
        self,
        image: torch.Tensor,
        segment: Segment,
        mode: str,
    ) -> str:
        """Record one SEG image encoding call."""

        assert tuple(image.shape) == (1, 4, 4, 3)
        self.calls = (*self.calls, (segment.label, mode))
        return f"data:image/png;base64,{segment.label}"


class _FakeConditioningEncoder:
    """Return visible conditioning entries for prompt chunks."""

    def __init__(self) -> None:
        """Initialize captured chunks."""

        self.chunks: tuple[str, ...] = ()

    def encode_batch(self, clip: Any, chunks: tuple[str, ...]) -> ConditioningBatch:
        """Encode prompts as simple strings."""

        self.chunks = chunks
        return ConditioningBatch(tuple(f"{clip}:{chunk}" for chunk in chunks))


class _ShortConditioningEncoder:
    """Return too few conditioning entries."""

    def encode_batch(self, clip: Any, chunks: tuple[str, ...]) -> ConditioningBatch:
        """Encode only the first prompt chunk."""

        _ = clip
        return ConditioningBatch((chunks[0],))


class _ProgressRecorder:
    """Record service progress updates."""

    def __init__(self) -> None:
        """Initialize captured update values."""

        self.updates: list[int] = []

    def update(self, value: int) -> None:
        """Record one progress advance."""

        self.updates.append(value)


def _service(
    llm: _FakeLLM | None = None,
    conditioning_encoder: _FakeConditioningEncoder
    | _ShortConditioningEncoder
    | None = None,
) -> TagSEGSWithExternalLLMService:
    """Create a service with fake boundaries."""

    return TagSEGSWithExternalLLMService(
        llm=llm or _FakeLLM(),
        image_encoder=_FakeImageEncoder(),
        conditioning_encoder=conditioning_encoder or _FakeConditioningEncoder(),
    )


def _native_segs(
    labels: tuple[str, ...],
    crop_region: CropRegion = DEFAULT_CROP_REGION,
) -> NativeSegs:
    """Create native SEGS with stable crop regions."""

    segments = tuple(
        Segment(
            cropped_image=None,
            cropped_mask=torch.ones((crop_region.height, crop_region.width)),
            confidence=1.0,
            crop_region=crop_region,
            bbox=BoundingBox(
                crop_region.left,
                crop_region.top,
                crop_region.right,
                crop_region.bottom,
            ),
            label=label,
        )
        for label in labels
    )
    return (4, 4), segments


def _image() -> torch.Tensor:
    """Return a small deterministic BHWC image."""

    return torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(1, 4, 4, 3) / 255.0
