# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for WD14 tagging of existing SEGS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.runtime.loaded_models import LoadedWD14Tagger
from simple_syrup.runtime.wd14_tagger import (
    FloatArray,
    WD14TagFormattingControls,
    WD14TagRecord,
)
from simple_syrup.services.tag_segs_with_wd14_service import TagSEGSWithWD14Service


def test_service_preserves_existing_segs_tag_and_conditioning_order() -> None:
    """Existing SEGS, crops, tags, and conditioning stay aligned by index."""

    progress = _ProgressRecorder()
    tagger = _FakeTagger(("tag first", "", "tag third"))
    encoder = _FakeEncoder()
    loaded_tagger = _loaded_tagger()
    service = TagSEGSWithWD14Service(
        tagger=tagger,
        encoder=encoder,
        progress_factory=lambda _total: progress,
    )
    segs = _native_segs(("first", "second", "third"))

    result = service.tag(
        image=_image(),
        segs=segs,
        clip="clip",
        wd14_tagger=loaded_tagger,
        tag_controls=_tag_controls(),
        universal_positive="masterpiece",
    )

    assert [segment.label for segment in result.segs[1]] == [
        "first",
        "second",
        "third",
    ]
    assert [tuple(crop.shape) for crop in tagger.crops] == [
        (1, 2, 2, 3),
        (1, 2, 2, 3),
        (1, 2, 2, 3),
    ]
    assert tagger.loaded_tagger is loaded_tagger
    assert encoder.chunks == (
        "masterpiece, tag first",
        "masterpiece",
        "masterpiece, tag third",
    )
    assert result.positive.entries == (
        "clip:masterpiece, tag first",
        "clip:masterpiece",
        "clip:masterpiece, tag third",
    )
    assert progress.updates == [1, 3, 1]


def test_service_rejects_empty_segs() -> None:
    """Tagging empty SEGS would not produce a selectable conditioning batch."""

    service = TagSEGSWithWD14Service(
        tagger=_FakeTagger(()),
        encoder=_FakeEncoder(),
    )

    with pytest.raises(ValueError, match="No SEGS"):
        service.tag(
            image=_image(),
            segs=((4, 4), ()),
            clip="clip",
            wd14_tagger=_loaded_tagger(),
            tag_controls=_tag_controls(),
            universal_positive="",
        )


def test_service_rejects_segs_image_header_mismatch() -> None:
    """SEGS must describe the image being cropped for tagging."""

    service = TagSEGSWithWD14Service(
        tagger=_FakeTagger(("tag",)),
        encoder=_FakeEncoder(),
    )

    with pytest.raises(ValueError, match="SEGS is 8x4, image is 4x4"):
        service.tag(
            image=_image(),
            segs=((8, 4), _native_segs(("first",))[1]),
            clip="clip",
            wd14_tagger=_loaded_tagger(),
            tag_controls=_tag_controls(),
            universal_positive="",
        )


def test_service_rejects_tagger_count_mismatch() -> None:
    """Dropping a tag would break SEGS alignment and is rejected."""

    service = TagSEGSWithWD14Service(
        tagger=_FakeTagger(("only one",)),
        encoder=_FakeEncoder(),
    )

    with pytest.raises(ValueError, match="returned 1 tag"):
        service.tag(
            image=_image(),
            segs=_native_segs(("first", "second")),
            clip="clip",
            wd14_tagger=_loaded_tagger(),
            tag_controls=_tag_controls(),
            universal_positive="",
        )


def test_service_rejects_conditioning_count_mismatch() -> None:
    """Dropping encoded conditioning would break SEGS alignment and is rejected."""

    service = TagSEGSWithWD14Service(
        tagger=_FakeTagger(("first", "second")),
        encoder=_ShortEncoder(),
    )

    with pytest.raises(ValueError, match="returned 1 entries for 2 SEGS"):
        service.tag(
            image=_image(),
            segs=_native_segs(("first", "second")),
            clip="clip",
            wd14_tagger=_loaded_tagger(),
            tag_controls=_tag_controls(),
            universal_positive="",
        )


class _FakeTagger:
    """Return fixed tag strings for ordered crops."""

    def __init__(self, tags: tuple[str, ...]) -> None:
        """Store the fixed tags."""

        self.tags = tags
        self.crops: tuple[torch.Tensor, ...] = ()
        self.loaded_tagger: LoadedWD14Tagger | None = None

    def tag_images(
        self,
        loaded_tagger: LoadedWD14Tagger,
        images: tuple[torch.Tensor, ...],
        controls: WD14TagFormattingControls,
        progress: object | None = None,
    ) -> tuple[str, ...]:
        """Return fixed tags and remember the crop order."""

        _ = controls
        if progress is not None:
            progress.update(len(images))  # type: ignore[attr-defined]
        self.loaded_tagger = loaded_tagger
        self.crops = images
        return self.tags


class _FakeEncoder:
    """Return visible conditioning values for prompt chunks."""

    def __init__(self) -> None:
        """Initialize captured chunks."""

        self.chunks: tuple[str, ...] = ()

    def encode_batch(self, clip: Any, chunks: tuple[str, ...]) -> ConditioningBatch:
        """Encode prompts as simple strings."""

        self.chunks = chunks
        return ConditioningBatch(tuple(f"{clip}:{chunk}" for chunk in chunks))


class _ShortEncoder:
    """Return too few conditioning entries for validation tests."""

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


def _native_segs(labels: tuple[str, ...]) -> NativeSegs:
    """Create native SEGS with stable two-pixel crop regions."""

    segments = tuple(
        Segment(
            cropped_image=None,
            cropped_mask=torch.ones((2, 2)),
            confidence=1.0,
            crop_region=CropRegion(index, index, index + 2, index + 2),
            bbox=BoundingBox(index, index, index + 2, index + 2),
            label=label,
        )
        for index, label in enumerate(labels)
    )
    return (4, 4), segments


def _image() -> torch.Tensor:
    """Return a small deterministic BHWC image."""

    return torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(1, 4, 4, 3) / 255.0


def _tag_controls() -> WD14TagFormattingControls:
    """Return valid WD14 controls for service tests."""

    return WD14TagFormattingControls(
        threshold=0.35,
        character_threshold=1.0,
        replace_underscore=True,
        trailing_comma=False,
        exclude_tags="",
    )


def _loaded_tagger() -> LoadedWD14Tagger:
    """Return a reusable loaded WD14 tagger test container."""

    return LoadedWD14Tagger(
        model_id="wd-eva02-large-tagger-v3",
        source="test",
        onnx_path=Path("wd-eva02-large-tagger-v3.onnx"),
        csv_path=Path("wd-eva02-large-tagger-v3.csv"),
        providers=("CPUExecutionProvider",),
        session=_FakeWD14Session(),
        tags=(WD14TagRecord("blue_hair", "0"),),
    )


class _FakeWD14Session:
    """Minimal WD14 session test double."""

    def get_inputs(self) -> list[object]:
        """Return no fake inputs."""

        return []

    def get_outputs(self) -> list[object]:
        """Return no fake outputs."""

        return []

    def run(
        self, output_names: list[str], feeds: dict[str, FloatArray]
    ) -> list[object]:
        """Return no fake outputs."""

        _ = output_names, feeds
        return []
