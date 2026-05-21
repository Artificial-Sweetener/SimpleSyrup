# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Tile & Tag SEGS service orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.segs import BoundingBox, CropRegion, NativeSegs, Segment
from simple_syrup.domain.tile_segs import TileSEGSControls
from simple_syrup.runtime.loaded_models import LoadedWD14Tagger
from simple_syrup.runtime.wd14_tagger import (
    FloatArray,
    WD14TagFormattingControls,
    WD14TagRecord,
)
from simple_syrup.services.tile_and_tag_segs_service import TileAndTagSEGSService


def test_service_preserves_segs_tag_and_conditioning_order() -> None:
    """SEGS, tile crops, tags, and conditioning stay aligned by index."""

    progress = _ProgressRecorder()
    tiler = _FakeTiler(_native_segs(("first", "second", "third")))
    tagger = _FakeTagger(("tag first", "", "tag third"))
    encoder = _FakeEncoder()
    loaded_tagger = _loaded_tagger()
    service = TileAndTagSEGSService(
        tiler=tiler,
        tagger=tagger,
        encoder=encoder,
        progress_factory=lambda _total: progress,
    )

    result = service.tile_and_tag(
        image=_image(),
        clip="clip",
        wd14_tagger=loaded_tagger,
        tile_controls=_tile_controls(),
        tag_controls=_tag_controls(),
        universal_positive="",
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
    assert encoder.chunks == ("tag first", "", "tag third")
    assert result.positive.entries == (
        "clip:tag first",
        "clip:",
        "clip:tag third",
    )
    assert progress.updates == [1, 3, 1]


def test_service_prefixes_every_tile_prompt_with_universal_positive() -> None:
    """Universal positive text is composed before conditioning encoding."""

    tiler = _FakeTiler(_native_segs(("first", "second", "third")))
    tagger = _FakeTagger(("tag first", "", "tag third"))
    encoder = _FakeEncoder()
    service = TileAndTagSEGSService(
        tiler=tiler,
        tagger=tagger,
        encoder=encoder,
    )

    result = service.tile_and_tag(
        image=_image(),
        clip="clip",
        wd14_tagger=_loaded_tagger(),
        tile_controls=_tile_controls(),
        tag_controls=_tag_controls(),
        universal_positive="masterpiece",
    )

    assert [segment.label for segment in result.segs[1]] == [
        "first",
        "second",
        "third",
    ]
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


def test_service_logs_universal_positive_presence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Completion logging records whether a universal positive was supplied."""

    service = TileAndTagSEGSService(
        tiler=_FakeTiler(_native_segs(("first",))),
        tagger=_FakeTagger(("tag first",)),
        encoder=_FakeEncoder(),
    )

    with caplog.at_level("INFO"):
        service.tile_and_tag(
            image=_image(),
            clip="clip",
            wd14_tagger=_loaded_tagger(),
            tile_controls=_tile_controls(),
            tag_controls=_tag_controls(),
            universal_positive=" masterwork ",
        )

    assert any(
        record.__dict__.get("universal_positive_present") is True
        for record in caplog.records
    )
    assert any(
        record.__dict__.get("wd14_model") == "wd-eva02-large-tagger-v3"
        for record in caplog.records
    )


def test_service_rejects_tagger_count_mismatch() -> None:
    """Dropping a tag would break SEGS alignment and is rejected."""

    service = TileAndTagSEGSService(
        tiler=_FakeTiler(_native_segs(("first", "second"))),
        tagger=_FakeTagger(("only one",)),
        encoder=_FakeEncoder(),
    )

    with pytest.raises(ValueError, match="returned 1 tag"):
        service.tile_and_tag(
            image=_image(),
            clip="clip",
            wd14_tagger=_loaded_tagger(),
            tile_controls=_tile_controls(),
            tag_controls=_tag_controls(),
            universal_positive="",
        )


def test_service_rejects_empty_segs() -> None:
    """A fully filtered image fails with an actionable error."""

    service = TileAndTagSEGSService(
        tiler=_FakeTiler(((4, 4), ())),
        tagger=_FakeTagger(()),
        encoder=_FakeEncoder(),
    )

    with pytest.raises(ValueError, match="No tile SEGS"):
        service.tile_and_tag(
            image=_image(),
            clip="clip",
            wd14_tagger=_loaded_tagger(),
            tile_controls=_tile_controls(),
            tag_controls=_tag_controls(),
            universal_positive="",
        )


def test_service_rejects_invalid_wd14_tagger() -> None:
    """Tile & Tag SEGS requires a loader-produced WD14_TAGGER object."""

    service = TileAndTagSEGSService(
        tiler=_FakeTiler(_native_segs(("first",))),
        tagger=_FakeTagger(("tag first",)),
        encoder=_FakeEncoder(),
    )

    with pytest.raises(TypeError, match="Load WD14 Tagger"):
        service.tile_and_tag(
            image=_image(),
            clip="clip",
            wd14_tagger=object(),
            tile_controls=_tile_controls(),
            tag_controls=_tag_controls(),
            universal_positive="",
        )


def test_conditioning_batch_selects_by_segment_order() -> None:
    """Detailer selection uses the same ordered batch produced by the service."""

    batch = ConditioningBatch(("first-cond", "second-cond"))

    assert batch.select(0) == "first-cond"
    assert batch.select(1) == "second-cond"
    assert batch.select(2) == "second-cond"


class _FakeTiler:
    """Return fixed native SEGS while capturing controls."""

    def __init__(self, segs: NativeSegs) -> None:
        """Store the fixed SEGS payload."""

        self.segs = segs
        self.calls: list[object] = []

    def build(
        self,
        image: torch.Tensor,
        controls: TileSEGSControls,
    ) -> NativeSegs:
        """Return the configured SEGS payload."""

        self.calls.append((image, controls))
        return self.segs


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


def _tile_controls() -> TileSEGSControls:
    """Return valid tile controls for service tests."""

    return TileSEGSControls(
        bbox_size=2,
        crop_factor=1.0,
        min_overlap=0,
        filter_segs_dilation=0,
        mask_irregularity=0.0,
        irregular_mask_mode="Reuse fast",
    )


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
