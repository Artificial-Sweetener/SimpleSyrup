# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for full-context detailer preview composition."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from PIL import Image

from simple_syrup.domain.segs import CropRegion
from simple_syrup.runtime import detail_previews
from simple_syrup.runtime.detail_previews import (
    DETAIL_PREVIEW_OUTLINE_RGB,
    DetailPreviewCompositor,
    DetailPreviewContext,
    build_detail_preview_geometry,
    fit_preview_size,
    prepare_detail_preview_callback,
    work_region_from_mask,
)


def test_fit_preview_size_preserves_small_dimensions() -> None:
    """Images already inside the cap keep their original dimensions."""

    assert fit_preview_size(100, 50, 512) == (100, 50)


def test_fit_preview_size_caps_long_side() -> None:
    """Large images are proportionally capped to the preview size."""

    assert fit_preview_size(1000, 500, 200) == (200, 100)
    assert fit_preview_size(500, 1000, 200) == (100, 200)


def test_preview_geometry_maps_crop_and_outline_boxes() -> None:
    """Preview geometry maps normal crops with an outside outline."""

    geometry = build_detail_preview_geometry(
        source_width=8,
        source_height=8,
        crop_region=CropRegion(2, 1, 6, 5),
        max_preview_resolution=8,
    )

    assert geometry.source_size == (8, 8)
    assert geometry.preview_size == (8, 8)
    assert geometry.crop_box == (2, 1, 6, 5)
    assert geometry.outline_box == (1, 0, 6, 5)


def test_preview_geometry_keeps_edge_outline_inside_image() -> None:
    """Outlines fall inside the image when a crop touches image edges."""

    geometry = build_detail_preview_geometry(
        source_width=8,
        source_height=8,
        crop_region=CropRegion(0, 0, 8, 8),
        max_preview_resolution=8,
    )

    assert geometry.crop_box == (0, 0, 8, 8)
    assert geometry.outline_box == (0, 0, 7, 7)


def test_preview_geometry_handles_downscaled_tiny_crop() -> None:
    """Heavily downscaled previews still produce valid crop boxes."""

    geometry = build_detail_preview_geometry(
        source_width=200,
        source_height=100,
        crop_region=CropRegion(50, 25, 51, 26),
        max_preview_resolution=50,
    )

    assert geometry.preview_size == (50, 25)
    assert geometry.crop_box[2] > geometry.crop_box[0]
    assert geometry.crop_box[3] > geometry.crop_box[1]
    assert geometry.outline_box[2] >= geometry.outline_box[0]
    assert geometry.outline_box[3] >= geometry.outline_box[1]


def test_compositor_darkens_background_pastes_crop_and_draws_outline() -> None:
    """Composed previews darken context, paste the crop, and draw the outline."""

    context = DetailPreviewContext(
        image=torch.ones((1, 8, 8, 3), dtype=torch.float32),
        work_region=CropRegion(2, 2, 6, 6),
        work_mask=torch.ones((4, 4), dtype=torch.float32),
    )
    compositor = DetailPreviewCompositor.from_context(
        context,
        max_preview_resolution=8,
    )

    output = compositor.compose(Image.new("RGB", (4, 4), (0, 255, 0)))

    assert output.size == (8, 8)
    assert output.getpixel((0, 0)) != (255, 255, 255)
    assert output.getpixel((3, 3)) == (0, 255, 0)
    assert output.getpixel((1, 1)) == DETAIL_PREVIEW_OUTLINE_RGB


def test_compositor_keeps_unmasked_crop_area_washed() -> None:
    """Only masked pixels inside the crop reveal the active crop preview."""

    crop_mask = torch.zeros((4, 4), dtype=torch.float32)
    crop_mask[1:3, 1:3] = 1.0
    context = DetailPreviewContext(
        image=torch.ones((1, 8, 8, 3), dtype=torch.float32),
        work_region=CropRegion(2, 2, 6, 6),
        work_mask=crop_mask,
    )
    compositor = DetailPreviewCompositor.from_context(
        context,
        max_preview_resolution=8,
    )

    output = compositor.compose(Image.new("RGB", (4, 4), (0, 255, 0)))

    assert output.getpixel((3, 3)) == (0, 255, 0)
    assert output.getpixel((2, 2)) != (0, 255, 0)
    assert output.getpixel((2, 2)) != (255, 255, 255)


def test_compositor_uses_preview_size_for_large_images() -> None:
    """Full-context composition uses preview resolution instead of source size."""

    context = DetailPreviewContext(
        image=torch.ones((1, 100, 200, 3), dtype=torch.float32),
        work_region=CropRegion(50, 25, 150, 75),
        work_mask=torch.ones((50, 100), dtype=torch.float32),
    )
    compositor = DetailPreviewCompositor.from_context(
        context,
        max_preview_resolution=50,
    )

    assert compositor.washed_background.size == (50, 25)


def test_full_image_mask_compositor_reveals_only_work_mask() -> None:
    """Full-canvas previews are revealed only through the provided work mask."""

    work_mask = torch.zeros((8, 8), dtype=torch.float32)
    work_mask[2:6, 3:7] = 1.0
    context = DetailPreviewContext(
        image=torch.ones((1, 8, 8, 3), dtype=torch.float32),
        work_region=work_region_from_mask(work_mask),
        work_mask=work_mask,
        sampled_region=CropRegion(0, 0, 8, 8),
    )
    compositor = DetailPreviewCompositor.from_context(
        context,
        max_preview_resolution=8,
    )

    output = compositor.compose(Image.new("RGB", (8, 8), (0, 255, 0)))

    assert output.getpixel((4, 3)) == (0, 255, 0)
    assert output.getpixel((0, 0)) != (0, 255, 0)
    assert output.getpixel((2, 1)) == DETAIL_PREVIEW_OUTLINE_RGB


def test_work_region_from_mask_returns_tight_bounds() -> None:
    """Work regions are the tight bounds around non-empty masks."""

    mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    mask[:, 2:6, 3:7] = 1.0

    assert work_region_from_mask(mask) == CropRegion(3, 2, 7, 6)


def test_work_region_from_empty_mask_fails() -> None:
    """Empty work masks fail before a confusing preview is produced."""

    with pytest.raises(ValueError, match="work mask"):
        work_region_from_mask(torch.zeros((8, 8), dtype=torch.float32))


def test_prepare_detail_preview_callback_sends_composed_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A previewer-backed callback sends a composed full-context image."""

    progress = _ProgressRecorder()
    monkeypatch.setattr(
        detail_previews,
        "_latent_preview",
        lambda: _LatentPreviewModule(_Previewer(), 8),
    )
    monkeypatch.setattr(
        detail_previews,
        "_comfy_utils",
        lambda: _ComfyUtilsModule(progress),
    )
    context = DetailPreviewContext(
        image=torch.ones((1, 8, 8, 3), dtype=torch.float32),
        work_region=CropRegion(2, 2, 6, 6),
        work_mask=torch.ones((4, 4), dtype=torch.float32),
    )

    callback = prepare_detail_preview_callback(_Model(), 4, context)
    callback(0, torch.zeros((1, 4, 4, 4)), torch.zeros((1, 4, 4, 4)), 4)

    assert progress.updates[0].value == 1
    assert progress.updates[0].total == 4
    preview = progress.updates[0].preview
    assert preview is not None
    assert preview[0] == "JPEG"
    assert preview[1].size == (8, 8)
    assert preview[2] == 8


def test_prepare_detail_preview_callback_without_previewer_skips_compositor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled ComfyUI previewer updates progress without composition."""

    progress = _ProgressRecorder()
    monkeypatch.setattr(
        detail_previews,
        "_latent_preview",
        lambda: _LatentPreviewModule(None, 8),
    )
    monkeypatch.setattr(
        detail_previews,
        "_comfy_utils",
        lambda: _ComfyUtilsModule(progress),
    )
    monkeypatch.setattr(
        DetailPreviewCompositor,
        "from_context",
        _raise_if_compositor_is_built,
    )
    context = DetailPreviewContext(
        image=torch.ones((1, 8, 8, 3), dtype=torch.float32),
        work_region=CropRegion(2, 2, 6, 6),
        work_mask=torch.ones((4, 4), dtype=torch.float32),
    )

    callback = prepare_detail_preview_callback(_Model(), 4, context)
    callback(1, torch.zeros((1, 4, 4, 4)), torch.zeros((1, 4, 4, 4)), 4)

    assert progress.updates == [_ProgressUpdate(2, 4, None)]


@dataclass(frozen=True)
class _ProgressUpdate:
    """Record one preview progress update."""

    value: int
    total: int
    preview: detail_previews.PreviewBytes | None


class _ProgressRecorder:
    """Record progress updates from a callback under test."""

    def __init__(self) -> None:
        """Create an empty update list."""

        self.updates: list[_ProgressUpdate] = []

    def update_absolute(
        self,
        value: int,
        total: int,
        preview: detail_previews.PreviewBytes | None = None,
    ) -> None:
        """Record one absolute progress update."""

        self.updates.append(_ProgressUpdate(value, total, preview))


class _ComfyUtilsModule:
    """Fake ComfyUI utils module for callback tests."""

    def __init__(self, progress: _ProgressRecorder) -> None:
        """Store the progress recorder returned by ProgressBar."""

        self._progress = progress

    def ProgressBar(self, steps: int) -> _ProgressRecorder:
        """Return the shared progress recorder."""

        del steps
        return self._progress


class _LatentPreviewModule:
    """Fake latent preview module for callback tests."""

    MAX_PREVIEW_RESOLUTION: int

    def __init__(self, previewer: _Previewer | None, max_resolution: int) -> None:
        """Store previewer lookup results."""

        self._previewer = previewer
        self.MAX_PREVIEW_RESOLUTION = max_resolution

    def get_previewer(
        self,
        device: object,
        latent_format: object,
    ) -> _Previewer | None:
        """Return the configured fake previewer."""

        del device, latent_format
        return self._previewer


class _Previewer:
    """Fake latent previewer returning a deterministic crop image."""

    def decode_latent_to_preview(self, x0: torch.Tensor) -> Image.Image:
        """Return a green crop preview."""

        del x0
        return Image.new("RGB", (4, 4), (0, 255, 0))


class _InnerModel:
    """Fake inner ComfyUI model carrying latent format."""

    latent_format = object()


class _Model:
    """Fake ComfyUI model carrying previewer lookup attributes."""

    load_device = object()
    model = _InnerModel()


def _raise_if_compositor_is_built(
    cls: type[DetailPreviewCompositor],
    context: DetailPreviewContext,
    max_preview_resolution: int,
) -> DetailPreviewCompositor:
    """Fail tests if disabled previews try to build composition state."""

    del cls, context, max_preview_resolution
    raise AssertionError("compositor should not be built without a previewer")
