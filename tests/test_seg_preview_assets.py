# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ComfyUI temporary-asset publication for SEG previews."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from pytest import MonkeyPatch

from simple_syrup.domain.segs import CropRegion
from simple_syrup.runtime.seg_preview_assets import ComfySegPreviewAssetPublisher
from simple_syrup.services.simple_preview_segs_service import (
    AtlasPlacement,
    SegPreviewDocument,
    SegPreviewRegion,
)


def test_publisher_stores_both_assets_and_builds_versioned_manifest(
    monkeypatch: MonkeyPatch,
) -> None:
    """The runtime adapter translates Comfy references without leaking tensors."""

    class _PreviewImage:
        """Return one distinct Comfy image reference per published tensor."""

        calls: ClassVar[list[torch.Tensor]] = []

        def __init__(self, image: torch.Tensor) -> None:
            self.calls.append(image)

        def as_dict(self) -> dict[str, object]:
            """Return the reference for the most recent call."""

            index = len(self.calls)
            return {
                "images": [
                    {
                        "filename": f"asset-{index}.png",
                        "subfolder": "preview",
                        "type": "temp",
                    }
                ]
            }

    fake_api = SimpleNamespace(UI=SimpleNamespace(PreviewImage=_PreviewImage))
    monkeypatch.setattr(
        "simple_syrup.runtime.seg_preview_assets.import_module",
        lambda _name: fake_api,
    )

    document = _document()
    manifest = ComfySegPreviewAssetPublisher().publish(document)

    assert _PreviewImage.calls == [
        document.image,
        document.atlas,
        *document.region_images,
    ]
    assert manifest.images == (
        {
            "filename": "asset-3.png",
            "subfolder": "preview",
            "type": "temp",
        },
    )
    assert manifest.manifest["version"] == 1
    assert manifest.manifest["preview"] == {
        "width": 4,
        "height": 3,
        "image": {
            "filename": "asset-1.png",
            "subfolder": "preview",
            "type": "temp",
        },
    }
    assert manifest.manifest["atlas"] == {
        "width": 2,
        "height": 1,
        "image": {
            "filename": "asset-2.png",
            "subfolder": "preview",
            "type": "temp",
        },
    }
    assert manifest.manifest["regions"] == [
        {
            "id": "seg-0001",
            "index": 0,
            "label": "subject",
            "confidence": 0.9,
            "area": 6,
            "color": "#f24236",
            "crop": {"x": 1, "y": 0, "width": 2, "height": 3},
            "atlas": {"x": 0, "y": 0, "width": 2, "height": 1},
        }
    ]


def test_publisher_rejects_malformed_comfy_image_reference(
    monkeypatch: MonkeyPatch,
) -> None:
    """Invalid host output fails explicitly before the frontend sees it."""

    class _InvalidPreviewImage:
        """Return a payload missing Comfy's required image reference fields."""

        def __init__(self, _image: torch.Tensor) -> None:
            pass

        def as_dict(self) -> dict[str, object]:
            """Return one intentionally incomplete reference."""

            return {"images": [{"filename": "asset.png"}]}

    fake_api = SimpleNamespace(UI=SimpleNamespace(PreviewImage=_InvalidPreviewImage))
    monkeypatch.setattr(
        "simple_syrup.runtime.seg_preview_assets.import_module",
        lambda _name: fake_api,
    )

    with pytest.raises(TypeError, match="reference is incomplete"):
        ComfySegPreviewAssetPublisher().publish(_document())


def _document() -> SegPreviewDocument:
    """Return one small document with stable transport metadata."""

    return SegPreviewDocument(
        source_width=4,
        source_height=3,
        preview_width=4,
        preview_height=3,
        image=torch.zeros((1, 3, 4, 3)),
        atlas=torch.ones((1, 1, 2, 3)),
        region_images=(torch.zeros((1, 3, 2, 4)),),
        regions=(
            SegPreviewRegion(
                region_id="seg-0001",
                index=0,
                label="subject",
                confidence=0.9,
                active_area=6,
                color="#f24236",
                crop=CropRegion(1, 0, 3, 3),
                atlas=AtlasPlacement(0, 0, 2, 1),
            ),
        ),
    )
