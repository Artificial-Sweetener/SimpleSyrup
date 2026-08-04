# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Simple Preview SEGS ComfyUI node."""

from __future__ import annotations

from typing import ClassVar

import torch
from pytest import MonkeyPatch

from simple_syrup.nodes.simple_preview_segs import SimplePreviewSEGS
from simple_syrup.runtime.seg_preview_assets import SegPreviewPublication
from simple_syrup.services.simple_preview_segs_service import SegPreviewDocument


def test_node_declares_terminal_passthrough_preview_contract() -> None:
    """The inspector remains insertable inline and executable as a terminal node."""

    inputs = SimplePreviewSEGS.INPUT_TYPES()["required"]

    assert tuple(inputs) == ("image", "segs")
    assert SimplePreviewSEGS.RETURN_TYPES == ("SEGS",)
    assert SimplePreviewSEGS.RETURN_NAMES == ("segs",)
    assert SimplePreviewSEGS.OUTPUT_NODE is True


def test_node_publishes_manifest_and_returns_original_segs_object(
    monkeypatch: MonkeyPatch,
) -> None:
    """Preview transport never replaces or mutates the connected SEGS payload."""

    segs: object = ((4, 5), [])
    document = SegPreviewDocument(
        source_width=5,
        source_height=4,
        preview_width=5,
        preview_height=4,
        image=torch.zeros((1, 4, 5, 3)),
        atlas=torch.zeros((1, 1, 1, 3)),
        region_images=(),
        regions=(),
    )

    class _Service:
        """Return the fixed preview document."""

        def build(self, *, image: object, segs: object) -> SegPreviewDocument:
            """Validate inputs before returning the document."""

            assert isinstance(image, torch.Tensor)
            assert segs == ((4, 5), ())
            return document

    class _Publisher:
        """Return one deterministic frontend manifest."""

        published: ClassVar[list[SegPreviewDocument]] = []

        def publish(self, value: SegPreviewDocument) -> SegPreviewPublication:
            """Record and publish the document."""

            self.published.append(value)
            return SegPreviewPublication(
                manifest={"version": 1, "regions": []},
                images=({"filename": "region.png", "subfolder": "", "type": "temp"},),
            )

    node = SimplePreviewSEGS()
    monkeypatch.setattr(SimplePreviewSEGS, "service_class", _Service)
    monkeypatch.setattr(SimplePreviewSEGS, "publisher_class", _Publisher)

    result = node.preview(image=torch.zeros((1, 4, 5, 3)), segs=segs)

    node_result = result["result"]
    assert isinstance(node_result, tuple)
    assert node_result == (segs,)
    assert node_result[0] is segs
    assert result["ui"] == {
        "images": [{"filename": "region.png", "subfolder": "", "type": "temp"}],
        "simple_syrup_segs_preview": [{"version": 1, "regions": []}],
    }
    assert _Publisher.published == [document]
