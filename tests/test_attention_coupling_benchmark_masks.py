# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify deterministic benchmark mask materialization."""

from pathlib import Path

import pytest
from PIL import Image

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter


def test_mask_writer_preserves_normalized_rectangle_geometry(tmp_path: Path) -> None:
    """Write exact binary rectangles beneath the owned Comfy input directory."""

    manifest = load_manifest()
    case = next(
        case for case in manifest.cases if case.case_id == "overlapping-regions"
    )
    input_root = tmp_path / "input"
    writer = MaskArtifactWriter(input_root, "benchmark-masks")

    names = writer.write_case(case, width=100, height=80)

    assert names == (
        "benchmark-masks__overlapping-regions__region-00.png",
        "benchmark-masks__overlapping-regions__region-01.png",
    )
    with Image.open(input_root / names[0]) as image:
        assert image.mode == "RGB"
        assert image.size == (100, 80)
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((57, 79)) == (255, 255, 255)
        assert image.getpixel((58, 0)) == (0, 0, 0)


@pytest.mark.parametrize("filename_prefix", ["../escape", "C:/escape", "Has Spaces"])
def test_mask_writer_rejects_unsafe_prefixes(
    tmp_path: Path,
    filename_prefix: str,
) -> None:
    """Reject prefixes that could escape or collide ambiguously."""

    with pytest.raises(ValueError, match="filename-safe"):
        MaskArtifactWriter(tmp_path / "input", filename_prefix)
