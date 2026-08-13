# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify decoded-pixel acceptance for exact P9.7 modifier paths."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from tools.regional_patch_interop_integration.image_evidence import (
    validate_exact_image_equivalence,
)


def test_exact_paths_ignore_png_metadata_and_require_equal_decoded_pixels(
    tmp_path: Path,
) -> None:
    """Accept container differences only when every decoded RGB pixel is equal."""

    baseline = tmp_path / "baseline.png"
    easycache = tmp_path / "easycache.png"
    _write_image(baseline, color="red")
    _write_image(easycache, color="red", comment="different container metadata")

    evidence = validate_exact_image_equivalence(
        {
            "anima-full-baseline": baseline,
            "anima-full-easycache": easycache,
        }
    )

    assert len(evidence) == 1
    assert evidence[0].changed_pixels == 0
    assert evidence[0].maximum_channel_delta == 0


def test_exact_path_rejects_any_decoded_pixel_change(tmp_path: Path) -> None:
    """Keep visual divergence from passing structural interoperability gates."""

    baseline = tmp_path / "baseline.png"
    optimized = tmp_path / "optimized.png"
    _write_image(baseline, color="red")
    _write_image(optimized, color="blue")

    with pytest.raises(ValueError, match="changed decoded baseline pixels"):
        validate_exact_image_equivalence(
            {
                "anima-full-baseline": baseline,
                "anima-full-optimized-attention": optimized,
            }
        )


def test_unrelated_accepted_subset_requires_no_exact_comparison(tmp_path: Path) -> None:
    """Keep focused result tests independent of omitted matrix comparisons."""

    assert validate_exact_image_equivalence({"other": tmp_path / "missing.png"}) == ()


def _write_image(path: Path, *, color: str, comment: str | None = None) -> None:
    """Write one small deterministic RGB image with optional PNG metadata."""

    metadata = PngImagePlugin.PngInfo()
    if comment is not None:
        metadata.add_text("comment", comment)
    Image.new("RGB", (4, 4), color).save(path, pnginfo=metadata)
