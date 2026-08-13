# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify readable labeled U11 review-sheet composition."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.sdxl_attention_coupling_integration.visual_artifacts import (
    build_visual_review_sheets,
)


def test_review_sheets_use_retained_sources_and_stable_labels(tmp_path: Path) -> None:
    """Create populated groups without cropping or rewriting source images."""

    source = tmp_path / "baseline.png"
    Image.new("RGB", (1024, 1024), "magenta").save(source)
    source_digest = source.read_bytes()
    result = tmp_path / "u11-result.json"
    result.write_text(
        json.dumps(
            {
                "status": "running",
                "observations": [
                    {
                        "artifact_id": "baseline--full",
                        "label": "No LoRA baseline",
                        "image_file": source.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    sheets = build_visual_review_sheets(result)

    assert [path.name for path in sheets] == ["characters.png"]
    assert source.read_bytes() == source_digest
    with Image.open(sheets[0]) as sheet:
        assert sheet.width == 1920
        assert sheet.height == 794
    manifest = json.loads(
        (tmp_path / "review-sheets" / "review-manifest.json").read_text("utf-8")
    )
    assert manifest["sheets"][0]["artifact_ids"] == ["baseline--full"]
