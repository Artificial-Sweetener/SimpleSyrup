# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify readable labeled U11 review-sheet composition."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.sdxl_attention_coupling_integration.visual_artifacts import (
    REVIEW_GROUPS,
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

    assert [path.name for path in sheets] == ["characters.png", "styles.png"]
    assert source.read_bytes() == source_digest
    with Image.open(sheets[0]) as sheet:
        assert sheet.width == 1920
        assert sheet.height == 794
    manifest = json.loads(
        (tmp_path / "review-sheets" / "review-manifest.json").read_text("utf-8")
    )
    assert manifest["sheets"][0]["artifact_ids"] == ["baseline--full"]
    assert manifest["sheets"][1]["artifact_ids"] == ["baseline--full"]


def test_style_review_orders_each_one_axis_control_before_its_candidate() -> None:
    """Present prompt, style-only, and multiple-adapter evidence in causal order."""

    styles = next(group for group in REVIEW_GROUPS if group.sheet_id == "styles")

    causal_order = (
        "global-style-control--full",
        "global-style-full-regional-control--full",
        "global-style-right-character-prompt-control--full",
        "global-style-regional-character--full",
        "multiple-left-style-prompt-control--full",
        "multiple-left-style-only-control--full",
        "multiple-left-scheduled-style-only--full",
        "multiple-left-midpoint-style-only--full",
        "multiple-left-minimal-window-style-only--full",
        "multiple-left--full",
    )
    assert (
        tuple(
            artifact_id
            for artifact_id in styles.artifact_ids
            if artifact_id in causal_order
        )
        == causal_order
    )


def test_schedule_mask_review_includes_current_route_completion_cases() -> None:
    """Keep every RA-06 original in the focused labeled comparison group."""

    group = next(item for item in REVIEW_GROUPS if item.sheet_id == "schedules-masks")

    assert group.artifact_ids[-3:] == (
        "ra06-independent-schedules--full",
        "ra06-soft-overlap--full",
        "ra06-uncovered-center--full",
    )
