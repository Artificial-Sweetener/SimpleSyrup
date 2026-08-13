# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test strict saved-image extraction from completed Comfy history."""

from __future__ import annotations

import pytest

from tools.comfy_api import JsonObject
from tools.comfy_integration.history_output import extract_saved_image


def _history(*, status: str = "success", images: object | None = None) -> JsonObject:
    """Build one narrow history record fixture."""

    resolved_images = (
        [{"filename": "image.png", "subfolder": "suite", "type": "output"}]
        if images is None
        else images
    )
    return {
        "status": {"completed": True, "status_str": status, "messages": []},
        "outputs": {"7": {"images": resolved_images}},
    }


def test_extracts_exactly_one_successful_saved_image() -> None:
    """Narrow the history object to the HTTP client's image value type."""

    reference = extract_saved_image(_history(), "7")

    assert reference.filename == "image.png"
    assert reference.subfolder == "suite"
    assert reference.output_type == "output"


def test_rejects_failed_prompt_history() -> None:
    """Preserve failure status rather than downloading misleading output."""

    with pytest.raises(RuntimeError, match="failed"):
        extract_saved_image(_history(status="error"), "7")

    incomplete_error = _history(status="error")
    status = incomplete_error["status"]
    assert isinstance(status, dict)
    status["completed"] = False
    with pytest.raises(RuntimeError, match="failed"):
        extract_saved_image(incomplete_error, "7")


@pytest.mark.parametrize("images", [[], [{}, {}], "image.png"])
def test_rejects_missing_or_ambiguous_saved_images(images: object) -> None:
    """Require the baseline's single expected output image."""

    with pytest.raises((TypeError, ValueError)):
        extract_saved_image(_history(images=images), "7")
