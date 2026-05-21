# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for grounded SAM model metadata."""

from __future__ import annotations

import json
from pathlib import Path

from simple_syrup.runtime.model_metadata import GroundedSAMModelMetadata
from test_helpers import FakeFolderPaths


def test_model_metadata_reports_source_and_expected_paths(tmp_path: Path) -> None:
    """Metadata includes catalog source URLs and local path diagnostics."""

    metadata = GroundedSAMModelMetadata(
        folder_paths_module=FakeFolderPaths(tmp_path)
    ).describe_selection(
        "sam_vit_b (375MB)",
        "GroundingDINO_SwinT_OGC (694MB)",
    )

    payload = json.loads(metadata)
    assert payload["sam"]["id"] == "sam_vit_b"
    assert payload["grounding_dino"]["id"] == "groundingdino_swint_ogc"
    assert "sam_vit_b_01ec64.pth" in metadata
    assert "groundingdino_swint_ogc.pth" in metadata
    assert "text_encoder_choices" in payload
    assert payload["vitmatte"][0]["id"] == "vitmatte-small-composition-1k"
    assert "vitmatte-base-composition-1k" in metadata
