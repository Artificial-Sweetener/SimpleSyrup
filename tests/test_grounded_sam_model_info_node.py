# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Grounded SAM Model Info node declaration."""

from __future__ import annotations

from typing import Any

from simple_syrup.nodes.grounded_sam_model_info import GroundedSAMModelInfo


def test_model_info_node_contract_constants() -> None:
    """Model info node constants match the public ComfyUI contract."""

    assert GroundedSAMModelInfo.RETURN_TYPES == ("STRING",)
    assert GroundedSAMModelInfo.RETURN_NAMES == ("model_info",)
    assert GroundedSAMModelInfo.FUNCTION == "describe"
    assert GroundedSAMModelInfo.CATEGORY == "SimpleSyrup/Masking"


def test_model_info_node_declares_expected_inputs() -> None:
    """Model info node exposes model selectors."""

    input_types: dict[str, dict[str, tuple[Any, ...]]] = (
        GroundedSAMModelInfo.INPUT_TYPES()
    )
    required = input_types["required"]

    assert set(required) == {"sam_model", "grounding_dino_model"}
    assert "sam_hq_vit_b (379MB)" in required["sam_model"][0]
    assert "GroundingDINO_SwinT_OGC (694MB)" in required["grounding_dino_model"][0]


def test_model_info_node_delegates_to_metadata_provider() -> None:
    """Node execution delegates metadata creation to its metadata provider."""

    class FakeMetadata:
        """Metadata double for model info."""

        def describe_selection(self, sam_model: str, grounding_dino_model: str) -> str:
            """Return deterministic metadata."""

            return f"{sam_model}|{grounding_dino_model}"

    node = GroundedSAMModelInfo()
    original = GroundedSAMModelInfo._metadata
    GroundedSAMModelInfo._metadata = FakeMetadata()  # type: ignore[assignment]
    try:
        result = node.describe("sam", "dino")
    finally:
        GroundedSAMModelInfo._metadata = original

    assert result == ("sam|dino",)
