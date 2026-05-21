# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Prompt Control prompt batch node schema."""

from __future__ import annotations

from simple_syrup.nodes_v3.encode_prompt_batch_with_prompt_control import (
    EncodePromptBatchWithPromptControl,
)


def test_prompt_control_prompt_batch_node_schema() -> None:
    """The v3 node exposes the planned lazy Prompt Control contract."""

    schema = EncodePromptBatchWithPromptControl.define_schema()

    assert schema.node_id == "SimpleSyrup.EncodePromptBatchWithPromptControl"
    assert schema.display_name == "Encode Prompt Batch w/ Prompt Control"
    assert schema.enable_expand is True
    assert schema.category == "SimpleSyrup/Conditioning"
    assert [output.io_type for output in schema.outputs] == [
        "CONDITIONING_BATCH",
        "CONDITIONING_BATCH",
    ]
    assert [output.id for output in schema.outputs] == ["positive", "negative"]


def test_prompt_control_prompt_batch_input_types() -> None:
    """The finalized v3 schema exposes Comfy-compatible input and output types."""

    inputs = EncodePromptBatchWithPromptControl.INPUT_TYPES()

    assert EncodePromptBatchWithPromptControl.RETURN_TYPES == [
        "CONDITIONING_BATCH",
        "CONDITIONING_BATCH",
    ]
    assert EncodePromptBatchWithPromptControl.RETURN_NAMES == [
        "positive",
        "negative",
    ]
    assert list(inputs["required"]) == [
        "clip",
        "positive_prompt",
        "negative_prompt",
        "separator",
    ]
    assert inputs["required"]["clip"][0] == "CLIP"
    assert inputs["required"]["separator"][1]["default"] == "[SEP]"
