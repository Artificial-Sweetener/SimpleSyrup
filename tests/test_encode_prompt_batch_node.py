# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for standard prompt batch encoding node."""

from __future__ import annotations

from typing import Any

import pytest

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.nodes.encode_prompt_batch import EncodePromptBatch


def test_encode_prompt_batch_contract() -> None:
    """Prompt batch node exposes the planned standard encoder shape."""

    inputs = EncodePromptBatch.INPUT_TYPES()

    assert EncodePromptBatch.RETURN_TYPES == (
        "CONDITIONING_BATCH",
        "CONDITIONING_BATCH",
    )
    assert EncodePromptBatch.RETURN_NAMES == ("positive", "negative")
    assert EncodePromptBatch.CATEGORY == "SimpleSyrup/Conditioning"
    assert list(inputs["required"]) == [
        "clip",
        "positive_prompt",
        "negative_prompt",
        "separator",
    ]
    assert inputs["required"]["clip"][0] == "CLIP"
    assert inputs["required"]["positive_prompt"][1]["default"] == ""
    assert inputs["required"]["negative_prompt"][1]["default"] == ""
    assert inputs["required"]["separator"][1]["default"] == "[SEP]"


def test_encode_prompt_batch_splits_and_encodes_each_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standard encoder batches positive and negative chunks independently."""

    monkeypatch.setattr(EncodePromptBatch, "encoder_class", _FakeEncoder)

    positive, negative = EncodePromptBatch().encode(
        clip="clip",
        positive_prompt="face [SEP] hair",
        negative_prompt="blur",
        separator="[SEP]",
    )

    assert isinstance(positive, ConditioningBatch)
    assert isinstance(negative, ConditioningBatch)
    assert positive.entries == ("clip:face", "clip:hair")
    assert negative.entries == ("clip:blur",)


def test_encode_prompt_batch_encodes_blank_and_empty_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank prompts and trailing separator chunks still become entries."""

    monkeypatch.setattr(EncodePromptBatch, "encoder_class", _FakeEncoder)

    positive, negative = EncodePromptBatch().encode(
        clip="clip",
        positive_prompt="",
        negative_prompt="bad [SEP]",
        separator="[SEP]",
    )

    assert isinstance(positive, ConditioningBatch)
    assert isinstance(negative, ConditioningBatch)
    assert positive.entries == ("clip:",)
    assert negative.entries == ("clip:bad", "clip:")


class _FakeEncoder:
    """Fake prompt encoder for node tests."""

    def encode_batch(
        self,
        clip: Any,
        chunks: tuple[str, ...],
    ) -> ConditioningBatch:
        """Return visible conditioning entries for assertions."""

        return ConditioningBatch(tuple(f"{clip}:{chunk}" for chunk in chunks))
