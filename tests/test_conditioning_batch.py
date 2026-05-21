# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SimpleSyrup conditioning batch domain behavior."""

from __future__ import annotations

import pytest

from simple_syrup.domain.conditioning_batch import (
    ConditioningBatch,
    select_conditioning,
    split_prompt_batch,
)


def test_split_prompt_batch_without_separator_returns_single_chunk() -> None:
    """Plain prompt text remains one prompt entry."""

    assert split_prompt_batch("cat", "[SEP]") == ("cat",)


def test_split_prompt_batch_trims_separator_whitespace() -> None:
    """Whitespace around separators does not become prompt text."""

    assert split_prompt_batch("cat [SEP] dog", "[SEP]") == ("cat", "dog")
    assert split_prompt_batch("cat[SEP]dog", "[SEP]") == ("cat", "dog")


def test_split_prompt_batch_preserves_blank_prompt_and_empty_chunks() -> None:
    """Blank prompts and trailing separator chunks remain explicit entries."""

    assert split_prompt_batch("", "[SEP]") == ("",)
    assert split_prompt_batch("cat [SEP]", "[SEP]") == ("cat", "")


def test_split_prompt_batch_rejects_empty_separator() -> None:
    """An empty separator would split between every character."""

    with pytest.raises(ValueError, match="separator must not be empty"):
        split_prompt_batch("cat", "")


def test_conditioning_batch_requires_entries() -> None:
    """A batch must contain at least one selectable entry."""

    with pytest.raises(ValueError, match="conditioning batch must contain"):
        ConditioningBatch(())


def test_conditioning_batch_selects_by_index_with_last_entry_fallback() -> None:
    """Indexes beyond the batch length reuse the final entry."""

    assert ConditioningBatch(("a",)).select(0) == "a"
    assert ConditioningBatch(("a",)).select(5) == "a"
    assert ConditioningBatch(("a", "b")).select(1) == "b"
    assert ConditioningBatch(("a", "b")).select(5) == "b"


def test_conditioning_batch_rejects_negative_indexes() -> None:
    """Negative indexes are invalid for per-SEG selection."""

    with pytest.raises(ValueError, match="conditioning batch index"):
        ConditioningBatch(("a",)).select(-1)


def test_select_conditioning_broadcasts_normal_conditioning() -> None:
    """Normal conditionings pass through unchanged for any valid index."""

    conditioning = object()

    assert select_conditioning(conditioning, 3) is conditioning


def test_select_conditioning_uses_batch_fallback() -> None:
    """Batch selection uses the same last-entry fallback policy."""

    assert select_conditioning(ConditioningBatch(("a", "b")), 5) == "b"
