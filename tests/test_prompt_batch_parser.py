# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test prompt-batch separator parsing behavior."""

from __future__ import annotations

import pytest

from simple_syrup.domain.prompt_batch_parser import split_prompt_batch


def test_split_prompt_batch_without_separator_returns_single_chunk() -> None:
    """Keep plain prompt text as one prompt entry."""

    assert split_prompt_batch("cat", "[SEP]") == ("cat",)


def test_split_prompt_batch_trims_separator_whitespace() -> None:
    """Exclude whitespace surrounding separators from prompt text."""

    assert split_prompt_batch("cat [SEP] dog", "[SEP]") == ("cat", "dog")
    assert split_prompt_batch("cat[SEP]dog", "[SEP]") == ("cat", "dog")


def test_split_prompt_batch_preserves_blank_prompt_and_empty_chunks() -> None:
    """Keep blank prompts and trailing separator chunks as explicit entries."""

    assert split_prompt_batch("", "[SEP]") == ("",)
    assert split_prompt_batch("cat [SEP]", "[SEP]") == ("cat", "")


def test_split_prompt_batch_treats_custom_separator_as_literal_text() -> None:
    """Escape regex syntax in a configured custom separator."""

    assert split_prompt_batch("cat .+ dog", ".+") == ("cat", "dog")


def test_split_prompt_batch_accepts_named_default_separators() -> None:
    """Discard organizational names while preserving prompt order."""

    assert split_prompt_batch(
        "global [SEP|Sky] clouds [SEP] field [SEP|Subject 2] person",
        "[SEP]",
    ) == ("global", "clouds", "field", "person")


def test_split_prompt_batch_preserves_empty_chunk_after_named_separator() -> None:
    """Keep a trailing named separator as an explicit blank prompt entry."""

    assert split_prompt_batch("cat [SEP|Unused region]", "[SEP]") == ("cat", "")


@pytest.mark.parametrize(
    "marker",
    (
        "[SEP|]",
        "[SEP|Sky",
        "[SEP|Sky\nClouds]",
        "[sep|Sky]",
    ),
)
def test_split_prompt_batch_keeps_malformed_or_wrong_case_markers(
    marker: str,
) -> None:
    """Leave text untouched when it is not a valid default separator marker."""

    prompt = f"cat {marker} dog"

    assert split_prompt_batch(prompt, "[SEP]") == (prompt,)


def test_split_prompt_batch_does_not_extend_custom_separator_grammar() -> None:
    """Recognize named markers only when the configured separator is `[SEP]`."""

    prompt = "cat [SEP|Sky] dog"

    assert split_prompt_batch(prompt, "---") == (prompt,)


def test_split_prompt_batch_rejects_empty_separator() -> None:
    """Reject a separator that would split between every character."""

    with pytest.raises(ValueError, match="separator must not be empty"):
        split_prompt_batch("cat", "")
