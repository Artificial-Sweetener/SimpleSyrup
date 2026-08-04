# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ordered workflow-file domain state."""

from __future__ import annotations

import pytest

from simple_syrup.domain.ordered_files import OrderedFileSelection


def test_preserves_scalar_duplicate_and_positional_state() -> None:
    """Workflow file selections are ordered multisets rather than sets."""

    scalar = OrderedFileSelection.require(
        "one.png",
        node_name="Loader",
        item_name="image",
    )
    duplicates = OrderedFileSelection.require(
        ["same.png", "middle.png", "same.png"],
        node_name="Loader",
        item_name="image",
    )

    assert scalar.paths == ("one.png",)
    assert duplicates.paths == ("same.png", "middle.png", "same.png")


def test_rejects_empty_and_malformed_workflow_values() -> None:
    """Invalid persisted values fail with workflow-facing context."""

    with pytest.raises(ValueError, match="Loader requires at least one image file"):
        OrderedFileSelection.require([], node_name="Loader", item_name="image")
    with pytest.raises(TypeError, match="image files must be non-empty strings"):
        OrderedFileSelection.require(
            ["valid.png", ""],
            node_name="Loader",
            item_name="image",
        )


def test_fingerprint_changes_with_order_duplicates_contents_and_context() -> None:
    """Every execution-relevant positional property participates in caching."""

    content = {"a.png": "aaa", "b.png": "bbb"}
    first = OrderedFileSelection(("a.png", "b.png"))
    reordered = OrderedFileSelection(("b.png", "a.png"))
    duplicate = OrderedFileSelection(("a.png", "a.png"))

    assert first.fingerprint(content.__getitem__) != reordered.fingerprint(
        content.__getitem__
    )
    assert first.fingerprint(content.__getitem__) != duplicate.fingerprint(
        content.__getitem__
    )
    assert first.fingerprint(
        content.__getitem__, context=("red",)
    ) != first.fingerprint(content.__getitem__, context=("blue",))
