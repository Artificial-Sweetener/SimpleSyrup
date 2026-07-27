# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for internal regional global-conditioning companions."""

from __future__ import annotations

import pytest

from simple_syrup.runtime.regional_conditioning_companion import (
    attach_global_companion,
    detach_global_companion,
)


def test_attach_and_detach_preserve_conditioning_without_mutation() -> None:
    """The internal companion round trip copies containers and metadata."""

    local = [["local tensor", {"hooks": "regional", "local": True}]]
    global_conditioning = [["global tensor", {"hooks": "regional", "global": True}]]

    attached = attach_global_companion(local, global_conditioning)
    detached, companion = detach_global_companion(attached)

    assert detached == local
    assert companion == global_conditioning
    assert attached is not local
    assert detached is not attached
    assert companion is not global_conditioning
    assert "simple_syrup.regional_global_companion" not in local[0][1]


def test_detach_without_companion_returns_a_clean_copy() -> None:
    """Ordinary conditioning remains ordinary after companion inspection."""

    conditioning = [["tensor", {"source": "ordinary"}]]

    detached, companion = detach_global_companion(conditioning)

    assert detached == conditioning
    assert detached is not conditioning
    assert companion is None


@pytest.mark.parametrize("value", [None, [], "conditioning"])
def test_attach_rejects_invalid_conditioning(value: object) -> None:
    """The internal node rejects empty or non-conditioning graph values."""

    with pytest.raises(TypeError, match="CONDITIONING"):
        attach_global_companion(value, [["global", {}]])
