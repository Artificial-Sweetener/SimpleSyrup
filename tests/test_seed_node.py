# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the reusable Seed ComfyUI node."""

from __future__ import annotations

from typing import Any

from simple_syrup.nodes.seed import Seed


def test_seed_node_contract_constants() -> None:
    """Seed node exposes the intended ComfyUI output contract."""

    assert Seed.RETURN_TYPES == ("INT",)
    assert Seed.RETURN_NAMES == ("seed",)
    assert Seed.FUNCTION == "execute"
    assert Seed.CATEGORY == "SimpleSyrup/Utilities"


def test_seed_node_declares_native_seed_input() -> None:
    """Seed input uses ComfyUI's native seed widget metadata."""

    input_types: dict[str, dict[str, tuple[Any, ...]]] = Seed.INPUT_TYPES()

    required = input_types["required"]
    assert tuple(required) == ("seed",)

    seed_input = required["seed"]
    assert seed_input[0] == "INT"

    metadata = seed_input[1]
    assert metadata["default"] == 0
    assert metadata["min"] == 0
    assert metadata["max"] == 0xFFFFFFFFFFFFFFFF
    assert metadata["control_after_generate"] is True


def test_seed_node_returns_seed_unchanged() -> None:
    """Execution returns the selected seed value without modification."""

    assert Seed().execute(123) == (123,)
