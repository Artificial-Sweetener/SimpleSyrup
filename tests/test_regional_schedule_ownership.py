# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify Prompt Control schedules retain one owner per runtime concern."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from simple_syrup.runtime.regional_attention_model_call_values import (
    uniform_model_call_sigma,
)

_REPOSITORY_ROOT = Path(__file__).parents[1]
_RUNTIME_SELECTORS = (
    "simple_syrup/domain/conditioning_schedule_selection.py",
    "simple_syrup/domain/regional_attention_selection.py",
    "simple_syrup/runtime/regional_attention_model_call.py",
    "simple_syrup/runtime/regional_lora_schedule_resolution.py",
    "simple_syrup/runtime/regional_lora/anima_schedule_wrapper.py",
)


def test_model_call_sigma_owner_handles_scalar_broadcast_and_materialized_calls() -> (
    None
):
    """Normalize Comfy call shapes once without changing uniformity validation."""

    scalar = torch.tensor(0.75)
    broadcast = scalar.expand(8)

    assert broadcast.stride() == (0,)
    assert uniform_model_call_sigma(scalar) == pytest.approx(0.75)
    assert uniform_model_call_sigma(broadcast) == pytest.approx(0.75)
    assert uniform_model_call_sigma(torch.full((4,), 0.75)) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="uniform"):
        uniform_model_call_sigma(torch.tensor([0.75, 0.5]))


def test_anima_schedule_wrapper_uses_the_shared_model_call_sigma_owner() -> None:
    """Reject restoration of a wrapper-local current-sigma implementation."""

    path = _REPOSITORY_ROOT / (
        "simple_syrup/runtime/regional_lora/anima_schedule_wrapper.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 2
        and node.module == "regional_attention_model_call_values"
        for alias in node.names
    }
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "uniform_model_call_sigma" in imported_names
    assert "_uniform_sigma" not in function_names


def test_runtime_schedule_selectors_do_not_import_prompt_control() -> None:
    """Keep Prompt Control syntax and lazy graph behavior outside selectors."""

    for relative_path in _RUNTIME_SELECTORS:
        path = _REPOSITORY_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = tuple(
            ast.unparse(node).lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
        )
        assert all("prompt_control" not in statement for statement in imports)
