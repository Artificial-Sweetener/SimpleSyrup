# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Latent Diagnostics ComfyUI node."""

from __future__ import annotations

from typing import Any

import torch

from simple_syrup.nodes.latent_diagnostics import LatentDiagnostics


def test_node_contract_constants() -> None:
    """The node declares the expected ComfyUI diagnostic contract."""

    assert LatentDiagnostics.RETURN_TYPES == ("LATENT", "STRING")
    assert LatentDiagnostics.RETURN_NAMES == ("latent", "report")
    assert LatentDiagnostics.FUNCTION == "analyze"
    assert LatentDiagnostics.CATEGORY == "SimpleSyrup/Utilities"
    assert LatentDiagnostics.OUTPUT_NODE is True


def test_node_declares_latent_input() -> None:
    """The node accepts one latent input."""

    input_types: dict[str, dict[str, tuple[Any, ...]]] = LatentDiagnostics.INPUT_TYPES()

    assert tuple(input_types["required"]) == ("latent",)
    assert input_types["required"]["latent"][0] == "LATENT"


def test_analyze_returns_passthrough_latent_and_ui_report() -> None:
    """The node returns the original latent alongside a visible text report."""

    latent = {"samples": torch.zeros((1, 16, 1, 8, 8))}

    result = LatentDiagnostics().analyze(latent)

    assert result["result"][0] is latent
    assert result["result"][1] == result["ui"]["text"][0]
    assert "shape: [1, 16, 1, 8, 8]" in result["result"][1]
    assert "mixture_of_diffusers_current_compatible: yes" in result["result"][1]
