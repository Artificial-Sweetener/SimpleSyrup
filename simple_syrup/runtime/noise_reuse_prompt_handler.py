# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Register capture coordination before Comfy validates or executes source sampling."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ..shared.logging import get_logger
from .noise_reuse_graph import rewrite_noise_reuse_graph

LOGGER = get_logger(__name__)
_MARKER = "_simple_syrup_noise_reuse_prompt_handler"


def register_noise_reuse_prompt_handler() -> None:
    """Register once when the native PromptServer becomes available."""
    try:
        instance = getattr(import_module("server").PromptServer, "instance", None)
    except (ImportError, AttributeError):
        return
    if instance is not None and not getattr(instance, _MARKER, False):
        instance.add_on_prompt_handler(_rewrite_prompt)
        setattr(instance, _MARKER, True)


def prepare_cube_prompt(prompt: dict[str, dict[str, object]]) -> None:
    """Reuse the authoritative capture planner on the final lowered Cube graph."""
    rewrite_noise_reuse_graph(prompt, import_module("nodes").NODE_CLASS_MAPPINGS)


def _rewrite_prompt(data: Any) -> Any:
    """Use normal error-producing nodes for invalid source provenance."""
    if isinstance(data, dict) and isinstance(data.get("prompt"), dict):
        rewrite_noise_reuse_graph(
            data["prompt"], import_module("nodes").NODE_CLASS_MAPPINGS
        )
    return data
