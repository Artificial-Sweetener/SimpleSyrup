# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Detect Prompt Control installation without importing its node modules."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path


@dataclass(frozen=True)
class PromptControlAvailability:
    """Describe whether Prompt Control can be resolved."""

    is_available: bool
    root_path: Path | None


def find_prompt_control_install(
    custom_nodes_root: Path | None = None,
) -> PromptControlAvailability:
    """Find Prompt Control code without importing Prompt Control nodes."""

    spec_availability = _find_prompt_control_from_python_path()
    if spec_availability.is_available:
        return spec_availability

    root = custom_nodes_root or default_custom_nodes_root()
    package_path = root / "comfyui-prompt-control" / "prompt_control"
    if (package_path / "nodes_lazy.py").is_file():
        return PromptControlAvailability(
            is_available=True,
            root_path=package_path.parent,
        )
    return PromptControlAvailability(is_available=False, root_path=None)


def prompt_control_is_available() -> bool:
    """Return whether Prompt Control can be advertised safely."""

    return find_prompt_control_install().is_available


def default_custom_nodes_root() -> Path:
    """Return the ComfyUI custom_nodes directory for this checkout."""

    return Path(__file__).resolve().parents[2].parent


def _find_prompt_control_from_python_path() -> PromptControlAvailability:
    """Inspect Python import specs without importing Prompt Control submodules."""

    spec = find_spec("prompt_control")
    if spec is None or spec.submodule_search_locations is None:
        return PromptControlAvailability(is_available=False, root_path=None)

    for location in spec.submodule_search_locations:
        package_path = Path(location)
        if (package_path / "nodes_lazy.py").is_file():
            return PromptControlAvailability(
                is_available=True,
                root_path=package_path.parent,
            )
    return PromptControlAvailability(is_available=False, root_path=None)
