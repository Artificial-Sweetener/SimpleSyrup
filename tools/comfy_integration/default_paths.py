# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive portable local paths for managed Comfy development tools."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

COMFY_ROOT_ENVIRONMENT_VARIABLE = "SIMPLE_SYRUP_COMFY_ROOT"


def default_comfy_root() -> Path:
    """Return the explicit or installation-derived Comfy application root."""

    configured = os.environ.get(COMFY_ROOT_ENVIRONMENT_VARIABLE)
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError(
                f"{COMFY_ROOT_ENVIRONMENT_VARIABLE} must be an absolute path."
            )
        return root.resolve()
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "main.py").is_file() and (candidate / "comfy").is_dir():
            return candidate
    raise RuntimeError(
        "Cannot locate the Comfy application root; set "
        f"{COMFY_ROOT_ENVIRONMENT_VARIABLE}."
    )


def default_benchmark_artifact_root(relative_path: str = "") -> Path:
    """Return a validated artifact directory beneath the active Comfy root."""

    return default_comfy_root() / "benchmark_artifacts" / _relative_parts(relative_path)


def default_input_root() -> Path:
    """Return the active Comfy input directory."""

    return default_comfy_root() / "input"


def default_custom_node_root(relative_path: str) -> Path:
    """Return a validated sibling custom-node path beneath the Comfy root."""

    return default_comfy_root() / "custom_nodes" / _relative_parts(relative_path)


def _relative_parts(relative_path: str) -> Path:
    """Normalize a portable relative path and reject root escape."""

    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    if normalized.is_absolute() or any(part == ".." for part in normalized.parts):
        raise ValueError("Managed Comfy path suffixes must stay relative.")
    return Path(*normalized.parts)
