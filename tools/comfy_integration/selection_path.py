# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parse Comfy selection names independently of the host path syntax."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def comfy_selection_path(value: str, *, minimum_parts: int = 1) -> Path:
    """Return one safe relative selection path using either separator style."""

    if minimum_parts < 1:
        raise ValueError("Minimum Comfy selection depth must be positive.")
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value.replace("\\", "/"))
    parts = posix_path.parts
    if (
        not value
        or windows_path.drive
        or windows_path.is_absolute()
        or posix_path.is_absolute()
        or len(parts) < minimum_parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("Comfy selection must be a safe relative path.")
    return Path(*parts)
