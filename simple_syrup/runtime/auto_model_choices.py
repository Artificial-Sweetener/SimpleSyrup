# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build component choices without duplicating automatic local artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from .auto_model_artifact import AutoModelArtifact


def automatic_component_choices(
    installed: Sequence[str],
    artifacts: Sequence[AutoModelArtifact],
    leading_choices: Sequence[str],
    folder_paths_module: ModuleType,
) -> list[str]:
    """Return leading choices plus local files not represented automatically."""

    if not artifacts:
        raise ValueError("Automatic component choices require at least one artifact.")
    folder_names = {artifact.folder_name for artifact in artifacts}
    if len(folder_names) != 1:
        raise ValueError("Automatic component artifacts must share one model category.")

    folder_name = next(iter(folder_names))
    automatic_names = {artifact.filename for artifact in artifacts}
    automatic_sizes = {artifact.file_size_bytes for artifact in artifacts}
    choices = list(dict.fromkeys(leading_choices))
    seen = set(choices)
    for choice in installed:
        if choice in seen or choice in automatic_names:
            continue
        if _installed_file_has_known_size(
            folder_paths_module,
            folder_name,
            choice,
            automatic_sizes,
        ):
            continue
        choices.append(choice)
        seen.add(choice)
    return choices


def _installed_file_has_known_size(
    folder_paths_module: ModuleType,
    folder_name: str,
    choice: str,
    automatic_sizes: set[int],
) -> bool:
    """Identify a likely automatic artifact without hashing during schema creation."""

    get_full_path: Any = getattr(folder_paths_module, "get_full_path", None)
    if not callable(get_full_path):
        return False
    path_value: Any = get_full_path(folder_name, choice)
    if path_value is None:
        return False
    try:
        path = Path(str(path_value))
        return path.is_file() and path.stat().st_size in automatic_sizes
    except OSError:
        return False


__all__ = ["automatic_component_choices"]
