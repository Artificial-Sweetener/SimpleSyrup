# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prevent executable repository code from acquiring machine-specific paths."""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE_ROOTS = (
    REPOSITORY_ROOT / "simple_syrup",
    REPOSITORY_ROOT / "tools",
    REPOSITORY_ROOT / "tests",
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:\\")
POSIX_MACHINE_PATH = re.compile(r"/(?:home|Users|mnt)/[^\s\"']+")


def test_executable_sources_have_no_machine_specific_path_literals() -> None:
    """Require host paths to enter through runtime configuration owners."""

    violations: list[str] = []
    for root in EXECUTABLE_ROOTS:
        for source_path in root.rglob("*.py"):
            source = source_path.read_text(encoding="utf-8")
            if WINDOWS_ABSOLUTE_PATH.search(source) or POSIX_MACHINE_PATH.search(
                source
            ):
                violations.append(str(source_path.relative_to(REPOSITORY_ROOT)))

    assert violations == []
