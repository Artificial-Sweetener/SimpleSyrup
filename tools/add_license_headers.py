#!/usr/bin/env python3
# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Add or update AGPLv3 license headers in SimpleSyrup-owned source files."""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_LINE = "SimpleSyrup - workflow-focused ComfyUI extensions for image generation"
COPYRIGHT_HOLDER = "Artificial Sweetener and contributors"
LICENSE_ID = "AGPL-3.0-or-later"
START_YEAR = 2026
SUPPORTED_SUFFIXES = frozenset((".py", ".pyi", ".ts", ".js", ".mjs", ".cjs"))
EXCLUDED_PARTS = frozenset(
    (
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "__pycache__",
    )
)
EXCLUDED_PREFIXES = (
    Path("simple_syrup/third_party"),
    Path("third_party/licenses"),
    Path("web/dist"),
)
PROJECT_MARKER_PATTERN = re.compile(
    r"^(?P<prefix>#|//) SimpleSyrup - .+$", re.MULTILINE
)


def _copyright_years(now: datetime | None = None) -> str:
    """Return the canonical copyright year text for the run date."""

    current_year = (now or datetime.now(UTC)).year
    if current_year <= START_YEAR:
        return str(START_YEAR)
    return f"{START_YEAR} - {current_year}"


def _comment_prefix(path: Path) -> str:
    """Return the line comment prefix for a supported source file."""

    if path.suffix in {".js", ".mjs", ".cjs", ".ts"}:
        return "//"
    return "#"


def _header(path: Path, now: datetime | None = None) -> str:
    """Return the canonical first-party license header for a source file."""

    prefix = _comment_prefix(path)
    return "\n".join(
        (
            f"{prefix} {PROJECT_LINE}",
            f"{prefix} Copyright (C) {_copyright_years(now)}  {COPYRIGHT_HOLDER}",
            f"{prefix} SPDX-License-Identifier: {LICENSE_ID}",
        )
    )


def _tracked_source_files() -> list[Path]:
    """Return git-tracked source files that should carry the project notice."""

    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Error running git ls-files: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    return sorted(
        path
        for path_text in result.stdout.splitlines()
        if _should_update(path := Path(path_text))
    )


def _should_update(path: Path) -> bool:
    """Return whether a repository path is SimpleSyrup-owned source."""

    if path.suffix not in SUPPORTED_SUFFIXES:
        return False
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    return not any(_is_relative_to(path, prefix) for prefix in EXCLUDED_PREFIXES)


def _is_relative_to(path: Path, prefix: Path) -> bool:
    """Return whether path is inside prefix without requiring Python 3.12."""

    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def _header_bounds(lines: list[str], prefix: str) -> tuple[int, int] | None:
    """Find an existing SimpleSyrup license header block in source lines."""

    start = None
    project_pattern = f"{prefix} SimpleSyrup - "
    for index, line in enumerate(lines):
        if line.startswith(project_pattern):
            start = index
            break

    if start is None:
        return None

    expected_final_prefix = f"{prefix} SPDX-License-Identifier:"
    for index in range(start, min(start + 8, len(lines))):
        if lines[index].startswith(expected_final_prefix):
            return start, index

    return None


def _insertion_index(lines: list[str], prefix: str) -> int:
    """Return the safe insertion point after shebangs or encoding directives."""

    index = 0
    if lines and lines[0].startswith("#!"):
        index += 1

    if prefix == "#" and len(lines) > index:
        encoding_line = lines[index]
        if encoding_line.startswith("#") and "coding" in encoding_line:
            index += 1

    return index


def update_header(path: Path) -> bool:
    """Add or normalize the project license header for one source file."""

    if not _should_update(path):
        return False

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"Skipping {path}: unable to read as UTF-8")
        return False

    lines = content.splitlines(keepends=True)
    prefix = _comment_prefix(path)
    header = _header(path)
    header_lines = [line + "\n" for line in header.splitlines()]
    bounds = _header_bounds(lines, prefix)

    if bounds is not None:
        start, end = bounds
        updated_lines = lines[:start] + header_lines + lines[end + 1 :]
        updated_content = "".join(updated_lines)
        if updated_content != content:
            path.write_text(updated_content, encoding="utf-8")
            print(f"Normalized header in {path}")
            return True
        return False

    if PROJECT_MARKER_PATTERN.search(content[:1500]):
        print(f"Skipping {path}: unknown SimpleSyrup header already present")
        return False

    insert_at = _insertion_index(lines, prefix)
    new_lines = lines[:insert_at] + header_lines + ["\n"] + lines[insert_at:]
    path.write_text("".join(new_lines), encoding="utf-8")
    print(f"Added header to {path}")
    return True


def main() -> None:
    """Update license headers in all tracked SimpleSyrup-owned source files."""

    files = _tracked_source_files()
    print(f"Found {len(files)} tracked source files.")
    changed = 0
    for path in files:
        if path.exists() and update_header(path):
            changed += 1
    print(f"Updated {changed} file(s).")


if __name__ == "__main__":
    main()
