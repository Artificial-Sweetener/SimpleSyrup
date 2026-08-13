# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Inspect repository revisions for managed P9.7 evidence."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RepositoryRevision:
    """Retain one repository's public name, HEAD, and worktree state."""

    name: str
    revision: str
    dirty: bool


def inspect_repository_revision(name: str, root: Path) -> RepositoryRevision:
    """Return exact Git HEAD and whether tracked or untracked work exists."""

    if not isinstance(name, str) or not name:
        raise ValueError("Repository evidence name must not be empty.")
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Repository evidence root does not exist: {name}.")
    revision = _git(resolved, "rev-parse", "HEAD").strip()
    if len(revision) != 40:
        raise ValueError(f"Repository {name} did not return a full Git revision.")
    dirty = bool(_git(resolved, "status", "--porcelain=v1", "--untracked-files=all"))
    return RepositoryRevision(name, revision, dirty)


def _git(root: Path, *arguments: str) -> str:
    """Run one read-only Git command through an argument list."""

    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    return completed.stdout
