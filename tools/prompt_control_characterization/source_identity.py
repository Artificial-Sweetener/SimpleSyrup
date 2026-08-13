# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the immutable installed Prompt Control source identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PINNED_VERSION = "3.0.0-beta.3"
PINNED_TRACKED_PATH_COUNT = 53
PINNED_TRACKED_TREE_SHA256 = (
    "eb45f34e976b01f3c4a98318c062ef3d54bfbbfef3e4e76ad8fd6f73b0f1204d"
)


@dataclass(frozen=True)
class PromptControlSourceIdentity:
    """Describe one validated Prompt Control installation without its path."""

    version: str
    tracked_path_count: int
    tracked_tree_sha256: str


def inspect_source(root: Path) -> PromptControlSourceIdentity:
    """Hash sorted tracked paths and bytes using the plan's canonical framing."""

    resolved = root.resolve()
    tracking = resolved / ".tracking"
    pyproject = resolved / "pyproject.toml"
    if not tracking.is_file() or not pyproject.is_file():
        raise ValueError(
            "Prompt Control root must contain .tracking and pyproject.toml."
        )
    version = _project_version(pyproject.read_text(encoding="utf-8"))
    paths = tuple(
        sorted(
            (
                line.strip().replace("\\", "/")
                for line in tracking.read_text(encoding="utf-8").splitlines()
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if not paths or any(not path for path in paths):
        raise ValueError("Prompt Control .tracking must contain nonempty paths.")
    digest = hashlib.sha256()
    for relative_path in paths:
        candidate = (resolved / Path(relative_path)).resolve()
        if not candidate.is_relative_to(resolved) or not candidate.is_file():
            raise ValueError(f"Invalid Prompt Control tracked path: {relative_path!r}.")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return PromptControlSourceIdentity(version, len(paths), digest.hexdigest())


def validate_pinned_source(identity: PromptControlSourceIdentity) -> None:
    """Reject any Prompt Control source other than the investigated install."""

    expected = PromptControlSourceIdentity(
        PINNED_VERSION,
        PINNED_TRACKED_PATH_COUNT,
        PINNED_TRACKED_TREE_SHA256,
    )
    if identity != expected:
        raise ValueError(
            "Prompt Control source does not match the pinned characterization identity."
        )


def _project_version(pyproject: str) -> str:
    """Read the sole project version assignment without a TOML dependency."""

    versions = []
    in_project = False
    for line in pyproject.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
        elif in_project and stripped.startswith("version"):
            _, separator, value = stripped.partition("=")
            if not separator:
                raise ValueError("Prompt Control project version is malformed.")
            versions.append(value.strip().strip('"'))
    if len(versions) != 1 or not versions[0]:
        raise ValueError("Prompt Control pyproject must define one project version.")
    return versions[0]
