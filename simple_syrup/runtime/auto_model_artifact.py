# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define trusted artifacts eligible for automatic local resolution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoModelArtifact:
    """Describe one checksum-pinned artifact and its canonical destination."""

    cache_id: str
    filename: str
    folder_name: str
    canonical_subfolder: str
    source_url: str
    source_repo: str
    description: str
    sha256: str
