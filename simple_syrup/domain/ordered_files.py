# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Domain model for validated, duplicate-preserving file selections."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderedFileSelection:
    """Represent non-empty workflow file positions without deduplication."""

    paths: tuple[str, ...]

    @classmethod
    def require(
        cls,
        files: Sequence[str],
        *,
        node_name: str,
        item_name: str,
    ) -> OrderedFileSelection:
        """Validate scalar-or-sequence workflow state into ordered paths."""

        ordered = (files,) if isinstance(files, str) else tuple(files)
        if not ordered:
            raise ValueError(f"{node_name} requires at least one {item_name} file.")
        if any(not isinstance(path, str) or not path for path in ordered):
            raise TypeError(f"{node_name} {item_name} files must be non-empty strings.")
        return cls(paths=ordered)

    def fingerprint(
        self,
        file_fingerprint: Callable[[str], str],
        *,
        context: Sequence[str] = (),
    ) -> str:
        """Hash context, every path position, and each file's content digest."""

        digest = hashlib.sha256()
        for value in context:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        for path in self.paths:
            encoded = path.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(file_fingerprint(path).encode("ascii"))
        return digest.hexdigest()
