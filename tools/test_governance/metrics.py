# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own stable fingerprints for reviewed test-governance facts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tools.architecture_governance.metrics import source_fingerprint

_INVENTORY_RULES = frozenset({"ISOLATED001", "LAYOUT001", "SERIAL001", "STUB001"})


def reviewed_state_fingerprint(
    root: Path,
    *,
    rule: str,
    candidates: tuple[str, ...],
    paths: tuple[str, ...],
) -> str:
    """Fingerprint the exact fact whose reviewed disposition must remain stable."""

    if rule not in _INVENTORY_RULES:
        return source_fingerprint(root, paths)
    digest = hashlib.sha256()
    for candidate in sorted(candidates):
        digest.update(candidate.encode("utf-8"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"
