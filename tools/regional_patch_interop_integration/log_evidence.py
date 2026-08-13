# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Publish hashed managed-Comfy logs and focused P9.7 diagnostics."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tools.comfy_api import JsonObject


def record_regional_patch_interop_logs(root: Path) -> JsonObject:
    """Hash complete logs and write one reviewable diagnostic slice."""

    resolved = root.resolve()
    evidence: JsonObject = {}
    diagnostic_lines: list[str] = []
    for name in ("comfy.stdout.log", "comfy.stderr.log"):
        path = resolved / name
        data = path.read_bytes()
        evidence[name] = {"sha256": _sha256(data), "size_bytes": len(data)}
        text = data.decode("utf-8", errors="replace")
        diagnostic_lines.extend(
            f"[{name}] {line}"
            for line in text.splitlines()
            if any(
                token in line.lower()
                for token in (
                    "p9.7",
                    "cache",
                    "attention",
                    "regional",
                    "lora",
                    "negpip",
                    "warning",
                    "error",
                )
            )
        )
    diagnostics_path = resolved / "diagnostics.log"
    diagnostic_text = "\n".join(diagnostic_lines)
    if diagnostic_text:
        diagnostic_text += "\n"
    diagnostics_path.write_text(diagnostic_text, encoding="utf-8")
    evidence["diagnostics.log"] = {
        "sha256": _sha256(diagnostic_text.encode()),
        "line_count": len(diagnostic_lines),
    }
    return evidence


def _sha256(value: bytes) -> str:
    """Hash one complete log artifact value."""

    return hashlib.sha256(value).hexdigest()
