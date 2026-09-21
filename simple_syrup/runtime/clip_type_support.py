# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate ComfyUI text-encoder type availability before model downloads."""

from __future__ import annotations

import importlib
from typing import Any


class ComfyClipTypeSupport:
    """Validate named ComfyUI CLIP types at the runtime boundary."""

    def require(self, clip_type_name: str) -> None:
        """Raise an actionable error when ComfyUI lacks a required CLIP type."""

        comfy_sd: Any = importlib.import_module("comfy.sd")
        if hasattr(comfy_sd.CLIPType, clip_type_name):
            return
        raise RuntimeError(
            f"This loader requires ComfyUI CLIP type '{clip_type_name}'. "
            "Update ComfyUI before using this node."
        )
