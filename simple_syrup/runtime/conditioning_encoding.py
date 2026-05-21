# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Runtime adapters for ComfyUI conditioning encoding."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from ..domain.conditioning_batch import ConditioningBatch


class ComfyConditioningEncoder:
    """Encode prompt chunks with ComfyUI's normal CLIP text encoder."""

    def encode(self, clip: Any, text: str) -> Any:
        """Encode a single prompt chunk into a normal Comfy conditioning."""

        nodes = import_module("nodes")
        return cast(Any, nodes.CLIPTextEncode().encode(clip, text)[0])

    def encode_batch(
        self,
        clip: Any,
        chunks: tuple[str, ...],
    ) -> ConditioningBatch:
        """Encode ordered prompt chunks into a conditioning batch."""

        return ConditioningBatch(tuple(self.encode(clip, chunk) for chunk in chunks))
