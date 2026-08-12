# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Describe Comfy's model-family-neutral diffusion wrapper executor."""

from __future__ import annotations

from typing import Protocol


class DiffusionWrapperExecutor(Protocol):
    """Expose the current diffusion object and nested wrapper invocation."""

    class_obj: object

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Execute the next diffusion wrapper or installed model forward."""
