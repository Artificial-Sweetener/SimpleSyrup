# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepare captured-noise dependencies for direct Cube queue execution."""

from __future__ import annotations

from ..runtime.noise_reuse_prompt_handler import prepare_cube_prompt


class NoiseReuseGraphPreparationService:
    """Share the capture graph authority with queues outside Comfy's prompt hook."""

    def prepare(self, prompt: dict[str, dict[str, object]]) -> None:
        """Install cache-keyed source capture before any sampler executes."""
        prepare_cube_prompt(prompt)
