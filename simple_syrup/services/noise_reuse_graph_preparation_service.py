# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Expose capture preparation through the existing public Cube capability."""

from __future__ import annotations

from ..runtime.noise_reuse_validation import prepare_noise_reuse_prompt


class NoiseReuseGraphPreparationService:
    """Delegate public host requests to the authoritative capture planner."""

    def prepare(self, prompt: dict[str, dict[str, object]]) -> None:
        """Install cache-keyed source capture before any sampler executes."""
        prepare_noise_reuse_prompt(prompt)
