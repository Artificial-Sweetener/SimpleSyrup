# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load a Comfy model before device-bound conditioning preprocessing."""

from __future__ import annotations

from comfy import model_management


class ComfyConditioningModelLoader:
    """Own official Comfy model residency for pre-sampling context conversion."""

    def load(self, model: object) -> None:
        """Load the exact patcher so model weights share its processing device."""

        model_management.load_model_gpu(model)


COMFY_CONDITIONING_MODEL_LOADER = ComfyConditioningModelLoader()
