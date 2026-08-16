# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Restore and load a Comfy source model before graph-sensitive preparation."""

from __future__ import annotations

from comfy import model_management


class ComfyConditioningModelLoader:
    """Own source-patcher restoration and pre-sampling model residency."""

    def load(self, model: object) -> None:
        """Load the source patcher so its graph and processing device are current."""

        model_management.load_model_gpu(model)


COMFY_CONDITIONING_MODEL_LOADER = ComfyConditioningModelLoader()
