# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive outer sampler work expectations for SDXL visual scenarios."""

from __future__ import annotations

from .matrix import SdxlIntegrationMode
from .visual_case_model import SdxlVisualCase


class SdxlVisualRuntimeExpectations:
    """Own outer sampler-call expectations across regional adapter topologies."""

    @staticmethod
    def expected_model_calls(
        case: SdxlVisualCase,
        mode: SdxlIntegrationMode,
    ) -> int:
        """Return the mode's native outer calls after validating scenario types."""

        if not isinstance(case, SdxlVisualCase):
            raise TypeError("SDXL visual runtime expectations require a case.")
        if not isinstance(mode, SdxlIntegrationMode):
            raise TypeError("SDXL visual runtime expectations require a mode.")
        return mode.expected_model_calls


SDXL_VISUAL_RUNTIME_EXPECTATIONS = SdxlVisualRuntimeExpectations()
