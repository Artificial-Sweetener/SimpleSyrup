# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for latent metadata diagnostics."""

from __future__ import annotations

from typing import Any, TypedDict

from ..services.latent_diagnostics_service import LatentDiagnosticsService

Latent = dict[str, Any]


class LatentDiagnosticsResult(TypedDict):
    """ComfyUI execution result with UI text and typed output data."""

    ui: dict[str, list[str]]
    result: tuple[Latent, str]


class LatentDiagnostics:
    """Expose safe latent tensor metadata for debugging model compatibility."""

    _service = LatentDiagnosticsService()

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    OUTPUT_TOOLTIPS = (
        "Input latent passed through unchanged for continued workflow use.",
        "Text report describing latent shape, dtype, device, and tiling fit.",
    )
    FUNCTION = "analyze"
    CATEGORY = "SimpleSyrup/Utilities"
    DESCRIPTION = (
        "Reports latent tensor shape, dtype, device, and tiling compatibility."
    )
    OUTPUT_NODE = True
    SEARCH_ALIASES = ["latent", "diagnostics", "inspect latent", "debug latent"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare the latent diagnostic input contract."""

        return {
            "required": {
                "latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Latent to inspect. The node reports metadata and passes "
                            "it through unchanged."
                        )
                    },
                )
            }
        }

    def analyze(self, latent: Latent) -> LatentDiagnosticsResult:
        """Return the input latent and a metadata report for ComfyUI display."""

        report = self._service.describe(latent)
        return {"ui": {"text": [report]}, "result": (latent, report)}
