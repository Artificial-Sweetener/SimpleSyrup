# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapter node for LayerStyle combined SAM model bundles."""

from __future__ import annotations


class LayerStyleSAMModelsAdapter:
    """Split a LayerStyle SAM/DINO bundle into conventional model sockets."""

    RETURN_TYPES = ("SAM_MODEL", "DINO_MODEL")
    RETURN_NAMES = ("sam_model", "dino_model")
    OUTPUT_TOOLTIPS = (
        "SAM model split from the LayerStyle bundle for SimpleSyrup mask nodes.",
        "DINO model split from the LayerStyle bundle for SimpleSyrup detection nodes.",
    )
    FUNCTION = "adapt"
    CATEGORY = "SimpleSyrup/Masking"
    DESCRIPTION = "Splits LayerStyle LS_SAM_MODELS into SAM_MODEL and DINO_MODEL."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        """Declare the LayerStyle bundle input."""

        return {
            "required": {
                "sam_models": (
                    "LS_SAM_MODELS",
                    {
                        "tooltip": (
                            "LayerStyle SAM/DINO bundle to split into separate "
                            "SimpleSyrup-compatible model sockets."
                        )
                    },
                )
            }
        }

    def adapt(self, sam_models: object) -> tuple[object, object]:
        """Return SAM and DINO objects from a LayerStyle-style bundle."""

        if not isinstance(sam_models, dict):
            raise TypeError(
                "LS_SAM_MODELS must be a mapping with SAM_MODEL and DINO_MODEL."
            )
        if "SAM_MODEL" not in sam_models or "DINO_MODEL" not in sam_models:
            raise ValueError("LS_SAM_MODELS must contain SAM_MODEL and DINO_MODEL.")
        return sam_models["SAM_MODEL"], sam_models["DINO_MODEL"]
