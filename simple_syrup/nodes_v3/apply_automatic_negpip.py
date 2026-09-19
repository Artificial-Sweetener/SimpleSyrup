# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Internal Comfy v3 node for model-family automatic NegPiP preparation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ..services.negpip_model_service import NEGPIP_MODEL_SERVICE

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        pass

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class ApplyAutomaticNegpipV3(_ComfyNodeBase):
    """Patch supported MODEL/CLIP pairs after a negative prompt-weight trigger."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the internal runtime patch boundary."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.ApplyAutomaticNegpip",
            display_name="Apply Automatic NegPiP (Internal)",
            category="SimpleSyrup/Internal",
            description=(
                "Internal model-family NegPiP preparation injected by Schedule & "
                "Encode Prompts after detecting a negative prompt weight."
            ),
            is_dev_only=True,
            inputs=[
                _comfy_io.Model.Input(
                    "model",
                    tooltip="MODEL inspected and cloned only when NegPiP is supported.",
                ),
                _comfy_io.Clip.Input(
                    "clip",
                    tooltip="CLIP cloned with the matching NegPiP encoder behavior.",
                ),
            ],
            outputs=[
                _comfy_io.Model.Output(
                    "model",
                    tooltip="MODEL carrying one supported NegPiP attention patch set.",
                ),
                _comfy_io.Clip.Output(
                    "clip",
                    tooltip="CLIP carrying matching negative-weight encoding behavior.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model: object, clip: object) -> tuple[object, object]:
        """Return the supported patched pair or the original unsupported pair."""

        return NEGPIP_MODEL_SERVICE.prepare(model, clip)
