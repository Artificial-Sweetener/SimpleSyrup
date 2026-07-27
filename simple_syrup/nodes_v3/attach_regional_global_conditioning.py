# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 internal node for regional global-prompt companions."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..runtime.regional_conditioning_companion import attach_global_companion

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class AttachRegionalGlobalConditioningV3(_ComfyNodeBase):
    """Attach the hooked global share required by one regional LoRA."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the internal companion-conditioning schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.AttachRegionalGlobalConditioning",
            display_name="Attach Regional Global Conditioning",
            category="SimpleSyrup/Internal",
            description=(
                "Keeps a region's global and local prompt shares under the same "
                "regional LoRA."
            ),
            inputs=[
                _comfy_io.Conditioning.Input(
                    "conditioning",
                    tooltip="Regional prompt conditioning to preserve.",
                ),
                _comfy_io.Conditioning.Input(
                    "global_conditioning",
                    tooltip=(
                        "Global prompt encoded with the same regional LoRA hooks."
                    ),
                ),
            ],
            outputs=[
                _comfy_io.Conditioning.Output(
                    "conditioning",
                    tooltip="Regional conditioning carrying its hooked global share.",
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning: object,
        global_conditioning: object,
    ) -> tuple[object]:
        """Attach the global companion to one regional conditioning."""

        return (attach_global_companion(conditioning, global_conditioning),)
