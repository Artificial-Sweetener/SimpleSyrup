# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 internal node for regional LoRA CLIP preparation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..runtime.regional_lora_hooks import prepare_regional_lora_clip

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class PrepareRegionalLoraHooksV3(_ComfyNodeBase):
    """Prepare only the CLIP work required by regional LoRA hooks."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the internal hook conversion schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.PrepareRegionalLoraHooks",
            display_name="Prepare Regional LoRA Hooks",
            category="SimpleSyrup/Internal",
            description=(
                "Keeps model-only regional LoRAs away from the text encoder while "
                "preparing text-encoder patches when present."
            ),
            inputs=[
                _comfy_io.Clip.Input(
                    "clip",
                    tooltip="CLIP model used to encode this regional prompt.",
                ),
                _comfy_io.Hooks.Input(
                    "hooks",
                    tooltip=("Prompt-Control LoRA hooks to inspect and prepare."),
                ),
            ],
            outputs=[
                _comfy_io.Clip.Output(
                    "clip",
                    tooltip=(
                        "Original CLIP for model-only LoRAs, or a hook-prepared "
                        "CLIP when text-encoder weights are present."
                    ),
                ),
                _comfy_io.Hooks.Output(
                    "hooks",
                    tooltip=("Regional LoRA hooks to attach after prompt encoding."),
                ),
            ],
        )

    @classmethod
    def execute(cls, clip: Any, hooks: object) -> tuple[Any, object]:
        """Return the appropriate encoding CLIP and unchanged model hooks."""

        return prepare_regional_lora_clip(clip, hooks)
