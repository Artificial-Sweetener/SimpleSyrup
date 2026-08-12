# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose internal stable identity labeling for regional LoRA hooks."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..runtime.regional_lora_hook_identity import label_regional_lora_hooks

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class LabelRegionalLoraHooksV3(_ComfyNodeBase):
    """Attach explicit adapter identities to cloned schedule-bearing hooks."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the internal ordered identity boundary."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.LabelRegionalLoraHooks",
            display_name="Label Regional LoRA Hooks",
            category="SimpleSyrup/Internal",
            description=(
                "Labels scheduled regional LoRA hooks with stable adapter "
                "identities without changing their weights or keyframes."
            ),
            inputs=[
                _comfy_io.Hooks.Input(
                    "hooks",
                    tooltip="Prompt Control hooks whose schedules remain unchanged.",
                ),
                _comfy_io.String.Input(
                    "adapter_identities_json",
                    tooltip=(
                        "Ordered JSON array with one stable identity per LoRA hook."
                    ),
                ),
            ],
            outputs=[
                _comfy_io.Hooks.Output(
                    "hooks",
                    tooltip="Cloned hooks carrying the supplied stable identities.",
                )
            ],
        )

    @classmethod
    def execute(cls, hooks: object, adapter_identities_json: str) -> tuple[object]:
        """Return one labeled clone of the supplied HookGroup."""

        return (label_regional_lora_hooks(hooks, adapter_identities_json),)
