# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the ordered native global-LoRA reference cases."""

from __future__ import annotations

from enum import StrEnum


class NativeGlobalLoraCase(StrEnum):
    """Name one ordinary native-Comfy global-LoRA condition."""

    TRIGGER_CONTROL = "native-trigger-control"
    GLOBAL_LORA = "native-global-lora"

    @property
    def applies_lora(self) -> bool:
        """Return whether this case loads the adapter on MODEL and CLIP."""

        return self is NativeGlobalLoraCase.GLOBAL_LORA

    @property
    def label(self) -> str:
        """Return the in-image condition label."""

        if self is NativeGlobalLoraCase.TRIGGER_CONTROL:
            return "A — NATIVE KSampler · TRIGGER ONLY · NO LoRA"
        return "B — NATIVE KSampler · GLOBAL MODEL + CLIP LoRA 0.65"


def ordered_native_global_lora_cases() -> tuple[NativeGlobalLoraCase, ...]:
    """Return the only valid causal execution order."""

    return (
        NativeGlobalLoraCase.TRIGGER_CONTROL,
        NativeGlobalLoraCase.GLOBAL_LORA,
    )
