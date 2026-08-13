# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Snapshot Comfy WeightHook payload state without resolving or mutating it."""

from __future__ import annotations

from dataclasses import dataclass

from comfy.hooks import WeightHook


@dataclass(frozen=True, slots=True)
class RegionalLoraHostPayload:
    """Retain one WeightHook's exact raw or initialized payload identities."""

    needs_resolution: bool
    raw_weights: object | None
    model_weights: object | None
    clip_weights: object | None

    def __post_init__(self) -> None:
        """Require one unambiguous raw or initialized host representation."""

        if not isinstance(self.needs_resolution, bool):
            raise TypeError("Regional LoRA payload resolution state must be boolean.")
        if self.needs_resolution:
            if self.raw_weights is None:
                raise ValueError(
                    "Unresolved regional LoRA payload requires raw weights."
                )
            if self.model_weights is not None or self.clip_weights is not None:
                raise ValueError(
                    "Unresolved regional LoRA payload cannot contain "
                    "initialized weights."
                )
        elif self.raw_weights is not None:
            raise ValueError(
                "Initialized regional LoRA payload cannot retain raw weights."
            )

    @classmethod
    def from_hook(cls, hook: WeightHook) -> RegionalLoraHostPayload:
        """Copy payload references from one installed Comfy WeightHook."""

        if not isinstance(hook, WeightHook):
            raise TypeError("Regional LoRA host payload requires a WeightHook.")
        if hook.need_weight_init:
            return cls(True, hook.weights, None, None)
        return cls(False, None, hook.weights, hook.weights_clip)

    @classmethod
    def unresolved(cls, weights: object) -> RegionalLoraHostPayload:
        """Declare one raw model-and-clip payload requiring Comfy resolution."""

        return cls(True, weights, None, None)

    @property
    def model_side_weights(self) -> object | None:
        """Return the exact payload Comfy presents to the model side."""

        return self.raw_weights if self.needs_resolution else self.model_weights
