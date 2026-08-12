# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select the WeightHooks that explicitly request regional model mutation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from comfy.hooks import HookGroup, WeightHook


@dataclass(frozen=True, slots=True)
class RegionalModelWeightHook:
    """Retain one model-active hook with its original order and strength."""

    hook_index: int
    hook: WeightHook
    model_strength: float

    def __post_init__(self) -> None:
        """Validate narrowed installed-Comfy hook state."""

        if isinstance(self.hook_index, bool) or not isinstance(self.hook_index, int):
            raise TypeError("Regional model hook index must be an integer.")
        if self.hook_index < 0:
            raise ValueError("Regional model hook index must be non-negative.")
        if not isinstance(self.hook, WeightHook):
            raise TypeError("Regional model hook must be a Comfy WeightHook.")
        if not math.isfinite(self.model_strength) or self.model_strength == 0.0:
            raise ValueError("Regional model hook strength must be finite and nonzero.")


@dataclass(frozen=True, slots=True)
class RegionalModelHookSelection:
    """Describe model-active hooks relative to all ordered WeightHooks."""

    weight_hook_count: int
    model_hooks: tuple[RegionalModelWeightHook, ...]

    def __post_init__(self) -> None:
        """Require ordered participants within the source HookGroup."""

        if isinstance(self.weight_hook_count, bool) or not isinstance(
            self.weight_hook_count, int
        ):
            raise TypeError("Regional WeightHook count must be an integer.")
        if self.weight_hook_count < 0:
            raise ValueError("Regional WeightHook count must be non-negative.")
        if not isinstance(self.model_hooks, tuple) or any(
            not isinstance(item, RegionalModelWeightHook) for item in self.model_hooks
        ):
            raise TypeError("Regional model hook selection has invalid entries.")
        indices = tuple(item.hook_index for item in self.model_hooks)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("Regional model hook indices must be unique and ordered.")
        if indices and indices[-1] >= self.weight_hook_count:
            raise ValueError("Regional model hook index exceeds the WeightHook count.")


class RegionalModelHookSelector:
    """Validate one HookGroup and select explicit model-side participants."""

    def select(
        self,
        hooks: object,
        *,
        source_label: str,
    ) -> RegionalModelHookSelection:
        """Return nonzero-model-strength WeightHooks without changing the source."""

        if not isinstance(source_label, str) or not source_label.strip():
            raise ValueError("Regional model hook source label must be non-empty.")
        if not isinstance(hooks, HookGroup):
            raise TypeError(f"{source_label} must contain a Comfy HookGroup.")
        if not isinstance(hooks.hooks, list):
            raise TypeError(f"{source_label} HookGroup hooks must be a list.")

        unsupported_hooks = tuple(
            hook for hook in hooks.hooks if not isinstance(hook, WeightHook)
        )
        if unsupported_hooks:
            unsupported_names = ", ".join(
                type(hook).__name__ for hook in unsupported_hooks
            )
            raise TypeError(
                f"{source_label} contains unsupported hooks: {unsupported_names}."
            )

        weight_hooks = tuple(
            hook for hook in hooks.hooks if isinstance(hook, WeightHook)
        )
        model_hooks: list[RegionalModelWeightHook] = []
        for hook_index, hook in enumerate(weight_hooks):
            model_strength = self._model_strength(
                getattr(hook, "_strength_model", None),
                source_label=source_label,
                hook_index=hook_index,
            )
            if model_strength == 0.0:
                continue
            model_hooks.append(
                RegionalModelWeightHook(hook_index, hook, model_strength)
            )
        return RegionalModelHookSelection(len(weight_hooks), tuple(model_hooks))

    @staticmethod
    def _model_strength(
        value: object,
        *,
        source_label: str,
        hook_index: int,
    ) -> float:
        """Narrow one host strength before it controls model participation."""

        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise TypeError(
                f"{source_label} WeightHook {hook_index} model strength must be "
                "finite numeric data."
            )
        return float(value)


REGIONAL_MODEL_HOOK_SELECTOR = RegionalModelHookSelector()
