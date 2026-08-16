# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own collision-safe Comfy attention replacement mutations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .model_patcher_mutations import (
    _require_bound_method,
    _require_dictionary_attribute,
)

Attn1ReplacementKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class ModelAttn1ReplacementsMutation:
    """Install one callback at exact standard-UNet attn1 replacement keys."""

    replacement_keys: tuple[Attn1ReplacementKey, ...]
    replacement: Callable[..., object]

    def __post_init__(self) -> None:
        """Require unique deterministic Comfy block keys and one callback."""

        if not callable(self.replacement):
            raise TypeError("MODEL attn1 replacement must be callable.")
        if not self.replacement_keys:
            raise ValueError("MODEL attn1 replacements require at least one key.")
        for key in self.replacement_keys:
            self._validate_key(key)
        if self.replacement_keys != tuple(sorted(set(self.replacement_keys))):
            raise ValueError("MODEL attn1 replacement keys must be unique and sorted.")

    def apply(self, model: object) -> None:
        """Validate the complete replacement surface before installing any key."""

        setter = _require_bound_method(
            model,
            "set_model_attn1_replace",
            ("patch", "block_name", "number", "transformer_index"),
        )
        model_options = _require_dictionary_attribute(model, "model_options")
        transformer_options = model_options.get("transformer_options")
        if not isinstance(transformer_options, dict):
            raise TypeError("MODEL transformer_options must be a dictionary.")
        patches_replace = transformer_options.get("patches_replace", {})
        if not isinstance(patches_replace, dict):
            raise TypeError(
                "MODEL transformer replacement patches must be a dictionary."
            )
        existing = patches_replace.get("attn1", {})
        if not isinstance(existing, dict):
            raise TypeError("Existing MODEL attn1 replacements must be a dictionary.")
        if any(not callable(callback) for callback in existing.values()):
            raise TypeError(
                "Existing MODEL attn1 replacements must contain only callables."
            )
        for key in self.replacement_keys:
            broad_key = key[:2]
            if key in existing or broad_key in existing:
                raise ValueError(
                    f"MODEL attn1 replacement for block {key!r} is already installed."
                )
        for block_name, number, transformer_index in self.replacement_keys:
            setter(self.replacement, block_name, number, transformer_index)

    @staticmethod
    def _validate_key(key: object) -> None:
        """Require one exact standard-UNet Comfy replacement key."""

        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError("MODEL attn1 replacement keys must contain three values.")
        block_name, number, transformer_index = key
        if block_name not in {"input", "middle", "output"}:
            raise ValueError("MODEL attn1 replacement block kind is unsupported.")
        for value in (number, transformer_index):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "MODEL attn1 replacement indices must be non-negative integers."
                )
