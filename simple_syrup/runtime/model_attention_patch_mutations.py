# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own paired Comfy attention input/output patch mutations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .model_patcher_mutations import (
    _require_bound_method,
    _require_callable_patch_list,
    _require_dictionary_attribute,
)


def _apply_paired_attention_patches(
    model: object,
    *,
    attention_name: str,
    input_patch: Callable[..., object],
    output_patch: Callable[..., object],
) -> None:
    """Validate and atomically install one paired attention callback surface."""

    if not callable(input_patch):
        raise TypeError(f"MODEL {attention_name} input patch must be callable.")
    if not callable(output_patch):
        raise TypeError(f"MODEL {attention_name} output patch must be callable.")
    input_setter = _require_bound_method(
        model,
        f"set_model_{attention_name}_patch",
        ("patch",),
    )
    output_setter = _require_bound_method(
        model,
        f"set_model_{attention_name}_output_patch",
        ("patch",),
    )
    model_options = _require_dictionary_attribute(model, "model_options")
    transformer_options = model_options.get("transformer_options")
    if not isinstance(transformer_options, dict):
        raise TypeError("MODEL transformer_options must be a dictionary.")
    if "patches" not in transformer_options:
        patches: dict[object, object] = {}
    else:
        patches_value = transformer_options["patches"]
        if not isinstance(patches_value, dict):
            raise TypeError("MODEL transformer patches must be a dictionary.")
        patches = patches_value
    input_name = f"{attention_name}_patch"
    output_name = f"{attention_name}_output_patch"
    input_exists = _require_callable_patch_list(patches, input_name)
    output_exists = _require_callable_patch_list(patches, output_name)
    if input_exists:
        raise ValueError(f"MODEL {attention_name} input patch is already installed.")
    if output_exists:
        raise ValueError(f"MODEL {attention_name} output patch is already installed.")
    input_setter(input_patch)
    output_setter(output_patch)


@dataclass(frozen=True)
class ModelAttn2PatchesMutation:
    """Install paired collision-safe attn2 input and output patches."""

    input_patch: Callable[..., object]
    output_patch: Callable[..., object]

    def apply(self, model: object) -> None:
        """Validate both attn2 surfaces before either mutation."""

        _apply_paired_attention_patches(
            model,
            attention_name="attn2",
            input_patch=self.input_patch,
            output_patch=self.output_patch,
        )
