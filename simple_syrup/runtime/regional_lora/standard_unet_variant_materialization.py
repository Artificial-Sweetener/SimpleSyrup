# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Materialize exact conventional weights for persistent regional variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from comfy import float as comfy_float
from comfy import lora, model_management, utils
from comfy.model_patcher import ModelPatcher, get_key_weight

from .standard_unet_variant_topology import StandardUnetRegionalVariant

_DIFFUSION_PREFIX = "diffusion_model."


@dataclass(frozen=True, slots=True)
class StandardUnetVariantParameter:
    """Bind one diffusion-relative path to an exact conventional tensor."""

    path: str
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        """Require one detached floating tensor and a valid relative path."""

        if not self.path or self.path.startswith(_DIFFUSION_PREFIX):
            raise ValueError("Variant parameter path must be diffusion-relative.")
        if (
            not isinstance(self.tensor, torch.Tensor)
            or not self.tensor.is_floating_point()
        ):
            raise TypeError("Variant parameter must be a floating tensor.")
        if self.tensor.requires_grad:
            raise ValueError("Variant parameter tensors must be detached.")


@dataclass(frozen=True, slots=True)
class StandardUnetMaterializedVariant:
    """Retain one region's exact complete replacement parameter bank."""

    region_index: int
    parameters: tuple[StandardUnetVariantParameter, ...]

    def __post_init__(self) -> None:
        """Require unique canonical parameter paths."""

        if self.region_index < 0 or not self.parameters:
            raise ValueError("Materialized regional variant must be nonempty.")
        paths = tuple(parameter.path for parameter in self.parameters)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("Materialized variant paths must be unique and sorted.")


class StandardUnetVariantMaterializer:
    """Apply Comfy's conventional patch math without mutating the source graph."""

    def materialize(
        self,
        model: object,
        variant: StandardUnetRegionalVariant,
        schedule_multipliers: tuple[float, ...],
    ) -> StandardUnetMaterializedVariant:
        """Return exact rounded regional weights over the live global MODEL."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("Standard UNet variant materialization requires a MODEL.")
        if not isinstance(variant, StandardUnetRegionalVariant):
            raise TypeError("Standard UNet variant materialization requires a variant.")
        self._validate_multipliers(variant, schedule_multipliers)
        patches_by_key: dict[str, list[tuple[object, ...]]] = {}
        for adapter in variant.adapters:
            strength = (
                adapter.model_strength * schedule_multipliers[adapter.composition_index]
            )
            for target in adapter.targets:
                patches_by_key.setdefault(target.path.parameter_key, []).append(
                    (
                        strength,
                        target.operation,
                        1.0,
                        target.path.offset,
                        None,
                    )
                )
        originals = model.get_key_patches(filter_prefix=_DIFFUSION_PREFIX)
        target_keys = tuple(
            sorted(
                set(patches_by_key)
                | {key for key, original in originals.items() if len(original) > 1}
            )
        )
        parameters = tuple(
            self._materialize_parameter(
                model,
                key,
                patches_by_key.get(key, []),
                originals=originals,
            )
            for key in target_keys
        )
        return StandardUnetMaterializedVariant(variant.region_index, parameters)

    @staticmethod
    def _materialize_parameter(
        model: ModelPatcher,
        key: str,
        patches: list[tuple[object, ...]],
        *,
        originals: dict[str, list[object]],
    ) -> StandardUnetVariantParameter:
        """Apply one ordered host patch list and conventional final rounding."""

        if not key.startswith(_DIFFUSION_PREFIX):
            raise ValueError("Regional variant target must belong to diffusion_model.")
        weight, set_function, convert_function = get_key_weight(model.model, key)
        if not isinstance(weight, torch.Tensor) or not isinstance(
            weight, torch.nn.Parameter
        ):
            raise TypeError(f"Regional variant target '{key}' must be a Parameter.")
        if set_function is not None or convert_function is not None:
            raise ValueError(
                f"Regional variant target '{key}' requires unsupported custom "
                "weight conversion."
            )
        original = originals.get(key)
        if original is None:
            raise RuntimeError(f"Regional variant target '{key}' lost base evidence.")
        base_entry = original[0]
        if (
            not isinstance(base_entry, tuple)
            or len(base_entry) != 2
            or not isinstance(base_entry[0], torch.Tensor)
        ):
            raise RuntimeError(f"Regional variant target '{key}' lost base weight.")
        base_weight = base_entry[0]
        temporary = model_management.cast_to_device(
            base_weight,
            weight.device,
            torch.float32,
            copy=True,
        )
        patched = lora.calculate_weight(
            [*original[1:], *patches],
            temporary,
            key,
            original_weights=originals,
        )
        if not isinstance(patched, torch.Tensor):
            raise TypeError(f"Regional variant target '{key}' returned no tensor.")
        rounded = comfy_float.stochastic_rounding(
            patched,
            weight.dtype,
            seed=utils.string_to_seed(key),
        ).detach()
        return StandardUnetVariantParameter(
            key.removeprefix(_DIFFUSION_PREFIX),
            rounded,
        )

    @staticmethod
    def _validate_multipliers(
        variant: StandardUnetRegionalVariant,
        multipliers: tuple[float, ...],
    ) -> None:
        """Require finite schedule values for every retained composition index."""

        if not isinstance(multipliers, tuple):
            raise TypeError("Standard UNet LoRA schedule multipliers must be a tuple.")
        indices = tuple(adapter.composition_index for adapter in variant.adapters)
        if not indices or max(indices) >= len(multipliers):
            raise ValueError("Standard UNet LoRA schedule is missing adapter values.")
        values = tuple(multipliers[index] for index in indices)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not torch.isfinite(torch.tensor(float(value))).item()
            for value in values
        ):
            raise ValueError("Standard UNet LoRA schedule values must be finite.")


STANDARD_UNET_VARIANT_MATERIALIZER = StandardUnetVariantMaterializer()
