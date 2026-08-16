# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt one Comfy host Linear to exact temporary execution parameters."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import comfy.ops
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class RegionalHostLinearParameters:
    """Retain one exact cast weight and optional bias during an operation."""

    weight: torch.Tensor
    bias: torch.Tensor | None

    def __post_init__(self) -> None:
        """Require an ordinary matrix weight and aligned optional bias."""

        if self.weight.ndim != 2 or not self.weight.is_floating_point():
            raise ValueError("Regional host Linear weight is invalid.")
        if self.bias is not None and (
            self.bias.ndim != 1
            or int(self.bias.shape[0]) != int(self.weight.shape[0])
            or self.bias.device != self.weight.device
        ):
            raise ValueError("Regional host Linear bias is invalid.")


class RegionalHostLinearParameterProvider:
    """Own Comfy cast, patch-function, interrupt, and offload semantics."""

    def __init__(self, module: nn.Module) -> None:
        """Retain one exact-class host module with Linear parameter state."""

        if not isinstance(module, nn.Module):
            raise TypeError("Regional host Linear provider requires a module.")
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError("Regional host Linear provider requires a matrix weight.")
        self._module = module

    @contextmanager
    def acquire(self, inputs: torch.Tensor) -> Iterator[RegionalHostLinearParameters]:
        """Yield exact live parameters and release Comfy offload ownership."""

        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("Regional host Linear inputs must be floating.")
        comfy.ops.run_every_op()
        if not self._uses_comfy_cast_contract():
            weight = getattr(self._module, "weight", None)
            bias = getattr(self._module, "bias", None)
            if not isinstance(weight, torch.Tensor) or (
                bias is not None and not isinstance(bias, torch.Tensor)
            ):
                raise TypeError("Host Linear returned invalid direct parameters.")
            yield RegionalHostLinearParameters(weight, bias)
            return
        weight, bias, offload = comfy.ops.cast_bias_weight(
            self._module,
            inputs,
            offloadable=True,
        )
        if not isinstance(weight, torch.Tensor) or (
            bias is not None and not isinstance(bias, torch.Tensor)
        ):
            raise TypeError("Comfy returned invalid Regional Linear parameters.")
        try:
            yield RegionalHostLinearParameters(weight, bias)
        finally:
            comfy.ops.uncast_bias_weight(self._module, weight, bias, offload)

    def _uses_comfy_cast_contract(self) -> bool:
        """Mirror the native Comfy Linear cast-path admission predicate."""

        return bool(getattr(self._module, "comfy_cast_weights", False)) or bool(
            len(getattr(self._module, "weight_function", ()))
            or len(getattr(self._module, "bias_function", ()))
        )
