# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt one UNet coupling execution to Comfy's paired attn2 callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .unet_attn2_execution import UnetAttn2Execution
from .unet_attn2_execution_resolver import UnetAttn2ExecutionResolver

_INVOCATION_KEY = "simple_syrup.unet_attn2_invocation"


@dataclass(frozen=True, slots=True)
class _UnetAttn2Invocation:
    """Bind one pending output callback to its exact execution authority."""

    execution: UnetAttn2Execution


class UnetAttn2PatchPair:
    """Own one collision-detecting input/output callback pair."""

    def __init__(self, resolver: UnetAttn2ExecutionResolver) -> None:
        """Retain the sole per-layer execution-resolution authority."""

        if not isinstance(resolver, UnetAttn2ExecutionResolver):
            raise TypeError("UNet attn2 patch pair requires an execution resolver.")
        self._resolver = resolver

    def input_patch(
        self,
        n: torch.Tensor,
        context_attn2: torch.Tensor,
        value_attn2: torch.Tensor,
        extra_options: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack supported rows before Comfy's native cross-attention call."""

        options = self._options(extra_options)
        if _INVOCATION_KEY in options:
            raise ValueError("UNet attn2 input callback already has pending state.")
        execution = self._resolver.resolve(n, context_attn2, options)
        if not isinstance(execution, UnetAttn2Execution):
            raise TypeError("UNet attn2 resolver returned an invalid execution.")
        expanded = execution.expand(n, context_attn2, value_attn2)
        options[_INVOCATION_KEY] = _UnetAttn2Invocation(execution)
        return expanded.query, expanded.context, expanded.value

    def output_patch(
        self,
        n: torch.Tensor,
        extra_options: dict[str, Any],
    ) -> torch.Tensor:
        """Restore the source batch before Comfy's residual and feed-forward path."""

        options = self._options(extra_options)
        invocation = options.pop(_INVOCATION_KEY, None)
        if not isinstance(invocation, _UnetAttn2Invocation):
            raise ValueError("UNet attn2 output callback has no matching input state.")
        return invocation.execution.blend(n)

    @staticmethod
    def _options(value: object) -> dict[str, Any]:
        """Narrow Comfy's callback-local options without copying its identity."""

        if not isinstance(value, dict):
            raise TypeError("UNet attn2 extra_options must be a dictionary.")
        return value
