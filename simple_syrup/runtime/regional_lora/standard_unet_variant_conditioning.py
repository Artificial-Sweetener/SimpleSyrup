# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve exact native conditioning for persistent standard-UNet variants."""

from __future__ import annotations

import torch

from ..regional_attention_execution_context import RegionalAttentionExecutionContext


class StandardUnetVariantConditioningResolver:
    """Select one exact CFG-aligned native context for each resident graph."""

    def __init__(self, execution_context: RegionalAttentionExecutionContext) -> None:
        """Retain the model-call-scoped aligned-context authority."""

        if not isinstance(execution_context, RegionalAttentionExecutionContext):
            raise TypeError(
                "Standard UNet variant conditioning requires an execution context."
            )
        self._execution_context = execution_context

    def global_context(self) -> torch.Tensor:
        """Return the active aligned global context for an uncovered base graph."""

        return self._execution_context.require_current().base_context

    def regional_context(self, region_index: int) -> torch.Tensor:
        """Return one exact native regional-or-base context per active batch row."""

        if isinstance(region_index, bool) or not isinstance(region_index, int):
            raise TypeError("Standard UNet variant region index must be an integer.")
        contexts = self._execution_context.require_current()
        if region_index < 0 or region_index >= len(contexts.regions):
            raise ValueError("Standard UNet variant region is outside conditioning.")
        region = contexts.regions[region_index]
        selected: list[torch.Tensor] = []
        uniform_entry: int | None = None
        uniform = True
        for row_index in range(int(contexts.base_context.shape[0])):
            active_entry: int | None = None
            for entry in region.entries:
                strength = entry.strengths[row_index]
                if strength not in (0.0, 1.0):
                    raise ValueError(
                        "Native standard UNet variant conditioning requires "
                        "binary regional entry strengths."
                    )
                if strength == 0.0:
                    continue
                if active_entry is not None:
                    raise ValueError(
                        "Native standard UNet variant conditioning requires at "
                        "most one active entry per batch row."
                    )
                active_entry = entry.entry_index
            if row_index == 0:
                uniform_entry = active_entry
            elif active_entry != uniform_entry:
                uniform = False
            source = (
                contexts.base_context
                if active_entry is None
                else region.entries[active_entry].context
            )
            selected.append(source[row_index])
        if uniform:
            return (
                contexts.base_context
                if uniform_entry is None
                else region.entries[uniform_entry].context
            )
        return torch.stack(selected, dim=0)
