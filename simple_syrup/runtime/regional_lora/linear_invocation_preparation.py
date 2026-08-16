# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare and reuse invariant regional Linear mask multipliers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .linear_execution_plan import RegionalLinearExecutionPlan


@dataclass(frozen=True, slots=True)
class RegionalLinearPreparedInvocation:
    """Retain combined declared-order multipliers for one mask signature."""

    multipliers: tuple[torch.Tensor | None, ...]


class RegionalLinearInvocationPreparationCache:
    """Own mask-local reuse of Linear invocation preparation."""

    def __init__(self) -> None:
        """Create one empty cache released with its operation-mask batch."""

        self._values: dict[tuple[object, ...], RegionalLinearPreparedInvocation] = {}

    def resolve(
        self,
        plan: RegionalLinearExecutionPlan,
        *,
        mask_multipliers: torch.Tensor,
        composition_indices: tuple[int, ...],
        schedule_strengths: tuple[float, ...],
        leading_shape: tuple[int, ...],
    ) -> RegionalLinearPreparedInvocation:
        """Return one prepared invocation for an exact scalar and shape contract."""

        signature = (
            leading_shape,
            schedule_strengths,
            tuple(
                tuple((use.composition_index, use.base_strength) for use in group.uses)
                for group in plan.groups
            ),
        )
        cached = self._values.get(signature)
        if cached is not None:
            return cached
        multipliers = self._multipliers(
            plan,
            mask_multipliers=mask_multipliers,
            composition_indices=composition_indices,
            schedule_strengths=schedule_strengths,
        )
        prepared = RegionalLinearPreparedInvocation(multipliers)
        self._values[signature] = prepared
        return prepared

    @staticmethod
    def _multipliers(
        plan: RegionalLinearExecutionPlan,
        *,
        mask_multipliers: torch.Tensor,
        composition_indices: tuple[int, ...],
        schedule_strengths: tuple[float, ...],
    ) -> tuple[torch.Tensor | None, ...]:
        """Combine contiguous repeated uses without synchronizing device state."""

        strengths = {
            use.composition_index: schedule_strengths[index]
            for index, use in enumerate(plan.uses)
        }
        combined: list[torch.Tensor | None] = []
        for group in plan.groups:
            multiplier: torch.Tensor | None = None
            for use in group.uses:
                scale = use.base_strength * strengths[use.composition_index]
                if scale == 0.0:
                    continue
                use_index = composition_indices.index(use.composition_index)
                contribution = mask_multipliers[use_index].squeeze(-1) * scale
                multiplier = (
                    contribution if multiplier is None else multiplier + contribution
                )
            if multiplier is not None and not bool(torch.count_nonzero(multiplier)):
                multiplier = None
            combined.append(multiplier)
        return tuple(combined)
