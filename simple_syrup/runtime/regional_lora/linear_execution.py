# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute generic regional ordinary-LoRA deltas around one original Linear call."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ...domain.regional_activation_geometry import RegionalActivationLayout
from .delta_execution import REGIONAL_LORA_DELTA_EXECUTOR, RegionalLoraDeltaExecutor
from .linear_execution_plan import RegionalLinearExecutionPlan
from .operation_mask_resolution import RegionalOperationMaskBatch


class RegionalLinearExecutor:
    """Call the original once and add exact active adapter groups in order."""

    def __init__(
        self,
        delta_executor: RegionalLoraDeltaExecutor = REGIONAL_LORA_DELTA_EXECUTOR,
    ) -> None:
        """Retain the existing model-neutral projection and accumulation owner."""

        if not isinstance(delta_executor, RegionalLoraDeltaExecutor):
            raise TypeError("Regional Linear execution requires a delta executor.")
        self._delta_executor = delta_executor

    def execute(
        self,
        original: Callable[..., object],
        inputs: torch.Tensor,
        *args: object,
        plan: RegionalLinearExecutionPlan,
        masks: RegionalOperationMaskBatch,
        schedule_strengths: tuple[float, ...],
        **kwargs: object,
    ) -> torch.Tensor:
        """Execute one original operation and ordered regional low-rank additions."""

        self._validate_call(inputs, plan, masks, schedule_strengths)
        original_output = original(inputs, *args, **kwargs)
        if not isinstance(original_output, torch.Tensor):
            raise TypeError("Regional Linear original operation must return a tensor.")
        expected_output_shape = (
            *inputs.shape[:-1],
            plan.groups[0].preparation.target.output_features,
        )
        if (
            tuple(original_output.shape) != expected_output_shape
            or original_output.device != inputs.device
            or original_output.dtype != inputs.dtype
        ):
            raise ValueError(
                "Regional Linear original output must match its target shape and "
                "input execution type."
            )
        multipliers = self._group_multipliers(
            plan,
            masks,
            schedule_strengths=schedule_strengths,
        )
        active = tuple(
            index
            for index, multiplier in enumerate(multipliers)
            if multiplier is not None
        )
        if not active:
            return original_output
        if len(active) == 1:
            index = active[0]
            multiplier = multipliers[index]
            if multiplier is None:
                raise AssertionError("Active Regional Linear multiplier disappeared.")
            return self._delta_executor.add_masked_delta(
                original_output,
                inputs,
                preparation=plan.groups[index].preparation,
                multiplier=multiplier,
            )
        deltas: list[torch.Tensor | None] = [None] * len(plan.groups)
        for indices, preparation in plan.rank_batches:
            selected = tuple(
                index for index in indices if multipliers[index] is not None
            )
            if not selected:
                continue
            if len(selected) == 1:
                index = selected[0]
                multiplier = multipliers[index]
                if multiplier is None:
                    raise AssertionError(
                        "Active Regional Linear multiplier disappeared."
                    )
                deltas[index] = self._delta_executor.masked_delta(
                    inputs,
                    preparation=plan.groups[index].preparation,
                    multiplier=multiplier,
                )
                continue
            selected_preparation = plan.preparation_for(
                selected_indices=selected,
                complete_indices=indices,
                complete=preparation,
            )
            batch = self._delta_executor.compatible_batch(
                inputs,
                preparation=selected_preparation,
                multipliers=tuple(
                    multiplier
                    for index in selected
                    if (multiplier := multipliers[index]) is not None
                ),
            )
            for local_index, group_index in enumerate(selected):
                deltas[group_index] = batch[..., local_index, :]
        return self._delta_executor.accumulate_ordered(
            original_output,
            tuple(delta for delta in deltas if delta is not None),
        )

    @staticmethod
    def _group_multipliers(
        plan: RegionalLinearExecutionPlan,
        masks: RegionalOperationMaskBatch,
        *,
        schedule_strengths: tuple[float, ...],
    ) -> tuple[torch.Tensor | None, ...]:
        """Combine contiguous repeated uses in declared floating addition order."""

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
                use_index = masks.composition_indices.index(use.composition_index)
                contribution = masks.multipliers[use_index].squeeze(-1) * scale
                multiplier = (
                    contribution if multiplier is None else multiplier + contribution
                )
            if multiplier is not None and not bool(torch.count_nonzero(multiplier)):
                multiplier = None
            combined.append(multiplier)
        return tuple(combined)

    @staticmethod
    def _validate_call(
        inputs: object,
        plan: object,
        masks: object,
        schedule_strengths: object,
    ) -> None:
        """Validate the explicit call contract before invoking the original."""

        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("Regional Linear inputs must be a floating tensor.")
        if not isinstance(plan, RegionalLinearExecutionPlan):
            raise TypeError("Regional Linear execution requires a typed plan.")
        if not isinstance(masks, RegionalOperationMaskBatch):
            raise TypeError("Regional Linear execution requires operation masks.")
        if masks.geometry.layout not in (
            RegionalActivationLayout.FLATTENED_SPATIAL_TOKENS,
            RegionalActivationLayout.CONSUMER_SPATIALIZED,
            RegionalActivationLayout.BRANCH_TOKENS,
        ):
            raise ValueError(
                "Regional Linear execution requires spatial or branch-token geometry."
            )
        if tuple(inputs.shape) != masks.geometry.invocation_shape:
            raise ValueError("Regional Linear input must match activation geometry.")
        if (
            masks.multipliers.device != inputs.device
            or masks.multipliers.dtype != inputs.dtype
        ):
            raise ValueError("Regional Linear masks must match input device and dtype.")
        if not isinstance(schedule_strengths, tuple) or len(schedule_strengths) != len(
            plan.uses
        ):
            raise ValueError("Regional Linear schedule strengths must align to uses.")
        if any(
            isinstance(strength, bool)
            or not isinstance(strength, int | float)
            or not math.isfinite(float(strength))
            for strength in schedule_strengths
        ):
            raise TypeError("Regional Linear schedule strengths must be finite.")
        if int(inputs.shape[-1]) != plan.groups[0].preparation.target.input_features:
            raise ValueError("Regional Linear input feature count is invalid.")
        if masks.composition_indices != tuple(
            use.composition_index for use in plan.uses
        ):
            raise ValueError(
                "Regional Linear operation masks must align to target uses."
            )


REGIONAL_LINEAR_EXECUTOR = RegionalLinearExecutor()
