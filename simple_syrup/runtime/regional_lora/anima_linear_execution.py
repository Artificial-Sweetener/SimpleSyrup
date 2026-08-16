# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Batch exact ordered multi-adapter deltas for one Anima linear target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..regional_lora_schedule_resolution import RegionalLoraScheduleResolution
from .anima_composition import AnimaRegionalLoraTargetGroup
from .anima_host_module_backing import (
    AnimaHostBackedLinearModule,
    AnimaHostModuleBacking,
)
from .anima_lora_weights import AnimaLoraWeightResolver
from .anima_projection_batch import (
    AnimaProjectionBatchRegistry,
    AnimaProjectionBatchRequest,
)
from .anima_schedule_context import AnimaRegionalLoraScheduleContext
from .delta_execution import (
    REGIONAL_LORA_DELTA_EXECUTOR,
    RegionalLoraDeltaExecutor,
)
from .preparation import (
    RegionalLoraCompatibleBatchPreparation,
    RegionalLoraTargetPreparation,
)


@dataclass(frozen=True, slots=True)
class _ExecutableTargetGroup:
    """Bind one contiguous composition group to its prepared-weight owner."""

    group: AnimaRegionalLoraTargetGroup
    preparation: RegionalLoraTargetPreparation


@dataclass(frozen=True, slots=True)
class _ExecutableRankBatch:
    """Bind compatible composition-group indices to one preparation owner."""

    indices: tuple[int, ...]
    preparation: RegionalLoraCompatibleBatchPreparation


class AnimaRegionalLoraCompositionLinearPatch(AnimaHostBackedLinearModule):
    """Call one original linear and add exact batched deltas in declared order."""

    def __init__(
        self,
        original: nn.Module,
        groups: tuple[AnimaRegionalLoraTargetGroup, ...],
        *,
        weight_resolver: AnimaLoraWeightResolver,
        schedule_context: AnimaRegionalLoraScheduleContext,
        delta_executor: RegionalLoraDeltaExecutor = REGIONAL_LORA_DELTA_EXECUTOR,
        projection_batches: AnimaProjectionBatchRegistry | None = None,
    ) -> None:
        """Retain one target's active ordered groups and scope-aware weights."""

        super().__init__()
        if not isinstance(original, nn.Module):
            raise TypeError("Anima LoRA composition original target must be a module.")
        if not isinstance(groups, tuple) or not groups:
            raise ValueError("Anima LoRA composition target groups cannot be empty.")
        if any(not isinstance(group, AnimaRegionalLoraTargetGroup) for group in groups):
            raise TypeError("Anima LoRA composition contains an invalid target group.")
        target_name = groups[0].target.adapter.target
        if any(group.target.adapter.target != target_name for group in groups):
            raise ValueError("Anima LoRA composition groups must share one target.")
        if not isinstance(weight_resolver, AnimaLoraWeightResolver):
            raise TypeError("Anima LoRA composition requires a weight resolver.")
        if not isinstance(schedule_context, AnimaRegionalLoraScheduleContext):
            raise TypeError("Anima LoRA composition requires a schedule context.")
        self._backing = AnimaHostModuleBacking(original)
        self._backing.install_direct_state(self)
        self.groups = groups
        self._weight_resolver = weight_resolver
        self._schedule_context = schedule_context
        self._delta_executor = delta_executor
        self._projection_batches = projection_batches
        self._executables = tuple(
            _ExecutableTargetGroup(
                group,
                RegionalLoraTargetPreparation(
                    adapter_identity=group.uses[
                        0
                    ].execution.adapter_plan.adapter_identity,
                    model_lineage=group.uses[0].execution.model_lineage,
                    target=group.target.adapter,
                    cache=group.uses[0].execution.cache,
                ),
            )
            for group in groups
        )
        self._rank_batches = self._build_rank_batches(self._executables)
        self._active_batch_preparations: dict[
            tuple[int, ...], RegionalLoraCompatibleBatchPreparation
        ] = {}

    def forward(self, inputs: torch.Tensor, *args: object, **kwargs: object) -> Any:
        """Apply masks before B, batch compatible groups, then add in exact order."""

        self._validate_inputs(inputs)
        resolution = self._schedule_context.require_current()
        if self._projection_batches is not None:
            coordinated_execution = self._projection_batches.resolve_execution(
                self,
                inputs,
                resolution,
            )
            if coordinated_execution is not None:
                original_output = self._backing.module(inputs, *args, **kwargs)
                if not isinstance(original_output, torch.Tensor):
                    raise TypeError(
                        "Anima LoRA composition original must return a tensor."
                    )
                return self._projection_batches.accumulate(
                    original_output,
                    coordinated_execution,
                )
        if len(self._executables) == 1:
            return self._forward_single_group(
                inputs,
                args,
                kwargs,
                resolution,
            )
        multipliers: tuple[torch.Tensor | None, ...] = tuple(
            self._combined_multiplier(executable.group, inputs, resolution)
            for executable in self._executables
        )
        original_output = self._backing.module(inputs, *args, **kwargs)
        if not isinstance(original_output, torch.Tensor):
            raise TypeError("Anima LoRA composition original must return a tensor.")
        active_indices = tuple(
            index
            for index, multiplier in enumerate(multipliers)
            if multiplier is not None
        )
        if len(active_indices) == 1:
            group_index = active_indices[0]
            multiplier = multipliers[group_index]
            if multiplier is None:
                raise AssertionError("Active Anima LoRA multiplier became unavailable.")
            return self._delta_executor.add_masked_delta(
                original_output,
                inputs,
                preparation=self._executables[group_index].preparation,
                multiplier=multiplier,
            )
        active_rank_batches = tuple(
            (
                rank_batch,
                tuple(
                    index
                    for index in rank_batch.indices
                    if multipliers[index] is not None
                ),
            )
            for rank_batch in self._rank_batches
        )
        active_rank_batches = tuple(value for value in active_rank_batches if value[1])
        if len(active_rank_batches) == 1:
            rank_batch, batch_indices = active_rank_batches[0]
            if batch_indices == active_indices:
                active_multipliers = tuple(
                    multiplier
                    for index in batch_indices
                    if (multiplier := multipliers[index]) is not None
                )
                return self._delta_executor.add_compatible_deltas(
                    original_output,
                    inputs,
                    preparation=self._preparation_for_indices(
                        rank_batch,
                        batch_indices,
                    ),
                    multipliers=active_multipliers,
                )
        group_outputs: list[torch.Tensor | None] = [None] * len(self.groups)
        for rank_batch in self._rank_batches:
            active_indices = tuple(
                index for index in rank_batch.indices if multipliers[index] is not None
            )
            if not active_indices:
                continue
            if len(active_indices) == 1:
                group_index = active_indices[0]
                multiplier = multipliers[group_index]
                if multiplier is None:
                    raise AssertionError(
                        "Active Anima LoRA multiplier became unavailable."
                    )
                group_outputs[group_index] = self._delta_executor.masked_delta(
                    inputs,
                    preparation=self._executables[group_index].preparation,
                    multiplier=multiplier,
                )
                continue
            active_multipliers = tuple(
                multiplier
                for index in active_indices
                if (multiplier := multipliers[index]) is not None
            )
            compatible_deltas = self._delta_executor.compatible_deltas(
                inputs,
                preparation=self._preparation_for_indices(
                    rank_batch,
                    active_indices,
                ),
                multipliers=active_multipliers,
            )
            for local_index, group_index in enumerate(active_indices):
                group_outputs[group_index] = compatible_deltas[local_index]
        active_outputs = tuple(
            group_output for group_output in group_outputs if group_output is not None
        )
        for group_output in active_outputs:
            if group_output.shape != original_output.shape:
                raise ValueError(
                    "Anima LoRA composition delta does not match original output."
                )
        return self._delta_executor.accumulate_ordered(
            original_output,
            active_outputs,
        )

    def projection_batch_request(
        self,
        inputs: torch.Tensor,
        resolution: RegionalLoraScheduleResolution,
    ) -> AnimaProjectionBatchRequest | None:
        """Expose one active single-group contract to a shared-input leader."""

        if len(self._executables) != 1:
            return None
        executable = self._executables[0]
        multiplier = self._combined_multiplier(
            executable.group,
            inputs,
            resolution,
        )
        if multiplier is None:
            return None
        return AnimaProjectionBatchRequest(executable.preparation, multiplier)

    def clear_device_cache(self) -> None:
        """Release every prepared tensor owned directly by this linear patch."""

        for executable in self._executables:
            executable.preparation.clear()
        for rank_batch in self._rank_batches:
            rank_batch.preparation.clear()
        for preparation in self._active_batch_preparations.values():
            preparation.clear()
        self._active_batch_preparations.clear()

    def _forward_single_group(
        self,
        inputs: torch.Tensor,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        resolution: RegionalLoraScheduleResolution,
    ) -> Any:
        """Execute one group without multi-group planning allocations."""

        executable = self._executables[0]
        multiplier = self._combined_multiplier(
            executable.group,
            inputs,
            resolution,
        )
        original_output = self._backing.module(inputs, *args, **kwargs)
        if not isinstance(original_output, torch.Tensor):
            raise TypeError("Anima LoRA composition original must return a tensor.")
        if multiplier is None:
            return original_output
        return self._delta_executor.add_masked_delta(
            original_output,
            inputs,
            preparation=executable.preparation,
            multiplier=multiplier,
        )

    def _combined_multiplier(
        self,
        group: AnimaRegionalLoraTargetGroup,
        inputs: torch.Tensor,
        resolution: RegionalLoraScheduleResolution,
    ) -> torch.Tensor | None:
        """Combine contiguous repeated uses in declared floating addition order."""

        return self._weight_resolver.combined_multiplier(group, inputs, resolution)

    def _preparation_for_indices(
        self,
        rank_batch: _ExecutableRankBatch,
        active_indices: tuple[int, ...],
    ) -> RegionalLoraCompatibleBatchPreparation:
        """Reuse full batches and cache exact active subsets without dense patches."""

        if active_indices == rank_batch.indices:
            return rank_batch.preparation
        existing = self._active_batch_preparations.get(active_indices)
        if existing is not None:
            return existing
        preparation = RegionalLoraCompatibleBatchPreparation(
            tuple(self._executables[index].preparation for index in active_indices)
        )
        self._active_batch_preparations[active_indices] = preparation
        return preparation

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        """Require floating inputs matching the shared target feature contract."""

        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("Anima LoRA composition input must be a floating tensor.")
        expected_features = self.groups[0].target.adapter.input_features
        if inputs.ndim < 1 or int(inputs.shape[-1]) != expected_features:
            raise ValueError(
                "Anima LoRA composition input feature dimension is invalid."
            )

    @staticmethod
    def _build_rank_batches(
        executables: tuple[_ExecutableTargetGroup, ...],
    ) -> tuple[_ExecutableRankBatch, ...]:
        """Group target indices by exact rank without changing addition order."""

        by_rank: dict[int, list[int]] = {}
        for index, executable in enumerate(executables):
            by_rank.setdefault(executable.group.target.adapter.rank, []).append(index)
        return tuple(
            _ExecutableRankBatch(
                tuple(indices),
                RegionalLoraCompatibleBatchPreparation(
                    tuple(executables[index].preparation for index in indices)
                ),
            )
            for indices in by_rank.values()
        )
