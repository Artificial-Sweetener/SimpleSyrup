# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute one base Linear plus exact compact regional low-rank deltas."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from .active_support import (
    REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER,
    RegionalLoraActiveSupportResolver,
)
from .host_linear_parameters import RegionalHostLinearParameterProvider
from .linear_execution_plan import RegionalLinearExecutionPlan
from .linear_performance_diagnostics import (
    REGIONAL_LINEAR_PERFORMANCE_DIAGNOSTICS,
    RegionalLinearPerformanceDiagnostics,
)
from .mapped_linear_projection import (
    REGIONAL_MAPPED_LINEAR_PROJECTOR,
    RegionalMappedLinearProjector,
)


@dataclass(frozen=True, slots=True)
class _ProjectedRegionalLinearDelta:
    """Retain one exact unique delta and its support-aligned group values."""

    delta: torch.Tensor
    indices: torch.Tensor | None
    multiplier_values: tuple[torch.Tensor, ...]
    group_indices: tuple[int, ...]


class RegionalPartitionedLinearExecutor:
    """Own base-once execution and exact sparse repeated-delta reuse."""

    def __init__(
        self,
        support_resolver: RegionalLoraActiveSupportResolver = (
            REGIONAL_LORA_ACTIVE_SUPPORT_RESOLVER
        ),
        diagnostics: RegionalLinearPerformanceDiagnostics = (
            REGIONAL_LINEAR_PERFORMANCE_DIAGNOSTICS
        ),
        mapped_projector: RegionalMappedLinearProjector = (
            REGIONAL_MAPPED_LINEAR_PROJECTOR
        ),
    ) -> None:
        """Retain focused active-support and aggregate diagnostics owners."""

        if not isinstance(support_resolver, RegionalLoraActiveSupportResolver):
            raise TypeError("Partitioned Linear support resolver is invalid.")
        if not isinstance(diagnostics, RegionalLinearPerformanceDiagnostics):
            raise TypeError("Partitioned Linear diagnostics are invalid.")
        if not isinstance(mapped_projector, RegionalMappedLinearProjector):
            raise TypeError("Partitioned Linear mapped projector is invalid.")
        self._support_resolver = support_resolver
        self._diagnostics = diagnostics
        self._mapped_projector = mapped_projector

    def execute(
        self,
        inputs: torch.Tensor,
        *,
        plan: RegionalLinearExecutionPlan,
        multipliers: tuple[torch.Tensor | None, ...],
        parameter_provider: RegionalHostLinearParameterProvider | None,
    ) -> torch.Tensor | None:
        """Return base plus every active group or decline an unsupported host call."""

        if parameter_provider is None or len(multipliers) != len(plan.groups):
            self._diagnostics.record_decline("unsupported-linear-contract")
            return None
        leading_shape = tuple(int(size) for size in inputs.shape[:-1])
        flattened = inputs.reshape(-1, int(inputs.shape[-1]))
        if not flattened.is_contiguous():
            self._diagnostics.record_decline("noncontiguous-input")
            return None
        active = tuple(
            index
            for index, multiplier in enumerate(multipliers)
            if multiplier is not None
        )
        with parameter_provider.acquire(inputs) as parameters:
            if (
                parameters.weight.device != inputs.device
                or parameters.weight.dtype != inputs.dtype
                or not parameters.weight.is_contiguous()
            ):
                self._diagnostics.record_decline("unsupported-host-parameters")
                return None
            result = functional.linear(flattened, parameters.weight, parameters.bias)
            if not active:
                self._diagnostics.record_partitioned((), ())
                return result.reshape(*leading_shape, int(result.shape[-1]))
            if self._mapped_projector.add(
                result,
                flattened,
                plan=plan.mapped_projection,
                multipliers=multipliers,
                leading_shape=leading_shape,
            ):
                self._diagnostics.record_mapped(active)
                return result.reshape(*leading_shape, int(result.shape[-1]))
            projected = self._project_unique_deltas(
                flattened,
                plan=plan,
                multipliers=multipliers,
                active=active,
                leading_shape=leading_shape,
            )
            by_group = {
                group_index: (projection, local_index)
                for projection in projected
                for local_index, group_index in enumerate(projection.group_indices)
            }
            for group_index in active:
                projection, local_index = by_group[group_index]
                addition = projection.delta * projection.multiplier_values[
                    local_index
                ].unsqueeze(-1)
                if projection.indices is None:
                    result = result + addition
                else:
                    result.index_add_(0, projection.indices, addition)
        self._diagnostics.record_partitioned(
            active,
            tuple(
                int(flattened.shape[0])
                if projection.indices is None
                else int(projection.indices.numel())
                for projection in projected
            ),
        )
        return result.reshape(*leading_shape, int(result.shape[-1]))

    def _project_unique_deltas(
        self,
        inputs: torch.Tensor,
        *,
        plan: RegionalLinearExecutionPlan,
        multipliers: tuple[torch.Tensor | None, ...],
        active: tuple[int, ...],
        leading_shape: tuple[int, ...],
    ) -> tuple[_ProjectedRegionalLinearDelta, ...]:
        """Project each exact A/B tensor pair once on its union active rows."""

        groups: dict[tuple[int, int], list[int]] = {}
        for group_index in active:
            target = plan.groups[group_index].preparation.target
            groups.setdefault((id(target.down), id(target.up)), []).append(group_index)
        projected: list[_ProjectedRegionalLinearDelta] = []
        for group_indices_list in groups.values():
            group_indices = tuple(group_indices_list)
            selected_multipliers = tuple(
                multiplier
                for group_index in group_indices
                if (multiplier := multipliers[group_index]) is not None
            )
            if len(selected_multipliers) != len(group_indices):
                raise AssertionError("Partitioned Linear multiplier disappeared.")
            support = self._support_resolver.resolve(
                selected_multipliers,
                leading_shape=leading_shape,
            )
            if support is None:
                indices = None
                selected_inputs = inputs
                multiplier_values = tuple(
                    multiplier.expand(leading_shape).reshape(-1)
                    for multiplier in selected_multipliers
                )
            else:
                indices = support.indices
                selected_inputs = torch.index_select(inputs, 0, indices)
                multiplier_values = support.multiplier_values
            preparation = plan.groups[group_indices[0]].preparation
            weights = preparation.weights(device=inputs.device, dtype=inputs.dtype)
            delta = functional.linear(
                functional.linear(selected_inputs, weights.down),
                weights.up,
            )
            projected.append(
                _ProjectedRegionalLinearDelta(
                    delta,
                    indices,
                    multiplier_values,
                    group_indices,
                )
            )
        return tuple(projected)


REGIONAL_PARTITIONED_LINEAR_EXECUTOR = RegionalPartitionedLinearExecutor()
