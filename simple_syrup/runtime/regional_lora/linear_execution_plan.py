# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build immutable generic Linear execution plans from bound Comfy operations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import comfy.model_patcher
import torch
from comfy.weight_adapter.lora import LoRAAdapter

from ...domain.regional_lora_plan import RegionalLoraAdapterIdentity
from .execution_cache import ModelCloneLineage, RegionalLoraExecutionCache
from .preparation import (
    RegionalLoraCompatibleBatchPreparation,
    RegionalLoraTargetPreparation,
)
from .standard_adapter import StandardLoraTarget
from .target_binding import (
    BoundRegionalLoraModuleClass,
    BoundRegionalLoraOperation,
    BoundRegionalLoraSpatialCapability,
)


@dataclass(frozen=True, slots=True)
class RegionalLinearOperationKey:
    """Identify one exact adapter tensor pair at one bound parameter target."""

    adapter_identity: RegionalLoraAdapterIdentity
    parameter_path: str
    down_identity: int
    up_identity: int


@dataclass(frozen=True, slots=True)
class RegionalLinearTargetUse:
    """Retain one ordered regional use and its immutable prepared-weight owner."""

    composition_index: int
    region_index: int
    operation_key: RegionalLinearOperationKey
    preparation: RegionalLoraTargetPreparation
    base_strength: float

    def __post_init__(self) -> None:
        """Require canonical indices, key, preparation, and finite base strength."""

        for name, value in (
            ("composition_index", self.composition_index),
            ("region_index", self.region_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Regional Linear {name} must be an integer.")
            if value < 0:
                raise ValueError(f"Regional Linear {name} must be non-negative.")
        if not isinstance(self.operation_key, RegionalLinearOperationKey):
            raise TypeError("Regional Linear use requires an operation key.")
        if not isinstance(self.preparation, RegionalLoraTargetPreparation):
            raise TypeError("Regional Linear use requires target preparation.")
        if not isinstance(self.base_strength, float) or not math.isfinite(
            self.base_strength
        ):
            raise TypeError("Regional Linear base strength must be finite.")


@dataclass(frozen=True, slots=True)
class RegionalLinearTargetGroup:
    """Combine only contiguous repeated uses of one exact adapter target."""

    uses: tuple[RegionalLinearTargetUse, ...]

    def __post_init__(self) -> None:
        """Require one nonempty contiguous exact-operation group."""

        if not isinstance(self.uses, tuple) or not self.uses:
            raise ValueError("Regional Linear target group cannot be empty.")
        first = self.uses[0]
        if any(use.operation_key != first.operation_key for use in self.uses):
            raise ValueError("Regional Linear group uses must share one operation.")

    @property
    def preparation(self) -> RegionalLoraTargetPreparation:
        """Return the shared cache-compatible target preparation."""

        return self.uses[0].preparation


@dataclass(slots=True)
class RegionalLinearExecutionPlan:
    """Retain declared-order groups and reusable compatible-rank preparations."""

    uses: tuple[RegionalLinearTargetUse, ...]
    groups: tuple[RegionalLinearTargetGroup, ...] = field(init=False)
    rank_batches: tuple[
        tuple[tuple[int, ...], RegionalLoraCompatibleBatchPreparation], ...
    ] = field(init=False)
    _active_batches: dict[tuple[int, ...], RegionalLoraCompatibleBatchPreparation] = (
        field(default_factory=dict, init=False, repr=False)
    )

    def __post_init__(self) -> None:
        """Canonicalize declared order once without changing adapter semantics."""

        if not isinstance(self.uses, tuple) or not self.uses:
            raise ValueError("Regional Linear execution plan cannot be empty.")
        if any(not isinstance(use, RegionalLinearTargetUse) for use in self.uses):
            raise TypeError("Regional Linear execution plan contains an invalid use.")
        composition = tuple(use.composition_index for use in self.uses)
        if composition != tuple(sorted(composition)):
            raise ValueError("Regional Linear uses must follow composition order.")
        groups = _contiguous_groups(self.uses)
        object.__setattr__(self, "groups", groups)
        by_contract: dict[tuple[int, int, int], list[int]] = {}
        for index, group in enumerate(groups):
            target = group.preparation.target
            contract = (target.rank, target.input_features, target.output_features)
            by_contract.setdefault(contract, []).append(index)
        batches = tuple(
            (
                tuple(indices),
                RegionalLoraCompatibleBatchPreparation(
                    tuple(groups[index].preparation for index in indices)
                ),
            )
            for indices in by_contract.values()
        )
        object.__setattr__(self, "rank_batches", batches)

    def clear(self) -> None:
        """Release locally retained prepared weights and compatible batches."""

        for group in self.groups:
            group.preparation.clear()
        for _indices, preparation in self.rank_batches:
            preparation.clear()
        for preparation in self._active_batches.values():
            preparation.clear()
        self._active_batches.clear()

    def preparation_for(
        self,
        *,
        selected_indices: tuple[int, ...],
        complete_indices: tuple[int, ...],
        complete: RegionalLoraCompatibleBatchPreparation,
    ) -> RegionalLoraCompatibleBatchPreparation:
        """Reuse one complete batch or cache an exact active compatible subset."""

        if selected_indices == complete_indices:
            return complete
        existing = self._active_batches.get(selected_indices)
        if existing is not None:
            return existing
        preparation = RegionalLoraCompatibleBatchPreparation(
            tuple(self.groups[index].preparation for index in selected_indices)
        )
        self._active_batches[selected_indices] = preparation
        return preparation


class RegionalLinearExecutionPlanFactory:
    """Adapt exact U2/U3/U4 evidence to existing neutral preparation owners."""

    def build(
        self,
        bindings: tuple[BoundRegionalLoraOperation, ...],
        *,
        model: comfy.model_patcher.ModelPatcher,
        cache: RegionalLoraExecutionCache,
    ) -> RegionalLinearExecutionPlan:
        """Return one plan without copying, moving, or mutating adapter tensors."""

        if not isinstance(bindings, tuple) or not bindings:
            raise ValueError("Regional Linear plan requires bound operations.")
        if not isinstance(model, comfy.model_patcher.ModelPatcher):
            raise TypeError("Regional Linear plan requires a Comfy ModelPatcher.")
        if not isinstance(cache, RegionalLoraExecutionCache):
            raise TypeError("Regional Linear plan requires an execution cache.")
        lineage = ModelCloneLineage.from_model(model)
        return RegionalLinearExecutionPlan(
            tuple(
                self._use(binding, lineage=lineage, cache=cache) for binding in bindings
            )
        )

    @staticmethod
    def _use(
        binding: BoundRegionalLoraOperation,
        *,
        lineage: ModelCloneLineage,
        cache: RegionalLoraExecutionCache,
    ) -> RegionalLinearTargetUse:
        """Adapt one exact ordinary matrix pair to a neutral preparation."""

        if not isinstance(binding, BoundRegionalLoraOperation) or not binding.bound:
            raise ValueError("Regional Linear execution requires a bound operation.")
        if binding.module_class is not BoundRegionalLoraModuleClass.LINEAR:
            raise ValueError("Regional Linear execution requires a Linear target.")
        if binding.spatial_capability is not (
            BoundRegionalLoraSpatialCapability.CONSUMER_SPATIALIZED
        ):
            raise ValueError(
                "Regional Linear execution requires proven consumer spatialization."
            )
        operation = binding.normalized_target.operation
        if not isinstance(operation, LoRAAdapter):
            raise TypeError("Regional Linear normalized operation must be LoRAAdapter.")
        up, down, _alpha, middle, dora, reshape = operation.weights
        if (
            not isinstance(down, torch.Tensor)
            or not isinstance(up, torch.Tensor)
            or down.ndim != 2
            or up.ndim != 2
            or middle is not None
            or dora is not None
            or reshape is not None
        ):
            raise ValueError("Regional Linear operation must be one matrix LoRA pair.")
        descriptor = binding.descriptor
        if descriptor.rank is None or descriptor.intrinsic_scale is None:
            raise ValueError("Regional Linear descriptor lacks executable metadata.")
        target = StandardLoraTarget(
            target=descriptor.target.parameter_path,
            down=down,
            up=up,
            rank=descriptor.rank,
            input_features=int(down.shape[1]),
            output_features=int(up.shape[0]),
            intrinsic_scale=descriptor.intrinsic_scale,
        )
        adapter_identity = descriptor.adapter.adapter_identity
        return RegionalLinearTargetUse(
            descriptor.adapter.composition_index,
            descriptor.adapter.region_index,
            RegionalLinearOperationKey(
                adapter_identity,
                descriptor.target.parameter_path,
                id(down),
                id(up),
            ),
            RegionalLoraTargetPreparation(
                adapter_identity,
                lineage,
                target,
                cache,
            ),
            float(descriptor.adapter.model_strength * descriptor.intrinsic_scale),
        )


def _contiguous_groups(
    uses: tuple[RegionalLinearTargetUse, ...],
) -> tuple[RegionalLinearTargetGroup, ...]:
    """Combine adjacent exact operations while preserving all other boundaries."""

    groups: list[list[RegionalLinearTargetUse]] = []
    for use in uses:
        if groups and groups[-1][-1].operation_key == use.operation_key:
            groups[-1].append(use)
        else:
            groups.append([use])
    return tuple(RegionalLinearTargetGroup(tuple(group)) for group in groups)


REGIONAL_LINEAR_EXECUTION_PLAN_FACTORY = RegionalLinearExecutionPlanFactory()
