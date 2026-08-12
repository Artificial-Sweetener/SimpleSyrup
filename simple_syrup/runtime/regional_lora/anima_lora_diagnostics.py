# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build Anima-only regional LoRA diagnostics from execution authorities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..regional_lora_schedule_resolution import RegionalLoraScheduleResolution
from .anima_composition import AnimaRegionalLoraComposition
from .anima_diagnostic_values import (
    AnimaAdapterUseDiagnostics,
)
from .anima_module_surface import AnimaModuleSurface


@dataclass(frozen=True, slots=True)
class AnimaLoraWorkDiagnostics:
    """Retain only low-rank work values owned by Anima LoRA execution."""

    low_rank_adapter_multiplier: float
    active_adapter_uses: int
    active_target_count: int
    target_use_count: int
    deduplicated_target_group_count: int
    compatible_projection_batch_count: int
    deduplicated_target_uses: int


@dataclass(frozen=True, slots=True)
class AnimaLoraExecutionDiagnostics:
    """Retain Anima adapter uses, cache cardinality, and low-rank work."""

    adapter_uses: tuple[AnimaAdapterUseDiagnostics, ...]
    prepared_cache_entries: int
    work: AnimaLoraWorkDiagnostics


class AnimaLoraDiagnosticsBuilder:
    """Derive LoRA diagnostics without mask, layout, CFG, or logging policy."""

    def __init__(
        self,
        surface: AnimaModuleSurface,
        composition: AnimaRegionalLoraComposition,
    ) -> None:
        """Precompute static adapter activity and target groups once."""

        if not isinstance(surface, AnimaModuleSurface):
            raise TypeError("Anima LoRA diagnostics require a module surface.")
        if not isinstance(composition, AnimaRegionalLoraComposition):
            raise TypeError("Anima LoRA diagnostics require a composition.")
        self._composition = composition
        self._static_activities = tuple(
            composition.static_activity(execution)
            for execution in composition.executions
        )
        self._groups_by_target = tuple(
            composition.groups_for_target(target.target_name)
            for target in surface.lora_targets
        )

    def build(
        self,
        resolution: RegionalLoraScheduleResolution,
    ) -> AnimaLoraExecutionDiagnostics:
        """Build one schedule-specific immutable adapter snapshot."""

        if not isinstance(resolution, RegionalLoraScheduleResolution):
            raise TypeError("Anima LoRA diagnostics require schedule resolution.")
        return AnimaLoraExecutionDiagnostics(
            adapter_uses=self._adapter_use_diagnostics(resolution),
            prepared_cache_entries=self._prepared_cache_entries(),
            work=self._work_estimates(resolution),
        )

    def _adapter_use_diagnostics(
        self,
        resolution: RegionalLoraScheduleResolution,
    ) -> tuple[AnimaAdapterUseDiagnostics, ...]:
        """Describe exact order and ownership through hashed source identities."""

        values: list[AnimaAdapterUseDiagnostics] = []
        for execution, activity in zip(
            self._composition.executions,
            self._static_activities,
            strict=True,
        ):
            effective_strength = resolution.effective_strength(
                execution.adapter_plan.composition_index
            )
            schedule_zero = activity.active and effective_strength == 0.0
            identity = execution.adapter_plan.adapter_identity.value
            token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            values.append(
                AnimaAdapterUseDiagnostics(
                    composition_index=execution.adapter_plan.composition_index,
                    adapter_token=token,
                    region_index=execution.adapter_plan.region_index,
                    branch=execution.adapter_plan.branch.value,
                    target_count=len(execution.admission.targets),
                    effective_strength=self._rounded(effective_strength),
                    active=activity.active and not schedule_zero,
                    pruning_reason=(
                        "schedule_zero"
                        if schedule_zero
                        else (
                            None
                            if activity.pruning_reason is None
                            else activity.pruning_reason.value
                        )
                    ),
                )
            )
        return tuple(values)

    def _work_estimates(
        self,
        resolution: RegionalLoraScheduleResolution,
    ) -> AnimaLoraWorkDiagnostics:
        """Estimate cardinality from authoritative composition target groups."""

        active = tuple(
            tuple(
                group
                for group in groups
                if any(
                    resolution.effective_strength(
                        use.execution.adapter_plan.composition_index
                    )
                    != 0.0
                    for use in group.uses
                )
            )
            for groups in self._groups_by_target
        )
        active = tuple(groups for groups in active if groups)
        target_use_count = sum(
            resolution.effective_strength(use.execution.adapter_plan.composition_index)
            != 0.0
            for groups in active
            for group in groups
            for use in group.uses
        )
        group_count = sum(len(groups) for groups in active)
        active_indices = {
            use.execution.adapter_plan.composition_index
            for groups in active
            for group in groups
            for use in group.uses
            if resolution.effective_strength(
                use.execution.adapter_plan.composition_index
            )
            != 0.0
        }
        compatible_batches = sum(
            len({group.target.adapter.rank for group in groups}) for groups in active
        )
        active_target_count = len(active)
        multiplier = (
            0.0 if active_target_count == 0 else group_count / active_target_count
        )
        return AnimaLoraWorkDiagnostics(
            low_rank_adapter_multiplier=self._rounded(multiplier),
            active_adapter_uses=len(active_indices),
            active_target_count=active_target_count,
            target_use_count=target_use_count,
            deduplicated_target_group_count=group_count,
            compatible_projection_batch_count=compatible_batches,
            deduplicated_target_uses=target_use_count - group_count,
        )

    def _prepared_cache_entries(self) -> int:
        """Sum unique execution-cache sizes without exposing cache keys."""

        caches = {
            id(execution.cache): execution.cache
            for execution in self._composition.executions
        }
        return sum(cache.size for cache in caches.values())

    @staticmethod
    def _rounded(value: float) -> float:
        """Stabilize diagnostic floats without affecting execution tensors."""

        return round(value, 6)
