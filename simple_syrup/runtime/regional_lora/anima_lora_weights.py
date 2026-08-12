# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve exact branch and spatial weights for Anima regional LoRA uses."""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

from ...domain.regional_attention import RegionalAttentionBranch
from ...domain.regional_lora_plan import RegionalLoraBranch
from ..regional_lora_schedule_resolution import RegionalLoraScheduleResolution
from .anima_branch_batch import AnimaRegionalBranchInvocation
from .anima_composition import AnimaRegionalLoraTargetGroup
from .anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)
from .anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocationContext,
    AnimaRegionalLoraAdapterExecution,
)
from .anima_lora_weight_categories import (
    AnimaLoraWeightCategory,
    anima_lora_weight_category,
)
from .anima_query_masks import AnimaQueryMaskBatch
from .anima_targets import AnimaLoraTarget


@dataclass(slots=True)
class _AnimaLoraMultiplierCacheSlot:
    """Retain reusable multiplier tensors for one exact spatial invocation."""

    masks: AnimaQueryMaskBatch | None
    chunk_gates: dict[tuple[object, ...], torch.Tensor] = field(default_factory=dict)
    spatial: dict[tuple[object, ...], torch.Tensor] = field(default_factory=dict)
    branches: dict[tuple[object, ...], torch.Tensor] = field(default_factory=dict)
    weighted: dict[tuple[object, ...], torch.Tensor] = field(default_factory=dict)
    combined: dict[tuple[object, ...], torch.Tensor] = field(default_factory=dict)


class AnimaLoraWeightResolver:
    """Own family-specific branch gating and pointwise mask alignment."""

    def __init__(
        self,
        *,
        cross_attention_context: AnimaCrossAttentionInvocationContext,
        branch_context: AnimaLoraBranchInvocationContext,
        spatial_context: AnimaLoraSpatialInvocationContext,
    ) -> None:
        """Retain the three task-local execution-scope authorities."""

        self._cross_attention_context = cross_attention_context
        self._branch_context = branch_context
        self._spatial_context = spatial_context
        self._cache: ContextVar[_AnimaLoraMultiplierCacheSlot | None] = ContextVar(
            "simple_syrup_anima_lora_multiplier_cache",
            default=None,
        )

    def weighted_multiplier(
        self,
        target: AnimaLoraTarget,
        execution: AnimaRegionalLoraAdapterExecution,
        inputs: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """Return one normalized schedule-scaled multiplier with shared reuse."""

        if not math.isfinite(strength):
            raise ValueError("Anima LoRA multiplier strength must be finite.")
        effective_strength = strength * target.adapter.intrinsic_scale
        leading_shape = tuple(int(size) for size in inputs.shape[:-1])
        slot = self._slot()
        key = self._weighted_request_key(
            target,
            execution,
            inputs,
            leading_shape,
            effective_strength,
        )
        existing = slot.weighted.get(key)
        if existing is not None:
            return existing
        raw = self.multiplier(target, execution, inputs)
        normalized = raw.to(device=inputs.device, dtype=inputs.dtype)
        while normalized.ndim < len(leading_shape):
            normalized = normalized.unsqueeze(-1)
        if normalized.ndim != len(leading_shape) or any(
            observed not in (1, expected)
            for observed, expected in zip(
                normalized.shape,
                leading_shape,
                strict=True,
            )
        ):
            raise ValueError(
                "Anima LoRA composition multiplier cannot broadcast to its input."
            )
        weighted = (
            normalized if effective_strength == 1.0 else normalized * effective_strength
        )
        slot.weighted[key] = weighted
        return weighted

    def clear(self) -> None:
        """Release the current task's retained multiplier tensors."""

        self._cache.set(None)

    def combined_multiplier(
        self,
        group: AnimaRegionalLoraTargetGroup,
        inputs: torch.Tensor,
        resolution: RegionalLoraScheduleResolution,
    ) -> torch.Tensor | None:
        """Combine one repeated group once per exact task-local request."""

        if not isinstance(group, AnimaRegionalLoraTargetGroup):
            raise TypeError("Anima LoRA combined multiplier requires a target group.")
        if not isinstance(resolution, RegionalLoraScheduleResolution):
            raise TypeError("Anima LoRA combined multiplier requires schedule state.")
        active = tuple(
            (use, strength)
            for use in group.uses
            if (
                strength := resolution.effective_strength(
                    use.execution.adapter_plan.composition_index
                )
            )
            != 0.0
        )
        if not active:
            return None
        leading_shape = tuple(int(size) for size in inputs.shape[:-1])
        cache_key = tuple(
            self._weighted_request_key(
                use.target,
                use.execution,
                inputs,
                leading_shape,
                strength,
            )
            for use, strength in active
        )
        slot = self._slot()
        existing = slot.combined.get(cache_key)
        if existing is not None:
            return existing
        combined: torch.Tensor | None = None
        for use, strength in active:
            contribution = self.weighted_multiplier(
                use.target,
                use.execution,
                inputs,
                strength,
            )
            combined = contribution if combined is None else combined + contribution
        if combined is None:
            raise AssertionError("Active Anima LoRA group produced no multiplier.")
        slot.combined[cache_key] = combined
        return combined

    def _weighted_request_key(
        self,
        target: AnimaLoraTarget,
        execution: AnimaRegionalLoraAdapterExecution,
        inputs: torch.Tensor,
        leading_shape: tuple[int, ...],
        strength: float,
    ) -> tuple[object, ...]:
        """Identify one reusable validated multiplier before raw resolution."""

        category = anima_lora_weight_category(target.family)
        if category is AnimaLoraWeightCategory.CROSS_ATTENTION:
            invocation: object = self._cross_attention_context.current_or_none()
            if invocation is None:
                raise RuntimeError(
                    "Anima cross-attention LoRA target executed outside its "
                    "regional branch batch."
                )
        elif category is AnimaLoraWeightCategory.ADALN:
            invocation = self._branch_context.current_or_none()
            if invocation is None:
                raise RuntimeError(
                    "Anima AdaLN LoRA target executed outside its regional "
                    "branch batch."
                )
        else:
            spatial = self._spatial_context.require_current()
            invocation = id(spatial.masks)
        plan = execution.adapter_plan
        return (
            category.value,
            invocation,
            id(execution.attention.active_contexts),
            plan.region_index,
            plan.branch,
            leading_shape,
            inputs.device,
            inputs.dtype,
            strength,
        )

    def multiplier(
        self,
        target: AnimaLoraTarget,
        execution: AnimaRegionalLoraAdapterExecution,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return one branch-only or pointwise multiplier for an adapter use."""

        family = target.family
        category = anima_lora_weight_category(family)
        slot = self._slot()
        if category is AnimaLoraWeightCategory.CROSS_ATTENTION:
            invocation = self._cross_attention_context.current_or_none()
            if invocation is None:
                raise RuntimeError(
                    "Anima cross-attention LoRA target executed outside its "
                    "regional branch batch."
                )
            return self._branch_multiplier(invocation, execution, inputs, slot)
        if category is AnimaLoraWeightCategory.ADALN:
            branch_invocation = self._branch_context.current_or_none()
            if branch_invocation is None:
                raise RuntimeError(
                    "Anima AdaLN LoRA target executed outside its regional "
                    "branch batch."
                )
            return self._branch_multiplier(
                branch_invocation,
                execution,
                inputs,
                slot,
            )
        spatial = self._spatial_context.require_current().masks.masks
        spatial_key = (
            "grid",
            id(execution.attention.active_contexts),
            execution.adapter_plan.region_index,
            execution.adapter_plan.branch,
            inputs.device,
            inputs.dtype,
        )
        cached_spatial = slot.spatial.get(spatial_key)
        if cached_spatial is None:
            region_masks = spatial[execution.adapter_plan.region_index]
            chunk_gate = self._chunk_gate(execution, inputs.device, inputs.dtype, slot)
            cached_spatial = region_masks * chunk_gate[:, None, None, None]
            slot.spatial[spatial_key] = cached_spatial
        if category is AnimaLoraWeightCategory.SELF_ATTENTION:
            if inputs.ndim != 3:
                raise ValueError("Anima self-attention LoRA input must use BxQxD.")
            flattened_key = ("flattened", *spatial_key[1:])
            multiplier = slot.spatial.get(flattened_key)
            if multiplier is None:
                multiplier = cached_spatial.flatten(start_dim=1)
                slot.spatial[flattened_key] = multiplier
            if tuple(multiplier.shape) != tuple(inputs.shape[:2]):
                raise ValueError(
                    "Anima self-attention LoRA masks do not match the query input."
                )
            return multiplier
        if category is AnimaLoraWeightCategory.MLP:
            if inputs.ndim != 5:
                raise ValueError("Anima MLP LoRA input must use BxTxHxWxD.")
            if tuple(cached_spatial.shape) != tuple(inputs.shape[:-1]):
                raise ValueError(
                    "Anima MLP LoRA masks do not match the pointwise input."
                )
            return cached_spatial
        raise AssertionError(f"Unhandled Anima LoRA target family {family}")

    def _branch_multiplier(
        self,
        invocation: AnimaRegionalBranchInvocation,
        execution: AnimaRegionalLoraAdapterExecution,
        inputs: torch.Tensor,
        slot: _AnimaLoraMultiplierCacheSlot,
    ) -> torch.Tensor:
        """Gate one canonical regional branch and matching conditioning chunks."""

        if inputs.ndim < 1 or int(inputs.shape[0]) != invocation.packed_batch_size:
            raise ValueError("Anima LoRA branch input has an invalid expanded batch.")
        key = (
            invocation,
            id(execution.attention.active_contexts),
            execution.adapter_plan.region_index,
            execution.adapter_plan.branch,
            inputs.device,
            inputs.dtype,
        )
        existing = slot.branches.get(key)
        if existing is not None:
            return existing
        chunk_gate = self._chunk_gate(execution, inputs.device, inputs.dtype, slot)
        if int(chunk_gate.shape[0]) != invocation.source_batch_size:
            raise ValueError("Anima LoRA branch batch does not match aligned chunks.")
        source_indices = torch.tensor(
            invocation.source_batch_indices,
            device=inputs.device,
            dtype=torch.int64,
        )
        source_gate = chunk_gate.index_select(0, source_indices)
        region_gate = inputs.new_tensor(
            tuple(
                region_index == execution.adapter_plan.region_index
                for region_index in invocation.region_indices
            )
        )
        multiplier = source_gate * region_gate
        slot.branches[key] = multiplier
        return multiplier

    def _chunk_gate(
        self,
        execution: AnimaRegionalLoraAdapterExecution,
        device: torch.device,
        dtype: torch.dtype,
        slot: _AnimaLoraMultiplierCacheSlot,
    ) -> torch.Tensor:
        """Return one scalar gate per aligned base batch item for plan branch."""

        contexts = execution.attention.active_contexts
        key = (
            id(contexts),
            execution.adapter_plan.branch,
            device,
            dtype,
        )
        existing = slot.chunk_gates.get(key)
        if existing is not None:
            return existing
        gate = torch.zeros(
            (int(contexts.base_context.shape[0]),),
            device=device,
            dtype=dtype,
        )
        expected_branch = (
            RegionalAttentionBranch.POSITIVE
            if execution.adapter_plan.branch is RegionalLoraBranch.POSITIVE
            else RegionalAttentionBranch.NEGATIVE
        )
        for chunk in contexts.chunks:
            if chunk.branch is expected_branch:
                gate[chunk.batch_start : chunk.batch_stop] = 1.0
        slot.chunk_gates[key] = gate
        return gate

    def _slot(self) -> _AnimaLoraMultiplierCacheSlot:
        """Return one task-local cache bounded to the active mask batch."""

        spatial = self._spatial_context.current_or_none()
        masks = None if spatial is None else spatial.masks
        slot = self._cache.get()
        if slot is None or slot.masks is not masks:
            slot = _AnimaLoraMultiplierCacheSlot(masks)
            self._cache.set(slot)
        return slot
