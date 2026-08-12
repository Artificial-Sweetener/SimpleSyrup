# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own immutable Anima adapter execution state and task-local call scopes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ...domain.regional_lora_plan import RegionalLoraAdapterPlan
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_branch_batch import AnimaRegionalBranchInvocation
from .anima_query_masks import AnimaQueryMaskBatch
from .anima_targets import AnimaLoraAdmission
from .execution_cache import ModelCloneLineage, RegionalLoraExecutionCache


@dataclass(frozen=True, slots=True)
class AnimaRegionalLoraAdapterExecution:
    """Bind one active adapter use to admitted targets and spatial authorities."""

    adapter_plan: RegionalLoraAdapterPlan
    admission: AnimaLoraAdmission
    attention: AnimaRegionalAttentionExecution
    model_lineage: ModelCloneLineage
    cache: RegionalLoraExecutionCache

    def __post_init__(self) -> None:
        """Validate one complete static adapter execution without device work."""

        if not isinstance(self.adapter_plan, RegionalLoraAdapterPlan):
            raise TypeError("Anima regional LoRA adapter plan has an invalid type.")
        if not isinstance(self.admission, AnimaLoraAdmission):
            raise TypeError("Anima regional LoRA admission has an invalid type.")
        if not self.admission.targets:
            raise ValueError("Anima regional LoRA admission must contain targets.")
        if not isinstance(self.attention, AnimaRegionalAttentionExecution):
            raise TypeError("Anima regional LoRA attention state has an invalid type.")
        if not isinstance(self.model_lineage, ModelCloneLineage):
            raise TypeError("Anima regional LoRA model lineage has an invalid type.")
        if not isinstance(self.cache, RegionalLoraExecutionCache):
            raise TypeError("Anima regional LoRA cache has an invalid type.")
        if self.adapter_plan.region_index >= self.attention.mask_bank.region_count:
            raise ValueError("Anima regional LoRA region exceeds the mask bank.")
        names = tuple(target.adapter.target for target in self.admission.targets)
        if len(names) != len(set(names)):
            raise ValueError(
                "Anima regional LoRA admission contains duplicate targets."
            )


class AnimaLoraBranchInvocationContext:
    """Own task-local branch identity for exact AdaLN branch evaluation."""

    def __init__(self) -> None:
        """Create an empty task-local branch slot."""

        self._current: ContextVar[AnimaRegionalBranchInvocation | None] = ContextVar(
            "simple_syrup_anima_lora_branch_invocation", default=None
        )

    def current_or_none(self) -> AnimaRegionalBranchInvocation | None:
        """Return the current AdaLN branch batch when active."""

        return self._current.get()

    @contextmanager
    def activate(self, invocation: AnimaRegionalBranchInvocation) -> Iterator[None]:
        """Publish one branch batch for exactly one AdaLN module call."""

        token = self._current.set(invocation)
        try:
            yield
        finally:
            self._current.reset(token)


@dataclass(frozen=True, slots=True)
class AnimaLoraSpatialInvocation:
    """Retain exact active regional masks for pointwise block projections."""

    masks: AnimaQueryMaskBatch


class AnimaLoraSpatialInvocationContext:
    """Own task-local pointwise mask state during one Anima block call."""

    def __init__(self) -> None:
        """Create an empty task-local spatial slot."""

        self._current: ContextVar[AnimaLoraSpatialInvocation | None] = ContextVar(
            "simple_syrup_anima_lora_spatial_invocation", default=None
        )

    def require_current(self) -> AnimaLoraSpatialInvocation:
        """Return active masks or fail outside the regional block patch."""

        invocation = self.current_or_none()
        if invocation is None:
            raise RuntimeError(
                "Anima regional LoRA spatial masks are unavailable outside the "
                "regional block patch."
            )
        return invocation

    def current_or_none(self) -> AnimaLoraSpatialInvocation | None:
        """Return the active spatial invocation without requiring its presence."""

        return self._current.get()

    @contextmanager
    def activate(self, invocation: AnimaLoraSpatialInvocation) -> Iterator[None]:
        """Publish one mask batch for exactly one regional block execution."""

        token = self._current.set(invocation)
        try:
            yield
        finally:
            self._current.reset(token)


ANIMA_LORA_BRANCH_CONTEXT = AnimaLoraBranchInvocationContext()
ANIMA_LORA_SPATIAL_CONTEXT = AnimaLoraSpatialInvocationContext()
