# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plan ordered, statically active Anima regional LoRA target groups."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...domain.regional_attention import RegionalAttentionBranch
from ...domain.regional_lora_plan import RegionalLoraBranch
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_execution_scope import AnimaRegionalLoraAdapterExecution
from .anima_targets import AnimaLoraTarget


@dataclass(frozen=True, slots=True)
class AnimaRegionalLoraTargetUse:
    """Bind one ordered active adapter execution to one admitted target."""

    execution: AnimaRegionalLoraAdapterExecution
    target: AnimaLoraTarget


class AnimaRegionalLoraPruningReason(StrEnum):
    """Name each exact static reason an adapter use has no execution work."""

    ZERO_STRENGTH = "zero_strength"
    ABSENT_BRANCH = "absent_branch"
    ZERO_COVERAGE = "zero_coverage"
    SCHEDULE_ZERO = "schedule_zero"


@dataclass(frozen=True, slots=True)
class AnimaRegionalLoraStaticActivity:
    """Describe whether one adapter use has work and its exact prune reason."""

    active: bool
    pruning_reason: AnimaRegionalLoraPruningReason | None

    def __post_init__(self) -> None:
        """Require active state and pruning reason to agree."""

        if self.active == (self.pruning_reason is not None):
            raise ValueError(
                "Anima regional LoRA activity must name only inactive pruning."
            )


@dataclass(frozen=True, slots=True)
class AnimaRegionalLoraTargetGroup:
    """Combine one contiguous repeated immutable-adapter target run."""

    uses: tuple[AnimaRegionalLoraTargetUse, ...]

    def __post_init__(self) -> None:
        """Require one target and one shared admitted tensor pair."""

        if not self.uses:
            raise ValueError("Anima regional LoRA target group cannot be empty.")
        first = self.uses[0]
        if any(
            use.target.adapter.target != first.target.adapter.target
            for use in self.uses
        ):
            raise ValueError("Grouped Anima regional LoRA uses must share one target.")
        if any(
            use.execution.adapter_plan.adapter_identity
            != first.execution.adapter_plan.adapter_identity
            for use in self.uses
        ):
            raise ValueError(
                "Grouped Anima regional LoRA uses must share one adapter identity."
            )
        if any(use.target.adapter is not first.target.adapter for use in self.uses):
            raise ValueError(
                "Repeated Anima adapter uses must share one admitted target object."
            )

    @property
    def target(self) -> AnimaLoraTarget:
        """Return the shared admitted target."""

        return self.uses[0].target

    @property
    def first_composition_index(self) -> int:
        """Return the declared order position of this contiguous group."""

        return self.uses[0].execution.adapter_plan.composition_index


@dataclass(frozen=True, slots=True)
class AnimaRegionalLoraComposition:
    """Own ordered active adapter uses sharing one regional attention call."""

    executions: tuple[AnimaRegionalLoraAdapterExecution, ...]

    def __post_init__(self) -> None:
        """Require canonical order plus shared model and attention authorities."""

        if not isinstance(self.executions, tuple) or not self.executions:
            raise ValueError("Anima regional LoRA composition cannot be empty.")
        if any(
            not isinstance(execution, AnimaRegionalLoraAdapterExecution)
            for execution in self.executions
        ):
            raise TypeError("Anima regional LoRA composition has an invalid execution.")
        indices = tuple(
            execution.adapter_plan.composition_index for execution in self.executions
        )
        if indices != tuple(range(len(self.executions))):
            raise ValueError(
                "Anima regional LoRA executions must retain contiguous declared order."
            )
        attention = self.executions[0].attention
        model_lineage = self.executions[0].model_lineage
        if any(execution.attention is not attention for execution in self.executions):
            raise ValueError(
                "Anima regional LoRA executions must share one attention authority."
            )
        if any(
            execution.model_lineage != model_lineage for execution in self.executions
        ):
            raise ValueError(
                "Anima regional LoRA executions must share one model lineage."
            )

    @property
    def attention(self) -> AnimaRegionalAttentionExecution:
        """Return the shared regional attention authority."""

        return self.executions[0].attention

    def groups_for_target(
        self,
        target_name: str,
    ) -> tuple[AnimaRegionalLoraTargetGroup, ...]:
        """Return active uses grouped only across contiguous identical adapters."""

        ordered_uses = tuple(
            AnimaRegionalLoraTargetUse(execution, target)
            for execution in self.executions
            if self.static_activity(execution).active
            for target in execution.admission.targets
            if target.adapter.target == target_name
        )
        groups: list[list[AnimaRegionalLoraTargetUse]] = []
        for use in ordered_uses:
            if groups and self._can_combine(groups[-1][-1], use):
                groups[-1].append(use)
            else:
                groups.append([use])
        return tuple(AnimaRegionalLoraTargetGroup(tuple(group)) for group in groups)

    @staticmethod
    def _can_combine(
        previous: AnimaRegionalLoraTargetUse,
        current: AnimaRegionalLoraTargetUse,
    ) -> bool:
        """Combine only adjacent uses of the exact same immutable target pair."""

        return (
            previous.execution.adapter_plan.adapter_identity
            == current.execution.adapter_plan.adapter_identity
            and previous.target.adapter is current.target.adapter
        )

    @staticmethod
    def static_activity(
        execution: AnimaRegionalLoraAdapterExecution,
    ) -> AnimaRegionalLoraStaticActivity:
        """Classify exact zero strength, absent branch, and canonical coverage."""

        if float(execution.adapter_plan.model_strength) == 0.0:
            return AnimaRegionalLoraStaticActivity(
                False,
                AnimaRegionalLoraPruningReason.ZERO_STRENGTH,
            )
        expected_branch = (
            RegionalAttentionBranch.POSITIVE
            if execution.adapter_plan.branch is RegionalLoraBranch.POSITIVE
            else RegionalAttentionBranch.NEGATIVE
        )
        if not execution.attention.dynamic_contexts and not any(
            chunk.branch is expected_branch
            for chunk in execution.attention.contexts.chunks
        ):
            return AnimaRegionalLoraStaticActivity(
                False,
                AnimaRegionalLoraPruningReason.ABSENT_BRANCH,
            )
        masks = execution.attention.mask_bank.conditioning_masks
        region = masks[execution.adapter_plan.region_index]
        if not bool((region != 0).any().item()):
            return AnimaRegionalLoraStaticActivity(
                False,
                AnimaRegionalLoraPruningReason.ZERO_COVERAGE,
            )
        return AnimaRegionalLoraStaticActivity(True, None)
