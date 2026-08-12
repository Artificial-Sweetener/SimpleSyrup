# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coordinate exact LoRA projection batches across shared Anima inputs."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

import torch

from ..regional_lora_schedule_resolution import RegionalLoraScheduleResolution
from .compatible_rank_projection import RegionalLoraCompatibleRankProjection
from .delta_execution import (
    REGIONAL_LORA_DELTA_EXECUTOR,
    RegionalLoraDeltaExecutor,
)
from .preparation import (
    RegionalLoraCompatibleBatchPreparation,
    RegionalLoraTargetPreparation,
)


@dataclass(frozen=True, slots=True)
class AnimaProjectionBatchRequest:
    """Expose one active target preparation and its exact pointwise multiplier."""

    preparation: RegionalLoraTargetPreparation
    multiplier: torch.Tensor


@dataclass(frozen=True, slots=True)
class AnimaProjectionBatchExecution:
    """Bind one target index to a shared active rank-space projection."""

    projection: RegionalLoraCompatibleRankProjection
    target_index: int


class AnimaProjectionBatchParticipant(Protocol):
    """Describe one linear patch that can join shared-input projection work."""

    def projection_batch_request(
        self,
        inputs: torch.Tensor,
        resolution: RegionalLoraScheduleResolution,
    ) -> AnimaProjectionBatchRequest | None:
        """Return one active single-group request or decline coordination."""


@dataclass(frozen=True, slots=True)
class _ProjectionBatchGroup:
    """Retain one leader-first immutable shared-input participant group."""

    group_id: str
    participants: tuple[AnimaProjectionBatchParticipant, ...]


@dataclass(slots=True)
class _ProjectionBatchSlot:
    """Retain follower executions for one exact leader input invocation."""

    group: _ProjectionBatchGroup
    inputs: torch.Tensor
    executions: dict[int, AnimaProjectionBatchExecution]


class AnimaProjectionBatchRegistry:
    """Own group registration and task-local leader/follower delta reuse."""

    def __init__(
        self,
        delta_executor: RegionalLoraDeltaExecutor = REGIONAL_LORA_DELTA_EXECUTOR,
    ) -> None:
        """Retain one delta executor and initialize an empty group inventory."""

        if not isinstance(delta_executor, RegionalLoraDeltaExecutor):
            raise TypeError("Anima projection batching requires a delta executor.")
        self._delta_executor = delta_executor
        self._groups: dict[int, _ProjectionBatchGroup] = {}
        self._preparations: dict[str, RegionalLoraCompatibleBatchPreparation] = {}
        self._slot: ContextVar[_ProjectionBatchSlot | None] = ContextVar(
            "simple_syrup_anima_projection_batch",
            default=None,
        )

    def register(
        self,
        group_id: str,
        participants: tuple[AnimaProjectionBatchParticipant, ...],
    ) -> None:
        """Register one leader-first group exactly once during mutation assembly."""

        if not group_id or len(participants) < 2:
            raise ValueError(
                "Anima projection batch requires a named multi-target group."
            )
        identities = tuple(id(participant) for participant in participants)
        if len(identities) != len(set(identities)):
            raise ValueError("Anima projection batch participants must be distinct.")
        if any(identity in self._groups for identity in identities):
            raise ValueError(
                "Anima projection batch participant is already registered."
            )
        group = _ProjectionBatchGroup(group_id, participants)
        for identity in identities:
            self._groups[identity] = group

    def resolve_execution(
        self,
        participant: AnimaProjectionBatchParticipant,
        inputs: torch.Tensor,
        resolution: RegionalLoraScheduleResolution,
    ) -> AnimaProjectionBatchExecution | None:
        """Return one shared-rank execution or decline without changing behavior."""

        group = self._groups.get(id(participant))
        if group is None:
            return None
        slot = self._slot.get()
        participant_id = id(participant)
        if (
            slot is not None
            and slot.group is group
            and slot.inputs is inputs
            and participant_id in slot.executions
        ):
            return slot.executions.pop(participant_id)
        if participant is not group.participants[0]:
            return None
        requests = tuple(
            member.projection_batch_request(inputs, resolution)
            for member in group.participants
        )
        if any(request is None for request in requests):
            return None
        active_requests = tuple(request for request in requests if request is not None)
        preparation = self._preparations.get(group.group_id)
        if preparation is None:
            preparation = RegionalLoraCompatibleBatchPreparation(
                tuple(request.preparation for request in active_requests)
            )
            self._preparations[group.group_id] = preparation
        projection = self._delta_executor.prepare_compatible_rank_projection(
            inputs,
            preparation=preparation,
            multipliers=tuple(request.multiplier for request in active_requests),
        )
        by_participant = {
            id(member): AnimaProjectionBatchExecution(projection, index)
            for index, member in enumerate(group.participants)
        }
        leader_output = by_participant.pop(participant_id)
        self._slot.set(_ProjectionBatchSlot(group, inputs, by_participant))
        return leader_output

    def accumulate(
        self,
        original_output: torch.Tensor,
        execution: AnimaProjectionBatchExecution,
    ) -> torch.Tensor:
        """Fuse one coordinated target directly into its original output."""

        if not isinstance(execution, AnimaProjectionBatchExecution):
            raise TypeError("Anima projection batching requires a typed execution.")
        return self._delta_executor.add_preprojected_target(
            original_output,
            execution.projection,
            execution.target_index,
        )

    def clear(self) -> None:
        """Release prepared batches and the current task's projection slot."""

        for preparation in self._preparations.values():
            preparation.clear()
        self._preparations.clear()
        self._slot.set(None)
