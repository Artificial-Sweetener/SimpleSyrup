# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select exact active processed Attention Coupling entries for one sigma."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .conditioning_schedule_selection import (
    CONDITIONING_SCHEDULE_SELECTION_POLICY,
    ConditioningScheduleSelectionPolicy,
    normalize_conditioning_sigma,
)
from .processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from .regional_attention import (
    RegionalAttentionBranch,
    require_non_negative_regional_attention_index,
)


@dataclass(frozen=True, slots=True)
class ActiveProcessedRegionalAttentionChunk:
    """Bind one Comfy chunk UUID to its active base and regional entries."""

    chunk_index: int
    branch: RegionalAttentionBranch
    base_entry: ProcessedRegionalAttentionEntry
    regional_entries: tuple[
        tuple[ProcessedRegionalAttentionEntry, ...] | None,
        ...,
    ]

    def __post_init__(self) -> None:
        """Validate canonical immutable active selection state."""

        require_non_negative_regional_attention_index(
            self.chunk_index,
            name="chunk_index",
        )
        if not isinstance(self.branch, RegionalAttentionBranch):
            raise TypeError("Regional attention chunk branch has an invalid type.")
        if not isinstance(self.base_entry, ProcessedRegionalAttentionEntry):
            raise TypeError("Regional attention chunk base entry has an invalid type.")
        if not isinstance(self.regional_entries, tuple):
            raise TypeError("Regional attention chunk regions must be a tuple.")
        for entries in self.regional_entries:
            if entries is not None and (
                not isinstance(entries, tuple)
                or any(
                    not isinstance(entry, ProcessedRegionalAttentionEntry)
                    for entry in entries
                )
            ):
                raise TypeError(
                    "Regional attention chunk contains invalid active entries."
                )


class RegionalAttentionSelectionService:
    """Match Comfy chunk UUIDs to active base and regional entry banks."""

    def __init__(
        self,
        schedule_policy: ConditioningScheduleSelectionPolicy | None = None,
    ) -> None:
        """Retain the focused schedule policy collaborator."""

        self._schedule_policy = schedule_policy or ConditioningScheduleSelectionPolicy()

    def select_chunks(
        self,
        plan: ProcessedRegionalAttentionPlan,
        *,
        cond_or_uncond: object,
        conditioning_uuids: object,
        sigma: float,
    ) -> tuple[ActiveProcessedRegionalAttentionChunk, ...]:
        """Select exact active entries in Comfy's supplied UUID order."""

        if not isinstance(plan, ProcessedRegionalAttentionPlan):
            raise TypeError("Regional attention selection requires a processed plan.")
        selectors = _selectors(cond_or_uncond)
        identities = _conditioning_uuids(conditioning_uuids)
        if len(selectors) != len(identities):
            raise ValueError(
                "Regional attention selectors and UUIDs must have equal lengths."
            )
        current_sigma = normalize_conditioning_sigma(sigma)
        return tuple(
            self._select_chunk(
                plan,
                chunk_index=chunk_index,
                selector=selector,
                conditioning_uuid=conditioning_uuid,
                sigma=current_sigma,
            )
            for chunk_index, (selector, conditioning_uuid) in enumerate(
                zip(selectors, identities, strict=True)
            )
        )

    def _select_chunk(
        self,
        plan: ProcessedRegionalAttentionPlan,
        *,
        chunk_index: int,
        selector: object,
        conditioning_uuid: UUID,
        sigma: float,
    ) -> ActiveProcessedRegionalAttentionChunk:
        """Resolve one branch UUID and all regional schedules atomically."""

        branch, contexts = _branch(plan, selector=selector, chunk_index=chunk_index)
        matching_base = tuple(
            entry
            for entry in contexts.base_context.entries
            if entry.uuid is conditioning_uuid
        )
        if len(matching_base) != 1:
            raise ValueError(
                f"Regional attention {branch.value} chunk {chunk_index} UUID "
                "does not identify exactly one processed base entry."
            )
        base_entry = matching_base[0]
        if not self._schedule_policy.is_active(base_entry.schedule, sigma=sigma):
            raise ValueError(
                f"Regional attention {branch.value} chunk {chunk_index} UUID is "
                f"inactive at sigma {sigma}."
            )
        return ActiveProcessedRegionalAttentionChunk(
            chunk_index=chunk_index,
            branch=branch,
            base_entry=base_entry,
            regional_entries=tuple(
                self._regional_entries(contexts, region_index, sigma=sigma)
                for region_index in range(plan.mask_bank.region_count)
            ),
        )

    def _regional_entries(
        self,
        branch: ProcessedRegionalAttentionBranch,
        region_index: int,
        *,
        sigma: float,
    ) -> tuple[ProcessedRegionalAttentionEntry, ...] | None:
        """Distinguish absent regions from authored regions with no active entry."""

        if region_index >= len(branch.regional_contexts):
            return None
        return self._active_entries(
            branch.regional_contexts[region_index].entries,
            sigma=sigma,
        )

    def _active_entries(
        self,
        entries: tuple[ProcessedRegionalAttentionEntry, ...],
        *,
        sigma: float,
    ) -> tuple[ProcessedRegionalAttentionEntry, ...]:
        """Retain every active entry in authored order at one finite sigma."""

        return tuple(
            entry
            for entry in entries
            if self._schedule_policy.is_active(entry.schedule, sigma=sigma)
        )


def _branch(
    plan: ProcessedRegionalAttentionPlan,
    *,
    selector: object,
    chunk_index: int,
) -> tuple[RegionalAttentionBranch, ProcessedRegionalAttentionBranch]:
    """Narrow one Comfy branch selector without accepting booleans."""

    if isinstance(selector, bool) or not isinstance(selector, int):
        raise TypeError(
            f"cond_or_uncond selector {chunk_index} must be integer 0 or 1."
        )
    if selector == 0:
        return RegionalAttentionBranch.POSITIVE, plan.positive
    if selector == 1:
        return RegionalAttentionBranch.NEGATIVE, plan.negative
    raise ValueError(
        f"cond_or_uncond selector {chunk_index} must be 0 or 1; observed {selector}."
    )


def _selectors(value: object) -> tuple[object, ...]:
    """Require Comfy's ordered selector container."""

    if not isinstance(value, list | tuple):
        raise TypeError("cond_or_uncond must be a list or tuple of chunk selectors.")
    return tuple(value)


def _conditioning_uuids(value: object) -> tuple[UUID, ...]:
    """Require exact Comfy UUID objects for every supplied chunk."""

    if not isinstance(value, list | tuple):
        raise TypeError("Conditioning UUIDs must be a list or tuple.")
    identities = tuple(value)
    if any(not isinstance(identity, UUID) for identity in identities):
        raise TypeError("Every conditioning identity must be a Comfy UUID.")
    return identities


REGIONAL_ATTENTION_SELECTION_SERVICE = RegionalAttentionSelectionService(
    CONDITIONING_SCHEDULE_SELECTION_POLICY
)
