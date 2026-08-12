# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve fresh Comfy runtime UUIDs to preprocessed attention entries."""

from __future__ import annotations

from uuid import UUID

import torch
from comfy.utils import repeat_to_batch_size

from ..domain.conditioning_schedule_selection import (
    CONDITIONING_SCHEDULE_SELECTION_POLICY,
    ConditioningScheduleSelectionPolicy,
)
from ..domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionPlan,
)


class RegionalAttentionRuntimeIdentityResolver:
    """Match active model contexts to the processor-owned UUID identities."""

    def __init__(
        self,
        schedule_policy: ConditioningScheduleSelectionPolicy | None = None,
    ) -> None:
        """Retain the authoritative conditioning schedule predicate."""

        self._schedule_policy = (
            schedule_policy or CONDITIONING_SCHEDULE_SELECTION_POLICY
        )

    def resolve(
        self,
        plan: ProcessedRegionalAttentionPlan,
        *,
        cond_or_uncond: object,
        base_context: torch.Tensor,
        sigma: float,
        latent_batch_size: int,
    ) -> tuple[UUID, ...]:
        """Return processor UUIDs in the exact active Comfy chunk order."""

        if not isinstance(plan, ProcessedRegionalAttentionPlan):
            raise TypeError("Runtime regional identity resolution requires a plan.")
        if not isinstance(cond_or_uncond, list | tuple):
            raise TypeError("Runtime cond_or_uncond must be a list or tuple.")
        selectors = tuple(cond_or_uncond)
        if not selectors:
            raise ValueError("Runtime regional identity resolution requires chunks.")
        if not isinstance(base_context, torch.Tensor) or base_context.ndim != 3:
            raise ValueError("Runtime base context must use BxSxD layout.")
        expected_batch = len(selectors) * latent_batch_size
        if int(base_context.shape[0]) != expected_batch:
            raise ValueError(
                "Runtime base context batch does not match CFG chunks and latent batch."
            )

        identities: list[UUID] = []
        for chunk_index, selector in enumerate(selectors):
            branch = self._branch(plan, selector, chunk_index)
            active = tuple(
                entry
                for entry in branch.base_context.entries
                if self._schedule_policy.is_active(entry.schedule, sigma=sigma)
            )
            if not active:
                raise ValueError(
                    f"Runtime regional chunk {chunk_index} has no active base entry "
                    f"at sigma {sigma}."
                )
            if len(active) == 1:
                identities.append(active[0].uuid)
                continue
            start = chunk_index * latent_batch_size
            stop = start + latent_batch_size
            observed = base_context[start:stop]
            matching = tuple(
                entry
                for entry in active
                if torch.equal(
                    repeat_to_batch_size(
                        entry.cross_attention,
                        latent_batch_size,
                    ).to(device=observed.device, dtype=observed.dtype),
                    observed,
                )
            )
            if not matching:
                raise ValueError(
                    f"Runtime regional chunk {chunk_index} does not match an active "
                    "processed base context."
                )
            identities.append(matching[0].uuid)
        return tuple(identities)

    @staticmethod
    def _branch(
        plan: ProcessedRegionalAttentionPlan,
        selector: object,
        chunk_index: int,
    ) -> ProcessedRegionalAttentionBranch:
        """Return the exact branch named by Comfy's integer selector."""

        if isinstance(selector, bool) or not isinstance(selector, int):
            raise TypeError(
                f"Runtime cond_or_uncond selector {chunk_index} must be 0 or 1."
            )
        if selector == 0:
            return plan.positive
        if selector == 1:
            return plan.negative
        raise ValueError(
            f"Runtime cond_or_uncond selector {chunk_index} must be 0 or 1; "
            f"observed {selector}."
        )


REGIONAL_ATTENTION_RUNTIME_IDENTITY_RESOLVER = (
    RegionalAttentionRuntimeIdentityResolver()
)
