# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build explicit compact Anima regional branch invocations for tests."""

from __future__ import annotations

from simple_syrup.runtime.regional_lora.anima_branch_batch import (
    AnimaRegionalBranchInvocation,
)


def uniform_branch_invocation(
    source_batch_size: int,
    branch_region_indices: tuple[int | None, ...],
) -> AnimaRegionalBranchInvocation:
    """Expand complete uniform branches into compact row-level identity."""

    return AnimaRegionalBranchInvocation(
        source_batch_size,
        tuple(
            source_index
            for _region_index in branch_region_indices
            for source_index in range(source_batch_size)
        ),
        tuple(
            region_index
            for region_index in branch_region_indices
            for _source_index in range(source_batch_size)
        ),
    )
