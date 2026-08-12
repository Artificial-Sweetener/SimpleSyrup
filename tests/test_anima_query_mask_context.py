# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove one shared Anima query-mask projection per model invocation."""

from __future__ import annotations

import torch
from regional_attention_test_values import single_entry_regions

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.masking.regional_mask_projection import (
    RegionalMaskForm,
    RegionalMaskProjectionMode,
)
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_query_mask_context import (
    AnimaQueryMaskContext,
)
from simple_syrup.runtime.regional_lora.anima_query_masks import (
    AnimaQueryMaskBatch,
    AnimaQueryMaskProjector,
)


class _RecordingQueryMaskProjector(AnimaQueryMaskProjector):
    """Count exact canonical projections while retaining production behavior."""

    def __init__(self) -> None:
        """Initialize production projection with an empty call history."""

        super().__init__()
        self.calls: list[
            tuple[
                RegionalMaskBank,
                AnimaActivationGeometry,
                int,
                RegionalMaskForm,
                RegionalMaskProjectionMode,
                torch.device,
                torch.dtype,
            ]
        ] = []

    def project(
        self,
        *,
        bank: RegionalMaskBank,
        geometry: AnimaActivationGeometry,
        latent_batch_size: int,
        form: RegionalMaskForm,
        mode: RegionalMaskProjectionMode,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AnimaQueryMaskBatch:
        """Record one projection contract and delegate its exact computation."""

        self.calls.append(
            (bank, geometry, latent_batch_size, form, mode, device, dtype)
        )
        return super().project(
            bank=bank,
            geometry=geometry,
            latent_batch_size=latent_batch_size,
            form=form,
            mode=mode,
            device=device,
            dtype=dtype,
        )


def test_context_projects_once_for_one_exact_invocation_contract() -> None:
    """Share one immutable batch and invalidate every changed cache authority."""

    projector = _RecordingQueryMaskProjector()
    context = AnimaQueryMaskContext(projector)
    execution = _execution()
    geometry = _geometry()

    first = context.resolve(
        execution,
        geometry,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    repeated = context.resolve(
        execution,
        geometry,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert repeated is first
    assert len(projector.calls) == 1

    changed_dtype = context.resolve(
        execution,
        geometry,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    changed_invocation = context.resolve(
        execution,
        _geometry(),
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    assert changed_dtype is not first
    assert changed_invocation is not changed_dtype
    assert len(projector.calls) == 3
    assert (
        tuple(call[3] for call in projector.calls)
        == (RegionalMaskForm.CONDITIONING,) * 3
    )
    assert tuple(call[6] for call in projector.calls) == (
        torch.float32,
        torch.float16,
        torch.float16,
    )


def _execution() -> AnimaRegionalAttentionExecution:
    """Build one valid single-region attention execution."""

    base = torch.zeros((1, 1, 1))
    masks = torch.ones((1, 2, 2))
    bank = RegionalMaskBank(
        planning_masks=masks.clone(),
        conditioning_masks=masks.clone(),
        canvas_width=2,
        canvas_height=2,
    )
    contexts = BatchedRegionalAttentionContexts(
        latent_batch_size=1,
        chunks=(
            RegionalAttentionChunkBatch(
                0,
                RegionalAttentionBranch.POSITIVE,
                0,
                1,
            ),
        ),
        base_context=base,
        regions=single_entry_regions((base.clone(),)),
    )
    return AnimaRegionalAttentionExecution(contexts, bank, (1.0,))


def _geometry() -> AnimaActivationGeometry:
    """Build a fresh identity for one two-by-two Anima query invocation."""

    return AnimaActivationGeometry(
        input_batch_size=1,
        activation_time=1,
        activation_height=2,
        activation_width=2,
        patch_temporal=1,
        patch_spatial=1,
        query_time=1,
        query_height=2,
        query_width=2,
        spatial_layout=None,
    )
