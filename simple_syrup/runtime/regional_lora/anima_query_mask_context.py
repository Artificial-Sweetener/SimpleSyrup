# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cache one projected Anima query-mask batch per exact model invocation."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import torch

from ...masking.regional_mask_projection import RegionalMaskForm
from .anima_activation_context import AnimaActivationGeometry
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_query_masks import AnimaQueryMaskBatch, AnimaQueryMaskProjector


@dataclass(frozen=True, slots=True)
class _AnimaQueryMaskSlot:
    """Bind one invocation identity and execution type to projected masks."""

    geometry: AnimaActivationGeometry
    execution: AnimaRegionalAttentionExecution
    device: torch.device
    dtype: torch.dtype
    masks: AnimaQueryMaskBatch


class AnimaQueryMaskContext:
    """Own task-local reuse of immutable masks across every patched block."""

    def __init__(self, projector: AnimaQueryMaskProjector | None = None) -> None:
        """Retain the canonical projector and initialize an empty task slot."""

        self._projector = projector or AnimaQueryMaskProjector()
        self._slot: ContextVar[_AnimaQueryMaskSlot | None] = ContextVar(
            "simple_syrup_anima_query_mask_slot",
            default=None,
        )

    def resolve(
        self,
        execution: AnimaRegionalAttentionExecution,
        geometry: AnimaActivationGeometry,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AnimaQueryMaskBatch:
        """Project once or return the exact active invocation's cached batch."""

        if not isinstance(execution, AnimaRegionalAttentionExecution):
            raise TypeError("Anima query masks require an attention execution.")
        if not isinstance(geometry, AnimaActivationGeometry):
            raise TypeError("Anima query masks require activation geometry.")
        target_device = torch.device(device)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("Anima query masks require a floating execution dtype.")
        slot = self._slot.get()
        if (
            slot is not None
            and slot.geometry is geometry
            and slot.execution is execution
            and slot.device == target_device
            and slot.dtype == dtype
        ):
            return slot.masks
        masks = self._projector.project(
            bank=execution.mask_bank,
            geometry=geometry,
            latent_batch_size=execution.active_contexts.latent_batch_size,
            form=RegionalMaskForm.CONDITIONING,
            mode=execution.projection_mode,
            device=target_device,
            dtype=dtype,
        )
        self._slot.set(
            _AnimaQueryMaskSlot(
                geometry,
                execution,
                target_device,
                dtype,
                masks,
            )
        )
        return masks

    def clear(self) -> None:
        """Release the current task's projected CUDA mask batch."""

        self._slot.set(None)


ANIMA_QUERY_MASK_CONTEXT = AnimaQueryMaskContext()
