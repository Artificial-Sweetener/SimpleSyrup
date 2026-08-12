# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own immutable regional Anima attention execution authorities."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...domain.regional_attention_batch import BatchedRegionalAttentionContexts
from ...domain.regional_attention_weights import RegionalAttentionWeightingPolicy
from ...domain.regional_mask_bank import RegionalMaskBank
from ...masking.regional_mask_projection import RegionalMaskProjectionMode
from ..regional_attention_execution_context import RegionalAttentionExecutionContext


@dataclass(frozen=True, slots=True)
class AnimaRegionalAttentionExecution:
    """Bind aligned contexts, masks, and static regional blend strengths."""

    contexts: BatchedRegionalAttentionContexts
    mask_bank: RegionalMaskBank
    region_strengths: tuple[float, ...]
    projection_mode: RegionalMaskProjectionMode = (
        RegionalMaskProjectionMode.CONTINUOUS_COVERAGE
    )
    execution_context: RegionalAttentionExecutionContext = field(
        default_factory=RegionalAttentionExecutionContext
    )
    dynamic_contexts: bool = False

    def __post_init__(self) -> None:
        """Require one canonical region count across every execution authority."""

        if not isinstance(self.contexts, BatchedRegionalAttentionContexts):
            raise TypeError(
                "Anima regional attention execution requires aligned contexts."
            )
        if not isinstance(self.mask_bank, RegionalMaskBank):
            raise TypeError("Anima regional attention execution requires a mask bank.")
        if len(self.contexts.regions) != self.mask_bank.region_count:
            raise ValueError(
                "Anima regional context count must match the canonical mask bank."
            )
        if not isinstance(self.region_strengths, tuple):
            raise TypeError("Anima regional strengths must be an immutable tuple.")
        if len(self.region_strengths) != self.mask_bank.region_count:
            raise ValueError(
                "Anima regional strength count must match the canonical mask bank."
            )
        if not isinstance(self.projection_mode, RegionalMaskProjectionMode):
            raise TypeError("Anima regional mask projection mode has an invalid type.")
        if not isinstance(
            self.execution_context,
            RegionalAttentionExecutionContext,
        ):
            raise TypeError("Anima regional execution context has an invalid type.")
        if not isinstance(self.dynamic_contexts, bool):
            raise TypeError("Anima regional dynamic-context state must be boolean.")
        RegionalAttentionWeightingPolicy().weights(
            self.mask_bank.conditioning_masks.reshape(
                self.mask_bank.region_count,
                -1,
            ),
            region_strengths=self.region_strengths,
        )

    @property
    def active_contexts(self) -> BatchedRegionalAttentionContexts:
        """Return the current model-call contexts or the static fallback batch."""

        return self.execution_context.current_or(self.contexts)
