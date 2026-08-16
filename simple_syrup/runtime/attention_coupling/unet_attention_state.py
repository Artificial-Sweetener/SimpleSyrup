# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bind one processed regional plan to standard-UNet call state."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..regional_attention_diagnostics import RegionalAttentionDiagnosticsBuilder
from ..regional_attention_execution_context import RegionalAttentionExecutionContext
from .standard_unet_attention_weighting import (
    STANDARD_UNET_ATTENTION_WEIGHTING_POLICY,
)
from .unet_attn2_resolution_cache import StandardUnetAttn2ResolutionCache


@dataclass(frozen=True, slots=True)
class StandardUnetAttentionState:
    """Retain one plan, static region strengths, and active call context."""

    plan: ProcessedRegionalAttentionPlan
    region_strengths: tuple[float, ...]
    diagnostics: RegionalAttentionDiagnosticsBuilder
    execution_context: RegionalAttentionExecutionContext = field(
        default_factory=RegionalAttentionExecutionContext
    )
    resolution_cache: StandardUnetAttn2ResolutionCache = field(
        default_factory=StandardUnetAttn2ResolutionCache
    )

    def __post_init__(self) -> None:
        """Require one canonical region authority across every state value."""

        if not isinstance(self.plan, ProcessedRegionalAttentionPlan):
            raise TypeError("Standard UNet attention state requires a processed plan.")
        if not isinstance(self.execution_context, RegionalAttentionExecutionContext):
            raise TypeError(
                "Standard UNet attention state requires an execution context."
            )
        if not isinstance(self.diagnostics, RegionalAttentionDiagnosticsBuilder):
            raise TypeError(
                "Standard UNet attention state requires regional diagnostics."
            )
        if self.diagnostics.mask_bank is not self.plan.mask_bank:
            raise ValueError(
                "Standard UNet diagnostics must share canonical mask-bank identity."
            )
        if not isinstance(self.resolution_cache, StandardUnetAttn2ResolutionCache):
            raise TypeError(
                "Standard UNet attention state requires a resolution cache."
            )
        STANDARD_UNET_ATTENTION_WEIGHTING_POLICY.weights(
            self.plan.mask_bank.conditioning_masks.reshape(
                self.plan.mask_bank.region_count,
                -1,
            ),
            region_strengths=self.region_strengths,
        )
