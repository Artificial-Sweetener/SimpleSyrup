# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Profile exact MODEL interoperability validation boundaries."""

from __future__ import annotations

from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_model_capabilities import RegionalModelCapabilities
from simple_syrup.runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropReport,
    RegionalModelPatchInteropValidator,
)

from .synchronized_phase_timing import measure_synchronized_phase, model_device


class ProfiledRegionalModelPatchInteropValidator(RegionalModelPatchInteropValidator):
    """Time exact production interop admission and execution validation."""

    def validate(
        self,
        model: object,
        capabilities: RegionalModelCapabilities,
    ) -> RegionalModelPatchInteropReport:
        """Delegate admission validation with synchronized model-device timing."""

        with measure_synchronized_phase(
            "interop_validation",
            device=model_device(model),
        ):
            return super().validate(model, capabilities)

    def validate_execution(
        self,
        report: RegionalModelPatchInteropReport,
        processed_plan: ProcessedRegionalAttentionPlan,
        execution_mode: RegionalAttentionExecutionMode,
    ) -> None:
        """Delegate execution validation while recording its exact duration."""

        with measure_synchronized_phase(
            "interop_execution_validation",
            device=None,
        ):
            return super().validate_execution(
                report,
                processed_plan,
                execution_mode,
            )
