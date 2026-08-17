# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Register focused benchmark-only Comfy v3 probe nodes."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .anima_regional_profile_node import StaticAnimaRegionalProfileV3
from .attention_coupling_phase_node import ProfiledKSamplerAttentionCouplingV3
from .clip_schedule_snapshot import SnapshotClipScheduleV3
from .cold_path_capture import (
    CaptureColdPathDiagnosticsV3,
    ReadColdPathDiagnosticsV3,
)
from .conditioning_batch_snapshot import SnapshotConditioningBatchV3
from .latent_completion import CompleteLatentV3
from .lora_execution_probe import InstrumentLoraModelV3, ReadLoraMetricsV3
from .materialization_parity_node import CompareMaterializationParityV3
from .model_modifier_snapshot import SnapshotModelModifierV3
from .operator_profile import ProfileIndexedModelCallV3, ReadOperatorProfileV3
from .prompt_control_expansion import SnapshotPromptControlExpansionV3
from .prompt_control_runtime import (
    InstrumentPromptControlModelV3,
    ReadPromptControlRuntimeV3,
)
from .prompt_control_snapshot import SnapshotPromptControlV3
from .regional_diagnostics_capture import (
    CaptureRegionalDiagnosticsV3,
    ReadRegionalDiagnosticsV3,
)
from .regional_hook_fixture import CreateRegionalHookFixtureV3
from .sampling_metrics import InstrumentModelV3, ReadMetricsV3

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyExtensionBase:
        """Type-checking base for the benchmark-only extension."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyExtensionBase = _comfy_api.ComfyExtension


class BenchmarkProbeExtension(_ComfyExtensionBase):
    """Register the focused benchmark instrumentation nodes."""

    async def get_node_list(self) -> list[type[object]]:
        """Return every benchmark-only node owned by this extension."""

        return [
            StaticAnimaRegionalProfileV3,
            ProfiledKSamplerAttentionCouplingV3,
            SnapshotClipScheduleV3,
            SnapshotConditioningBatchV3,
            InstrumentModelV3,
            ReadMetricsV3,
            CompleteLatentV3,
            CaptureColdPathDiagnosticsV3,
            ReadColdPathDiagnosticsV3,
            ProfileIndexedModelCallV3,
            ReadOperatorProfileV3,
            InstrumentLoraModelV3,
            ReadLoraMetricsV3,
            CompareMaterializationParityV3,
            SnapshotModelModifierV3,
            SnapshotPromptControlV3,
            SnapshotPromptControlExpansionV3,
            InstrumentPromptControlModelV3,
            ReadPromptControlRuntimeV3,
            CaptureRegionalDiagnosticsV3,
            ReadRegionalDiagnosticsV3,
            CreateRegionalHookFixtureV3,
        ]


async def comfy_entrypoint() -> BenchmarkProbeExtension:
    """Return the benchmark-only Comfy extension."""

    return BenchmarkProbeExtension()
