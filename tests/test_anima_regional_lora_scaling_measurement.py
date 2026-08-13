# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify per-repeat scaling cache, work, tensor, and backend evidence."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from regional_lora_test_values import single_target_execution
from torch import nn

from simple_syrup.domain.regional_lora_plan import RegionalLoraBranch
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import (
    AnimaLoraTargetModule,
    AnimaModuleSurface,
)
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraTargetFamily
from tools.anima_regional_lora_performance import (
    matrix_measurement as measurement_module,
)
from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.matrix_profile import (
    PreparedScalingProfile,
)
from tools.anima_regional_lora_performance.measurement import (
    PerformanceSequenceMeasurement,
)
from tools.anima_regional_lora_performance.runtime_profile import (
    PerformanceRuntimeProfile,
)


def test_scaling_measurement_reads_production_work_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish work through diagnostics and cache owners without reinterpretation."""

    execution = single_target_execution(
        AnimaLoraTargetFamily.SELF_ATTN_Q,
        RegionalLoraBranch.POSITIVE,
    )
    definition = default_scaling_manifest().profiles[1]
    composition = AnimaRegionalLoraComposition((execution,))
    target = execution.admission.targets[0]
    surface = AnimaModuleSurface(
        cast(Any, object()),
        (),
        (
            AnimaLoraTargetModule(
                target.adapter.target,
                target.block_index,
                target.family,
                nn.Linear(2, 2),
                2,
                2,
                "(input)",
            ),
        ),
    )
    runtime = PerformanceRuntimeProfile(
        definition,
        object(),
        {"transformer_options": {}},
        (execution,),
        (execution.cache,),
        composition,
    )
    profile = PreparedScalingProfile(definition, runtime, surface, None)
    output = torch.tensor([[[1.25, -0.5]]])
    monkeypatch.setattr(
        measurement_module,
        "execute_call_sequence",
        lambda *args, **kwargs: PerformanceSequenceMeasurement(
            12.5,
            4096,
            "a" * 64,
            30,
            output,
        ),
    )

    observation = measurement_module.measure_scaling_repeat(
        profile,
        repeat_index=2,
        latent=torch.zeros(1),
        context=torch.zeros(1),
        sample_sigmas=torch.linspace(1.0, 0.0, 31),
        call_count=30,
    )

    assert observation.profile_id == definition.profile_id
    assert observation.repeat_index == 2
    assert observation.output_float32.dtype is torch.float32
    assert torch.equal(observation.output_float32, output.float())
    assert observation.cache_entries_before == 0
    assert observation.cache_entries_after == 0
    assert observation.attention_override_calls == 0
    assert observation.work.active_adapter_uses == 1
    assert observation.work.active_target_count == 1
    assert observation.work.target_use_count == 1
    assert observation.work.deduplicated_target_group_count == 1
    assert observation.work.compatible_projection_batch_count == 1
    assert observation.work.deduplicated_target_uses == 0


def test_scaling_measurement_rejects_negative_repeat() -> None:
    """Fail before model execution on an invalid result coordinate."""

    with pytest.raises(ValueError, match="non-negative"):
        measurement_module.measure_scaling_repeat(
            cast(PreparedScalingProfile, object()),
            repeat_index=-1,
            latent=torch.zeros(1),
            context=torch.zeros(1),
            sample_sigmas=torch.ones(1),
            call_count=1,
        )
