# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove Attention Coupling projected state follows Comfy MODEL detach."""

from __future__ import annotations

from typing import Any

import torch
from comfy.patcher_extension import CallbacksMP
from regional_attention_test_values import single_entry_regions

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_attention_device_cache_lifecycle import (
    AnimaAttentionDeviceCacheLifecycle,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_query_activity import (
    AnimaRegionalQueryActivityContext,
)
from simple_syrup.runtime.regional_lora.anima_query_mask_context import (
    AnimaQueryMaskContext,
)

_DETACH_KEY = "simple_syrup.anima_attention_device_cache"


def test_detach_releases_and_lazily_reprojects_attention_device_state() -> None:
    """Drop projected tensors idempotently and reproduce their exact values."""

    query_masks = AnimaQueryMaskContext()
    query_activity = AnimaRegionalQueryActivityContext(query_masks)
    lifecycle = AnimaAttentionDeviceCacheLifecycle(query_activity, query_masks)
    model = _patcher()
    lifecycle.mutation().apply(model)
    execution = _execution()
    geometry = AnimaActivationGeometry(1, 1, 2, 2, 1, 1, 1, 2, 2, None)

    first = query_activity.resolve(
        execution,
        geometry,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert (
        query_activity.resolve(
            execution,
            geometry,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        is first
    )

    model.detach()
    model.detach()

    second = query_activity.resolve(
        execution,
        geometry,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert second is not first
    assert second.masks is not first.masks
    torch.testing.assert_close(second.masks.masks, first.masks.masks)
    torch.testing.assert_close(
        second.attention_weights.base,
        first.attention_weights.base,
    )
    assert model.get_callbacks(CallbacksMP.ON_DETACH, _DETACH_KEY) == [
        lifecycle.release
    ]


def _execution() -> AnimaRegionalAttentionExecution:
    """Create one regional attention execution over a two-by-two mask."""

    context = torch.zeros((1, 1, 1))
    contexts = BatchedRegionalAttentionContexts(
        1,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
        context,
        single_entry_regions((context.clone(),)),
    )
    masks = torch.tensor([[[1.0, 0.5], [0.0, 1.0]]])
    return AnimaRegionalAttentionExecution(
        contexts,
        RegionalMaskBank(masks.clone(), masks, 2, 2),
        (1.0,),
    )


def _patcher() -> Any:
    """Create a real CPU Comfy MODEL patcher."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(torch.nn.Linear(2, 2), device, device)
