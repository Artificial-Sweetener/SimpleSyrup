# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify generic regional operation device state follows Comfy detach."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from comfy.patcher_extension import CallbacksMP

from simple_syrup.runtime.regional_lora.execution_cache import (
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.operation_cache_lifecycle import (
    REGIONAL_OPERATION_DETACH_CALLBACK_KEY,
    RegionalOperationCacheLifecycle,
)


@dataclass
class _Plan:
    """Record every local preparation release."""

    clears: int = 0

    def clear(self) -> None:
        """Record one detach release."""

        self.clears += 1


def test_detach_callback_releases_every_plan_and_shared_cache() -> None:
    """Release all generic owners on complete and clone-switch detach forms."""

    first = _Plan()
    second = _Plan()
    cache = RegionalLoraExecutionCache()
    lifecycle = RegionalOperationCacheLifecycle((first, second), cache)
    model = _patcher()
    lifecycle.mutation().apply(model)

    model.detach(unpatch_all=False)
    model.detach(unpatch_all=True)

    assert (first.clears, second.clears) == (2, 2)
    assert cache.size == 0
    assert model.get_callbacks(
        CallbacksMP.ON_DETACH,
        REGIONAL_OPERATION_DETACH_CALLBACK_KEY,
    ) == [lifecycle.release]


def _patcher() -> Any:
    """Return one real Comfy patcher for callback execution."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(torch.nn.Linear(2, 2), device, device)
