# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize cross-template standard-UNet residency handoff."""

from __future__ import annotations

import torch
from comfy.model_patcher import ModelPatcher
from torch import nn

from simple_syrup.runtime.regional_lora.standard_unet_variant_residency_handoff import (
    StandardUnetVariantResidencyHandoff,
)


def test_distinct_template_allocations_are_not_comfy_clones() -> None:
    """Show why Comfy retains separately allocated roots from one lineage."""

    first = _patcher(nn.Linear(1, 1, bias=False))
    second = _patcher(nn.Linear(1, 1, bias=False))
    second.clone_base_uuid = first.clone_base_uuid

    assert first.clone_base_uuid == second.clone_base_uuid
    assert first.model is not second.model
    assert not first.is_clone(second)


def test_handoff_unloads_only_the_previous_distinct_allocation() -> None:
    """Release stale resident weights while preserving exact-root requeues."""

    calls: list[tuple[ModelPatcher, bool, bool]] = []

    def unload(
        model: ModelPatcher,
        *,
        unload_additional_models: bool,
        all_devices: bool,
    ) -> None:
        calls.append((model, unload_additional_models, all_devices))

    handoff = StandardUnetVariantResidencyHandoff(unload)
    first = _cuda_patcher(nn.Linear(1, 1, bias=False))
    first_clone = first.clone()
    second = _cuda_patcher(nn.Linear(1, 1, bias=False))

    assert not handoff.activate(first)
    assert not handoff.activate(first_clone)
    assert handoff.activate(second)
    assert calls == [(first_clone, False, False)]


def test_handoff_ignores_cpu_and_clear_forgets_prior_identity() -> None:
    """Avoid host unloading for CPU roots and after explicit cache cleanup."""

    calls: list[ModelPatcher] = []

    def unload(model: ModelPatcher, **_kwargs: object) -> None:
        calls.append(model)

    handoff = StandardUnetVariantResidencyHandoff(unload)
    first = _cuda_patcher(nn.Linear(1, 1, bias=False))
    second = _cuda_patcher(nn.Linear(1, 1, bias=False))

    assert not handoff.activate(_patcher(nn.Linear(1, 1, bias=False)))
    assert not handoff.activate(first)
    handoff.clear()
    assert not handoff.activate(second)
    assert calls == []


def _patcher(model: nn.Module) -> ModelPatcher:
    """Return one static patcher without allocating CUDA tensors."""

    device = torch.device("cpu")
    return ModelPatcher(model, device, device)


def _cuda_patcher(model: nn.Module) -> ModelPatcher:
    """Return one CUDA-addressed patcher without allocating CUDA tensors."""

    patcher = _patcher(model)
    patcher.load_device = torch.device("cuda:0")
    return patcher
