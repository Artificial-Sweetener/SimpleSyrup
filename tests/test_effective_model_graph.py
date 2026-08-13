# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify read-only effective model graph resolution through object patches."""

from __future__ import annotations

from types import SimpleNamespace

import comfy.model_patcher
import pytest
import torch
from torch import nn

from simple_syrup.runtime.regional_lora.effective_model_graph import (
    EffectiveModelGraphError,
    EffectiveModelGraphResolver,
)


def test_resolver_reads_base_graph_without_mutation() -> None:
    """Return exact nested base objects when no active patch owns the path."""

    patcher = _patcher()
    layer = patcher.model.diffusion_model.layer
    before = _snapshot(patcher)

    result = EffectiveModelGraphResolver().resolve(
        patcher,
        "diffusion_model.layer.weight",
    )

    assert result.value is layer.weight
    assert result.active_object_patch_paths == ()
    assert _snapshot(patcher) == before


def test_resolver_traverses_one_unapplied_parent_object_patch() -> None:
    """Observe a pending replacement's children without applying the patch."""

    patcher = _patcher()
    original = patcher.model.diffusion_model.layer
    replacement = nn.Linear(4, 6, bias=False)
    patcher.add_object_patch("diffusion_model.layer", replacement)

    module = EffectiveModelGraphResolver().resolve(
        patcher,
        "diffusion_model.layer",
    )
    parameter = EffectiveModelGraphResolver().resolve(
        patcher,
        "diffusion_model.layer.weight",
    )

    assert module.value is replacement
    assert parameter.value is replacement.weight
    assert module.active_object_patch_paths == ("diffusion_model.layer",)
    assert parameter.active_object_patch_paths == ("diffusion_model.layer",)
    assert patcher.model.diffusion_model.layer is original


def test_resolver_uses_exact_parameter_patch() -> None:
    """Observe exact pending parameter ownership independently of its module."""

    patcher = _patcher()
    replacement = nn.Parameter(torch.ones((6, 4)), requires_grad=False)
    patcher.add_object_patch("diffusion_model.layer.weight", replacement)

    result = EffectiveModelGraphResolver().resolve(
        patcher,
        "diffusion_model.layer.weight",
    )

    assert result.value is replacement
    assert result.active_object_patch_paths == ("diffusion_model.layer.weight",)


def test_resolver_rejects_overlapping_parent_and_child_patches() -> None:
    """Reject insertion-order-dependent effective graph ownership."""

    patcher = _patcher()
    patcher.add_object_patch("diffusion_model.layer", nn.Linear(4, 6, bias=False))
    patcher.add_object_patch(
        "diffusion_model.layer.weight",
        nn.Parameter(torch.ones((6, 4)), requires_grad=False),
    )

    with pytest.raises(EffectiveModelGraphError, match="overlapping object patch"):
        EffectiveModelGraphResolver().resolve(
            patcher,
            "diffusion_model.layer.weight",
        )


@pytest.mark.parametrize("path", ["", ".layer", "layer.", "layer..weight"])
def test_resolver_rejects_invalid_or_missing_paths(path: str) -> None:
    """Fail clearly for malformed and absent effective graph paths."""

    patcher = _patcher()
    with pytest.raises((ValueError, EffectiveModelGraphError)):
        EffectiveModelGraphResolver().resolve(patcher, path)
    with pytest.raises(EffectiveModelGraphError, match="does not exist"):
        EffectiveModelGraphResolver().resolve(patcher, "diffusion_model.missing")


def _patcher() -> comfy.model_patcher.ModelPatcher:
    """Return one real Comfy patcher around a minimal nested graph."""

    model = nn.Module()
    model.diffusion_model = nn.Module()
    model.diffusion_model.layer = nn.Linear(4, 6, bias=False)
    return comfy.model_patcher.ModelPatcher(
        model,
        torch.device("cpu"),
        torch.device("cpu"),
    )


def _snapshot(patcher: comfy.model_patcher.ModelPatcher) -> SimpleNamespace:
    """Snapshot graph and patch mapping identity without copying tensors."""

    return SimpleNamespace(
        model=patcher.model,
        layer=patcher.model.diffusion_model.layer,
        weight=patcher.model.diffusion_model.layer.weight,
        object_patches=dict(patcher.object_patches),
        backups=dict(patcher.object_patches_backup),
    )
