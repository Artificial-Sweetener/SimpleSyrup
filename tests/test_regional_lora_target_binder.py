# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify atomic regional LoRA target binding against real Comfy patchers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import comfy.model_patcher
import pytest
import torch
from comfy.weight_adapter.lora import LoRAAdapter
from torch import nn

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.regional_lora.comfy_adapter_resolution import (
    ComfyAdapterResolutionIssue,
    ComfyAdapterResolutionIssueCode,
    ComfyAdapterTargetPath,
    ComfyNormalizedAdapterTarget,
    ComfyRegionalAdapterResolution,
    ComfyRegionalLoraResolution,
)
from simple_syrup.runtime.regional_lora.resolved_operation_translator import (
    ComfyResolvedOperationTranslator,
)
from simple_syrup.runtime.regional_lora.target_binder import RegionalLoraTargetBinder
from simple_syrup.runtime.regional_lora.target_binding import (
    BoundRegionalLoraModuleClass,
    BoundRegionalLoraSpatialCapability,
    RegionalLoraBindingIssueCode,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload


def test_binder_retains_exact_identities_and_never_mutates_source() -> None:
    """Bind a Linear target by identity while preserving all patcher state."""

    source = _patcher()
    resolution = _resolution(
        ((_adapter(0), (_linear_target("diffusion_model.layer.weight"),)),)
    )
    operations = ComfyResolvedOperationTranslator().translate(resolution)
    before = _snapshot(source)

    result = RegionalLoraTargetBinder().bind(
        source=source,
        candidate=source,
        resolution=resolution,
        operations=operations,
    )

    assert result.admissible
    assert result.entries[0].descriptor is operations.entries[0]
    assert (
        result.entries[0].normalized_target is resolution.adapters[0].model_targets[0]
    )
    model = cast(_TestModel, source.model)
    assert result.entries[0].module is model.diffusion_model.layer
    assert result.entries[0].parameter is model.diffusion_model.layer.weight
    assert result.entries[0].module_class is BoundRegionalLoraModuleClass.LINEAR
    assert (
        result.entries[0].spatial_capability
        is BoundRegionalLoraSpatialCapability.GLOBAL_ONLY
    )
    assert _snapshot(source) == before


def test_binder_accepts_direct_clone_and_explicit_linear_capability() -> None:
    """Bind one direct clone and retain consumer-owned spatialization evidence."""

    source = _patcher()
    candidate = source.clone()
    path = "diffusion_model.layer.weight"
    resolution = _resolution(((_adapter(0), (_linear_target(path),)),))

    result = RegionalLoraTargetBinder().bind(
        source=source,
        candidate=candidate,
        resolution=resolution,
        operations=ComfyResolvedOperationTranslator().translate(resolution),
        linear_spatial_capabilities={
            path: BoundRegionalLoraSpatialCapability.CONSUMER_SPATIALIZED
        },
    )

    assert result.admissible
    assert (
        result.entries[0].spatial_capability
        is BoundRegionalLoraSpatialCapability.CONSUMER_SPATIALIZED
    )


def test_binder_classifies_convolution_as_direct() -> None:
    """Derive direct spatial capability from the observed Conv2d target."""

    source = _patcher()
    source.model.diffusion_model.conv = nn.Conv2d(4, 6, 3, bias=False)
    resolution = _resolution(
        ((_adapter(0), (_conv_target("diffusion_model.conv.weight"),)),)
    )

    result = RegionalLoraTargetBinder().bind(
        source=source,
        candidate=source,
        resolution=resolution,
        operations=ComfyResolvedOperationTranslator().translate(resolution),
    )

    assert result.admissible
    assert result.entries[0].module_class is BoundRegionalLoraModuleClass.CONVOLUTION_2D
    assert (
        result.entries[0].spatial_capability
        is BoundRegionalLoraSpatialCapability.DIRECT
    )


def test_binder_reads_pending_object_patch_without_applying_it() -> None:
    """Bind through the effective replacement graph while leaving it pending."""

    source = _patcher()
    original = source.model.diffusion_model.layer
    replacement = nn.Linear(4, 6, bias=False)
    source.add_object_patch("diffusion_model.layer", replacement)
    resolution = _resolution(
        ((_adapter(0), (_linear_target("diffusion_model.layer.weight"),)),)
    )
    before = _snapshot(source)

    result = RegionalLoraTargetBinder().bind(
        source=source,
        candidate=source,
        resolution=resolution,
        operations=ComfyResolvedOperationTranslator().translate(resolution),
    )

    assert result.admissible
    assert result.entries[0].module is replacement
    assert result.entries[0].parameter is replacement.weight
    assert result.entries[0].active_object_patch_paths == ("diffusion_model.layer",)
    assert source.model.diffusion_model.layer is original
    assert _snapshot(source) == before


def test_binder_aggregates_target_resolution_and_rejection_failures() -> None:
    """Return all independent failures without partial graph mutation."""

    source = _patcher()
    source.model.diffusion_model.unsupported = nn.Embedding(4, 6)
    source.model.diffusion_model.bad_shape = nn.Linear(4, 6, bias=False)
    source.model.diffusion_model.dora = nn.Linear(4, 6, bias=False)
    targets = (
        _linear_target("diffusion_model.missing.weight"),
        _linear_target("diffusion_model.unsupported.weight"),
        _linear_target("diffusion_model.bad_shape.weight", output_size=7),
        _linear_target("diffusion_model.dora.weight", dora=True),
    )
    issue = ComfyAdapterResolutionIssue(
        0,
        "adapter-0.safetensors",
        ComfyAdapterResolutionIssueCode.UNSUPPORTED_OPERATION,
        "host characterization issue",
    )
    resolution = _resolution(((_adapter(0), targets),), issues=(issue,))
    before = _snapshot(source)

    result = RegionalLoraTargetBinder().bind(
        source=source,
        candidate=source,
        resolution=resolution,
        operations=ComfyResolvedOperationTranslator().translate(resolution),
    )

    assert {entry.issues[0].code for entry in result.entries} == {
        RegionalLoraBindingIssueCode.TARGET_NOT_FOUND,
        RegionalLoraBindingIssueCode.MODULE_UNSUPPORTED,
        RegionalLoraBindingIssueCode.SHAPE_INCOMPATIBLE,
        RegionalLoraBindingIssueCode.OPERATION_REJECTED,
    }
    assert RegionalLoraBindingIssueCode.RESOLUTION_ISSUE in {
        issue.code for issue in result.issues
    }
    assert not result.admissible
    assert _snapshot(source) == before


def test_binder_rejects_duplicate_ownership_within_one_adapter() -> None:
    """Reject repeated target ownership while retaining both pieces of evidence."""

    source = _patcher()
    target = _linear_target("diffusion_model.layer.weight")
    resolution = _resolution(((_adapter(0), (target, target)),))

    result = RegionalLoraTargetBinder().bind(
        source=source,
        candidate=source,
        resolution=resolution,
        operations=ComfyResolvedOperationTranslator().translate(resolution),
    )

    assert len(result.entries) == 2
    assert all(
        entry.issues[0].code is RegionalLoraBindingIssueCode.DUPLICATE_OWNERSHIP
        for entry in result.entries
    )


def test_binder_allows_multiple_adapters_to_compose_on_one_parameter() -> None:
    """Permit ordered adapter composition without mistaking it for duplication."""

    source = _patcher()
    resolution = _resolution(
        (
            (_adapter(0), (_linear_target("diffusion_model.layer.weight"),)),
            (_adapter(1), (_linear_target("diffusion_model.layer.weight"),)),
        )
    )

    result = RegionalLoraTargetBinder().bind(
        source=source,
        candidate=source,
        resolution=resolution,
        operations=ComfyResolvedOperationTranslator().translate(resolution),
    )

    assert result.admissible
    assert len(result.entries) == 2


def test_binder_reports_u2_u3_identity_mismatch() -> None:
    """Refuse positional guessing when normalized and translated identities differ."""

    source = _patcher()
    one = _resolution(
        ((_adapter(0), (_linear_target("diffusion_model.layer.weight"),)),)
    )
    two = _resolution(
        (
            (
                _adapter(0),
                (
                    _linear_target("diffusion_model.layer.weight"),
                    _linear_target("diffusion_model.layer.weight"),
                ),
            ),
        )
    )

    result = RegionalLoraTargetBinder().bind(
        source=source,
        candidate=source,
        resolution=two,
        operations=ComfyResolvedOperationTranslator().translate(one),
    )

    assert RegionalLoraBindingIssueCode.RESOLUTION_MISMATCH in {
        issue.code for issue in result.issues
    }
    assert len(result.entries) == 1


def test_binder_rejects_unrelated_patcher_before_target_access() -> None:
    """Reject unrelated graph lineage without returning partial target entries."""

    source = _patcher()
    resolution = _resolution(
        ((_adapter(0), (_linear_target("diffusion_model.layer.weight"),)),)
    )

    result = RegionalLoraTargetBinder().bind(
        source=source,
        candidate=_patcher(),
        resolution=resolution,
        operations=ComfyResolvedOperationTranslator().translate(resolution),
    )

    assert result.entries == ()
    assert result.issues[0].code is RegionalLoraBindingIssueCode.SOURCE_LINEAGE


def test_binder_rejects_invalid_linear_capability_declaration() -> None:
    """Accept only explicit Linear capabilities owned by actual consumers."""

    source = _patcher()
    resolution = _resolution(
        ((_adapter(0), (_linear_target("diffusion_model.layer.weight"),)),)
    )

    with pytest.raises(ValueError, match="consumer-spatialized or global-only"):
        RegionalLoraTargetBinder().bind(
            source=source,
            candidate=source,
            resolution=resolution,
            operations=ComfyResolvedOperationTranslator().translate(resolution),
            linear_spatial_capabilities={
                "diffusion_model.layer.weight": (
                    BoundRegionalLoraSpatialCapability.DIRECT
                )
            },
        )


def _patcher() -> comfy.model_patcher.ModelPatcher:
    """Return a real Comfy patcher with a minimal nested diffusion graph."""

    model = _TestModel()
    return comfy.model_patcher.ModelPatcher(
        model, torch.device("cpu"), torch.device("cpu")
    )


def _snapshot(patcher: comfy.model_patcher.ModelPatcher) -> SimpleNamespace:
    """Capture all graph identities and pending patch state used by the binder."""

    model = cast(_TestModel, patcher.model)
    return SimpleNamespace(
        model=model,
        diffusion_model=model.diffusion_model,
        layer=model.diffusion_model.layer,
        weight=model.diffusion_model.layer.weight,
        object_patches=dict(patcher.object_patches),
        backups=dict(patcher.object_patches_backup),
    )


class _TestDiffusionModel(nn.Module):
    """Provide typed target modules for binder tests."""

    def __init__(self) -> None:
        """Create one Linear target and permit focused test additions."""

        super().__init__()
        self.layer = nn.Linear(4, 6, bias=False)


class _TestModel(nn.Module):
    """Provide the typed Comfy model root used by binder tests."""

    def __init__(self) -> None:
        """Expose one diffusion-model owner below the model root."""

        super().__init__()
        self.diffusion_model = _TestDiffusionModel()


def _resolution(
    adapters: tuple[
        tuple[RegionalLoraAdapterPlan, tuple[ComfyNormalizedAdapterTarget, ...]], ...
    ],
    *,
    issues: tuple[ComfyAdapterResolutionIssue, ...] = (),
) -> ComfyRegionalLoraResolution:
    """Build ordered normalized evidence around the supplied exact targets."""

    return ComfyRegionalLoraResolution(
        tuple(
            ComfyRegionalAdapterResolution(
                adapter,
                RegionalLoraHostPayload(False, None, {}, None),
                targets,
                (),
                (),
            )
            for adapter, targets in adapters
        ),
        issues,
    )


def _linear_target(
    path: str,
    *,
    output_size: int = 6,
    dora: bool = False,
) -> ComfyNormalizedAdapterTarget:
    """Create one exact normalized matrix-pair target."""

    operation = LoRAAdapter(
        {"up", "down"},
        (
            torch.ones((output_size, 2)),
            torch.ones((2, 4)),
            None,
            None,
            torch.ones(output_size) if dora else None,
            None,
        ),
    )
    return ComfyNormalizedAdapterTarget(
        ComfyAdapterTargetPath(path, None),
        operation,
        "LoRAAdapter",
        ("down", "up"),
        not dora,
    )


def _conv_target(path: str) -> ComfyNormalizedAdapterTarget:
    """Create one exact normalized Conv2d LoRA target."""

    operation = LoRAAdapter(
        {"up", "down"},
        (torch.ones((6, 2)), torch.ones((2, 4, 3, 3)), None, None, None, None),
    )
    return ComfyNormalizedAdapterTarget(
        ComfyAdapterTargetPath(path, None),
        operation,
        "LoRAAdapter",
        ("down", "up"),
        True,
    )


def _adapter(composition_index: int) -> RegionalLoraAdapterPlan:
    """Create one valid ordered regional adapter plan."""

    return RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity(f"adapter-{composition_index}.safetensors"),
        composition_index,
        composition_index,
        RegionalLoraBranch.POSITIVE,
        0.8,
        (RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
    )
