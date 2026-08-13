# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify complete clone-local generic operation mutation assembly."""

from __future__ import annotations

from typing import Any

import comfy.model_patcher
import comfy.ops
import comfy_aimdo.host_buffer  # type: ignore[import-untyped]
import pytest
import torch
from comfy.patcher_extension import PatcherInjection, WrappersMP
from comfy.weight_adapter.lora import LoRAAdapter
from torch import nn

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE
from simple_syrup.runtime.regional_lora.comfy_adapter_resolution import (
    ComfyAdapterTargetPath,
    ComfyNormalizedAdapterTarget,
    ComfyRegionalAdapterResolution,
    ComfyRegionalLoraResolution,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.host_operation_backing import (
    RegionalLinearOperationPatch,
)
from simple_syrup.runtime.regional_lora.operation_mutation_assembly import (
    RegionalOperationMutationAssembler,
)
from simple_syrup.runtime.regional_lora.resolved_operation_translator import (
    ComfyResolvedOperationTranslator,
)
from simple_syrup.runtime.regional_lora.target_binder import RegionalLoraTargetBinder
from simple_syrup.runtime.regional_lora.target_binding import (
    BoundRegionalLoraSpatialCapability,
    RegionalLoraBindingResult,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload


def test_assembler_preserves_existing_wrapper_and_installs_one_direct_clone() -> None:
    """Build one grouped target replacement without changing source state."""

    source = _patcher()

    def wrapper(*args: object) -> tuple[object, ...]:
        """Return characterized wrapper arguments."""

        return args

    source.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "external", wrapper)
    binding = _binding(source)
    mutations = RegionalOperationMutationAssembler().assemble(
        binding,
        model=source,
        cache=RegionalLoraExecutionCache(),
    )
    original = source.model.diffusion_model.linear
    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        mutations,
        operation="universal regional operation assembly regression",
    )

    assert derived.parent is source
    assert source.object_patches == {}
    assert derived.get_wrappers(WrappersMP.DIFFUSION_MODEL, "external") == [wrapper]
    assert tuple(derived.object_patches) == ("diffusion_model.linear",)
    replacement = derived.object_patches["diffusion_model.linear"]
    assert isinstance(replacement, RegionalLinearOperationPatch)
    derived.patch_model(load_weights=False)
    try:
        assert derived.model.diffusion_model.linear is replacement
    finally:
        derived.unpatch_model(unpatch_weights=False)
    assert derived.model.diffusion_model.linear is original


def test_assembler_rejects_existing_injection_before_creating_mutations() -> None:
    """Fail before clone mutation while injection ownership is unresolved."""

    source = _patcher()
    source.set_injections(
        "external",
        [PatcherInjection(lambda model: None, lambda model: None)],
    )

    with pytest.raises(ValueError, match="existing MODEL injections"):
        RegionalOperationMutationAssembler().assemble(
            _binding(source),
            model=source,
            cache=RegionalLoraExecutionCache(),
        )

    assert source.object_patches == {}


def test_assembler_rejects_incomplete_binding_without_partial_plan() -> None:
    """Require complete admission before constructing any replacement shell."""

    source = _patcher()
    complete = _binding(source)
    issue = complete.entries[0].issues
    incomplete = RegionalLoraBindingResult(complete.entries, issue or (object(),))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="complete admission"):
        RegionalOperationMutationAssembler().assemble(
            incomplete,
            model=source,
            cache=RegionalLoraExecutionCache(),
        )


def test_real_lowvram_load_unload_and_reentry_preserve_public_path() -> None:
    """Exercise Comfy's host writes and graph restoration around a replacement."""

    source = _patcher(comfy_linear=True)
    binding = _binding(source)
    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        RegionalOperationMutationAssembler().assemble(
            binding,
            model=source,
            cache=RegionalLoraExecutionCache(),
        ),
        operation="regional operation low-VRAM regression",
    )
    original = source.model.diffusion_model.linear
    derived.patch_model(
        device_to=torch.device("cpu"),
        lowvram_model_memory=1,
        load_weights=True,
    )
    replacement = derived.model.diffusion_model.linear
    try:
        assert isinstance(replacement, RegionalLinearOperationPatch)
        assert replacement.comfy_cast_weights is True
        assert replacement._host_backing.module.__dict__["comfy_cast_weights"] is True
        assert (
            replacement.weight_function
            is replacement._host_backing.module.weight_function
        )
        torch.testing.assert_close(
            replacement(torch.ones((1, 2))),
            torch.ones((1, 2)),
        )
    finally:
        derived.unpatch_model(torch.device("cpu"), unpatch_weights=True)

    assert derived.model.diffusion_model.linear is original
    assert derived.object_patches_backup == {}
    derived.patch_model(load_weights=False)
    try:
        assert isinstance(
            derived.model.diffusion_model.linear,
            RegionalLinearOperationPatch,
        )
    finally:
        derived.unpatch_model(unpatch_weights=False)


def test_real_full_load_and_detach_release_shared_prepared_cache() -> None:
    """Load normally, prepare one device entry, then release it on detach."""

    source = _patcher(comfy_linear=True)
    cache = RegionalLoraExecutionCache()
    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        RegionalOperationMutationAssembler().assemble(
            _binding(source),
            model=source,
            cache=cache,
        ),
        operation="regional operation full-load regression",
    )
    original = source.model.diffusion_model.linear
    derived.patch_model(
        device_to=torch.device("cpu"),
        lowvram_model_memory=0,
        load_weights=True,
    )
    replacement = derived.model.diffusion_model.linear
    assert isinstance(replacement, RegionalLinearOperationPatch)
    preparation = replacement._plan.groups[0].preparation
    first = preparation.weights(device=torch.device("cpu"), dtype=torch.float32)
    assert cache.size == 1

    derived.detach(unpatch_all=True)

    assert derived.model.diffusion_model.linear is original
    assert cache.size == 0
    second = preparation.weights(device=torch.device("cpu"), dtype=torch.float32)
    assert second is not first
    assert cache.size == 1


def test_global_weight_patch_does_not_mutate_source_parameter_storage() -> None:
    """Keep global LoRA-style weight updates inside the derived patch window."""

    source = _patcher(comfy_linear=True)
    source_weight = source.model.diffusion_model.linear.weight
    source_before = source_weight.detach().clone()
    binding = _binding(source)
    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        RegionalOperationMutationAssembler().assemble(
            binding,
            model=source,
            cache=RegionalLoraExecutionCache(),
        ),
        operation="regional operation global-patch regression",
    )
    derived.add_patches(
        {
            "diffusion_model.linear.weight": (
                "diff",
                (torch.ones_like(source_weight),),
            )
        }
    )

    derived.patch_model(device_to=torch.device("cpu"), load_weights=True)
    try:
        replacement = derived.model.diffusion_model.linear
        assert isinstance(replacement, RegionalLinearOperationPatch)
        assert not torch.equal(replacement.weight, source_before)
        torch.testing.assert_close(source_weight, source_before)
    finally:
        derived.unpatch_model(torch.device("cpu"), unpatch_weights=True)

    assert derived.model.diffusion_model.linear is source.model.diffusion_model.linear
    torch.testing.assert_close(source_weight, source_before)


def test_dynamic_patcher_clone_retains_dynamic_owner_and_pending_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise installed dynamic clone state without requiring native host buffers."""

    class HostBufferStub:
        """Stand in only for unavailable native allocator construction."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Accept the installed constructor surface without allocating."""

            del args, kwargs

    monkeypatch.setattr(comfy_aimdo.host_buffer, "HostBuffer", HostBufferStub)
    source = _dynamic_patcher()
    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        RegionalOperationMutationAssembler().assemble(
            _binding(source),
            model=source,
            cache=RegionalLoraExecutionCache(),
        ),
        operation="regional operation dynamic regression",
    )

    try:
        assert type(derived) is comfy.model_patcher.ModelPatcherDynamic
        assert derived.is_dynamic()
        assert derived.parent is source
        assert derived.model is source.model
        assert derived.model.dynamic_pins is source.model.dynamic_pins
        assert tuple(derived.object_patches) == ("diffusion_model.linear",)
        assert source.object_patches == {}
        original = source.model.diffusion_model.linear
        derived.patch_model(load_weights=False)
        assert isinstance(
            derived.model.diffusion_model.linear,
            RegionalLinearOperationPatch,
        )
        derived.unpatch_model(unpatch_weights=False)
        assert derived.model.diffusion_model.linear is original
    finally:
        derived.detach(unpatch_all=False)
        source.detach(unpatch_all=False)


def _binding(
    patcher: comfy.model_patcher.ModelPatcher,
) -> RegionalLoraBindingResult:
    """Return one complete consumer-spatialized Linear binding."""

    target = ComfyNormalizedAdapterTarget(
        ComfyAdapterTargetPath("diffusion_model.linear.weight", None),
        LoRAAdapter(
            {"up", "down"},
            (
                torch.eye(2),
                torch.eye(2),
                None,
                None,
                None,
                None,
            ),
        ),
        "LoRAAdapter",
        ("down", "up"),
        True,
    )
    plan = RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity("assembly.safetensors"),
        0,
        0,
        RegionalLoraBranch.POSITIVE,
        1.0,
        (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
    )
    resolution = ComfyRegionalLoraResolution(
        (
            ComfyRegionalAdapterResolution(
                plan,
                RegionalLoraHostPayload(False, None, {}, None),
                (target,),
                (),
                (),
            ),
        ),
        (),
    )
    operations = ComfyResolvedOperationTranslator().translate(resolution)
    return RegionalLoraTargetBinder().bind(
        source=patcher,
        candidate=patcher,
        resolution=resolution,
        operations=operations,
        linear_spatial_capabilities={
            "diffusion_model.linear.weight": (
                BoundRegionalLoraSpatialCapability.CONSUMER_SPATIALIZED
            )
        },
    )


def _patcher(*, comfy_linear: bool = False) -> Any:
    """Return one real Comfy patcher around a two-feature target."""

    linear: nn.Module
    if comfy_linear:
        linear = comfy.ops.disable_weight_init.Linear(2, 2, bias=False)
        linear.weight = nn.Parameter(torch.eye(2), requires_grad=False)
    else:
        linear = nn.Linear(2, 2, bias=False)
        linear.weight = nn.Parameter(torch.eye(2), requires_grad=False)
    root = nn.Module()
    root.diffusion_model = nn.Module()
    root.diffusion_model.linear = linear
    device = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(root, device, device)


def _dynamic_patcher() -> comfy.model_patcher.ModelPatcherDynamic:
    """Return one installed dynamic GPU patcher without loading any weights."""

    root = nn.Module()
    root.diffusion_model = nn.Module()
    linear = comfy.ops.disable_weight_init.Linear(
        2,
        2,
        bias=False,
        device="cpu",
    )
    linear.weight = nn.Parameter(
        torch.eye(2),
        requires_grad=False,
    )
    root.diffusion_model.linear = linear
    return comfy.model_patcher.ModelPatcherDynamic(
        root,
        torch.device("cuda:0"),
        torch.device("cpu"),
    )
