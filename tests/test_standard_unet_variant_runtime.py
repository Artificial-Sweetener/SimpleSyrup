# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify complete persistent-variant installation and detach cleanup."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import torch
from comfy.ldm.modules.attention import SpatialTransformer
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import CallbacksMP, WrappersMP
from torch import nn

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.attention_coupling.unet import StandardUnetAttentionBackend
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_lora.comfy_adapter_resolution import (
    ComfyAdapterTargetPath,
    ComfyNormalizedAdapterTarget,
    ComfyRegionalAdapterResolution,
    ComfyRegionalLoraResolution,
)
from simple_syrup.runtime.regional_lora.standard_unet_native_admission import (
    StandardUnetNativeLoraAdmission,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_template import (
    STANDARD_UNET_VARIANT_TEMPLATE_CACHE,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation


class _Diffusion(nn.Module):
    """Expose one conventional target and a standard-style private forward."""

    def __init__(self) -> None:
        """Create one deterministic base projection."""

        super().__init__()
        self.layer = nn.Linear(1, 1, bias=False)
        self.layer.weight.data.fill_(1.0)
        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SpatialTransformer(
                        in_channels=32,
                        n_heads=1,
                        d_head=32,
                        depth=1,
                        context_dim=32,
                        use_checkpoint=False,
                    )
                )
            ]
        )

    def _forward(
        self, inputs: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        """Run the generic target projection."""

        del args, kwargs
        return cast(torch.Tensor, self.layer(inputs))


class _Root(nn.Module):
    """Expose the diffusion module through a Comfy patcher root."""

    def __init__(self) -> None:
        """Create the source diffusion graph."""

        super().__init__()
        self.diffusion_model = _Diffusion()


def test_runtime_reuses_two_exact_template_variants_across_request_detach() -> None:
    """Retain exact shared variants while clearing only request-local state."""

    source = _Root()
    device = torch.device("cpu")
    patcher = ModelPatcher(source, device, device)
    admission, state = _admission_and_state()

    def first_wrapper(function: object, arguments: object) -> object:
        """Represent one request-local upstream model wrapper."""

        del function, arguments
        return None

    patcher.set_model_unet_function_wrapper(first_wrapper)

    built = StandardUnetAttentionBackend().derive(
        model=patcher,
        state=state,
        admission=admission,
    )
    derived = built.model
    assert isinstance(derived, ModelPatcher)
    assert "patches" not in derived.model_options["transformer_options"]
    replacements = derived.model_options["transformer_options"].get(
        "patches_replace",
        {},
    )
    assert isinstance(replacements, dict)
    assert "attn1" not in replacements
    variant_root = derived.object_patches["diffusion_model"]
    assert isinstance(variant_root, nn.Module)
    container = variant_root.simple_syrup_regional_variants
    assert isinstance(container, nn.ModuleDict)
    assert len(container) == 2
    assert (
        len(
            derived.get_wrappers(
                WrappersMP.PREPARE_SAMPLING,
                "simple_syrup.standard_unet_static_variant_residency",
            )
        )
        == 1
    )
    weights = tuple(
        float(cast(_Diffusion, shell).layer.weight.item())
        for shell in container.values()
    )

    assert weights == (2.0, 3.0)
    assert float(source.diffusion_model.layer.weight.item()) == 1.0
    assert derived.model_options["model_function_wrapper"] is first_wrapper

    def second_wrapper(function: object, arguments: object) -> object:
        """Represent a changed request-local upstream model wrapper."""

        del function, arguments
        return None

    patcher.set_model_unet_function_wrapper(second_wrapper)
    static_source = derived.model.diffusion_model
    derived.object_patches_backup["diffusion_model"] = static_source
    derived.model.diffusion_model = variant_root

    second = (
        StandardUnetAttentionBackend()
        .derive(
            model=patcher,
            state=state,
            admission=admission,
        )
        .model
    )
    assert isinstance(second, ModelPatcher)
    assert second.model is derived.model
    assert second.object_patches["diffusion_model"] is variant_root
    assert second.model_options["model_function_wrapper"] is second_wrapper

    callback = derived.get_callbacks(
        CallbacksMP.ON_DETACH,
        "simple_syrup.standard_unet_regional_variants",
    )[0]
    callback(derived, False)
    assert len(container) == 2
    callback(derived, True)

    assert len(container) == 2
    STANDARD_UNET_VARIANT_TEMPLATE_CACHE.clear()


def _admission_and_state() -> tuple[
    StandardUnetNativeLoraAdmission,
    StandardUnetAttentionState,
]:
    """Return paired generic adapter resolution and matching attention state."""

    plans = tuple(
        _adapter(index, region, branch)
        for index, (region, branch) in enumerate(
            (
                (0, RegionalLoraBranch.POSITIVE),
                (1, RegionalLoraBranch.POSITIVE),
                (0, RegionalLoraBranch.NEGATIVE),
                (1, RegionalLoraBranch.NEGATIVE),
            )
        )
    )
    payload_values = (object(), object())
    payloads = (
        RegionalLoraHostPayload.unresolved(payload_values[0]),
        RegionalLoraHostPayload.unresolved(payload_values[1]),
        RegionalLoraHostPayload.unresolved(payload_values[0]),
        RegionalLoraHostPayload.unresolved(payload_values[1]),
    )
    adaptation = RegionalLoraPlanAdaptation(RegionalLoraPlan(plans), payloads)
    results = tuple(
        ComfyRegionalAdapterResolution(
            plan,
            payload,
            (_target(1.0 if plan.region_index == 0 else 2.0),),
            (),
            (),
        )
        for plan, payload in zip(plans, payloads, strict=True)
    )
    admission = StandardUnetNativeLoraAdmission(
        adaptation,
        ComfyRegionalLoraResolution(results, ()),
    )
    masks = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    processed = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _context(0, None),
            (_context(1, 0), _context(2, 1)),
        ),
        ProcessedRegionalAttentionBranch(
            _context(0, None),
            (_context(1, 0), _context(2, 1)),
        ),
        RegionalMaskBank(masks.clone(), masks.clone(), 2, 1),
        adaptation.plan,
    )
    state = StandardUnetAttentionState(
        processed,
        (1.0, 1.0),
        RegionalAttentionDiagnosticsBuilder(processed.mask_bank, backend="generic"),
    )
    return admission, state


def _adapter(
    index: int,
    region: int,
    branch: RegionalLoraBranch,
) -> RegionalLoraAdapterPlan:
    """Return one fixed-strength generic adapter use."""

    return RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity(f"adapter-{region}"),
        index,
        region,
        branch,
        1.0,
        (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
    )


def _target(delta: float) -> ComfyNormalizedAdapterTarget:
    """Return one conventional additive target operation."""

    return ComfyNormalizedAdapterTarget(
        ComfyAdapterTargetPath("diffusion_model.layer.weight", None),
        ("diff", (torch.tensor([[delta]]),)),
        "LegacyDiff",
        ("source",),
        False,
    )


def _context(index: int, region: int | None) -> ProcessedRegionalAttentionContext:
    """Return one always-active generic processed context."""

    return ProcessedRegionalAttentionContext(
        index,
        region,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                ConditioningScheduleRange(None, None, None, None),
                torch.zeros((1, 1, 1)),
                1.0,
            ),
        ),
    )
