# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify fail-closed native standard-UNet adapter admission."""

from __future__ import annotations

from typing import ClassVar

import pytest

from simple_syrup.domain.regional_lora_plan import (
    EMPTY_REGIONAL_LORA_PLAN,
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
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
from simple_syrup.runtime.regional_lora.standard_unet_native_admission import (
    StandardUnetNativeLoraAdmission,
    StandardUnetNativeLoraAdmissionService,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation


class _Resolver:
    """Return predeclared installed-Comfy resolution evidence."""

    result: ClassVar[ComfyRegionalLoraResolution]
    calls: ClassVar[list[tuple[RegionalLoraPlanAdaptation, object]]] = []

    def resolve(
        self,
        adaptation: RegionalLoraPlanAdaptation,
        *,
        model: object,
    ) -> ComfyRegionalLoraResolution:
        """Record and return one resolution."""

        type(self).calls.append((adaptation, model))
        return type(self).result


def test_empty_adaptation_skips_resolution() -> None:
    """Keep prompt-only standard-UNet preparation allocation-free."""

    adaptation = RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ())
    _Resolver.calls = []

    admitted = StandardUnetNativeLoraAdmissionService(_Resolver()).admit(
        "model", adaptation
    )

    assert isinstance(admitted, StandardUnetNativeLoraAdmission)
    assert admitted.adaptation is adaptation
    assert admitted.resolution is None
    assert _Resolver.calls == []


def test_complete_native_target_resolution_is_retained() -> None:
    """Preserve host-normalized target evidence without operation translation."""

    adaptation = _adaptation()
    _Resolver.result = _resolution(adaptation)
    _Resolver.calls = []

    admitted = StandardUnetNativeLoraAdmissionService(_Resolver()).admit(
        "model", adaptation
    )

    assert admitted.resolution is _Resolver.result
    assert _Resolver.calls == [(adaptation, "model")]


@pytest.mark.parametrize(
    "code",
    [
        ComfyAdapterResolutionIssueCode.RESOLUTION_FAILED,
        ComfyAdapterResolutionIssueCode.INVALID_TARGET_PATH,
    ],
)
def test_unusable_native_resolution_fails_closed(
    code: ComfyAdapterResolutionIssueCode,
) -> None:
    """Reject host resolution failures before model loading or mutation."""

    adaptation = _adaptation()
    _Resolver.result = _resolution(adaptation, issue_code=code)

    with pytest.raises(ValueError, match="native adapter admission failed"):
        StandardUnetNativeLoraAdmissionService(_Resolver()).admit("model", adaptation)


def test_model_active_adapter_without_targets_fails_closed() -> None:
    """Reject a nonzero model hook that Comfy cannot apply to the MODEL."""

    adaptation = _adaptation()
    _Resolver.result = _resolution(adaptation, include_target=False)

    with pytest.raises(ValueError, match="resolved no model targets"):
        StandardUnetNativeLoraAdmissionService(_Resolver()).admit("model", adaptation)


def test_native_host_operations_are_not_rejected_by_operation_route_policy() -> None:
    """Let installed Comfy execute target forms supported by native patching."""

    adaptation = _adaptation()
    _Resolver.result = _resolution(
        adaptation,
        issue_code=ComfyAdapterResolutionIssueCode.UNSUPPORTED_OPERATION,
    )

    admitted = StandardUnetNativeLoraAdmissionService(_Resolver()).admit(
        "model", adaptation
    )

    assert admitted.resolution is _Resolver.result


def _adaptation() -> RegionalLoraPlanAdaptation:
    """Return one generic model-active adapter use."""

    adapter = RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity("adapter-token"),
        0,
        0,
        RegionalLoraBranch.POSITIVE,
        1.0,
        (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
    )
    payload = RegionalLoraHostPayload.unresolved({"source": object()})
    return RegionalLoraPlanAdaptation(RegionalLoraPlan((adapter,)), (payload,))


def _resolution(
    adaptation: RegionalLoraPlanAdaptation,
    *,
    include_target: bool = True,
    issue_code: ComfyAdapterResolutionIssueCode | None = None,
) -> ComfyRegionalLoraResolution:
    """Return generic resolution evidence for one adapted use."""

    adapter = adaptation.plan.adapters[0]
    payload = adaptation.adapter_payloads[0]
    targets = (
        (
            ComfyNormalizedAdapterTarget(
                ComfyAdapterTargetPath("diffusion_model.layer.weight", None),
                object(),
                "HostNativeOperation",
                ("source",),
                False,
            ),
        )
        if include_target
        else ()
    )
    issues = (
        (
            ComfyAdapterResolutionIssue(
                adapter.composition_index,
                adapter.adapter_identity.value,
                issue_code,
                "generic issue",
            ),
        )
        if issue_code is not None
        else ()
    )
    result = ComfyRegionalAdapterResolution(
        adapter,
        payload,
        targets,
        (),
        issues,
    )
    return ComfyRegionalLoraResolution((result,), issues)
