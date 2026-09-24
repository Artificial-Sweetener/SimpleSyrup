# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify branch-neutral persistent standard-UNet variant topology."""

from __future__ import annotations

import pytest

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.runtime.regional_lora.comfy_adapter_resolution import (
    ComfyAdapterTargetPath,
    ComfyNormalizedAdapterTarget,
    ComfyRegionalAdapterResolution,
    ComfyRegionalLoraResolution,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_topology import (
    StandardUnetVariantTopologyBuilder,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload


def test_cfg_pairs_collapse_into_ordered_region_variants() -> None:
    """Preserve adapter order while eliminating duplicate CFG model work."""

    payloads = (object(), object())
    resolution = ComfyRegionalLoraResolution(
        (
            _result(0, 1, "first", RegionalLoraBranch.POSITIVE, payloads[0]),
            _result(1, 0, "second", RegionalLoraBranch.POSITIVE, payloads[1]),
            _result(2, 1, "first", RegionalLoraBranch.NEGATIVE, payloads[0]),
            _result(3, 0, "second", RegionalLoraBranch.NEGATIVE, payloads[1]),
        ),
        (),
    )

    topology = StandardUnetVariantTopologyBuilder().build(resolution)

    assert tuple(variant.region_index for variant in topology.variants) == (0, 1)
    assert topology.variants[0].adapters[0].composition_index == 1
    assert topology.variants[1].adapters[0].composition_index == 0


@pytest.mark.parametrize("difference", ["region", "payload", "target"])
def test_cfg_pair_asymmetry_fails_closed(difference: str) -> None:
    """Reject branch-dependent model weights before persistent construction."""

    shared = object()
    positive = _result(0, 0, "adapter", RegionalLoraBranch.POSITIVE, shared)
    negative = _result(
        1,
        1 if difference == "region" else 0,
        "adapter",
        RegionalLoraBranch.NEGATIVE,
        object() if difference == "payload" else shared,
        target_name="other.weight" if difference == "target" else "layer.weight",
    )

    with pytest.raises(ValueError, match="CFG pair|CFG pairs"):
        StandardUnetVariantTopologyBuilder().build(
            ComfyRegionalLoraResolution((positive, negative), ())
        )


def _result(
    composition_index: int,
    region_index: int,
    identity: str,
    branch: RegionalLoraBranch,
    payload_weights: object,
    *,
    target_name: str = "layer.weight",
) -> ComfyRegionalAdapterResolution:
    """Return one model-neutral resolved adapter use."""

    plan = RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity(identity),
        composition_index,
        region_index,
        branch,
        1.0,
        (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
    )
    payload = RegionalLoraHostPayload.unresolved(payload_weights)
    target = ComfyNormalizedAdapterTarget(
        ComfyAdapterTargetPath(f"diffusion_model.{target_name}", None),
        object(),
        "HostOperation",
        (identity,),
        False,
    )
    return ComfyRegionalAdapterResolution(plan, payload, (target,), (), ())
