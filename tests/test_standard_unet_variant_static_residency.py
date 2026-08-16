# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact static residency for complete standard-UNet variant roots."""

from __future__ import annotations

import pytest
import torch
from comfy import model_management
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
from torch import nn

from simple_syrup.runtime.regional_lora.standard_unet_variant_residency_handoff import (
    StandardUnetVariantResidencyHandoff,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_static_residency import (
    StandardUnetStaticVariantResidency,
)


class _Root(nn.Module):
    """Expose a replaceable diffusion module and one retained parameter."""

    def __init__(self, width: int) -> None:
        """Create deterministic source and non-diffusion weights."""

        super().__init__()
        self.diffusion_model = nn.Linear(width, width, bias=False)
        self.retained = nn.Parameter(torch.ones(3))


def test_static_residency_publishes_enlarged_size_and_wrapper() -> None:
    """Make Comfy account for the complete replacement before model loading."""

    source = _Root(2)
    patcher = _patcher(source)
    replacement = nn.Sequential(source.diffusion_model, nn.Linear(4, 4, bias=False))
    residency = StandardUnetStaticVariantResidency(
        patcher,
        source.diffusion_model,
        replacement,
    )

    residency.apply(patcher)

    assert patcher.model_size() == residency.required_bytes
    assert residency.required_bytes > model_management.module_size(source)
    assert (
        len(
            patcher.get_wrappers(
                WrappersMP.PREPARE_SAMPLING,
                "simple_syrup.standard_unet_static_variant_residency",
            )
        )
        == 1
    )


@pytest.mark.parametrize(
    ("capacity_delta", "force_offload", "expected_full_load"),
    ((1, False, True), (-1, False, False), (1, True, False)),
)
def test_static_residency_forwards_capacity_and_offload_policy(
    monkeypatch: pytest.MonkeyPatch,
    capacity_delta: int,
    force_offload: bool,
    expected_full_load: bool,
) -> None:
    """Force only safe static residency and preserve explicit offload."""

    source = _Root(2)
    patcher = _patcher(source)
    patcher.load_device = torch.device("cuda")
    residency = StandardUnetStaticVariantResidency(
        patcher,
        source.diffusion_model,
        nn.Linear(4, 4, bias=False),
        StandardUnetVariantResidencyHandoff(lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        model_management,
        "maximum_vram_for_weights",
        lambda _device: residency.required_bytes + capacity_delta,
    )
    captured: dict[str, object] = {}

    def executor(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "prepared"

    result = residency.prepare_sampling(
        executor,
        patcher,
        (1, 4, 8, 8),
        {"positive": ()},
        force_full_load=False,
        force_offload=force_offload,
    )

    assert result == "prepared"
    forwarded = captured["kwargs"]
    assert isinstance(forwarded, dict)
    assert forwarded["force_offload"] is force_offload
    assert forwarded["force_full_load"] is expected_full_load


def test_static_residency_hands_off_before_host_preparation() -> None:
    """Release a different regional allocation before Comfy loads this root."""

    events: list[str] = []

    def unload(_model: ModelPatcher, **_kwargs: object) -> None:
        events.append("unload")

    handoff = StandardUnetVariantResidencyHandoff(unload)
    first_source = _Root(2)
    first = _patcher(first_source)
    first.load_device = torch.device("cuda:0")
    second_source = _Root(2)
    second = _patcher(second_source)
    second.load_device = torch.device("cuda:0")
    first_residency = StandardUnetStaticVariantResidency(
        first,
        first_source.diffusion_model,
        nn.Linear(4, 4, bias=False),
        handoff,
    )
    second_residency = StandardUnetStaticVariantResidency(
        second,
        second_source.diffusion_model,
        nn.Linear(4, 4, bias=False),
        handoff,
    )

    def prepare(*_args: object, **_kwargs: object) -> str:
        events.append("prepare")
        return "prepared"

    first_residency.prepare_sampling(
        prepare,
        first,
        (1, 4, 8, 8),
        {},
        force_offload=True,
    )
    events.clear()
    result = second_residency.prepare_sampling(
        prepare,
        second,
        (1, 4, 8, 8),
        {},
        force_offload=True,
    )

    assert result == "prepared"
    assert events == ["unload", "prepare"]


def _patcher(root: nn.Module) -> ModelPatcher:
    """Return one ordinary static Comfy patcher."""

    device = torch.device("cpu")
    return ModelPatcher(root, device, device)
