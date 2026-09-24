# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the native full-context Attention Coupling node boundary."""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from simple_syrup.nodes_v3.ksampler_attention_coupling import (
    KSamplerAttentionCouplingV3,
)


class _RecordingAttentionService:
    """Record one complete node delegation request."""

    calls: ClassVar[list[dict[str, Any]]] = []
    output: ClassVar[dict[str, Any]] = {"samples": torch.ones((1, 16, 1, 2, 2))}

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        """Return a recognizable latent after recording all values."""

        type(self).calls.append(kwargs)
        return self.output


def test_schema_exposes_stable_full_context_contract_and_guidance() -> None:
    """Declare the stable node id, exact inputs, and regional LoRA policy."""

    schema = KSamplerAttentionCouplingV3.define_schema()

    assert schema.node_id == "SimpleSyrup.KSamplerAttentionCoupling"
    assert schema.display_name == "KSampler (Attention Coupling)"
    assert [item.id for item in schema.inputs] == [
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "region_masks",
        "regional_prompt_weight",
        "region_mask_feather",
        "latent_image",
        "denoise",
    ]
    inputs = {item.id: item for item in schema.inputs}
    assert inputs["region_masks"].optional is True
    assert inputs["regional_prompt_weight"].default == 1.0
    assert [output.id for output in schema.outputs] == ["latent"]
    guidance = " ".join(
        [schema.description, *(item.tooltip for item in schema.inputs)]
    ).lower()
    for required in (
        "anima",
        "sdxl",
        "global lora",
        "regional lora",
        "regional model-side hooks",
        "independent schedules",
        "declared order",
        "runtime scales",
        "unsupported",
        "fail",
    ):
        assert required in guidance


def test_node_delegates_the_complete_request_once() -> None:
    """Keep all orchestration outside the Comfy-facing node class."""

    original = KSamplerAttentionCouplingV3.sampling_service_class
    KSamplerAttentionCouplingV3.sampling_service_class = _RecordingAttentionService  # type: ignore[assignment]
    _RecordingAttentionService.calls = []
    masks = torch.ones((2, 8, 8))
    latent = {"samples": torch.zeros((1, 16, 1, 2, 2))}
    try:
        (output,) = KSamplerAttentionCouplingV3.execute(
            model="model",
            seed=9,
            steps=30,
            cfg=1.5,
            sampler_name="euler",
            scheduler="simple",
            positive="positive-batch",
            negative="negative-batch",
            region_masks=masks,
            regional_prompt_weight=0.8,
            region_mask_feather=12,
            latent_image=latent,
            denoise=0.9,
        )
    finally:
        KSamplerAttentionCouplingV3.sampling_service_class = original

    assert output is _RecordingAttentionService.output
    assert _RecordingAttentionService.calls == [
        {
            "model": "model",
            "seed": 9,
            "steps": 30,
            "cfg": 1.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": "positive-batch",
            "negative": "negative-batch",
            "region_masks": masks,
            "regional_prompt_weight": 0.8,
            "region_mask_feather": 12,
            "latent_image": latent,
            "denoise": 0.9,
        }
    ]


def test_node_forwards_disconnected_masks_as_an_ordinary_request() -> None:
    """Allow the optional mask socket to remain disconnected."""

    original = KSamplerAttentionCouplingV3.sampling_service_class
    KSamplerAttentionCouplingV3.sampling_service_class = _RecordingAttentionService  # type: ignore[assignment]
    _RecordingAttentionService.calls = []
    latent = {"samples": torch.zeros((1, 16, 1, 3, 5))}
    try:
        KSamplerAttentionCouplingV3.execute(
            model="model",
            seed=4,
            steps=20,
            cfg=3.0,
            sampler_name="euler",
            scheduler="normal",
            positive="positive",
            negative="negative",
            latent_image=latent,
            denoise=0.45,
        )
    finally:
        KSamplerAttentionCouplingV3.sampling_service_class = original

    assert _RecordingAttentionService.calls[0]["region_masks"] is None
    assert _RecordingAttentionService.calls[0]["latent_image"] is latent
    assert _RecordingAttentionService.calls[0]["denoise"] == 0.45
