# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the native tiled Attention Coupling v3 node boundary."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import torch

from simple_syrup.nodes_v3.ksampler_tiled_attention_coupling import (
    KSamplerTiledAttentionCouplingV3,
)


class _RecordingTiledAttentionService:
    """Record complete tiled node requests and expose controlled failure."""

    calls: ClassVar[list[dict[str, Any]]] = []
    output: ClassVar[dict[str, Any]] = {"samples": torch.ones((1, 16, 1, 4, 8))}
    failure: ClassVar[Exception | None] = None

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        """Record one request, then return or raise the configured result."""

        type(self).calls.append(kwargs)
        failure = type(self).failure
        if failure is not None:
            raise failure
        return type(self).output


def test_schema_exposes_stable_tiled_attention_coupling_contract() -> None:
    """Declare exact workflow inputs, defaults, output, and LoRA guidance."""

    schema = KSamplerTiledAttentionCouplingV3.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert schema.node_id == "SimpleSyrup.KSamplerAttentionCouplingTiled"
    assert schema.display_name == "KSampler (Attention Coupling + Tiled Diffusion)"
    assert schema.category == "SimpleSyrup/Sampling"
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
        "diffusion_mode",
        "latent_tile_width",
        "latent_tile_height",
        "latent_tile_overlap",
        "latent_tile_batch_size",
    ]
    assert [output.id for output in schema.outputs] == ["latent"]
    assert inputs["diffusion_mode"].options == [
        "multidiffusion",
        "mixture_of_diffusers",
    ]
    assert inputs["diffusion_mode"].default == "multidiffusion"
    assert inputs["latent_tile_width"].default == 128
    assert inputs["latent_tile_height"].default == 128
    assert inputs["latent_tile_overlap"].default == 16
    assert inputs["latent_tile_batch_size"].default == 4
    assert all(item.tooltip for item in (*schema.inputs, *schema.outputs))
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
        "inactive",
        "quality",
        "multidiffusion",
        "mixture of diffusers",
        "unsupported",
        "fail",
    ):
        assert required in guidance


def test_node_delegates_every_tiled_attention_input_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep preparation, pruning, sampling, and fusion outside the node."""

    monkeypatch.setattr(
        KSamplerTiledAttentionCouplingV3,
        "sampling_service_class",
        _RecordingTiledAttentionService,
    )
    _RecordingTiledAttentionService.calls = []
    _RecordingTiledAttentionService.failure = None
    masks = torch.ones((2, 32, 64))
    latent = {"samples": torch.zeros((1, 16, 1, 4, 8))}

    (output,) = KSamplerTiledAttentionCouplingV3.execute(
        model="model",
        seed=17,
        steps=28,
        cfg=1.25,
        sampler_name="euler",
        scheduler="simple",
        positive="positive-batch",
        negative="negative-batch",
        region_masks=masks,
        regional_prompt_weight=0.85,
        region_mask_feather=10,
        latent_image=latent,
        denoise=0.9,
        diffusion_mode="mixture_of_diffusers",
        latent_tile_width=96,
        latent_tile_height=80,
        latent_tile_overlap=24,
        latent_tile_batch_size=3,
    )

    assert output is _RecordingTiledAttentionService.output
    assert _RecordingTiledAttentionService.calls == [
        {
            "diffusion_mode": "mixture_of_diffusers",
            "model": "model",
            "seed": 17,
            "steps": 28,
            "cfg": 1.25,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": "positive-batch",
            "negative": "negative-batch",
            "region_masks": masks,
            "regional_prompt_weight": 0.85,
            "region_mask_feather": 10,
            "latent_image": latent,
            "denoise": 0.9,
            "latent_tile_width": 96,
            "latent_tile_height": 80,
            "latent_tile_overlap": 24,
            "latent_tile_batch_size": 3,
            "preview_context": None,
            "differential_diffusion": False,
        }
    ]


def test_node_preserves_actionable_service_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate the application boundary's validated failure unchanged."""

    monkeypatch.setattr(
        KSamplerTiledAttentionCouplingV3,
        "sampling_service_class",
        _RecordingTiledAttentionService,
    )
    _RecordingTiledAttentionService.calls = []
    _RecordingTiledAttentionService.failure = ValueError(
        "unsupported Anima regional LoRA target"
    )

    with pytest.raises(ValueError, match="unsupported Anima regional LoRA target"):
        KSamplerTiledAttentionCouplingV3.execute(
            model="model",
            seed=17,
            steps=28,
            cfg=1.25,
            sampler_name="euler",
            scheduler="simple",
            positive="positive-batch",
            negative="negative-batch",
            region_masks=torch.ones((1, 8, 8)),
            regional_prompt_weight=0.85,
            region_mask_feather=10,
            latent_image={"samples": torch.zeros((1, 16, 1, 2, 2))},
        )

    _RecordingTiledAttentionService.failure = None
