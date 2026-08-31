# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the native Contextual Attention Coupling v3 node boundary."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import torch

from simple_syrup.domain.context_segs import ContextSegmentSequence
from simple_syrup.nodes_v3.ksampler_contextual_attention_coupling import (
    KSamplerContextualAttentionCouplingV3,
)
from simple_syrup.services.contextual_diffusion_sampling_service import (
    ContextualDiffusionSamplingResult,
)


class _RecordingContextualAttentionService:
    """Record complete combined requests and expose controlled failure."""

    calls: ClassVar[list[dict[str, Any]]] = []
    output = ContextualDiffusionSamplingResult(
        {"samples": torch.ones((1, 16, 1, 4, 8))},
        ((32, 64), ContextSegmentSequence(())),
    )
    failure: ClassVar[Exception | None] = None

    def sample(self, **kwargs: Any) -> ContextualDiffusionSamplingResult:
        """Record one request, then return or raise the configured result."""

        type(self).calls.append(kwargs)
        failure = type(self).failure
        if failure is not None:
            raise failure
        return type(self).output


def test_schema_exposes_stable_contextual_attention_coupling_contract() -> None:
    """Declare exact workflow inputs, outputs, defaults, and LoRA guidance."""

    schema = KSamplerContextualAttentionCouplingV3.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert schema.node_id == "SimpleSyrup.KSamplerAttentionCouplingContextual"
    assert schema.display_name == (
        "KSampler (Attention Coupling + Contextual Diffusion)"
    )
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
        "latent_context_size",
        "latent_context_overlap",
        "latent_context_batch_size",
        "global_weight",
        "global_steps",
        "global_decay",
        "segs",
    ]
    assert [output.id for output in schema.outputs] == ["latent", "contexts_segs"]
    assert inputs["segs"].optional
    assert inputs["diffusion_mode"].options == [
        "multidiffusion",
        "mixture_of_diffusers",
    ]
    assert inputs["latent_context_size"].default == 96
    assert inputs["latent_context_overlap"].default == 32
    assert inputs["latent_context_batch_size"].default == 4
    assert inputs["global_weight"].default == 1.0
    assert inputs["global_steps"].default == 1
    assert inputs["global_decay"].default == 0.5
    assert inputs["regional_prompt_weight"].default == 1.0
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
        "full quality",
        "inactive",
        "local",
        "reduced-global",
        "segs",
        "unsupported",
        "fail",
    ):
        assert required in guidance


def test_node_delegates_every_contextual_attention_input_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep preparation, planning, correction, and sampling outside the node."""

    monkeypatch.setattr(
        KSamplerContextualAttentionCouplingV3,
        "sampling_service_class",
        _RecordingContextualAttentionService,
    )
    _RecordingContextualAttentionService.calls = []
    _RecordingContextualAttentionService.failure = None
    masks = torch.ones((2, 32, 64))
    latent = {"samples": torch.zeros((1, 16, 1, 4, 8))}
    segs = object()

    output, contexts = KSamplerContextualAttentionCouplingV3.execute(
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
        latent_context_size=112,
        latent_context_overlap=24,
        latent_context_batch_size=3,
        global_weight=0.75,
        global_steps=4,
        global_decay=0.25,
        segs=segs,
    )

    assert output is _RecordingContextualAttentionService.output.latent
    assert contexts is _RecordingContextualAttentionService.output.contexts
    assert _RecordingContextualAttentionService.calls == [
        {
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
            "diffusion_mode": "mixture_of_diffusers",
            "latent_context_size": 112,
            "latent_context_overlap": 24,
            "latent_context_batch_size": 3,
            "global_weight": 0.75,
            "global_steps": 4,
            "global_decay": 0.25,
            "segs": segs,
        }
    ]


def test_node_preserves_actionable_service_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate the combined application's validated failure unchanged."""

    monkeypatch.setattr(
        KSamplerContextualAttentionCouplingV3,
        "sampling_service_class",
        _RecordingContextualAttentionService,
    )
    _RecordingContextualAttentionService.calls = []
    _RecordingContextualAttentionService.failure = ValueError(
        "unsupported Anima regional LoRA target"
    )

    with pytest.raises(ValueError, match="unsupported Anima regional LoRA target"):
        KSamplerContextualAttentionCouplingV3.execute(
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

    _RecordingContextualAttentionService.failure = None
