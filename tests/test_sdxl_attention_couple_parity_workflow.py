# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the bounded standard-UNet parity graph contracts."""

from __future__ import annotations

from pathlib import Path

from sdxl_visual_test_inventory import visual_inventory

from tools.comfy_api import JsonObject
from tools.sdxl_attention_couple_parity.cases import (
    SdxlAttentionCoupleParityCase,
    parity_case,
)
from tools.sdxl_attention_couple_parity.global_style import ParityGlobalStyle
from tools.sdxl_attention_couple_parity.workflow import (
    BuiltParityWorkflow,
    ParityBackend,
    build_parity_workflow,
)


def test_parity_graphs_share_native_sdxl_controls_and_only_change_backend(
    tmp_path: Path,
) -> None:
    """Keep prompts and sampling equal while selecting two execution owners."""

    visual_inventory(tmp_path)
    case = parity_case()
    reference = _build(ParityBackend.REFERENCE, case)
    candidate = _build(ParityBackend.CANDIDATE, case)

    assert _inputs(reference.prompt, "CLIPTextEncodeSDXL") == _inputs(
        candidate.prompt, "CLIPTextEncodeSDXL"
    )
    assert all(
        inputs["target_width"] == 1536 and inputs["target_height"] == 1536
        for inputs in _inputs(reference.prompt, "CLIPTextEncodeSDXL")
        if isinstance(inputs, dict)
    )
    assert _inputs(reference.prompt, "EmptyLatentImage") == _inputs(
        candidate.prompt, "EmptyLatentImage"
    )
    reference_sampler = _sole(reference.prompt, "KSampler")
    candidate_sampler = _sole(candidate.prompt, "SimpleSyrup.KSamplerAttentionCoupling")
    for key in (
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "latent_image",
        "denoise",
    ):
        assert reference_sampler[key] == candidate_sampler[key]
    assert candidate_sampler["regional_prompt_weight"] == 0.4
    assert candidate_sampler["region_mask_feather"] == 0
    solid_masks = _inputs(reference.prompt, "SolidMask")
    assert [mask["value"] for mask in solid_masks if isinstance(mask, dict)] == [
        0.6,
        0.4,
    ]
    assert all(
        isinstance(mask, dict) and mask["width"] == 1536 and mask["height"] == 1536
        for mask in solid_masks
    )
    composites = _inputs(reference.prompt, "MaskComposite")
    assert len(composites) == 2
    assert all(
        isinstance(inputs, dict) and inputs["operation"] == "multiply"
        for inputs in composites
    )


def test_parity_graphs_load_no_adapter_nodes(tmp_path: Path) -> None:
    """Prevent model or adapter identity from entering the prompt-only control."""

    visual_inventory(tmp_path)
    case = parity_case()
    for backend in ParityBackend:
        built = _build(backend, case)
        classes = {str(node["class_type"]) for node in built.prompt.values()}
        assert not any("lora" in class_name.casefold() for class_name in classes)
    reference = _build(ParityBackend.REFERENCE, case)
    assert "AttentionCouplePPM" in {
        str(node["class_type"]) for node in reference.prompt.values()
    }


def test_global_style_is_identical_and_base_only_across_backends() -> None:
    """Apply one global adapter equally without leaking triggers into regions."""

    style = ParityGlobalStyle(
        lora_name=r"owned\style.safetensors",
        strength=0.65,
        prompt_g="anonymous style G",
        prompt_l="anonymous style L",
    )
    reference = _build(ParityBackend.REFERENCE, parity_case(), global_style=style)
    candidate = _build(ParityBackend.CANDIDATE, parity_case(), global_style=style)

    assert _inputs(reference.prompt, "LoraLoader") == _inputs(
        candidate.prompt, "LoraLoader"
    )
    loader = _sole(reference.prompt, "LoraLoader")
    assert loader["strength_model"] == 0.65
    assert loader["strength_clip"] == 0.65
    assert loader["lora_name"] == r"owned\style.safetensors"
    for built in (reference, candidate):
        encodes = _inputs(built.prompt, "CLIPTextEncodeSDXL")
        assert len(encodes) == 6
        assert isinstance(encodes[0], dict)
        assert "anonymous style G" in str(encodes[0]["text_g"])
        assert "anonymous style L" in str(encodes[0]["text_l"])
        assert all(
            "anonymous style G" not in str(inputs["text_g"])
            and "anonymous style L" not in str(inputs["text_l"])
            for inputs in encodes[1:]
            if isinstance(inputs, dict)
        )


def _build(
    backend: ParityBackend,
    case: SdxlAttentionCoupleParityCase,
    *,
    global_style: ParityGlobalStyle | None = None,
) -> BuiltParityWorkflow:
    """Build one backend graph through fixed anonymous test controls."""

    return build_parity_workflow(
        backend=backend,
        run_id="run",
        checkpoint_name=r"owned\checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        case=case,
        global_style=global_style,
    )


def _inputs(prompt: dict[str, JsonObject], class_type: str) -> list[object]:
    """Return ordered inputs for every node of one class."""

    return [
        node["inputs"] for node in prompt.values() if node["class_type"] == class_type
    ]


def _sole(prompt: dict[str, JsonObject], class_type: str) -> JsonObject:
    """Return the sole node input object for one class."""

    values = _inputs(prompt, class_type)
    assert len(values) == 1
    value = values[0]
    assert isinstance(value, dict)
    return value
