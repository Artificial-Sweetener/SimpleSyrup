# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify isolated automatic NegPiP proof workflow construction."""

from __future__ import annotations

from typing import cast

import pytest

from tools.negpip_integration.workflow import (
    REFINER_BASE_PROMPT,
    REFINER_SWITCH_STEP,
    NegpipFixtureSelections,
    NegpipLiveFamily,
    NegpipLiveWorkflowBuilder,
)


@pytest.fixture
def builder() -> NegpipLiveWorkflowBuilder:
    """Return a builder with deterministic nested Comfy selections."""

    return NegpipLiveWorkflowBuilder(
        NegpipFixtureSelections(
            sd1_checkpoint=r"proof\sd1.safetensors",
            sdxl_checkpoint=r"proof\sdxl.safetensors",
            sdxl_refiner_checkpoint=r"proof\sdxl-refiner.safetensors",
            anima_diffusion=r"proof\anima.safetensors",
            anima_text_encoder=r"proof\anima-te.safetensors",
            krea2_diffusion=r"proof\krea2.safetensors",
            krea2_text_encoder=r"proof\krea2-te.safetensors",
            qwen_image_vae=r"proof\qwen-image-vae.safetensors",
        )
    )


@pytest.mark.parametrize("family", tuple(NegpipLiveFamily))
def test_triggered_workflow_samples_and_requires_runtime_evidence(
    builder: NegpipLiveWorkflowBuilder,
    family: NegpipLiveFamily,
) -> None:
    """Every family uses the public node and synchronized callback observer."""

    built = builder.build(family, run_id=f"proof:{family.value}", trigger=True)
    class_types = [str(node["class_type"]) for node in built.prompt.values()]
    schedule = next(
        node
        for node in built.prompt.values()
        if node["class_type"] == "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl"
    )

    inputs = cast(dict[str, object], schedule["inputs"])
    assert "bright (red:-1.0) jacket" in str(inputs["positive_prompt"])
    expected_sampler = (
        "KSamplerAdvanced" if family is NegpipLiveFamily.SDXL_REFINER else "KSampler"
    )
    assert expected_sampler in class_types
    assert "VAEDecode" in class_types
    assert "SaveImage" in class_types
    assert "SimpleSyrupBenchmark.InstrumentNegpipModel" in class_types
    assert "SimpleSyrupBenchmark.ReadNegpipRuntime" in class_types
    assert built.runtime_node_id is not None
    assert built.conditioning_node_id is not None


@pytest.mark.parametrize("family", tuple(NegpipLiveFamily))
def test_control_workflow_proves_automatic_gate_stays_off(
    builder: NegpipLiveWorkflowBuilder,
    family: NegpipLiveFamily,
) -> None:
    """A no-negative-weight control saves a sampled unmodified image."""

    built = builder.build(family, run_id=f"control:{family.value}", trigger=False)
    class_types = [str(node["class_type"]) for node in built.prompt.values()]

    assert "SimpleSyrupBenchmark.SnapshotModelModifier" in class_types
    expected_sampler = (
        "KSamplerAdvanced" if family is NegpipLiveFamily.SDXL_REFINER else "KSampler"
    )
    assert expected_sampler in class_types
    assert "VAEDecode" in class_types
    assert "SaveImage" in class_types
    assert "SimpleSyrupBenchmark.InstrumentNegpipModel" not in class_types
    assert built.runtime_node_id is None
    assert built.conditioning_node_id is None


def test_family_workflows_select_native_loaders_and_latents(
    builder: NegpipLiveWorkflowBuilder,
) -> None:
    """Use real family-specific loader and latent contracts."""

    expected = {
        NegpipLiveFamily.SD1: {"CheckpointLoaderSimple", "EmptyLatentImage"},
        NegpipLiveFamily.SDXL: {"CheckpointLoaderSimple", "EmptyLatentImage"},
        NegpipLiveFamily.SDXL_REFINER: {
            "CheckpointLoaderSimple",
            "EmptyLatentImage",
        },
        NegpipLiveFamily.ANIMA: {
            "SimpleSyrup.SimpleLoadAnima",
            "EmptyCosmosLatentVideo",
        },
        NegpipLiveFamily.KREA2: {
            "UNETLoader",
            "CLIPLoader",
            "VAELoader",
            "EmptySD3LatentImage",
        },
    }

    for family, required in expected.items():
        built = builder.build(family, run_id=family.value, trigger=True)
        class_types = {str(node["class_type"]) for node in built.prompt.values()}
        assert required.issubset(class_types)


def test_refiner_workflow_runs_base_then_refiner_sampling(
    builder: NegpipLiveWorkflowBuilder,
) -> None:
    """Use SDXL base for high noise and the probed refiner for low noise."""

    built = builder.build(
        NegpipLiveFamily.SDXL_REFINER,
        run_id="refiner",
        trigger=True,
    )
    samplers = [
        node
        for node in built.prompt.values()
        if node["class_type"] == "KSamplerAdvanced"
    ]

    assert len(samplers) == 2
    base_inputs = cast(dict[str, object], samplers[0]["inputs"])
    refiner_inputs = cast(dict[str, object], samplers[1]["inputs"])
    assert base_inputs["add_noise"] == "enable"
    assert base_inputs["end_at_step"] == REFINER_SWITCH_STEP
    assert base_inputs["return_with_leftover_noise"] == "enable"
    assert refiner_inputs["add_noise"] == "disable"
    assert refiner_inputs["start_at_step"] == REFINER_SWITCH_STEP
    assert refiner_inputs["end_at_step"] == 24
    base_positive = next(
        node
        for node in built.prompt.values()
        if node["class_type"] == "CLIPTextEncode"
        and cast(dict[str, object], node["inputs"])["text"] == REFINER_BASE_PROMPT
    )
    base_text = cast(dict[str, object], base_positive["inputs"])["text"]
    assert isinstance(base_text, str)
    assert "red" not in base_text


def test_ppm_baseline_prepatches_the_same_schedule_path(
    builder: NegpipLiveWorkflowBuilder,
) -> None:
    """Put pinned PPM before Schedule & Encode as the behavioral oracle."""

    built = builder.build(
        NegpipLiveFamily.SD1,
        run_id="ppm-baseline",
        trigger=True,
        baseline_ppm=True,
    )
    class_types = [str(node["class_type"]) for node in built.prompt.values()]

    assert built.mode == "ppm_baseline"
    assert class_types.count("CLIPNegPip") == 1
    assert (
        class_types.count("SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl") == 1
    )


def test_ppm_baseline_rejects_an_untriggered_workflow(
    builder: NegpipLiveWorkflowBuilder,
) -> None:
    """Keep ordinary controls free from every NegPiP patch."""

    with pytest.raises(ValueError, match="requires a negative weight"):
        builder.build(
            NegpipLiveFamily.SD1,
            run_id="invalid",
            trigger=False,
            baseline_ppm=True,
        )
