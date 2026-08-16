# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize the matched SDXL no-LoRA comparison graphs."""

from __future__ import annotations

from typing import cast

from tools.comfy_api import JsonObject
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_no_lora_comparison import (
    build_no_lora_comparison_workflows,
    without_image_outputs,
)


def test_comparison_uses_matched_locked_controls_without_any_lora() -> None:
    """Keep graph families distinct while matching prompts and sampling settings."""

    prompts = _prompts()
    built = build_no_lora_comparison_workflows(
        run_id="comparison",
        checkpoint_name="checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        prompts=prompts,
    )
    plain_types = _types(built.plain.prompt)
    regional_types = _types(built.regional.prompt)

    assert plain_types.count("KSampler") == 1
    assert "SimpleSyrup.KSamplerAttentionCoupling" not in plain_types
    assert regional_types.count("SimpleSyrup.KSamplerAttentionCoupling") == 1
    assert "KSampler" not in regional_types
    assert not any("Lora" in value for value in plain_types + regional_types)
    assert "CreateHookLora" not in regional_types
    assert built.plain_positive_g == "global positive, left positive, right positive"
    assert built.plain_negative_g == "global negative, left negative, right negative"
    assert "[SEP]" not in built.plain_positive_g
    assert ":" not in _save_prefix(built.plain.prompt)
    assert ":" not in _save_prefix(built.regional.prompt)
    _assert_locked_sampler(built.plain.prompt, "KSampler")
    _assert_locked_sampler(
        built.regional.prompt,
        "SimpleSyrup.KSamplerAttentionCoupling",
    )


def test_warmups_execute_sampling_without_creating_images() -> None:
    """Keep warmup graph work matched without hidden decoded artifacts."""

    built = build_no_lora_comparison_workflows(
        run_id="comparison",
        checkpoint_name="checkpoint.safetensors",
        mask_names=("left.png", "right.png"),
        prompts=_prompts(),
    )

    for source in (built.plain.prompt, built.regional.prompt):
        warmed = without_image_outputs(source)
        types = _types(warmed)
        assert "VAEDecode" not in types
        assert "SaveImage" not in types
        assert "SimpleSyrupBenchmark.ReadMetrics" in types
        assert any("KSampler" in value for value in types)


def _assert_locked_sampler(prompt: dict[str, JsonObject], class_type: str) -> None:
    """Require the same seed, step, CFG, sampler, and scheduler controls."""

    sampler = next(node for node in prompt.values() if node["class_type"] == class_type)
    inputs = cast(JsonObject, sampler["inputs"])
    assert inputs["seed"] == 7429113057
    assert inputs["steps"] == 30
    assert inputs["cfg"] == 5.0
    assert inputs["sampler_name"] == "euler_ancestral"
    assert inputs["scheduler"] == "karras"
    if class_type != "KSampler":
        assert inputs["regional_prompt_weight"] == 1.0


def _types(prompt: dict[str, JsonObject]) -> tuple[str, ...]:
    """Return exact graph node classes."""

    return tuple(str(node["class_type"]) for node in prompt.values())


def _save_prefix(prompt: dict[str, JsonObject]) -> str:
    """Return the sole filesystem-facing SaveImage prefix."""

    saved = next(node for node in prompt.values() if node["class_type"] == "SaveImage")
    inputs = cast(JsonObject, saved["inputs"])
    prefix = inputs["filename_prefix"]
    if not isinstance(prefix, str):
        raise AssertionError("SaveImage prefix must be text.")
    return prefix


def _prompts() -> SdxlVisualPromptSet:
    """Return distinct external-like prompt sections without inventory identity."""

    return SdxlVisualPromptSet(
        base_positive_g="global positive",
        base_positive_l="global positive",
        base_negative_g="global negative",
        base_negative_l="global negative",
        left_positive_g="left positive",
        left_positive_l="left positive",
        right_positive_g="right positive",
        right_positive_l="right positive",
        left_negative_g="left negative",
        left_negative_l="left negative",
        right_negative_g="right negative",
        right_negative_l="right negative",
    )
