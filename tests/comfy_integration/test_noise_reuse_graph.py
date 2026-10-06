# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove receiver planning, strict image provenance and source cache dependencies."""

from copy import deepcopy
from typing import Any

import pytest

from simple_syrup.runtime.noise_reuse_graph import (
    rewrite_noise_reuse_graph,
    source_lineage,
)


def graph() -> dict[str, Any]:
    """Build shared source branches without invoking model IO."""
    return {
        "source": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["loader", 0],
                "latent_image": ["empty", 0],
                "sampler_name": "euler",
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["source", 0], "vae": ["loader", 2]},
        },
        "recover": {
            "class_type": "SimpleSyrup.LatentFromImage",
            "inputs": {"image": ["decode", 0]},
        },
        "upscale": {
            "class_type": "SimpleSyrup.SimpleUpscaleLatent",
            "inputs": {"latent": ["recover", 0], "method": "Auto", "scale_factor": 3.0},
        },
        "reuse": {
            "class_type": "SimpleSyrup.NoiseReuseOptions",
            "inputs": {"enabled": True, "finishing_steps": 0},
        },
        "receive": {
            "class_type": "SimpleSyrup.KSampler",
            "inputs": {
                "model": ["loader", 0],
                "latent_image": ["upscale", 0],
                "steps": 16,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.2,
                "options": ["reuse", 0],
            },
        },
    }


def test_image_bridge_resolves_original_latent_and_both_consumers_share_capture() -> (
    None
):
    """Coordinate before source execution and preserve direct latent consumers."""
    prompt = graph()
    prompt["finish"] = {
        "class_type": "SimpleSyrup.NoiseReuseOptions",
        "inputs": {"finishing_steps": 1},
    }
    prompt["receive_finish"] = deepcopy(prompt["receive"])
    prompt["receive_finish"]["inputs"]["options"] = ["finish", 0]
    assert source_lineage(prompt, ["upscale", 0], {}) == ("source", '[{"scale": 3.0}]')
    rewrite_noise_reuse_graph(prompt, {})
    captures = [
        node
        for node in prompt.values()
        if node["class_type"] == "SimpleSyrup.TrajectoryCaptureModel"
    ]
    requests = [
        node
        for node in prompt.values()
        if node["class_type"] == "SimpleSyrup.NoiseCaptureRequest"
    ]
    assert len(captures) == 1 and len(requests) == 2
    assert requests[1]["inputs"]["previous"] == ["receive.noise_capture_request", 0]
    assert prompt["source"]["inputs"]["model"] == ["source.trajectory_capture", 0]
    assert prompt["decode"]["inputs"]["samples"] == ["source.sampling_provenance", 0]
    assert prompt["source.sampling_provenance"]["inputs"]["latent"] == ["source", 0]
    assert requests[0]["inputs"]["source_latent"] == ["empty", 0]
    assert not any(node["class_type"] == "VAEEncode" for node in prompt.values())


def test_missing_or_edited_provenance_creates_actionable_error() -> None:
    """Reject pixel edits before any receiving sampler can substitute a fallback."""
    prompt = graph()
    prompt["decode"]["class_type"] = "ImageScale"
    with pytest.raises(ValueError, match="image provenance"):
        source_lineage(prompt, ["upscale", 0], {})
    rewrite_noise_reuse_graph(prompt, {})
    assert prompt["receive"]["inputs"]["latent_image"] == [
        "receive.noise_reuse_error",
        0,
    ]
    assert (
        prompt["receive.noise_reuse_error"]["class_type"]
        == "SimpleSyrup.NoiseReuseError"
    )


def test_disabled_reuse_leaves_native_graph_unchanged() -> None:
    """Avoid upstream capture or new GPU work when the option is switched off."""
    prompt = graph()
    prompt["upscale"]["inputs"]["method"] = "bilinear"
    prompt["reuse"]["inputs"]["enabled"] = False
    before = deepcopy(prompt)
    rewrite_noise_reuse_graph(prompt, {})
    assert prompt == before


def test_auto_only_requests_format_provenance_without_state_capture() -> None:
    """Identify compatible models without perturbing the original source schedule."""
    prompt = graph()
    prompt["reuse"]["inputs"]["enabled"] = False
    rewrite_noise_reuse_graph(prompt, {})
    assert "requests" not in prompt["source.trajectory_capture"]["inputs"]
    assert not any(
        node["class_type"] == "SimpleSyrup.NoiseCaptureRequest"
        for node in prompt.values()
    )


def test_auto_without_sampler_lineage_keeps_native_fallback_input() -> None:
    """Allow unknown encoded latents while keeping noise reuse source checks strict."""
    prompt = graph()
    prompt["reuse"]["inputs"]["enabled"] = False
    prompt["upscale"]["inputs"]["latent"] = ["encode", 0]
    prompt["encode"] = {"class_type": "VAEEncode", "inputs": {}}
    before = deepcopy(prompt)
    rewrite_noise_reuse_graph(prompt, {})
    assert prompt == before
    prompt["reuse"]["inputs"]["enabled"] = True
    rewrite_noise_reuse_graph(prompt, {})
    assert prompt["upscale"]["inputs"]["latent"] == ["encode", 0]
    assert prompt["receive"]["inputs"]["latent_image"] == [
        "receive.noise_reuse_error",
        0,
    ]


def test_requeued_coordinated_prompt_is_idempotent() -> None:
    """Keep history requeues from installing duplicate keyed source wrappers."""
    prompt = graph()
    rewrite_noise_reuse_graph(prompt, {})
    before = deepcopy(prompt)
    rewrite_noise_reuse_graph(prompt, {})
    assert prompt == before


def test_receiving_settings_are_live_source_cache_dependencies() -> None:
    """Invalidate source capture when the receiving schedule or options change."""
    prompt = graph()
    prompt["receive"]["inputs"]["steps"] = ["live_steps", 0]
    prompt["receive"]["inputs"]["denoise"] = ["live_denoise", 0]
    rewrite_noise_reuse_graph(prompt, {})
    request = prompt["receive.noise_capture_request"]["inputs"]
    assert request["steps"] == ["live_steps", 0]
    assert request["denoise"] == ["live_denoise", 0]
    assert request["options"] == ["reuse", 0]
    capture = prompt["source.trajectory_capture"]["inputs"]
    assert capture["requests"] == ["receive.noise_capture_request", 0]


@pytest.mark.parametrize(
    "kind",
    [
        "SimpleSyrup.KSamplerAttentionCoupling",
        "SimpleSyrup.KSamplerAttentionCouplingContextual",
        "SimpleSyrup.KSamplerAttentionCouplingTiled",
        "SimpleSyrup.KSamplerPromptByRegion",
        "SimpleSyrup.KSamplerPromptByTiledRegion",
        "ExternalCompatibleSampler",
    ],
)
def test_sampling_contract_discovers_regional_and_external_sources(kind: str) -> None:
    """Capture new sampler nodes naturally when they expose the native contract."""
    prompt = graph()
    prompt["source"]["class_type"] = kind
    prompt["source"]["inputs"]["sampler_name"] = "er_sde"
    prompt["source"]["_meta"] = {"title": "Original pajamas"}
    prompt["receive"]["_meta"] = {"title": "Upscale pajamas"}
    rewrite_noise_reuse_graph(prompt, {})
    capture = prompt["source.trajectory_capture"]["inputs"]
    assert capture["sampler_name"] == "er_sde"
    assert "Original pajamas" in capture["graph_context"]
    assert "Upscale pajamas" in capture["graph_context"]
    assert kind in capture["graph_context"]


@pytest.mark.parametrize("advanced", [False, True])
def test_custom_sampling_capture_preserves_sampler_and_guider_consumers(
    advanced: bool,
) -> None:
    """Clone only the source guider branch while observing any connected solver."""
    prompt = graph()
    source = prompt["source"]
    source["class_type"] = "SamplerCustomAdvanced" if advanced else "SamplerCustom"
    source["inputs"].pop("sampler_name")
    source["inputs"].update(sampler=["select", 0], sigmas=["sigmas", 0])
    prompt["select"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "dpmpp_2m"},
    }
    prompt["guidance"] = {
        "class_type": "CFGGuider",
        "inputs": {"model": ["loader", 0], "cfg": 4.0},
    }
    original_guider = deepcopy(prompt["guidance"])
    if advanced:
        source["inputs"].pop("model")
        source["inputs"]["guider"] = ["guidance", 0]
    rewrite_noise_reuse_graph(prompt, {})
    assert prompt["guidance"] == original_guider
    assert source["inputs"]["sampler"] == ["select", 0]
    capture = prompt["source.trajectory_capture"]["inputs"]
    assert capture["sampler_name"] == "dpmpp_2m"
    assert capture["sampler"] == ["select", 0]
    if advanced:
        assert source["inputs"]["guider"] == ["source.trajectory_guider", 0]
        assert prompt["source.trajectory_guider"]["inputs"]["model"] == [
            "source.trajectory_capture",
            0,
        ]
    else:
        assert source["inputs"]["model"] == ["source.trajectory_capture", 0]
