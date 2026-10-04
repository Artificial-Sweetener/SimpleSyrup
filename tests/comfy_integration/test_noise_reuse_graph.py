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
