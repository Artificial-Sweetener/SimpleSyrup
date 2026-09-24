# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test the production-node incoming Anima baseline graph."""

from __future__ import annotations

from tools.comfy_integration.anima_fixture_selections import (
    ANIMA_DIFFUSION_SELECTION,
    ANIMA_TEXT_ENCODER_SELECTION,
    ANIMA_VAE_SELECTION,
)
from tools.comfy_integration.baseline_workflow import build_baseline_workflow


def test_baseline_is_small_deterministic_loader_to_saved_image_graph() -> None:
    """Fix the exact real workflow nodes, settings, and connections."""

    workflow = build_baseline_workflow("run-identity")

    assert workflow.save_node_id == "7"
    assert workflow.required_node_ids == frozenset(
        {
            "SimpleSyrup.SimpleLoadAnima",
            "CLIPTextEncode",
            "EmptyCosmosLatentVideo",
            "KSampler",
            "VAEDecode",
            "SaveImage",
        }
    )
    assert workflow.prompt["1"]["inputs"] == {
        "diffusion_model": ANIMA_DIFFUSION_SELECTION,
        "quantization": "Original",
        "diffusion_weight_dtype": "default",
        "text_encoder": ANIMA_TEXT_ENCODER_SELECTION,
        "text_encoder_device": "default",
        "vae": ANIMA_VAE_SELECTION,
    }
    assert workflow.prompt["5"]["inputs"] == {
        "model": ["1", 0],
        "seed": 1_029_384_756,
        "steps": 1,
        "cfg": 4.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "positive": ["2", 0],
        "negative": ["3", 0],
        "latent_image": ["4", 0],
        "denoise": 1.0,
    }
    assert workflow.prompt["7"]["inputs"] == {
        "images": ["6", 0],
        "filename_prefix": "simple_syrup_integration/run-identity",
    }
