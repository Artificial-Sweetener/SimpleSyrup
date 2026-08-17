# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build the smallest real incoming Anima baseline workflow."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject
from tools.comfy_integration.anima_fixture_selections import (
    ANIMA_DIFFUSION_SELECTION,
    ANIMA_TEXT_ENCODER_SELECTION,
    ANIMA_VAE_SELECTION,
)


@dataclass(frozen=True)
class BuiltBaselineWorkflow:
    """Return the graph and saved-image output node identity."""

    prompt: dict[str, JsonObject]
    save_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every class required from clean node discovery."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_baseline_workflow(run_id: str) -> BuiltBaselineWorkflow:
    """Build one deterministic production-node loader-to-image graph."""

    graph = _Graph()
    loader = graph.add(
        "SimpleSyrup.SimpleLoadAnima",
        diffusion_model=ANIMA_DIFFUSION_SELECTION,
        quantization="Original",
        diffusion_weight_dtype="default",
        text_encoder=ANIMA_TEXT_ENCODER_SELECTION,
        text_encoder_device="default",
        vae=ANIMA_VAE_SELECTION,
    )
    positive = graph.add("CLIPTextEncode", clip=[loader, 1], text="a red cube")
    negative = graph.add("CLIPTextEncode", clip=[loader, 1], text="low quality")
    latent = graph.add(
        "EmptyCosmosLatentVideo", width=256, height=256, length=1, batch_size=1
    )
    sampled = graph.add(
        "KSampler",
        model=[loader, 0],
        seed=1_029_384_756,
        steps=1,
        cfg=4.0,
        sampler_name="euler",
        scheduler="simple",
        positive=[positive, 0],
        negative=[negative, 0],
        latent_image=[latent, 0],
        denoise=1.0,
    )
    decoded = graph.add("VAEDecode", samples=[sampled, 0], vae=[loader, 2])
    saved = graph.add(
        "SaveImage",
        images=[decoded, 0],
        filename_prefix=f"simple_syrup_integration/{run_id}",
    )
    return BuiltBaselineWorkflow(graph.prompt, saved)


class _Graph:
    """Own deterministic numeric node identities."""

    def __init__(self) -> None:
        """Initialize an empty graph."""

        self.prompt: dict[str, JsonObject] = {}

    def add(self, class_type: str, **inputs: object) -> str:
        """Append one node and return its identity."""

        node_id = str(len(self.prompt) + 1)
        self.prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id
