# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build native and regional image-producing Anima comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject

from .cases import AnimaVisualCase, AnimaVisualMode
from .inventory import AnimaVisualInventory


@dataclass(frozen=True, slots=True)
class BuiltAnimaVisualWorkflow:
    """Expose one graph and its image and metrics terminal nodes."""

    prompt: dict[str, JsonObject]
    save_node_id: str
    metrics_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every live Comfy node class in this graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_anima_visual_workflow(
    *,
    run_id: str,
    case: AnimaVisualCase,
    inventory: AnimaVisualInventory,
    mask_names: tuple[str, str],
) -> BuiltAnimaVisualWorkflow:
    """Build one locked 30-step native or regional Anima workflow."""

    graph = _Graph()
    loader = graph.add(
        "SimpleSyrup.SimpleLoadAnima",
        diffusion_model=inventory.diffusion_model,
        quantization="Original",
        diffusion_weight_dtype="default",
        text_encoder=inventory.text_encoder,
        text_encoder_device="default",
        vae=inventory.vae,
    )
    latent = graph.add(
        "EmptyCosmosLatentVideo",
        width=1024,
        height=1024,
        length=1,
        batch_size=1,
    )
    model: object = [loader, 0]
    region_masks: object | None = None
    if case.mode is AnimaVisualMode.PLAIN:
        positive = graph.add(
            "CLIPTextEncode",
            clip=[loader, 1],
            text=_join(
                inventory.global_positive,
                inventory.left_positive,
                inventory.right_positive,
            ),
        )
        negative = graph.add(
            "CLIPTextEncode",
            clip=[loader, 1],
            text=_join(
                inventory.global_negative,
                inventory.left_negative,
                inventory.right_negative,
            ),
        )
        positive_link: object = [positive, 0]
        negative_link: object = [negative, 0]
    else:
        left_positive = inventory.left_positive
        if case.mode is AnimaVisualMode.REGIONAL_LORA:
            left_positive = f"{left_positive} <lora:{inventory.regional_lora}:1>"
        encoded = graph.add(
            "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
            model=[loader, 0],
            clip=[loader, 1],
            positive_prompt="[SEP]".join(
                (
                    inventory.global_positive,
                    left_positive,
                    inventory.right_positive,
                )
            ),
            negative_prompt="[SEP]".join(
                (
                    inventory.global_negative,
                    _join(inventory.global_negative, inventory.left_negative),
                    _join(inventory.global_negative, inventory.right_negative),
                )
            ),
        )
        masks = graph.add(
            "SimpleSyrup.LoadMaskBatch",
            image={"__value__": list(mask_names)},
            channel="red",
        )
        region_masks = [masks, 0]
        model = [encoded, 0]
        positive_link = [encoded, 1]
        negative_link = [encoded, 2]
    metrics_run_id = f"{run_id}:{case.mode.value}:metrics"
    instrumented = graph.add(
        "SimpleSyrupBenchmark.InstrumentModel",
        model=model,
        run_id=metrics_run_id,
    )
    sampler_inputs: dict[str, object] = {
        "model": [instrumented, 0],
        "seed": 1_029_384_756,
        "steps": 30,
        "cfg": 1.0,
        "sampler_name": "er_sde",
        "scheduler": "simple",
        "positive": positive_link,
        "negative": negative_link,
        "latent_image": [latent, 0],
        "denoise": 1.0,
    }
    sampler_type = "KSampler"
    if case.mode is not AnimaVisualMode.PLAIN:
        if region_masks is None:
            raise RuntimeError("Regional attention requires explicit masks.")
        sampler_type = "SimpleSyrup.KSamplerAttentionCoupling"
        sampler_inputs.update(
            {
                "region_masks": region_masks,
                "regional_prompt_weight": 1.0,
                "region_mask_feather": 0,
            }
        )
    sampled = graph.add(sampler_type, **sampler_inputs)
    metrics = graph.add(
        "SimpleSyrupBenchmark.ReadMetrics",
        latent=[sampled, 0],
        run_id=metrics_run_id,
    )
    decoded = graph.add("VAEDecode", samples=[metrics, 0], vae=[loader, 2])
    saved = graph.add(
        "SaveImage",
        images=[decoded, 0],
        filename_prefix=f"anima_visual_performance/{run_id}/{case.mode.value}",
    )
    return BuiltAnimaVisualWorkflow(graph.prompt, saved, metrics)


def _join(*parts: str) -> str:
    """Join complete prompt segments without changing authored weights."""

    return ", ".join(part for part in parts if part.strip())


class _Graph:
    """Own deterministic numeric node identities for one workflow."""

    def __init__(self) -> None:
        """Initialize an empty graph."""

        self.prompt: dict[str, JsonObject] = {}

    def add(self, class_type: str, **inputs: object) -> str:
        """Append one node and return its stable identity."""

        node_id = str(len(self.prompt) + 1)
        self.prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id
